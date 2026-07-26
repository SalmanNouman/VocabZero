from __future__ import annotations

import argparse
import asyncio
import logging

import os
import uuid

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
import numpy as np
import uvicorn

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse

from pydantic import BaseModel

from vocab_zero.core.dictionary import DictionaryManager, LexiconEntry
from vocab_zero.core.engine import TranslationEngine
from vocab_zero.core.engine_factory import build_engine
from vocab_zero.core.models import AudioConfig, TranslationResult
from vocab_zero.utils.audio import acoustic_hash, extract_whisper_embedding

load_dotenv(find_dotenv())

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
MAX_AUDIO_SIZE = 160000  # Maximum samples (~10 seconds at 16kHz)
MAX_BODY_SIZE = MAX_AUDIO_SIZE * 8 + 4096  # ~1.3 MB: audio floats as JSON text + overhead

logger = logging.getLogger(__name__)


class TranslateRequest(BaseModel):
    source_term: str | None = None
    audio_data: list[float] | None = None
    context: str | None = None


class FeedbackRequestData(BaseModel):
    source_term: str
    target_term: str
    audio_data: list[float] | None = None
    context: str | None = None


class AutocompleteRequest(BaseModel):
    sentence: str
    context: str | None = None


def api_success(data: dict[str, object]) -> dict[str, object]:
    return {"ok": True, "data": data}


def api_error(code: str, message: str) -> dict[str, object]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _cosine_distance(query: np.ndarray, query_norm: float, vector: list[float]) -> float | None:
    """Cosine distance (0=identical, 1=orthogonal) or None if it can't be computed."""
    candidate = np.array(vector, dtype=np.float64)
    candidate_norm = np.linalg.norm(candidate)
    if query_norm <= 0 or candidate_norm <= 0:
        return None
    similarity = float(np.dot(query, candidate) / (query_norm * candidate_norm))
    return 1.0 - max(0.0, min(1.0, similarity))


def perform_audio_matching_candidates(
    audio_data: list[float], dictionary: DictionaryManager, audio_config: AudioConfig | None = None
) -> list[tuple[LexiconEntry, float]]:
    if len(audio_data) > MAX_AUDIO_SIZE:
        logger.warning("Audio payload exceeds maximum size: %d samples", len(audio_data))
        return []

    query_vector = extract_whisper_embedding(audio_data)
    if not query_vector:
        return []

    # Primary path: cosine nearest-neighbour search over stored embeddings.
    results = dictionary.vector_store.search_by_vector(query_vector, k=5)
    if results:
        return [(r.entry, 1.0 - r.score) for r in results]

    # Fallback: in-memory cosine match when the vector store is empty but the
    # dictionary holds embeddings (e.g. mocked entries in tests).
    query = np.array(query_vector, dtype=np.float64)
    query_norm = np.linalg.norm(query)
    candidates: list[tuple[LexiconEntry, float]] = []
    for entry in dictionary.iter_entries():
        distances = [
            d for d in (_cosine_distance(query, query_norm, vec) for vec in entry.embeddings) if d is not None
        ]
        if distances:
            candidates.append((entry, min(distances)))

    candidates.sort(key=lambda x: x[1])
    return candidates


def build_translation_response(result: TranslationResult) -> dict[str, object]:
    response_data = {
        "translated_text": result.translated_text,
        "confidence": result.confidence,
        "source": result.source,
        "status": result.status,
        "context_used": result.context_used,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "feedback_request": None,
    }

    if result.feedback_request:
        response_data["feedback_request"] = {
            "source_term": result.feedback_request.source_term,
            "context": result.feedback_request.context,
            "candidate_matches": result.feedback_request.candidate_matches,
            "reason": result.feedback_request.reason,
        }
    return response_data


def process_feedback(
    source_term: str,
    target_term: str,
    audio_data: list[float] | None,
    context: str | None,
    engine: TranslationEngine,
) -> tuple[LexiconEntry, TranslationResult]:

    embedding: list[float] | None = None
    if audio_data is not None and len(audio_data) > 0:
        if len(audio_data) > MAX_AUDIO_SIZE:
            logger.warning(
                "Audio payload exceeds maximum size in feedback: %d samples", len(audio_data)
            )
            raise ValueError(f"Audio payload exceeds maximum size of {MAX_AUDIO_SIZE} samples")

        embedding = extract_whisper_embedding(audio_data)
        if not embedding:
            raise ValueError("Failed to extract a speech embedding from the provided audio")
        logger.debug("Generated Whisper embedding for feedback (dim %d)", len(embedding))

    stripped_source = source_term.strip()

    stripped_target = target_term.strip()

    if not stripped_source or not stripped_target:
        raise ValueError("source_term and target_term must be non-empty after stripping whitespace")

    existing_entry = engine.dictionary.lookup(stripped_source)

    if existing_entry is not None:
        embeddings = list(existing_entry.embeddings)

        if embedding is not None:
            embeddings.append(embedding)

        context_examples = list(existing_entry.context_examples)

        if context and context not in context_examples:
            context_examples.append(context)

        entry = LexiconEntry(
            source_term=stripped_source,
            target_term=stripped_target,
            confidence=1.0,
            context_examples=context_examples,
            embeddings=embeddings,
        )

    else:
        effective_source = stripped_source

        if embedding is not None:
            effective_source = acoustic_hash([embedding])

        entry = LexiconEntry(
            source_term=effective_source,
            target_term=stripped_target,
            confidence=1.0,
            context_examples=[context] if context else [],
            embeddings=[embedding] if embedding is not None else [],
        )

    result = engine.persist_learned_entry(entry)

    return entry, result


async def periodic_pruning_task(dictionary: DictionaryManager, interval: float = 300.0):

    while True:
        try:
            await asyncio.sleep(interval)

            logger.info("Running template bank clustering background task...")

            loop = asyncio.get_running_loop()

            await loop.run_in_executor(None, dictionary.prune_templates, 5)

            logger.info("Template bank clustering background task finished.")

        except asyncio.CancelledError:
            break

        except Exception as e:
            logger.error("Error in template clustering background task: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):

    dictionary_path = os.getenv("VOCABZERO_DICTIONARY", "lexicon.json")

    vector_db_path = os.getenv("VOCABZERO_VECTOR_DB", None)

    app.state.engine = build_engine(dictionary_path=dictionary_path, vector_db_path=vector_db_path)

    app.state.pruning_task = asyncio.create_task(
        periodic_pruning_task(app.state.engine.dictionary, 300.0)
    )

    yield

    app.state.pruning_task.cancel()

    try:
        await app.state.pruning_task

    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        raise HTTPException(status_code=413, detail="Request body too large")
    return await call_next(request)


@app.get("/")
async def get_index():

    if not INDEX_HTML.exists():
        return api_error("missing_static", "Frontend static HTML file is missing")

    return FileResponse(INDEX_HTML)


@app.post("/api/translate")
async def translate(request: Request, payload: TranslateRequest):

    engine = request.app.state.engine

    # --- PCM Audio Template Matching (Phase 2) ---

    if payload.audio_data is not None:
        if len(payload.audio_data) == 0:
            if not payload.source_term:
                return api_error(
                    "invalid_input", "Empty audio_data without source_term is not valid"
                )
        elif len(payload.audio_data) > MAX_AUDIO_SIZE:
            return api_error("audio_too_large", f"Audio exceeds {MAX_AUDIO_SIZE} samples")
        else:
            # Whisper inference runs synchronously in the main thread: it is
            # fast (~30ms) and PyTorch's forward pass is unstable when dispatched
            # to FastAPI's background thread pool.
            try:
                candidates = perform_audio_matching_candidates(
                    payload.audio_data,
                    engine.dictionary,
                    engine.audio_config,
                )
            except Exception:
                logger.exception("Whisper embedding/matching failed for audio translate")
                return api_error(
                    "extraction_failed", "Failed to process audio for matching"
                )

            best_match, conf = engine.rerank_acoustic_candidates(candidates, payload.context)
            logger.info(
                "Audio translate query. Found %d candidates; best match: %s (conf %.4f)",
                len(candidates),
                best_match.target_term if best_match else None,
                conf,
            )
            for _entry, dist in candidates[:5]:
                logger.info("  Candidate: %s (distance %.4f)", _entry.target_term, dist)

            if best_match is not None:
                response_data = {
                    "translated_text": best_match.target_term,
                    "confidence": conf,
                    "source": "acoustic_matching",
                    "status": "translated",
                    "context_used": [],
                    "error_code": None,
                    "error_message": None,
                    "feedback_request": None,
                }

            else:
                source_id = payload.source_term or f"unknown_{uuid.uuid4().hex}"

                reason_msg = "No matching embedding below threshold"

                if candidates:
                    nearest_dist = min(dist for _, dist in candidates)
                    if nearest_dist >= engine.audio_config.match_distance_threshold:
                        reason_msg = f"Match distance too high ({nearest_dist:.4f})"
                    else:
                        reason_msg = (
                            "Below distance threshold but failed the confidence/ambiguity "
                            f"gate (nearest distance {nearest_dist:.4f})"
                        )

                response_data = {
                    "translated_text": "",
                    "confidence": 0.0,
                    "source": "none",
                    "status": "requires_feedback",
                    "context_used": [],
                    "error_code": None,
                    "error_message": None,
                    "feedback_request": {
                        "source_term": source_id,
                        "context": payload.context or "rohingya_audio",
                        "candidate_matches": [c[0].target_term for c in candidates[:3]],
                        "reason": reason_msg,
                    },
                }

            return api_success(response_data)

    # --- Fallback Text Translation ---

    if payload.source_term is None and payload.audio_data is None:
        return api_error("invalid_input", "Either source_term or audio_data must be provided")

    logger.info("Translate query for source_term (hash: %s)", hash(payload.source_term) & 0xFFFF)

    result = engine.translate(payload.source_term, payload.context)

    if result.status == "error":
        return api_error(
            result.error_code or "translation_failed",
            result.error_message or "Unknown translation error",
        )

    return api_success(build_translation_response(result))


@app.post("/api/feedback")
async def feedback(request: Request, payload: FeedbackRequestData):

    engine = request.app.state.engine

    logger.info("Feedback received: teaching new mapping")

    try:
        entry, result = process_feedback(
            payload.source_term,
            payload.target_term,
            payload.audio_data,
            payload.context,
            engine,
        )

    except ValueError as e:
        message = str(e)
        if "exceeds maximum size" in message:
            return api_error("audio_too_large", message)
        if "Failed to extract" in message:
            return api_error("extraction_failed", message)

        return api_error("invalid_input", message)

    except Exception:
        logger.exception("Failed to extract a speech embedding from feedback audio")
        return api_error(
            "extraction_failed",
            "Failed to extract a speech embedding from the provided audio",
        )

    if result.status == "error":
        return api_error(
            result.error_code or "persistence_failed",
            result.error_message or "Failed to persist lexicon entry",
        )

    # Return the entry's actual persisted source_term (may be an acoustic hash

    # rather than the caller-supplied source_term) so clients can keep their

    # displayed signature in sync with the real lexicon key.

    return api_success({"status": "learned", "source_term": entry.source_term})


@app.post("/api/autocomplete")
async def autocomplete(request: Request, payload: AutocompleteRequest):

    engine = request.app.state.engine

    if not engine.llm_client:
        return api_success({"suggestions": []})

    try:
        response = engine.llm_client.translate(payload.sentence, context=payload.context)

        if response and response.translation:
            suggestions = [w.strip() for w in response.translation.split(",") if w.strip()]

            return api_success({"suggestions": suggestions})

    except Exception as e:
        logger.error("Error in autocomplete: %s", e)

    return api_success({"suggestions": []})


@app.get("/api/lexicon")
async def get_lexicon(request: Request):

    engine = request.app.state.engine

    entries = []

    for entry in engine.dictionary.iter_entries():
        entries.append(
            {
                "source_term": entry.source_term,
                "target_term": entry.target_term,
                "confidence": entry.confidence,
                "context_examples": entry.context_examples,
            }
        )

    return api_success(entries)


@app.delete("/api/lexicon/{source_term}")
async def delete_lexicon_entry(request: Request, source_term: str):

    engine = request.app.state.engine

    normalized = source_term.strip()

    existing = engine.dictionary.lookup(normalized)

    if existing is None:
        return api_error("not_found", f"Entry '{source_term}' not found in lexicon")

    if not engine.dictionary.delete(normalized):
        return api_error("not_found", f"Entry '{source_term}' not found in lexicon")

    try:
        engine.dictionary.save()

    except (OSError, IOError):
        engine.dictionary.upsert(existing)

        return api_error("persistence_failed", "Failed to persist lexicon deletion")

    return api_success({"deleted": True})


@app.get("/api/audio_config")
async def get_audio_config(request: Request):

    engine = request.app.state.engine

    cfg = engine.audio_config

    return api_success(
        {
            "match_distance_threshold": cfg.match_distance_threshold,
            "min_confidence_gate": cfg.min_confidence_gate,
            "ambiguity_margin_ratio": cfg.ambiguity_margin_ratio,
            "ambiguity_confidence_floor": cfg.ambiguity_confidence_floor,
            "sample_rate": cfg.sample_rate,
        }
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="VocabZero API - Project Hail Mary Interface Server"
    )

    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")

    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")

    parser.add_argument("--dictionary", default="lexicon.json", help="Path to dictionary file")

    parser.add_argument("--vector-db", default=None, help="Path to vector database")

    args = parser.parse_args()

    os.environ["VOCABZERO_DICTIONARY"] = args.dictionary

    if args.vector_db:
        os.environ["VOCABZERO_VECTOR_DB"] = args.vector_db

    uvicorn.run("vocab_zero.interfaces.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
