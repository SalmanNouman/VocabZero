from __future__ import annotations

import argparse
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NamedTuple

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vocab_zero.core.dictionary import DictionaryManager, LexiconEntry
from vocab_zero.core.engine import TranslationEngine
from vocab_zero.core.models import TranslationResult
from vocab_zero.interfaces.cli import build_engine
from vocab_zero.utils.audio import dtw_distance, extract_mfcc

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

DTW_MATCH_THRESHOLD = 15.0
MAX_AUDIO_SIZE = 160000  # Maximum samples (~10 seconds at 16kHz)

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


class WebSocketTranslateMessage(BaseModel):
    type: str
    source_term: str | None = None
    audio_data: list[float] | None = None
    context: str | None = None


class WebSocketFeedbackMessage(BaseModel):
    type: str
    source_term: str
    target_term: str
    audio_data: list[float] | None = None
    context: str | None = None


def api_success(data: dict[str, object]) -> dict[str, object]:
    return {"ok": True, "data": data}


def api_error(code: str, message: str) -> dict[str, object]:
    return {"ok": False, "error": {"code": code, "message": message}}


class AudioMatchResult(NamedTuple):
    best_match: LexiconEntry | None
    min_distance: float


def perform_audio_matching(
    audio_data: list[float], dictionary: DictionaryManager
) -> AudioMatchResult:
    if len(audio_data) > MAX_AUDIO_SIZE:
        logger.warning("Audio payload exceeds maximum size: %d samples", len(audio_data))
        return AudioMatchResult(None, float("inf"))
    query_mfcc = extract_mfcc(audio_data, sample_rate=16000)
    if not query_mfcc:
        return AudioMatchResult(None, float("inf"))
    expected_dims = len(query_mfcc[0])

    best_match = None
    min_dist = float("inf")

    for entry in dictionary.iter_entries():
        for template in entry.mfcc_templates:
            if template:
                if any(len(frame) != expected_dims for frame in template):
                    logger.warning("Skipping invalid mfcc_templates for '%s'", entry.source_term)
                    continue
                dist = dtw_distance(query_mfcc, template)
                if dist < min_dist:
                    min_dist = dist
                    best_match = entry

    return AudioMatchResult(best_match, min_dist)


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
    mfcc_templates: list[list[list[float]]] = []
    if audio_data is not None and len(audio_data) > 0:
        if len(audio_data) > MAX_AUDIO_SIZE:
            logger.warning("Audio payload exceeds maximum size in feedback: %d samples", len(audio_data))
            raise ValueError(f"Audio payload exceeds maximum size of {MAX_AUDIO_SIZE} samples")
        template = extract_mfcc(audio_data, sample_rate=16000)
        logger.debug(
            "Generated MFCC template for feedback (shape %dx%d)",
            len(template),
            len(template[0]) if template else 0,
        )
        mfcc_templates = [template]

    stripped_source = source_term.strip()
    stripped_target = target_term.strip()
    if not stripped_source or not stripped_target:
        raise ValueError("source_term and target_term must be non-empty after stripping whitespace")

    entry = LexiconEntry(
        source_term=stripped_source,
        target_term=stripped_target,
        confidence=1.0,
        context_examples=[context] if context else [],
        mfcc_templates=mfcc_templates,
    )

    result = engine.persist_learned_entry(entry)
    return entry, result


def build_audio_match_response(
    match_result: AudioMatchResult,
    source_term: str | None,
    context: str | None,
) -> dict[str, object]:
    best_match, min_dist = match_result

    if best_match is not None and min_dist < DTW_MATCH_THRESHOLD:
        conf = float(max(0.5, min(1.0, 1.0 - (min_dist / DTW_MATCH_THRESHOLD))))
        return {
            "translated_text": best_match.target_term,
            "confidence": conf,
            "source": "acoustic_matching",
            "status": "translated",
            "context_used": [],
            "error_code": None,
            "error_message": None,
            "feedback_request": None,
        }

    # No match: trigger human feedback
    source_id = source_term or f"unknown_{uuid.uuid4().hex}"
    reason_msg = (
        f"DTW distance too high ({min_dist:.4f})"
        if best_match
        else "No templates in lexicon"
    )
    return {
        "translated_text": "",
        "confidence": 0.0,
        "source": "none",
        "status": "requires_feedback",
        "context_used": [],
        "error_code": None,
        "error_message": None,
        "feedback_request": {
            "source_term": source_id,
            "context": context or "rohingya_audio",
            "candidate_matches": [best_match.target_term] if best_match else [],
            "reason": reason_msg,
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Read configuration from env for the lifespan execution
    dictionary_path = os.getenv("VOCABZERO_DICTIONARY", "lexicon.json")
    vector_db_path = os.getenv("VOCABZERO_VECTOR_DB", None)
    app.state.engine = build_engine(
        dictionary_path=dictionary_path, vector_db_path=vector_db_path
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def get_index():
    if not INDEX_HTML.exists():
        return api_error("missing_static", "Frontend static HTML file is missing")
    return FileResponse(INDEX_HTML)


@app.post("/api/translate")
async def translate(request: Request, payload: TranslateRequest):
    engine = request.app.state.engine

    # --- PCM Audio Template Matching (Phase 2) ---
    if payload.audio_data is not None and len(payload.audio_data) > 0:
        # TODO: This is O(N) per request - consider indexing/caching MFCC templates
        # or using vector DB for pre-filtering as lexicon grows
        match_result = perform_audio_matching(payload.audio_data, engine.dictionary)

        logger.info(
            "Audio translate query. Best DTW match: '%s' with distance: %.4f",
            match_result.best_match.target_term if match_result.best_match else 'None',
            match_result.min_distance,
        )

        return api_success(
            build_audio_match_response(match_result, payload.source_term, payload.context)
        )

    # --- Fallback Text Translation ---
    if payload.source_term is None and payload.audio_data is None:
        return api_error(
            "invalid_input", "Either source_term or audio_data must be provided"
        )

    logger.info("Translate query: %s", payload.source_term)
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
    logger.info("Taught mapping: %s -> %s", payload.source_term, payload.target_term)

    try:
        _, result = process_feedback(
            payload.source_term,
            payload.target_term,
            payload.audio_data,
            payload.context,
            engine,
        )
    except ValueError as e:
        if "audio" in str(e).lower():
            return api_error("audio_too_large", str(e))
        return api_error("invalid_input", str(e))

    if result.status == "error":
        return api_error(
            result.error_code or "persistence_failed",
            result.error_message or "Failed to persist lexicon entry",
        )

    return api_success({"status": "learned"})


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


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    engine = websocket.app.state.engine
    logger.info("WebSocket connection established")

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "translate":
                try:
                    msg = WebSocketTranslateMessage.model_validate(data)
                except Exception:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "invalid_input",
                            "message": "Invalid translate message format",
                        }
                    )
                    continue

                if msg.audio_data is not None and len(msg.audio_data) > 0:
                    match_result = perform_audio_matching(msg.audio_data, engine.dictionary)

                    logger.info(
                        "WebSocket audio translate. Best DTW match: '%s' with distance: %.4f",
                        match_result.best_match.target_term if match_result.best_match else 'None',
                        match_result.min_distance,
                    )

                    response_data = build_audio_match_response(match_result, msg.source_term, msg.context)
                    response_data["type"] = "translation" if response_data["status"] == "translated" else "feedback_request"
                    await websocket.send_json(response_data)
                else:
                    if msg.source_term is None:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "code": "invalid_input",
                                "message": "Either source_term or audio_data must be provided",
                            }
                        )
                        continue

                    result = engine.translate(msg.source_term, msg.context)

                    if result.status == "error":
                        await websocket.send_json(
                            {
                                "type": "error",
                                "code": result.error_code or "translation_failed",
                                "message": result.error_message or "Unknown translation error",
                            }
                        )
                    else:
                        response_data = build_translation_response(result)
                        response_data["type"] = "translation"
                        await websocket.send_json(response_data)

            elif message_type == "feedback":
                try:
                    msg = WebSocketFeedbackMessage.model_validate(data)
                except Exception:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "invalid_input",
                            "message": "Invalid feedback message format",
                        }
                    )
                    continue

                logger.info("WebSocket feedback: %s -> %s", msg.source_term, msg.target_term)

                try:
                    _, result = process_feedback(
                        msg.source_term,
                        msg.target_term,
                        msg.audio_data,
                        msg.context,
                        engine,
                    )
                except ValueError as e:
                    error_code = "audio_too_large" if "audio" in str(e).lower() else "invalid_input"
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": error_code,
                            "message": str(e),
                        }
                    )
                    continue

                if result.status == "error":
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": result.error_code or "persistence_failed",
                            "message": result.error_message or "Failed to persist lexicon entry",
                        }
                    )
                else:
                    await websocket.send_json(
                        {
                            "type": "feedback_acknowledged",
                            "status": "learned",
                        }
                    )

            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "unknown_message_type",
                        "message": f"Unknown message type: {message_type}",
                    }
                )

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "server_error",
                    "message": "Internal server error",
                }
            )
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VocabZero API - Project Hail Mary Interface Server"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host address to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run server on"
    )
    parser.add_argument(
        "--dictionary", default="lexicon.json", help="Path to dictionary file"
    )
    parser.add_argument(
        "--vector-db", default=None, help="Path to vector database"
    )

    args = parser.parse_args()

    os.environ["VOCABZERO_DICTIONARY"] = args.dictionary
    if args.vector_db:
        os.environ["VOCABZERO_VECTOR_DB"] = args.vector_db

    uvicorn.run("vocab_zero.interfaces.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
