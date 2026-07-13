from __future__ import annotations



import argparse

import asyncio

import logging

import os

import uuid

from contextlib import asynccontextmanager

from pathlib import Path



import uvicorn

from fastapi import FastAPI, Request

from fastapi.responses import FileResponse

from pydantic import BaseModel



from vocab_zero.core.dictionary import DictionaryManager, LexiconEntry

from vocab_zero.core.engine import TranslationEngine

from vocab_zero.core.engine_factory import build_engine

from vocab_zero.core.models import AudioConfig, TranslationResult

from vocab_zero.utils.audio import acoustic_hash, dtw_distance, extract_mfcc



STATIC_DIR = Path(__file__).parent / "static"

INDEX_HTML = STATIC_DIR / "index.html"



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







class AutocompleteRequest(BaseModel):

    sentence: str

    context: str | None = None





class CalibrationSample(BaseModel):

    label: str

    audio_data: list[float]





class CalibrationApply(BaseModel):

    dtw_threshold_36: float | None = None

    dtw_threshold_12: float | None = None

    min_confidence_gate: float | None = None

    persist: bool = True





def api_success(data: dict[str, object]) -> dict[str, object]:

    return {"ok": True, "data": data}





def api_error(code: str, message: str) -> dict[str, object]:

    return {"ok": False, "error": {"code": code, "message": message}}





def perform_audio_matching_candidates(

    audio_data: list[float], dictionary: DictionaryManager, audio_config: AudioConfig | None = None

) -> list[tuple[LexiconEntry, float]]:

    if len(audio_data) > MAX_AUDIO_SIZE:

        logger.warning("Audio payload exceeds maximum size: %d samples", len(audio_data))

        return []

    query_mfcc = extract_mfcc(audio_data, audio_config=audio_config)

    if not query_mfcc:

        return []

    expected_dims = len(query_mfcc[0])



    candidates = []



    for entry in dictionary.iter_entries():

        min_entry_dist = float("inf")

        # Evaluate against all stored templates (k-NN style)

        for template in entry.mfcc_templates:

            if not template:

                continue

            if any(len(frame) != expected_dims for frame in template):

                logger.warning("Skipping invalid template frame dimensions for '%s'", entry.source_term)

                continue

            dist = dtw_distance(query_mfcc, template)

            if dist < min_entry_dist:

                min_entry_dist = dist



        if min_entry_dist < float("inf"):

            candidates.append((entry, min_entry_dist))



    # Sort candidates by distance

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

    mfcc_template = None

    if audio_data is not None and len(audio_data) > 0:

        if len(audio_data) > MAX_AUDIO_SIZE:

            logger.warning("Audio payload exceeds maximum size in feedback: %d samples", len(audio_data))

            raise ValueError(f"Audio payload exceeds maximum size of {MAX_AUDIO_SIZE} samples")

        mfcc_template = extract_mfcc(audio_data, audio_config=engine.audio_config)

        logger.debug(

            "Generated MFCC template for feedback (shape %dxD)",

            len(mfcc_template),

        )



    stripped_source = source_term.strip()

    stripped_target = target_term.strip()

    if not stripped_source or not stripped_target:

        raise ValueError("source_term and target_term must be non-empty after stripping whitespace")



    existing_entry = engine.dictionary.lookup(stripped_source)

    if existing_entry is not None:

        mfcc_templates = list(existing_entry.mfcc_templates)

        if mfcc_template is not None:

            mfcc_templates.append(mfcc_template)



        context_examples = list(existing_entry.context_examples)

        if context and context not in context_examples:

            context_examples.append(context)



        entry = LexiconEntry(

            source_term=stripped_source,

            target_term=stripped_target,

            confidence=1.0,

            context_examples=context_examples,

            mfcc_templates=mfcc_templates,

        )

    else:

        effective_source = stripped_source

        if mfcc_template is not None:

            effective_source = acoustic_hash(mfcc_template)

        entry = LexiconEntry(

            source_term=effective_source,

            target_term=stripped_target,

            confidence=1.0,

            context_examples=[context] if context else [],

            mfcc_templates=[mfcc_template] if mfcc_template is not None else [],

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

    app.state.engine = build_engine(

        dictionary_path=dictionary_path, vector_db_path=vector_db_path

    )

    app.state.calibration_samples: dict[str, list[list[list[float]]]] = {}

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

        candidates = perform_audio_matching_candidates(

            payload.audio_data, engine.dictionary, engine.audio_config

        )

        

        logger.info(

            "Audio translate query. Found %d candidates.",

            len(candidates),

        )



        best_match, conf = engine.rerank_acoustic_candidates(candidates, payload.context)



        for entry, dist in candidates[:5]:

            logger.info(

                "  Candidate '%s' -> '%s' (DTW dist: %.4f, threshold: %.4f, ratio: %.1f%%)",

                entry.source_term, entry.target_term, dist,

                engine.audio_config.dtw_threshold,

                dist / engine.audio_config.dtw_threshold * 100,

            )



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

            reason_msg = "No matching acoustic template below threshold"

            if candidates:

                reason_msg = f"DTW distance too high ({candidates[0][1]:.4f})"

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

        entry, result = process_feedback(

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

    return api_success({

        "dtw_threshold_36": cfg.dtw_threshold_36,

        "dtw_threshold_12": cfg.dtw_threshold_12,

        "dtw_threshold": cfg.dtw_threshold,

        "min_confidence_gate": cfg.min_confidence_gate,

        "use_deltas": cfg.use_deltas,

        "use_cmvn": cfg.use_cmvn,

        "use_vtln": cfg.use_vtln,

        "use_liftering": cfg.use_liftering,

        "sample_rate": cfg.sample_rate,

    })





@app.post("/api/calibrate/sample")

async def calibrate_sample(request: Request, payload: CalibrationSample):

    engine = request.app.state.engine

    label = payload.label.strip()

    if not label:

        return api_error("invalid_input", "Label must be non-empty")



    if len(payload.audio_data) > MAX_AUDIO_SIZE:

        return api_error("audio_too_large", f"Audio exceeds {MAX_AUDIO_SIZE} samples")



    if len(payload.audio_data) < 2400:

        return api_error("audio_too_short", "Audio too short for MFCC extraction")



    mfcc = extract_mfcc(payload.audio_data, audio_config=engine.audio_config)

    if not mfcc:

        return api_error("extraction_failed", "Could not extract MFCC features from audio")



    samples = request.app.state.calibration_samples

    if label not in samples:

        samples[label] = []

    samples[label].append(mfcc)



    total = sum(len(v) for v in samples.values())

    logger.info("Calibration sample added: label='%s' (%d total samples)", label, total)

    return api_success({

        "label": label,

        "sample_count": len(samples[label]),

        "total_samples": total,

        "labels": {k: len(v) for k, v in samples.items()},

    })





@app.post("/api/calibrate/compute")

async def calibrate_compute(request: Request):

    engine = request.app.state.engine

    samples = request.app.state.calibration_samples



    if len(samples) < 2:

        return api_error("insufficient_data", "Need at least 2 different labels to calibrate")



    for label, templates in samples.items():

        if len(templates) < 2:

            return api_error(

                "insufficient_data",

                f"Label '{label}' needs at least 2 recordings (has {len(templates)})",

            )



    loop = asyncio.get_running_loop()



    def _compute_distances() -> dict[str, object]:

        intra_distances: list[float] = []

        inter_distances: list[float] = []



        labels = list(samples.keys())

        for i, label_a in enumerate(labels):

            templates_a = samples[label_a]

            # Intra-class: same label, different recordings

            for j in range(len(templates_a)):

                for k in range(j + 1, len(templates_a)):

                    dist = dtw_distance(templates_a[j], templates_a[k])

                    intra_distances.append(dist)



            # Inter-class: different labels

            for label_b in labels[i + 1:]:

                templates_b = samples[label_b]

                for ta in templates_a:

                    for tb in templates_b:

                        dist = dtw_distance(ta, tb)

                        inter_distances.append(dist)



        if not intra_distances or not inter_distances:

            return {"error": "Not enough pairwise distances to compute"}



        max_intra = max(intra_distances)

        min_inter = min(inter_distances)

        mean_intra = sum(intra_distances) / len(intra_distances)

        mean_inter = sum(inter_distances) / len(inter_distances)



        # Suggested threshold: midpoint between max intra and min inter, biased toward safety

        if min_inter > max_intra:

            suggested = max_intra + (min_inter - max_intra) * 0.4

        else:

            suggested = (max_intra + min_inter) / 2.0



        separation = min_inter / max_intra if max_intra > 0 else float("inf")



        return {

            "intra_class": {

                "min": round(min(intra_distances), 4),

                "max": round(max_intra, 4),

                "mean": round(mean_intra, 4),

                "count": len(intra_distances),

            },

            "inter_class": {

                "min": round(min_inter, 4),

                "max": round(max(inter_distances), 4),

                "mean": round(mean_inter, 4),

                "count": len(inter_distances),

            },

            "suggested_threshold_36": round(suggested, 4) if engine.audio_config.use_deltas else None,

            "suggested_threshold_12": round(suggested, 4) if not engine.audio_config.use_deltas else None,

            "suggested_threshold": round(suggested, 4),

            "separation_ratio": round(separation, 4),

            "well_separated": separation > 1.5,

            "sample_counts": {k: len(v) for k, v in samples.items()},

        }



    result = await loop.run_in_executor(None, _compute_distances)

    if "error" in result:

        return api_error("computation_failed", result["error"])

    return api_success(result)





@app.post("/api/calibrate/apply")

async def calibrate_apply(request: Request, payload: CalibrationApply):

    engine = request.app.state.engine

    updates: dict[str, float] = {}



    if payload.dtw_threshold_36 is not None:

        if payload.dtw_threshold_36 <= 0:

            return api_error("invalid_value", "dtw_threshold_36 must be positive")

        updates["dtw_threshold_36"] = payload.dtw_threshold_36



    if payload.dtw_threshold_12 is not None:

        if payload.dtw_threshold_12 <= 0:

            return api_error("invalid_value", "dtw_threshold_12 must be positive")

        updates["dtw_threshold_12"] = payload.dtw_threshold_12



    if payload.min_confidence_gate is not None:

        if not (0.0 <= payload.min_confidence_gate <= 1.0):

            return api_error("invalid_value", "min_confidence_gate must be between 0.0 and 1.0")

        updates["min_confidence_gate"] = payload.min_confidence_gate



    if not updates:

        return api_error("no_changes", "No threshold values provided")



    engine.audio_config = engine.audio_config.model_copy(update=updates)

    logger.info("Applied calibration: %s", updates)



    if payload.persist:

        dictionary_path = os.getenv("VOCABZERO_DICTIONARY", "lexicon.json")

        calibration_path = Path(dictionary_path).parent / "calibration.json"

        try:

            engine.audio_config.save_calibration(calibration_path)

            logger.info("Calibration persisted to %s", calibration_path)

        except (OSError, IOError) as exc:

            logger.error("Failed to persist calibration: %s", exc)

            return api_error("persistence_failed", f"Applied in-memory but failed to save: {exc}")



    return api_success({

        "dtw_threshold_36": engine.audio_config.dtw_threshold_36,

        "dtw_threshold_12": engine.audio_config.dtw_threshold_12,

        "dtw_threshold": engine.audio_config.dtw_threshold,

        "min_confidence_gate": engine.audio_config.min_confidence_gate,

        "persisted": payload.persist,

    })





@app.delete("/api/calibrate/samples")

async def calibrate_clear(request: Request):

    request.app.state.calibration_samples = {}

    return api_success({"cleared": True})





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

