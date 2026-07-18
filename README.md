# VocabZero

Open-source Dynamic Lexicon-Learning Translator that dynamically learns translation pairs (lexicons) in real-time.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Quickstart

```bash
# Install all dependencies (including dev tools)
uv sync --dev

# Optional: configure the backend through a local dotenv file
cp .env.example .env

# Run the test suite
uv run pytest

# Lint the codebase
uv run ruff check

# Build the package
uv build
```

## FastAPI Server Usage

The `vocabzero-api` entry point serves a FastAPI application with a built-in web frontend for real-time acoustic translation and lexicon management.

### Starting the Server

```bash
# Development server with auto-reload (localhost only)
uv run uvicorn vocab_zero.interfaces.api:app --host 127.0.0.1 --port 8000 --reload

# Production (no reload) — bind to localhost; use a reverse proxy with auth for remote access
uv run uvicorn vocab_zero.interfaces.api:app --host 127.0.0.1 --port 8000
```

The server starts at `http://localhost:8000`. Open this URL in a browser to access the web frontend.

### Web Frontend

The built-in frontend (`src/vocab_zero/interfaces/static/index.html`) provides a "Project Hail Mary" speech interface for short recorded phrases. The frontend splits each recording into word segments using offline Silero VAD, with an RMS energy gate as a fallback. Each segment is sent to `/api/translate` for acoustic matching. Matched words appear as cyan bubbles; unknown words appear as red dashed bubbles that can be clicked to teach a translation.

The interface supports the **Teach Dialog**: when an unknown sound is detected, a dialog appears with context-aware autocomplete suggestions (powered by the LLM if configured). Enter the English translation and click "Teach Sound" to save it to the lexicon.

### Configuration

The server reads these environment variables at startup:

Copy `.env.example` to `.env` to configure these values locally. The application loads `.env` automatically at startup, while explicitly set environment variables take precedence.

- `VOCABZERO_DICTIONARY`: path to the lexicon JSON file (default: `lexicon.json`)
- `VOCABZERO_VECTOR_DB`: path to ChromaDB vector storage (empty string = disabled)
- `OPENAI_API_KEY`: API key for LLM-powered translation and autocomplete (empty = disabled)
- `OPENAI_BASE_URL`: base URL for OpenAI-compatible API
- `LLM_MODEL_NAME`: model name (default: `gpt-4o-mini`)
- `LLM_PROVIDER`: LLM provider to use (`openai`, `none`)
- `VOCABZERO_CMVN` / `VOCABZERO_VTLN` / `VOCABZERO_LIFTERING` / `VOCABZERO_DELTAS`: audio normalization toggles (accept `1`/`true`/`yes`/`on`; all default on)
- `VOCABZERO_DTW_THRESHOLD_36` / `VOCABZERO_DTW_THRESHOLD_12`: normalized DTW thresholds (defaults `1.8` / `1.2`)
- `VOCABZERO_MIN_CONFIDENCE`: minimum accepted acoustic confidence (default `0.6`)
- `VOCABZERO_AMBIGUITY_MARGIN_RATIO`: normalized distance margin used to identify ambiguity and restore full confidence when exceeded (default `0.15`)
- `VOCABZERO_AMBIGUITY_CONFIDENCE_FLOOR`: confidence multiplier floor for ambiguous acoustic-only matches (default `0.4`)
- `VOCABZERO_DTW_BAND_RATIO`: Sakoe-Chiba DTW band width as a fraction of the longer sequence (default `0.2`)
- `VOCABZERO_MAX_LENGTH_RATIO`: maximum template/query frame-length ratio (default `2.5`)
- `VOCABZERO_TEMPLATE_AGG_K`: number of nearest valid templates averaged per lexicon entry (default `3`)

### REST API Endpoints

All endpoints return `{ ok: boolean, data?: T, error?: { code, message } }`.

#### `POST /api/translate`

Translate a source term or audio signal.

```json
{
  "source_term": "optional text signature",
  "audio_data": [0.0, 0.1, ...],
  "context": "optional context label"
}
```

If `audio_data` is provided, the engine extracts MFCC features and matches against stored templates via DTW. If `source_term` is provided without audio, it does a dictionary lookup. Returns `translated`, `low_confidence`, `requires_feedback`, or `error`.

#### `POST /api/feedback`

Teach the engine a new translation pair. When `audio_data` is provided and the `source_term` doesn't exist in the lexicon, a deterministic acoustic hash (`sound_<8hex>`) is auto-generated as the source term.

```json
{
  "source_term": "200_500",
  "target_term": "hello",
  "audio_data": [0.0, 0.1, ...],
  "context": "greeting"
}
```

Returns `{"status": "learned", "source_term": "<canonical_or_generated_id>"}`. The `source_term` field returns the canonical source term used (auto-generated `sound_<8hex>` if the input did not match an existing entry).

#### `POST /api/autocomplete`

Get context-aware translation suggestions for a partially-translated sentence. Requires LLM to be configured.

```json
{
  "sentence": "hello [unknown] world",
  "context": "greeting"
}
```

Returns `{"suggestions": ["word1", "word2"]}`.

#### `GET /api/lexicon`

List all lexicon entries with source term, target term, confidence, and context examples.

#### `DELETE /api/lexicon/{source_term}`

Remove a lexicon entry by its source term.

#### `GET /api/audio_config`

Returns the current audio processing configuration including DTW thresholds, confidence controls, template matching safeguards, and normalization flags.

```json
{
  "sample_rate": 16000,
  "dtw_threshold_36": 1.8,
  "dtw_threshold_12": 1.2,
  "dtw_threshold": 1.8,
  "min_confidence_gate": 0.6,
  "ambiguity_margin_ratio": 0.15,
  "ambiguity_confidence_floor": 0.4,
  "dtw_band_ratio": 0.2,
  "max_length_ratio": 2.5,
  "template_agg_k": 3,
  "use_deltas": true,
  "use_cmvn": true,
  "use_vtln": true,
  "use_liftering": true
}
```

#### `POST /api/calibrate/sample`

Submit a calibration recording for a specific label. Audio is validated (2400–160000 samples), MFCC features are extracted, and stored under the label. Per-label cap: 20 samples. Total cap: 200 samples.

```json
{
  "label": "phrase_1",
  "audio_data": [0.0, 0.1, ...]
}
```

Returns `{"label": "phrase_1", "sample_count": 3, "total_samples": 9, "labels": {...}}`.

#### `POST /api/calibrate/compute`

Compute DTW distance statistics from all collected calibration samples. Requires at least 2 labels with 2 recordings each. Returns intra-class and inter-class distance distributions, a suggested threshold, and separation ratio.

```json
{
  "intra_class": { "min": 0.5, "max": 1.2, "mean": 0.8, "count": 6 },
  "inter_class": { "min": 2.1, "max": 3.5, "mean": 2.8, "count": 12 },
  "suggested_threshold": 1.56,
  "separation_ratio": 1.75,
  "well_separated": true,
  "sample_counts": { "phrase_1": 3, "phrase_2": 3, "phrase_3": 3 }
}
```

#### `POST /api/calibrate/apply`

Apply calibrated DTW thresholds and confidence gate. Persists values to `calibration.json`.

```json
{
  "dtw_threshold_36": 1.56,
  "dtw_threshold_12": 1.0,
  "min_confidence_gate": 0.6
}
```

Returns `{"status": "applied", "dtw_threshold_36": 1.56, ...}`.

#### `DELETE /api/calibrate/samples`

Clear all collected calibration samples. Must be called before starting a new calibration session.

## Vector DB Client

The `VectorStoreClient` provides semantic search capabilities using ChromaDB as an embedded vector database. When using the default embedding function, the embedding model (~80MB) will be downloaded on first use. For deterministic behavior in tests, inject a custom embedding function. ChromaDB storage persists to `.chroma/` by default (git-ignored), relative to the process working directory unless an explicit `persist_dir` is supplied.
