# VocabZero

Open-source Dynamic Lexicon-Learning Translator that dynamically learns translation pairs (lexicons) in real-time.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Quickstart

```bash
# Install all dependencies (including dev tools)
uv sync --dev

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
# Development server with auto-reload
uv run uvicorn vocab_zero.interfaces.api:app --host 0.0.0.0 --port 8000 --reload

# Production (no reload)
uv run uvicorn vocab_zero.interfaces.api:app --host 0.0.0.0 --port 8000
```

The server starts at `http://localhost:8000`. Open this URL in a browser to access the web frontend.

### Web Frontend

The built-in frontend (`src/vocab_zero/interfaces/static/index.html`) provides a "Project Hail Mary" speech interface for short recorded phrases. The frontend splits each recording into word segments using silence detection. Each segment is sent to `/api/translate` for acoustic matching. Matched words appear as cyan bubbles; unknown words appear as red dashed bubbles that can be clicked to teach a translation.

The interface supports the **Teach Dialog**: when an unknown sound is detected, a dialog appears with context-aware autocomplete suggestions (powered by the LLM if configured). Enter the English translation and click "Teach Sound" to save it to the lexicon.

### Configuration

The server reads these environment variables at startup:

- `VOCABZERO_DICTIONARY`: path to the lexicon JSON file (default: `lexicon.json`)
- `VOCABZERO_VECTOR_DB`: path to ChromaDB vector storage (empty string = disabled)
- `OPENAI_API_KEY`: API key for LLM-powered translation and autocomplete (empty = disabled)
- `OPENAI_BASE_URL`: base URL for OpenAI-compatible API
- `LLM_PROVIDER`: LLM provider to use (`openai`, `none`)
- `VOCABZERO_CMVN` / `VOCABZERO_VTLN` / `VOCABZERO_LIFTERING` / `VOCABZERO_DELTAS`: audio normalization toggles (accept `1`/`true`/`yes`/`on`; all default on)

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

Returns `{"status": "learned"}`.

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

## Vector DB Client

The `VectorStoreClient` provides semantic search capabilities using ChromaDB as an embedded vector database. When using the default embedding function, the embedding model (~80MB) will be downloaded on first use. For deterministic behavior in tests, inject a custom embedding function. ChromaDB storage persists to `.chroma/` by default (git-ignored), relative to the process working directory unless an explicit `persist_dir` is supplied.
