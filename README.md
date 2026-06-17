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

## CLI Usage

The `vocabzero-cli` command provides an interactive terminal interface for translation:

```bash
# Basic usage with default dictionary (lexicon.json)
uv run vocabzero-cli

# Specify custom dictionary path
uv run vocabzero-cli --dictionary /path/to/lexicon.json

# Enable vector database for semantic search (optional)
uv run vocabzero-cli --dictionary /path/to/lexicon.json --vector-db /path/to/vector_db
```

The CLI supports:

- Interactive translation: type a term and see the translation result
- Human feedback: when the engine requests feedback, provide the correct translation or skip
- Context display: shows confidence, source, and context used for each translation
- Quit commands: type `quit` or `exit` to leave the CLI

## Serial Usage

The `vocabzero-serial` command connects a host computer to an M5Stick-style device over newline-delimited JSON:

```bash
# Windows
uv run vocabzero-serial --port COM3

# POSIX
uv run vocabzero-serial --port /dev/ttyUSB0

# Custom dictionary and vector database
uv run vocabzero-serial --port COM3 --dictionary /path/to/lexicon.json --vector-db /path/to/vector_db
```

Inbound device packets:

```json
{"type":"translate","frequencies":[220,440,660],"context":"optional label"}
{"type":"feedback_response","status":"provided","value":"target term"}
```

Frequencies must be a non-empty integer list and are normalized into sorted underscore-joined source terms; invalid packets receive error frames. Non-JSON lines are translated as raw source text. Host responses use `translation_result`, `feedback_request`, and `error` frames. Firmware, audio capture, and DSP are out of scope for this package.

To enable LLM-powered translation, set the `OPENAI_API_KEY` environment variable.

`DictionaryManager` creates `lexicon.json` in the current working directory on first save unless a custom path is supplied.

## Vector DB Client

The `VectorStoreClient` provides semantic search capabilities using ChromaDB as an embedded vector database. When using the default embedding function, the embedding model (~80MB) will be downloaded on first use. For deterministic behavior in tests, inject a custom embedding function. ChromaDB storage persists to `.chroma/` by default (git-ignored), relative to the process working directory unless an explicit `persist_dir` is supplied.
