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

`DictionaryManager` creates `lexicon.json` in the current working directory on first save unless a custom path is supplied.
