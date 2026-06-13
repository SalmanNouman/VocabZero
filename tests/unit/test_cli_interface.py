from __future__ import annotations

from typing import Callable

from vocab_zero.core.dictionary import LexiconEntry
from vocab_zero.core.models import FeedbackRequest, TranslationResult
from vocab_zero.interfaces import cli as cli_module
from vocab_zero.interfaces.cli import CLIInterface, build_engine


class FakeEngine:
    """Fake engine for testing CLI without real translation logic."""

    def __init__(self) -> None:
        self.translate_calls: list[tuple[str, str | None]] = []
        self.on_feedback_required: Callable[[FeedbackRequest], LexiconEntry | None] | None = None

    def translate(self, source_term: str, context: str | None = None) -> TranslationResult:
        self.translate_calls.append((source_term, context))
        return TranslationResult(
            translated_text=f"translated_{source_term}",
            confidence=0.9,
            source="dictionary",
            status="translated",
        )


class FakeVectorStore:
    def __init__(self, persist_dir: str) -> None:
        self.persist_dir = persist_dir


class FakeOpenAIClient:
    def __init__(self, config: object) -> None:
        self.config = config


def test_cli_interface_display_translated():
    """Test display_translation for translated status."""
    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, output_func=fake_output)

    result = TranslationResult(
        translated_text="hello",
        confidence=0.95,
        source="dictionary",
        status="translated",
        context_used=["greeting"],
    )
    cli.display_translation(result)

    assert any("Translation: hello" in line for line in outputs)
    assert any("Confidence: 0.95" in line for line in outputs)
    assert any("Source: dictionary" in line for line in outputs)
    assert any("Context used: greeting" in line for line in outputs)


def test_cli_interface_display_low_confidence():
    """Test display_translation for low_confidence status."""
    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, output_func=fake_output)

    result = TranslationResult(
        translated_text="hello",
        confidence=0.3,
        source="llm_inference",
        status="low_confidence",
        context_used=["context1", "context2"],
    )
    cli.display_translation(result)

    assert any("low confidence" in line for line in outputs)
    assert any("Confidence: 0.30" in line for line in outputs)


def test_cli_interface_display_requires_feedback():
    """Test display_translation for requires_feedback status."""
    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, output_func=fake_output)

    result = TranslationResult(
        translated_text="",
        confidence=0.0,
        source="none",
        status="requires_feedback",
        feedback_request=FeedbackRequest(
            source_term="unknown",
            context="some context",
            candidate_matches=["candidate1", "candidate2"],
            reason="Term not found",
        ),
    )
    cli.display_translation(result)

    assert any("Feedback required" in line for line in outputs)
    assert any("Term not found" in line for line in outputs)
    assert any("candidate1" in line for line in outputs)


def test_cli_interface_display_feedback_declined():
    """Test display_translation for feedback_declined status."""
    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, output_func=fake_output)

    result = TranslationResult(
        translated_text="",
        confidence=0.0,
        source="dictionary",
        status="feedback_declined",
    )
    cli.display_translation(result)

    assert any("Feedback skipped" in line for line in outputs)
    assert any("no translation provided" in line for line in outputs)


def test_cli_interface_display_learned():
    """Test display_translation for learned status."""
    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, output_func=fake_output)

    result = TranslationResult(
        translated_text="learned_translation",
        confidence=1.0,
        source="human_feedback",
        status="learned",
        error_code="semantic_index_update_failed",
    )
    cli.display_translation(result)

    assert any("Learned: learned_translation" in line for line in outputs)
    assert any("Warning: semantic_index_update_failed" in line for line in outputs)


def test_cli_interface_display_error():
    """Test display_translation for error status."""
    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, output_func=fake_output)

    result = TranslationResult(
        translated_text="",
        confidence=0.0,
        source="none",
        status="error",
        error_code="invalid_input",
        error_message="Source term cannot be empty",
    )
    cli.display_translation(result)

    assert any("Error: Source term cannot be empty" in line for line in outputs)
    assert any("Error code: invalid_input" in line for line in outputs)


def test_cli_interface_request_feedback_accept():
    """Test request_human_feedback when user provides a translation."""
    inputs = ["target_translation"]
    input_index = [0]

    def fake_input(prompt: str) -> str:
        idx = input_index[0]
        input_index[0] += 1
        return inputs[idx]

    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, input_func=fake_input, output_func=fake_output)

    request = FeedbackRequest(
        source_term="source",
        context="context",
        candidate_matches=["candidate1"],
        reason="low confidence",
    )

    result = cli.request_human_feedback(request)

    assert result is not None
    assert result.source_term == "source"
    assert result.target_term == "target_translation"
    assert result.confidence == 1.0
    assert result.context_examples == ["context"]


def test_cli_interface_request_feedback_reject():
    """Test request_human_feedback when user skips (empty input)."""
    inputs = [""]
    input_index = [0]

    def fake_input(prompt: str) -> str:
        idx = input_index[0]
        input_index[0] += 1
        return inputs[idx]

    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, input_func=fake_input, output_func=fake_output)

    request = FeedbackRequest(source_term="source", context="context")

    result = cli.request_human_feedback(request)

    assert result is None


def test_cli_interface_request_feedback_no_context():
    """Test request_human_feedback with no context."""
    inputs = ["target"]
    input_index = [0]

    def fake_input(prompt: str) -> str:
        idx = input_index[0]
        input_index[0] += 1
        return inputs[idx]

    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, input_func=fake_input, output_func=fake_output)

    request = FeedbackRequest(source_term="source", context=None)

    result = cli.request_human_feedback(request)

    assert result is not None
    assert result.context_examples == []


def test_cli_interface_run_quit():
    """Test run() loop with quit command."""
    inputs = ["quit"]
    input_index = [0]

    def fake_input(prompt: str) -> str:
        idx = input_index[0]
        input_index[0] += 1
        return inputs[idx]

    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, input_func=fake_input, output_func=fake_output)

    cli.run()

    assert any("Goodbye!" in line for line in outputs)
    assert len(engine.translate_calls) == 0


def test_cli_interface_run_translation():
    """Test run() loop with a translation request."""
    inputs = ["hello", "quit"]
    input_index = [0]

    def fake_input(prompt: str) -> str:
        idx = input_index[0]
        input_index[0] += 1
        return inputs[idx]

    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, input_func=fake_input, output_func=fake_output)

    cli.run()

    assert len(engine.translate_calls) == 1
    assert engine.translate_calls[0] == ("hello", None)
    assert any("translated_hello" in line for line in outputs)


def test_cli_interface_run_empty_input():
    """Test run() loop skips empty input."""
    inputs = ["", "quit"]
    input_index = [0]

    def fake_input(prompt: str) -> str:
        idx = input_index[0]
        input_index[0] += 1
        return inputs[idx]

    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, input_func=fake_input, output_func=fake_output)

    cli.run()

    assert any("empty input - skipped" in line for line in outputs)
    assert len(engine.translate_calls) == 0


def test_cli_interface_run_keyboard_interrupt():
    """Test run() loop handles KeyboardInterrupt gracefully."""

    def fake_input(prompt: str) -> str:
        raise KeyboardInterrupt()

    outputs: list[str] = []

    def fake_output(text: str) -> None:
        outputs.append(text)

    engine = FakeEngine()
    cli = CLIInterface(engine, input_func=fake_input, output_func=fake_output)

    cli.run()

    assert any("Goodbye!" in line for line in outputs)


def test_cli_interface_feedback_callback_bound():
    """Test that CLI binds its feedback method to the engine."""
    engine = FakeEngine()
    cli = CLIInterface(engine)

    assert engine.on_feedback_required is not None
    assert engine.on_feedback_required == cli.request_human_feedback


def test_build_engine_with_dictionary_only(tmp_path):
    """Test build_engine with only dictionary (no vector DB or LLM)."""
    dict_path = tmp_path / "lexicon.json"
    dict_path.write_text("{}")

    engine = build_engine(dictionary_path=str(dict_path), vector_db_path=None)

    assert engine is not None
    assert engine.dictionary is not None
    assert engine.vector_store is None
    assert engine.llm_client is None


def test_build_engine_with_vector_db(tmp_path, monkeypatch):
    """Test build_engine with vector DB enabled."""
    dict_path = tmp_path / "lexicon.json"
    dict_path.write_text("{}")
    vector_path = tmp_path / "vector_db"

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(cli_module, "VectorStoreClient", FakeVectorStore)

    engine = build_engine(dictionary_path=str(dict_path), vector_db_path=str(vector_path))

    assert engine is not None
    assert engine.dictionary is not None
    assert engine.vector_store is not None
    assert isinstance(engine.vector_store, FakeVectorStore)
    assert engine.vector_store.persist_dir == str(vector_path)
    assert engine.llm_client is None


def test_build_engine_with_llm(tmp_path, monkeypatch):
    """Test build_engine with LLM client when API key is present."""
    dict_path = tmp_path / "lexicon.json"
    dict_path.write_text("{}")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli_module, "OpenAICompatibleClient", FakeOpenAIClient)

    engine = build_engine(dictionary_path=str(dict_path), vector_db_path=None)

    assert engine is not None
    assert engine.dictionary is not None
    assert engine.vector_store is None
    assert engine.llm_client is not None
    assert isinstance(engine.llm_client, FakeOpenAIClient)


def test_main_wires_args_and_runs_cli(monkeypatch, tmp_path):
    """Test main() parses args, builds an engine, and runs the CLI."""
    dictionary_path = tmp_path / "lexicon.json"
    vector_path = tmp_path / "vector_db"
    calls: dict[str, object] = {}

    def fake_build_engine(dictionary_path: str, vector_db_path: str | None = None) -> FakeEngine:
        calls["dictionary_path"] = dictionary_path
        calls["vector_db_path"] = vector_db_path
        return FakeEngine()

    class FakeCLI:
        def __init__(self, engine: FakeEngine) -> None:
            calls["engine"] = engine
            self.engine = engine

        def run(self) -> None:
            calls["ran"] = True

    monkeypatch.setattr(cli_module, "build_engine", fake_build_engine)
    monkeypatch.setattr(cli_module, "CLIInterface", FakeCLI)
    monkeypatch.setattr(
        "sys.argv",
        [
            "vocabzero-cli",
            "--dictionary",
            str(dictionary_path),
            "--vector-db",
            str(vector_path),
        ],
    )

    cli_module.main()

    assert calls["dictionary_path"] == str(dictionary_path)
    assert calls["vector_db_path"] == str(vector_path)
    assert isinstance(calls["engine"], FakeEngine)
    assert calls["ran"] is True
