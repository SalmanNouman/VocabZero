from __future__ import annotations

import json
from typing import Callable

import pytest
import serial

from vocab_zero.core.dictionary import LexiconEntry
from vocab_zero.core.models import FeedbackRequest, TranslationResult
from vocab_zero.interfaces import serial_port as serial_module
from vocab_zero.interfaces.serial_port import SerialConfig, SerialInterface


class FakeEngine:
    def __init__(self) -> None:
        self.translate_calls: list[tuple[str, str | None]] = []
        self.on_feedback_required: Callable[[FeedbackRequest], LexiconEntry | None] | None = None
        self.results: list[TranslationResult] = [
            TranslationResult(
                translated_text="translated",
                confidence=0.9,
                source="dictionary",
                status="translated",
            )
        ]

    def translate(self, source_term: str, context: str | None = None) -> TranslationResult:
        self.translate_calls.append((source_term, context))
        return self.results.pop(0)


class FakeSerial:
    def __init__(
        self,
        lines: list[bytes] | None = None,
        fail_on_read: bool = False,
        fail_on_write: bool = False,
        **kwargs: object,
    ) -> None:
        self.lines = lines or []
        self.fail_on_read = fail_on_read
        self.fail_on_write = fail_on_write
        self.kwargs = kwargs
        self.writes: list[bytes] = []
        self.closed = False

    def readline(self) -> bytes:
        if self.fail_on_read:
            raise serial.SerialException("disconnected")
        if self.lines:
            return self.lines.pop(0)
        raise serial.SerialException("disconnected")

    def write(self, data: bytes) -> int:
        if self.fail_on_write:
            raise serial.SerialException("disconnected")
        self.writes.append(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


def make_interface(serial_obj: FakeSerial | None = None, feedback_timeout: float = 0.01) -> SerialInterface:
    engine = FakeEngine()
    config = SerialConfig(port="COM3", feedback_timeout=feedback_timeout, read_timeout=0.01)
    interface = SerialInterface(engine, config, serial_factory=lambda **kwargs: serial_obj or FakeSerial(**kwargs))
    interface._serial = serial_obj or FakeSerial()
    return interface


def written_frame(fake_serial: FakeSerial, index: int = 0) -> dict[str, object]:
    return json.loads(fake_serial.writes[index].decode("utf-8"))


def test_constructor_binds_feedback_callback() -> None:
    engine = FakeEngine()
    interface = SerialInterface(engine, SerialConfig(port="COM3"), serial_factory=lambda **kwargs: FakeSerial(**kwargs))

    assert engine.on_feedback_required == interface.request_human_feedback


def test_normalize_frequencies_sorts_and_joins() -> None:
    interface = make_interface()

    assert interface._normalize_frequencies([660, 220, 440]) == "220_440_660"


def test_parse_raw_non_json_line_as_translate_packet() -> None:
    interface = make_interface()

    packet = interface._parse_line("raw term")

    assert packet == {"type": "translate", "source_term": "raw term", "context": None}


@pytest.mark.parametrize(
    ("line", "message_part"),
    [
        ("[]", "JSON object"),
        ('{"type":"unknown"}', "Unknown"),
        ('{"type":"translate","frequencies":[]}', "cannot be empty"),
        ('{"type":"translate","frequencies":"bad"}', "frequencies list"),
        ('{"type":"translate","frequencies":[220,"bad"]}', "integers"),
    ],
)
def test_parse_invalid_packets_return_invalid_packet(line: str, message_part: str) -> None:
    interface = make_interface()

    packet = interface._parse_line(line)

    assert packet is not None
    assert packet["type"] == "error"
    assert packet["code"] == "invalid_packet"
    assert message_part in packet["message"]


def test_parse_translate_packet_normalizes_context() -> None:
    interface = make_interface()

    packet = interface._parse_line('{"type":"translate","frequencies":[440,220],"context":"ctx"}')

    assert packet == {"type": "translate", "source_term": "220_440", "context": "ctx"}


@pytest.mark.parametrize(
    "result",
    [
        TranslationResult(translated_text="ok", confidence=0.9, source="dictionary", status="translated"),
        TranslationResult(translated_text="low", confidence=0.3, source="dictionary", status="low_confidence"),
        TranslationResult(
            translated_text="",
            source="none",
            status="requires_feedback",
            feedback_request=FeedbackRequest(source_term="x"),
        ),
        TranslationResult(translated_text="", source="none", status="feedback_declined"),
        TranslationResult(translated_text="new", confidence=1.0, source="human_feedback", status="learned"),
        TranslationResult(translated_text="", source="none", status="error", error_code="bad", error_message="Bad"),
    ],
)
def test_display_translation_serializes_statuses(result: TranslationResult) -> None:
    fake_serial = FakeSerial()
    interface = make_interface(fake_serial)

    interface.display_translation(result)

    frame = written_frame(fake_serial)
    assert frame["type"] == "translation_result"
    assert frame["status"] == result.status
    assert frame["text"] == result.translated_text


@pytest.mark.parametrize(
    ("status", "value", "expected_target"),
    [("declined", None, None), ("selected", "target", "target"), ("provided", "target", "target")],
)
def test_feedback_response_statuses(status: str, value: str | None, expected_target: str | None) -> None:
    fake_serial = FakeSerial()
    interface = make_interface(fake_serial)
    packet: dict[str, object] = {"type": "feedback_response", "status": status}
    if value is not None:
        packet["value"] = value
    interface._packet_queue.put(packet)

    entry = interface.request_human_feedback(FeedbackRequest(source_term="source", context="ctx"))

    request_frame = written_frame(fake_serial)
    assert request_frame["type"] == "feedback_request"
    if expected_target is None:
        assert entry is None
    else:
        assert entry is not None
        assert entry.target_term == expected_target
        assert entry.context_examples == ["ctx"]


def test_invalid_feedback_status_writes_error() -> None:
    fake_serial = FakeSerial()
    interface = make_interface(fake_serial)
    interface._packet_queue.put({"type": "feedback_response", "status": "bad"})

    entry = interface.request_human_feedback(FeedbackRequest(source_term="source"))

    assert entry is None
    assert written_frame(fake_serial, 1)["code"] == "invalid_feedback"


def test_feedback_timeout_returns_none() -> None:
    interface = make_interface(feedback_timeout=0.001)

    assert interface.request_human_feedback(FeedbackRequest(source_term="source")) is None


def test_feedback_wait_defers_translate_packets() -> None:
    fake_serial = FakeSerial()
    interface = make_interface(fake_serial)
    translate_packet = {"type": "translate", "source_term": "next", "context": None}
    interface._packet_queue.put(translate_packet)
    interface._packet_queue.put({"type": "feedback_response", "status": "declined"})

    assert interface.request_human_feedback(FeedbackRequest(source_term="source")) is None
    assert interface._next_packet() == translate_packet


def test_feedback_response_processed_before_deferred_packet() -> None:
    fake_serial = FakeSerial()
    interface = make_interface(fake_serial)
    translate_packet = {"type": "translate", "source_term": "next", "context": None}
    interface._packet_queue.put(translate_packet)
    interface._packet_queue.put({"type": "feedback_response", "status": "selected", "value": "target"})

    entry = interface.request_human_feedback(FeedbackRequest(source_term="source", context="ctx"))

    assert entry is not None
    assert entry.target_term == "target"
    assert entry.context_examples == ["ctx"]
    assert interface._next_packet() == translate_packet


def test_reader_enqueues_packets_and_disconnect_sentinel() -> None:
    fake_serial = FakeSerial(lines=[b'{"type":"translate","frequencies":[2,1]}\n'])
    interface = make_interface(fake_serial)

    interface._read_loop()
    packet = interface._packet_queue.get_nowait()

    assert packet["type"] == "translate"
    assert packet["source_term"] == "1_2"

    failing_serial = FakeSerial(fail_on_read=True)
    failing_interface = make_interface(failing_serial)
    failing_interface._read_loop()
    disconnect = failing_interface._packet_queue.get_nowait()

    assert disconnect["type"] == "disconnect"


def test_run_loop_dispatches_translate_and_stops() -> None:
    fake_serial = FakeSerial()
    engine = FakeEngine()
    config = SerialConfig(port="COM3", read_timeout=0.001, max_reconnect_attempts=0)
    interface = SerialInterface(engine, config, serial_factory=lambda **kwargs: fake_serial)
    interface._packet_queue.put({"type": "translate", "source_term": "source", "context": "ctx"})
    interface._packet_queue.put({"type": "disconnect"})

    interface.run()

    assert engine.translate_calls == [("source", "ctx")]
    assert written_frame(fake_serial)["type"] == "translation_result"
    assert fake_serial.closed is True


def test_reconnect_retries_until_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    serials = [FakeSerial(), FakeSerial()]
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> FakeSerial:
        calls.append(kwargs)
        return serials[min(len(calls) - 1, len(serials) - 1)]

    interface = SerialInterface(
        FakeEngine(),
        SerialConfig(port="COM3", read_timeout=0.001, reconnect_delay=0, max_reconnect_attempts=1),
        serial_factory=factory,
    )
    interface._packet_queue.put({"type": "disconnect"})
    interface._packet_queue.put({"type": "disconnect"})
    monkeypatch.setattr(serial_module.time, "sleep", lambda seconds: None)

    interface.run()

    assert len(calls) == 2
    assert serials[0].closed is True
    assert serials[1].closed is True


def test_main_wires_args_and_runs_serial(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    def fake_build_engine(dictionary_path: str, vector_db_path: str | None = None) -> FakeEngine:
        calls["dictionary_path"] = dictionary_path
        calls["vector_db_path"] = vector_db_path
        return FakeEngine()

    class FakeSerialInterface:
        def __init__(self, engine: FakeEngine, config: SerialConfig) -> None:
            calls["engine"] = engine
            calls["config"] = config

        def run(self) -> None:
            calls["ran"] = True

    dictionary_path = tmp_path / "lexicon.json"
    vector_path = tmp_path / "vector"
    monkeypatch.setattr(serial_module, "build_engine", fake_build_engine)
    monkeypatch.setattr(serial_module, "SerialInterface", FakeSerialInterface)
    monkeypatch.setattr(
        "sys.argv",
        [
            "vocabzero-serial",
            "--port",
            "COM3",
            "--baud-rate",
            "9600",
            "--dictionary",
            str(dictionary_path),
            "--vector-db",
            str(vector_path),
            "--feedback-timeout",
            "5",
        ],
    )

    serial_module.main()

    assert calls["dictionary_path"] == str(dictionary_path)
    assert calls["vector_db_path"] == str(vector_path)
    config = calls["config"]
    assert isinstance(config, SerialConfig)
    assert config.port == "COM3"
    assert config.baud_rate == 9600
    assert config.feedback_timeout == 5
    assert calls["ran"] is True
