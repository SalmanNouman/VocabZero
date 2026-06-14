from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, TypedDict

import serial

from vocab_zero.core.dictionary import LexiconEntry
from vocab_zero.core.engine import TranslationEngine
from vocab_zero.core.models import FeedbackRequest, TranslationResult
from vocab_zero.interfaces.base import BaseInterface
from vocab_zero.interfaces.cli import build_engine

PacketType = Literal["translate", "feedback_response", "error", "disconnect"]


class SerialConnection(Protocol):
    def readline(self) -> bytes: ...

    def write(self, data: bytes) -> int | None: ...

    def close(self) -> None: ...


class InboundPacket(TypedDict, total=False):
    type: PacketType
    source_term: str
    context: str | None
    status: str
    value: str
    code: str
    message: str


@dataclass(frozen=True)
class SerialConfig:
    port: str
    baud_rate: int = 115200
    read_timeout: float = 1.0
    write_timeout: float = 1.0
    reconnect_delay: float = 2.0
    feedback_timeout: float = 30.0
    max_reconnect_attempts: int = 3


SerialFactory = Callable[..., SerialConnection]


class SerialInterface(BaseInterface):
    def __init__(
        self,
        engine: TranslationEngine,
        config: SerialConfig,
        serial_factory: SerialFactory | None = None,
    ) -> None:
        self.engine = engine
        self.config = config
        self.serial_factory = serial_factory or serial.Serial
        self.engine.on_feedback_required = self.request_human_feedback
        self._packet_queue: queue.Queue[InboundPacket] = queue.Queue()
        self._deferred_packets: queue.Queue[InboundPacket] = queue.Queue()
        self._stop_event = threading.Event()
        self._write_lock = threading.Lock()
        self._serial: SerialConnection | None = None
        self._reader_thread: threading.Thread | None = None
        self._feedback_wait_active = False

    def run(self) -> None:
        for attempt in range(self.config.max_reconnect_attempts + 1):
            try:
                self._connect()
                break
            except serial.SerialException:
                if attempt >= self.config.max_reconnect_attempts:
                    return
                time.sleep(self.config.reconnect_delay)

        reconnect_attempts = 0

        while not self._stop_event.is_set():
            packet = self._next_packet()
            if packet is None:
                continue

            if packet["type"] == "disconnect":
                reconnect_attempts += 1
                if reconnect_attempts > self.config.max_reconnect_attempts:
                    self._stop_event.set()
                    break
                self._close_serial()
                time.sleep(self.config.reconnect_delay)
                for attempt in range(self.config.max_reconnect_attempts - reconnect_attempts + 1):
                    try:
                        self._connect()
                        break
                    except serial.SerialException:
                        if attempt >= self.config.max_reconnect_attempts - reconnect_attempts:
                            self._stop_event.set()
                            break
                        time.sleep(self.config.reconnect_delay)
                continue

            reconnect_attempts = 0
            self._dispatch_packet(packet)

        self._close_serial()

    def display_translation(self, result: TranslationResult) -> None:
        frame: dict[str, object] = {
            "type": "translation_result",
            "status": result.status,
            "text": result.translated_text,
            "confidence": result.confidence,
            "source": result.source,
            "context_used": result.context_used,
        }
        if result.error_code:
            frame["error_code"] = result.error_code
        if result.error_message:
            frame["error_message"] = result.error_message
        if result.feedback_request is not None:
            frame["feedback_request"] = self._feedback_request_frame(result.feedback_request)
        self._write_frame(frame)

    def request_human_feedback(self, request: FeedbackRequest) -> LexiconEntry | None:
        self._feedback_wait_active = True
        self._write_frame(self._feedback_request_frame(request))
        deadline = time.monotonic() + self.config.feedback_timeout
        deferred_during_wait: list[InboundPacket] = []

        try:
            while not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                try:
                    packet = self._packet_queue.get(timeout=min(remaining, 0.1))
                except queue.Empty:
                    continue

                if packet["type"] == "feedback_response":
                    for deferred_packet in deferred_during_wait:
                        self._deferred_packets.put(deferred_packet)
                    return self._feedback_entry(request, packet)

                deferred_during_wait.append(packet)

            for deferred_packet in deferred_during_wait:
                self._deferred_packets.put(deferred_packet)
            return None
        finally:
            self._feedback_wait_active = False

    def stop(self) -> None:
        self._stop_event.set()
        self._close_serial()

    def _connect(self) -> None:
        self._serial = self.serial_factory(
            port=self.config.port,
            baudrate=self.config.baud_rate,
            timeout=self.config.read_timeout,
            write_timeout=self.config.write_timeout,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _close_serial(self) -> None:
        if self._serial is None:
            return
        try:
            self._serial.close()
        except serial.SerialException:
            pass
        finally:
            self._serial = None

    def _read_loop(self) -> None:
        active_serial = self._serial
        if active_serial is None:
            return

        while not self._stop_event.is_set():
            try:
                line = active_serial.readline()
            except serial.SerialException:
                self._packet_queue.put(
                    {
                        "type": "disconnect",
                        "code": "serial_disconnected",
                        "message": "Serial port disconnected",
                    }
                )
                return

            packet = self._parse_line(line)
            if packet is not None:
                self._packet_queue.put(packet)

    def _next_packet(self, timeout: float | None = None) -> InboundPacket | None:
        try:
            return self._deferred_packets.get_nowait()
        except queue.Empty:
            pass

        try:
            return self._packet_queue.get(
                timeout=timeout if timeout is not None else self.config.read_timeout
            )
        except queue.Empty:
            return None

    def _dispatch_packet(self, packet: InboundPacket) -> None:
        packet_type = packet["type"]
        if packet_type == "translate":
            result = self.engine.translate(packet["source_term"], packet.get("context"))
            self.display_translation(result)
            return

        if packet_type == "feedback_response":
            if not self._feedback_wait_active:
                self._write_error("unexpected_feedback", "Feedback response received without active request")
            return

        if packet_type == "error":
            self._write_error(packet["code"], packet["message"])

    def _parse_line(self, line: bytes | str) -> InboundPacket | None:
        text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
        text = text.strip()
        if not text:
            return None

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"type": "translate", "source_term": text, "context": None}

        if not isinstance(data, dict):
            return self._invalid_packet("Packet must be a JSON object")

        frame_type = data.get("type")
        if frame_type == "translate":
            return self._parse_translate(data)
        if frame_type == "feedback_response":
            return self._parse_feedback_response(data)
        return self._invalid_packet("Unknown packet type")

    def _parse_translate(self, data: dict[object, object]) -> InboundPacket:
        frequencies = data.get("frequencies")
        if not isinstance(frequencies, list):
            return self._invalid_packet("Translate packet requires a frequencies list")
        if not frequencies:
            return self._invalid_packet("Translate packet frequencies cannot be empty")
        if not all(type(frequency) is int for frequency in frequencies):
            return self._invalid_packet("Frequencies must be integers")

        context = data.get("context")
        return {
            "type": "translate",
            "source_term": self._normalize_frequencies(frequencies),
            "context": context if isinstance(context, str) else None,
        }

    def _parse_feedback_response(self, data: dict[object, object]) -> InboundPacket:
        status = data.get("status")
        if not isinstance(status, str):
            return self._invalid_packet("Feedback response requires a status")

        value = data.get("value")
        packet: InboundPacket = {"type": "feedback_response", "status": status}
        if isinstance(value, str):
            packet["value"] = value
        return packet

    def _normalize_frequencies(self, frequencies: list[int]) -> str:
        return "_".join(str(frequency) for frequency in sorted(frequencies))

    def _feedback_request_frame(self, request: FeedbackRequest) -> dict[str, object]:
        return {
            "type": "feedback_request",
            "source_term": request.source_term,
            "context": request.context,
            "candidates": request.candidate_matches,
            "reason": request.reason,
        }

    def _feedback_entry(self, request: FeedbackRequest, packet: InboundPacket) -> LexiconEntry | None:
        status = packet["status"]
        if status == "declined":
            return None
        if status not in ("selected", "provided"):
            self._write_error("invalid_feedback", "Feedback response status is invalid")
            return None

        value = packet.get("value", "").strip()
        if not value:
            self._write_error("invalid_feedback", "Feedback response requires a value")
            return None

        return LexiconEntry(
            source_term=request.source_term,
            target_term=value,
            confidence=1.0,
            context_examples=[request.context] if request.context else [],
        )

    def _invalid_packet(self, message: str) -> InboundPacket:
        return {"type": "error", "code": "invalid_packet", "message": message}

    def _write_error(self, code: str, message: str) -> None:
        self._write_frame({"type": "error", "code": code, "message": message})

    def _write_frame(self, data: dict[str, object]) -> None:
        active_serial = self._serial
        if active_serial is None:
            return

        payload = (json.dumps(data, separators=(",", ":")) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                active_serial.write(payload)
            except serial.SerialException:
                self._packet_queue.put(
                    {
                        "type": "disconnect",
                        "code": "serial_disconnected",
                        "message": "Serial port disconnected",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="VocabZero Serial - NDJSON serial translator")
    parser.add_argument("--port", required=True, help="Serial port name, for example COM3 or /dev/ttyUSB0")
    parser.add_argument("--baud-rate", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--dictionary", default="lexicon.json", help="Path to the dictionary JSON file")
    parser.add_argument("--vector-db", default=None, help="Path to the vector database")
    parser.add_argument("--read-timeout", type=float, default=1.0, help="Serial read timeout in seconds")
    parser.add_argument("--write-timeout", type=float, default=1.0, help="Serial write timeout in seconds")
    parser.add_argument("--reconnect-delay", type=float, default=2.0, help="Reconnect delay in seconds")
    parser.add_argument("--feedback-timeout", type=float, default=30.0, help="Feedback wait timeout in seconds")
    parser.add_argument("--max-reconnect-attempts", type=int, default=3, help="Maximum reconnect attempts")

    args = parser.parse_args()
    engine = build_engine(dictionary_path=args.dictionary, vector_db_path=args.vector_db)
    config = SerialConfig(
        port=args.port,
        baud_rate=args.baud_rate,
        read_timeout=args.read_timeout,
        write_timeout=args.write_timeout,
        reconnect_delay=args.reconnect_delay,
        feedback_timeout=args.feedback_timeout,
        max_reconnect_attempts=args.max_reconnect_attempts,
    )
    SerialInterface(engine, config).run()
