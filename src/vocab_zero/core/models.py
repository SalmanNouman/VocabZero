from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

TranslationStatus = Literal["translated", "low_confidence", "requires_feedback", "feedback_declined", "learned", "error"]
TranslationSource = Literal["dictionary", "llm_inference", "human_feedback", "none"]


class FeedbackRequest(BaseModel):
    source_term: str
    context: str | None = None
    candidate_matches: list[str] = Field(default_factory=list)
    reason: str = ""


class TranslationResult(BaseModel):
    translated_text: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: TranslationSource = "none"
    status: TranslationStatus = "error"
    context_used: list[str] = Field(default_factory=list)
    feedback_request: FeedbackRequest | None = None
    error_code: str | None = None
    error_message: str | None = None


class LLMResponse(BaseModel):
    translation: str
    reasoning: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TranslationConfig(BaseModel):
    high_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    low_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    rag_k: int = Field(default=5, ge=1, le=50)
    timeout_seconds: float = Field(default=30.0, ge=1.0)
    retry_count: int = Field(default=1, ge=0, le=3)
    llm_confidence_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    vector_confidence_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    max_context_length: int = Field(default=500, ge=100)
    model_name: str = Field(default="gpt-4o-mini")
    api_key: str | None = None
    base_url: str | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> TranslationConfig:
        if self.high_threshold <= self.low_threshold:
            raise ValueError(
                f"high_threshold ({self.high_threshold}) must be strictly greater than low_threshold ({self.low_threshold})"
            )
        return self

    @model_validator(mode="after")
    def validate_weights(self) -> TranslationConfig:
        total = self.llm_confidence_weight + self.vector_confidence_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"llm_confidence_weight ({self.llm_confidence_weight}) + vector_confidence_weight ({self.vector_confidence_weight}) must equal 1.0, got {total}"
            )
        return self

    @classmethod
    def from_env(cls) -> TranslationConfig:
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
        )


class AudioConfig(BaseModel):
    sample_rate: int = Field(default=16000, ge=8000)
    num_cepstrum: int = Field(default=13, ge=2, le=40)
    num_filters: int = Field(default=26, ge=10, le=80)
    nfft: int = Field(default=512, ge=128)
    preemph: float = Field(default=0.97, ge=0.0, le=1.0)
    use_cmvn: bool = True
    use_vtln: bool = True
    use_liftering: bool = True
    lifter_coef: int = Field(default=22, ge=1, le=50)
    use_deltas: bool = True
    # Calibrated via /api/calibrate or set via VOCABZERO_DTW_THRESHOLD_36/12
    # env vars. Same-word pairs should typically fall below 1.0; different-word
    # pairs above 2.0. Use the calibration wizard to find exact values for your
    # mic and environment.
    dtw_threshold_36: float = Field(default=1.8, gt=0.0)
    dtw_threshold_12: float = Field(default=1.2, gt=0.0)
    min_confidence_gate: float = Field(default=0.6, ge=0.0, le=1.0)
    ambiguity_margin_ratio: float = Field(default=0.15, gt=0.0, le=1.0)
    ambiguity_confidence_floor: float = Field(default=0.4, ge=0.0, le=1.0)
    dtw_band_ratio: float = Field(default=0.2, ge=0.0)
    max_length_ratio: float = Field(default=2.5, ge=1.0)
    template_agg_k: int = Field(default=3, ge=1)

    @property
    def dtw_threshold(self) -> float:
        return self.dtw_threshold_36 if self.use_deltas else self.dtw_threshold_12

    @classmethod
    def from_env(cls) -> AudioConfig:
        def _env_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        def _env_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            return float(raw.strip())

        def _env_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None:
                return default
            return int(raw.strip())

        return cls(
            use_cmvn=_env_bool("VOCABZERO_CMVN", True),
            use_vtln=_env_bool("VOCABZERO_VTLN", True),
            use_liftering=_env_bool("VOCABZERO_LIFTERING", True),
            use_deltas=_env_bool("VOCABZERO_DELTAS", True),
            dtw_threshold_36=_env_float("VOCABZERO_DTW_THRESHOLD_36", 1.8),
            dtw_threshold_12=_env_float("VOCABZERO_DTW_THRESHOLD_12", 1.2),
            min_confidence_gate=_env_float("VOCABZERO_MIN_CONFIDENCE", 0.6),
            ambiguity_margin_ratio=_env_float("VOCABZERO_AMBIGUITY_MARGIN_RATIO", 0.15),
            ambiguity_confidence_floor=_env_float("VOCABZERO_AMBIGUITY_CONFIDENCE_FLOOR", 0.4),
            dtw_band_ratio=_env_float("VOCABZERO_DTW_BAND_RATIO", 0.2),
            max_length_ratio=_env_float("VOCABZERO_MAX_LENGTH_RATIO", 2.5),
            template_agg_k=_env_int("VOCABZERO_TEMPLATE_AGG_K", 3),
        )

    def save_calibration(self, path: Path) -> None:
        """Persist DTW thresholds and confidence gate to a calibration file."""
        data = {
            "dtw_threshold_36": self.dtw_threshold_36,
            "dtw_threshold_12": self.dtw_threshold_12,
            "min_confidence_gate": self.min_confidence_gate,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp_path.replace(path)

    @classmethod
    def load_calibration(cls, path: Path, base: AudioConfig) -> AudioConfig:
        """Load calibration from file, merging with a base AudioConfig."""
        logger = logging.getLogger(__name__)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            updates: dict[str, float] = {}
            if "dtw_threshold_36" in data:
                updates["dtw_threshold_36"] = float(data["dtw_threshold_36"])
            if "dtw_threshold_12" in data:
                updates["dtw_threshold_12"] = float(data["dtw_threshold_12"])
            if "min_confidence_gate" in data:
                updates["min_confidence_gate"] = float(data["min_confidence_gate"])
            if updates:
                logger.info("Loaded calibration from %s: %s", path, updates)
                return base.model_copy(update=updates)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to load calibration from %s: %s", path, exc)
        return base
