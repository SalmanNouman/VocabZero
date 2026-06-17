from __future__ import annotations

import os
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
