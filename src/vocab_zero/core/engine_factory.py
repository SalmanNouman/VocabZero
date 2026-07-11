from __future__ import annotations

import logging
import os
from pathlib import Path

from vocab_zero.core.dictionary import DictionaryManager
from vocab_zero.core.engine import TranslationEngine
from vocab_zero.core.llm_client import GemmaClient, OpenAICompatibleClient
from vocab_zero.core.models import AudioConfig, TranslationConfig
from vocab_zero.core.vector_db import VectorStoreClient


def build_engine(
    dictionary_path: str = "lexicon.json",
    vector_db_path: str | None = None,
    audio_config: AudioConfig | None = None,
) -> TranslationEngine:
    """Factory function to build a TranslationEngine with configured components.

    Respects the following environment variables:

    - ``LLM_PROVIDER``: ``"gemma"`` to use the local Gemma model or OpenAI-compatible
      endpoint, or ``"openai"`` (default) to use any OpenAI-compatible endpoint.
    - ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` / ``LLM_MODEL_NAME``: used when
      provider is ``"openai"`` or ``"gemma"`` with an OpenAI-compatible endpoint.
    - ``VOCABZERO_CMVN`` / ``VOCABZERO_VTLN`` / ``VOCABZERO_LIFTERING`` /
      ``VOCABZERO_DELTAS``: toggle audio normalization layers (1/0).
    - ``VOCABZERO_DTW_THRESHOLD_36`` / ``VOCABZERO_DTW_THRESHOLD_12``: DTW
      distance thresholds for 36-dim and 12-dim features.
    - ``VOCABZERO_MIN_CONFIDENCE``: minimum confidence gate for accepting matches.

    If a ``calibration.json`` file exists alongside the dictionary, its threshold
    values are loaded and override the defaults / env vars.
    """
    _logger = logging.getLogger(__name__)
    dictionary = DictionaryManager(path=dictionary_path)

    vector_store: VectorStoreClient | None = None
    if vector_db_path:
        vector_store = VectorStoreClient(persist_dir=vector_db_path)

    config = TranslationConfig.from_env()
    effective_audio_config = audio_config or AudioConfig.from_env()

    calibration_path = Path(dictionary_path).parent / "calibration.json"
    if calibration_path.exists():
        effective_audio_config = AudioConfig.load_calibration(calibration_path, effective_audio_config)
        _logger.info("Loaded calibration from %s", calibration_path)

    llm_client: GemmaClient | OpenAICompatibleClient | None = None
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "gemma":
        llm_client = GemmaClient(config=config)
    elif config.api_key or config.base_url:
        llm_client = OpenAICompatibleClient(config)

    return TranslationEngine(
        dictionary=dictionary,
        vector_store=vector_store,
        llm_client=llm_client,
        config=config,
        audio_config=effective_audio_config,
    )
