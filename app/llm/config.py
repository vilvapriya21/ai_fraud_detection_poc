"""Environment-backed configuration for switchable LangChain LLM providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


SUPPORTED_PROVIDERS = {"openai", "groq", "ollama"}
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class LlmConfig:
    """Provider-specific connection values without exposing credential contents."""

    provider: str
    openai_api_key: str | None
    openai_model: str | None
    groq_api_key: str | None
    groq_model: str | None
    ollama_base_url: str
    ollama_model: str | None

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "LlmConfig":
        """Read supported provider settings from environment variables."""

        values = environment if environment is not None else os.environ
        return cls(
            provider=values.get("LLM_PROVIDER", "").strip().lower(),
            openai_api_key=cls._optional_value(values.get("OPENAI_API_KEY")),
            openai_model=cls._optional_value(values.get("OPENAI_MODEL")),
            groq_api_key=cls._optional_value(values.get("GROQ_API_KEY")),
            groq_model=cls._optional_value(values.get("GROQ_MODEL")),
            ollama_base_url=values.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()
            or DEFAULT_OLLAMA_BASE_URL,
            ollama_model=cls._optional_value(values.get("OLLAMA_MODEL")),
        )

    @property
    def provider_supported(self) -> bool:
        """Whether the selected provider is one supported by the application."""

        return self.provider in SUPPORTED_PROVIDERS

    @staticmethod
    def _optional_value(value: str | None) -> str | None:
        """Normalize empty environment values to ``None``."""

        if value is None:
            return None
        return value.strip() or None
