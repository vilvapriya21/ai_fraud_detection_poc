"""Small factory for provider-neutral LangChain chat-model clients."""

from __future__ import annotations

from typing import Any

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.llm.config import LlmConfig


def create_chat_model(config: LlmConfig | None = None) -> Any | None:
    """Create the selected chat model or return ``None`` when it is not configured."""

    selected = config or LlmConfig.from_environment()
    if not selected.provider_supported:
        return None
    if selected.provider == "openai":
        return _openai_model(selected)
    if selected.provider == "groq":
        return _groq_model(selected)
    return _ollama_model(selected)


def _openai_model(config: LlmConfig) -> ChatOpenAI | None:
    """Create an OpenAI LangChain client when both credential and model are configured."""

    if not config.openai_api_key or not config.openai_model:
        return None
    return ChatOpenAI(
        model=config.openai_model,
        api_key=config.openai_api_key,
        temperature=0,
        max_tokens=300,
        timeout=15,
    )


def _groq_model(config: LlmConfig) -> ChatGroq | None:
    """Create a Groq LangChain client when both credential and model are configured."""

    if not config.groq_api_key or not config.groq_model:
        return None
    return ChatGroq(
        model=config.groq_model,
        api_key=config.groq_api_key,
        temperature=0,
        max_tokens=300,
        timeout=15,
    )


def _ollama_model(config: LlmConfig) -> ChatOllama | None:
    """Create an Ollama LangChain client when a local model is configured."""

    if not config.ollama_model:
        return None
    return ChatOllama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        temperature=0,
        num_predict=300,
    )
