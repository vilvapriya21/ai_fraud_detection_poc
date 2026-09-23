"""Tests for switchable provider configuration without real API calls."""

from app.llm.config import LlmConfig
from app.llm import factory


def test_openai_provider_uses_openai_environment_values(monkeypatch: object) -> None:
    """OpenAI selection creates the common chat interface with only configured values."""

    captured: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"provider": "openai"}

    monkeypatch.setattr(factory, "ChatOpenAI", fake_openai)
    config = LlmConfig.from_environment(
        {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"}
    )

    assert factory.create_chat_model(config) == {"provider": "openai"}
    assert captured["model"] == "test-model"
    assert captured["api_key"] == "test-key"


def test_groq_provider_uses_groq_environment_values(monkeypatch: object) -> None:
    """Groq selection uses the LangChain Groq integration without calling its API."""

    captured: dict[str, object] = {}

    def fake_groq(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"provider": "groq"}

    monkeypatch.setattr(factory, "ChatGroq", fake_groq)
    config = LlmConfig.from_environment(
        {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test-key", "GROQ_MODEL": "test-model"}
    )

    assert factory.create_chat_model(config) == {"provider": "groq"}
    assert captured["model"] == "test-model"
    assert captured["api_key"] == "test-key"


def test_ollama_provider_uses_local_base_url(monkeypatch: object) -> None:
    """Ollama selection configures a local client without making a network request."""

    captured: dict[str, object] = {}

    def fake_ollama(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"provider": "ollama"}

    monkeypatch.setattr(factory, "ChatOllama", fake_ollama)
    config = LlmConfig.from_environment(
        {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "OLLAMA_MODEL": "local-test-model",
        }
    )

    assert factory.create_chat_model(config) == {"provider": "ollama"}
    assert captured["base_url"] == "http://localhost:11434"
    assert captured["model"] == "local-test-model"


def test_missing_or_invalid_provider_configuration_returns_none() -> None:
    """Unconfigured providers defer to the existing investigation fallback."""

    assert factory.create_chat_model(LlmConfig.from_environment({"LLM_PROVIDER": "groq"})) is None
    assert factory.create_chat_model(LlmConfig.from_environment({"LLM_PROVIDER": "unknown"})) is None
