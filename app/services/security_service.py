"""Input and retrieved-content safety controls for investigation workflows."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal


logger = logging.getLogger(__name__)

PROMPT_INJECTION_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
    r"reveal\s+(?:the\s+)?(?:system|developer)\s+(?:prompt|message|instructions)",
    r"(?:system|developer)\s+prompt",
    r"follow\s+these\s+instructions\s+instead",
)
JAILBREAK_PATTERNS = (
    r"jailbreak",
    r"do\s+anything\s+now",
    r"bypass\s+(?:the\s+)?(?:safety|guardrails|restrictions)",
    r"unrestricted\s+mode",
)
SENSITIVE_REQUEST_PATTERNS = (
    r"\b(?:api[ _-]?key|password|credential|access[ _-]?token|secret)\b",
    r"\b(?:social\s+security|ssn|cvv|credit\s+card\s+number|personal\s+identifiable\s+information|\bpii\b)\b",
    r"\b(?:customer|user)\s+(?:id|identifier)\b",
)
SENSITIVE_VALUE_PATTERNS = (
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b(?:\d[ -]*?){13,16}\b",
    r"\bsk-[A-Za-z0-9_-]{10,}\b",
    r"\b(?:api[ _-]?key|password|secret)\s*[:=]",
)


@dataclass(frozen=True)
class SecurityDecision:
    """The sanitized request fields and a possible unsafe-content classification."""

    question: str
    transaction_description: str
    blocked_category: Literal[
        "prompt_injection",
        "jailbreak",
        "sensitive_data_request",
        "unsafe_content",
    ] | None = None

    @property
    def allowed(self) -> bool:
        """Whether the request may proceed to retrieval or agent tools."""

        return self.blocked_category is None


class SecurityService:
    """Block unsafe instructions and keep only non-sensitive security event metadata."""

    def __init__(self) -> None:
        """Create an in-memory, content-free security event log."""

        self._events: list[dict[str, str]] = []
        self._event_lock = Lock()

    def validate_investigation_input(
        self,
        question: str,
        transaction_description: str,
        endpoint: str,
    ) -> SecurityDecision:
        """Sanitize request text and block prompt attacks or sensitive-data requests."""

        safe_question = self.sanitize_text(question, endpoint)
        safe_description = self.sanitize_text(transaction_description, endpoint)
        combined_text = f"{safe_question}\n{safe_description}"
        category = self._unsafe_category(combined_text)
        if category is not None:
            self._record_event("blocked", category, endpoint)
        return SecurityDecision(safe_question, safe_description, category)

    def sanitize_text(self, text: str, endpoint: str) -> str:
        """Remove control characters and invisible Unicode controls without logging text."""

        sanitized = re.sub(r"[\x00-\x1f\x7f\u200b-\u200d\ufeff]", " ", text)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        if sanitized != text:
            self._record_event("sanitized", "unsafe_content", endpoint)
        return sanitized

    def filter_untrusted_documents(
        self,
        documents: list[tuple[Any, float]],
        endpoint: str = "rag_retrieval",
    ) -> list[tuple[Any, float]]:
        """Exclude retrieved documents that contain instruction-hijack or sensitive content."""

        safe_documents: list[tuple[Any, float]] = []
        for document, score in documents:
            content = getattr(document, "page_content", "")
            if self._unsafe_category(content) is None:
                safe_documents.append((document, score))
            else:
                self._record_event("blocked", "unsafe_content", endpoint)
        return safe_documents

    def generated_output_is_safe(self, text: str, endpoint: str = "rag_generation") -> bool:
        """Reject generated text that appears to contain unsafe instructions or sensitive values."""

        if self._unsafe_category(text) is not None or any(
            re.search(pattern, text, flags=re.IGNORECASE) for pattern in SENSITIVE_VALUE_PATTERNS
        ):
            self._record_event("blocked", "unsafe_content", endpoint)
            return False
        return True

    def recent_events(self) -> list[dict[str, str]]:
        """Return copies of content-free security event metadata for local diagnostics."""

        with self._event_lock:
            return list(self._events)

    @staticmethod
    def blocked_response(category: str) -> dict[str, str | bool]:
        """Build a generic safe response that never echoes submitted content."""

        return {
            "blocked": True,
            "category": category,
            "message": "Request blocked by safety controls.",
        }

    @staticmethod
    def _unsafe_category(
        text: str,
    ) -> Literal["prompt_injection", "jailbreak", "sensitive_data_request"] | None:
        """Classify unsafe text with deterministic, case-insensitive safety patterns."""

        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS):
            return "prompt_injection"
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in JAILBREAK_PATTERNS):
            return "jailbreak"
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SENSITIVE_REQUEST_PATTERNS):
            return "sensitive_data_request"
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SENSITIVE_VALUE_PATTERNS):
            return "sensitive_data_request"
        return None

    def _record_event(self, action: str, category: str, endpoint: str) -> None:
        """Log event metadata only; request and document contents are intentionally excluded."""

        event = {"action": action, "category": category, "endpoint": endpoint}
        with self._event_lock:
            self._events.append(event)
        logger.warning(
            "security_event action=%s category=%s endpoint=%s",
            action,
            category,
            endpoint,
        )


security_service = SecurityService()
