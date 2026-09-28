"""Schemas for safe blocked responses from investigation endpoints."""

from typing import Literal

from pydantic import BaseModel


SecurityCategory = Literal[
    "prompt_injection",
    "jailbreak",
    "sensitive_data_request",
    "unsafe_content",
]


class SecurityBlockedResponse(BaseModel):
    """A generic response that does not reflect unsafe request content."""

    blocked: Literal[True] = True
    category: SecurityCategory
    message: str = "Request blocked by safety controls."


class SecurityEvent(BaseModel):
    """Content-free metadata describing one security control event."""

    action: Literal["blocked", "sanitized"]
    category: SecurityCategory
    endpoint: str


class SecurityEventsResponse(BaseModel):
    """The in-memory content-free security event log."""

    events: list[SecurityEvent]
