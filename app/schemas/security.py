"""Schemas for safe blocked responses from investigation endpoints."""

from typing import Literal

from pydantic import BaseModel


class SecurityBlockedResponse(BaseModel):
    """A generic response that does not reflect unsafe request content."""

    blocked: Literal[True] = True
    category: Literal["prompt_injection", "jailbreak", "sensitive_data_request", "unsafe_content"]
    message: str = "Request blocked by safety controls."
