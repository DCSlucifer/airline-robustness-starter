"""Pydantic models for the AI assistant's structured data."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolSelection(BaseModel):
    """A tool chosen by the LLM router, with its raw (unvalidated) arguments."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AssistantResult(BaseModel):
    """The full result of one What-If turn, suitable for rendering and logging."""

    query: str
    tool_name: str
    arguments: dict[str, Any]
    metrics: dict[str, Any]
    explanation: str
