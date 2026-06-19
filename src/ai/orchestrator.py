"""Orchestrates one What-If turn: route -> guardrails -> execute -> explain."""
from __future__ import annotations
from typing import Optional

import networkx as nx

from .schemas import AssistantResult
from .tools import TOOL_SPECS, run_tool
from .guardrails import validate_and_clamp
from .llm_client import LLMClient

__all__ = ["run_whatif"]


def run_whatif(
    query: str,
    G: nx.DiGraph,
    router: LLMClient,
    explainer: Optional[LLMClient] = None,
) -> AssistantResult:
    """Map a natural-language query to a tool call, run it, and explain the result."""
    explainer = explainer or router

    selection = router.select_tool(query, TOOL_SPECS)
    safe_args = validate_and_clamp(selection.name, selection.arguments, G)
    metrics = run_tool(selection.name, safe_args, G)
    explanation = explainer.explain(query, selection.name, metrics)

    return AssistantResult(
        query=query,
        tool_name=selection.name,
        arguments=safe_args,
        metrics=metrics,
        explanation=explanation,
    )
