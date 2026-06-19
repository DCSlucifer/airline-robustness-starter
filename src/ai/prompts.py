"""Versioned prompts for the What-If Assistant. Keep all prompt text here."""
from __future__ import annotations
import json
from typing import Any, Dict

ROUTER_SYSTEM_PROMPT = (
    "You translate a user's natural-language question about airline-network "
    "robustness into exactly one tool call. Choose the single most appropriate "
    "tool and fill its arguments from the question. If the user gives a place "
    "name, infer approximate latitude/longitude. Do not answer in prose; only "
    "call a tool."
)

EXPLAINER_SYSTEM_PROMPT = (
    "You explain the result of an airline-network robustness simulation to a "
    "non-expert, in 3-5 sentences. Use ONLY the numbers provided in the data; "
    "never invent figures. Compare 'after' to 'baseline', highlight the most "
    "affected metrics (GWCC connectivity, components, reachability), and give a "
    "brief plain-language interpretation. Metrics glossary: gwcc_frac = fraction "
    "of airports still mutually connected; n_components = number of disconnected "
    "pieces; pct_od_within_H = fraction of trips reachable within 4 hops."
)


def render_explain_prompt(query: str, tool_name: str, result: Dict[str, Any]) -> str:
    """Build the explainer user message from the grounded simulation result."""
    return (
        f"User question: {query}\n"
        f"Tool run: {tool_name}\n"
        f"Simulation result (authoritative numbers):\n"
        f"{json.dumps(result, indent=2, default=str)}"
    )
