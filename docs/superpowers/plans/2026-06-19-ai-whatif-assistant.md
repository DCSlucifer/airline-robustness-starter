# AI What-If Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a natural-language "Ask AI" layer that maps user questions to the existing attack/defense simulation functions via LLM function calling, runs the real simulation, and explains the grounded result.

**Architecture:** A new `src/ai/` package, fully decoupled from the simulation code. An `LLMClient` interface hides the provider; a tool registry maps JSON tool schemas to the existing simulation functions; an orchestrator runs route → guardrails → execute → explain. All logic is testable offline against a `FakeLLMClient`; only the thin `ClaudeClient` adapter touches the network.

**Tech Stack:** Python 3.10+, `anthropic` SDK (default provider, swappable), `pydantic` (already transitively available via other deps; pinned here), `networkx`, `pytest`, Streamlit.

## Global Constraints

- Python 3.10+ (matches existing `pyproject.toml`).
- The LLM **never** computes or invents a numeric metric — every number comes from `src/metrics.topological_report`. The LLM only selects a tool + arguments and writes prose.
- Default models: router `claude-haiku-4-5`, explainer `claude-sonnet-4-6`. Model IDs are exact strings, no date suffixes.
- Provider is swappable: all Anthropic SDK calls live only in `src/ai/llm_client.py::ClaudeClient`. No other module imports `anthropic`.
- Existing simulation modules (`attacks.py`, `defenses.py`, `metrics.py`, etc.) are **not modified** by Phases 1-3 except imports.
- Every parameter the LLM supplies is validated and clamped in `src/ai/guardrails.py` before any simulation runs. No LLM-supplied value can break the app.
- New code uses type hints and module docstrings consistent with the existing codebase.
- This plan covers the **core assistant (Phases 1-3)** from the spec. Phase 0 (repo hardening) and Phase 4 (findings writeup) are separate plans.

---

## File Structure

**Phase 1 — tool layer + structured output (no UI, no network needed for tests):**
- Create `src/ai/__init__.py` — package marker, exports.
- Create `src/ai/schemas.py` — Pydantic models `ToolSelection`, `AssistantResult`.
- Create `src/ai/tools.py` — `TOOL_SPECS` (Anthropic tool schemas) + `run_tool(name, args, G)` dispatch to existing simulation functions.
- Create `src/ai/guardrails.py` — `validate_and_clamp(name, args, G)` + `GuardrailError`.
- Create `src/ai/prompts.py` — `ROUTER_SYSTEM_PROMPT`, `EXPLAINER_SYSTEM_PROMPT`, `render_explain_prompt`.
- Create `src/ai/llm_client.py` — `LLMClient` protocol, `FakeLLMClient`, `ClaudeClient`.

**Phase 2 — orchestrator + Streamlit panel:**
- Create `src/ai/orchestrator.py` — `run_whatif(query, G, router, explainer)`.
- Modify `src/app/streamlit_app.py` — add an "Ask AI" panel that calls the orchestrator.

**Phase 3 — eval harness + observability:**
- Create `data/ai_golden_set.json` — labelled NL queries.
- Create `src/ai/eval.py` — golden-set runner + accuracy scoring.
- Create `src/ai/tracing.py` — per-turn structured logging.

**Tests:**
- Create `tests/ai/__init__.py`, `tests/ai/conftest.py` (shared tiny-graph fixture).
- Create `tests/ai/test_schemas.py`, `test_tools.py`, `test_guardrails.py`, `test_llm_client.py`, `test_orchestrator.py`, `test_eval.py`, `test_tracing.py`.

---

## Task 1: Pydantic schemas

**Files:**
- Create: `src/ai/__init__.py`
- Create: `src/ai/schemas.py`
- Create: `tests/ai/__init__.py`
- Test: `tests/ai/test_schemas.py`

**Interfaces:**
- Produces: `ToolSelection(name: str, arguments: dict[str, Any])`; `AssistantResult(query: str, tool_name: str, arguments: dict, metrics: dict, explanation: str)`. Later tasks import both from `src.ai.schemas`.

- [ ] **Step 1: Write the failing test**

Create `tests/ai/__init__.py` as an empty file, then create `tests/ai/test_schemas.py`:

```python
from src.ai.schemas import ToolSelection, AssistantResult


def test_tool_selection_holds_name_and_arguments():
    sel = ToolSelection(name="targeted_attack", arguments={"metric": "degree", "k": 10})
    assert sel.name == "targeted_attack"
    assert sel.arguments["k"] == 10


def test_assistant_result_roundtrips_to_dict():
    res = AssistantResult(
        query="what if we lose 5 hubs?",
        tool_name="targeted_attack",
        arguments={"metric": "degree", "k": 5},
        metrics={"baseline": {}, "after": {}},
        explanation="Connectivity dropped.",
    )
    d = res.model_dump()
    assert d["tool_name"] == "targeted_attack"
    assert d["explanation"] == "Connectivity dropped."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ai/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ai/__init__.py`:

```python
"""AI What-If Assistant: natural-language interface over the simulation tools."""
```

Create `src/ai/schemas.py`:

```python
"""Pydantic models for the AI assistant's structured data."""
from __future__ import annotations
from typing import Any, Dict

from pydantic import BaseModel, Field


class ToolSelection(BaseModel):
    """A tool chosen by the LLM router, with its raw (unvalidated) arguments."""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class AssistantResult(BaseModel):
    """The full result of one What-If turn, suitable for rendering and logging."""
    query: str
    tool_name: str
    arguments: Dict[str, Any]
    metrics: Dict[str, Any]
    explanation: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ai/test_schemas.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai/__init__.py src/ai/schemas.py tests/ai/__init__.py tests/ai/test_schemas.py
git commit -m "feat(ai): add Pydantic schemas for assistant tool selection and result"
```

---

## Task 2: Tool registry and dispatch

**Files:**
- Create: `src/ai/tools.py`
- Create: `tests/ai/conftest.py`
- Test: `tests/ai/test_tools.py`

**Interfaces:**
- Consumes: `topological_report` from `src.metrics`; `targeted_node_removal`, `edge_betweenness_attack`, `geographic_attack_radius`, `community_bridge_attack` from `src.attacks`; `greedy_edge_addition` from `src.defenses`.
- Produces: `TOOL_SPECS: list[dict]` (Anthropic tool schemas, each with `name`, `description`, `input_schema`, `strict: True`); `TOOL_NAMES: set[str]`; `run_tool(name: str, args: dict, G: nx.DiGraph) -> dict` returning `{"baseline": dict, "after": dict, "removed_nodes": list, "removed_edges": list, "added_edges": list}` (keys present per tool).

- [ ] **Step 1: Write the failing test**

Create `tests/ai/conftest.py`:

```python
"""Shared fixtures for AI assistant tests."""
import networkx as nx
import pytest


@pytest.fixture
def small_graph() -> nx.DiGraph:
    """A tiny airline-like directed graph with coordinates."""
    G = nx.DiGraph()
    coords = {
        "AAA": (40.0, -74.0),
        "BBB": (41.0, -73.0),
        "CCC": (34.0, -118.0),
        "DDD": (51.5, -0.1),
        "EEE": (48.8, 2.3),
        "FFF": (35.7, 139.7),
    }
    for iata, (lat, lon) in coords.items():
        G.add_node(iata, lat=lat, lon=lon, name=iata)
    edges = [
        ("AAA", "BBB"), ("BBB", "AAA"), ("AAA", "CCC"), ("CCC", "AAA"),
        ("CCC", "FFF"), ("FFF", "CCC"), ("DDD", "EEE"), ("EEE", "DDD"),
        ("BBB", "DDD"), ("DDD", "BBB"), ("AAA", "DDD"), ("DDD", "AAA"),
    ]
    G.add_edges_from(edges)
    return G
```

Create `tests/ai/test_tools.py`:

```python
import networkx as nx

from src.ai.tools import TOOL_SPECS, TOOL_NAMES, run_tool


def test_tool_specs_are_well_formed():
    assert TOOL_NAMES  # non-empty
    for spec in TOOL_SPECS:
        assert spec["name"] in TOOL_NAMES
        assert "description" in spec and spec["description"]
        schema = spec["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert spec["strict"] is True


def test_run_targeted_attack_returns_grounded_metrics(small_graph):
    result = run_tool("targeted_attack", {"metric": "degree", "k": 2}, small_graph)
    assert set(result) >= {"baseline", "after", "removed_nodes"}
    assert len(result["removed_nodes"]) == 2
    # Removing nodes never increases node count.
    assert result["after"]["n_nodes"] < result["baseline"]["n_nodes"]


def test_run_geographic_attack_removes_nearby_nodes(small_graph):
    # ~ New York area; AAA(40,-74) and BBB(41,-73) are close.
    result = run_tool(
        "geographic_attack", {"lat": 40.5, "lon": -73.5, "radius_km": 300}, small_graph
    )
    assert "AAA" in result["removed_nodes"]
    assert "BBB" in result["removed_nodes"]
    assert "FFF" not in result["removed_nodes"]  # Tokyo, far away


def test_run_defend_adds_edges(small_graph):
    result = run_tool("defend", {"budget": 1, "max_distance_km": 20000}, small_graph)
    assert "added_edges" in result
    assert "after" in result


def test_run_tool_rejects_unknown_tool(small_graph):
    import pytest
    with pytest.raises(KeyError):
        run_tool("nonexistent", {}, small_graph)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ai/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai.tools'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ai/tools.py`:

```python
"""Tool registry mapping LLM-callable schemas to the real simulation functions.

The LLM selects a tool and arguments; ``run_tool`` executes the corresponding
simulation on the graph and returns grounded metrics. The LLM never computes
these numbers.
"""
from __future__ import annotations
from typing import Any, Dict, List

import networkx as nx

from ..metrics import topological_report
from ..attacks import (
    targeted_node_removal,
    edge_betweenness_attack,
    geographic_attack_radius,
    community_bridge_attack,
)
from ..defenses import greedy_edge_addition

__all__ = ["TOOL_SPECS", "TOOL_NAMES", "run_tool"]

_METRIC_ENUM = ["degree", "betweenness", "pagerank"]

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "targeted_attack",
        "description": (
            "Remove the k most central airports (hubs) ranked by a centrality "
            "metric. Use for scenarios like 'what if we lose the busiest hubs' "
            "or 'attack the most connected airports'."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": _METRIC_ENUM},
                "k": {"type": "integer", "description": "Number of airports to remove"},
            },
            "required": ["metric", "k"],
            "additionalProperties": False,
        },
    },
    {
        "name": "geographic_attack",
        "description": (
            "Disable all airports within a radius (km) of a latitude/longitude. "
            "Use for regional disasters: hurricanes, earthquakes, 'what if a "
            "storm hits a region'."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "radius_km": {"type": "number"},
            },
            "required": ["lat", "lon", "radius_km"],
            "additionalProperties": False,
        },
    },
    {
        "name": "edge_attack",
        "description": (
            "Remove the m highest-betweenness routes (critical bridge connections). "
            "Use for 'what if key routes are cut' scenarios."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "m": {"type": "integer", "description": "Number of routes to remove"},
            },
            "required": ["m"],
            "additionalProperties": False,
        },
    },
    {
        "name": "community_bridge_attack",
        "description": (
            "Remove m routes that bridge different network communities/regions. "
            "Use for 'what if connections between regions are severed'."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "m": {"type": "integer", "description": "Number of bridge routes to remove"},
            },
            "required": ["m"],
            "additionalProperties": False,
        },
    },
    {
        "name": "defend",
        "description": (
            "Add up to 'budget' new routes (within max_distance_km) to improve "
            "connectivity. Use for 'how can we make the network more resilient' "
            "or 'add routes to recover'."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "budget": {"type": "integer", "description": "Number of routes to add"},
                "max_distance_km": {"type": "number"},
            },
            "required": ["budget", "max_distance_km"],
            "additionalProperties": False,
        },
    },
]

TOOL_NAMES = {spec["name"] for spec in TOOL_SPECS}


def run_tool(name: str, args: Dict[str, Any], G: nx.DiGraph) -> Dict[str, Any]:
    """Execute the named simulation tool with validated args; return grounded metrics."""
    if name not in TOOL_NAMES:
        raise KeyError(f"Unknown tool: {name}")

    baseline = topological_report(G, fast_mode=True)

    if name == "targeted_attack":
        H, log = targeted_node_removal(
            G, k=args["k"], metric=args["metric"], adaptive=True, fast_mode=True
        )
        removed = [e["removed_node"] for e in log if e.get("removed_node")]
        return {
            "baseline": baseline,
            "after": topological_report(H, fast_mode=True),
            "removed_nodes": removed,
        }

    if name == "geographic_attack":
        H, info = geographic_attack_radius(
            G, (args["lat"], args["lon"]), args["radius_km"]
        )
        return {
            "baseline": baseline,
            "after": topological_report(H, fast_mode=True),
            "removed_nodes": list(info.get("removed_nodes", [])),
        }

    if name == "edge_attack":
        H, log = edge_betweenness_attack(G, m=args["m"], adaptive=True, fast_mode=True)
        removed = [e["removed_edge"] for e in log if e.get("removed_edge")]
        return {
            "baseline": baseline,
            "after": topological_report(H, fast_mode=True),
            "removed_edges": removed,
        }

    if name == "community_bridge_attack":
        H, info = community_bridge_attack(G, m=args["m"])
        return {
            "baseline": baseline,
            "after": topological_report(H, fast_mode=True),
            "removed_edges": list(info.get("removed_edges", [])),
        }

    # name == "defend"
    H, log = greedy_edge_addition(
        G, budget=args["budget"], max_distance_km=args["max_distance_km"], fast_mode=True
    )
    added = [e for entry in log for e in entry.get("added_edges", [])]
    return {
        "baseline": baseline,
        "after": topological_report(H, fast_mode=True),
        "added_edges": added,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ai/test_tools.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai/tools.py tests/ai/conftest.py tests/ai/test_tools.py
git commit -m "feat(ai): add tool registry mapping LLM schemas to simulation functions"
```

---

## Task 3: Guardrails

**Files:**
- Create: `src/ai/guardrails.py`
- Test: `tests/ai/test_guardrails.py`

**Interfaces:**
- Consumes: `TOOL_NAMES` from `src.ai.tools`.
- Produces: `GuardrailError(ValueError)`; `validate_and_clamp(name: str, args: dict, G: nx.DiGraph) -> dict` returning a new clamped args dict. Raises `GuardrailError` for unknown tool, missing required key, or invalid enum.

- [ ] **Step 1: Write the failing test**

Create `tests/ai/test_guardrails.py`:

```python
import pytest

from src.ai.guardrails import validate_and_clamp, GuardrailError


def test_caps_k_at_node_count(small_graph):
    out = validate_and_clamp("targeted_attack", {"metric": "degree", "k": 9999}, small_graph)
    assert out["k"] == small_graph.number_of_nodes()


def test_floors_k_at_one(small_graph):
    out = validate_and_clamp("targeted_attack", {"metric": "degree", "k": 0}, small_graph)
    assert out["k"] == 1


def test_caps_budget_at_ten(small_graph):
    out = validate_and_clamp(
        "defend", {"budget": 50, "max_distance_km": 3000}, small_graph
    )
    assert out["budget"] == 10


def test_rejects_unknown_tool(small_graph):
    with pytest.raises(GuardrailError):
        validate_and_clamp("rm_rf", {}, small_graph)


def test_rejects_bad_metric(small_graph):
    with pytest.raises(GuardrailError):
        validate_and_clamp("targeted_attack", {"metric": "evil", "k": 3}, small_graph)


def test_rejects_missing_required_arg(small_graph):
    with pytest.raises(GuardrailError):
        validate_and_clamp("targeted_attack", {"metric": "degree"}, small_graph)


def test_clamps_radius_to_positive(small_graph):
    out = validate_and_clamp(
        "geographic_attack", {"lat": 40.0, "lon": -74.0, "radius_km": -5}, small_graph
    )
    assert out["radius_km"] >= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ai/test_guardrails.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai.guardrails'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ai/guardrails.py`:

```python
"""Validation and clamping of LLM-supplied tool arguments.

No LLM-supplied value reaches a simulation function without passing through
``validate_and_clamp``. This is the production-safety boundary.
"""
from __future__ import annotations
from typing import Any, Dict

import networkx as nx

from .tools import TOOL_NAMES

__all__ = ["validate_and_clamp", "GuardrailError"]

_METRICS = {"degree", "betweenness", "pagerank"}
_MAX_BUDGET = 10
_MAX_RADIUS_KM = 20000.0
_MAX_DISTANCE_KM = 20000.0

_REQUIRED = {
    "targeted_attack": ("metric", "k"),
    "geographic_attack": ("lat", "lon", "radius_km"),
    "edge_attack": ("m",),
    "community_bridge_attack": ("m",),
    "defend": ("budget", "max_distance_km"),
}


class GuardrailError(ValueError):
    """Raised when LLM-supplied arguments are unusable or unsafe."""


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def validate_and_clamp(name: str, args: Dict[str, Any], G: nx.DiGraph) -> Dict[str, Any]:
    """Return a new args dict with safe, in-range values. Raise GuardrailError if invalid."""
    if name not in TOOL_NAMES:
        raise GuardrailError(f"Tool '{name}' is not in the allowed tool list")

    for key in _REQUIRED[name]:
        if key not in args:
            raise GuardrailError(f"Missing required argument '{key}' for tool '{name}'")

    out: Dict[str, Any] = dict(args)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if "metric" in out and out["metric"] not in _METRICS:
        raise GuardrailError(f"Invalid metric '{out['metric']}'")

    if "k" in out:
        out["k"] = int(_clamp(int(out["k"]), 1, max(1, n_nodes)))
    if "m" in out:
        out["m"] = int(_clamp(int(out["m"]), 1, max(1, n_edges)))
    if "budget" in out:
        out["budget"] = int(_clamp(int(out["budget"]), 1, _MAX_BUDGET))
    if "radius_km" in out:
        out["radius_km"] = _clamp(float(out["radius_km"]), 1.0, _MAX_RADIUS_KM)
    if "max_distance_km" in out:
        out["max_distance_km"] = _clamp(float(out["max_distance_km"]), 100.0, _MAX_DISTANCE_KM)

    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ai/test_guardrails.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai/guardrails.py tests/ai/test_guardrails.py
git commit -m "feat(ai): add guardrails to validate and clamp LLM tool arguments"
```

---

## Task 4: Prompts and LLM client (interface + FakeLLMClient + ClaudeClient)

**Files:**
- Create: `src/ai/prompts.py`
- Create: `src/ai/llm_client.py`
- Test: `tests/ai/test_llm_client.py`

**Interfaces:**
- Consumes: `ToolSelection` from `src.ai.schemas`.
- Produces:
  - `prompts.ROUTER_SYSTEM_PROMPT: str`, `prompts.EXPLAINER_SYSTEM_PROMPT: str`, `prompts.render_explain_prompt(query: str, tool_name: str, result: dict) -> str`.
  - `llm_client.LLMClient` (Protocol with `select_tool(query: str, tools: list[dict]) -> ToolSelection` and `explain(query: str, tool_name: str, result: dict) -> str`).
  - `llm_client.FakeLLMClient(selection: ToolSelection, explanation: str = "...")` implementing both methods with canned values.
  - `llm_client.ClaudeClient(api_key: str | None = None, router_model="claude-haiku-4-5", explainer_model="claude-sonnet-4-6")`. Its constructor accepts an injectable `_client` for testing.

- [ ] **Step 1: Write the failing test**

Create `tests/ai/test_llm_client.py`:

```python
from src.ai.schemas import ToolSelection
from src.ai.llm_client import FakeLLMClient, ClaudeClient
from src.ai.prompts import render_explain_prompt


def test_fake_client_returns_canned_selection_and_explanation():
    fake = FakeLLMClient(
        selection=ToolSelection(name="targeted_attack", arguments={"metric": "degree", "k": 3}),
        explanation="Hubs removed; connectivity fell.",
    )
    sel = fake.select_tool("lose 3 hubs", tools=[])
    assert sel.name == "targeted_attack"
    assert fake.explain("q", "targeted_attack", {}) == "Hubs removed; connectivity fell."


def test_render_explain_prompt_includes_metrics():
    prompt = render_explain_prompt(
        "what if?", "targeted_attack", {"baseline": {"gwcc_frac": 1.0}, "after": {"gwcc_frac": 0.6}}
    )
    assert "targeted_attack" in prompt
    assert "gwcc_frac" in prompt


class _StubBlock:
    def __init__(self, type, name=None, input=None, text=None):
        self.type = type
        self.name = name
        self.input = input
        self.text = text


class _StubResponse:
    def __init__(self, content):
        self.content = content


class _StubMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _StubAnthropic:
    def __init__(self, response):
        self.messages = _StubMessages(response)


def test_claude_client_parses_tool_use_block():
    response = _StubResponse([_StubBlock("tool_use", name="geographic_attack",
                                         input={"lat": 40.0, "lon": -74.0, "radius_km": 500})])
    client = ClaudeClient(_client=_StubAnthropic(response))
    sel = client.select_tool("storm near NYC", tools=[{"name": "geographic_attack"}])
    assert sel.name == "geographic_attack"
    assert sel.arguments["radius_km"] == 500
    # Router uses the cheap model and forces a tool call.
    assert client._client.messages.last_kwargs["model"] == "claude-haiku-4-5"
    assert client._client.messages.last_kwargs["tool_choice"] == {"type": "any"}


def test_claude_client_parses_explanation_text():
    response = _StubResponse([_StubBlock("text", text="Connectivity dropped 40%.")])
    client = ClaudeClient(_client=_StubAnthropic(response))
    out = client.explain("q", "targeted_attack", {"baseline": {}, "after": {}})
    assert out == "Connectivity dropped 40%."
    assert client._client.messages.last_kwargs["model"] == "claude-sonnet-4-6"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ai/test_llm_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai.prompts'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ai/prompts.py`:

```python
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
```

Create `src/ai/llm_client.py`:

```python
"""Provider-swappable LLM client. Only ClaudeClient touches the network/SDK."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol

from .schemas import ToolSelection
from .prompts import (
    ROUTER_SYSTEM_PROMPT,
    EXPLAINER_SYSTEM_PROMPT,
    render_explain_prompt,
)

__all__ = ["LLMClient", "FakeLLMClient", "ClaudeClient"]


class LLMClient(Protocol):
    """The orchestrator depends only on this interface, never on a provider SDK."""

    def select_tool(self, query: str, tools: List[Dict[str, Any]]) -> ToolSelection: ...

    def explain(self, query: str, tool_name: str, result: Dict[str, Any]) -> str: ...


class FakeLLMClient:
    """Deterministic client for offline tests; returns preset values."""

    def __init__(self, selection: ToolSelection, explanation: str = "(no explanation)"):
        self._selection = selection
        self._explanation = explanation

    def select_tool(self, query: str, tools: List[Dict[str, Any]]) -> ToolSelection:
        return self._selection

    def explain(self, query: str, tool_name: str, result: Dict[str, Any]) -> str:
        return self._explanation


class ClaudeClient:
    """Anthropic-backed client. The only module that imports `anthropic`.

    `_client` is injectable for tests; in production it is constructed from
    the `anthropic` SDK using ANTHROPIC_API_KEY or an explicit api_key (BYOK).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        router_model: str = "claude-haiku-4-5",
        explainer_model: str = "claude-sonnet-4-6",
        _client: Any = None,
    ):
        self.router_model = router_model
        self.explainer_model = explainer_model
        if _client is not None:
            self._client = _client
        else:
            import anthropic  # imported lazily so tests don't require the SDK
            self._client = (
                anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            )

    def select_tool(self, query: str, tools: List[Dict[str, Any]]) -> ToolSelection:
        resp = self._client.messages.create(
            model=self.router_model,
            max_tokens=1024,
            system=ROUTER_SYSTEM_PROMPT,
            tools=tools,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": query}],
        )
        block = next(b for b in resp.content if b.type == "tool_use")
        return ToolSelection(name=block.name, arguments=dict(block.input))

    def explain(self, query: str, tool_name: str, result: Dict[str, Any]) -> str:
        resp = self._client.messages.create(
            model=self.explainer_model,
            max_tokens=1024,
            system=EXPLAINER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": render_explain_prompt(query, tool_name, result)}],
        )
        return next(b.text for b in resp.content if b.type == "text")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ai/test_llm_client.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai/prompts.py src/ai/llm_client.py tests/ai/test_llm_client.py
git commit -m "feat(ai): add prompts and provider-swappable LLM client (Fake + Claude)"
```

---

## Task 5: Orchestrator

**Files:**
- Create: `src/ai/orchestrator.py`
- Test: `tests/ai/test_orchestrator.py`

**Interfaces:**
- Consumes: `TOOL_SPECS`, `run_tool` from `src.ai.tools`; `validate_and_clamp`, `GuardrailError` from `src.ai.guardrails`; `LLMClient` from `src.ai.llm_client`; `ToolSelection`, `AssistantResult` from `src.ai.schemas`.
- Produces: `run_whatif(query: str, G: nx.DiGraph, router: LLMClient, explainer: LLMClient | None = None) -> AssistantResult`. If `explainer` is None, `router` is used for both. Raises `GuardrailError` on unsafe args.

- [ ] **Step 1: Write the failing test**

Create `tests/ai/test_orchestrator.py`:

```python
import pytest

from src.ai.schemas import ToolSelection, AssistantResult
from src.ai.llm_client import FakeLLMClient
from src.ai.orchestrator import run_whatif
from src.ai.guardrails import GuardrailError


def test_run_whatif_end_to_end_grounded(small_graph):
    fake = FakeLLMClient(
        selection=ToolSelection(name="targeted_attack", arguments={"metric": "degree", "k": 2}),
        explanation="Two hubs removed; connectivity decreased.",
    )
    result = run_whatif("what if we lose the 2 biggest hubs?", small_graph, fake)
    assert isinstance(result, AssistantResult)
    assert result.tool_name == "targeted_attack"
    assert result.arguments["k"] == 2
    assert result.metrics["after"]["n_nodes"] < result.metrics["baseline"]["n_nodes"]
    assert result.explanation == "Two hubs removed; connectivity decreased."


def test_run_whatif_clamps_unsafe_arguments(small_graph):
    fake = FakeLLMClient(
        selection=ToolSelection(name="targeted_attack", arguments={"metric": "degree", "k": 9999}),
        explanation="ok",
    )
    result = run_whatif("destroy everything", small_graph, fake)
    assert result.arguments["k"] == small_graph.number_of_nodes()


def test_run_whatif_rejects_unknown_tool(small_graph):
    fake = FakeLLMClient(
        selection=ToolSelection(name="hack_the_mainframe", arguments={}),
        explanation="ok",
    )
    with pytest.raises(GuardrailError):
        run_whatif("do something bad", small_graph, fake)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ai/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai.orchestrator'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ai/orchestrator.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ai/test_orchestrator.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai/orchestrator.py tests/ai/test_orchestrator.py
git commit -m "feat(ai): add orchestrator wiring route -> guardrails -> execute -> explain"
```

---

## Task 6: Streamlit "Ask AI" panel

**Files:**
- Modify: `src/app/streamlit_app.py` (add an "Ask AI" panel in the left column, after the DEFENSE block near line 378)
- Add dependency: `requirements.txt`

**Interfaces:**
- Consumes: `run_whatif` from `src.ai.orchestrator`; `ClaudeClient` from `src.ai.llm_client`; `GuardrailError` from `src.ai.guardrails`.
- Produces: UI only. No new exported symbols.

This task is UI glue; verify by manual smoke test (Step 4) since Streamlit UI is not unit-tested in this repo.

- [ ] **Step 1: Add the `anthropic` dependency**

Edit `requirements.txt` — add this line at the end:

```
anthropic>=0.69
```

- [ ] **Step 2: Add the import**

In `src/app/streamlit_app.py`, after the existing `from src.viz import ...` line (near line 100), add:

```python
from src.ai.orchestrator import run_whatif
from src.ai.llm_client import ClaudeClient
from src.ai.guardrails import GuardrailError
```

- [ ] **Step 3: Add the panel UI**

In `src/app/streamlit_app.py`, inside `with left:`, immediately after the "Commit current state as new baseline" button block (ends near line 398), add:

```python
    st.caption("ASK AI")
    api_key = st.text_input("Anthropic API key", type="password", key="ai_key",
                            help="Your key is used only for this session (BYOK).")
    ai_query = st.text_input("Ask a what-if question", key="ai_query",
                             placeholder="What if a storm hits the US East Coast?")
    if st.button("Ask AI", use_container_width=True):
        if not api_key:
            st.warning("Enter an Anthropic API key to use the assistant.")
        elif not ai_query:
            st.warning("Type a question first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    G_for_ai = st.session_state.get("G_base") or st.session_state.get("G")
                    client = ClaudeClient(api_key=api_key)
                    result = run_whatif(ai_query, G_for_ai, client)
                    st.session_state["ai_result"] = result.model_dump()
                except GuardrailError as e:
                    st.error(f"Unsafe request: {e}")
                except Exception as e:
                    st.error(f"AI error: {e}")

    ai_result = st.session_state.get("ai_result")
    if ai_result:
        st.markdown(f"**Tool:** `{ai_result['tool_name']}`  ")
        st.caption(f"args: {ai_result['arguments']}")
        st.write(ai_result["explanation"])
```

- [ ] **Step 4: Manual smoke test**

Run: `python -m streamlit run src/app/streamlit_app.py`
Then in the browser: open sidebar → Load default data → in the left column "ASK AI", paste an Anthropic API key, type "What happens if we lose the 5 busiest hubs?", click "Ask AI".
Expected: the panel shows `Tool: targeted_attack`, args including `k`, and a 3-5 sentence grounded explanation. No traceback.

Also confirm the existing test suite is unaffected:
Run: `python -m pytest tests/ -q`
Expected: all prior tests + new AI tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/app/streamlit_app.py requirements.txt
git commit -m "feat(ai): add Ask AI panel to Streamlit app (BYOK Claude)"
```

---

## Task 7: Eval harness (golden set + accuracy scoring)

**Files:**
- Create: `data/ai_golden_set.json`
- Create: `src/ai/eval.py`
- Test: `tests/ai/test_eval.py`

**Interfaces:**
- Consumes: `LLMClient` from `src.ai.llm_client`; `TOOL_SPECS` from `src.ai.tools`.
- Produces:
  - `eval.load_golden_set(path: str) -> list[dict]` — each item `{"query": str, "expected_tool": str}`.
  - `eval.score_tool_selection(client: LLMClient, golden: list[dict]) -> dict` returning `{"total": int, "correct": int, "accuracy": float, "details": list[dict]}`.

- [ ] **Step 1: Write the failing test**

Create `tests/ai/test_eval.py`:

```python
from src.ai.schemas import ToolSelection
from src.ai.llm_client import FakeLLMClient
from src.ai.eval import score_tool_selection, load_golden_set


class _RoutingFake:
    """Returns a tool selection based on a keyword map (simulates a router)."""
    def __init__(self, keyword_to_tool):
        self._map = keyword_to_tool

    def select_tool(self, query, tools):
        for kw, tool in self._map.items():
            if kw in query.lower():
                return ToolSelection(name=tool, arguments={})
        return ToolSelection(name="targeted_attack", arguments={})

    def explain(self, query, tool_name, result):
        return ""


def test_score_tool_selection_computes_accuracy():
    golden = [
        {"query": "lose the biggest hubs", "expected_tool": "targeted_attack"},
        {"query": "a storm hits the coast", "expected_tool": "geographic_attack"},
    ]
    client = _RoutingFake({"hub": "targeted_attack", "storm": "geographic_attack"})
    report = score_tool_selection(client, golden)
    assert report["total"] == 2
    assert report["correct"] == 2
    assert report["accuracy"] == 1.0


def test_score_tool_selection_counts_wrong_choices():
    golden = [{"query": "a storm hits", "expected_tool": "geographic_attack"}]
    client = FakeLLMClient(ToolSelection(name="targeted_attack", arguments={}))
    report = score_tool_selection(client, golden)
    assert report["correct"] == 0
    assert report["accuracy"] == 0.0
    assert report["details"][0]["expected"] == "geographic_attack"
    assert report["details"][0]["actual"] == "targeted_attack"


def test_load_golden_set_reads_repo_file():
    items = load_golden_set("data/ai_golden_set.json")
    assert len(items) >= 10
    assert all("query" in it and "expected_tool" in it for it in items)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ai/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai.eval'`

- [ ] **Step 3: Write minimal implementation**

Create `data/ai_golden_set.json`:

```json
[
  {"query": "What if we lose the 5 busiest hubs?", "expected_tool": "targeted_attack"},
  {"query": "Attack the 10 most connected airports", "expected_tool": "targeted_attack"},
  {"query": "Remove the top airports by betweenness", "expected_tool": "targeted_attack"},
  {"query": "What happens if a hurricane hits the US East Coast?", "expected_tool": "geographic_attack"},
  {"query": "An earthquake strikes near Tokyo, disabling nearby airports", "expected_tool": "geographic_attack"},
  {"query": "Simulate a regional disaster within 500 km of London", "expected_tool": "geographic_attack"},
  {"query": "Cut the 15 most critical routes", "expected_tool": "edge_attack"},
  {"query": "What if key bridge connections between airports are severed?", "expected_tool": "edge_attack"},
  {"query": "Remove routes that link different regions together", "expected_tool": "community_bridge_attack"},
  {"query": "Sever the connections bridging separate communities", "expected_tool": "community_bridge_attack"},
  {"query": "How can we add routes to make the network more resilient?", "expected_tool": "defend"},
  {"query": "Add 5 new connections to improve connectivity", "expected_tool": "defend"}
]
```

Create `src/ai/eval.py`:

```python
"""Evaluation harness: measures tool-selection accuracy on a labelled golden set."""
from __future__ import annotations
import json
from typing import Any, Dict, List

from .llm_client import LLMClient
from .tools import TOOL_SPECS

__all__ = ["load_golden_set", "score_tool_selection"]


def load_golden_set(path: str) -> List[Dict[str, str]]:
    """Load labelled (query, expected_tool) pairs from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_tool_selection(client: LLMClient, golden: List[Dict[str, str]]) -> Dict[str, Any]:
    """Run the router over the golden set and report tool-selection accuracy."""
    details: List[Dict[str, Any]] = []
    correct = 0
    for item in golden:
        selection = client.select_tool(item["query"], TOOL_SPECS)
        is_correct = selection.name == item["expected_tool"]
        correct += int(is_correct)
        details.append({
            "query": item["query"],
            "expected": item["expected_tool"],
            "actual": selection.name,
            "correct": is_correct,
        })
    total = len(golden)
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "details": details,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ai/test_eval.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add data/ai_golden_set.json src/ai/eval.py tests/ai/test_eval.py
git commit -m "feat(ai): add eval harness measuring tool-selection accuracy on golden set"
```

---

## Task 8: Observability / tracing

**Files:**
- Create: `src/ai/tracing.py`
- Test: `tests/ai/test_tracing.py`

**Interfaces:**
- Produces: `tracing.Trace` dataclass with fields `query, tool_name, arguments, latency_ms, ok, error`; `tracing.trace_turn(query, fn) -> tuple[Any, Trace]` that times `fn()` (a zero-arg callable returning an `AssistantResult`), captures success/error, and returns `(result_or_None, Trace)`. `tracing.format_trace(trace) -> str` for a one-line log.

- [ ] **Step 1: Write the failing test**

Create `tests/ai/test_tracing.py`:

```python
from src.ai.schemas import AssistantResult
from src.ai.tracing import trace_turn, format_trace, Trace


def _ok_result():
    return AssistantResult(
        query="q", tool_name="targeted_attack", arguments={"k": 3},
        metrics={}, explanation="done",
    )


def test_trace_turn_records_success():
    result, trace = trace_turn("q", _ok_result)
    assert isinstance(trace, Trace)
    assert result.explanation == "done"
    assert trace.ok is True
    assert trace.tool_name == "targeted_attack"
    assert trace.latency_ms >= 0
    assert trace.error is None


def test_trace_turn_records_failure():
    def boom():
        raise ValueError("bad args")

    result, trace = trace_turn("q", boom)
    assert result is None
    assert trace.ok is False
    assert "bad args" in trace.error
    assert trace.tool_name is None


def test_format_trace_is_single_line():
    _, trace = trace_turn("q", _ok_result)
    line = format_trace(trace)
    assert "\n" not in line
    assert "targeted_attack" in line
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ai/test_tracing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai.tracing'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ai/tracing.py`:

```python
"""Lightweight per-turn tracing for observability (latency, tool, success/error)."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

__all__ = ["Trace", "trace_turn", "format_trace"]


@dataclass
class Trace:
    query: str
    tool_name: Optional[str]
    arguments: Optional[dict]
    latency_ms: float
    ok: bool
    error: Optional[str]


def trace_turn(query: str, fn: Callable[[], Any]) -> Tuple[Optional[Any], Trace]:
    """Run fn() (returning an AssistantResult), timing it and capturing any error."""
    start = time.perf_counter()
    try:
        result = fn()
        latency_ms = (time.perf_counter() - start) * 1000.0
        return result, Trace(
            query=query,
            tool_name=getattr(result, "tool_name", None),
            arguments=getattr(result, "arguments", None),
            latency_ms=latency_ms,
            ok=True,
            error=None,
        )
    except Exception as e:  # noqa: BLE001 - tracing records all failures
        latency_ms = (time.perf_counter() - start) * 1000.0
        return None, Trace(
            query=query,
            tool_name=None,
            arguments=None,
            latency_ms=latency_ms,
            ok=False,
            error=str(e),
        )


def format_trace(trace: Trace) -> str:
    """Render a trace as a single log line."""
    status = "ok" if trace.ok else f"error={trace.error!r}"
    return (
        f"[whatif] tool={trace.tool_name} args={trace.arguments} "
        f"latency_ms={trace.latency_ms:.0f} {status} query={trace.query!r}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ai/test_tracing.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai/tracing.py tests/ai/test_tracing.py
git commit -m "feat(ai): add per-turn tracing for latency and success/error observability"
```

---

## Final verification

- [ ] **Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all existing tests (69) plus the new AI tests pass, no failures.

- [ ] **Confirm provider isolation**

Run: `python -c "import subprocess,sys; out=subprocess.run([sys.executable,'-c','import ast,glob; [print(f) for f in glob.glob(\"src/**/*.py\",recursive=True) if \"import anthropic\" in open(f,encoding=\"utf-8\").read() and not f.endswith(\"llm_client.py\")]'])"`
Expected: no output (only `llm_client.py` imports `anthropic`).

---

## Self-Review notes (already applied)

- **Spec coverage:** function-calling tool layer (Task 2), structured output via Pydantic + strict tool schemas (Tasks 1, 2, 4), guardrails (Task 3), provider-swappable client / cost routing (Task 4), orchestrator + Streamlit panel (Tasks 5-6), eval harness (Task 7), observability (Task 8), grounding principle enforced (LLM only selects tools; numbers come from `run_tool`). Disciplined prompts isolated in `prompts.py` (Task 4).
- **Type consistency:** `ToolSelection`/`AssistantResult` field names are stable across Tasks 1, 4, 5, 8; `run_tool`/`validate_and_clamp`/`select_tool`/`explain` signatures match between producer and consumer tasks.
- **Deferred provider:** every task except Task 6 (UI) is testable offline via `FakeLLMClient`; Task 4 tests `ClaudeClient` with an injected stub, so no network or API key is needed in CI.
