"""Offline evaluation harness: measures router tool-selection and argument accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..llm_client import LLMClient
from ..tools import TOOL_SPECS
from .golden_set import GOLDEN_SET

__all__ = ["EvalCaseResult", "EvalReport", "evaluate", "format_report"]


@dataclass
class EvalCaseResult:
    query: str
    expected_tool: str
    predicted_tool: str
    tool_correct: bool
    args_correct: bool


@dataclass
class EvalReport:
    results: list[EvalCaseResult]
    tool_accuracy: float
    arg_accuracy: float

    @property
    def n_cases(self) -> int:
        return len(self.results)


def _args_match(expected: dict[str, Any], predicted: dict[str, Any]) -> bool:
    return all(predicted.get(k) == v for k, v in expected.items())


def evaluate(client: LLMClient, dataset: list[dict[str, Any]] = GOLDEN_SET) -> EvalReport:
    """Run the client's router over the dataset and score tool + argument accuracy."""
    results: list[EvalCaseResult] = []
    for case in dataset:
        sel = client.select_tool(case["query"], TOOL_SPECS)
        tool_ok = sel.name == case["expected_tool"]
        args_ok = tool_ok and _args_match(case.get("expected_args", {}), sel.arguments)
        results.append(
            EvalCaseResult(
                query=case["query"],
                expected_tool=case["expected_tool"],
                predicted_tool=sel.name,
                tool_correct=tool_ok,
                args_correct=args_ok,
            )
        )
    n = len(results) or 1
    tool_acc = sum(r.tool_correct for r in results) / n
    arg_acc = sum(r.args_correct for r in results) / n
    return EvalReport(results=results, tool_accuracy=tool_acc, arg_accuracy=arg_acc)


def format_report(report: EvalReport) -> str:
    """Render a human-readable accuracy summary."""
    lines = [
        f"Cases: {report.n_cases}",
        f"Tool-selection accuracy: {report.tool_accuracy:.1%}",
        f"Argument accuracy: {report.arg_accuracy:.1%}",
        "",
    ]
    for r in report.results:
        mark = "OK " if r.tool_correct else "XX "
        lines.append(f"{mark}{r.expected_tool:24s} <- {r.query}")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual run, needs ANTHROPIC_API_KEY
    import os

    from ..llm_client import ClaudeClient

    client = ClaudeClient(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    print(format_report(evaluate(client)))


if __name__ == "__main__":  # pragma: no cover
    main()
