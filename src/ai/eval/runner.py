"""Router evaluation harness with an opt-in provider-backed command-line runner."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..factory import make_client, resolve_provider
from ..llm_client import LLMClient, LLMConfigurationError
from ..tools import TOOL_SPECS
from .golden_set import GOLDEN_SET

__all__ = [
    "EvalCaseResult",
    "EvalReport",
    "evaluate",
    "format_report",
    "make_eval_client",
    "main",
]


_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


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


def make_eval_client(
    provider: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> LLMClient:
    """Build the selected live evaluation client from environment configuration."""
    environment = os.environ if environ is None else environ
    configured_provider = provider if provider is not None else environment.get("LLM_PROVIDER")
    canonical_provider = resolve_provider(configured_provider)
    key_name = _API_KEY_ENV[canonical_provider]
    api_key = environment.get(key_name, "").strip()
    if not api_key:
        raise LLMConfigurationError(
            f"{key_name} is required to evaluate the {canonical_provider} provider; "
            f"set it in the environment before running the evaluator"
        )
    return make_client(canonical_provider, api_key=api_key)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM router tool selection against the local golden set."
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "claude"],
        default=None,
        help="Provider to evaluate (default: LLM_PROVIDER, then openai)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        client = make_eval_client(args.provider)
    except LLMConfigurationError as exc:
        parser.error(str(exc))
    print(format_report(evaluate(client)))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
