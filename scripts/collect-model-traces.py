#!/usr/bin/env python
"""Records what the model actually did on each corpus fixture, as JSONL.

An eval run answers "did it pass". Fine-tuning needs the trace behind the
verdict: the request, the tools that were offered, every call the model made,
every refusal the harness returned, the final answer and which assertions the
oracle failed. None of that survives an eval run — EvalStore records verdicts
and counts by contract, and each case runs in a temporary workspace that is
deleted — so this driver re-executes the model fixtures and serialises the
evidence the runner already produces.

Passing cases are recorded too. On the same fixture, a passing trace and a
failing one differ only in what the model chose, which is the pair a preference
dataset needs; keeping only failures throws away the positive half.

    uv run python scripts/collect-model-traces.py --output datasets/model-failures/traces.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness import (  # noqa: E402
    EngineReadiness,
    EvalTier,
    HuggingFaceTokenEstimator,
    ModelMessage,
    ModelRequest,
    ModelRole,
    OllamaRuntime,
    ToolResult,
    load_config,
)
from harness.brave_browser import BraveEgressGuard  # noqa: E402
from harness.evals import (  # noqa: E402
    BENCH_DATE,
    ModelCaseRunner,
    RegressionFixture,
    load_eval_catalog,
)
from harness.evals.runner import EvalCaseSpec  # noqa: E402
from harness.system_prompt import build_system_prompt, load_operator_notes  # noqa: E402

# The protocol's recorded orders. A seed is what varies a local run, so more
# repetitions of the same seed add copies, not evidence.
PROTOCOL_SEEDS = (104729, 130363, 155921)
EXCERPT_LIMIT = 2000


def _excerpt(value: object) -> Any:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= EXCERPT_LIMIT:
        return value
    return {"truncated_json": text[:EXCERPT_LIMIT], "original_length": len(text)}


def _result_record(result: ToolResult) -> Mapping[str, Any]:
    error = dict(result.error) if result.error is not None else None
    return {
        "tool_call_id": result.tool_call_id,
        "status": result.status.value,
        "retryable": result.retryable,
        "error": error,
        "producer": result.meta.get("producer"),
        "taints": result.meta.get("taints", []),
        "data": _excerpt(result.data),
    }


def _refusals(results: Sequence[ToolResult]) -> list[Mapping[str, Any]]:
    """Every call the harness refused: the correction signal a fine-tune wants."""
    return [
        {
            "tool_call_id": result.tool_call_id,
            "code": (result.error or {}).get("code"),
            "message": (result.error or {}).get("message"),
        }
        for result in results
        if result.status.value == "blocked"
    ]


def _offered_tools(config: Any) -> list[str]:
    # The eval policy holds every grant, so the model is offered every enabled tool.
    return sorted(
        definition.name
        for definition in config.tool_registry.model_tools
        if definition.status == "enabled"
    )


def _record(
    fixture: RegressionFixture,
    seed: int,
    result: Any,
    offered: Sequence[str],
    system_prompt: str,
) -> Mapping[str, Any]:
    evaluation = result.evaluation
    assertions = evaluation.assertions if evaluation is not None else ()
    oracle = fixture.oracle
    return {
        "fixture_id": fixture.id,
        "fixture_type": fixture.type,
        "tags": list(fixture.tags),
        "seed": seed,
        "verdict": evaluation.verdict.value if evaluation is not None else "not_evaluated",
        "prompt": {
            "system": system_prompt,
            "user": fixture.stimulus.get("user_request"),
            "offered_tools": list(offered),
            "workspace": _excerpt(fixture.stimulus.get("workspace")),
        },
        "observed": {
            "tool_calls": [
                {"id": call.id, "name": call.name, "arguments": _excerpt(dict(call.arguments))}
                for call in result.evidence.tool_calls
            ],
            "tool_results": [_result_record(item) for item in result.evidence.tool_results],
            "final_response": result.evidence.response,
            "terminal_outcome": {
                "kind": (
                    result.evidence.terminal_outcome_kind.value
                    if result.evidence.terminal_outcome_kind is not None
                    else None
                ),
                "reason_code": result.evidence.terminal_outcome_reason,
                "detail": result.evidence.terminal_outcome_detail,
            },
        },
        "harness_refusals": _refusals(result.evidence.tool_results),
        "expected": {
            "typed_assertions": [
                assertion.model_dump(mode="json") for assertion in oracle.typed_assertions
            ],
            "explanation": list(oracle.explanation),
        },
        "failed_assertions": [
            {"operator": item.operator, "detail": item.explanation}
            for item in assertions
            if item.verdict.value != "pass"
        ],
        "metrics": dict(result.metrics),
        "security_violations": result.security_violations,
    }


async def _warm_up(runtime: OllamaRuntime) -> None:
    """One throwaway generation before the battery starts.

    The seed is honoured — the same prompt at the same seed repeats exactly — but
    the first generation after the model loads does not match the ones that
    follow. Without this, whichever case happens to run first is measured under
    conditions no other case sees.
    """
    await runtime.generate(
        ModelRequest(
            messages=(ModelMessage(role=ModelRole.USER, content="ok"),),
            tools=(),
            options={},
            seed=0,
            max_output_tokens=8,
            think=False,
        )
    )


async def collect(seeds: Sequence[int], output: Path, tokenizer: Path) -> int:
    config = load_config()
    catalog = load_eval_catalog(
        ROOT / "evals/fixtures/regressions.json",
        ROOT / "evals/experiments.json",
        contract_root=ROOT,
    )
    runtime = OllamaRuntime(
        base_url="http://127.0.0.1:11434",
        model=config.runtime_profile.model.id,
        expected_digest=config.runtime_profile.profile_digest_sha256,
        timeout=config.loop.model_generation_timeout_seconds,
    )
    verification = await runtime.verify_profile()
    if not verification.ready:
        print(f"RuntimeProfile is not ready: {verification.reason_code}", file=sys.stderr)
        return 2

    await _warm_up(runtime)
    operator_notes = load_operator_notes(ROOT / "SYSTEM-PROMPT.md")
    runner = ModelCaseRunner(
        config=config,
        runtime=runtime,
        estimator=HuggingFaceTokenEstimator(
            tokenizer,
            expected_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        ),
        operator_notes=operator_notes,
        runtime_readiness=EngineReadiness(ready=True),
        browser_guard=BraveEgressGuard(),
    )
    fixtures = [item for item in catalog.dataset.fixtures if runner.supports(item.type)]
    offered = _offered_tools(config)
    system_prompt = build_system_prompt(config, today=BENCH_DATE, operator_notes=operator_notes)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    try:
        with output.open("w", encoding="utf-8") as handle:
            for seed in seeds:
                for order_index, fixture in enumerate(fixtures):
                    result = await runner.run_case(
                        EvalCaseSpec(
                            run_id="trace-collection",
                            arm_id="trace",
                            fixture=fixture,
                            seed=seed,
                            order_index=order_index,
                            tier=EvalTier.MODEL_SMOKE,
                        )
                    )
                    record = _record(fixture, seed, result, offered, system_prompt)
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    verdict = str(record["verdict"])
                    counts[verdict] = counts.get(verdict, 0) + 1
                    print(f"{seed} {fixture.id}: {verdict}", file=sys.stderr)
    finally:
        await runtime.aclose()
    print(json.dumps({"output": str(output), "records": sum(counts.values()), "verdicts": counts}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets/model-failures/traces.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help="repeatable; defaults to the three recorded protocol seeds",
    )
    # O tokenizer é o do perfil da rota, pelo mesmo diretório que a bancada usa.
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / f".harness/tokenizers/{load_config().runtime_profile.id}.json",
    )
    arguments = parser.parse_args()
    if not arguments.tokenizer.is_file():
        print(f"tokenizer not found: {arguments.tokenizer}", file=sys.stderr)
        return 2
    seeds = tuple(arguments.seeds) if arguments.seeds else PROTOCOL_SEEDS
    return asyncio.run(collect(seeds, arguments.output, arguments.tokenizer))


if __name__ == "__main__":
    raise SystemExit(main())
