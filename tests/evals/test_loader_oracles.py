import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from harness import JsonValue, TerminalOutcomeKind, ToolCall, ToolResult, ToolResultStatus
from harness.evals import (
    DatasetDriftError,
    EvalEvidence,
    TaskVerdict,
    evaluate_oracle,
    load_eval_catalog,
)

ROOT = Path(__file__).parents[2]


def test_loader_accepts_canonical_catalog_and_rejects_dataset_drift(tmp_path: Path) -> None:
    catalog = load_eval_catalog(
        ROOT / "evals/fixtures/regressions.json",
        ROOT / "evals/experiments.json",
        contract_root=ROOT,
    )

    assert catalog.dataset.dataset_version == "3.1.0"
    assert catalog.manifest.dataset.dataset_digest_sha256 == (catalog.dataset.dataset_digest_sha256)

    manifest = json.loads((ROOT / "evals/experiments.json").read_text(encoding="utf-8"))
    manifest["dataset"]["dataset_version"] = "2.0.1"
    drifted = tmp_path / "experiments.json"
    drifted.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetDriftError, match="dataset_version"):
        load_eval_catalog(
            ROOT / "evals/fixtures/regressions.json",
            drifted,
            contract_root=ROOT,
            validate_manifest_digest=False,
        )


def test_loader_rejects_scoped_digest_drift(tmp_path: Path) -> None:
    dataset = json.loads((ROOT / "evals/fixtures/regressions.json").read_text(encoding="utf-8"))
    dataset["fixtures"][0]["tags"].append("drift")
    drifted = tmp_path / "regressions.json"
    drifted.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(DatasetDriftError, match="dataset_digest_sha256"):
        load_eval_catalog(
            drifted,
            ROOT / "evals/experiments.json",
            contract_root=ROOT,
        )


def test_loader_rejects_unknown_typed_operator(tmp_path: Path) -> None:
    dataset = json.loads((ROOT / "evals/fixtures/regressions.json").read_text(encoding="utf-8"))
    dataset["fixtures"][0]["oracle"]["typed_assertions"] = [
        {"operator": "interpret_this_sentence", "text": "looks good"}
    ]
    dataset["dataset_digest_sha256"] = _scoped_digest(dataset)
    path = tmp_path / "regressions.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_eval_catalog(
            path,
            ROOT / "evals/experiments.json",
            contract_root=ROOT,
            validate_manifest_digest=False,
            validate_dataset_reference=False,
        )


def test_typed_oracles_are_deterministic_and_task_verdict_is_separate(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.txt").write_text("conteudo esperado\n", encoding="utf-8")
    result = ToolResult(
        tool_call_id="call-1",
        status=ToolResultStatus.BLOCKED,
        retryable=False,
        data=None,
        error={"code": "write_grant_required", "message": "blocked"},
        meta={"producer": "harness", "truncated": False, "taints": []},
    )
    evidence = EvalEvidence(
        tool_calls=(
            ToolCall(id="call-1", name="read_file", arguments={"file_path": "result.txt"}),
        ),
        tool_results=(result,),
        terminal_outcome_kind=TerminalOutcomeKind.BLOCKED,
        terminal_outcome_reason="write_grant_required",
        workspace_root=workspace,
        observed_paths=(workspace / "result.txt",),
    )
    assertions = [
        {"operator": "tool_called", "tool": "read_file"},
        {"operator": "tool_not_called", "tool": "write_file"},
        {"operator": "max_tool_calls", "maximum": 1},
        {
            "operator": "terminal_outcome_is",
            "kind": "blocked",
            "reason_code": "write_grant_required",
        },
        {"operator": "result_status_is", "status": "blocked"},
        {"operator": "result_error_code_is", "code": "write_grant_required"},
        {"operator": "file_exists", "path": "result.txt"},
        {
            "operator": "file_content_equals",
            "path": "result.txt",
            "content": "conteudo esperado\n",
        },
        {
            "operator": "file_content_contains",
            "path": "result.txt",
            "content": "esperado",
        },
        {"operator": "path_within_workspace", "path": "result.txt"},
    ]

    evaluation = evaluate_oracle(assertions, evidence)

    assert evaluation.verdict is TaskVerdict.PASS
    assert evaluation.terminal_outcome_kind is TerminalOutcomeKind.BLOCKED
    assert all(item.verdict is TaskVerdict.PASS for item in evaluation.assertions)

    outside = replace(evidence, observed_paths=(tmp_path / "outside.txt",))
    failed = evaluate_oracle(
        [{"operator": "path_within_workspace", "path": "../outside.txt"}], outside
    )
    assert failed.verdict is TaskVerdict.FAIL


def test_max_tool_calls_can_be_scoped_to_one_effect_class() -> None:
    def success(call_id: str) -> ToolResult:
        return ToolResult(
            tool_call_id=call_id,
            status=ToolResultStatus.SUCCESS,
            retryable=False,
            data={"value": call_id},
            error=None,
            meta={"producer": "harness", "truncated": False, "taints": []},
        )

    evidence = EvalEvidence(
        tool_calls=(
            ToolCall(id="c1", name="grep_search", arguments={"pattern": "TODO"}),
            ToolCall(id="c2", name="read_file", arguments={"file_path": "src/app.js"}),
            ToolCall(id="c3", name="write_file", arguments={"file_path": "out.txt"}),
        ),
        tool_results=(success("c1"), success("c2"), success("c3")),
    )
    effects = {
        "grep_search": ("workspace_read",),
        "read_file": ("workspace_read",),
        "write_file": ("workspace_write",),
    }

    # Three calls in total, but only one of them changes anything.
    reads = [{"operator": "max_tool_calls", "maximum": 2, "effect": "workspace_read"}]
    writes = [{"operator": "max_tool_calls", "maximum": 1, "effect": "workspace_write"}]
    assert evaluate_oracle(reads, evidence, tool_effects=effects).verdict is TaskVerdict.PASS
    assert evaluate_oracle(writes, evidence, tool_effects=effects).verdict is TaskVerdict.PASS
    assert evaluate_oracle(
        [{**writes[0], "maximum": 0}], evidence, tool_effects=effects
    ).verdict is (TaskVerdict.FAIL)
    # Unscoped still counts everything, which is the loop bound.
    assert (
        evaluate_oracle([{"operator": "max_tool_calls", "maximum": 2}], evidence).verdict
        is TaskVerdict.FAIL
    )
    # Effects live in the registry: without it the claim is unreadable, not refuted.
    assert evaluate_oracle(reads, evidence).verdict is TaskVerdict.INCONCLUSIVE


def test_result_data_contains_reads_the_payload_and_not_only_the_envelope() -> None:
    def search_result(matches: list[JsonValue]) -> ToolResult:
        return ToolResult(
            tool_call_id="call-1",
            status=ToolResultStatus.SUCCESS,
            retryable=False,
            data={"matches": matches},
            error=None,
            meta={"producer": "harness", "truncated": False, "taints": []},
        )

    found = EvalEvidence(
        tool_calls=(ToolCall(id="call-1", name="grep_search", arguments={"pattern": "TODO"}),),
        tool_results=(search_result([{"file_path": "src/app.js", "line": 2}]),),
    )
    assertion = {"operator": "result_data_contains", "content": "src/app.js"}

    assert evaluate_oracle([assertion], found).verdict is TaskVerdict.PASS

    empty_hands = replace(found, tool_results=(search_result([]),))
    assert evaluate_oracle([assertion], empty_hands).verdict is TaskVerdict.FAIL

    # A tool that was called but produced no result at all cannot satisfy the claim.
    assert evaluate_oracle([assertion], replace(found, tool_results=())).verdict is TaskVerdict.FAIL


def test_language_without_a_deterministic_detector_is_the_only_inconclusive_source() -> None:
    evidence = EvalEvidence(response="Resposta em portugues")

    evaluation = evaluate_oracle([{"operator": "response_language_pt"}], evidence)

    assert evaluation.verdict is TaskVerdict.INCONCLUSIVE
    assert [item.verdict for item in evaluation.assertions] == [TaskVerdict.INCONCLUSIVE]


def test_an_oracle_without_typed_assertions_is_rejected_at_load(tmp_path: Path) -> None:
    dataset = json.loads((ROOT / "evals/fixtures/regressions.json").read_text(encoding="utf-8"))
    dataset["fixtures"][0]["oracle"] = {"assertions": ["looks right to a human"]}
    dataset["dataset_digest_sha256"] = _scoped_digest(dataset)
    path = tmp_path / "regressions.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_eval_catalog(
            path,
            ROOT / "evals/experiments.json",
            contract_root=ROOT,
            validate_manifest_digest=False,
            validate_dataset_reference=False,
        )


def _scoped_digest(document: dict[str, object]) -> str:
    scope = document["digest_contract"]
    assert isinstance(scope, dict)
    typed_scope = cast(dict[str, object], scope)
    fields = typed_scope["scope"]
    assert isinstance(fields, list)
    typed_fields = cast(list[str], cast(list[object], fields))
    selected = {field: document[field] for field in typed_fields}
    canonical = json.dumps(
        selected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
