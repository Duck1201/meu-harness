import asyncio
import hashlib
from pathlib import Path
from typing import cast

import pytest

from harness import load_config
from harness.brave_browser import browser_devtools_available
from harness.conversation_store import ConversationStore
from harness.domain import MUTATION_EFFECT
from harness.evals import (
    BrowserBenchCaseRunner,
    CompositeCaseRunner,
    ContractCaseRunner,
    EvalCaseSpec,
    EvalTier,
    ModelCaseRunner,
    RegressionFixture,
    TaskVerdict,
    build_live_runner,
    load_eval_catalog,
)
from harness.evals.bench import (
    BENCH_HOSTNAME,
    BROWSER_REQUIRED_PAGE,
    HTTP_READABLE_PAGE,
    BenchEgressGuard,
    BenchServer,
)
from harness.evals.model_runner import WaivedWriteGate
from harness.ports import ConfirmationRequest, ModelRuntime, TokenEstimator

# Mesma sonda cacheada de tests/test_brave_browser.py: `@cache` garante um único
# launch por sessão de pytest mesmo com os dois módulos coletados.
brave_required = pytest.mark.skipif(
    not browser_devtools_available(),
    reason="This host cannot open a browser debugging port",
)


def _fixtures() -> dict[str, RegressionFixture]:
    root = Path(__file__).resolve().parents[2]
    catalog = load_eval_catalog(
        root / "evals/fixtures/regressions.json",
        root / "evals/experiments.json",
        contract_root=root,
    )
    return {fixture.id: fixture for fixture in catalog.dataset.fixtures}


def _spec(fixture: RegressionFixture) -> EvalCaseSpec:
    return EvalCaseSpec(
        run_id="run-1",
        arm_id="arm-1",
        fixture=fixture,
        seed=104729,
        order_index=0,
        tier=EvalTier.EXPERIMENT,
    )


def test_bench_pages_sit_on_both_sides_of_the_extraction_threshold() -> None:
    # The readable page must be usable over HTTP alone; the other one only after
    # JavaScript runs, which is what makes the escalation observable.
    assert len(HTTP_READABLE_PAGE) > 120
    assert b"carregando" in BROWSER_REQUIRED_PAGE
    assert b"<script>" in BROWSER_REQUIRED_PAGE


def test_bench_guard_pins_only_the_bench_hostname() -> None:
    async def scenario() -> None:
        async with BenchServer() as bench:
            guard = BenchEgressGuard(bench.port)

            pinned = await guard.resolve(bench.url("/readable"))
            assert pinned.hostname == BENCH_HOSTNAME
            assert [address.host for address in pinned.addresses] == ["127.0.0.1"]

    asyncio.run(scenario())


def test_repeated_identical_fetch_produces_one_network_effect() -> None:
    async def scenario() -> None:
        runner = BrowserBenchCaseRunner(registry=load_config().tool_registry)

        result = await runner.run_case(_spec(_fixtures()["mechanical_repeat_guard"]))

        assert result.evaluation is not None
        assert result.evaluation.verdict is TaskVerdict.PASS
        # Containment is the count of network effects, not a warning: two identical
        # calls, both successful, one request reaching the bench.
        assert result.metrics["data_egress_events"] == 1.0
        assert result.metrics["browser_escalations"] == 0.0
        assert [item.meta["cache_hit"] for item in result.evidence.tool_results] == [False, True]

    asyncio.run(scenario())


@brave_required
def test_browser_escalation_happens_only_for_the_page_that_needs_it() -> None:
    async def scenario() -> None:
        runner = BrowserBenchCaseRunner(registry=load_config().tool_registry)

        result = await runner.run_case(_spec(_fixtures()["web_http_first_browser_by_symptom"]))

        assert result.evaluation is not None
        assert result.evaluation.verdict is TaskVerdict.PASS
        assert result.metrics["browser_escalations"] == 1.0
        assert result.metrics["untrusted_web_taint_violations"] == 0.0
        assert result.security_violations == 0
        producers = [result_item.meta["producer"] for result_item in result.evidence.tool_results]
        assert producers == ["web_fetch", "browser"]

    asyncio.run(scenario())


def test_composite_routes_each_fixture_type_to_a_runner_that_supports_it() -> None:
    config = load_config()
    registry = config.tool_registry
    contract = ContractCaseRunner(config=config)
    bench = BrowserBenchCaseRunner(registry=registry)
    composite = CompositeCaseRunner((contract, bench))

    assert contract.supports("privacy_gate")
    assert not contract.supports("egress_boundary")
    assert bench.supports("egress_boundary")
    assert composite.supports("privacy_gate")
    assert composite.supports("executor_automation")
    assert not composite.supports("model_task")

    with pytest.raises(ValueError):
        CompositeCaseRunner(())


def test_every_fixture_type_in_the_dataset_has_a_runner() -> None:
    """A type without a runner is dropped in silence by EvalService._fixtures.

    Such a fixture never runs, never fails and protects nothing, which is how
    three of them survived a release. supports() reads no collaborator, so the
    model runner can be built with placeholders for this check alone — and the
    composition is the script's own, so a signature that drifts breaks here
    instead of on the bench.
    """
    config = load_config()
    composite = build_live_runner(
        config,
        ModelCaseRunner(
            config=config,
            runtime=cast(ModelRuntime, None),
            estimator=cast(TokenEstimator, None),
            operator_notes="",
        ),
    )

    unsupported = sorted(
        {fixture.type for fixture in _fixtures().values() if not composite.supports(fixture.type)}
    )

    assert unsupported == []


async def _gate_decision(
    store_path: Path,
    *,
    waived: bool,
    reason_code: str,
) -> tuple[bool, str, bool]:
    store = ConversationStore(store_path)
    await store.initialize()
    revision = await store.create_workspace(str(store_path.parent))
    conversation = await store.create_conversation(revision.workspace_id)
    if waived:
        await store.waive_confirmation(conversation.id, MUTATION_EFFECT)
    gate = WaivedWriteGate(store)
    request = ConfirmationRequest(
        id="confirmation-1",
        conversation_id=conversation.id,
        turn_id="turn-1",
        request_id="request-1",
        step_sequence=1,
        reason_code=reason_code,
        tool_calls=(),
    )
    decision = await gate.confirm(request)
    return decision.approved, decision.reason_code, await gate.will_announce(request)


def test_the_bench_gate_answers_a_write_from_the_waiver_the_runner_wrote(tmp_path: Path) -> None:
    # A dispensa gravada no store é a resposta, e a decisão tomada sem perguntar
    # entra na history como "waived" — a mesma palavra que a produção usa.
    approved, reason_code, announced = asyncio.run(
        _gate_decision(
            tmp_path / "c.sqlite3", waived=True, reason_code="write_confirmation_required"
        )
    )

    assert approved
    assert reason_code == "write_confirmation_waived"
    assert not announced


def test_the_bench_gate_denies_a_write_nobody_waived(tmp_path: Path) -> None:
    approved, reason_code, _ = asyncio.run(
        _gate_decision(
            tmp_path / "c.sqlite3", waived=False, reason_code="write_confirmation_required"
        )
    )

    assert not approved
    assert reason_code == "write_confirmation_required"


def test_the_bench_gate_still_denies_a_tainted_write_under_the_same_waiver(tmp_path: Path) -> None:
    # Escrita derivada de conteúdo da web é outra pergunta, e a dispensa nunca a
    # respondeu (ADR 0008). Ligar o gate da bancada não pode ter afrouxado isso.
    approved, reason_code, _ = asyncio.run(
        _gate_decision(
            tmp_path / "c.sqlite3", waived=True, reason_code="web_taint_confirmation_required"
        )
    )

    assert not approved
    assert reason_code == "web_taint_confirmation_required"


def test_an_empty_operator_block_freezes_the_digest_of_the_empty_string() -> None:
    runner = ModelCaseRunner(
        config=load_config(),
        runtime=cast(ModelRuntime, None),
        estimator=cast(TokenEstimator, None),
        operator_notes="",
    )

    # "sem texto do Operator" é fato medido, não campo ausente.
    assert runner.operator_prompt_digest == hashlib.sha256(b"").hexdigest()


def test_a_different_operator_block_freezes_a_different_digest() -> None:
    config = load_config()

    def digest_for(notes: str) -> str:
        return ModelCaseRunner(
            config=config,
            runtime=cast(ModelRuntime, None),
            estimator=cast(TokenEstimator, None),
            operator_notes=notes,
        ).operator_prompt_digest

    assert digest_for("Prefira respostas curtas.") != digest_for("Responda sempre em inglês.")
