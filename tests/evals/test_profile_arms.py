"""Um braço de bake-off troca o modelo, e só o modelo.

Antes disto o runner rodava todo braço contra o perfil da rota: um experimento
que comparasse dois modelos compararia o mesmo modelo com ele mesmo, com rótulos
diferentes, e reportaria a diferença como resultado.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from harness import TerminalOutcomeKind, load_config
from harness.config import (
    QWEN_REASONING_MARKUP,
    QWEN_TOOL_MARKUP,
    HarnessConfig,
    RuntimeProfileConfig,
)
from harness.domain import JsonValue
from harness.evals import (
    EvalCaseSpec,
    EvalTier,
    ModelCaseRunner,
    RegressionFixture,
    load_eval_catalog,
)
from harness.ports import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRuntime,
    TokenEstimator,
    ToolSchema,
)

_CHALLENGER = "bench_challenger"


def _fixture() -> RegressionFixture:
    root = Path(__file__).resolve().parents[2]
    catalog = load_eval_catalog(
        root / "evals/fixtures/regressions.json",
        root / "evals/experiments.json",
        contract_root=root,
    )
    return next(f for f in catalog.dataset.fixtures if f.id == "glob_for_paths_not_grep")


def _config_with_challenger() -> HarnessConfig:
    """Controle com a marcação do Qwen e Challenger com a do Gemma, seja qual for o ativo."""
    config = load_config()
    control = config.runtime_profile.model_copy(
        update={
            "tool_markup_leak_markers": QWEN_TOOL_MARKUP,
            "reasoning_leak_markers": QWEN_REASONING_MARKUP,
        }
    )
    challenger = control.model_copy(
        update={
            "id": _CHALLENGER,
            "status": "challenger",
            "release_eligible": False,
            "tool_markup_leak_markers": ("<|tool_call>", "<tool_call|>"),
        }
    )
    others = tuple(
        profile for profile in config.model_profiles.runtime_profiles if profile.id != control.id
    )
    profiles = config.model_profiles.model_copy(
        update={"runtime_profiles": (control, *others, challenger)}
    )
    return config.model_copy(update={"model_profiles": profiles})


def _spec(**settings: JsonValue) -> EvalCaseSpec:
    return EvalCaseSpec(
        run_id="run-1",
        arm_id="arm-1",
        fixture=_fixture(),
        seed=104729,
        order_index=0,
        tier=EvalTier.EXPERIMENT,
        settings=settings,
    )


class _Runtime:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(content=self._answer)


class _FlatEstimator:
    @property
    def validated(self) -> bool:
        return True

    def estimate(self, messages: Sequence[ModelMessage], tools: Sequence[ToolSchema]) -> int:
        del tools
        return sum(len(message.content or "") for message in messages) // 4


class _Switch:
    def __init__(self, runtime: _Runtime) -> None:
        self._runtime = runtime
        self.activated: list[str] = []

    async def activate(self, profile: RuntimeProfileConfig) -> tuple[ModelRuntime, TokenEstimator]:
        self.activated.append(profile.id)
        return self._runtime, _FlatEstimator()


def _runner(
    control: _Runtime, switch: _Switch | None, config: HarnessConfig | None = None
) -> ModelCaseRunner:
    return ModelCaseRunner(
        config=config or _config_with_challenger(),
        runtime=control,
        estimator=_FlatEstimator(),
        operator_notes="",
        runtime_switch=switch,
    )


def test_an_arm_that_names_a_profile_runs_on_that_profile() -> None:
    control = _Runtime("Nenhum arquivo encontrado.")
    challenger = _Runtime("Nenhum arquivo encontrado.")
    switch = _Switch(challenger)

    asyncio.run(_runner(control, switch).run_case(_spec(runtime_profile=_CHALLENGER)))

    assert switch.activated == [_CHALLENGER]
    assert challenger.requests
    assert not control.requests


def test_an_arm_on_the_route_profile_does_not_switch() -> None:
    control = _Runtime("Nenhum arquivo encontrado.")
    switch = _Switch(_Runtime("x"))
    route_profile = load_config().runtime_profile.id

    asyncio.run(_runner(control, switch).run_case(_spec(runtime_profile=route_profile)))

    assert switch.activated == []
    assert control.requests


def test_an_arm_that_names_a_profile_without_a_switch_is_refused() -> None:
    runner = _runner(_Runtime("x"), None)

    with pytest.raises(ValueError, match="no switch"):
        asyncio.run(runner.run_case(_spec(runtime_profile=_CHALLENGER)))


def test_the_thinking_arm_reaches_the_request() -> None:
    on = _Runtime("Nenhum arquivo encontrado.")
    off = _Runtime("Nenhum arquivo encontrado.")

    asyncio.run(_runner(on, None).run_case(_spec(thinking=True)))
    asyncio.run(_runner(off, None).run_case(_spec(thinking=False)))

    assert all(request.think for request in on.requests)
    assert not any(request.think for request in off.requests)


def test_the_leak_markers_come_from_the_profile_that_answers() -> None:
    """A marcação de tool call da outra família vaza como resposta sem isto."""
    leaked = '<|tool_call>call:glob_search{pattern:"**/*.md"}<tool_call|>'

    on_control = asyncio.run(_runner(_Runtime(leaked), None).run_case(_spec()))
    challenger = _Runtime(leaked)
    on_challenger = asyncio.run(
        _runner(_Runtime("x"), _Switch(challenger)).run_case(_spec(runtime_profile=_CHALLENGER))
    )

    # O detector do Qwen não conhece a marcação do Gemma e deixa passar.
    assert on_control.evidence.terminal_outcome_kind is TerminalOutcomeKind.COMPLETED
    assert on_challenger.evidence.terminal_outcome_kind is TerminalOutcomeKind.FAILED
    assert on_challenger.evidence.terminal_outcome_reason == "malformed_model_response_limit"


def test_for_runtime_profile_changes_only_the_route_profile() -> None:
    config = _config_with_challenger()

    derived = config.for_runtime_profile(_CHALLENGER)

    assert derived.runtime_profile.id == _CHALLENGER
    assert derived.loop == config.loop
    assert derived.context == config.context
    assert derived.tool_registry == config.tool_registry
    with pytest.raises(ValueError, match="not found"):
        config.for_runtime_profile("missing")
