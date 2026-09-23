"""Runners for the tiers that need the real model or the real browser.

``ContractCaseRunner`` covers what a deterministic executor can prove on its own.
Everything else in the corpus — a task the model has to solve, a browser that has
to escalate by symptom — needs the actual RuntimeProfile, and until now no runner
existed for ``EvalTier.MODEL_SMOKE`` or ``EvalTier.EXPERIMENT``: ``EvalService``
looked one up, found nothing, and blocked the run with ``runner_not_configured``.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit

from ..agent_engine import AgentEngine, ResponseMarkup
from ..brave_browser import BraveBrowserCapability, BraveEgressGuard
from ..composite_tools import CompositeToolExecutor
from ..config import HarnessConfig, RuntimeProfileConfig, ToolRegistryConfig
from ..context_builder import ContextBuilder, ModelViewFormat
from ..conversation_store import ConversationStore
from ..corpus_service import CorpusRetriever
from ..corpus_tools import CorpusToolExecutor
from ..domain import (
    CORPUS_EFFECT,
    MUTATION_EFFECT,
    WAIVABLE_CONFIRMATION_REASONS,
    Grant,
    JsonValue,
    SessionPolicy,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    waived_reason_code,
)
from ..local_tools import RegistryToolExecutor
from ..ports import (
    ConfirmationDecision,
    ConfirmationRequest,
    CorpusAnswerJudge,
    EmbeddingRuntime,
    EngineReadiness,
    ModelRuntime,
    NullEventSink,
    TextTokenCounter,
    TokenEstimator,
    ToolExecutor,
    ToolSchema,
)
from ..system_prompt import build_system_prompt
from ..web_tools import WebToolExecutor
from .bench import BENCH_HOSTNAME, SEARCH_PATH, BenchEgressGuard, BenchServer
from .language import PortugueseDetector
from .models import RegressionFixture
from .oracles import EvalEvidence, evaluate_oracle, unsupported_claims
from .runner import (
    CaseRunner,
    CaseRunResult,
    ContractCaseRunner,
    EvalCaseSpec,
    EvalCorpus,
    build_eval_corpus,
    security_violations,
)

_ALL_GRANTS = ("WorkspaceRootGrant", "WriteGrant", "WebAccessGrant")

# The corpus reads the same prompt on every run. Taking the host date here would
# make two runs of the same arm differ by the day they ran, and every recorded
# trace differ from the one before it, for a fact no fixture measures. It becomes
# a field of the fixture the day a fixture depends on the date.
BENCH_DATE = date(2026, 1, 1)

# The fixture names a symptom per URL; the bench serves one page for each symptom.
_BENCH_PATHS = {"readable": "/readable", "js-only": "/js-only"}

# A fixture that declares a network_stub gets its URL rewritten to the bench page
# with that symptom, so a model task never depends on a third-party site.
_STUB_PAGES = {
    "known_url_returns_readable_html": "/readable",
    "known_url_requires_browser": "/js-only",
}
_URL_PATTERN = re.compile(r"https?://\S+")


def _model_view_format(settings: Mapping[str, JsonValue]) -> ModelViewFormat:
    """Qual render de ModelView o braço pede. Fora do experimento, o do contrato."""
    declared = settings.get("model_view_format")
    if declared == "xml":
        return "xml"
    if declared in (None, "json"):
        return "json"
    raise ValueError(f"unknown model_view_format in arm settings: {declared!r}")


def _sampling(settings: Mapping[str, JsonValue], name: str, default: float) -> float:
    declared = settings.get(name)
    if declared is None:
        return default
    if isinstance(declared, bool) or not isinstance(declared, int | float):
        raise ValueError(f"arm setting {name} must be a number: {declared!r}")
    return float(declared)


def _flag(settings: Mapping[str, JsonValue], name: str, default: bool) -> bool:
    declared = settings.get(name)
    if declared is None:
        return default
    if not isinstance(declared, bool):
        raise ValueError(f"arm setting {name} must be a boolean: {declared!r}")
    return declared


def _count(settings: Mapping[str, JsonValue], name: str, default: int) -> int:
    declared = settings.get(name)
    if declared is None:
        return default
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        raise ValueError(f"arm setting {name} must be a positive integer: {declared!r}")
    return declared


class RuntimeSwitch(Protocol):
    """Brings up the runtime a bake-off arm names, and releases the one before it.

    Arms run one after another, so a switch happens once per arm, not per case.
    Two models do not share an 8 GB card, and the implementation is expected to
    unload the previous one before loading the next.
    """

    async def activate(
        self, profile: RuntimeProfileConfig
    ) -> tuple[ModelRuntime, TokenEstimator]: ...


def _eval_policy(permissions: Sequence[str] = _ALL_GRANTS) -> SessionPolicy:
    now = datetime.now(UTC)
    return SessionPolicy(
        conversation_id="eval",
        grants=tuple(
            Grant(
                id=f"eval-{permission}",
                conversation_id="eval",
                permission=permission,
                scope="workspace" if permission != "WebAccessGrant" else "public-network",
                granted_at=now,
            )
            for permission in permissions
        ),
    )


class CompositeCaseRunner:
    """Routes each fixture to the first runner that supports its type."""

    def __init__(self, runners: Sequence[CaseRunner]) -> None:
        if not runners:
            raise ValueError("a composite runner needs at least one runner")
        self._runners = tuple(runners)

    def supports(self, fixture_type: str) -> bool:
        return any(runner.supports(fixture_type) for runner in self._runners)

    async def run_case(self, spec: EvalCaseSpec) -> CaseRunResult:
        for runner in self._runners:
            if runner.supports(spec.fixture.type):
                return await runner.run_case(spec)
        raise ValueError(f"no runner supports fixture type: {spec.fixture.type}")


class BrowserBenchCaseRunner:
    """Replays a web_fetch automation fixture against the deterministic bench.

    The fixture states the symptom per URL (how many characters HTTP alone
    extracts); the bench serves one page above the escalation threshold and one
    below it, and the assertions read back which engine produced each result.
    """

    _SUPPORTED_TYPES = frozenset({"executor_automation", "egress_boundary"})

    def __init__(
        self,
        *,
        registry: ToolRegistryConfig,
        browser_guard: BraveEgressGuard | None = None,
    ) -> None:
        self._registry = registry
        self._browser_guard = browser_guard or BraveEgressGuard()

    def supports(self, fixture_type: str) -> bool:
        return fixture_type in self._SUPPORTED_TYPES

    async def run_case(self, spec: EvalCaseSpec) -> CaseRunResult:
        if spec.fixture.type == "egress_boundary":
            return await self._run_boundary(spec)
        calls_spec = spec.fixture.stimulus.get("calls")
        if not isinstance(calls_spec, Sequence) or isinstance(calls_spec, str):
            raise ValueError(f"fixture has no bench calls: {spec.fixture.id}")

        started = time.monotonic()
        async with BenchServer() as bench:
            guard = BenchEgressGuard(bench.port)
            executor = WebToolExecutor(
                registry=self._registry,
                session_policy=_eval_policy(("WebAccessGrant",)),
                egress_guard=guard,
                browser_capability=BraveBrowserCapability(
                    egress_guard=guard,
                    guard=self._browser_guard,
                ),
                browser_egress_guard=self._browser_guard,
            )
            calls: list[ToolCall] = []
            results: list[ToolResult] = []
            for index, raw in enumerate(calls_spec):
                if not isinstance(raw, Mapping):
                    raise ValueError(f"invalid bench call in fixture: {spec.fixture.id}")
                entry = cast(Mapping[str, JsonValue], raw)
                call = ToolCall(
                    id=f"bench-{index}",
                    name="web_fetch",
                    arguments={"url": bench.url(self._bench_path(entry))},
                )
                calls.append(call)
                results.append(await executor.execute(call))
            bench_requests = bench.requests

        evidence = EvalEvidence(
            tool_calls=tuple(calls),
            tool_results=tuple(results),
        )
        producers = [str(result.meta.get("producer", "")) for result in results]
        evaluation = evaluate_oracle(
            spec.fixture.oracle.typed_assertions,
            evidence,
            tool_effects=self._registry.effects_by_tool,
        )
        return CaseRunResult(
            evidence=evidence,
            metrics={
                "latency_ms": (time.monotonic() - started) * 1000,
                "browser_escalations": float(producers.count("browser")),
                "data_egress_events": float(len(bench_requests)),
                "untrusted_web_taint_violations": float(
                    sum(
                        "UntrustedWebTaint" not in cast(list[str], result.meta.get("taints", []))
                        for result in results
                        if result.status is ToolResultStatus.SUCCESS
                    )
                ),
            },
            security_violations=security_violations(spec.fixture, evaluation),
            evaluation=evaluation,
        )

    async def _run_boundary(self, spec: EvalCaseSpec) -> CaseRunResult:
        """Proves the bench exception does not exist on the production route.

        The grant is present and the guard is the production one, so the only
        thing that can refuse the bench address is the egress policy itself.
        """
        raw_call = spec.fixture.stimulus.get("tool_call")
        if not isinstance(raw_call, Mapping):
            raise ValueError(f"boundary fixture has no tool_call: {spec.fixture.id}")
        entry = cast(Mapping[str, JsonValue], raw_call)
        raw_arguments = entry.get("arguments")
        arguments: Mapping[str, JsonValue] = (
            cast(Mapping[str, JsonValue], raw_arguments)
            if isinstance(raw_arguments, Mapping)
            else {}
        )
        call = ToolCall(
            id="boundary-0",
            name=str(entry.get("name", "web_fetch")),
            arguments=arguments,
        )
        executor = WebToolExecutor(
            registry=self._registry,
            session_policy=_eval_policy(("WebAccessGrant",)),
        )
        result = await executor.execute(call)
        evidence = EvalEvidence(tool_calls=(call,), tool_results=(result,))
        evaluation = evaluate_oracle(
            spec.fixture.oracle.typed_assertions,
            evidence,
            tool_effects=self._registry.effects_by_tool,
        )
        return CaseRunResult(
            evidence=evidence,
            metrics={"data_egress_events": 0.0},
            security_violations=security_violations(spec.fixture, evaluation),
            evaluation=evaluation,
        )

    def _bench_path(self, entry: Mapping[str, JsonValue]) -> str:
        url = entry.get("url")
        if not isinstance(url, str):
            raise ValueError("bench call has no url")
        tail = urlsplit(url).path.rsplit("/", 1)[-1]
        path = _BENCH_PATHS.get(tail)
        if path is None:
            raise ValueError(f"no bench page for {url}")
        return path


class _CorpusArmRetrieval:
    """The retrieval the harness runs before the first AgentStep, for one arm.

    The production one reads the grant from the ConversationStore and generates a
    standalone English query first. Here the grant is the arm — that is the whole
    point of the comparison — and there is no rewrite: it costs a generation per
    case and, measured against three acervos, changed no injected passage at all.
    Making it a variable belongs to its own experiment, not to the background of
    this one.
    """

    def __init__(self, *, corpus_id: str | None, retriever: CorpusRetriever) -> None:
        self._corpus_id = corpus_id
        self._retriever = retriever
        self.injected_passages: int | None = None

    async def for_turn(
        self,
        conversation_id: str,
        question: str,
    ) -> Mapping[str, JsonValue] | None:
        del conversation_id
        if self._corpus_id is None:
            # O braço sem acervo não recupera nada, e isso é fato medido: zero
            # passagens, não "a automação não rodou".
            self.injected_passages = 0
            return None
        retrieval = await self._retriever.retrieve(self._corpus_id, question)
        self.injected_passages = len(retrieval.chunks)
        return {
            "corpus_id": retrieval.corpus_id,
            "search_query": None,
            **cast(Mapping[str, JsonValue], retrieval.payload()),
        }


class WaivedWriteGate:
    """Reads the waiver the bench already wrote, and denies everything else.

    Sem Operator na bancada, alguém tem que responder a confirmação. O gate padrão
    nega, e toda fixture de escrita terminava em ``write_confirmation_required`` —
    recusa do gate medida como falha de tarefa. Este lê a mesma dispensa que o
    Operator concede, com a mesma lista de razões que a produção usa, então o que
    ele aprova é o que ela aprovaria. O resto continua negado: escrita sob taint não
    é dispensável (ADR 0008), e um gate que espera resposta prenderia o Turn até o
    timeout de duração.
    """

    def __init__(self, store: ConversationStore) -> None:
        self._store = store

    async def _waived(self, request: ConfirmationRequest) -> bool:
        return request.reason_code in WAIVABLE_CONFIRMATION_REASONS and MUTATION_EFFECT in (
            await self._store.waived_confirmations(request.conversation_id)
        )

    async def will_announce(self, request: ConfirmationRequest) -> bool:
        return not await self._waived(request)

    async def confirm(self, request: ConfirmationRequest) -> ConfirmationDecision:
        if await self._waived(request):
            return ConfirmationDecision(
                approved=True,
                reason_code=waived_reason_code(request.reason_code),
            )
        return ConfirmationDecision(approved=False, reason_code=request.reason_code)


class ModelCaseRunner:
    """Runs a fixture's user request through the real AgentEngine and RuntimeProfile."""

    _SUPPORTED_TYPES = frozenset(
        {"model_task", "loop_recovery", "capability_gate", "corpus_answer"}
    )

    def __init__(
        self,
        *,
        config: HarnessConfig,
        runtime: ModelRuntime,
        estimator: TokenEstimator,
        operator_notes: str,
        runtime_readiness: EngineReadiness | None = None,
        browser_guard: BraveEgressGuard | None = None,
        embedder: EmbeddingRuntime | None = None,
        runtime_switch: RuntimeSwitch | None = None,
        answer_judge: CorpusAnswerJudge | None = None,
    ) -> None:
        self._config = config
        self._runtime = runtime
        self._estimator = estimator
        self._runtime_switch = runtime_switch
        self._answer_judge = answer_judge
        # Sem embedder, o acervo da fixture é montado pelo determinístico da
        # bancada, e o que chega ao modelo é a passagem que o hash sorteou. Com
        # ele, é a passagem que a produção entregaria — inclusive nenhuma.
        self._embedder = embedder
        self._operator_notes = operator_notes
        # Congelado por run ao lado dos digests de contrato: bloco vazio recebe o
        # digest da string vazia, porque "sem texto do Operator" é fato medido e
        # não campo ausente.
        self.operator_prompt_digest = hashlib.sha256(operator_notes.encode("utf-8")).hexdigest()
        self._runtime_readiness = runtime_readiness or EngineReadiness(ready=True)
        self._browser_guard = browser_guard or BraveEgressGuard()
        self._language_detector = PortugueseDetector()

    def supports(self, fixture_type: str) -> bool:
        return fixture_type in self._SUPPORTED_TYPES

    def _judge_for(self, settings: Mapping[str, JsonValue]) -> CorpusAnswerJudge | None:
        """O braço liga o juiz; sem juiz configurado, pedir um é erro, não silêncio."""
        declared = settings.get("answer_judge", "disabled")
        if declared == "disabled":
            return None
        if declared != "advisory":
            raise ValueError(f"unknown answer_judge in arm settings: {declared!r}")
        if self._answer_judge is None:
            raise ValueError("arm asks for the answer judge and none is configured")
        return self._answer_judge

    async def _arm_runtime(
        self, settings: Mapping[str, JsonValue]
    ) -> tuple[HarnessConfig, ModelRuntime, TokenEstimator]:
        """O perfil que o braço declara, ou o da rota quando ele não declara nenhum.

        Um braço de bake-off troca o modelo e nada mais. Sem switch configurado, um
        braço que pede outro perfil é erro: rodar o perfil errado com o rótulo
        certo é o jeito de o relatório comparar um modelo com ele mesmo.
        """
        declared = settings.get("runtime_profile")
        if declared is None or declared == self._config.runtime_profile.id:
            return self._config, self._runtime, self._estimator
        if not isinstance(declared, str):
            raise ValueError(f"arm setting runtime_profile must be a string: {declared!r}")
        if self._runtime_switch is None:
            raise ValueError(f"arm asks for runtime profile {declared} and no switch is set")
        config = self._config.for_runtime_profile(declared)
        runtime, estimator = await self._runtime_switch.activate(config.runtime_profile)
        return config, runtime, estimator

    async def run_case(self, spec: EvalCaseSpec) -> CaseRunResult:
        raw_request = spec.fixture.stimulus.get("user_request")
        if not isinstance(raw_request, str):
            raise ValueError(f"fixture has no user_request: {spec.fixture.id}")
        config, runtime, estimator = await self._arm_runtime(spec.settings)

        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="he-model-") as temporary:
            base = Path(temporary)
            workspace = base / "w"
            workspace.mkdir()
            _seed_workspace(workspace, spec.fixture)

            corpus = (
                await build_eval_corpus(
                    spec.fixture,
                    base / "corpora",
                    embedder=self._embedder,
                    counter=(estimator if isinstance(estimator, TextTokenCounter) else None),
                    judge=self._judge_for(spec.settings),
                )
                if "corpus_documents" in spec.fixture.stimulus
                else None
            )
            granted = _corpus_granted(spec)
            retrieval = (
                _CorpusArmRetrieval(
                    corpus_id=corpus.corpus_id if granted else None,
                    retriever=corpus.retriever,
                )
                if corpus is not None
                else None
            )

            async with BenchServer() as bench:
                request = _bench_request(raw_request, spec.fixture, bench)
                store = ConversationStore(base / "c.sqlite3")
                await store.initialize()
                revision = await store.create_workspace(str(workspace))
                conversation = await store.create_conversation(revision.workspace_id)
                # No Operator watches a bench run, and the default gate denies, so
                # every fixture that writes would measure a refusal instead of the
                # model. The waiver is the Operator's own mechanism, used here in
                # the open rather than a gate that only evals have.
                await store.waive_confirmation(conversation.id, MUTATION_EFFECT)
                # Um acervo é a fonte, não uma fonte a mais. Com WebAccessGrant o
                # braço sem Corpus sai para a web, volta com UntrustedWebTaint e o
                # gate encerra o Turn — medido: nove de quinze casos terminaram em
                # web_taint_confirmation_required sem resposta nenhuma. Isso compara
                # acervo contra web barrada, e o experimento declara comparar acervo
                # contra memória.
                policy = _eval_policy(
                    ("WorkspaceRootGrant", "WriteGrant") if corpus is not None else _ALL_GRANTS
                )
                if corpus is not None and granted:
                    policy = _with_corpus_grant(policy, corpus.corpus_id)
                executor = self._executor(workspace, policy, bench, corpus)
                engine = AgentEngine(
                    store=store,
                    runtime=runtime,
                    tool_executor=executor,
                    context_builder=ContextBuilder(
                        estimator,
                        context_window=self._config.context.initial_budget_tokens,
                        output_budget=self._config.loop.max_output_tokens,
                        model_view_format=_model_view_format(spec.settings),
                        automation_role=config.runtime_profile.internal_automation_role,
                    ),
                    event_sink=NullEventSink(),
                    confirmation_gate=WaivedWriteGate(store),
                    system_prompt=build_system_prompt(
                        config,
                        today=BENCH_DATE,
                        operator_notes=self._operator_notes,
                    ),
                    tool_schemas=self._tool_schemas(policy),
                    model_options={
                        "temperature": _sampling(
                            spec.settings,
                            "temperature",
                            config.execution_route.sampling.temperature,
                        ),
                        "presence_penalty": _sampling(
                            spec.settings,
                            "presence_penalty",
                            config.execution_route.sampling.presence_penalty,
                        ),
                    },
                    seed=spec.seed,
                    max_model_invocations=_count(spec.settings, "max_steps", config.loop.max_steps),
                    max_tool_calls_per_step=self._config.loop.max_tool_calls_per_step,
                    max_tool_calls_per_turn=self._config.loop.max_tool_calls_per_turn,
                    max_read_calls_per_turn=self._config.loop.max_read_calls_per_turn,
                    max_malformed_model_attempts=(self._config.loop.max_malformed_model_attempts),
                    tool_effects=self._config.tool_registry.effects_by_tool,
                    max_turn_duration_seconds=self._config.loop.max_turn_duration_seconds,
                    runtime_readiness=self._runtime_readiness,
                    think=_flag(
                        spec.settings, "thinking", config.execution_route.sampling.thinking
                    ),
                    response_markup=ResponseMarkup.of(config.runtime_profile),
                    turn_retrieval=retrieval,
                )
                turn = await engine.run(conversation.id, request)

            steps = await store.list_agent_steps(turn.id)
            history = await store.list_canonical_history(conversation.id)
            calls = tuple(call for step in steps for call in step.tool_calls)
            results = tuple(result for step in steps for result in step.tool_results)
            response = next(
                (
                    str(entry.payload.get("content", ""))
                    for entry in reversed(history)
                    if entry.kind.value == "final_response"
                ),
                None,
            )
            outcome = turn.terminal_outcome
            evidence = EvalEvidence(
                tool_calls=calls,
                tool_results=results,
                terminal_outcome_kind=outcome.kind if outcome is not None else None,
                terminal_outcome_reason=outcome.reason_code if outcome is not None else None,
                terminal_outcome_detail=outcome.detail if outcome is not None else None,
                workspace_root=workspace,
                observed_paths=(),
                response=response,
                injected_passages=retrieval.injected_passages if retrieval is not None else None,
            )
            evaluation = evaluate_oracle(
                spec.fixture.oracle.typed_assertions,
                evidence,
                language_detector=self._language_detector,
                tool_effects=self._config.tool_registry.effects_by_tool,
            )
            return CaseRunResult(
                evidence=evidence,
                metrics={
                    "latency_ms": (time.monotonic() - started) * 1000,
                    "steps_to_terminal": float(len(steps)),
                    "extra_tool_calls": float(max(0, len(calls) - _expected_calls(spec.fixture))),
                    "tool_noop_rate": _noop_rate(results),
                    # A malformed response or a tool call emitted as prose is caught
                    # by AgentEngine and never reaches the Operator, so no oracle can
                    # see it. Counting it is the only way the corpus reports how often
                    # the model produces something the harness had to throw away.
                    "rejected_model_attempts": float(
                        sum(entry.kind.value == "rejected_model_attempt" for entry in history)
                    ),
                    # A metade qualitativa do gate de acervo, contada: quantas
                    # alegações o caso fez que o material não sustenta.
                    "unsupported_claims": float(unsupported_claims(evaluation)),
                    # O oráculo é o mesmo nos dois braços — tem que ser, ou a
                    # comparação não compara nada. Quem mostra que os braços
                    # rodaram experimentos diferentes é esta métrica.
                    **(
                        {"injected_passages": float(retrieval.injected_passages or 0)}
                        if retrieval is not None
                        else {}
                    ),
                },
                security_violations=security_violations(spec.fixture, evaluation),
                evaluation=evaluation,
            )

    def _tool_schemas(self, policy: SessionPolicy) -> tuple[ToolSchema, ...]:
        effective = policy.effective_grants
        return tuple(
            definition.tool_schema()
            for definition in self._config.tool_registry.model_tools
            if definition.status == "enabled"
            and all(grant in effective for grant in definition.required_grants)
        )

    def _executor(
        self,
        workspace: Path,
        policy: SessionPolicy,
        bench: BenchServer,
        corpus: EvalCorpus | None = None,
    ) -> ToolExecutor:
        registry = self._config.tool_registry
        local: ToolExecutor = RegistryToolExecutor(
            registry=registry,
            workspace_root=workspace,
            session_policy=policy,
            max_read_bytes=self._config.context.max_tool_read_bytes,
            max_search_bytes=self._config.context.max_tool_search_bytes,
        )
        guard = BenchEgressGuard(bench.port)
        web: ToolExecutor = WebToolExecutor(
            registry=registry,
            session_policy=policy,
            egress_guard=guard,
            # The bench answers the provider endpoint, so web_search is exercised
            # without the corpus reaching a real search engine.
            search_endpoint=bench.url(SEARCH_PATH),
            browser_capability=BraveBrowserCapability(
                egress_guard=guard,
                guard=self._browser_guard,
            ),
            browser_egress_guard=self._browser_guard,
        )
        corpus_executor: ToolExecutor | None = (
            CorpusToolExecutor(
                registry=registry,
                session_policy=policy,
                retriever=corpus.retriever,
            )
            if corpus is not None
            else None
        )
        return CompositeToolExecutor(
            routes={
                definition.name: self._route(definition.effects, local, web, corpus_executor)
                for definition in registry.model_tools
            }
        )

    @staticmethod
    def _route(
        effects: Sequence[str],
        local: ToolExecutor,
        web: ToolExecutor,
        corpus: ToolExecutor | None,
    ) -> ToolExecutor:
        """Rota pelo efeito, como em produção — nunca pelo nome da tool."""
        if CORPUS_EFFECT in effects and corpus is not None:
            return corpus
        if "data_egress" in effects:
            return web
        return local


def build_live_runner(
    config: HarnessConfig,
    model_runner: ModelCaseRunner,
    *,
    browser_guard: BraveEgressGuard | None = None,
) -> CompositeCaseRunner:
    """A composição de runners que cobre todo o corpus, montada num lugar só.

    Escrita duas vezes — no script de campanha e no teste que verifica se todo
    `type` do dataset tem runner — ela ficou meses quebrada no script:
    `ContractCaseRunner` trocou `registry=` por `config=` e só o teste
    acompanhou. Com uma função só, o teste monta o mesmo caminho do script com
    colaboradores de mentira e a deriva de assinatura aparece sem GPU.
    """
    return CompositeCaseRunner(
        (
            ContractCaseRunner(config=config),
            BrowserBenchCaseRunner(registry=config.tool_registry, browser_guard=browser_guard),
            model_runner,
        )
    )


def _corpus_granted(spec: EvalCaseSpec) -> bool:
    """Quem decide o grant é o braço; a fixture só diz o que vale fora de um.

    É esta linha que faz `no_corpus` e `corpus_granted` rodarem experimentos
    diferentes. Sem ela o braço vira rótulo, os dois lados executam o mesmo
    trabalho e a diferença que aparecer no relatório é ruído de geração.
    """
    declared = spec.settings.get("corpus_granted")
    if declared is not None:
        return declared is not False
    return spec.fixture.stimulus.get("corpus_granted", True) is not False


def _with_corpus_grant(policy: SessionPolicy, corpus_id: str) -> SessionPolicy:
    return SessionPolicy(
        conversation_id=policy.conversation_id,
        grants=(
            *policy.grants,
            Grant(
                id="eval-CorpusGrant",
                conversation_id=policy.conversation_id,
                permission="CorpusGrant",
                scope=corpus_id,
                granted_at=datetime.now(UTC),
            ),
        ),
    )


def _bench_request(request: str, fixture: RegressionFixture, bench: BenchServer) -> str:
    stub = fixture.stimulus.get("network_stub")
    path = _STUB_PAGES.get(stub) if isinstance(stub, str) else None
    if path is None:
        return request
    return _URL_PATTERN.sub(bench.url(path), request)


def _seed_workspace(workspace: Path, fixture: RegressionFixture) -> None:
    entries = fixture.stimulus.get("workspace")
    if not isinstance(entries, Sequence) or isinstance(entries, str):
        return
    for raw in entries:
        # A fixture describes its workspace either as "path" or as
        # {"path": ..., "content": ...}; both have to produce a real file, or the
        # model is asked to search an empty directory and loops until the limit.
        if isinstance(raw, str):
            name, content = _split_seed_entry(raw)
        elif isinstance(raw, Mapping):
            entry = cast(Mapping[str, JsonValue], raw)
            raw_name = entry.get("path")
            if not isinstance(raw_name, str):
                continue
            raw_content = entry.get("content")
            name = raw_name
            content = raw_content if isinstance(raw_content, str) else ""
        else:
            continue
        if not name or Path(name).is_absolute() or ".." in Path(name).parts:
            continue
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _split_seed_entry(entry: str) -> tuple[str, str]:
    """Splits "src/app.js contains TODO: validate input" into path and content."""
    path, separator, description = entry.partition(" ")
    if not separator:
        return entry.strip(), ""
    return path.strip(), description.strip()


def _expected_calls(fixture: RegressionFixture) -> int:
    extra = fixture.oracle.model_extra or {}
    maximum = extra.get("max_tool_calls")
    return maximum if isinstance(maximum, int) else 0


def _noop_rate(results: Sequence[ToolResult]) -> float:
    if not results:
        return 0.0
    empty = sum(result.status is ToolResultStatus.EMPTY for result in results)
    return empty / len(results)


__all__ = [
    "BENCH_HOSTNAME",
    "BrowserBenchCaseRunner",
    "CompositeCaseRunner",
    "ModelCaseRunner",
    "RuntimeSwitch",
    "build_live_runner",
]
