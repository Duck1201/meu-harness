import hashlib
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from ..agent_engine import AgentEngine
from ..config import CorpusConfig, HarnessConfig, ToolRegistryConfig, load_config
from ..context_builder import ContextBuilder, ContextTurn
from ..conversation_store import ConversationStore
from ..corpus_ingestion import build_document, embeddable_texts, extract, source_digest
from ..corpus_service import CorpusLibrary, CorpusRetriever
from ..corpus_tools import CorpusToolExecutor
from ..domain import (
    CanonicalHistoryEntry,
    CanonicalHistoryEntryKind,
    Grant,
    JsonValue,
    SessionPolicy,
    TerminalOutcomeKind,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from ..local_tools import RegistryToolExecutor
from ..ports import (
    ConfirmationPreview,
    CorpusAnswerJudge,
    EmbeddingRuntime,
    EngineReadiness,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NullEventSink,
    TextTokenCounter,
    ToolBatchPreflight,
    ToolSchema,
)
from ..web_tools import WebToolExecutor
from .models import EvalTier, RegressionFixture, TaskVerdict
from .oracles import EvalEvidence, OracleEvaluation, evaluate_oracle


@dataclass(frozen=True, slots=True)
class EvalCaseSpec:
    run_id: str
    arm_id: str
    fixture: RegressionFixture
    seed: int
    order_index: int
    tier: EvalTier
    # O que o braço muda em relação ao contrato. Vinha sendo gravado no EvalStore
    # e parava ali: dois braços declarados diferentes rodavam idênticos, e a
    # comparação media ruído. Quem aplica é o runner, campo por campo.
    settings: Mapping[str, JsonValue] = field(
        default_factory=lambda: cast(Mapping[str, JsonValue], {})
    )


@dataclass(frozen=True, slots=True)
class CaseRunResult:
    evidence: EvalEvidence
    metrics: Mapping[str, float]
    security_violations: int = 0
    evaluation: OracleEvaluation | None = None


class CaseRunner(Protocol):
    def supports(self, fixture_type: str) -> bool: ...

    async def run_case(self, spec: EvalCaseSpec) -> CaseRunResult: ...


class ContractCaseRunner:
    _SUPPORTED_TYPES = frozenset(
        {
            "executor_contract",
            "state_machine",
            "context_builder",
            "privacy_gate",
            "corpus_contract",
        }
    )

    def __init__(self, *, config: HarnessConfig) -> None:
        # Recebe o contrato inteiro, não só o registry: o executor de fixture
        # precisa dos mesmos tetos de bytes que produção, e dois argumentos
        # soltos em cada chamador seriam duas chances de divergir.
        self._registry = config.tool_registry
        self._context = config.context

    def supports(self, fixture_type: str) -> bool:
        return fixture_type in self._SUPPORTED_TYPES

    async def run_case(self, spec: EvalCaseSpec) -> CaseRunResult:
        if not self.supports(spec.fixture.type):
            raise ValueError(
                f"ContractCaseRunner does not support fixture type: {spec.fixture.type}"
            )
        with tempfile.TemporaryDirectory(prefix="he-") as temporary:
            base = Path(temporary)
            workspace = base / "w"
            workspace.mkdir()
            outside = base / "o"
            outside.mkdir()
            (outside / "secret.txt").write_text("synthetic", encoding="utf-8")
            stimulus = spec.fixture.stimulus
            if "workspace_setup" in stimulus:
                (workspace / "link-outside").symlink_to(outside, target_is_directory=True)

            if spec.fixture.type == "corpus_contract":
                calls = _fixture_calls(spec.fixture)
                results = await _run_corpus_contract(
                    spec.fixture,
                    base / "corpora",
                    registry=self._registry,
                    calls=calls,
                )
                evidence = _evidence(
                    workspace=workspace,
                    calls=calls,
                    effective_calls=calls,
                    results=results,
                )
            elif spec.fixture.type == "context_builder":
                evidence = _run_context_builder_contract()
                results: tuple[ToolResult, ...] = ()
            elif spec.fixture.id == "final_step_has_no_tools":
                evidence = await _run_final_step_contract(base, workspace, spec.seed)
                results = evidence.tool_results
            elif spec.fixture.type in {"privacy_gate", "state_machine"}:
                calls = _fixture_calls(spec.fixture)
                executor = WebToolExecutor(
                    registry=self._registry,
                    session_policy=SessionPolicy(conversation_id="eval"),
                )
                results = tuple([await executor.execute(call) for call in calls])
                effective_calls = () if spec.fixture.type == "privacy_gate" else calls
                evidence = _evidence(
                    workspace=workspace,
                    calls=calls,
                    effective_calls=effective_calls,
                    results=results,
                )
            else:
                calls = _fixture_calls(spec.fixture)
                executor = RegistryToolExecutor(
                    registry=self._registry,
                    workspace_root=workspace,
                    session_policy=_contract_policy(),
                    max_read_bytes=self._context.max_tool_read_bytes,
                    max_search_bytes=self._context.max_tool_search_bytes,
                )
                results = tuple([await executor.execute(call) for call in calls])
                evidence = _evidence(
                    workspace=workspace,
                    calls=calls,
                    effective_calls=calls,
                    results=results,
                )
            evaluation = evaluate_oracle(
                spec.fixture.oracle.typed_assertions,
                evidence,
                tool_effects=self._registry.effects_by_tool,
            )
            return CaseRunResult(
                evidence=evidence,
                metrics={},
                security_violations=security_violations(spec.fixture, evaluation),
                evaluation=evaluation,
            )


@dataclass(frozen=True, slots=True)
class EvalCorpus:
    """Um acervo montado a partir da própria fixture, com o que consultá-lo."""

    corpus_id: str
    retriever: CorpusRetriever
    config: CorpusConfig


async def build_eval_corpus(
    fixture: RegressionFixture,
    directory: Path,
    *,
    embedder: EmbeddingRuntime | None = None,
    counter: TextTokenCounter | None = None,
    judge: CorpusAnswerJudge | None = None,
) -> EvalCorpus:
    """Monta o Corpus que a fixture descreve, para quem quiser consultá-lo.

    Sem embedder, usa o determinístico: vetor por palavra, piso rebaixado e
    chunks curtos, para o CI rodar sem GPU e sem Ollama. Ele não mede semântica
    e não finge medir, então as fixtures de contrato — que só provam o gate — são
    suas.

    Com o embedder de verdade, monta o mesmo acervo que a produção montaria:
    `bge-m3`, chunks do contrato e o piso do contrato. É o que as fixtures
    `corpus_answer` precisam, porque nelas a pergunta é se a passagem que chega
    ao modelo é a que a produção teria entregado. Medido: com o embedder de hash,
    uma pergunta sobre servidor de e-mail traz a passagem do proxy, que o piso de
    0.53 no `bge-m3` recusa (similaridade abaixo do piso, contra 0.76 na pergunta
    que a passagem responde). Sem essa distinção, o experimento mediria o modelo
    diante de uma passagem que ele nunca veria.
    """
    embedder = embedder or _HashingEmbedder()
    counter = counter or _WordCounter()
    real = not isinstance(embedder, _HashingEmbedder)
    config = load_config().corpus if real else _corpus_eval_config()
    library = CorpusLibrary(
        directory,
        embedding_model=embedder.model,
        embedding_dimensions=embedder.dimensions,
    )
    corpus = await library.create(name=str(fixture.stimulus.get("corpus_name", "Corpus")))
    documents = fixture.stimulus.get("corpus_documents")
    if isinstance(documents, Sequence) and not isinstance(documents, str):
        for item in documents:
            if not isinstance(item, Mapping):
                continue
            entry = cast(Mapping[str, JsonValue], item)
            filename = str(entry.get("filename", "document.md"))
            text = str(entry.get("text", ""))
            data = text.encode("utf-8")
            draft = build_document(
                extract(filename, data, html_extractor=config.ingestion.html_extractor),
                origin_kind=str(entry.get("origin_kind", "upload")),
                origin_ref=str(entry.get("origin_ref", filename)),
                source_digest=source_digest(data),
                counter=counter,
                chunk_tokens=config.ingestion.chunk_target_tokens if real else 64,
                overlap_tokens=config.ingestion.chunk_overlap_tokens if real else 8,
            )
            await library.store(corpus.id).add_document(
                draft, await embedder.embed(embeddable_texts(draft))
            )
    return EvalCorpus(
        corpus_id=corpus.id,
        retriever=CorpusRetriever(
            library=library,
            embedder=embedder,
            counter=counter,
            config=config,
            judge=judge,
        ),
        config=config,
    )


def corpus_policy(corpus_id: str | None) -> SessionPolicy:
    """Concede — ou não — o CorpusGrant que dá acesso a um acervo."""
    return _corpus_policy(corpus_id)


async def _run_corpus_contract(
    fixture: RegressionFixture,
    directory: Path,
    *,
    registry: ToolRegistryConfig,
    calls: Sequence[ToolCall],
) -> tuple[ToolResult, ...]:
    """Roda `corpus_search` contra um Corpus montado a partir da própria fixture.

    Aqui o grant vem do stimulus e não do braço, de propósito: o oráculo de cada
    uma destas fixtures é escrito para o valor que ela declara — sem grant exige
    `blocked`, com grant exige `success` ou `empty`. Deixar um braço sobrescrever
    isso quebraria a fixture em vez de medir o braço.
    """
    corpus = await build_eval_corpus(fixture, directory)
    granted = fixture.stimulus.get("corpus_granted", True) is not False
    executor = CorpusToolExecutor(
        registry=registry,
        session_policy=_corpus_policy(corpus.corpus_id if granted else None),
        retriever=corpus.retriever,
    )
    return tuple([await executor.execute(call) for call in calls])


_EVAL_EMBEDDING_DIMENSIONS = 64


class _HashingEmbedder:
    """Vetor determinístico por palavra. Não mede sentido, e não finge medir."""

    model = "eval_hashing_embedder"
    dimensions = _EVAL_EMBEDDING_DIMENSIONS

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            buckets = [0.0] * _EVAL_EMBEDDING_DIMENSIONS
            for word in text.lower().split():
                digest = hashlib.sha256(word.encode("utf-8")).digest()
                buckets[digest[0] % _EVAL_EMBEDDING_DIMENSIONS] += 1.0
            norm = sum(value * value for value in buckets) ** 0.5 or 1.0
            vectors.append(tuple(value / norm for value in buckets))
        return tuple(vectors)


class _WordCounter:
    def count_text(self, text: str) -> int:
        return max(1, len(text.split()))


def _corpus_eval_config() -> CorpusConfig:
    """O piso do contrato foi medido no bge-m3; aqui a escala é outra.

    Baixá-lo para o embedder da bancada é o que mantém a fixture medindo o gate
    em vez de medir a coincidência de dois vetores de brinquedo.
    """
    config = load_config().corpus
    return config.model_copy(
        update={"retrieval": config.retrieval.model_copy(update={"dense_similarity_floor": 0.3})}
    )


def _corpus_policy(corpus_id: str | None) -> SessionPolicy:
    if corpus_id is None:
        return SessionPolicy(conversation_id="eval")
    return SessionPolicy(
        conversation_id="eval",
        grants=(
            Grant(
                id="eval-corpus",
                conversation_id="eval",
                permission="CorpusGrant",
                scope=corpus_id,
                granted_at=datetime.now(UTC),
            ),
        ),
    )


def _contract_policy() -> SessionPolicy:
    now = datetime.now(UTC)
    return SessionPolicy(
        conversation_id="eval",
        grants=(
            Grant(
                id="eval-root",
                conversation_id="eval",
                permission="WorkspaceRootGrant",
                scope="workspace",
                granted_at=now,
            ),
            Grant(
                id="eval-write",
                conversation_id="eval",
                permission="WriteGrant",
                scope="workspace",
                granted_at=now,
            ),
        ),
    )


def _fixture_calls(fixture: RegressionFixture) -> tuple[ToolCall, ...]:
    stimulus = fixture.stimulus
    raw_call = stimulus.get("tool_call")
    if isinstance(raw_call, Mapping):
        return (_tool_call(cast(Mapping[str, JsonValue], raw_call), 0),)
    attempts = stimulus.get("attempts")
    if attempts is None:
        attempts = stimulus.get("generated_calls")
    if isinstance(attempts, Sequence) and not isinstance(attempts, str):
        calls: list[ToolCall] = []
        for index, raw_attempt in enumerate(attempts):
            if not isinstance(raw_attempt, Mapping):
                raise ValueError(f"invalid attempt in fixture: {fixture.id}")
            attempt = cast(Mapping[str, JsonValue], raw_attempt)
            name = attempt.get("tool")
            if name is None:
                name = attempt.get("name")
            if not isinstance(name, str):
                raise ValueError(f"fixture attempt has no tool: {fixture.id}")
            raw_arguments = attempt.get("arguments")
            arguments = (
                cast(Mapping[str, JsonValue], raw_arguments)
                if isinstance(raw_arguments, Mapping)
                else {key: value for key, value in attempt.items() if key not in {"tool", "name"}}
            )
            calls.append(ToolCall(id=f"eval-{index}", name=name, arguments=arguments))
        return tuple(calls)
    raise ValueError(f"deterministic fixture has no executable tool call: {fixture.id}")


def _evidence(
    *,
    workspace: Path,
    calls: tuple[ToolCall, ...],
    effective_calls: tuple[ToolCall, ...],
    results: tuple[ToolResult, ...],
) -> EvalEvidence:
    blocked = next(
        (result for result in results if result.status is ToolResultStatus.BLOCKED),
        None,
    )
    reason = _result_error_code(blocked) if blocked is not None else None
    return EvalEvidence(
        tool_calls=effective_calls,
        tool_results=results,
        terminal_outcome_kind=(TerminalOutcomeKind.BLOCKED if blocked is not None else None),
        terminal_outcome_reason=reason,
        workspace_root=workspace,
        observed_paths=_observed_paths(workspace, calls),
    )


async def _run_final_step_contract(base: Path, workspace: Path, seed: int) -> EvalEvidence:
    store = ConversationStore(base / "c.sqlite3")
    await store.initialize()
    revision = await store.create_workspace(str(workspace))
    conversation = await store.create_conversation(revision.workspace_id)
    await store.enqueue_request(conversation.id, "synthetic final-step contract")
    runtime = _FinalStepRuntime()
    engine = AgentEngine(
        store=store,
        runtime=runtime,
        tool_executor=_NoopToolExecutor(),
        context_builder=ContextBuilder(
            _DeterministicEstimator(), context_window=128, output_budget=32
        ),
        system_prompt="contract",
        event_sink=NullEventSink(),
        tool_schemas=(ToolSchema(name="contract_tool", description="contract", parameters={}),),
        seed=seed,
        model_options={},
        max_model_invocations=1,
        runtime_readiness=EngineReadiness(ready=True),
    )
    turn = await engine.start_next_turn(conversation.id)
    if turn is None or turn.terminal_outcome is None:
        raise RuntimeError("final-step contract did not produce a TerminalOutcome")
    steps = await store.list_agent_steps(turn.id)
    return EvalEvidence(
        tool_calls=tuple(call for step in steps for call in step.tool_calls),
        tool_results=tuple(result for step in steps for result in step.tool_results),
        terminal_outcome_kind=turn.terminal_outcome.kind,
        terminal_outcome_reason=turn.terminal_outcome.reason_code,
        workspace_root=workspace,
    )


def _run_context_builder_contract() -> EvalEvidence:
    now = datetime.now(UTC)
    entries = (
        CanonicalHistoryEntry(
            id="result-1",
            sequence=1,
            conversation_id="eval",
            turn_id="turn",
            kind=CanonicalHistoryEntryKind.TOOL_RESULT,
            payload={
                "tool_call_id": "call-1",
                "status": "success",
                "retryable": False,
                "data": {"value": "synthetic"},
                "error": None,
                "meta": {"producer": "eval", "truncated": False, "taints": []},
            },
            created_at=now,
        ),
        CanonicalHistoryEntry(
            id="rejected-1",
            sequence=2,
            conversation_id="eval",
            turn_id="turn",
            kind=CanonicalHistoryEntryKind.REJECTED_MODEL_ATTEMPT,
            payload={"code": "synthetic_rejection"},
            created_at=now,
        ),
        CanonicalHistoryEntry(
            id="result-2",
            sequence=3,
            conversation_id="eval",
            turn_id="turn",
            kind=CanonicalHistoryEntryKind.TOOL_RESULT,
            payload={
                "tool_call_id": "call-2",
                "status": "success",
                "retryable": False,
                "data": {"value": "synthetic"},
                "error": None,
                "meta": {"producer": "eval", "truncated": False, "taints": []},
            },
            created_at=now,
        ),
    )
    builder = ContextBuilder(_DeterministicEstimator(), context_window=128, output_budget=32)
    _ = builder.build(
        system="contract",
        tool_schemas=(),
        completed_turns=(),
        current_turn=ContextTurn(turn_id="turn", entries=entries),
    )
    return EvalEvidence()


class _DeterministicEstimator:
    validated = True

    def estimate(self, messages: Sequence[ModelMessage], tools: Sequence[ToolSchema]) -> int:
        return len(messages) + len(tools)


class _FinalStepRuntime:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        if request.tools:
            raise RuntimeError("AgentEngine offered tools on the final step")
        return ModelResponse(content="synthetic final response")


class _NoopToolExecutor:
    async def preflight(self, calls: Sequence[ToolCall]) -> ToolBatchPreflight:
        del calls
        return ToolBatchPreflight(allowed=True)

    async def execute(self, call: ToolCall) -> ToolResult:
        raise RuntimeError(f"unexpected final-step tool execution: {call.name}")

    async def preview(self, call: ToolCall) -> ConfirmationPreview | None:
        del call
        return None


def _tool_call(raw: Mapping[str, JsonValue], index: int) -> ToolCall:
    name = raw.get("name")
    arguments = raw.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise ValueError("fixture tool_call requires name and arguments")
    return ToolCall(
        id=f"eval-{index}",
        name=name,
        arguments=cast(Mapping[str, JsonValue], arguments),
    )


def _observed_paths(workspace: Path, calls: Sequence[ToolCall]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for call in calls:
        raw = call.arguments.get("file_path")
        if isinstance(raw, str):
            paths.append(Path(raw) if Path(raw).is_absolute() else workspace / raw)
    return tuple(paths)


def _result_error_code(result: ToolResult | None) -> str | None:
    if result is None or result.error is None:
        return None
    code = result.error.get("code")
    return code if isinstance(code, str) else None


_SECURITY_TAGS = frozenset(
    {"security", "privacy", "web_access_grant", "web_taint", "symlink", "eval_bench"}
)


def security_violations(fixture: RegressionFixture, evaluation: OracleEvaluation | None) -> int:
    """Counts security expectations the case failed to meet.

    A blocked call is the harness doing its job, not a violation: half the corpus
    exists precisely to trigger a refusal, and counting refusals would make the
    zero_violations promotion gate impossible to pass by construction. What a
    violation means is that a security-relevant fixture did not get the outcome
    its oracle demands — the attempt went through, or the refusal was the wrong one.
    """
    if evaluation is None or not _SECURITY_TAGS.intersection(fixture.tags):
        return 0
    return int(evaluation.verdict is TaskVerdict.FAIL)
