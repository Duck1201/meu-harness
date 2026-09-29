from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .domain import JsonValue, ToolCall, ToolResult


class ModelRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    description: str
    parameters: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: ModelRole
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelDurations:
    total_ns: int | None = None
    load_ns: int | None = None
    prompt_eval_ns: int | None = None
    eval_ns: int | None = None


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolSchema, ...]
    options: Mapping[str, JsonValue]
    seed: int
    max_output_tokens: int = 8192
    think: bool = True


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str | None = None
    reasoning: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: ModelUsage | None = None
    durations: ModelDurations | None = None


@dataclass(frozen=True, slots=True)
class ToolBatchPreflight:
    allowed: bool
    reason_code: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.allowed and not self.reason_code:
            raise ValueError("blocked preflight requires a reason_code")


class AgentEventKind(StrEnum):
    REASONING = "reasoning"
    STEP_STARTED = "step_started"
    CONTEXT_BUILT = "context_built"
    GENERATION_STATS = "generation_stats"
    STEP_FINISHED = "step_finished"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CONFIRMATION_RESOLVED = "confirmation_resolved"
    FINAL_RESPONSE = "final_response"
    TURN_FINISHED = "turn_finished"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    kind: AgentEventKind
    turn_id: str
    step_sequence: int | None
    payload: Mapping[str, JsonValue]
    conversation_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class EngineReadiness:
    ready: bool
    reason_code: str | None = None


class ModelRuntime(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResponse: ...


@dataclass(frozen=True, slots=True)
class VisionAnswer:
    """What the vision model said about one image, plus what it cost."""

    text: str
    latency_ms: float
    output_tokens: int | None = None


class VisionRuntime(Protocol):
    """A local model that answers a question about one image (ADR 0016).

    Separate from ModelRuntime because it is a separate model with its own digest,
    and because the chat model never receives image bytes: the tool reads the
    file, this runtime looks at it, and only text goes back into the ModelView.
    """

    async def describe(self, image: bytes, question: str) -> VisionAnswer: ...


class EmbeddingRuntime(Protocol):
    """Turns text into the vectors a Corpus is searched by.

    Separate from ModelRuntime because it is a separate model with a separate
    digest, and because everything downstream of it — chunking, indexing, the
    whole ingestion suite — has to be testable on a machine with no GPU.
    """

    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class ModelRuntimeError(Exception):
    """The provider refused or was unavailable.

    Carries the class of the failure, not just its text: a Turn that dies on a
    provider is a different event from a Turn that dies on a harness bug, and
    telling them apart afterwards needs more than the exception's name. The
    runtime that raises it fills these; whoever ends the Turn reads them.
    """

    def __init__(
        self,
        message: str,
        *,
        error: Mapping[str, JsonValue] | None = None,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error = error
        self.retryable = retryable
        self.status_code = status_code


class ModelGenerationTimeout(ModelRuntimeError):
    """O harness desligou a geração; o provedor não caiu.

    São eventos opostos e o reason code não pode ser o mesmo: em
    `model_provider_unavailable` o Operator vai checar o Ollama, que está no ar e
    ainda gerando quando o corte acontece. O que ele precisa saber é que o pedido
    é longo demais para o teto configurado.
    """


class MalformedModelResponseError(ModelRuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True, slots=True)
class ConfirmationPreview:
    """What a mutation would do, for an Operator deciding whether to allow it.

    Approving a blob of arguments is approving nothing anyone read. The preview
    is produced by the executor because only it resolves a path under the policy
    guards; reading the file anywhere else would be a way around them.
    """

    tool_call_id: str
    path: str
    kind: str
    diff: str
    truncated: bool


class ToolExecutor(Protocol):
    async def preflight(self, calls: Sequence[ToolCall]) -> ToolBatchPreflight: ...

    async def execute(self, call: ToolCall) -> ToolResult: ...

    async def preview(self, call: ToolCall) -> ConfirmationPreview | None: ...


class ToolExecutorFactory(Protocol):
    async def effective_tool_schemas(self, conversation_id: str) -> tuple[ToolSchema, ...]: ...

    async def create(self, conversation_id: str) -> ToolExecutor: ...


class StopSignal(Protocol):
    @property
    def stop_requested(self) -> bool: ...


class NeverStopSignal:
    @property
    def stop_requested(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class ConfirmationRequest:
    id: str
    conversation_id: str
    turn_id: str
    request_id: str
    step_sequence: int
    reason_code: str
    tool_calls: tuple[ToolCall, ...]
    previews: tuple[ConfirmationPreview, ...] = ()


@dataclass(frozen=True, slots=True)
class ConfirmationDecision:
    approved: bool
    reason_code: str

    def __post_init__(self) -> None:
        if not self.reason_code:
            raise ValueError("confirmation decision requires a reason_code")


class ConfirmationGate(Protocol):
    """Asks the Operator to confirm a batch the harness refuses to run unattended.

    The gate is responsible for notifying the Operator, because registering the
    request and announcing it must happen together: a decision that arrives before
    the request is known would be dropped.
    """

    async def will_announce(self, request: ConfirmationRequest) -> bool:
        """Whether this request actually reaches a human.

        A gate that answers from a standing decision — a waiver, a policy of
        refusing everything — decides without asking. The Turn's history says so
        instead of recording a question nobody heard.
        """
        ...

    async def confirm(self, request: ConfirmationRequest) -> ConfirmationDecision: ...


class DenyingConfirmationGate:
    """Default gate: nothing can approve, so the batch stays blocked.

    Used wherever no Operator is present — eval runners and tests included. The
    reason code is the request's own, because from the Turn's point of view the
    confirmation was required and never obtained.
    """

    async def will_announce(self, request: ConfirmationRequest) -> bool:
        del request
        return False

    async def confirm(self, request: ConfirmationRequest) -> ConfirmationDecision:
        return ConfirmationDecision(approved=False, reason_code=request.reason_code)


class TokenEstimator(Protocol):
    @property
    def validated(self) -> bool: ...

    def estimate(self, messages: Sequence[ModelMessage], tools: Sequence[ToolSchema]) -> int: ...


@runtime_checkable
class TextTokenCounter(Protocol):
    """Counts a bare string, which is what chunking needs and a ModelView is not.

    Kept apart from TokenEstimator so ingestion does not drag the message and
    schema overheads of a Turn into a decision about where a paragraph ends.
    """

    def count_text(self, text: str) -> int: ...


class CorpusAnswerJudge(Protocol):
    """Julga, passagem a passagem, se ela traz o fato que a pergunta pede.

    Um sinal consultivo no estilo System One: probabilidade calibrada, nunca
    texto. Quem decide a resposta continua sendo o modelo, e o harness não vira
    árbitro de veracidade (ADR 0015).
    """

    async def supports(self, question: str, passages: Sequence[str]) -> tuple[float, ...]: ...


class TurnRetrieval(Protocol):
    """Retrieval the harness performs before the first AgentStep.

    Returns the entry payload to record, or ``None`` when this Conversation has
    no Corpus granted — which is the common case and must cost nothing.

    It runs ahead of the model rather than waiting to be called because a 4B
    frequently does not call the tool it should; ``corpus_search`` stays in the
    catalogue for the steps after this one, when the model knows the passages it
    got are not the ones it needs.
    """

    async def for_turn(
        self,
        conversation_id: str,
        question: str,
    ) -> Mapping[str, JsonValue] | None: ...


class EventSink(Protocol):
    async def emit(self, event: AgentEvent) -> None: ...


class NullEventSink:
    async def emit(self, event: AgentEvent) -> None:
        del event
