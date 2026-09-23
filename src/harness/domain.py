import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | Sequence[JsonValue] | Mapping[str, JsonValue]

# The effect class the registry gives every tool that changes the Workspace. Named
# once because three layers key on it: the mutation lock, the confirmation gate and
# the waiver the Operator can grant.
MUTATION_EFFECT = "workspace_write"

# The effect a Corpus read declares, and the grant that answers for it. The grant's
# scope carries which Corpus was authorized, so the conversation's selection and its
# authorization are the same fact instead of two that can disagree.
CORPUS_EFFECT = "corpus_read"
CORPUS_GRANT = "CorpusGrant"

# Inference by a local model other than the chat model, over input the harness
# already authorized under another effect. It needs no grant of its own — nothing
# leaves the machine — and exists so a tool that runs a second model on the GPU is
# offered only when that model is configured, and spends the full call budget
# instead of the cheap read one (ADR 0016).
LOCAL_INFERENCE_EFFECT = "local_inference"

# The mark every derivation of web content keeps. Written here because the Corpus
# applies it at ingestion, long before any Turn sees the Chunk.
UNTRUSTED_WEB_TAINT = "UntrustedWebTaint"

# The refusal each grant produces when it is missing. Shared data, not a shared
# gate: both executors still check their own calls, they just name the refusal the
# same way, so the Operator dialog can answer either one.
_GRANT_REASON_CODES = {
    "WorkspaceRootGrant": "workspace_root_grant_required",
    "WriteGrant": "write_grant_required",
    "WebAccessGrant": "web_access_grant_required",
    "CorpusGrant": "corpus_grant_required",
}


def grant_reason_code(grant: str) -> str:
    return _GRANT_REASON_CODES.get(grant, "grant_required")


# Reasons whose question is "may the model write here": the Operator can answer
# them once for the whole Conversation. A tainted write is never one of them.
# They live here, and not with the gate that reads them, because more than one
# gate does: the Operator's and the bench's have to agree on what a waiver covers
# or the corpus measures a different system than production runs.
WAIVABLE_CONFIRMATION_REASONS = frozenset({"write_confirmation_required", "write_grant_required"})

# Yolo answers every question the gate can ask, including the tainted write. The
# Operator turns it on knowing that a page the model read can now drive a write
# without being announced — see docs/adr/0008-operator-yolo-mode.md. Each such
# call is still recorded, as "waived" and never as "approved".
YOLO_CONFIRMATION_REASONS = WAIVABLE_CONFIRMATION_REASONS | {
    "web_taint_confirmation_required",
    "web_access_grant_required",
}


def waived_reason_code(requested: str) -> str:
    """Answers in the words of the question: …_required becomes …_waived."""
    return f"{requested.removesuffix('_required')}_waived"


class RequestStatus(StrEnum):
    QUEUED = "queued"
    CANCELED = "canceled"
    DEQUEUED = "dequeued"


class TurnStatus(StrEnum):
    ACTIVE = "active"
    FINISHED = "finished"


class ToolResultStatus(StrEnum):
    SUCCESS = "success"
    EMPTY = "empty"
    BLOCKED = "blocked"
    FAILED = "failed"


class TerminalOutcomeKind(StrEnum):
    COMPLETED = "completed"
    LIMIT_REACHED = "limit_reached"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    FAILED = "failed"
    ABANDONED = "abandoned"


class CanonicalHistoryEntryKind(StrEnum):
    USER_MESSAGE = "user_message"
    MODEL_ATTEMPT = "model_attempt"
    REJECTED_MODEL_ATTEMPT = "rejected_model_attempt"
    TOOL_RESULT = "tool_result"
    FINAL_RESPONSE = "final_response"
    INTERNAL_AUTOMATION = "internal_automation"


@dataclass(frozen=True, slots=True)
class WorkspaceRevision:
    workspace_id: str
    revision: int
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    workspace_id: str
    created_at: datetime
    updated_at: datetime
    last_active_at: datetime
    name: str = "New conversation"
    archived_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Corpus:
    """An acervo the Operator assembled, with a store of its own.

    Counts travel with it because the only reason to list corpora is to choose
    one, and a corpus with no Documents is a different choice from a full one.
    """

    id: str
    name: str
    description: str
    embedding_model: str
    embedding_dimensions: int
    created_at: datetime
    updated_at: datetime
    document_count: int = 0
    chunk_count: int = 0


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    origin_kind: str
    origin_ref: str
    title: str
    source_digest: str
    taints: tuple[str, ...]
    ingested_at: datetime
    chunk_count: int = 0


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A Chunk on its way to the model, address included.

    ``text`` is sliced from the stored Document and never rewritten: an answer
    can only be checked against a passage that says what the source said.
    """

    id: str
    document_id: str
    document_title: str
    origin_kind: str
    origin_ref: str
    location: str
    text: str
    score: float
    taints: tuple[str, ...]
    # A fusão por rank ordena mas não mede: o primeiro colocado pontua igual
    # sendo ótimo ou péssimo. A similaridade é a única das duas que diz o quanto
    # a passagem se parece com a pergunta, e é ela que o Operator lê na UI.
    similarity: float | None = None


@dataclass(frozen=True, slots=True)
class PendingRequest:
    id: str
    conversation_id: str
    sequence: int
    content: str
    status: RequestStatus
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Grant:
    id: str
    conversation_id: str
    permission: str
    scope: str
    granted_at: datetime
    expires_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SessionPolicy:
    conversation_id: str
    grants: tuple[Grant, ...] = ()

    @property
    def effective_grants(self) -> frozenset[str]:
        now = datetime.now(UTC)
        return frozenset(
            grant.permission
            for grant in self.grants
            if grant.conversation_id == self.conversation_id
            and grant.granted_at <= now
            and (grant.expires_at is None or grant.expires_at > now)
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, JsonValue]
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class ResultPayload:
    data: JsonValue
    error: Mapping[str, JsonValue] | None

    def __post_init__(self) -> None:
        _require_mapping_or_none(self.error)
        if self.data is not None and self.error is not None:
            raise ValueError("result data and error cannot both be non-null")


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_call_id: str
    status: ToolResultStatus
    retryable: bool
    data: JsonValue
    error: Mapping[str, JsonValue] | None
    meta: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        ResultPayload(data=self.data, error=self.error)

    @property
    def payload(self) -> ResultPayload:
        return ResultPayload(data=self.data, error=self.error)


@dataclass(frozen=True, slots=True)
class AgentStep:
    id: str
    turn_id: str
    sequence: int
    seed: int
    tool_calls: tuple[ToolCall, ...]
    tool_results: tuple[ToolResult, ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    kind: TerminalOutcomeKind
    reason_code: str
    recorded_at: datetime
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.reason_code:
            raise ValueError("terminal outcome reason_code must not be empty")


@dataclass(frozen=True, slots=True)
class CanonicalHistoryEntry:
    id: str
    sequence: int
    conversation_id: str
    turn_id: str
    kind: CanonicalHistoryEntryKind
    payload: Mapping[str, JsonValue]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Feedback:
    id: str
    conversation_id: str
    rating: int
    comment: str | None
    created_at: datetime
    turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class DomainEvent:
    id: str
    sequence: int
    event_type: str
    payload: Mapping[str, JsonValue]
    idempotency_key: str
    occurred_at: datetime
    conversation_id: str | None = None
    published_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ObservabilityEvent:
    id: str
    sequence: int
    event_type: str
    payload: Mapping[str, JsonValue]
    occurred_at: datetime
    turn_id: str | None = None
    step_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class Turn:
    id: str
    conversation_id: str
    request_id: str
    status: TurnStatus
    started_at: datetime
    base_seed: int
    ended_at: datetime | None = None
    terminal_outcome: TerminalOutcome | None = None


def _require_mapping_or_none(value: object) -> None:
    if value is not None and not isinstance(value, Mapping):
        raise TypeError("result error must be a mapping or null")


def effective_tool_calls(
    calls: Sequence[ToolCall],
    results: Sequence[ToolResult],
) -> tuple[ToolCall, ...]:
    """Drops repeats that told the Turn nothing it did not already have.

    A call is a repeat when an earlier *successful* call in the same Turn had the
    same name and arguments and came back with a byte-identical payload. Nothing
    is cached to decide this: the tool ran, the filesystem or the network was
    consulted, and only then did the two payloads turn out to be the same — which
    keeps ``never_cache_workspace_reads`` intact, TOCTOU included, while refusing
    to charge a Turn twice for one piece of information.

    Repeats that come back *different* are not repeats: something changed, and
    both readings are real.

    Only success is discounted. A refusal repeated verbatim is not free
    information, it is a Turn stuck on the same mistake, and making it cost
    nothing would turn every refusal into an unlimited retry.
    """
    by_id = {result.tool_call_id: result for result in results}
    seen: dict[str, str] = {}
    effective: list[ToolCall] = []
    for call in calls:
        result = by_id.get(call.id)
        if result is None or result.status is not ToolResultStatus.SUCCESS:
            effective.append(call)
            continue
        signature = tool_call_signature(call)
        payload = _canonical(result.status.value, {"data": result.data, "error": result.error})
        if seen.get(signature) == payload:
            continue
        seen[signature] = payload
        effective.append(call)
    return tuple(effective)


def tool_call_signature(call: ToolCall) -> str:
    """Identity of a call for repeat detection: the name and the arguments."""
    return _canonical(call.name, call.arguments)


def _canonical(prefix: str, value: object) -> str:
    return prefix + "\x00" + json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
