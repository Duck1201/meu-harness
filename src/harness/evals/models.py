from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain import JsonValue, TerminalOutcomeKind, ToolResultStatus


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    NOT_EVALUATED = "not_evaluated"


class EvalTier(StrEnum):
    CONTRACT = "contract"
    MODEL_SMOKE = "model_smoke"
    EXPERIMENT = "experiment"


class EvalPhase(StrEnum):
    PILOT = "pilot"
    PROMOTION = "promotion"


class EvalRunStatus(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELING = "canceling"
    CANCELED = "canceled"
    BLOCKED = "blocked"
    FAILED = "failed"


class DigestContract(EvalModel):
    algorithm: Literal["sha256"]
    canonicalization: Literal["sorted_keys_compact_utf8"]
    scope: tuple[str, ...]


class ToolCalled(EvalModel):
    operator: Literal["tool_called"]
    tool: str = Field(min_length=1)


class ToolNotCalled(EvalModel):
    operator: Literal["tool_not_called"]
    tool: str = Field(min_length=1)


class MaxToolCalls(EvalModel):
    """Bounds the calls a Turn spends, optionally only those of one effect class.

    Without ``effect`` this is every effective call, which is the loop bound.
    With one, it is the budget for that class alone: a read inside the workspace
    jail is not the same expense as a write or an egress, and a fixture that
    means "change nothing" should say so instead of counting everything.
    """

    operator: Literal["max_tool_calls"]
    maximum: int = Field(ge=0)
    effect: str | None = None


class TerminalOutcomeIs(EvalModel):
    operator: Literal["terminal_outcome_is"]
    kind: TerminalOutcomeKind
    reason_code: str | None = None


class ResultStatusIs(EvalModel):
    operator: Literal["result_status_is"]
    status: ToolResultStatus
    tool_call_id: str | None = None


class ResultErrorCodeIs(EvalModel):
    operator: Literal["result_error_code_is"]
    code: str = Field(min_length=1)
    tool_call_id: str | None = None


class ResultDataContains(EvalModel):
    """Asserts that some ResultPayload actually surfaced the expected finding.

    The other result operators read the envelope — status, error class, producer.
    This one reads ``data``, which is where a search says which file matched or a
    read says what it returned. Without it, "the tool was called" is all a fixture
    can claim, and a tool that answers nothing useful still passes.
    """

    operator: Literal["result_data_contains"]
    content: str = Field(min_length=1)
    tool_call_id: str | None = None


class ResultProducerIs(EvalModel):
    """Which engine produced a ResultPayload — "web_fetch" for HTTP, "browser" after escalation."""

    operator: Literal["result_producer_is"]
    producer: str = Field(min_length=1)
    tool_call_id: str | None = None


class ResultTaintIs(EvalModel):
    """Whether a ResultPayload declared a provenance mark.

    Taint is what makes the confirmation gate tighten for the rest of the Turn, so
    a fixture that cannot assert it cannot prove the barrier exists.
    """

    operator: Literal["result_taint_is"]
    taint: str = Field(min_length=1)
    present: bool = True
    tool_call_id: str | None = None


class FileExists(EvalModel):
    operator: Literal["file_exists"]
    path: str = Field(min_length=1)


class FileContentEquals(EvalModel):
    operator: Literal["file_content_equals"]
    path: str = Field(min_length=1)
    content: str


class FileContentContains(EvalModel):
    operator: Literal["file_content_contains"]
    path: str = Field(min_length=1)
    content: str


class PathWithinWorkspace(EvalModel):
    operator: Literal["path_within_workspace"]
    path: str = Field(min_length=1)


class ResponseLanguagePt(EvalModel):
    operator: Literal["response_language_pt"]


class ResponseContains(EvalModel):
    """Whether the answer the Operator reads carries a given string.

    The corpus experiment needs it in both directions: with ``present`` the
    fixture asserts the fact the acervo holds actually reached the answer, and
    with ``present=False`` it asserts the model did not state what no passage
    supports. Comparison is case-insensitive because the model is free to
    capitalise as it likes; everything else is a literal match, so a fixture
    stays a fixture instead of becoming a judgement.

    ``unless_admits_ignorance`` narrows a prohibition to the answer that fills
    the gap. A literal match cannot tell "the timeout code is ERR_ORIGIN_2049"
    from "the only code in the manual is ERR_ORIGIN_2049, and it is for something
    else"; with the flag, naming the forbidden string counts as a claim only
    when the answer does not also admit the Corpus is silent. The price is
    explicit: an answer that admits ignorance and then guesses the neighbouring
    fact anyway passes this assertion.
    """

    operator: Literal["response_contains"]
    content: str = Field(min_length=1)
    present: bool = True
    unless_admits_ignorance: bool = False

    @model_validator(mode="after")
    def _exemption_needs_a_prohibition(self) -> Self:
        if self.unless_admits_ignorance and self.present:
            raise ValueError("unless_admits_ignorance only applies with present=false")
        return self


class ResponseAdmitsIgnorance(EvalModel):
    """Whether the answer says it does not know instead of filling the gap.

    A phrase list, not a model: what counts as admitting ignorance is decided
    here, in the open, and the fixture that depends on it can be read without
    running anything. It is a heuristic over pt-BR wording and it says so — a
    refusal phrased in a way this list does not carry scores as not admitting,
    which is the safe direction for a measure about invention.
    """

    operator: Literal["response_admits_ignorance"]


class InjectedPassages(EvalModel):
    """How many Corpus passages the harness injected before the first AgentStep.

    This is what makes an arm an arm: the granted arm has to show passages and
    the baseline has to show none, or the two arms ran the same experiment twice.
    Evidence with no retrieval at all scores inconclusive rather than zero — "the
    automation never ran" and "the acervo answered nothing" are different facts.
    """

    operator: Literal["injected_passages"]
    minimum: int | None = Field(default=None, ge=0)
    maximum: int | None = Field(default=None, ge=0)


type TypedAssertion = Annotated[
    ToolCalled
    | ToolNotCalled
    | MaxToolCalls
    | TerminalOutcomeIs
    | ResultStatusIs
    | ResultErrorCodeIs
    | ResultDataContains
    | ResultProducerIs
    | ResultTaintIs
    | FileExists
    | FileContentEquals
    | FileContentContains
    | PathWithinWorkspace
    | ResponseLanguagePt
    | ResponseContains
    | ResponseAdmitsIgnorance
    | InjectedPassages,
    Field(discriminator="operator"),
]


class OracleDefinition(EvalModel):
    """An oracle is executable or it is not an oracle.

    Free-form assertions used to live here and scored INCONCLUSIVE, so a fixture
    carrying them never passed and never failed while still looking covered.
    Prose now has exactly one home, ``explanation``, which nothing evaluates.
    """

    model_config = ConfigDict(extra="allow")

    typed_assertions: tuple[TypedAssertion, ...] = Field(min_length=1)
    explanation: tuple[str, ...] = ()


class RegressionFixture(EvalModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    tags: tuple[str, ...]
    origin: str = Field(min_length=1)
    stimulus: Mapping[str, JsonValue]
    oracle: OracleDefinition


class RegressionDataset(EvalModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal[2]
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    dataset_digest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    digest_contract: DigestContract
    fixtures: tuple[RegressionFixture, ...]


class DatasetReference(EvalModel):
    path: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    dataset_digest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PromotionProtocol(EvalModel):
    model_config = ConfigDict(extra="allow")

    pilot_runs_per_arm: Literal[15]
    promotion_runs_per_arm: Literal[50]
    recorded_orders_or_seeds: Literal[3]
    security_gate: Literal["zero_violations"]
    # Acima disso a execução não decide nada: o par inconclusivo sai da
    # comparação, e uma execução que perde mais de um quinto dos pares mediu
    # pouco demais para promover ou reprovar seja o que for.
    max_inconclusive_pair_rate: float = Field(ge=0.0, le=1.0)


class ExperimentArm(EvalModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)


class ExperimentDefinition(EvalModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    runtime_profile: str = Field(min_length=1)
    execution_route: str = Field(min_length=1)
    fixture_tags: tuple[str, ...]
    arms: tuple[ExperimentArm, ...]


class ExperimentManifest(EvalModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal[2]
    manifest_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    manifest_digest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    digest_contract: DigestContract
    dataset: DatasetReference
    contract_digests: Mapping[str, str]
    promotion_protocol: PromotionProtocol
    experiments: tuple[ExperimentDefinition, ...]


class EvalCatalog(EvalModel):
    dataset: RegressionDataset
    manifest: ExperimentManifest
