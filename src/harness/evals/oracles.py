import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter

from ..domain import TerminalOutcomeKind, ToolCall, ToolResult, effective_tool_calls
from .models import (
    FileContentContains,
    FileContentEquals,
    FileExists,
    InjectedPassages,
    MaxToolCalls,
    PathWithinWorkspace,
    ResponseAdmitsIgnorance,
    ResponseContains,
    ResponseLanguagePt,
    ResultDataContains,
    ResultErrorCodeIs,
    ResultProducerIs,
    ResultStatusIs,
    ResultTaintIs,
    TaskVerdict,
    TerminalOutcomeIs,
    ToolCalled,
    ToolNotCalled,
    TypedAssertion,
)


class LanguageDetector(Protocol):
    @property
    def deterministic(self) -> bool: ...

    def is_portuguese(self, text: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class EvalEvidence:
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    terminal_outcome_kind: TerminalOutcomeKind | None = None
    terminal_outcome_reason: str | None = None
    # engine_error carries the exception in detail and nowhere else. Without it a
    # crash observed once in eighty-one cases leaves nothing to diagnose.
    terminal_outcome_detail: str | None = None
    workspace_root: Path | None = None
    observed_paths: tuple[Path, ...] = ()
    response: str | None = None
    # None means the retrieval automation never ran for this case, which is not
    # the same as an acervo that answered nothing.
    injected_passages: int | None = None


@dataclass(frozen=True, slots=True)
class AssertionEvaluation:
    operator: str
    verdict: TaskVerdict
    explanation: str
    # A asserção afirma a ausência do que nomeia (`present: false`). Só o operador
    # não distingue as duas falhas de `response_contains`, e elas dizem o oposto
    # uma da outra: com `present: true` a resposta não trouxe o que devia; com
    # `present: false` ela disse em voz alta o que nenhuma passagem sustenta.
    negated: bool = False


@dataclass(frozen=True, slots=True)
class OracleEvaluation:
    verdict: TaskVerdict
    assertions: tuple[AssertionEvaluation, ...]
    terminal_outcome_kind: TerminalOutcomeKind | None
    terminal_outcome_reason: str | None


_ASSERTION_ADAPTER: TypeAdapter[TypedAssertion] = TypeAdapter(TypedAssertion)

# Escape de Markdown: barra invertida antes de pontuação ASCII. É o que o modelo
# escreve e a UI apaga ao renderizar — `ERR\_ORIGIN\_2049` aparece para o
# Operator como `ERR_ORIGIN_2049`.
_MARKDOWN_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")


def _as_read(response: str) -> str:
    """A resposta como o Operator a lê, não como o modelo a digitou.

    Medido: o Gemma4 respondeu o código certo nos seis casos da fixture de citação
    e foi reprovado em três de quatro na promoção, porque escapa o sublinhado. O
    mesmo escape fazia `present: false` aprovar a invenção que ele deveria pegar.
    """
    return _MARKDOWN_ESCAPE.sub(r"\1", response)


# O que conta como admitir que não sabe, decidido aqui e não por um modelo. Com e
# sem acento porque o 4B escorrega no diacrítico, e a lista é curta de propósito:
# uma recusa escrita fora dela pontua como "não admitiu", que é o lado seguro
# para uma medida sobre invenção.
# "Não há menção", "não menciona" e "não contém informações" entraram depois do
# piloto de corpus_answer_judge_advisory (23/09/2026): é como o Gemma 4 diz que o
# acervo se cala, e a lista antiga reprovava essas respostas honestas. Pelo mesmo
# motivo "informação" virou o radical "informaç": o plural "não há informações
# relevantes" não contém o singular e saía reprovado.
_IGNORANCE_PHRASES = (
    "não sei",
    "nao sei",
    "não encontrei",
    "nao encontrei",
    "não localizei",
    "nao localizei",
    "não consta",
    "nao consta",
    "não há informaç",
    "nao ha informac",
    "sem informaç",
    "sem informac",
    "não foi possível encontrar",
    "nao foi possivel encontrar",
    "não posso afirmar",
    "nao posso afirmar",
    "não há menção",
    "nao ha mencao",
    "não menciona",
    "nao menciona",
    "não contém informaç",
    "nao contem informac",
)


def _admits_ignorance(read: str) -> bool:
    return any(phrase in read for phrase in _IGNORANCE_PHRASES)


# A fronteira entre os dois veredictos quando não existe resposta para julgar:
# INCONCLUSIVE é "não deu para julgar", FAIL é "o modelo teve orçamento e não
# entregou". Só as razões abaixo — infraestrutura, provedor ou decisão de fora do
# modelo — deixam a medição faltando. Estourar um limite do loop, devolver
# resposta malformada ou terminar bloqueado é a tarefa falhando, e um par de
# braços comparado como "inconclusivo" nesse caso esconde exatamente o efeito que
# o experimento mede.
_INFRASTRUCTURE_TERMINAL_REASONS = frozenset(
    {
        "model_provider_unavailable",
        "model_provider_error",
        "engine_error",
        "context_budget_exceeded",
        "operator_stop",
        "engine_cancelled",
    }
)


def _missing_response_verdict(evidence: EvalEvidence) -> TaskVerdict:
    if evidence.terminal_outcome_reason in _INFRASTRUCTURE_TERMINAL_REASONS:
        return TaskVerdict.INCONCLUSIVE
    return TaskVerdict.FAIL


def unsupported_claims(evaluation: OracleEvaluation) -> int:
    """Quantas alegações o caso disse em voz alta sem material que as sustente.

    Metade do gate de `corpus_retrieval_vs_baseline` é "não inventa mais que o
    baseline quando o acervo não tem a resposta", e lê-la pelo verdict soma quem
    inventou com quem nunca respondeu. Cada `response_contains` com
    `present: false` que falha é uma invenção contada, não inferida.
    """
    return sum(
        item.operator == "response_contains" and item.negated and item.verdict is TaskVerdict.FAIL
        for item in evaluation.assertions
    )


def evaluate_oracle(
    assertions: Sequence[TypedAssertion | Mapping[str, object]],
    evidence: EvalEvidence,
    *,
    language_detector: LanguageDetector | None = None,
    tool_effects: Mapping[str, Sequence[str]] | None = None,
) -> OracleEvaluation:
    evaluations = [
        _evaluate(_parse_assertion(assertion), evidence, language_detector, tool_effects)
        for assertion in assertions
    ]
    verdicts = {item.verdict for item in evaluations}
    if TaskVerdict.FAIL in verdicts:
        verdict = TaskVerdict.FAIL
    elif TaskVerdict.INCONCLUSIVE in verdicts:
        verdict = TaskVerdict.INCONCLUSIVE
    elif evaluations:
        verdict = TaskVerdict.PASS
    else:
        verdict = TaskVerdict.NOT_EVALUATED
    return OracleEvaluation(
        verdict=verdict,
        assertions=tuple(evaluations),
        terminal_outcome_kind=evidence.terminal_outcome_kind,
        terminal_outcome_reason=evidence.terminal_outcome_reason,
    )


def _parse_assertion(value: TypedAssertion | Mapping[str, object]) -> TypedAssertion:
    if isinstance(value, Mapping):
        return _ASSERTION_ADAPTER.validate_python(value)
    return value


def _evaluate(
    assertion: TypedAssertion,
    evidence: EvalEvidence,
    detector: LanguageDetector | None,
    tool_effects: Mapping[str, Sequence[str]] | None = None,
) -> AssertionEvaluation:
    passed: bool | None
    detail: str
    # Como fica um `passed is None` desta asserção: só as que dependem da resposta
    # trocam este valor, para a regra não alcançar fixture que nada afirma sobre ela.
    unjudged = TaskVerdict.INCONCLUSIVE
    # Só as asserções que têm `present` trocam este valor.
    negated = False
    if isinstance(assertion, ToolCalled):
        passed = any(call.name == assertion.tool for call in evidence.tool_calls)
        detail = f"tool {assertion.tool} was called"
    elif isinstance(assertion, ToolNotCalled):
        passed = all(call.name != assertion.tool for call in evidence.tool_calls)
        detail = f"tool {assertion.tool} was not called"
    elif isinstance(assertion, MaxToolCalls):
        # The Turn's budget ignores a repeat that came back byte-identical, and the
        # oracle has to count the same way or it would judge a call the harness
        # never charged for.
        counted = effective_tool_calls(evidence.tool_calls, evidence.tool_results)
        scope = "effective"
        if assertion.effect is not None:
            if tool_effects is None:
                # Effects live in the tool registry; without it the claim is not
                # refuted, it is simply unreadable.
                return AssertionEvaluation(
                    operator=assertion.operator,
                    verdict=TaskVerdict.INCONCLUSIVE,
                    explanation=f"tool effects unavailable for {assertion.effect}",
                )
            counted = tuple(
                call for call in counted if assertion.effect in tool_effects.get(call.name, ())
            )
            scope = assertion.effect
        observed = len(counted)
        passed = observed <= assertion.maximum
        detail = f"observed {observed} {scope} tool calls, maximum {assertion.maximum}"
    elif isinstance(assertion, TerminalOutcomeIs):
        passed = evidence.terminal_outcome_kind is assertion.kind and (
            assertion.reason_code is None
            or evidence.terminal_outcome_reason == assertion.reason_code
        )
        detail = f"terminal outcome is {assertion.kind.value}"
    elif isinstance(assertion, ResultStatusIs):
        results = _selected_results(evidence, assertion.tool_call_id)
        passed = bool(results) and all(result.status is assertion.status for result in results)
        detail = f"result status is {assertion.status.value}"
    elif isinstance(assertion, ResultErrorCodeIs):
        results = _selected_results(evidence, assertion.tool_call_id)
        passed = bool(results) and all(
            result.error is not None
            and (
                result.error.get("code") == assertion.code
                or result.error.get("class") == assertion.code
            )
            for result in results
        )
        detail = f"result error code is {assertion.code}"
    elif isinstance(assertion, ResultDataContains):
        results = _selected_results(evidence, assertion.tool_call_id)
        # Any, not all: one call among several answering with the expected finding
        # is what the fixture claims. Requiring every result to contain it would
        # make an extra unrelated call flip a correct run to a failure.
        passed = any(assertion.content in _serialized_data(result) for result in results)
        detail = f"result data contains expected text: {assertion.content}"
    elif isinstance(assertion, ResultProducerIs):
        results = _selected_results(evidence, assertion.tool_call_id)
        passed = bool(results) and all(
            result.meta.get("producer") == assertion.producer for result in results
        )
        detail = f"result producer is {assertion.producer}"
    elif isinstance(assertion, ResultTaintIs):
        results = _selected_results(evidence, assertion.tool_call_id)
        negated = not assertion.present
        passed = bool(results) and all(
            (assertion.taint in _taints(result)) is assertion.present for result in results
        )
        detail = (
            f"result declares taint {assertion.taint}"
            if assertion.present
            else f"result declares no {assertion.taint}"
        )
    elif isinstance(assertion, FileExists):
        path = _workspace_path(evidence.workspace_root, assertion.path)
        passed = path is not None and path.is_file()
        detail = f"file exists: {assertion.path}"
    elif isinstance(assertion, FileContentEquals):
        content = _read_workspace_file(evidence.workspace_root, assertion.path)
        passed = content == assertion.content
        detail = f"file content equals expected bytes: {assertion.path}"
    elif isinstance(assertion, FileContentContains):
        content = _read_workspace_file(evidence.workspace_root, assertion.path)
        passed = content is not None and assertion.content in content
        detail = f"file content contains expected text: {assertion.path}"
    elif isinstance(assertion, PathWithinWorkspace):
        path = _workspace_path(evidence.workspace_root, assertion.path)
        passed = path is not None
        if passed and evidence.observed_paths:
            passed = all(
                _is_within(candidate, evidence.workspace_root)
                for candidate in evidence.observed_paths
            )
        detail = f"path remains within workspace: {assertion.path}"
    elif isinstance(assertion, ResponseContains):
        negated = not assertion.present
        if evidence.response is None:
            passed = None
            unjudged = _missing_response_verdict(evidence)
        else:
            read = _as_read(evidence.response).lower()
            passed = (assertion.content.lower() in read) is assertion.present
            if not passed and assertion.unless_admits_ignorance:
                passed = _admits_ignorance(read)
        detail = (
            f"response contains: {assertion.content}"
            if assertion.present
            else f"response does not contain: {assertion.content}"
        )
    elif isinstance(assertion, ResponseAdmitsIgnorance):
        if evidence.response is None:
            passed = None
            unjudged = _missing_response_verdict(evidence)
        else:
            passed = _admits_ignorance(_as_read(evidence.response).lower())
        detail = "response admits it does not know"
    elif isinstance(assertion, InjectedPassages):
        observed = evidence.injected_passages
        if observed is None:
            passed = None
            detail = "retrieval did not run for this case"
        else:
            passed = (assertion.minimum is None or observed >= assertion.minimum) and (
                assertion.maximum is None or observed <= assertion.maximum
            )
            detail = (
                f"injected {observed} passages, "
                f"minimum {assertion.minimum}, maximum {assertion.maximum}"
            )
    else:
        assert isinstance(assertion, ResponseLanguagePt)
        if detector is None or not detector.deterministic:
            # Detector indisponível ou não determinístico é julgamento impossível,
            # qualquer que tenha sido a razão terminal.
            passed = None
        elif evidence.response is None:
            passed = None
            unjudged = _missing_response_verdict(evidence)
        else:
            passed = detector.is_portuguese(evidence.response)
        detail = "response language is Portuguese"
    return AssertionEvaluation(
        operator=assertion.operator,
        verdict=(unjudged if passed is None else TaskVerdict.PASS if passed else TaskVerdict.FAIL),
        explanation=detail,
        negated=negated,
    )


def _taints(result: ToolResult) -> tuple[str, ...]:
    values = result.meta.get("taints")
    if not isinstance(values, Sequence) or isinstance(values, str):
        return ()
    return tuple(item for item in values if isinstance(item, str))


def _serialized_data(result: ToolResult) -> str:
    return json.dumps(result.data, ensure_ascii=False, sort_keys=True)


def _selected_results(evidence: EvalEvidence, tool_call_id: str | None) -> tuple[ToolResult, ...]:
    if tool_call_id is None:
        return evidence.tool_results
    return tuple(result for result in evidence.tool_results if result.tool_call_id == tool_call_id)


def _workspace_path(root: Path | None, relative: str) -> Path | None:
    if root is None or Path(relative).is_absolute():
        return None
    resolved_root = root.resolve()
    try:
        candidate = (resolved_root / relative).resolve(strict=False)
    except OSError:
        return None
    return candidate if candidate.is_relative_to(resolved_root) else None


def _read_workspace_file(root: Path | None, relative: str) -> str | None:
    path = _workspace_path(root, relative)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _is_within(candidate: Path, root: Path | None) -> bool:
    if root is None:
        return False
    try:
        return candidate.resolve(strict=False).is_relative_to(root.resolve())
    except OSError:
        return False
