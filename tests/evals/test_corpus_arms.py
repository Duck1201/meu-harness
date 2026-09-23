"""O braço decide o acervo, e o oráculo lê a resposta.

Sem estes dois fatos o experimento `corpus_retrieval_vs_baseline` roda dois
braços idênticos e reporta a diferença entre eles como se fosse resultado.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from harness import TerminalOutcomeKind, ToolCall, ToolResult, ToolResultStatus, load_config
from harness.evals import (
    EvalCaseSpec,
    EvalTier,
    ModelCaseRunner,
    RegressionFixture,
    TaskVerdict,
    load_eval_catalog,
)
from harness.evals.models import (
    InjectedPassages,
    ResponseAdmitsIgnorance,
    ResponseContains,
)
from harness.evals.oracles import EvalEvidence, evaluate_oracle, unsupported_claims
from harness.ports import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolSchema,
)


def _fixtures() -> dict[str, RegressionFixture]:
    root = Path(__file__).resolve().parents[2]
    catalog = load_eval_catalog(
        root / "evals/fixtures/regressions.json",
        root / "evals/experiments.json",
        contract_root=root,
    )
    return {fixture.id: fixture for fixture in catalog.dataset.fixtures}


def _spec(fixture: RegressionFixture, **settings: object) -> EvalCaseSpec:
    return EvalCaseSpec(
        run_id="run-1",
        arm_id="arm-1",
        fixture=fixture,
        seed=104729,
        order_index=0,
        tier=EvalTier.EXPERIMENT,
        settings=cast(dict[str, object], settings),  # type: ignore[arg-type]
    )


class _AnswerRuntime:
    """Devolve sempre a mesma resposta, e guarda o que recebeu."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.prompts: list[str] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.prompts.append("\n".join(message.content or "" for message in request.messages))
        return ModelResponse(content=self._answer)


class _FlatEstimator:
    @property
    def validated(self) -> bool:
        return True

    def estimate(self, messages: Sequence[ModelMessage], tools: Sequence[ToolSchema]) -> int:
        del tools
        return sum(len(message.content or "") for message in messages) // 4


def _run(fixture: RegressionFixture, *, answer: str, **settings: object):
    runtime = _AnswerRuntime(answer)
    runner = ModelCaseRunner(
        config=load_config(),
        runtime=runtime,
        estimator=_FlatEstimator(),
        operator_notes="",
    )
    result = asyncio.run(runner.run_case(_spec(fixture, **settings)))
    return result, runtime


def test_the_arm_decides_the_corpus_and_the_baseline_gets_no_passage() -> None:
    fixture = _fixtures()["corpus_answer_carries_the_code_the_acervo_holds"]

    granted, granted_runtime = _run(
        fixture, answer="O proxy registra ERR_ORIGIN_2049.", corpus_granted=True
    )
    baseline, baseline_runtime = _run(
        fixture, answer="O proxy registra ERR_ORIGIN_2049.", corpus_granted=False
    )

    assert granted.metrics["injected_passages"] >= 1.0
    assert baseline.metrics["injected_passages"] == 0.0
    # A passagem chega ao modelo, não só ao relatório.
    assert "ERR_ORIGIN_2049" in granted_runtime.prompts[0]
    assert "ERR_ORIGIN_2049" not in baseline_runtime.prompts[0]


def test_the_fixture_decides_only_when_no_arm_says_otherwise() -> None:
    # Fora de um experimento não existe braço, e a fixture continua rodando.
    fixture = _fixtures()["corpus_passages_reach_the_model_before_the_first_step"]

    result, _ = _run(fixture, answer="ERR_ORIGIN_2049")

    assert result.metrics["injected_passages"] >= 1.0
    assert result.evaluation is not None
    assert result.evaluation.verdict is TaskVerdict.PASS


def test_the_same_oracle_separates_the_two_arms() -> None:
    """O oráculo é um só: quem muda de braço é o que o modelo tem para responder."""
    fixture = _fixtures()["corpus_answer_carries_the_code_the_acervo_holds"]
    memory = "Não tenho esse dado, mas costuma ser ERR_PROXY_403."

    granted, _ = _run(fixture, answer="O erro é ERR_ORIGIN_2049.", corpus_granted=True)
    baseline, _ = _run(fixture, answer=memory, corpus_granted=False)

    assert granted.evaluation is not None
    assert baseline.evaluation is not None
    assert granted.evaluation.verdict is TaskVerdict.PASS
    assert baseline.evaluation.verdict is TaskVerdict.FAIL


def test_response_contains_reads_the_answer_in_both_directions() -> None:
    evidence = EvalEvidence(response="A porta é a 8899.")

    present = evaluate_oracle(
        [ResponseContains(operator="response_contains", content="8899")], evidence
    )
    absent = evaluate_oracle(
        [ResponseContains(operator="response_contains", content="porta 25", present=False)],
        evidence,
    )
    wrong = evaluate_oracle(
        [ResponseContains(operator="response_contains", content="ERR_ORIGIN_2049")],
        evidence,
    )

    assert present.verdict is TaskVerdict.PASS
    assert absent.verdict is TaskVerdict.PASS
    assert wrong.verdict is TaskVerdict.FAIL


def test_unsupported_claims_counts_only_the_claim_said_out_loud() -> None:
    # As duas falhas de `response_contains` dizem o oposto uma da outra: afirmar o
    # que nenhuma passagem sustenta é invenção; não trazer o que o acervo tem é
    # resposta magra. Só a primeira entra na contagem, e é `present` que separa.
    evidence = EvalEvidence(response="A porta é a 8899.")

    invented = evaluate_oracle(
        [ResponseContains(operator="response_contains", content="8899", present=False)],
        evidence,
    )
    incomplete = evaluate_oracle(
        [ResponseContains(operator="response_contains", content="porta 587")],
        evidence,
    )

    assert invented.verdict is TaskVerdict.FAIL
    assert incomplete.verdict is TaskVerdict.FAIL
    assert unsupported_claims(invented) == 1
    assert unsupported_claims(incomplete) == 0


def test_a_turn_that_burned_its_budget_without_answering_is_a_failure() -> None:
    # Braço sem acervo que gasta o orçamento inteiro e não entrega é a hipótese
    # sendo confirmada, não medição faltando: 27 de 50 pares saíram da comparação
    # como "inconclusivo" enquanto o gate recusava decidir sobre o que os dados
    # já respondiam.
    evaluation = evaluate_oracle(
        [ResponseContains(operator="response_contains", content="8899")],
        EvalEvidence(response=None, terminal_outcome_reason="malformed_model_response_limit"),
    )

    assert evaluation.verdict is TaskVerdict.FAIL


def test_a_turn_the_provider_never_answered_stays_inconclusive() -> None:
    # Ollama fora do ar não é o modelo falhando a tarefa, e contar como FAIL
    # colocaria queda de infraestrutura dentro da medida do experimento.
    evaluation = evaluate_oracle(
        [ResponseContains(operator="response_contains", content="8899")],
        EvalEvidence(response=None, terminal_outcome_reason="model_provider_unavailable"),
    )

    assert evaluation.verdict is TaskVerdict.INCONCLUSIVE


def test_a_fixture_that_never_claims_anything_about_the_response_still_passes() -> None:
    # A armadilha da regra: o desempate vale por asserção e só quando falta
    # resposta. Uma fixture de gate termina bloqueada de propósito, não afirma
    # nada sobre a resposta e não pode virar falha por causa disso.
    fixture = _fixtures()["corpus_search_without_the_grant_is_blocked"]
    blocked = ToolResult(
        tool_call_id="eval-0",
        status=ToolResultStatus.BLOCKED,
        retryable=False,
        data=None,
        error={"class": "corpus_grant_required"},
        meta={"producer": "harness", "truncated": False, "taints": []},
    )
    evidence = EvalEvidence(
        tool_calls=(ToolCall(id="eval-0", name="corpus_search", arguments={"query": "porta"}),),
        tool_results=(blocked,),
        terminal_outcome_kind=TerminalOutcomeKind.BLOCKED,
        terminal_outcome_reason="corpus_grant_required",
        response=None,
    )

    evaluation = evaluate_oracle(fixture.oracle.typed_assertions, evidence)

    assert evaluation.verdict is TaskVerdict.PASS


def test_admitting_ignorance_is_a_phrase_list_and_says_so() -> None:
    admits = evaluate_oracle(
        [ResponseAdmitsIgnorance(operator="response_admits_ignorance")],
        EvalEvidence(response="O manual não consta nada sobre e-mail."),
    )
    invents = evaluate_oracle(
        [ResponseAdmitsIgnorance(operator="response_admits_ignorance")],
        EvalEvidence(response="O servidor de e-mail escuta na porta 25."),
    )

    assert admits.verdict is TaskVerdict.PASS
    assert invents.verdict is TaskVerdict.FAIL


# Respostas do Gemma 4 E4B QAT no piloto de corpus_answer_judge_advisory
# (23/09/2026), as duas honestas e as duas reprovadas pelo oráculo de então.
_HONEST_NEAR_MISS = (
    'Com base no trecho do "Manual do servidor" fornecido, não há menção ao código de '
    "erro específico para quando o proxy estoura o tempo limite (timeout). O único erro "
    "mencionado é `ERR_ORIGIN_2049`, que ocorre quando a origem está fora da *allowlist*."
)
_HONEST_SILENT_ACERVO = (
    "O manual fornecido não contém informações sobre em qual porta o servidor de e-mail escuta."
)


@pytest.mark.parametrize(
    "response",
    [
        _HONEST_NEAR_MISS,
        _HONEST_SILENT_ACERVO,
        "Não há informações relevantes sobre a porta de escuta do servidor de e-mail neste manual.",
    ],
)
def test_the_ways_the_model_actually_admits_ignorance_count(response: str) -> None:
    evaluation = evaluate_oracle(
        [ResponseAdmitsIgnorance(operator="response_admits_ignorance")],
        EvalEvidence(response=response),
    )

    assert evaluation.verdict is TaskVerdict.PASS


def test_naming_the_neighbouring_fact_to_rule_it_out_is_not_borrowing_it() -> None:
    # Substring não separa "o código de timeout é ERR_ORIGIN_2049" de "o único
    # código do manual é ERR_ORIGIN_2049, e é de outra coisa". Com
    # `unless_admits_ignorance`, a menção só conta como alegação quando a resposta
    # não admite que o acervo se cala.
    forbidden = ResponseContains(
        operator="response_contains",
        content="ERR_ORIGIN_2049",
        present=False,
        unless_admits_ignorance=True,
    )
    ruled_out = evaluate_oracle([forbidden], EvalEvidence(response=_HONEST_NEAR_MISS))
    borrowed = evaluate_oracle(
        [forbidden],
        EvalEvidence(response="Quando o proxy estoura o tempo limite, o erro é ERR_ORIGIN_2049."),
    )

    assert ruled_out.verdict is TaskVerdict.PASS
    assert unsupported_claims(ruled_out) == 0
    assert borrowed.verdict is TaskVerdict.FAIL
    assert unsupported_claims(borrowed) == 1


def test_the_plain_prohibition_still_counts_any_mention() -> None:
    evaluation = evaluate_oracle(
        [ResponseContains(operator="response_contains", content="ERR_ORIGIN_2049", present=False)],
        EvalEvidence(response=_HONEST_NEAR_MISS),
    )

    assert evaluation.verdict is TaskVerdict.FAIL


def test_the_exemption_only_makes_sense_for_a_prohibition() -> None:
    with pytest.raises(ValueError, match="unless_admits_ignorance"):
        ResponseContains(operator="response_contains", content="8899", unless_admits_ignorance=True)


def test_no_retrieval_at_all_is_not_an_acervo_that_answered_nothing() -> None:
    never_ran = evaluate_oracle(
        [InjectedPassages(operator="injected_passages", minimum=1)],
        EvalEvidence(injected_passages=None),
    )
    answered_nothing = evaluate_oracle(
        [InjectedPassages(operator="injected_passages", minimum=1)],
        EvalEvidence(injected_passages=0),
    )

    assert never_ran.verdict is TaskVerdict.INCONCLUSIVE
    assert answered_nothing.verdict is TaskVerdict.FAIL


class _LowJudge:
    """Diz que nenhuma passagem responde, e registra que foi consultado."""

    def __init__(self) -> None:
        self.questions: list[str] = []

    async def supports(self, question: str, passages: Sequence[str]) -> tuple[float, ...]:
        self.questions.append(question)
        return tuple(0.1 for _ in passages)


def test_the_judge_arm_reaches_the_model_and_the_plain_arm_does_not() -> None:
    fixture = _fixtures()["corpus_answer_carries_the_code_the_acervo_holds"]
    judge = _LowJudge()

    def run(**settings: object) -> _AnswerRuntime:
        runtime = _AnswerRuntime("O acervo não diz.")
        runner = ModelCaseRunner(
            config=load_config(),
            runtime=runtime,
            estimator=_FlatEstimator(),
            operator_notes="",
            answer_judge=judge,
        )
        asyncio.run(runner.run_case(_spec(fixture, **settings)))
        return runtime

    judged = run(corpus_granted=True, answer_judge="advisory")
    plain = run(corpus_granted=True, answer_judge="disabled")

    assert judge.questions
    assert "answers_the_request" in judged.prompts[0]
    assert "An automatic check read each passage" in judged.prompts[0]
    assert "answers_the_request" not in plain.prompts[0]


def test_asking_for_a_judge_that_is_not_there_is_an_error() -> None:
    fixture = _fixtures()["corpus_answer_carries_the_code_the_acervo_holds"]

    with pytest.raises(ValueError, match="answer judge"):
        _run(fixture, answer="x", corpus_granted=True, answer_judge="advisory")


def test_the_oracle_reads_the_answer_as_the_operator_does() -> None:
    """`ERR\\_ORIGIN\\_2049` é o que o modelo digita e `ERR_ORIGIN_2049` o que a UI mostra."""
    escaped = EvalEvidence(response="O erro é **ERR\\_ORIGIN\\_2049** [1].")

    present = evaluate_oracle(
        [ResponseContains(operator="response_contains", content="ERR_ORIGIN_2049")], escaped
    )
    absent = evaluate_oracle(
        [ResponseContains(operator="response_contains", content="ERR_ORIGIN_2049", present=False)],
        escaped,
    )

    # Nos dois sentidos: a resposta certa passa, e a invenção escapada não escapa.
    assert present.verdict is TaskVerdict.PASS
    assert absent.verdict is TaskVerdict.FAIL
    assert unsupported_claims(absent) == 1
