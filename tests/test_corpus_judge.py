"""O juiz de Corpus anota passagens e nunca decide a resposta (ADR 0015)."""

import asyncio
import json
import math
from collections.abc import Mapping, Sequence

import httpx
import pytest

from harness.corpus_judge import (
    INSTRUCTION,
    AnswerJudgeUnavailableError,
    OllamaRerankerAnswerJudge,
    judge_prompt,
    yes_probability,
)
from harness.corpus_service import (
    CITATION_INSTRUCTION,
    NOTHING_FOUND_INSTRUCTION,
    UNSUPPORTED_INSTRUCTION,
    Retrieval,
)
from harness.domain import RetrievedChunk


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=text[:8],
        document_id="doc",
        document_title="Manual",
        origin_kind="upload",
        origin_ref="manual.md",
        location="4.1 Proxy",
        text=text,
        score=0.1,
        taints=(),
    )


def _retrieval(coverage: tuple[float, ...] | None, chunks: int = 2) -> Retrieval:
    return Retrieval(
        corpus_id="c",
        corpus_name="Manual",
        query="q",
        lexical_query=None,
        chunks=tuple(_chunk(f"passagem {index}") for index in range(chunks)),
        coverage=coverage,
        coverage_threshold=0.5,
    )


def _instruction(retrieval: Retrieval) -> object:
    payload = retrieval.payload()
    assert isinstance(payload, Mapping)
    return payload["instruction"]


def test_without_a_judge_the_payload_is_what_it_always_was() -> None:
    retrieval = _retrieval(None)
    payload = retrieval.payload()

    assert isinstance(payload, Mapping)
    assert payload["instruction"] == CITATION_INSTRUCTION
    passages = payload["passages"]
    assert isinstance(passages, list)
    assert all(isinstance(p, Mapping) and "answers_the_request" not in p for p in passages)


def test_each_passage_carries_what_the_judge_said() -> None:
    payload = _retrieval((0.91, 0.12)).payload()

    assert isinstance(payload, Mapping)
    passages = payload["passages"]
    assert isinstance(passages, list)
    assert [p["answers_the_request"] for p in passages if isinstance(p, Mapping)] == [0.91, 0.12]
    assert payload["instruction"] == CITATION_INSTRUCTION


def test_only_a_unanimous_no_changes_the_instruction() -> None:
    assert _instruction(_retrieval((0.49, 0.2))) == UNSUPPORTED_INSTRUCTION
    assert _instruction(_retrieval((0.49, 0.5))) == CITATION_INSTRUCTION
    assert _instruction(_retrieval((), chunks=0)) == NOTHING_FOUND_INSTRUCTION


_DIGEST = "a" * 64
_TAG = "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0"


def _ollama(
    answers: Sequence[Mapping[str, float]], *, digest: str = _DIGEST
) -> tuple[httpx.MockTransport, list[Mapping[str, object]]]:
    """Um Ollama de mentira: /api/tags com a tag do juiz e /api/generate com logprobs."""
    pending = list(answers)
    sent: list[Mapping[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": _TAG, "digest": digest}]})
        body = json.loads(request.content)
        sent.append(body)
        top = [{"token": token, "logprob": value} for token, value in pending.pop(0).items()]
        return httpx.Response(200, json={"response": "no", "logprobs": [{"top_logprobs": top}]})

    return httpx.MockTransport(handler), sent


def _judge(transport: httpx.MockTransport) -> OllamaRerankerAnswerJudge:
    return OllamaRerankerAnswerJudge(
        base_url="http://ollama", model=_TAG, expected_digest=_DIGEST, transport=transport
    )


def test_each_passage_is_one_raw_prompt_scored_yes_against_no() -> None:
    transport, sent = _ollama(
        [{"yes": math.log(0.8), "no": math.log(0.2)}, {"no": -0.1, "No": -2.0, "yes": -5.0}]
    )

    coverage = asyncio.run(_judge(transport).supports("Qual o erro?", ["p1", "p2"]))

    assert coverage[0] == pytest.approx(0.8)
    assert coverage[1] == pytest.approx(1 / (1 + math.exp(-0.1 + 5.0)))
    assert [body["prompt"] for body in sent] == [
        judge_prompt("Qual o erro?", "p1"),
        judge_prompt("Qual o erro?", "p2"),
    ]
    assert all(body["raw"] is True and body["logprobs"] is True for body in sent)
    assert INSTRUCTION in str(sent[0]["prompt"])


def test_a_token_outside_the_top_list_counts_as_the_floor() -> None:
    assert yes_probability({"yes": -0.05, "Yes": -3.0}) == pytest.approx(
        1 / (1 + math.exp(-3.0 + 0.05))
    )


def test_the_judge_never_scores_on_a_tag_the_contract_did_not_pin() -> None:
    transport, sent = _ollama([{"yes": 0.0}], digest="b" * 64)

    with pytest.raises(AnswerJudgeUnavailableError) as raised:
        asyncio.run(_judge(transport).supports("q", ["p"]))

    assert raised.value.code == "model_digest_mismatch"
    assert sent == []
