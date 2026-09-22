"""O juiz de Corpus anota passagens e nunca decide a resposta (ADR 0015)."""

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from harness.corpus_judge import AnswerJudgeUnavailableError, LayaAnswerJudge
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


class _Agent:
    def __init__(self, answers: Sequence[float]) -> None:
        self._answers = list(answers)
        self.states: list[Mapping[str, str]] = []

    def system_one(
        self, state: Mapping[str, str], questions: Mapping[str, Mapping[str, object]]
    ) -> Mapping[str, object]:
        self.states.append(state)
        assert questions["a"]["type"] == "noul"
        return {"answers": {"a": {"type": "noul", "noul": self._answers.pop(0)}}}


def test_the_judge_asks_one_noul_per_passage(tmp_path: Path) -> None:
    agent = _Agent([0.8, 0.1])
    judge = LayaAnswerJudge(tmp_path, weights_sha256="0" * 64, agent=agent)

    coverage = asyncio.run(judge.supports("Qual o erro?", ["p1", "p2"]))

    assert coverage == (0.8, 0.1)
    assert agent.states == [
        {"pergunta": "Qual o erro?", "passagem": "p1"},
        {"pergunta": "Qual o erro?", "passagem": "p2"},
    ]


def _checkpoint(tmp_path: Path, *, tokenizer: bool = True) -> str:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    if tokenizer:
        (tmp_path / "tokenizer").mkdir()
        (tmp_path / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    return hashlib.sha256(b"weights").hexdigest()


@pytest.mark.parametrize(
    ("setup", "code"),
    [
        ("missing", "judge_not_installed"),
        ("no_tokenizer", "judge_tokenizer_missing"),
        ("swapped", "judge_digest_mismatch"),
    ],
)
def test_the_judge_never_loads_what_the_contract_did_not_pin(
    tmp_path: Path, setup: str, code: str
) -> None:
    digest = "0" * 64
    if setup == "no_tokenizer":
        digest = _checkpoint(tmp_path, tokenizer=False)
    elif setup == "swapped":
        _checkpoint(tmp_path)
    judge = LayaAnswerJudge(tmp_path, weights_sha256=digest)

    with pytest.raises(AnswerJudgeUnavailableError) as raised:
        asyncio.run(judge.supports("q", ["p"]))

    assert raised.value.code == code
