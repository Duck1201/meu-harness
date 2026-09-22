"""Juiz local "esta passagem traz o que foi perguntado?" sobre o Laya (ADR 0015).

O Laya é um modelo System One no molde do Jev: recebe estado e uma pergunta
tipada e devolve probabilidade calibrada, sem gerar texto. Aqui ele responde um
`noul` por passagem recuperada — uma de cada vez, porque a janela do checkpoint
multilíngue é de 1024 tokens e seis passagens não cabem juntas.

O modelo roda em CPU para não disputar a VRAM com o modelo de chat, e só de um
diretório local fixado por revisão: o pacote baixa do Hugging Face quando não
acha o arquivo, e um juiz que sai para a rede no meio do Turn é egress que
ninguém autorizou.
"""

import asyncio
import hashlib
import importlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

# Formulação medida em 24 pares rotulados em pt-BR (2026-09-22): a mesma
# pergunta em inglês ficou em 62,5% de acerto, esta em 79%. O quase-acerto — mesmo
# assunto, dado ausente — é o caso difícil, e o critério "false" o nomeia.
_QUESTION: Mapping[str, object] = {
    "type": "noul",
    "instructions": (
        "A `passagem` contém a resposta exata para a `pergunta`? "
        "Mesmo assunto sem o dado pedido não conta."
    ),
    "criteria": {
        "true": "a passagem traz o dado pedido",
        "false": "o dado pedido não está na passagem",
    },
}


class _Agent(Protocol):
    def system_one(
        self, state: Mapping[str, str], questions: Mapping[str, Mapping[str, object]]
    ) -> Mapping[str, object]: ...


class AnswerJudgeUnavailableError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class LayaAnswerJudge:
    """Implementa `CorpusAnswerJudge` com o checkpoint Laya de um diretório local."""

    def __init__(self, model_dir: Path, *, weights_sha256: str, agent: _Agent | None = None):
        self._model_dir = model_dir
        self._weights_sha256 = weights_sha256.removeprefix("sha256:").lower()
        self._agent = agent
        self._lock = asyncio.Lock()

    async def supports(self, question: str, passages: Sequence[str]) -> tuple[float, ...]:
        agent = await self._loaded()
        return await asyncio.to_thread(self._judge, agent, question, tuple(passages))

    async def _loaded(self) -> _Agent:
        async with self._lock:
            if self._agent is None:
                self._agent = await asyncio.to_thread(self._load)
            return self._agent

    def _load(self) -> _Agent:
        weights = self._model_dir / "model.safetensors"
        if not weights.is_file():
            raise AnswerJudgeUnavailableError("judge_not_installed", f"{weights} não existe.")
        # Sem o tokenizer no diretório, o pacote busca o do encoder no Hub.
        if not (self._model_dir / "tokenizer" / "tokenizer.json").is_file():
            raise AnswerJudgeUnavailableError(
                "judge_tokenizer_missing", "O checkpoint local não traz o tokenizer."
            )
        if _sha256(weights) != self._weights_sha256:
            raise AnswerJudgeUnavailableError(
                "judge_digest_mismatch", "Os pesos do juiz não são os que o contrato fixa."
            )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            laya = importlib.import_module("laya")
        except ImportError as error:
            raise AnswerJudgeUnavailableError(
                "judge_dependency_missing", "O juiz precisa de `uv sync --extra judge`."
            ) from error
        return cast(_Agent, laya.load(str(self._model_dir), device="cpu"))

    @staticmethod
    def _judge(agent: _Agent, question: str, passages: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(
            _noul(agent.system_one({"pergunta": question, "passagem": passage}, {"a": _QUESTION}))
            for passage in passages
        )


def _noul(result: Mapping[str, object]) -> float:
    answers = cast(Mapping[str, Mapping[str, object]], result["answers"])
    value = answers["a"]["noul"]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AnswerJudgeUnavailableError("judge_invalid_answer", f"noul inválido: {value!r}")
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["AnswerJudgeUnavailableError", "LayaAnswerJudge"]
