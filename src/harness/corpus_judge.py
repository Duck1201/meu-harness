"""Juiz local "esta passagem traz o que foi perguntado?" sobre o Qwen3-Reranker (ADR 0015).

O Qwen3-Reranker é um modelo de 0,6B treinado para responder "yes" ou "no" a um
par consulta/documento sob uma instrução escrita. A probabilidade de "yes" contra
"no" no primeiro token gerado é o `noul` que um juiz no molde do Jev devolveria:
número calibrado, texto nenhum. A instrução é o que o faz servir aqui — ele julga
se o documento afirma o fato pedido, não só se trata do mesmo assunto, e é esse o
caso em que um reranker de relevância pura erra.

Roda no mesmo Ollama que já serve o chat e o `bge-m3`, em Q8, pela rota crua de
`/api/generate` com logprobs: um token por passagem, sem template, sem raciocínio.
Em CPU, pelo transformers, o mesmo modelo custava 13 s por Turn de seis passagens
de 520 tokens; na placa custa 1,2 s e 867 MB de VRAM. A tag é conferida pelo
digest antes do primeiro julgamento, como a do modelo de chat.
"""

import asyncio
import math
from collections.abc import Mapping, Sequence

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from .ollama_runtime import OllamaRuntime

# Medida em 60 pares rotulados em pt-BR, metade quase-acertos (2026-09-22): com
# esta instrução, AUC 0,948 e 85% de acerto no limiar 0,5 pelo Ollama em Q8; com a
# instrução de fábrica ("Given a web search query..."), 68%. O que ela nomeia é o
# quase-acerto.
INSTRUCTION = (
    "Dada uma pergunta, julgue se o documento afirma explicitamente o fato específico "
    "que a pergunta pede. Documento do mesmo assunto sem esse fato não conta."
)
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_TOP_LOGPROBS = 20
# Uma passagem injetada tem no máximo 512 tokens; a janela cobre pergunta,
# instrução e formato com folga, e fica pequena para caber ao lado do chat.
_CONTEXT_TOKENS = 2048


def judge_prompt(question: str, passage: str) -> str:
    """O prompt inteiro no formato do Qwen3-Reranker, com a instrução do harness."""
    body = f"<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: {passage}"
    return f"{_PREFIX}{body}{_SUFFIX}"


def yes_probability(top_logprobs: Mapping[str, float]) -> float:
    """P("yes") contra P("no"), só entre os dois — o resto da distribuição se cancela.

    Um dos dois pode ficar fora dos 20 mais prováveis; aí ele vale no máximo o
    menor logprob listado, e usar esse teto é a estimativa conservadora.
    """
    if not top_logprobs:
        raise AnswerJudgeUnavailableError("judge_invalid_answer", "O juiz não devolveu logprobs.")
    floor = min(top_logprobs.values())
    yes = top_logprobs.get("yes", floor)
    no = top_logprobs.get("no", floor)
    return 1.0 / (1.0 + math.exp(no - yes))


class AnswerJudgeUnavailableError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _TopLogprob(_Wire):
    token: str
    logprob: float


class _TokenLogprob(_Wire):
    top_logprobs: tuple[_TopLogprob, ...] = ()


class _GenerateResponse(_Wire):
    logprobs: tuple[_TokenLogprob, ...] = ()


class OllamaRerankerAnswerJudge:
    """Implementa `CorpusAnswerJudge` com o Qwen3-Reranker servido pelo Ollama."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        expected_digest: str,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._verifier = OllamaRuntime(
            base_url=base_url, model=model, expected_digest=expected_digest, transport=transport
        )
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )
        self._verified = False
        self._lock = asyncio.Lock()

    async def supports(self, question: str, passages: Sequence[str]) -> tuple[float, ...]:
        await self._verify()
        scores: list[float] = []
        for passage in passages:
            scores.append(await self._score(question, passage))
        return tuple(scores)

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._verifier.aclose()

    async def _verify(self) -> None:
        async with self._lock:
            if self._verified:
                return
            verification = await self._verifier.verify_profile()
            if not verification.ready:
                raise AnswerJudgeUnavailableError(
                    verification.reason_code or "judge_not_ready",
                    f"O juiz {self._model} não está pronto no Ollama.",
                )
            self._verified = True

    async def _score(self, question: str, passage: str) -> float:
        response = await self._client.post(
            "/api/generate",
            json={
                "model": self._model,
                "prompt": judge_prompt(question, passage),
                "raw": True,
                "stream": False,
                "logprobs": True,
                "top_logprobs": _TOP_LOGPROBS,
                "options": {"num_predict": 1, "temperature": 0, "num_ctx": _CONTEXT_TOKENS},
            },
        )
        if not response.is_success:
            raise AnswerJudgeUnavailableError(
                "judge_http_error", f"O Ollama respondeu {response.status_code} ao juiz."
            )
        try:
            wire = _GenerateResponse.model_validate_json(response.content)
        except ValidationError as error:
            raise AnswerJudgeUnavailableError(
                "judge_invalid_answer", "Resposta do juiz fora do formato."
            ) from error
        if not wire.logprobs:
            raise AnswerJudgeUnavailableError(
                "judge_invalid_answer", "O juiz não devolveu logprobs."
            )
        first = wire.logprobs[0]
        return yes_probability({item.token: item.logprob for item in first.top_logprobs})


__all__ = [
    "INSTRUCTION",
    "AnswerJudgeUnavailableError",
    "OllamaRerankerAnswerJudge",
    "judge_prompt",
    "yes_probability",
]
