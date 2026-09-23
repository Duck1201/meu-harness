"""O modelo de visão local por trás de `describe_image` (ADR 0016).

Um segundo modelo no mesmo Ollama, com digest próprio, que responde uma pergunta
sobre uma imagem. O modelo de chat nunca recebe os bytes: ele pede, este runtime
olha, e só texto volta ao ModelView.

A escolha foi medida numa bancada de nove imagens com gabarito exato — cupom,
tabela com `24.7` ao lado de `247`, terminal, código, diálogo, gráfico, texto
miúdo, uma senha ilegível e um preço que não existe. Por este runtime, a
temperatura 0, o Qwen3.5-2B faz 9/9 (`scripts/vision-bench.py`), e nas duas
armadilhas diz que não dá para ler, onde o mitos e o MiniCPM inventaram. O modelo
de chat ativo também enxerga, mas embaralha dígitos pequenos, e esse é o erro que
o gate de visão existe para impedir.
"""

import asyncio
import base64
import time

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from .ollama_runtime import OllamaRuntime
from .ports import ModelRuntimeError, VisionAnswer

# O mesmo texto da bancada que escolheu o modelo: mudar a instrução é mudar o
# que foi medido. Em português porque a resposta volta para um Operator em
# pt-BR e o modelo de visão responde na língua em que é instruído.
VISION_SYSTEM_PROMPT = (
    "Você descreve imagens para um agente. Responda em português do Brasil, de forma curta. "
    "Transcreva números e códigos exatamente como aparecem. Se não conseguir ler algo com "
    "certeza, ou se a informação não estiver na imagem, diga isso claramente em vez de adivinhar."
)
_SEED = 0
# Leitura é transcrição, não criação: amostrar só troca dígito. Medido em dez seeds
# (2026-09-23): a 0,2 o modelo inventou a senha ilegível 3 vezes e leu "118" como
# "18" em 5; a 0 foi honesto 10/10 e leu "118" 10/10.
_TEMPERATURE = 0.0


class VisionRuntimeError(ModelRuntimeError):
    """O modelo de visão recusou, caiu ou não é o que o contrato fixa."""


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _Message(_Wire):
    content: str = ""


class _ChatResponse(_Wire):
    message: _Message
    eval_count: int | None = None


class OllamaVisionRuntime:
    """Implementa `VisionRuntime` sobre `/api/chat` com o campo `images` do Ollama."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        expected_digest: str,
        max_output_tokens: int,
        context_tokens: int,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._context_tokens = context_tokens
        self._verifier = OllamaRuntime(
            base_url=base_url, model=model, expected_digest=expected_digest, transport=transport
        )
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )
        self._verified = False
        self._lock = asyncio.Lock()

    async def describe(self, image: bytes, question: str) -> VisionAnswer:
        await self._verify()
        started = time.monotonic()
        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "think": False,
                    "messages": [
                        {"role": "system", "content": VISION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": question,
                            "images": [base64.b64encode(image).decode("ascii")],
                        },
                    ],
                    "options": {
                        "temperature": _TEMPERATURE,
                        "seed": _SEED,
                        "num_predict": self._max_output_tokens,
                        "num_ctx": self._context_tokens,
                    },
                },
            )
        except httpx.HTTPError as error:
            # Só a classe da falha: a mensagem do httpx pode trazer o corpo, e o
            # corpo é a imagem.
            raise VisionRuntimeError(
                "vision model unreachable",
                error={"code": "vision_unavailable", "message": type(error).__name__},
                retryable=True,
            ) from error
        if not response.is_success:
            raise VisionRuntimeError(
                "vision model refused",
                error={"code": "vision_http_error", "message": str(response.status_code)},
                retryable=response.status_code >= 500,
                status_code=response.status_code,
            )
        try:
            wire = _ChatResponse.model_validate_json(response.content)
        except ValidationError as error:
            raise VisionRuntimeError(
                "vision model answered outside the format",
                error={"code": "vision_invalid_answer", "message": "invalid chat response"},
            ) from error
        return VisionAnswer(
            text=wire.message.content.strip(),
            latency_ms=round((time.monotonic() - started) * 1000, 1),
            output_tokens=wire.eval_count,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._verifier.aclose()

    async def _verify(self) -> None:
        async with self._lock:
            if self._verified:
                return
            verification = await self._verifier.verify_profile()
            if not verification.ready:
                raise VisionRuntimeError(
                    f"vision model {self._model} is not ready",
                    error={
                        "code": verification.reason_code or "vision_not_ready",
                        "message": self._model,
                    },
                )
            self._verified = True


__all__ = ["VISION_SYSTEM_PROMPT", "OllamaVisionRuntime", "VisionRuntimeError"]
