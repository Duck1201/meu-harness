"""OCR de PDF escaneado para o Corpus (ADR 0018).

Só para o PDF sem camada de texto, que antes era recusado: medido no Kurose
(2026-09-29), refazer por OCR um PDF que já tem camada troca um conjunto de
erros por outro — corrigia "Citan1os" e "Rl." e perdia acentos ("possível" virou
"possivel", o que a perna lexical não casa) — ao custo de 8 a 12 s por página.
Onde não há camada, qualquer leitura é melhor que recusar o arquivo.

A página é renderizada pelo `pdftoppm` (poppler-utils) e lida pelo GLM-OCR no
Ollama, fixado por digest. O texto entra marcado com `OcrTranscribedTaint`: é a
leitura de um modelo, não a camada do arquivo, e a passagem diz isso ao modelo
de chat.
"""

from __future__ import annotations

import asyncio
import base64
import re
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from .ollama_runtime import OllamaRuntime
from .ports import ModelRuntimeError

# O prompt que o GLM-OCR reconhece para transcrição de texto corrido.
OCR_PROMPT = "Text Recognition:"
# Transcrição, não criação: amostrar só troca letra.
_TEMPERATURE = 0.0
_SEED = 0


class OcrError(ModelRuntimeError):
    """O OCR não leu a página: fora do ar, recusou, ou não é o modelo do contrato."""


class PdfPageRenderer(Protocol):
    async def render(self, pdf: Path, page: int) -> bytes: ...


class PageOcr(Protocol):
    async def transcribe(self, image: bytes) -> str: ...


class PdftoppmRenderer:
    """Renderiza uma página em PNG com o `pdftoppm` do host."""

    def __init__(self, executable: str, *, dpi: int, timeout: float = 120.0) -> None:
        self._executable = executable
        self._dpi = dpi
        self._timeout = timeout

    async def render(self, pdf: Path, page: int) -> bytes:
        with tempfile.TemporaryDirectory(prefix="harness-ocr-") as directory:
            target = Path(directory) / "page"
            process = await asyncio.create_subprocess_exec(
                self._executable,
                "-f",
                str(page),
                "-l",
                str(page),
                "-r",
                str(self._dpi),
                "-png",
                "-singlefile",
                str(pdf),
                str(target),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, error = await asyncio.wait_for(process.communicate(), self._timeout)
            except TimeoutError:
                process.kill()
                raise OcrError(
                    "page render timed out",
                    error={"code": "ocr_render_timeout", "message": f"page {page}"},
                ) from None
            rendered = target.with_suffix(".png")
            if process.returncode != 0 or not rendered.is_file():
                raise OcrError(
                    "page render failed",
                    error={
                        "code": "ocr_render_failed",
                        "message": error.decode("utf-8", "replace")[:200],
                    },
                )
            return rendered.read_bytes()


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _Message(_Wire):
    content: str = ""


class _ChatResponse(_Wire):
    message: _Message


class OllamaPageOcr:
    """Implementa `PageOcr` sobre `/api/chat`, com o digest conferido uma vez."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        expected_digest: str,
        max_output_tokens: int,
        context_tokens: int,
        timeout: float = 300.0,
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

    async def transcribe(self, image: bytes) -> str:
        await self._verify()
        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": OCR_PROMPT,
                            "images": [base64.b64encode(image).decode("ascii")],
                        }
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
            raise OcrError(
                "ocr model unreachable",
                error={"code": "ocr_unavailable", "message": type(error).__name__},
                retryable=True,
            ) from error
        if not response.is_success:
            raise OcrError(
                "ocr model refused",
                error={"code": "ocr_http_error", "message": str(response.status_code)},
                retryable=response.status_code >= 500,
                status_code=response.status_code,
            )
        try:
            wire = _ChatResponse.model_validate_json(response.content)
        except ValidationError as error:
            raise OcrError(
                "ocr model answered outside the format",
                error={"code": "ocr_invalid_answer", "message": "invalid chat response"},
            ) from error
        return clean_transcription(wire.message.content)

    async def _verify(self) -> None:
        async with self._lock:
            if self._verified:
                return
            verification = await self._verifier.verify_profile()
            if not verification.ready:
                raise OcrError(
                    f"ocr model {self._model} is not ready",
                    error={
                        "code": verification.reason_code or "ocr_not_ready",
                        "message": self._model,
                    },
                )
            self._verified = True

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._verifier.aclose()


_FENCE = re.compile(r"^```[a-z]*\s*$", re.MULTILINE)
# Menor bloco que conta como repetição: abaixo disso, frase curta que o livro
# repete de verdade ("Figura 2.1") seria cortada junto.
_MIN_REPEAT_CHARS = 120


def clean_transcription(text: str) -> str:
    """Tira as cercas de Markdown e corta a volta que o modelo repete sem parar.

    Medido na capa do Kurose: o GLM-OCR transcreveu a página e recomeçou dentro
    de uma cerca ```markdown, repetindo até esgotar os 4096 tokens. O que vem
    depois da primeira repetição de um bloco longo não é página, é o laço.
    """
    stripped = _FENCE.sub("", text).strip()
    return _without_repetition(stripped)


def _without_repetition(text: str) -> str:
    size = len(text)
    for start in range(size):
        window = text[start : start + _MIN_REPEAT_CHARS]
        if len(window) < _MIN_REPEAT_CHARS:
            break
        again = text.find(window, start + _MIN_REPEAT_CHARS)
        if again >= 0:
            return text[:again].rstrip()
    return text


async def transcribe_pdf(
    data: bytes,
    pages: int,
    *,
    renderer: PdfPageRenderer,
    ocr: PageOcr,
    on_page: Callable[[int], Awaitable[None]] | None = None,
) -> list[str]:
    """O texto de cada página, na ordem, lido pelo OCR uma página por vez."""
    with tempfile.TemporaryDirectory(prefix="harness-ocr-pdf-") as directory:
        pdf = Path(directory) / "source.pdf"
        pdf.write_bytes(data)
        texts: list[str] = []
        for page in range(1, pages + 1):
            texts.append(await ocr.transcribe(await renderer.render(pdf, page)))
            if on_page is not None:
                await on_page(page)
        return texts


__all__ = [
    "OCR_PROMPT",
    "OcrError",
    "OllamaPageOcr",
    "PageOcr",
    "PdfPageRenderer",
    "PdftoppmRenderer",
    "clean_transcription",
    "transcribe_pdf",
]
