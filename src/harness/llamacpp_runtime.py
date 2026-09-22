"""ModelRuntime sobre o `llama-server` do llama.cpp (ADR 0013).

O Ollama não dá conta de dois dos candidatos do bake-off: GGUF puxado do Hugging
Face não ganha o renderer/parser da família, e o tool calling desses perfis
depende do template jinja embutido no arquivo. O `llama-server` aplica esse
template e devolve as tool calls já separadas na API compatível com OpenAI, que
é o que este adaptador fala.

A identidade do modelo aqui é o arquivo: o servidor carrega um GGUF, não uma tag,
então o digest que prova o perfil é o SHA-256 do arquivo que `/props` diz estar
carregado.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Self, cast
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .domain import JsonValue, ToolCall
from .ports import (
    MalformedModelResponseError,
    ModelDurations,
    ModelGenerationTimeout,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelRuntimeError,
    ModelUsage,
    ToolSchema,
)


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _WireFunction(_Wire):
    name: str
    arguments: str | Mapping[str, JsonValue] | None = None


class _WireToolCall(_Wire):
    id: str | None = None
    function: _WireFunction


class _WireMessage(_Wire):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[_WireToolCall, ...] | None = None


class _WireChoice(_Wire):
    message: _WireMessage


class _WireUsage(_Wire):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _WireTimings(_Wire):
    prompt_ms: float | None = None
    predicted_ms: float | None = None


class _WireChatResponse(_Wire):
    choices: tuple[_WireChoice, ...] = Field(min_length=1)
    usage: _WireUsage | None = None
    timings: _WireTimings | None = None


class _WireGenerationSettings(_Wire):
    n_ctx: int | None = None


class _WireProps(_Wire):
    model_path: str | None = None
    default_generation_settings: _WireGenerationSettings | None = None


class LlamaCppRuntimeError(ModelRuntimeError):
    """Falha do provedor com a classe do llama-server anexada."""


class LlamaCppProfileVerification(BaseModel):
    model_config = ConfigDict(frozen=True)

    ready: bool
    model: str
    expected_digest: str
    observed_digest: str | None = None
    reason_code: str | None = None


@lru_cache(maxsize=16)
def _file_digest(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns  # parte da chave do cache: arquivo trocado é arquivo novo
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


async def gguf_digest(path: Path) -> str:
    """SHA-256 de um GGUF, calculado fora do loop e lembrado enquanto o arquivo não mudar."""
    stat = path.stat()
    return await asyncio.to_thread(_file_digest, str(path), stat.st_size, stat.st_mtime_ns)


class LlamaCppRuntime:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        expected_digest: str,
        context_window: int,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._expected_digest = expected_digest.removeprefix("sha256:").lower()
        self._context_window = context_window
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        response = await self._request(
            "POST", "/v1/chat/completions", json=_chat_request(self._model, request)
        )
        try:
            wire = _WireChatResponse.model_validate_json(response.content)
        except ValidationError as error:
            raise MalformedModelResponseError(
                "llama-server returned an invalid chat response",
                raw=_safe_payload(response),
            ) from error
        message = wire.choices[0].message
        tool_calls = tuple(_tool_call(call, response) for call in message.tool_calls or ())
        content = message.content or None
        if content is None and not tool_calls:
            raise MalformedModelResponseError(
                "llama-server response has neither content nor tool calls",
                raw=_safe_payload(response),
            )
        return ModelResponse(
            content=content,
            reasoning=message.reasoning_content or None,
            tool_calls=tool_calls,
            usage=_usage(wire),
            durations=_durations(wire),
        )

    async def verify_profile(self) -> LlamaCppProfileVerification:
        response = await self._request("GET", "/props")
        try:
            props = _WireProps.model_validate_json(response.content)
        except ValidationError as error:
            raise MalformedModelResponseError("llama-server returned invalid props") from error
        if props.model_path is None or not Path(props.model_path).is_file():
            return self._verification(False, reason_code="model_not_installed")
        observed = await gguf_digest(Path(props.model_path))
        if observed != self._expected_digest:
            return self._verification(False, observed, "model_digest_mismatch")
        # A janela é do servidor, não do pedido: um llama-server subido com outro
        # -c corta o ModelView que o orçamento do contrato achou que cabia.
        settings = props.default_generation_settings
        if settings is None or settings.n_ctx != self._context_window:
            return self._verification(False, observed, "context_window_mismatch")
        return self._verification(True, observed)

    async def health(self) -> LlamaCppProfileVerification:
        return await self.verify_profile()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    def _verification(
        self, ready: bool, observed: str | None = None, reason_code: str | None = None
    ) -> LlamaCppProfileVerification:
        return LlamaCppProfileVerification(
            ready=ready,
            model=self._model,
            expected_digest=self._expected_digest,
            observed_digest=observed,
            reason_code=reason_code,
        )

    async def _request(
        self, method: str, url: str, *, json: Mapping[str, object] | None = None
    ) -> httpx.Response:
        try:
            response = await self._client.request(method, url, json=json)
        except httpx.TimeoutException as error:
            raise ModelGenerationTimeout(
                str(error),
                error={"code": "llama_cpp_generation_timeout", "message": str(error)},
            ) from error
        except httpx.HTTPError as error:
            raise LlamaCppRuntimeError(
                str(error),
                error={"code": "llama_cpp_transport_error", "message": str(error)},
                retryable=True,
            ) from error
        if response.is_success:
            return response
        payload = _http_error_payload(response)
        raise LlamaCppRuntimeError(
            str(payload["message"]),
            error=payload,
            retryable=response.status_code in {408, 429, 503} or response.status_code >= 500,
            status_code=response.status_code,
        )


def _chat_request(model: str, request: ModelRequest) -> Mapping[str, object]:
    options = dict(request.options)
    body: dict[str, object] = {
        **options,
        "model": model,
        "messages": [_message(message) for message in request.messages],
        "stream": False,
        "seed": request.seed,
        "max_tokens": request.max_output_tokens,
        # Liga e desliga o raciocínio pelo template, que é quem sabe fazê-lo; o
        # texto pensado volta separado em reasoning_content e nunca no corpo.
        "chat_template_kwargs": {"enable_thinking": request.think},
    }
    if request.tools:
        body["tools"] = [_tool(tool) for tool in request.tools]
    return body


def _message(message: ModelMessage) -> Mapping[str, object]:
    wire: dict[str, object] = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    if message.role is ModelRole.TOOL:
        if message.tool_call_id is not None:
            wire["tool_call_id"] = message.tool_call_id
        if message.name is not None:
            wire["name"] = message.name
    return wire


def _tool(tool: ToolSchema) -> Mapping[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _tool_call(call: _WireToolCall, response: httpx.Response) -> ToolCall:
    raw = call.function.arguments
    if raw is None or raw == "":
        arguments: Mapping[str, JsonValue] = {}
    elif isinstance(raw, str):
        try:
            decoded = cast(object, json.loads(raw))
        except ValueError as error:
            raise MalformedModelResponseError(
                "llama-server tool call arguments are not JSON", raw=_safe_payload(response)
            ) from error
        if not isinstance(decoded, dict):
            raise MalformedModelResponseError(
                "llama-server tool call arguments are not an object", raw=_safe_payload(response)
            )
        arguments = cast(dict[str, JsonValue], decoded)
    else:
        arguments = raw
    return ToolCall(
        id=call.id or f"llamacpp-{uuid4()}",
        name=call.function.name,
        arguments=arguments,
    )


def _usage(response: _WireChatResponse) -> ModelUsage | None:
    usage = response.usage
    if usage is None or (usage.prompt_tokens is None and usage.completion_tokens is None):
        return None
    return ModelUsage(
        input_tokens=usage.prompt_tokens or 0,
        output_tokens=usage.completion_tokens or 0,
    )


def _durations(response: _WireChatResponse) -> ModelDurations | None:
    timings = response.timings
    if timings is None or (timings.prompt_ms is None and timings.predicted_ms is None):
        return None
    prompt_ns = _ns(timings.prompt_ms)
    eval_ns = _ns(timings.predicted_ms)
    return ModelDurations(
        total_ns=(prompt_ns or 0) + (eval_ns or 0),
        prompt_eval_ns=prompt_ns,
        eval_ns=eval_ns,
    )


def _ns(milliseconds: float | None) -> int | None:
    return None if milliseconds is None else round(milliseconds * 1_000_000)


def _http_error_payload(response: httpx.Response) -> Mapping[str, JsonValue]:
    message: str = response.reason_phrase
    try:
        decoded = cast(object, response.json())
    except ValueError:
        decoded = None
    if isinstance(decoded, dict):
        error = cast(dict[str, object], decoded).get("error")
        if isinstance(error, dict):
            text = cast(dict[str, object], error).get("message")
            if isinstance(text, str):
                message = text
        elif isinstance(error, str):
            message = error
    return {
        "code": "llama_cpp_http_error",
        "message": message,
        "status_code": response.status_code,
    }


def _safe_payload(response: httpx.Response) -> Mapping[str, JsonValue] | None:
    try:
        decoded = cast(object, response.json())
    except ValueError:
        return None
    if not isinstance(decoded, dict):
        return None
    return cast(Mapping[str, JsonValue], _without_reasoning(cast(JsonValue, decoded)))


def _without_reasoning(value: JsonValue) -> JsonValue:
    # Reasoning é transitório por contrato: nem o payload bruto de uma resposta
    # rejeitada o leva para o CanonicalHistory.
    if isinstance(value, Mapping):
        return {
            key: _without_reasoning(child)
            for key, child in value.items()
            if key.lower() not in {"reasoning", "reasoning_content", "thinking"}
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_without_reasoning(child) for child in value]
    return value


__all__ = [
    "LlamaCppProfileVerification",
    "LlamaCppRuntime",
    "LlamaCppRuntimeError",
    "gguf_digest",
]
