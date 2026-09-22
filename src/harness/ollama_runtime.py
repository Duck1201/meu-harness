from collections.abc import Mapping, Sequence
from typing import Literal, Self, cast
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


class OllamaWireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class OllamaFunction(OllamaWireModel):
    name: str = Field(min_length=1)
    arguments: Mapping[str, JsonValue] | None = None
    description: str | None = None
    parameters: Mapping[str, JsonValue] | None = None


class OllamaToolCall(OllamaWireModel):
    function: OllamaFunction


class OllamaTool(OllamaWireModel):
    type: Literal["function"] = "function"
    function: OllamaFunction


class OllamaMessage(OllamaWireModel):
    role: ModelRole
    content: str
    thinking: str | None = None
    tool_calls: tuple[OllamaToolCall, ...] = ()
    tool_name: str | None = None


class OllamaChatRequest(OllamaWireModel):
    model: str
    messages: tuple[OllamaMessage, ...] = Field(min_length=1)
    tools: tuple[OllamaTool, ...] = ()
    stream: Literal[False] = False
    think: bool = True
    options: Mapping[str, JsonValue]


class OllamaChatResponse(OllamaWireModel):
    model: str
    message: OllamaMessage
    done: bool
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class OllamaEmbedRequest(OllamaWireModel):
    model: str
    input: tuple[str, ...] = Field(min_length=1)
    truncate: bool = True


class OllamaEmbedResponse(OllamaWireModel):
    model: str
    embeddings: tuple[tuple[float, ...], ...] = Field(min_length=1)


class OllamaTag(OllamaWireModel):
    name: str
    model: str | None = None
    digest: str


class OllamaTagsResponse(OllamaWireModel):
    models: tuple[OllamaTag, ...]


class OllamaRuntimeError(ModelRuntimeError):
    """A provider failure with Ollama's own class attached.

    The fields live on ModelRuntimeError so the engine can end a Turn on them
    without importing this module or knowing which runtime is behind the port.
    """


class OllamaProfileVerification(BaseModel):
    model_config = ConfigDict(frozen=True)

    ready: bool
    model: str
    expected_digest: str
    observed_digest: str | None = None
    reason_code: str | None = None


class OllamaRuntime:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        expected_digest: str,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._expected_digest = _normalize_digest(expected_digest)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        wire_request = _chat_request(self._model, request)
        response = await self._request(
            "POST",
            "/api/chat",
            json=wire_request.model_dump(mode="json", exclude_none=True),
        )
        try:
            wire_response = OllamaChatResponse.model_validate_json(response.content)
        except ValidationError as error:
            raise MalformedModelResponseError(
                "Ollama returned an invalid chat response",
                raw=_safe_response_payload(response),
            ) from error

        tool_calls = tuple(
            ToolCall(
                id=f"ollama-{uuid4()}",
                name=call.function.name,
                arguments=call.function.arguments or {},
            )
            for call in wire_response.message.tool_calls
        )
        content = wire_response.message.content or None
        if content is None and not tool_calls:
            raise MalformedModelResponseError(
                "Ollama response has neither content nor tool calls",
                raw=_safe_response_payload(response),
            )
        return ModelResponse(
            content=content,
            reasoning=wire_response.message.thinking,
            tool_calls=tool_calls,
            usage=_usage(wire_response),
            durations=_durations(wire_response),
        )

    async def verify_profile(self) -> OllamaProfileVerification:
        return await _verify_tag(self._client, self._model, self._expected_digest)

    async def health(self) -> OllamaProfileVerification:
        return await self.verify_profile()

    async def loaded_models(self) -> tuple[str, ...]:
        """Os modelos que o Ollama mantém carregados agora, pelo /api/ps."""
        response = await self._request("GET", "/api/ps")
        try:
            loaded = OllamaTagsResponse.model_validate_json(response.content)
        except ValidationError as error:
            raise MalformedModelResponseError("Ollama returned invalid ps data") from error
        return tuple(item.name for item in loaded.models)

    async def unload(self, model: str | None = None) -> None:
        """Tira um modelo da VRAM já, em vez de esperar o keep_alive vencer.

        A bancada troca de modelo entre braços numa placa de 8 GB; dois modelos
        de chat não cabem juntos, e o segundo carregaria parcialmente em CPU e
        seria medido mais lento do que é.
        """
        await self._request(
            "POST",
            "/api/generate",
            json={"model": model or self._model, "keep_alive": 0},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
    ) -> httpx.Response:
        return await _request(self._client, method, url, json=json)


class OllamaEmbeddingRuntime:
    """The embedding half of the same installation.

    It is its own model with its own digest, so it verifies itself the way the
    chat model does: an installation that answers with another build is not the
    one the RuntimeProfile declared, and every vector already indexed was
    produced by that declared one.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        expected_digest: str,
        dimensions: int,
        timeout: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._expected_digest = _normalize_digest(expected_digest)
        self._dimensions = dimensions
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        request = OllamaEmbedRequest(model=self._model, input=tuple(texts))
        response = await _request(
            self._client,
            "POST",
            "/api/embed",
            json=request.model_dump(mode="json"),
        )
        try:
            wire = OllamaEmbedResponse.model_validate_json(response.content)
        except ValidationError as error:
            raise MalformedModelResponseError(
                "Ollama returned an invalid embedding response",
                raw=_safe_response_payload(response),
            ) from error
        if len(wire.embeddings) != len(texts):
            raise MalformedModelResponseError(
                "Ollama returned a different number of embeddings than inputs"
            )
        # A vector of another width would be indexed happily by nobody: vec0 fixes
        # the column width at creation, and a silent mismatch here would only show
        # up as an insert failure with no idea which model produced it.
        for vector in wire.embeddings:
            if len(vector) != self._dimensions:
                raise MalformedModelResponseError(
                    f"embedding model returned {len(vector)} dimensions, "
                    f"expected {self._dimensions}"
                )
        return wire.embeddings

    async def verify_profile(self) -> OllamaProfileVerification:
        return await _verify_tag(self._client, self._model, self._expected_digest)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()


async def _request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json: Mapping[str, object] | None = None,
) -> httpx.Response:
    try:
        response = await client.request(method, url, json=json)
    except httpx.TimeoutException as error:
        # Quem desligou foi este cliente, com o Ollama ainda gerando do outro
        # lado. Tratar isso como transporte quebrado fazia o engine dizer que o
        # provedor estava fora do ar justamente quando ele estava trabalhando.
        raise ModelGenerationTimeout(
            str(error),
            error={"code": "ollama_generation_timeout", "message": str(error)},
        ) from error
    except httpx.HTTPError as error:
        payload: Mapping[str, JsonValue] = {
            "code": "ollama_transport_error",
            "message": str(error),
        }
        raise OllamaRuntimeError(
            str(error),
            error=payload,
            retryable=True,
        ) from error
    if response.is_success:
        return response
    payload = _http_error_payload(response)
    raise OllamaRuntimeError(
        str(payload["message"]),
        error=payload,
        retryable=response.status_code in {408, 429} or response.status_code >= 500,
        status_code=response.status_code,
    )


async def _verify_tag(
    client: httpx.AsyncClient,
    model: str,
    expected_digest: str,
) -> OllamaProfileVerification:
    response = await _request(client, "GET", "/api/tags")
    try:
        tags = OllamaTagsResponse.model_validate_json(response.content)
    except ValidationError as error:
        raise MalformedModelResponseError("Ollama returned invalid tags data") from error
    tag = next((item for item in tags.models if model in {item.name, item.model}), None)
    if tag is None:
        return OllamaProfileVerification(
            ready=False,
            model=model,
            expected_digest=expected_digest,
            reason_code="model_not_installed",
        )
    observed = _normalize_digest(tag.digest)
    if observed != expected_digest:
        return OllamaProfileVerification(
            ready=False,
            model=model,
            expected_digest=expected_digest,
            observed_digest=observed,
            reason_code="model_digest_mismatch",
        )
    return OllamaProfileVerification(
        ready=True,
        model=model,
        expected_digest=expected_digest,
        observed_digest=observed,
    )


def _chat_request(model: str, request: ModelRequest) -> OllamaChatRequest:
    options = dict(request.options)
    options["seed"] = request.seed
    options["num_predict"] = request.max_output_tokens
    return OllamaChatRequest(
        model=model,
        messages=tuple(_message(message) for message in request.messages),
        tools=tuple(_tool(tool) for tool in request.tools),
        stream=False,
        think=request.think,
        options=options,
    )


def _message(message: ModelMessage) -> OllamaMessage:
    calls = tuple(
        OllamaToolCall(function=OllamaFunction(name=call.name, arguments=call.arguments))
        for call in message.tool_calls
    )
    return OllamaMessage(
        role=message.role,
        content=message.content,
        tool_calls=calls,
        tool_name=message.name if message.role is ModelRole.TOOL else None,
    )


def _tool(tool: ToolSchema) -> OllamaTool:
    return OllamaTool(
        function=OllamaFunction(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
        )
    )


def _usage(response: OllamaChatResponse) -> ModelUsage | None:
    if response.prompt_eval_count is None and response.eval_count is None:
        return None
    return ModelUsage(
        input_tokens=response.prompt_eval_count or 0,
        output_tokens=response.eval_count or 0,
    )


def _durations(response: OllamaChatResponse) -> ModelDurations | None:
    values = (
        response.total_duration,
        response.load_duration,
        response.prompt_eval_duration,
        response.eval_duration,
    )
    if all(value is None for value in values):
        return None
    return ModelDurations(
        total_ns=response.total_duration,
        load_ns=response.load_duration,
        prompt_eval_ns=response.prompt_eval_duration,
        eval_ns=response.eval_duration,
    )


def _http_error_payload(response: httpx.Response) -> Mapping[str, JsonValue]:
    # Not every failure answers in Ollama's dialect: a proxy in the way returns
    # HTML with its 502. Reading that body has to fail into the status we already
    # have, or the ValueError escapes the raise it was building and the Operator
    # is told the harness crashed instead of the provider refusing.
    decoded: Mapping[str, JsonValue]
    try:
        decoded = _safe_json_object(response)
    except ValueError:
        decoded = {}
    message = decoded.get("error")
    return {
        "code": "ollama_http_error",
        "message": message if isinstance(message, str) else response.reason_phrase,
        "status_code": response.status_code,
    }


def _safe_response_payload(response: httpx.Response) -> Mapping[str, JsonValue] | None:
    try:
        return _remove_reasoning(_safe_json_object(response))
    except ValueError:
        return None


def _safe_json_object(response: httpx.Response) -> Mapping[str, JsonValue]:
    decoded = cast(object, response.json())
    if not isinstance(decoded, dict):
        raise ValueError("Ollama response is not a JSON object")
    return cast(dict[str, JsonValue], decoded)


def _remove_reasoning(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {
        key: _remove_reasoning_value(child)
        for key, child in value.items()
        if key.lower() not in {"reasoning", "thinking"}
    }


def _remove_reasoning_value(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return _remove_reasoning(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_remove_reasoning_value(child) for child in value]
    return value


def _normalize_digest(value: str) -> str:
    return value.removeprefix("sha256:").lower()
