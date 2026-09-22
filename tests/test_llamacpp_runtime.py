import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from harness import ModelMessage, ModelRequest, ModelRole, ToolCall, ToolSchema
from harness.llamacpp_runtime import LlamaCppRuntime, LlamaCppRuntimeError
from harness.ports import MalformedModelResponseError, ModelGenerationTimeout

_DIGEST = "a" * 64


def _runtime(handler: httpx.MockTransport, *, digest: str = _DIGEST) -> LlamaCppRuntime:
    return LlamaCppRuntime(
        base_url="http://llama",
        model="cove-4b",
        expected_digest=digest,
        context_window=65536,
        transport=handler,
    )


def _request(*, think: bool = True) -> ModelRequest:
    return ModelRequest(
        messages=(
            ModelMessage(role=ModelRole.SYSTEM, content="sys"),
            ModelMessage(role=ModelRole.USER, content="leia o README"),
            ModelMessage(
                role=ModelRole.ASSISTANT,
                content="",
                tool_calls=(ToolCall(id="c1", name="read_file", arguments={"file_path": "a"}),),
            ),
            ModelMessage(role=ModelRole.TOOL, content="{}", tool_call_id="c1", name="read_file"),
        ),
        tools=(ToolSchema(name="read_file", description="d", parameters={"type": "object"}),),
        options={"temperature": 0.3, "presence_penalty": 0},
        seed=7,
        max_output_tokens=128,
        think=think,
    )


def test_the_request_speaks_openai_and_the_answer_comes_back_split() -> None:
    sent: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "vou ler",
                            "tool_calls": [
                                {
                                    "id": "x9",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"file_path": "README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 12},
                "timings": {"prompt_ms": 10.5, "predicted_ms": 200.0},
            },
        )

    async def scenario() -> None:
        async with _runtime(httpx.MockTransport(handler)) as runtime:
            response = await runtime.generate(_request(think=False))
        assert response.content is None
        assert response.reasoning == "vou ler"
        assert response.tool_calls == (
            ToolCall(id="x9", name="read_file", arguments={"file_path": "README.md"}),
        )
        assert response.usage is not None and response.usage.input_tokens == 40
        assert response.durations is not None and response.durations.eval_ns == 200_000_000

    asyncio.run(scenario())

    body = sent[0]
    assert body["seed"] == 7 and body["max_tokens"] == 128 and body["stream"] is False
    assert body["temperature"] == 0.3
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    messages = body["messages"]
    assert isinstance(messages, list)
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == '{"file_path": "a"}'
    assert messages[3] == {
        "role": "tool",
        "content": "{}",
        "tool_call_id": "c1",
        "name": "read_file",
    }


def test_arguments_that_are_not_json_are_a_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        call = {"function": {"name": "read_file", "arguments": "{file_path: README"}}
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [call]}}]})

    async def scenario() -> None:
        async with _runtime(httpx.MockTransport(handler)) as runtime:
            with pytest.raises(MalformedModelResponseError):
                await runtime.generate(_request())

    asyncio.run(scenario())


def test_a_call_without_id_gets_one() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        call = {"function": {"name": "glob_search", "arguments": {"pattern": "*.md"}}}
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [call]}}]})

    async def scenario() -> None:
        async with _runtime(httpx.MockTransport(handler)) as runtime:
            response = await runtime.generate(_request())
        assert response.tool_calls[0].id.startswith("llamacpp-")
        assert response.tool_calls[0].arguments == {"pattern": "*.md"}

    asyncio.run(scenario())


def test_http_errors_and_timeouts_keep_their_class() -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503, json={"error": {"message": "Loading model"}})

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async def scenario() -> None:
        async with _runtime(httpx.MockTransport(refused)) as runtime:
            with pytest.raises(LlamaCppRuntimeError) as raised:
                await runtime.generate(_request())
        assert raised.value.retryable
        assert raised.value.error is not None
        assert raised.value.error["message"] == "Loading model"
        async with _runtime(httpx.MockTransport(slow)) as runtime:
            with pytest.raises(ModelGenerationTimeout):
                await runtime.generate(_request())

    asyncio.run(scenario())


def test_the_profile_is_the_file_the_server_loaded(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF weights")
    digest = hashlib.sha256(b"GGUF weights").hexdigest()

    def server(n_ctx: int) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/props"
            return httpx.Response(
                200,
                json={
                    "model_path": str(gguf),
                    "default_generation_settings": {"n_ctx": n_ctx},
                },
            )

        return httpx.MockTransport(handler)

    async def scenario() -> None:
        async with _runtime(server(65536), digest=digest) as runtime:
            assert (await runtime.verify_profile()).ready
        async with _runtime(server(65536)) as runtime:
            other = await runtime.verify_profile()
        assert other.reason_code == "model_digest_mismatch"
        assert other.observed_digest == digest
        async with _runtime(server(8192), digest=digest) as runtime:
            short = await runtime.verify_profile()
        assert short.reason_code == "context_window_mismatch"

    asyncio.run(scenario())
