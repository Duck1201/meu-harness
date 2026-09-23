"""O runtime de visão fala o dialeto de imagem do Ollama e nunca devolve a imagem."""

import asyncio
import base64
import json
from typing import Any, cast

import httpx
import pytest

from harness.vision_runtime import VISION_SYSTEM_PROMPT, OllamaVisionRuntime, VisionRuntimeError

_TAG = "qwen3.5:2b-q4_K_M"
_DIGEST = "c" * 64
_IMAGE = b"\x89PNG\r\n\x1a\n" + b"pixels" * 10


def _ollama(
    *, digest: str = _DIGEST, status: int = 200
) -> tuple[httpx.MockTransport, list[dict[str, object]]]:
    sent: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": _TAG, "digest": digest}]})
        body = json.loads(request.content)
        sent.append(body)
        if status != 200:
            return httpx.Response(
                status, json={"error": "boom " + body["messages"][1]["images"][0]}
            )
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": " O total é R$ 53,28. "},
                "eval_count": 9,
            },
        )

    return httpx.MockTransport(handler), sent


def _runtime(transport: httpx.MockTransport) -> OllamaVisionRuntime:
    return OllamaVisionRuntime(
        base_url="http://ollama",
        model=_TAG,
        expected_digest=_DIGEST,
        max_output_tokens=700,
        context_tokens=8192,
        transport=transport,
    )


def test_the_image_goes_in_the_images_field_and_text_comes_back() -> None:
    transport, sent = _ollama()

    answer = asyncio.run(_runtime(transport).describe(_IMAGE, "Qual o total?"))

    assert answer.text == "O total é R$ 53,28."
    assert answer.output_tokens == 9
    body = sent[0]
    assert body["model"] == _TAG and body["think"] is False and body["stream"] is False
    messages = cast(list[dict[str, Any]], body["messages"])
    assert messages[0] == {"role": "system", "content": VISION_SYSTEM_PROMPT}
    assert messages[1]["content"] == "Qual o total?"
    assert base64.b64decode(cast(list[str], messages[1]["images"])[0]) == _IMAGE


def test_a_model_the_contract_did_not_pin_is_never_asked() -> None:
    transport, sent = _ollama(digest="d" * 64)

    with pytest.raises(VisionRuntimeError) as raised:
        asyncio.run(_runtime(transport).describe(_IMAGE, "q"))

    assert raised.value.error is not None
    assert raised.value.error["code"] == "model_digest_mismatch"
    assert sent == []


def test_a_refusal_never_echoes_the_image() -> None:
    transport, _ = _ollama(status=500)

    with pytest.raises(VisionRuntimeError) as raised:
        asyncio.run(_runtime(transport).describe(_IMAGE, "q"))

    encoded = base64.b64encode(_IMAGE).decode()
    assert raised.value.retryable
    assert encoded not in str(raised.value)
    assert encoded not in json.dumps(raised.value.error)
