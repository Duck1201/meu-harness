"""describe_image lê do Workspace com a política de sempre e só devolve texto (ADR 0016)."""

import asyncio
import json
import struct
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from harness import load_config
from harness.domain import Grant, JsonValue, SessionPolicy, ToolCall, ToolResultStatus
from harness.ports import ModelRuntimeError, VisionAnswer
from harness.vision_tools import CAVEAT, VisionToolExecutor, image_format, image_size

_PNG = (
    b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 640, 480) + b"\x08\x02" * 20
)


class _Vision:
    def __init__(self, text: str = "O total é R$ 53,28.", error: ModelRuntimeError | None = None):
        self.calls: list[tuple[bytes, str]] = []
        self._text = text
        self._error = error

    async def describe(self, image: bytes, question: str) -> VisionAnswer:
        self.calls.append((image, question))
        if self._error is not None:
            raise self._error
        return VisionAnswer(text=self._text, latency_ms=12.5, output_tokens=7)


def _policy(*grants: str) -> SessionPolicy:
    now = datetime.now(UTC)
    return SessionPolicy(
        conversation_id="c",
        grants=tuple(
            Grant(id=f"g-{g}", conversation_id="c", permission=g, scope="workspace", granted_at=now)
            for g in grants
        ),
    )


def _executor(
    root: Path, vision: _Vision, *grants: str, max_bytes: int | None = None
) -> VisionToolExecutor:
    config = load_config()
    vision_config = config.vision
    if max_bytes is not None:
        vision_config = vision_config.model_copy(update={"max_image_bytes": max_bytes})
    return VisionToolExecutor(
        registry=config.tool_registry,
        workspace_root=root,
        session_policy=_policy(*(grants or ("WorkspaceRootGrant",))),
        runtime=vision,
        config=vision_config,
    )


def _call(path: str, question: str = "Qual o total?") -> ToolCall:
    return ToolCall(
        id="call-1", name="describe_image", arguments={"file_path": path, "question": question}
    )


def test_the_answer_comes_back_with_the_caveat_and_the_image_identity(tmp_path: Path) -> None:
    (tmp_path / "prints").mkdir()
    (tmp_path / "prints" / "cupom.png").write_bytes(_PNG)
    vision = _Vision()

    result = asyncio.run(_executor(tmp_path, vision).execute(_call("prints/cupom.png")))

    assert result.status is ToolResultStatus.SUCCESS
    assert vision.calls == [(_PNG, "Qual o total?")]
    data = cast(Mapping[str, JsonValue], result.data)
    assert data["answer"] == "O total é R$ 53,28."
    assert data["caveat"] == CAVEAT
    image = cast(Mapping[str, JsonValue], data["image"])
    assert image["path"] == "prints/cupom.png" and image["format"] == "png"
    assert (image["width"], image["height"]) == (640, 480)
    assert image["bytes"] == len(_PNG)
    assert result.meta["producer"] == "local_vision" and result.meta["taints"] == []


def test_the_image_bytes_never_reach_the_result(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(_PNG)

    result = asyncio.run(_executor(tmp_path, _Vision()).execute(_call("a.png")))

    serialized = json.dumps({"data": result.data, "meta": result.meta}, ensure_ascii=False)
    assert "IHDR" not in serialized
    assert "iVBOR" not in serialized  # o começo do PNG em base64


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("/etc/passwd", "absolute_path_not_allowed"),
        ("../fora.png", "path_traversal_not_allowed"),
        (".env", "sensitive_path_denied"),
        ("link.png", "path_outside_workspace"),
    ],
)
def test_the_workspace_path_policy_applies(tmp_path: Path, path: str, code: str) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-fora.png"
    outside.write_bytes(_PNG)
    (tmp_path / "link.png").symlink_to(outside)
    vision = _Vision()

    result = asyncio.run(_executor(tmp_path, vision).execute(_call(path)))

    assert result.status is ToolResultStatus.BLOCKED
    assert result.error is not None and result.error["code"] == code
    assert vision.calls == []


def test_the_ceiling_is_checked_before_a_byte_is_read(tmp_path: Path) -> None:
    (tmp_path / "grande.png").write_bytes(_PNG)
    vision = _Vision()

    result = asyncio.run(
        _executor(tmp_path, vision, max_bytes=len(_PNG) - 1).execute(_call("grande.png"))
    )

    assert result.status is ToolResultStatus.BLOCKED
    assert result.error is not None and result.error["code"] == "image_too_large"
    assert vision.calls == []


def test_an_extension_proves_nothing(tmp_path: Path) -> None:
    (tmp_path / "falso.png").write_text("não sou uma imagem", encoding="utf-8")
    vision = _Vision()

    result = asyncio.run(_executor(tmp_path, vision).execute(_call("falso.png")))

    assert result.status is ToolResultStatus.FAILED
    assert result.error is not None and result.error["code"] == "not_an_image"
    assert vision.calls == []


def test_without_the_workspace_grant_nothing_is_read(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(_PNG)
    vision = _Vision()
    executor = _executor(tmp_path, vision, "WriteGrant")

    preflight = asyncio.run(executor.preflight([_call("a.png")]))
    result = asyncio.run(executor.execute(_call("a.png")))

    assert not preflight.allowed and preflight.reason_code == "workspace_root_grant_required"
    assert result.status is ToolResultStatus.BLOCKED
    assert vision.calls == []


def test_a_vision_failure_is_failed_not_blocked(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(_PNG)
    down = ModelRuntimeError(
        "down", error={"code": "vision_unavailable", "message": "ConnectError"}, retryable=True
    )

    failed = asyncio.run(_executor(tmp_path, _Vision(error=down)).execute(_call("a.png")))
    silent = asyncio.run(_executor(tmp_path, _Vision(text="")).execute(_call("a.png")))

    assert failed.status is ToolResultStatus.FAILED and failed.retryable
    assert failed.error is not None and failed.error["code"] == "vision_unavailable"
    assert silent.status is ToolResultStatus.FAILED
    assert silent.error is not None and silent.error["code"] == "vision_empty_answer"


def test_formats_and_sizes_come_from_the_bytes() -> None:
    jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    jpeg += b"\xff\xc0\x00\x11\x08" + struct.pack(">HH", 200, 300) + b"\x03" + b"\x00" * 12
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8X" + b"\x00" * 8 + (99).to_bytes(3, "little")
    webp += (49).to_bytes(3, "little")

    assert image_format(_PNG[:16]) == "png"
    assert image_format(jpeg[:16]) == "jpeg"
    assert image_format(webp[:16]) == "webp"
    assert image_format(b"GIF89a" + b"\x00" * 10) is None
    assert image_size(jpeg, "jpeg") == (300, 200)
    assert image_size(webp, "webp") == (100, 50)
