"""O executor por trás do efeito `local_inference`: `describe_image` (ADR 0016).

Lê uma imagem do Workspace com a mesma política de caminho de `read_file` —
raiz, symlink, arquivo sensível, caminho negado pelo host — e pergunta ao modelo
de visão local. O gate do WorkspaceRootGrant vive aqui também, como vive em cada
executor que realiza um efeito: quem lê o arquivo é quem responde por ele.

Os bytes da imagem só existem entre o disco e o runtime. O ToolResult leva a
resposta, o aviso de incerteza e a identidade da imagem — digest, tamanho,
formato e dimensões —, nunca o conteúdo, e é isso que chega ao CanonicalHistory.
"""

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator, FormatChecker

from .config import ToolDefinitionConfig, ToolRegistryConfig, VisionConfig
from .domain import (
    LOCAL_INFERENCE_EFFECT,
    JsonValue,
    SessionPolicy,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    grant_reason_code,
)
from .ports import ConfirmationPreview, ModelRuntimeError, ToolBatchPreflight, VisionRuntime
from .workspace_paths import PathPolicyError, WorkspacePathPolicy, relative_parts

_PRODUCER = "local_vision"
# O que o gate de visão pede em palavras: a leitura é de um modelo pequeno, e o
# erro que ele comete — dígito trocado em texto miúdo — é silencioso. Em inglês
# porque é voltado ao modelo, como toda instrução do harness.
CAVEAT = (
    "Automatic reading by a small local vision model. Numbers, codes and small text can be "
    "misread; say so when the answer depends on them, and do not present the reading as certain."
)


def image_format(head: bytes) -> str | None:
    """PNG, JPEG ou WebP pelos magic bytes; a extensão do arquivo não prova nada."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def image_size(data: bytes, image_type: str) -> tuple[int, int] | None:
    """Largura e altura do cabeçalho, sem decodificar a imagem; None se não souber."""
    try:
        if image_type == "png" and len(data) >= 24:
            width, height = struct.unpack(">II", data[16:24])
            return width, height
        if image_type == "jpeg":
            return _jpeg_size(data)
        if image_type == "webp":
            return _webp_size(data)
    except struct.error:
        return None
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE):
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return width, height
        length = struct.unpack(">H", data[index + 2 : index + 4])[0]
        index += 2 + length
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8 " and len(data) >= 30:
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


class _Refusal(Exception):
    def __init__(self, status: ToolResultStatus, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


class VisionToolExecutor:
    def __init__(
        self,
        *,
        registry: ToolRegistryConfig,
        workspace_root: str | Path,
        session_policy: SessionPolicy,
        runtime: VisionRuntime,
        config: VisionConfig,
        host_denied_paths: Sequence[str] = (),
    ) -> None:
        self._registry = {
            definition.name: definition
            for definition in registry.model_tools
            if LOCAL_INFERENCE_EFFECT in definition.effects
        }
        self._paths = WorkspacePathPolicy(
            Path(workspace_root).resolve(strict=True), host_denied_paths
        )
        self._effective_grants = session_policy.effective_grants
        self._runtime = runtime
        self._config = config

    async def preflight(self, calls: Sequence[ToolCall]) -> ToolBatchPreflight:
        seen: set[str] = set()
        for raw in calls:
            call = self._normalized(raw)
            try:
                if not call.id or call.id in seen:
                    raise _Refusal(
                        ToolResultStatus.BLOCKED,
                        "duplicate_tool_call_id",
                        "Tool call IDs must be non-empty and unique.",
                    )
                seen.add(call.id)
                self._validate(call)
            except _Refusal as refusal:
                return ToolBatchPreflight(
                    allowed=False, reason_code=refusal.code, detail=refusal.detail
                )
        return ToolBatchPreflight(allowed=True)

    async def execute(self, call: ToolCall) -> ToolResult:
        call = self._normalized(call)
        try:
            parts = self._validate(call)
            data, image_type = self._read(parts)
        except _Refusal as refusal:
            return _error(call, refusal.status, refusal.code, refusal.detail)
        question = cast(str, call.arguments["question"]).strip()
        try:
            answer = await self._runtime.describe(data, question)
        except ModelRuntimeError as error:
            code = error.error.get("code") if error.error else None
            return _error(
                call,
                ToolResultStatus.FAILED,
                code if isinstance(code, str) else "vision_unavailable",
                "The local vision model did not answer.",
                retryable=error.retryable,
            )
        if not answer.text:
            return _error(
                call,
                ToolResultStatus.FAILED,
                "vision_empty_answer",
                "The vision model said nothing.",
            )
        size = image_size(data, image_type)
        image: dict[str, JsonValue] = {
            "path": "/".join(parts),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "format": image_type,
            "width": size[0] if size else None,
            "height": size[1] if size else None,
        }
        return ToolResult(
            tool_call_id=call.id,
            status=ToolResultStatus.SUCCESS,
            retryable=False,
            data={"answer": answer.text, "caveat": CAVEAT, "image": image},
            error=None,
            meta={
                "producer": _PRODUCER,
                "truncated": False,
                # Imagem do Workspace segue a regra de read_file: o Operator é quem
                # põe o arquivo lá, e ler não o marca como vindo da web (ADR 0016).
                "taints": [],
                "latency_ms": answer.latency_ms,
            },
        )

    async def preview(self, call: ToolCall) -> ConfirmationPreview | None:
        del call
        return None

    def _read(self, parts: tuple[str, ...]) -> tuple[bytes, str]:
        try:
            path = self._paths.resolve_read(parts, strict=True)
        except FileNotFoundError:
            raise _Refusal(
                ToolResultStatus.FAILED, "path_not_found", "The image does not exist."
            ) from None
        except PermissionError:
            raise _Refusal(
                ToolResultStatus.FAILED, "filesystem_permission_denied", "The image cannot be read."
            ) from None
        except PathPolicyError as issue:
            raise _Refusal(ToolResultStatus.BLOCKED, issue.code, issue.detail) from None
        if not path.is_file():
            raise _Refusal(ToolResultStatus.FAILED, "not_a_file", "The path is not a file.")
        # O teto vem antes do primeiro byte lido: read_file carrega o arquivo
        # inteiro e só então pagina, e uma imagem de 2 GB não pode ter a mesma sorte.
        size = path.stat().st_size
        if size > self._config.max_image_bytes:
            raise _Refusal(
                ToolResultStatus.BLOCKED,
                "image_too_large",
                f"The image has {size} bytes; the limit is {self._config.max_image_bytes}.",
            )
        data = path.read_bytes()
        image_type = image_format(data[:16])
        if image_type is None or image_type not in self._config.accepted_formats:
            accepted = ", ".join(self._config.accepted_formats)
            raise _Refusal(
                ToolResultStatus.FAILED, "not_an_image", f"Only {accepted} images can be read."
            )
        return data, image_type

    def _normalized(self, call: ToolCall) -> ToolCall:
        definition = self._registry.get(call.name)
        if definition is None:
            return call
        arguments = definition.normalized_arguments(call.arguments)
        return call if arguments == call.arguments else replace(call, arguments=arguments)

    def _validate(self, call: ToolCall) -> tuple[str, ...]:
        definition = self._definition(call)
        validator = Draft202012Validator(
            cast(Mapping[str, Any], definition.parameters), format_checker=FormatChecker()
        )
        errors = sorted(
            validator.iter_errors(call.arguments),  # pyright: ignore[reportUnknownMemberType]
            key=lambda error: list(error.path),
        )
        if errors:
            raise _Refusal(ToolResultStatus.BLOCKED, "invalid_tool_arguments", errors[0].message)
        if not cast(str, call.arguments.get("question", "")).strip():
            raise _Refusal(
                ToolResultStatus.BLOCKED,
                "invalid_tool_arguments",
                "The question must not be blank.",
            )
        # Por efeito, nunca por nome: os grants vêm do registry, como nos outros executores.
        missing = [g for g in definition.required_grants if g not in self._effective_grants]
        if missing:
            raise _Refusal(
                ToolResultStatus.BLOCKED,
                grant_reason_code(missing[0]),
                f"The {missing[0]} is required for this effect.",
            )
        try:
            parts = relative_parts(cast(str, call.arguments["file_path"]), allow_dot=False)
            self._paths.check(parts)
            self._paths.resolve_read(parts, strict=False)
        except PathPolicyError as issue:
            raise _Refusal(ToolResultStatus.BLOCKED, issue.code, issue.detail) from None
        return parts

    def _definition(self, call: ToolCall) -> ToolDefinitionConfig:
        definition = self._registry.get(call.name)
        if definition is None:
            raise _Refusal(ToolResultStatus.BLOCKED, "unknown_tool", "Unknown vision tool.")
        if definition.status != "enabled":
            raise _Refusal(
                ToolResultStatus.BLOCKED, "tool_not_enabled", "The requested tool is not enabled."
            )
        return definition


def _error(
    call: ToolCall, status: ToolResultStatus, code: str, message: str, *, retryable: bool = False
) -> ToolResult:
    error: Mapping[str, JsonValue] = {"code": code, "message": message}
    return ToolResult(
        tool_call_id=call.id,
        status=status,
        retryable=retryable,
        data=None,
        error=error,
        meta={"producer": "harness", "truncated": False, "taints": []},
    )


__all__ = ["CAVEAT", "VisionToolExecutor", "image_format", "image_size"]
