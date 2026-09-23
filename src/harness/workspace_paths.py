"""Política de caminho do Workspace, compartilhada por todo executor que lê arquivo.

Saiu de `local_tools.py` quando `describe_image` passou a ler do Workspace num
executor próprio (ADR 0016). Duas cópias da regra de symlink ou de arquivo
sensível seriam duas regras que um dia divergem, e é a que ficou para trás que
deixa um `.env` sair.
"""

from collections.abc import Sequence
from pathlib import Path, PureWindowsPath

_CREDENTIAL_DIRECTORIES = frozenset(
    {".aws", ".azure", ".credentials", ".gnupg", ".kube", ".ssh", "credentials"}
)
_PRIVATE_KEY_NAMES = frozenset({"id_dsa", "id_ecdsa", "id_ed25519", "id_rsa", "identity"})
_PRIVATE_KEY_SUFFIXES = frozenset({".jks", ".key", ".p12", ".pem", ".pfx", ".pkcs12"})
_ENV_EXAMPLE_SUFFIXES = (".example", ".sample", ".template")


class PathPolicyError(Exception):
    """Um caminho que a política recusa, com o código que o ToolResult carrega."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def relative_parts(value: str, *, allow_dot: bool, allow_glob: bool = False) -> tuple[str, ...]:
    if "\x00" in value:
        raise PathPolicyError("nul_in_path", "Paths cannot contain NUL bytes.")
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise PathPolicyError("absolute_path_not_allowed", "Paths must be workspace-relative.")
    if "\\" in value:
        raise PathPolicyError("invalid_path", "Paths must use forward slashes.")
    raw_parts = value.split("/")
    if any(part == ".." for part in raw_parts):
        raise PathPolicyError(
            "path_traversal_not_allowed", "Parent traversal is not allowed in paths."
        )
    parts = tuple(part for part in raw_parts if part not in {"", "."})
    if not parts and not allow_dot:
        raise PathPolicyError("invalid_path", "The path must identify a workspace entry.")
    if not allow_glob and any(any(char in part for char in "*?[") for part in parts):
        raise PathPolicyError("invalid_path", "Wildcard characters are not allowed in this path.")
    return parts


def is_sensitive(parts: tuple[str, ...]) -> bool:
    for part in parts:
        lowered = part.lower()
        if lowered == ".git" or lowered in _CREDENTIAL_DIRECTORIES:
            return True
        if lowered == ".env" or (
            lowered.startswith(".env.") and not lowered.endswith(_ENV_EXAMPLE_SUFFIXES)
        ):
            return True
        if lowered in _PRIVATE_KEY_NAMES or Path(lowered).suffix in _PRIVATE_KEY_SUFFIXES:
            return True
    return False


def is_prefix(prefix: tuple[str, ...], path: tuple[str, ...]) -> bool:
    return len(prefix) <= len(path) and path[: len(prefix)] == prefix


class WorkspacePathPolicy:
    """Onde um executor pode ler: dentro da raiz, fora do sensível e do negado."""

    def __init__(self, root: Path, host_denied_paths: Sequence[str] = ()) -> None:
        self.root = root
        self._host_denied = tuple(
            relative_parts(path, allow_dot=True) for path in host_denied_paths
        )

    def check(self, parts: tuple[str, ...], *, literal_only: bool = False) -> None:
        checked = tuple(
            part for part in parts if not literal_only or not any(char in part for char in "*?[")
        )
        if is_sensitive(checked):
            raise PathPolicyError(
                "sensitive_path_denied", "Access to this sensitive path is denied."
            )
        if any(is_prefix(denied, parts) for denied in self._host_denied):
            raise PathPolicyError("host_path_denied", "The host policy denies this path.")

    def resolve_read(self, parts: tuple[str, ...], *, strict: bool) -> Path:
        """O caminho resolvido — symlink seguido — ainda dentro da raiz e permitido."""
        try:
            resolved = self.root.joinpath(*parts).resolve(strict=strict)
        except FileNotFoundError:
            raise
        except PermissionError:
            raise
        except (OSError, RuntimeError):
            raise PathPolicyError(
                "path_validation_failed", "The path could not be validated."
            ) from None
        if not resolved.is_relative_to(self.root):
            raise PathPolicyError(
                "path_outside_workspace", "The resolved path is outside the workspace."
            )
        self.check(resolved.relative_to(self.root).parts)
        return resolved


__all__ = [
    "PathPolicyError",
    "WorkspacePathPolicy",
    "is_prefix",
    "is_sensitive",
    "relative_parts",
]
