import json
import os
import stat
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, field_validator

# Campos que já existiram em disco e hoje não existem mais. Os modelos são
# extra="forbid", então um arquivo antigo deixaria o servidor sem subir; descartar
# a chave é mais honesto que fingir que ela ainda significa algo.
_LEGACY_KEYS = frozenset({"brave_credential_ref", "brave_api_key"})


def _without_legacy_keys(payload: bytes) -> Mapping[str, object]:
    parsed: object = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("host configuration must be a JSON object")
    fields = cast(dict[str, object], parsed)
    return {key: value for key, value in fields.items() if key not in _LEGACY_KEYS}


class HostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    allowed_workspace_roots: tuple[Path, ...]
    tokenizer_path: Path
    tokenizer_digest: str
    state_dir: Path
    allowed_origins: tuple[str, ...]
    searxng_url: str | None = None
    ollama_url: str = "http://127.0.0.1:11434"
    browser_executable: Path | None = None
    # Só pesam quando o perfil da rota é servido pelo llama.cpp (ADR 0013). O
    # arquivo do modelo é do host e o digest dele é do contrato; o servidor em si
    # é subido por `scripts/llama-server.sh` a partir destes mesmos campos.
    llama_server_url: str = "http://127.0.0.1:8081"
    llama_server_executable: Path | None = None
    gguf_paths: Mapping[str, Path] = {}

    @field_validator("allowed_workspace_roots")
    @classmethod
    def validate_workspace_roots(cls, roots: tuple[Path, ...]) -> tuple[Path, ...]:
        validated: list[Path] = []
        for root in roots:
            canonical = _canonical_path(root, must_exist=True)
            if not canonical.is_dir():
                raise ValueError("allowed workspace roots must be directories")
            if canonical not in validated:
                validated.append(canonical)
        return tuple(validated)

    @field_validator("tokenizer_path", "state_dir")
    @classmethod
    def validate_absolute_path(cls, path: Path) -> Path:
        return _canonical_path(path, must_exist=False)

    @field_validator("tokenizer_digest")
    @classmethod
    def validate_tokenizer_digest(cls, value: str) -> str:
        digest = value.removeprefix("sha256:").lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("tokenizer digest must be a SHA-256 digest")
        return digest

    @field_validator("browser_executable")
    @classmethod
    def validate_browser_executable(cls, path: Path | None) -> Path | None:
        # Declarar o binário é o que tira a escalação do web_fetch do palpite: sem
        # isso o harness varre o PATH e o comportamento muda com a máquina.
        if path is None:
            return None
        canonical = _canonical_path(path, must_exist=True)
        if not os.access(canonical, os.X_OK):
            raise ValueError("browser_executable must be executable")
        return canonical

    @field_validator("llama_server_executable")
    @classmethod
    def validate_llama_server_executable(cls, path: Path | None) -> Path | None:
        if path is None:
            return None
        canonical = _canonical_path(path, must_exist=True)
        if not os.access(canonical, os.X_OK):
            raise ValueError("llama_server_executable must be executable")
        return canonical

    @field_validator("gguf_paths")
    @classmethod
    def validate_gguf_paths(cls, paths: Mapping[str, Path]) -> Mapping[str, Path]:
        return {profile: _canonical_path(path, must_exist=False) for profile, path in paths.items()}

    @field_validator("ollama_url", "llama_server_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        parsed = urlsplit(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("runtime URLs must be HTTP or HTTPS URLs")
        return value.strip()

    @field_validator("searxng_url")
    @classmethod
    def validate_searxng_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        url = value.strip()
        if not url:
            return None
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("searxng_url must be an HTTP or HTTPS URL")
        return url

    @field_validator("allowed_origins")
    @classmethod
    def validate_allowed_origins(cls, origins: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(dict.fromkeys(origins))
        if any(not _valid_origin(origin) for origin in validated):
            raise ValueError("allowed origins must be HTTP or HTTPS origins")
        return validated


# Campos que o formulário de setup e a aba Configurações não carregam. Quem grava
# pelo formulário reescreve o arquivo inteiro, e sem isto apagaria o servidor
# llama.cpp e os GGUFs que o Operator declarou à mão.
_FIELDS_OUTSIDE_THE_FORM = (
    "llama_server_url",
    "llama_server_executable",
    "gguf_paths",
)


def keeping_fields_outside_the_form(new: HostConfig, previous: HostConfig | None) -> HostConfig:
    if previous is None:
        return new
    return new.model_copy(
        update={field: getattr(previous, field) for field in _FIELDS_OUTSIDE_THE_FORM}
    )


class HostConfigStore:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        selected = Path(path) if path is not None else default_host_config_path(environ)
        self.path = selected.expanduser().resolve(strict=False)

    @property
    def credentials_path(self) -> Path:
        return self.path.with_name("credentials.json")

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> HostConfig:
        return HostConfig.model_validate(_without_legacy_keys(_read_private_file(self.path)))

    def load_optional(self) -> HostConfig | None:
        try:
            return self.load()
        except FileNotFoundError:
            return None

    def write(self, config: HostConfig) -> None:
        payload = config.model_dump(mode="json")
        _atomic_write_json(self.path, payload)


class _CredentialFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    operator_password_hash: str | None = None


class CredentialStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)

    def write_operator_password_hash(self, value: str) -> None:
        self._write(operator_password_hash=_normalize_secret(value))

    def read_operator_password_hash(self) -> str | None:
        try:
            return self._load().operator_password_hash
        except (OSError, ValueError):
            return None

    def _load(self) -> _CredentialFile:
        return _CredentialFile.model_validate(_without_legacy_keys(_read_private_file(self.path)))

    def _write(self, **fields: str) -> None:
        payload: dict[str, object] = {"schema_version": 1, "operator_password_hash": None}
        with suppress(OSError, ValueError):
            payload["operator_password_hash"] = self._load().operator_password_hash
        payload.update(fields)
        _atomic_write_json(self.path, payload)


def default_host_config_path(environ: Mapping[str, str] | None = None) -> Path:
    environment = os.environ if environ is None else environ
    configured = environment.get("XDG_CONFIG_HOME")
    base = Path(configured).expanduser() if configured else Path.home() / ".config"
    if not base.is_absolute():
        raise ValueError("XDG_CONFIG_HOME must be an absolute path")
    return (base / "harness-2" / "host.json").resolve(strict=False)


def default_state_dir(environ: Mapping[str, str] | None = None) -> Path:
    environment = os.environ if environ is None else environ
    configured = environment.get("XDG_STATE_HOME")
    base = Path(configured).expanduser() if configured else Path.home() / ".local/state"
    if not base.is_absolute():
        raise ValueError("XDG_STATE_HOME must be an absolute path")
    return (base / "harness-2").resolve(strict=False)


def _canonical_path(path: Path, *, must_exist: bool) -> Path:
    if not path.is_absolute():
        raise ValueError("host paths must be absolute")
    try:
        canonical = path.resolve(strict=must_exist)
    except OSError as error:
        raise ValueError("host path could not be resolved") from error
    if path != canonical:
        raise ValueError("host paths must be canonical")
    return canonical


def _normalize_secret(value: str) -> str:
    normalized = value.strip()
    if not normalized or "\n" in normalized or "\r" in normalized:
        raise ValueError("credential must be a non-empty single line")
    return normalized


def _valid_origin(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and parsed.path == ""
        and parsed.query == ""
        and parsed.fragment == ""
        and (port is None or 0 < port < 65536)
    )


def _read_private_file(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise OSError("configuration file must be private and regular")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 65536):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        content = (
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        written = 0
        while written < len(content):
            written += os.write(descriptor, content[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()
