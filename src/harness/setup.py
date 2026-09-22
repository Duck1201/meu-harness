import hashlib
import hmac
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict
from tokenizers import Tokenizer

from .host_config import (
    CredentialStore,
    HostConfig,
    HostConfigStore,
    default_state_dir,
    keeping_fields_outside_the_form,
)


class SetupSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_workspace_roots: tuple[Path, ...]
    tokenizer_path: Path
    state_dir: Path
    allowed_origins: tuple[str, ...]
    searxng_url: str | None = None
    ollama_url: str = "http://127.0.0.1:11434"
    browser_executable: Path | None = None


class SetupStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configured: bool
    required: bool
    restart_required: bool
    token_expires_at: datetime | None
    suggested_state_dir: Path
    suggested_tokenizer_path: Path


class SetupError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class SetupController:
    def __init__(
        self,
        host_store: HostConfigStore,
        *,
        credential_store: CredentialStore | None = None,
        reopen: bool = False,
        token: str | None = None,
        ttl_seconds: float = 600,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("setup token TTL must be positive")
        self._host_store = host_store
        self._credential_store = credential_store or CredentialStore(host_store.credentials_path)
        self._now = now or _utc_now
        current = self._now()
        if current.tzinfo is None:
            raise ValueError("setup clock must return timezone-aware datetimes")
        self._configured = host_store.exists()
        self._active = reopen or not self._configured
        self._restart_required = False
        self._token = token or secrets.token_urlsafe(32) if self._active else None
        self._expires_at = current + timedelta(seconds=ttl_seconds) if self._active else None
        self._lock = Lock()

    @property
    def status(self) -> SetupStatus:
        suggested_state_dir = default_state_dir()
        return SetupStatus(
            configured=self._configured,
            required=self._active,
            restart_required=self._restart_required,
            token_expires_at=self._expires_at if self._active else None,
            suggested_state_dir=suggested_state_dir,
            suggested_tokenizer_path=suggested_state_dir / "tokenizer.json",
        )

    def boot_console_message(self, setup_url: str = "/setup") -> str | None:
        if not self._active or self._token is None:
            return None
        return f"Harness setup: {setup_url} token={self._token}"

    def authorize(self, supplied_token: str | None) -> None:
        with self._lock:
            self._authorize(supplied_token)

    def complete(self, supplied_token: str | None, submission: SetupSubmission) -> None:
        with self._lock:
            self._authorize(supplied_token)
            # Reabrir o setup (`--setup`) sobre um host já configurado reescreve o
            # arquivo; o que o formulário não mostra continua valendo.
            self._host_store.write(
                keeping_fields_outside_the_form(
                    validated_host_config(submission), self._host_store.load_optional()
                )
            )
            self._configured = True
            self._active = False
            self._restart_required = True
            self._token = None
            self._expires_at = None

    def _authorize(self, supplied_token: str | None) -> None:
        if not self._active or self._token is None:
            raise SetupError(
                "setup_not_available",
                "Setup is not available for this boot.",
                status_code=409,
            )
        expires_at = self._expires_at
        if expires_at is None or self._now() >= expires_at:
            raise SetupError(
                "setup_token_expired",
                "The setup token has expired.",
                status_code=410,
            )
        if supplied_token is None or not hmac.compare_digest(supplied_token, self._token):
            raise SetupError(
                "invalid_setup_token",
                "The setup token is invalid.",
                status_code=403,
            )


def validated_host_config(submission: SetupSubmission) -> HostConfig:
    """Valida uma submissão e devolve o HostConfig — usada pelo setup e pelo painel."""
    tokenizer_path = submission.tokenizer_path
    if not tokenizer_path.is_absolute():
        raise SetupError(
            "invalid_setup_configuration",
            "Setup paths must be absolute and canonical.",
            status_code=422,
        )
    try:
        canonical_tokenizer = tokenizer_path.resolve(strict=True)
    except OSError as error:
        raise SetupError(
            "invalid_tokenizer",
            "The tokenizer file could not be validated.",
            status_code=422,
        ) from error
    if canonical_tokenizer != tokenizer_path or not canonical_tokenizer.is_file():
        raise SetupError(
            "invalid_tokenizer",
            "The tokenizer file could not be validated.",
            status_code=422,
        )

    try:
        with canonical_tokenizer.open("rb") as tokenizer_file:
            actual_digest = hashlib.file_digest(tokenizer_file, "sha256").hexdigest()
    except OSError as error:
        raise SetupError(
            "invalid_tokenizer",
            "The tokenizer file could not be validated.",
            status_code=422,
        ) from error
    try:
        Tokenizer.from_file(  # pyright: ignore[reportUnknownMemberType]
            str(canonical_tokenizer)
        )
    except Exception as error:
        raise SetupError(
            "invalid_tokenizer",
            "The tokenizer file could not be validated.",
            status_code=422,
        ) from error

    origins = tuple(dict.fromkeys(submission.allowed_origins))
    if any(not _valid_origin(origin) for origin in origins):
        raise SetupError(
            "invalid_allowed_origin",
            "Allowed origins must be HTTP or HTTPS origins.",
            status_code=422,
        )
    try:
        config = HostConfig(
            allowed_workspace_roots=submission.allowed_workspace_roots,
            tokenizer_path=canonical_tokenizer,
            tokenizer_digest=actual_digest,
            state_dir=submission.state_dir,
            allowed_origins=origins,
            searxng_url=submission.searxng_url,
            ollama_url=submission.ollama_url,
            browser_executable=submission.browser_executable,
        )
    except ValueError as error:
        raise SetupError(
            "invalid_setup_configuration",
            "Setup paths and workspace roots must be absolute, canonical, and valid.",
            status_code=422,
        ) from error
    return config


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


def _utc_now() -> datetime:
    return datetime.now(UTC)
