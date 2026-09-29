import json
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

import harness.host_config as host_config_module
from harness import CredentialStore, HostConfig, HostConfigStore


def test_host_config_round_trips_as_private_versioned_json(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    state_dir = tmp_path / "state"
    store = HostConfigStore(tmp_path / "config" / "host.json")
    config = HostConfig(
        schema_version=1,
        allowed_workspace_roots=(workspace.resolve(),),
        tokenizer_path=tokenizer.resolve(),
        tokenizer_digest="0" * 64,
        state_dir=state_dir.resolve(),
        allowed_origins=("http://127.0.0.1:8000",),
        searxng_url="http://127.0.0.1:8080/search",
    )

    result = store.write(config)

    assert result is None
    assert store.load() == config
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert json.loads(store.path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_credential_store_keeps_the_operator_password_private(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials.json")

    result = store.write_operator_password_hash("pbkdf2_sha256$1$00$11")

    assert result is None
    assert store.read_operator_password_hash() == "pbkdf2_sha256$1$00$11"
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "operator_password_hash": "pbkdf2_sha256$1$00$11",
    }
    assert CredentialStore(tmp_path / "missing.json").read_operator_password_hash() is None


def test_files_written_before_the_brave_removal_still_load(tmp_path: Path) -> None:
    """extra="forbid" would otherwise leave an upgraded host unable to boot."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    host_path = tmp_path / "config" / "host.json"
    host_path.parent.mkdir()
    host_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "allowed_workspace_roots": [str(workspace.resolve())],
                "tokenizer_path": str(tokenizer.resolve()),
                "tokenizer_digest": "0" * 64,
                "state_dir": str((tmp_path / "state").resolve()),
                "allowed_origins": ["http://127.0.0.1:8000"],
                "brave_credential_ref": "brave_api_key",
            }
        ),
        encoding="utf-8",
    )
    host_path.chmod(0o600)
    credentials = tmp_path / "config" / "credentials.json"
    credentials.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "brave_api_key": "old-secret",
                "operator_password_hash": "pbkdf2_sha256$1$00$11",
            }
        ),
        encoding="utf-8",
    )
    credentials.chmod(0o600)

    loaded = HostConfigStore(host_path).load()

    assert loaded.searxng_url is None
    assert CredentialStore(credentials).read_operator_password_hash() == "pbkdf2_sha256$1$00$11"


def test_host_config_replace_is_atomic_when_the_final_swap_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    store = HostConfigStore(tmp_path / "config" / "host.json")
    original = HostConfig(
        allowed_workspace_roots=(workspace.resolve(),),
        tokenizer_path=tokenizer.resolve(),
        tokenizer_digest="1" * 64,
        state_dir=(tmp_path / "state-one").resolve(),
        allowed_origins=("http://operator.test",),
    )
    store.write(original)
    updated = original.model_copy(update={"state_dir": (tmp_path / "state-two").resolve()})

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("simulated interrupted swap")

    monkeypatch.setattr(host_config_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted swap"):
        store.write(updated)

    assert store.load() == original
    assert list(store.path.parent.glob(".*.tmp")) == []


def test_default_host_path_uses_xdg_config_home(tmp_path: Path) -> None:
    store = HostConfigStore(environ={"XDG_CONFIG_HOME": str(tmp_path)})

    assert store.path == (tmp_path / "meu-harness" / "host.json").resolve()


def test_host_config_rejects_nonexistent_roots_relative_paths_and_invalid_urls(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")

    def build(
        *,
        roots: tuple[Path, ...] = (),
        state_dir: Path | None = None,
        origins: tuple[str, ...] = ("http://operator.test",),
        searxng_url: str | None = None,
        ollama_url: str = "http://127.0.0.1:11434",
    ) -> HostConfig:
        return HostConfig(
            allowed_workspace_roots=roots or (tmp_path.resolve(),),
            tokenizer_path=tokenizer.resolve(),
            tokenizer_digest="2" * 64,
            state_dir=state_dir or (tmp_path / "state").resolve(),
            allowed_origins=origins,
            searxng_url=searxng_url,
            ollama_url=ollama_url,
        )

    with pytest.raises(ValidationError, match="could not be resolved"):
        build(roots=(tmp_path / "missing",))
    with pytest.raises(ValidationError, match="absolute"):
        build(state_dir=Path("relative"))
    with pytest.raises(ValidationError, match="origins"):
        build(origins=("http://operator.test/path",))
    with pytest.raises(ValidationError, match="searxng_url"):
        build(searxng_url="not-a-url")
    with pytest.raises(ValidationError, match="ollama_url"):
        build(ollama_url="ftp://127.0.0.1:11434")


def test_browser_executable_must_be_an_absolute_executable_path(tmp_path: Path) -> None:
    """Declaring the binary is what stops web_fetch escalation from depending on PATH."""
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    browser = tmp_path / "chromium"
    browser.write_text("#!/bin/sh\n", encoding="utf-8")

    def build(browser_executable: Path | None) -> HostConfig:
        return HostConfig(
            allowed_workspace_roots=(tmp_path.resolve(),),
            tokenizer_path=tokenizer.resolve(),
            tokenizer_digest="2" * 64,
            state_dir=(tmp_path / "state").resolve(),
            allowed_origins=("http://operator.test",),
            browser_executable=browser_executable,
        )

    with pytest.raises(ValidationError, match="executable"):
        build(browser.resolve())
    browser.chmod(0o700)
    assert build(browser.resolve()).browser_executable == browser.resolve()
    assert build(None).browser_executable is None
    with pytest.raises(ValidationError, match="absolute"):
        build(Path("chromium"))


def test_saving_the_form_keeps_the_llama_server_declared_by_hand(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def config(**extra: object) -> HostConfig:
        return HostConfig.model_validate(
            {
                "allowed_workspace_roots": [str(workspace.resolve())],
                "tokenizer_path": str((tmp_path / "tokenizer.json").resolve()),
                "tokenizer_digest": "0" * 64,
                "state_dir": str((tmp_path / "state").resolve()),
                "allowed_origins": ["http://127.0.0.1:8000"],
                **extra,
            }
        )

    by_hand = config(
        llama_server_url="http://127.0.0.1:8091",
        gguf_paths={"cove_4b_llamacpp": str(tmp_path / "CoVe-4B.Q4_K_M.gguf")},
    )
    from_the_form = config(searxng_url="http://127.0.0.1:8080/search")

    saved = host_config_module.keeping_fields_outside_the_form(from_the_form, by_hand)

    assert saved.searxng_url == "http://127.0.0.1:8080/search"
    assert saved.llama_server_url == "http://127.0.0.1:8091"
    assert saved.gguf_paths == by_hand.gguf_paths
    assert host_config_module.keeping_fields_outside_the_form(from_the_form, None) == from_the_form
