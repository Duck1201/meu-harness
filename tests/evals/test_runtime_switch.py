"""O switch deixa um modelo por vez na placa e cada braço com o seu tokenizer."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from harness import load_config
from harness.config import RuntimeBackend, RuntimeIdentityConfig, RuntimeProfileConfig
from harness.evals.runtime_switch import (
    BackendLauncher,
    ManagedRuntime,
    ProfileRuntimeSwitch,
    RuntimeSwitchError,
)
from harness.ports import ModelMessage, ModelRequest, ModelResponse, ToolSchema


class _Runtime:
    def __init__(self, name: str) -> None:
        self.name = name
        self.generated = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:
        del request
        self.generated += 1
        return ModelResponse(content="ok")

    async def aclose(self) -> None:
        return None


class _Launcher:
    def __init__(self, log: list[str], backend: str) -> None:
        self._log = log
        self._backend = backend

    async def start(self, profile: RuntimeProfileConfig) -> ManagedRuntime:
        self._log.append(f"start:{profile.id}")
        return _Runtime(profile.id)

    async def stop(self) -> None:
        self._log.append(f"stop:{self._backend}")

    async def vacate(self) -> None:
        self._log.append(f"vacate:{self._backend}")


class _Estimator:
    @property
    def validated(self) -> bool:
        return True

    def estimate(self, messages: Sequence[ModelMessage], tools: Sequence[ToolSchema]) -> int:
        del messages, tools
        return 0


def _profile(profile_id: str, backend: RuntimeBackend) -> RuntimeProfileConfig:
    return load_config().runtime_profile.model_copy(
        update={"id": profile_id, "runtime": RuntimeIdentityConfig(backend=backend)}
    )


def _switch(tmp_path: Path, log: list[str]) -> ProfileRuntimeSwitch:
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.json").write_text(f'{{"{name}": 1}}', encoding="utf-8")
    launchers: dict[RuntimeBackend, BackendLauncher] = {
        RuntimeBackend.OLLAMA: _Launcher(log, "ollama"),
        RuntimeBackend.LLAMA_CPP: _Launcher(log, "llama_cpp"),
    }
    return ProfileRuntimeSwitch(
        launchers=launchers,
        tokenizer_dir=tmp_path,
        estimator_factory=lambda path, digest: _Estimator(),
    )


def test_the_same_profile_is_activated_once_and_warmed_up(tmp_path: Path) -> None:
    log: list[str] = []
    switch = _switch(tmp_path, log)
    profile = _profile("a", RuntimeBackend.OLLAMA)

    first, _ = asyncio.run(switch.activate(profile))
    second, _ = asyncio.run(switch.activate(profile))

    assert first is second
    assert log.count("start:a") == 1
    assert isinstance(first, _Runtime) and first.generated == 1


def test_a_new_profile_stops_the_old_and_vacates_the_other_backend(tmp_path: Path) -> None:
    log: list[str] = []
    switch = _switch(tmp_path, log)

    async def scenario() -> None:
        await switch.activate(_profile("a", RuntimeBackend.OLLAMA))
        await switch.activate(_profile("b", RuntimeBackend.LLAMA_CPP))

    asyncio.run(scenario())

    assert log == ["vacate:llama_cpp", "start:a", "stop:ollama", "vacate:ollama", "start:b"]


def test_the_tokenizer_digest_is_recorded_per_profile(tmp_path: Path) -> None:
    switch = _switch(tmp_path, [])

    async def scenario() -> None:
        await switch.activate(_profile("a", RuntimeBackend.OLLAMA))
        await switch.activate(_profile("c", RuntimeBackend.OLLAMA))

    asyncio.run(scenario())

    assert set(switch.tokenizer_digests) == {"a", "c"}
    assert switch.tokenizer_digests["a"] != switch.tokenizer_digests["c"]


def test_a_profile_without_its_tokenizer_does_not_start(tmp_path: Path) -> None:
    log: list[str] = []
    switch = _switch(tmp_path, log)

    with pytest.raises(RuntimeSwitchError) as raised:
        asyncio.run(switch.activate(_profile("missing", RuntimeBackend.OLLAMA)))

    assert raised.value.reason_code == "tokenizer_file_missing"
    assert log == []


def test_a_backend_without_launcher_is_refused(tmp_path: Path) -> None:
    switch = ProfileRuntimeSwitch(launchers={}, tokenizer_dir=tmp_path)

    with pytest.raises(RuntimeSwitchError) as raised:
        asyncio.run(switch.activate(_profile("a", RuntimeBackend.LLAMA_CPP)))

    assert raised.value.reason_code == "backend_not_configured:llama_cpp"
