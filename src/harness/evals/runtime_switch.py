"""Troca de modelo entre braços de bake-off.

Um experimento que compara modelos roda os braços em sequência, e cada braço
precisa do seu modelo carregado sozinho na placa. Esta é a única peça da bancada
que conhece mais de um RuntimeProfile ao mesmo tempo; o resto do runner recebe o
runtime já pronto e não sabe que houve troca.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from ..config import RuntimeBackend, RuntimeProfileConfig
from ..llamacpp_runtime import LlamaCppRuntime, gguf_digest
from ..ollama_runtime import OllamaRuntime
from ..ports import ModelMessage, ModelRequest, ModelRole, ModelRuntime, TokenEstimator
from ..token_estimator import HuggingFaceTokenEstimator


class RuntimeSwitchError(RuntimeError):
    """O perfil pedido não está pronto; o braço não roda com outro no lugar."""

    def __init__(self, profile_id: str, reason_code: str) -> None:
        super().__init__(f"{profile_id}: {reason_code}")
        self.profile_id = profile_id
        self.reason_code = reason_code


class ManagedRuntime(ModelRuntime, Protocol):
    """Um runtime que a bancada liga e desliga."""

    async def aclose(self) -> None: ...


class BackendLauncher(Protocol):
    """Sobe o runtime de um backend para um perfil, já verificado pelo digest."""

    async def start(self, profile: RuntimeProfileConfig) -> ManagedRuntime: ...

    async def stop(self) -> None: ...

    async def vacate(self) -> None:
        """Libera a placa de tudo que este backend carregou, inclusive fora da bancada."""
        ...


async def warm_up(runtime: ModelRuntime) -> None:
    """One throwaway generation before a battery starts on a freshly loaded model.

    The seed is honoured — the same prompt at the same seed repeats exactly — but
    the first generation after the model loads does not match the ones that
    follow. Without this, whichever case happens to run first is measured under
    conditions no other case sees.
    """
    await runtime.generate(
        ModelRequest(
            messages=(ModelMessage(role=ModelRole.USER, content="ok"),),
            tools=(),
            options={},
            seed=0,
            max_output_tokens=8,
            think=False,
        )
    )


class OllamaLauncher:
    """Perfis servidos pelo Ollama: verifica a tag e descarrega o que sobrar.

    O Ollama carrega sob demanda e mantém o modelo anterior até o keep_alive
    vencer. Com dois modelos de chat na mesma placa de 8 GB o segundo vai parcial
    para a CPU, e o braço seria medido mais lento do que o modelo é. Por isso a
    subida descarrega tudo que não é o próprio modelo nem o embedding da bancada.
    """

    def __init__(self, *, base_url: str, timeout: float, keep_loaded: frozenset[str]) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._keep_loaded = keep_loaded
        self._current: OllamaRuntime | None = None

    async def start(self, profile: RuntimeProfileConfig) -> ManagedRuntime:
        runtime = OllamaRuntime(
            base_url=self._base_url,
            model=profile.model.id,
            expected_digest=profile.profile_digest_sha256,
            timeout=self._timeout,
        )
        verification = await runtime.verify_profile()
        if not verification.ready:
            await runtime.aclose()
            raise RuntimeSwitchError(profile.id, verification.reason_code or "not_ready")
        await self.release_others(runtime, keep=profile.model.id)
        self._current = runtime
        return runtime

    async def release_others(self, runtime: OllamaRuntime, *, keep: str | None) -> None:
        for model in await runtime.loaded_models():
            if model != keep and model not in self._keep_loaded:
                await runtime.unload(model)

    async def stop(self) -> None:
        if self._current is None:
            return
        await self._current.unload()
        await self._current.aclose()
        self._current = None

    async def vacate(self) -> None:
        # O braço controle roda no runtime que o script montou, fora do switch, e
        # o modelo dele continua carregado quando um braço de outro backend sobe.
        await self.stop()
        probe = OllamaRuntime(
            base_url=self._base_url, model="", expected_digest="0" * 64, timeout=self._timeout
        )
        try:
            await self.release_others(probe, keep=None)
        finally:
            await probe.aclose()


class LlamaCppLauncher:
    """Sobe um `llama-server` por perfil e o derruba quando o braço acaba.

    Os argumentos do servidor são do contrato (`installation.server_args`), e o
    arquivo é do host (`host.json#gguf_paths`). O digest do arquivo é conferido
    antes de subir: um GGUF trocado em disco sob o mesmo nome é outro modelo, e
    subir o servidor para descobrir isso custaria o load inteiro.
    """

    def __init__(
        self,
        *,
        executable: Path,
        base_url: str,
        gguf_paths: Mapping[str, Path],
        timeout: float,
        startup_timeout: float = 240.0,
        log_dir: Path | None = None,
    ) -> None:
        self._executable = executable
        self._base_url = base_url.rstrip("/")
        self._gguf_paths = dict(gguf_paths)
        self._timeout = timeout
        self._startup_timeout = startup_timeout
        self._log_dir = log_dir
        self._process: asyncio.subprocess.Process | None = None
        self._runtime: LlamaCppRuntime | None = None

    async def start(self, profile: RuntimeProfileConfig) -> ManagedRuntime:
        path = self._gguf_paths.get(profile.id)
        if path is None or not path.is_file():
            raise RuntimeSwitchError(profile.id, "model_not_installed")
        expected = str(profile.installation.get("gguf_sha256", ""))
        if await gguf_digest(path) != expected:
            raise RuntimeSwitchError(profile.id, "model_digest_mismatch")
        await self.stop()
        parsed = urlsplit(self._base_url)
        arguments = [
            "-m",
            str(path),
            "--host",
            parsed.hostname or "127.0.0.1",
            "--port",
            str(parsed.port or 8081),
            "--alias",
            profile.model.id,
            *server_arguments(profile),
        ]
        log = (
            (self._log_dir / f"llama-server-{profile.id}.log").open("ab")
            if self._log_dir is not None
            else None
        )
        self._process = await asyncio.create_subprocess_exec(
            str(self._executable),
            *arguments,
            stdout=log or asyncio.subprocess.DEVNULL,
            stderr=log or asyncio.subprocess.DEVNULL,
        )
        await self._wait_until_healthy(profile.id)
        runtime = LlamaCppRuntime(
            base_url=self._base_url,
            model=profile.model.id,
            expected_digest=expected,
            context_window=profile.context_window,
            timeout=self._timeout,
        )
        verification = await runtime.verify_profile()
        if not verification.ready:
            await runtime.aclose()
            await self.stop()
            raise RuntimeSwitchError(profile.id, verification.reason_code or "not_ready")
        self._runtime = runtime
        return runtime

    async def _wait_until_healthy(self, profile_id: str) -> None:
        deadline = time.monotonic() + self._startup_timeout
        async with httpx.AsyncClient(base_url=self._base_url, timeout=5) as client:
            while time.monotonic() < deadline:
                if self._process is not None and self._process.returncode is not None:
                    raise RuntimeSwitchError(profile_id, "server_exited")
                try:
                    if (await client.get("/health")).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
        await self.stop()
        raise RuntimeSwitchError(profile_id, "server_start_timeout")

    async def stop(self) -> None:
        if self._runtime is not None:
            await self._runtime.aclose()
            self._runtime = None
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=30)
        except TimeoutError:
            process.kill()
            await process.wait()

    async def vacate(self) -> None:
        await self.stop()


def server_arguments(profile: RuntimeProfileConfig) -> Sequence[str]:
    """Os argumentos do `llama-server` que o contrato declara para o perfil."""
    raw = profile.installation.get("server_args", [])
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise RuntimeSwitchError(profile.id, "invalid_server_args")
    return [str(item) for item in raw]


class ProfileRuntimeSwitch:
    """Implementa `RuntimeSwitch`: um perfil ativo por vez, com seu tokenizer.

    O tokenizer é do perfil, não do host: o orçamento de contexto de um braço
    Gemma medido com o vocabulário do Qwen cortaria o ModelView no lugar errado.
    Cada perfil lê `<tokenizer_dir>/<profile_id>.json`, e o digest calculado fica
    em `tokenizer_digests` para ser congelado junto do resultado.
    """

    def __init__(
        self,
        *,
        launchers: Mapping[RuntimeBackend, BackendLauncher],
        tokenizer_dir: Path,
        tokenizer_overrides: Mapping[str, Path] | None = None,
        estimator_factory: Callable[[Path, str], TokenEstimator] | None = None,
    ) -> None:
        self._launchers = dict(launchers)
        self._tokenizer_dir = tokenizer_dir
        # O perfil funcional já tem tokenizer no host antes de existir bancada de
        # modelos; o script o aponta aqui em vez de exigir uma cópia no diretório.
        self._tokenizer_overrides = dict(tokenizer_overrides or {})
        self._estimator_factory = estimator_factory or _hugging_face_estimator
        self._active: tuple[str, ModelRuntime, TokenEstimator] | None = None
        self._active_backend: RuntimeBackend | None = None
        self.tokenizer_digests: dict[str, str] = {}

    async def activate(self, profile: RuntimeProfileConfig) -> tuple[ModelRuntime, TokenEstimator]:
        if self._active is not None and self._active[0] == profile.id:
            return self._active[1], self._active[2]
        launcher = self._launchers.get(profile.runtime.backend)
        if launcher is None:
            backend = profile.runtime.backend.value
            raise RuntimeSwitchError(profile.id, f"backend_not_configured:{backend}")
        estimator = self._estimator(profile)
        await self.release()
        for backend, other in self._launchers.items():
            if backend is not profile.runtime.backend:
                await other.vacate()
        runtime = await launcher.start(profile)
        self._active_backend = profile.runtime.backend
        await warm_up(runtime)
        self._active = (profile.id, runtime, estimator)
        return runtime, estimator

    async def release(self) -> None:
        """Desliga o perfil ativo; o próximo braço sobe numa placa vazia."""
        if self._active_backend is not None:
            await self._launchers[self._active_backend].stop()
        self._active = None
        self._active_backend = None

    def _estimator(self, profile: RuntimeProfileConfig) -> TokenEstimator:
        path = self._tokenizer_overrides.get(profile.id, self._tokenizer_dir / f"{profile.id}.json")
        if not path.is_file():
            raise RuntimeSwitchError(profile.id, "tokenizer_file_missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.tokenizer_digests[profile.id] = digest
        return self._estimator_factory(path, digest)


def _hugging_face_estimator(path: Path, digest: str) -> TokenEstimator:
    return HuggingFaceTokenEstimator(path, expected_sha256=digest)


__all__ = [
    "BackendLauncher",
    "LlamaCppLauncher",
    "ManagedRuntime",
    "OllamaLauncher",
    "ProfileRuntimeSwitch",
    "RuntimeSwitchError",
    "server_arguments",
    "warm_up",
]
