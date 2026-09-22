"""Troca de modelo entre braços de bake-off.

Um experimento que compara modelos roda os braços em sequência, e cada braço
precisa do seu modelo carregado sozinho na placa. Esta é a única peça da bancada
que conhece mais de um RuntimeProfile ao mesmo tempo; o resto do runner recebe o
runtime já pronto e não sabe que houve troca.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from ..config import RuntimeBackend, RuntimeProfileConfig
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
    "ManagedRuntime",
    "OllamaLauncher",
    "ProfileRuntimeSwitch",
    "RuntimeSwitchError",
    "warm_up",
]
