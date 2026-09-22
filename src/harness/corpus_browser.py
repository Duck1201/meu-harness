"""Rota de browser da coleta de Corpus, sobre o Scrapling (ADR 0014).

Um site que monta o conteúdo em JavaScript chega ao crawl HTTP como casca: o
HTML vem, o texto não. Para essa página, e só para ela, o scraper pede uma
renderização a este módulo, que abre um Chromium efêmero pelo Scrapling.

O risco é o browser. Uma página renderizada pede dezenas de subrecursos que o
Operator nunca viu, e qualquer um pode apontar para a rede interna. Por isso toda
requisição do browser passa pelo mesmo EgressGuard da rota HTTP, registrado em
`page.route` antes da navegação; WebSocket é recusado; e a instalação do guard é
conferida depois do fetch — o Scrapling engole exceção do `page_setup` e navega
assim mesmo, então sem a conferência o guard falharia aberto.

O que fica fora, e está em ADR 0014: a resolução do browser é dele, então entre a
validação e a conexão há a mesma janela de DNS rebinding que `RELEASE-PENDING.md`
já registra, e requisição de service worker não passa por `page.route`.
"""

import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, cast
from urllib.parse import urlsplit

from .web_tools import EgressGuard, EgressPolicyError, EgressResolutionError


class _Request(Protocol):
    @property
    def url(self) -> str: ...


class BrowserRoute(Protocol):
    """O pedaço de `playwright.async_api.Route` que o guard usa."""

    @property
    def request(self) -> _Request: ...

    async def abort(self, error_code: str | None = None) -> None: ...

    async def fallback(self) -> None: ...


class _WebSocketRoute(Protocol):
    async def close(self) -> None: ...


class BrowserPage(Protocol):
    async def route(self, url: str, handler: Callable[[BrowserRoute], Awaitable[None]]) -> None: ...

    async def route_web_socket(
        self, url: str, handler: Callable[[_WebSocketRoute], Awaitable[None]]
    ) -> None: ...


class _Rendered(Protocol):
    @property
    def status(self) -> int: ...

    @property
    def url(self) -> str: ...

    @property
    def html_content(self) -> str: ...


class _BrowserFetcher(Protocol):
    async def async_fetch(self, url: str, **kwargs: object) -> _Rendered: ...


# Esquemas que não saem para a rede: o próprio documento e o que ele gera em memória.
_LOCAL_SCHEMES = frozenset({"data", "blob", "about"})


@dataclass(slots=True)
class _GuardState:
    installed: bool = False
    refused: int = 0


class EgressVettingRoute:
    """O handler de `page.route`: cada requisição do browser passa pelo guard."""

    def __init__(self, guard: EgressGuard, state: _GuardState) -> None:
        self._guard = guard
        self._state = state

    async def __call__(self, route: BrowserRoute) -> None:
        url = route.request.url
        scheme = urlsplit(url).scheme.casefold()
        if scheme in _LOCAL_SCHEMES:
            await route.fallback()
            return
        try:
            await self._guard.resolve(url)
        except (EgressPolicyError, EgressResolutionError):
            self._state.refused += 1
            await route.abort("blockedbyclient")
            return
        # `fallback`, não `continue_`: os bloqueios que o próprio Scrapling
        # registrou (recursos pesados, domínios de anúncio) ainda têm a vez deles.
        await route.fallback()


class BrowserRenderError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ScraplingBrowserRenderer:
    """Renderiza uma URL num Chromium efêmero do Scrapling, atrás do EgressGuard.

    `stealth` escolhe o StealthyFetcher, que passa por proteção anti-bot; ele só
    serve a coleta que o Operator dispara, e os termos do site continuam valendo.
    """

    def __init__(
        self,
        *,
        guard: EgressGuard | None = None,
        stealth: bool = False,
        timeout_milliseconds: int = 30_000,
        fetcher: _BrowserFetcher | None = None,
    ) -> None:
        self._guard = guard or EgressGuard()
        self._stealth = stealth
        self._timeout = timeout_milliseconds
        self._fetcher = fetcher

    async def render(self, url: str) -> str:
        try:
            await self._guard.resolve(url)
        except (EgressPolicyError, EgressResolutionError) as error:
            raise BrowserRenderError("egress_refused", str(error)) from error
        state = _GuardState()
        vetting = EgressVettingRoute(self._guard, state)

        async def refuse_socket(socket: _WebSocketRoute) -> None:
            await socket.close()

        async def page_setup(page: BrowserPage) -> None:
            await page.route("**/*", vetting)
            await page.route_web_socket("**/*", refuse_socket)
            state.installed = True

        fetcher = self._fetcher or _load_fetcher(stealth=self._stealth)
        rendered = await fetcher.async_fetch(
            url,
            headless=True,
            page_setup=page_setup,
            disable_resources=True,
            network_idle=True,
            timeout=self._timeout,
        )
        if not state.installed:
            raise BrowserRenderError(
                "egress_guard_not_installed",
                "O browser navegou sem o guard de rede; a página foi descartada.",
            )
        # O documento final também passa pelo guard: um redirect feito pelo
        # próprio browser não voltou por `_get`.
        try:
            await self._guard.resolve(rendered.url or url)
        except (EgressPolicyError, EgressResolutionError) as error:
            raise BrowserRenderError("egress_refused", str(error)) from error
        if rendered.status != 200:
            raise BrowserRenderError("http_error", f"O browser recebeu {rendered.status}.")
        return rendered.html_content


def _load_fetcher(*, stealth: bool) -> _BrowserFetcher:
    """Os fetchers de browser são um extra opcional (`uv sync --extra browser`)."""
    try:
        module = importlib.import_module("scrapling.fetchers")
    except ImportError as error:
        raise BrowserRenderError(
            "browser_route_unavailable",
            "A rota de browser precisa de `uv sync --extra browser` e `scrapling install`.",
        ) from error
    name = "StealthyFetcher" if stealth else "DynamicFetcher"
    return cast(_BrowserFetcher, getattr(module, name))


__all__ = [
    "BrowserPage",
    "BrowserRenderError",
    "BrowserRoute",
    "EgressVettingRoute",
    "ScraplingBrowserRenderer",
]
