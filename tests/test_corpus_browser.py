"""A rota de browser da coleta: cada requisição do Chromium passa pelo guard.

Não sobe browser nenhum. O que se prova aqui é o contrato que o Scrapling recebe
— o handler de `page.route`, a recusa de WebSocket e a conferência de que o guard
foi instalado — com um fetcher dublê que faz o que o Scrapling faz.
"""

import asyncio
import socket
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import cast

import pytest

from harness.config import load_config
from harness.corpus_browser import BrowserRenderError, ScraplingBrowserRenderer
from harness.corpus_scraper import Scraper
from harness.web_tools import EgressGuard, GuardedResolver, HttpResponse, ResolvedTarget

_ADDRESSES = {
    "site.test": "93.184.216.34",
    "cdn.test": "93.184.216.35",
    "intranet.test": "10.0.0.5",
    "metadata.test": "169.254.169.254",
}


async def _lookup(
    host: str, port: int, family: int, type_: int
) -> Sequence[tuple[int, int, int, str, tuple[str, int]]]:
    del family, type_
    address = _ADDRESSES.get(host)
    if address is None:
        raise OSError("unknown host")
    return ((socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port)),)


def _guard() -> EgressGuard:
    return EgressGuard(GuardedResolver(lookup=_lookup))


class _Request:
    def __init__(self, url: str) -> None:
        self.url = url


class _Route:
    def __init__(self, url: str) -> None:
        self.request = _Request(url)
        self.outcome: str | None = None

    async def abort(self, error_code: str | None = None) -> None:
        self.outcome = f"abort:{error_code}"

    async def fallback(self) -> None:
        self.outcome = "fallback"


class _Socket:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _Page:
    def __init__(self) -> None:
        self.routes: list[Callable[[_Route], Awaitable[None]]] = []
        self.sockets: list[Callable[[_Socket], Awaitable[None]]] = []

    async def route(self, url: str, handler: Callable[[_Route], Awaitable[None]]) -> None:
        del url
        self.routes.append(handler)

    async def route_web_socket(
        self, url: str, handler: Callable[[_Socket], Awaitable[None]]
    ) -> None:
        del url
        self.sockets.append(handler)


class _Rendered:
    def __init__(self, url: str, status: int = 200) -> None:
        self.url = url
        self.status = status
        self.html_content = "<html><body><p>conteudo montado por javascript</p></body></html>"


class _Fetcher:
    """Faz o que o Scrapling faz: chama page_setup e pede os subrecursos."""

    def __init__(
        self,
        subresources: Sequence[str] = (),
        *,
        run_setup: bool = True,
        final_url: str | None = None,
    ) -> None:
        self._subresources = subresources
        self._run_setup = run_setup
        self._final_url = final_url
        self.routes: list[_Route] = []
        self.socket = _Socket()

    async def async_fetch(self, url: str, **kwargs: object) -> _Rendered:
        page = _Page()
        setup = cast(Callable[[_Page], Awaitable[None]], kwargs["page_setup"])
        if self._run_setup:
            await setup(page)
        for resource in (url, *self._subresources):
            route = _Route(resource)
            for handler in reversed(page.routes):
                await handler(route)
            self.routes.append(route)
        for handler in page.sockets:
            await handler(self.socket)
        return _Rendered(self._final_url or url)


def _render(fetcher: _Fetcher, url: str = "https://site.test/app") -> str:
    renderer = ScraplingBrowserRenderer(guard=_guard(), fetcher=fetcher)
    return asyncio.run(renderer.render(url))


def test_every_browser_request_passes_the_guard() -> None:
    fetcher = _Fetcher(
        (
            "https://cdn.test/app.js",
            "http://intranet.test/admin",
            "http://metadata.test/latest/meta-data/",
            "data:image/png;base64,AAAA",
        )
    )

    html = _render(fetcher)

    outcomes = {route.request.url: route.outcome for route in fetcher.routes}
    assert "conteudo montado" in html
    assert outcomes["https://site.test/app"] == "fallback"
    assert outcomes["https://cdn.test/app.js"] == "fallback"
    assert outcomes["http://intranet.test/admin"] == "abort:blockedbyclient"
    assert outcomes["http://metadata.test/latest/meta-data/"] == "abort:blockedbyclient"
    assert outcomes["data:image/png;base64,AAAA"] == "fallback"
    assert fetcher.socket.closed


def test_a_page_rendered_without_the_guard_is_discarded() -> None:
    """O Scrapling engole exceção do page_setup e navega assim mesmo."""
    with pytest.raises(BrowserRenderError) as raised:
        _render(_Fetcher(run_setup=False))

    assert raised.value.code == "egress_guard_not_installed"


def test_a_private_seed_or_final_url_is_refused() -> None:
    with pytest.raises(BrowserRenderError) as seed:
        _render(_Fetcher(), "http://intranet.test/")
    with pytest.raises(BrowserRenderError) as redirected:
        _render(_Fetcher(final_url="http://intranet.test/painel"))

    assert seed.value.code == "egress_refused"
    assert redirected.value.code == "egress_refused"


class _Transport:
    def __init__(self, pages: Mapping[str, bytes]) -> None:
        self._pages = pages

    async def request(
        self,
        target: ResolvedTarget,
        *,
        headers: Mapping[str, str],
        max_bytes: int,
        timeout_seconds: float,
    ) -> HttpResponse:
        del headers, max_bytes, timeout_seconds
        body = self._pages.get(target.url.split("?")[0])
        if body is None:
            return HttpResponse(status=404, headers={"Content-Type": "text/plain"}, body=b"")
        return HttpResponse(status=200, headers={"Content-Type": "text/html"}, body=body)


class _Renderer:
    def __init__(self, *, fail: bool = False) -> None:
        self.rendered: list[str] = []
        self._fail = fail

    async def render(self, url: str) -> str:
        self.rendered.append(url)
        if self._fail:
            raise BrowserRenderError("http_error", "boom")
        return "<html><body><p>" + "texto renderizado " * 30 + "</p></body></html>"


def _collect(renderer: _Renderer, mode: str) -> list[bytes]:
    config = load_config().corpus.scraper
    crawl = config.html_crawl.model_copy(
        update={
            "prefer_sitemap": False,
            "respect_robots_txt": False,
            "max_depth": 0,
            "browser_escalation": config.html_crawl.browser_escalation.model_copy(
                update={"mode": mode}
            ),
        }
    )

    async def no_sleep(seconds: float) -> None:
        del seconds

    scraper = Scraper(
        config=config.model_copy(update={"html_crawl": crawl}),
        egress_guard=_guard(),
        transport=_Transport(
            {
                "https://site.test/casca": b"<html><body><div id=app></div></body></html>",
                "https://site.test/rica": (
                    b"<html><body><p>" + b"texto que o http ja trouxe " * 20 + b"</p></body></html>"
                ),
            }
        ),
        sleep=no_sleep,
        renderer=renderer,
    )

    async def scenario() -> list[bytes]:
        pages: list[bytes] = []
        for seed in ("https://site.test/casca", "https://site.test/rica"):
            plan = await scraper.plan(seed)
            pages.extend([page.data async for page in scraper.collect(plan)])
        return pages

    return asyncio.run(scenario())


def test_only_the_thin_page_pays_for_a_browser() -> None:
    renderer = _Renderer()

    shell, rich = _collect(renderer, "symptom")

    assert renderer.rendered == ["https://site.test/casca"]
    assert b"texto renderizado" in shell
    assert b"texto que o http ja trouxe" in rich


def test_escalation_off_or_failing_keeps_the_http_page() -> None:
    disabled = _Renderer()
    failing = _Renderer(fail=True)

    off = _collect(disabled, "disabled")
    kept = _collect(failing, "symptom")

    assert disabled.rendered == []
    assert b"id=app" in off[0]
    assert failing.rendered == ["https://site.test/casca"]
    assert b"id=app" in kept[0]
