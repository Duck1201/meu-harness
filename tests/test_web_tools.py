import asyncio
import json
import socket
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pytest

from harness import (
    AiohttpHttpTransport,
    BrowserPage,
    EgressGuard,
    EgressPolicyError,
    EngineReadiness,
    Grant,
    GuardedResolver,
    HttpResponse,
    ResolvedAddress,
    ResolvedTarget,
    SessionPolicy,
    ToolCall,
    WebToolExecutor,
    load_config,
)


class FakeLookup:
    def __init__(self, addresses: Sequence[tuple[int, str]]) -> None:
        self.addresses = addresses
        self.calls: list[tuple[str, int]] = []

    async def __call__(
        self,
        host: str,
        port: int,
        family: int,
        type_: int,
    ) -> Sequence[tuple[int, int, int, str, tuple[str, int] | tuple[str, int, int, int]]]:
        del family, type_
        self.calls.append((host, port))
        return tuple(
            (
                address_family,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (address, port) if address_family == socket.AF_INET else (address, port, 0, 0),
            )
            for address_family, address in self.addresses
        )


class FakeHttpTransport:
    def __init__(self, responses: Sequence[HttpResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[ResolvedTarget, Mapping[str, str]]] = []

    async def request(
        self,
        target: ResolvedTarget,
        *,
        headers: Mapping[str, str],
        max_bytes: int,
        timeout_seconds: float,
    ) -> HttpResponse:
        del max_bytes, timeout_seconds
        self.requests.append((target, headers))
        return self.responses.pop(0)


class SlowHttpTransport:
    async def request(
        self,
        target: ResolvedTarget,
        *,
        headers: Mapping[str, str],
        max_bytes: int,
        timeout_seconds: float,
    ) -> HttpResponse:
        del target, headers, max_bytes, timeout_seconds
        await asyncio.sleep(1)
        raise AssertionError("The executor timeout must cancel the transport")


class FakeBrowser:
    def __init__(self, page: BrowserPage) -> None:
        self.page = page
        self.urls: list[str] = []

    async def fetch(
        self,
        url: str,
        *,
        max_bytes: int,
        timeout_seconds: float,
    ) -> BrowserPage:
        del max_bytes, timeout_seconds
        self.urls.append(url)
        return self.page


class FakeBrowserGuard:
    def __init__(self, ready: bool) -> None:
        self.ready = ready

    async def readiness(self) -> EngineReadiness:
        return EngineReadiness(ready=self.ready, reason_code=None if self.ready else "not_ready")


def web_policy(*permissions: str) -> SessionPolicy:
    now = datetime.now(UTC)
    return SessionPolicy(
        conversation_id="conversation-1",
        grants=tuple(
            Grant(
                id=f"grant-{permission}",
                conversation_id="conversation-1",
                permission=permission,
                scope="public-web",
                granted_at=now,
            )
            for permission in permissions
        ),
    )


def public_guard() -> EgressGuard:
    return EgressGuard(GuardedResolver(lookup=FakeLookup(((socket.AF_INET, "93.184.216.34"),))))


def test_egress_guard_rejects_non_http_and_userinfo() -> None:
    async def scenario() -> None:
        guard = public_guard()
        for url, expected_code in {
            "ftp://example.com/file": "disallowed_url_scheme",
            "https://operator:secret@example.com/": "url_userinfo_not_allowed",
        }.items():
            with pytest.raises(EgressPolicyError) as error:
                await guard.resolve(url)
            assert error.value.code == expected_code

    asyncio.run(scenario())


def test_egress_guard_lets_local_and_private_destinations_through() -> None:
    # O harness tem um Operator só, na própria máquina: a rede dele (outros
    # containers, o roteador, a LAN) é destino legítimo, não ameaça.
    async def scenario() -> None:
        for url, address in {
            "http://localhost:8080/": "127.0.0.1",
            "http://127.0.0.1:18789/": "127.0.0.1",
            "http://192.168.0.1/": "192.168.0.1",
            "http://[::1]/": "::1",
        }.items():
            family = socket.AF_INET6 if ":" in address else socket.AF_INET
            guard = EgressGuard(GuardedResolver(lookup=FakeLookup(((family, address),))))
            target = await guard.resolve(url)
            assert [item.host for item in target.addresses] == [address]

    asyncio.run(scenario())


def test_web_preflight_requires_grant_and_validates_registry_schema() -> None:
    async def scenario() -> None:
        transport = FakeHttpTransport(())
        missing_grant = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy(),
            egress_guard=public_guard(),
            http_transport=transport,
        )
        granted = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=transport,
        )

        grant_result = await missing_grant.preflight(
            (ToolCall(id="fetch", name="web_fetch", arguments={"url": "https://example.com"}),)
        )
        schema_result = await granted.preflight(
            (
                ToolCall(
                    id="search",
                    name="web_search",
                    arguments={"query": "harness", "unexpected": True},
                ),
            )
        )

        assert grant_result.allowed is False
        assert grant_result.reason_code == "web_access_grant_required"
        assert schema_result.allowed is False
        assert schema_result.reason_code == "invalid_tool_arguments"
        assert transport.requests == []

    asyncio.run(scenario())


def test_web_fetch_revalidates_redirect_before_following_it() -> None:
    async def scenario() -> None:
        transport = FakeHttpTransport(
            (HttpResponse(status=302, headers={"Location": "ftp://example.com/secret"}, body=b""),)
        )
        executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=transport,
        )

        result = await executor.execute(
            ToolCall(id="fetch", name="web_fetch", arguments={"url": "https://example.com"})
        )

        assert result.status.value == "blocked"
        assert result.error is not None
        assert result.error["code"] == "disallowed_url_scheme"
        assert len(transport.requests) == 1

    asyncio.run(scenario())


def test_web_fetch_extracts_html_once_and_discards_raw_markup() -> None:
    async def scenario() -> None:
        html = b"""
            <html><head><title>Useful reference</title><style>hidden style</style></head>
            <body><nav>hidden navigation</nav><main>
            <h1>Harness security reference</h1>
            <p>This public article contains a deliberately substantial explanation of guarded
            network access, deterministic fetching, and safe handling of untrusted web content.</p>
            <p>Read the <a href="/details">detailed implementation notes</a> for examples.</p>
            <script>hidden script</script><aside>hidden aside</aside></main>
            <footer>hidden footer</footer></body></html>
        """
        transport = FakeHttpTransport(
            (
                HttpResponse(
                    status=200,
                    headers={"Content-Type": "text/html; charset=utf-8"},
                    body=html,
                ),
            )
        )
        executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=transport,
        )
        call = ToolCall(
            id="fetch",
            name="web_fetch",
            arguments={"url": "https://example.com/article", "max_chars": 12000},
        )

        result = await executor.execute(call)

        assert result.status.value == "success"
        assert isinstance(result.data, Mapping)
        content = result.data["content"]
        assert isinstance(content, str)
        assert "Harness security reference" in content
        assert "detailed implementation notes (https://example.com/details)" in content
        assert all(
            hidden not in content
            for hidden in (
                "hidden style",
                "hidden navigation",
                "hidden script",
                "hidden aside",
                "hidden footer",
            )
        )
        assert "<html" not in repr(result)
        assert result.meta == {
            "producer": "web_fetch",
            "truncated": False,
            "taints": ["UntrustedWebTaint"],
            "final_url": "https://example.com/article",
            "cache_hit": False,
        }

    asyncio.run(scenario())


def test_web_fetch_cuts_an_oversized_body_and_reuses_exact_url_cache() -> None:
    async def scenario() -> None:
        oversized = b"A page far larger than the body cap still has readable text on top. " * 5
        oversized_transport = FakeHttpTransport(
            (HttpResponse(status=200, headers={"Content-Type": "text/plain"}, body=oversized),)
        )
        oversized_executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=oversized_transport,
            max_response_bytes=200,
        )
        large = await oversized_executor.execute(
            ToolCall(id="large", name="web_fetch", arguments={"url": "https://example.com/large"})
        )
        # Uma página maior que o cap é cortada, não descartada: o que coube
        # continua chegando ao modelo.
        assert large.status.value == "success"
        assert isinstance(large.data, Mapping)
        assert large.data["content"] == oversized[:200].decode()

        content = (
            b"A sufficiently long plain text response demonstrates exact URL caching without "
            b"making a second HTTP request during the same executor instance and Turn."
        )
        cached_transport = FakeHttpTransport(
            (HttpResponse(status=200, headers={"Content-Type": "text/plain"}, body=content),)
        )
        cached_executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=cached_transport,
        )
        first = await cached_executor.execute(
            ToolCall(id="first", name="web_fetch", arguments={"url": "https://example.com/cache"})
        )
        second = await cached_executor.execute(
            ToolCall(id="second", name="web_fetch", arguments={"url": "https://example.com/cache"})
        )

        assert first.status.value == "success"
        assert second.status.value == "success"
        assert second.tool_call_id == "second"
        assert second.meta["cache_hit"] is True
        assert len(cached_transport.requests) == 1

    asyncio.run(scenario())


def test_short_extraction_escalates_and_says_so_when_no_browser_is_configured() -> None:
    async def scenario() -> None:
        short_executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=FakeHttpTransport(
                (
                    HttpResponse(
                        status=200,
                        headers={"Content-Type": "text/html"},
                        body=b"<p>short</p>",
                    ),
                )
            ),
        )
        short = await short_executor.execute(
            ToolCall(id="short", name="web_fetch", arguments={"url": "https://example.com/short"})
        )
        # An extraction under the calibrated threshold is the declared escalation
        # symptom, so the executor asks for a browser instead of giving up on HTTP.
        assert short.status.value == "failed"
        assert short.error is not None
        assert short.error["code"] == "browser_escalation_unavailable"

        unavailable_executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=FakeHttpTransport(
                (
                    HttpResponse(
                        status=403,
                        headers={"Content-Type": "text/html"},
                        body=b"challenge",
                    ),
                )
            ),
        )
        unavailable = await unavailable_executor.execute(
            ToolCall(
                id="browser",
                name="web_fetch",
                arguments={"url": "https://example.com/challenge"},
            )
        )
        assert unavailable.status.value == "failed"
        assert unavailable.error is not None
        assert unavailable.error["code"] == "browser_escalation_unavailable"

    asyncio.run(scenario())


def test_web_fetch_enforces_redirect_content_type_and_character_limits() -> None:
    async def scenario() -> None:
        redirect_transport = FakeHttpTransport(
            (
                HttpResponse(status=302, headers={"Location": "/one"}, body=b""),
                HttpResponse(status=302, headers={"Location": "/two"}, body=b""),
            )
        )
        redirect_executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=redirect_transport,
            max_redirects=1,
        )
        redirected = await redirect_executor.execute(
            ToolCall(
                id="redirects",
                name="web_fetch",
                arguments={"url": "https://example.com/start"},
            )
        )
        assert redirected.status.value == "blocked"
        assert redirected.error is not None
        assert redirected.error["code"] == "redirect_limit_exceeded"
        assert len(redirect_transport.requests) == 2
        assert len({headers["User-Agent"] for _, headers in redirect_transport.requests}) == 1

        unsupported_executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=FakeHttpTransport(
                (
                    HttpResponse(
                        status=200,
                        headers={"Content-Type": "image/png"},
                        body=b"not an image",
                    ),
                )
            ),
        )
        unsupported = await unsupported_executor.execute(
            ToolCall(
                id="content-type",
                name="web_fetch",
                arguments={"url": "https://example.com/image"},
            )
        )
        assert unsupported.status.value == "failed"
        assert unsupported.error is not None
        assert unsupported.error["code"] == "unsupported_content_type"

        long_content = b"x" * 140
        limited_executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=FakeHttpTransport(
                (
                    HttpResponse(
                        status=200,
                        headers={"Content-Type": "text/plain"},
                        body=long_content,
                    ),
                )
            ),
        )
        limited = await limited_executor.execute(
            ToolCall(
                id="character-limit",
                name="web_fetch",
                arguments={"url": "https://example.com/long", "max_chars": 100},
            )
        )
        assert limited.status.value == "success"
        assert isinstance(limited.data, Mapping)
        assert limited.data["content"] == "x" * 100
        # Truncou: o sinal é esse. Não há parâmetro de offset para prometer.
        assert limited.meta["truncated"] is True
        assert "continuation" not in limited.meta

    asyncio.run(scenario())


def test_web_search_reads_searxng_limits_results_and_normalizes_cache_key() -> None:
    async def scenario() -> None:
        provider_body = json.dumps(
            {
                "query": "harness security",
                "results": [
                    {"title": "One", "url": "https://one.example", "content": "First"},
                    {"title": "Two", "url": "https://two.example", "content": "Second"},
                    {"title": "Three", "url": "https://three.example", "content": "Third"},
                ],
            }
        ).encode()
        transport = FakeHttpTransport(
            (
                HttpResponse(
                    status=200,
                    headers={"Content-Type": "application/json"},
                    body=provider_body,
                ),
            )
        )
        executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=transport,
            search_endpoint="https://searx.example/search",
        )

        first = await executor.execute(
            ToolCall(
                id="search-1",
                name="web_search",
                arguments={"query": "  Harness   Security  ", "limit": 2},
            )
        )
        second = await executor.execute(
            ToolCall(
                id="search-2",
                name="web_search",
                arguments={"query": "harness security", "limit": 2},
            )
        )

        assert first.status.value == "success"
        assert isinstance(first.data, Mapping)
        results = first.data["results"]
        assert isinstance(results, Sequence)
        assert len(results) == 2
        assert first.meta == {
            "producer": "web_search",
            "truncated": True,
            "taints": ["UntrustedWebTaint"],
            "engine": "searxng",
            "cache_hit": False,
        }
        assert second.tool_call_id == "search-2"
        assert second.meta["cache_hit"] is True
        assert len(transport.requests) == 1
        target, headers = transport.requests[0]
        assert target.url.startswith("https://searx.example/search?")
        assert "q=Harness+Security" in target.url
        assert "format=json" in target.url
        assert "X-Subscription-Token" not in headers
        assert headers["User-Agent"].startswith("Harness/2.0")

    asyncio.run(scenario())


def test_web_search_falls_back_to_duckduckgo_when_searxng_does_not_answer() -> None:
    async def scenario() -> None:
        lite_html = (
            b"<html><body><table>"
            b'<tr><td><a class="result-link" href="https://one.example">One</a></td></tr>'
            b'<tr><td class="result-snippet">Primeiro resultado.</td></tr>'
            b'<tr><td><a class="result-link" href="https://two.example">Two</a></td></tr>'
            b'<tr><td class="result-snippet">Segundo resultado.</td></tr>'
            b"</table></body></html>"
        )
        transport = FakeHttpTransport(
            (
                HttpResponse(status=502, headers={"Content-Type": "text/html"}, body=b"nope"),
                HttpResponse(status=200, headers={"Content-Type": "text/html"}, body=lite_html),
            )
        )
        executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=transport,
            search_endpoint="https://searx.example/search",
        )

        result = await executor.execute(
            ToolCall(id="search-1", name="web_search", arguments={"query": "harness", "limit": 1})
        )

        assert result.status.value == "success"
        assert result.meta["engine"] == "duckduckgo"
        assert result.meta["truncated"] is True
        assert isinstance(result.data, Mapping)
        assert result.data["results"] == [
            {
                "title": "One",
                "url": "https://one.example",
                "description": "Primeiro resultado.",
            }
        ]
        assert len(transport.requests) == 2
        assert transport.requests[1][0].url.startswith("https://lite.duckduckgo.com/lite/?")

    asyncio.run(scenario())


def test_web_search_without_any_provider_reachable_is_failed_not_blocked() -> None:
    async def scenario() -> None:
        transport = FakeHttpTransport(
            (HttpResponse(status=503, headers={"Content-Type": "text/html"}, body=b""),)
        )
        executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=transport,
        )

        result = await executor.execute(
            ToolCall(id="search-1", name="web_search", arguments={"query": "harness"})
        )

        # Nobody answered, which is a provider failure — never a harness decision.
        assert result.status.value == "failed"
        assert result.retryable is True
        assert result.error is not None
        assert result.error["code"] == "provider_unavailable"

    asyncio.run(scenario())


def test_aiohttp_transport_connects_to_pinned_address_without_second_dns_lookup() -> None:
    async def scenario() -> None:
        received = bytearray()

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            received.extend(await reader.readuntil(b"\r\n\r\n"))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 12\r\n"
                b"Connection: close\r\n\r\npinned route"
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        socket_address = server.sockets[0].getsockname()
        port = int(socket_address[1])
        target = ResolvedTarget(
            url=f"http://does-not-resolve.invalid:{port}/resource",
            hostname="does-not-resolve.invalid",
            port=port,
            addresses=(
                ResolvedAddress(
                    host="127.0.0.1",
                    port=port,
                    family=socket.AF_INET,
                    proto=socket.IPPROTO_TCP,
                ),
            ),
        )
        try:
            response = await AiohttpHttpTransport().request(
                target,
                headers={"User-Agent": "Harness/2.0 test"},
                max_bytes=1024,
                timeout_seconds=2,
            )
        finally:
            server.close()
            await server.wait_closed()

        assert response.status == 200
        assert response.body == b"pinned route"
        assert b"Host: does-not-resolve.invalid:" in received
        assert b"User-Agent: Harness/2.0 test" in received
        assert b"Cookie:" not in received

    asyncio.run(scenario())


def test_web_response_timeout_is_enforced_around_injected_transport() -> None:
    async def scenario() -> None:
        executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=SlowHttpTransport(),
            timeout_seconds=0.01,
        )

        result = await executor.execute(
            ToolCall(id="slow", name="web_fetch", arguments={"url": "https://example.com/slow"})
        )

        assert result.status.value == "failed"
        assert result.retryable is True
        assert result.error is not None
        assert result.error["code"] == "response_timeout"

    asyncio.run(scenario())


def test_browser_capability_opens_only_after_browser_egress_guard_is_ready() -> None:
    async def scenario() -> None:
        page = BrowserPage(
            final_url="https://example.com/rendered",
            content_type="text/html",
            body=(
                b"<main><h1>Rendered article</h1><p>The guarded browser returned enough useful "
                b"content to demonstrate that extraction remains shared with the ordinary HTTP "
                b"path and that raw markup never enters the ToolResult.</p></main>"
            ),
        )
        browser = FakeBrowser(page)
        not_ready = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=FakeHttpTransport(
                (
                    HttpResponse(
                        status=403,
                        headers={"Content-Type": "text/html"},
                        body=b"challenge",
                    ),
                )
            ),
            browser_capability=browser,
            browser_egress_guard=FakeBrowserGuard(False),
        )
        denied = await not_ready.execute(
            ToolCall(id="denied", name="web_fetch", arguments={"url": "https://example.com"})
        )
        assert denied.status.value == "failed"
        assert denied.error is not None
        assert denied.error["code"] == "browser_escalation_unavailable"
        assert browser.urls == []

        ready = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=FakeHttpTransport(
                (
                    HttpResponse(
                        status=403,
                        headers={"Content-Type": "text/html"},
                        body=b"challenge",
                    ),
                )
            ),
            browser_capability=browser,
            browser_egress_guard=FakeBrowserGuard(True),
        )
        rendered = await ready.execute(
            ToolCall(id="ready", name="web_fetch", arguments={"url": "https://example.com"})
        )
        assert rendered.status.value == "success"
        assert rendered.meta["producer"] == "browser"
        assert browser.urls == ["https://example.com"]
        assert "<main" not in repr(rendered)

    asyncio.run(scenario())


def test_get_weather_resolves_the_place_then_reads_the_forecast() -> None:
    async def scenario() -> None:
        geocoding = json.dumps(
            {
                "results": [
                    {
                        "name": "Recife",
                        "latitude": -8.05,
                        "longitude": -34.9,
                        "country": "Brazil",
                        "timezone": "America/Recife",
                    }
                ]
            }
        ).encode()
        forecast = json.dumps(
            {
                "current": {"temperature_2m": 29.4, "weather_code": 2},
                "current_units": {"temperature_2m": "°C"},
                "daily": {"time": ["2026-08-12"], "temperature_2m_max": [31.0]},
                "daily_units": {"temperature_2m_max": "°C"},
            }
        ).encode()
        transport = FakeHttpTransport(
            (
                HttpResponse(
                    status=200, headers={"Content-Type": "application/json"}, body=geocoding
                ),
                HttpResponse(
                    status=200, headers={"Content-Type": "application/json"}, body=forecast
                ),
            )
        )
        executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=transport,
        )

        result = await executor.execute(
            ToolCall(
                id="weather-1",
                name="get_weather",
                arguments={"location": "Recife", "days": 1},
            )
        )

        assert result.status.value == "success"
        assert isinstance(result.data, Mapping)
        place = result.data["place"]
        assert isinstance(place, Mapping)
        assert place["latitude"] == -8.05
        assert place["country"] == "Brazil"
        assert result.meta["taints"] == ["UntrustedWebTaint"]
        assert result.meta["producer"] == "open_meteo"
        assert len(transport.requests) == 2
        assert "latitude=-8.05" in transport.requests[1][0].url

    asyncio.run(scenario())


def test_get_weather_without_a_grant_is_blocked_and_an_unknown_place_is_empty() -> None:
    async def scenario() -> None:
        ungranted = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy(),
            egress_guard=public_guard(),
            http_transport=FakeHttpTransport(()),
        )
        blocked = await ungranted.execute(
            ToolCall(id="weather-1", name="get_weather", arguments={"location": "Recife"})
        )

        assert blocked.status.value == "blocked"
        assert blocked.error is not None
        assert blocked.error["code"] == "web_access_grant_required"

        empty_executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=FakeHttpTransport(
                (
                    HttpResponse(
                        status=200,
                        headers={"Content-Type": "application/json"},
                        body=b'{"generationtime_ms": 0.1}',
                    ),
                )
            ),
        )
        empty = await empty_executor.execute(
            ToolCall(id="weather-2", name="get_weather", arguments={"location": "Atlantis"})
        )

        assert empty.status.value == "empty"
        assert empty.error is None

    asyncio.run(scenario())


def test_link_menus_do_not_crowd_out_the_readable_text() -> None:
    async def scenario() -> None:
        menu = "".join(
            f'<li><a href="https://x{index}.example.com/">Idioma {index}</a></li>'
            for index in range(60)
        )
        body = (
            "<html><body><ul>"
            + menu
            + '</ul><p>Brasília é a capital federal do Brasil, com <a href="'
            'https://example.com/page#nota">nota</a> e <a href="https://outro.example.com/">um '
            "destino real</a> no meio do texto corrido que o modelo precisa mesmo ler.</p>"
            "</body></html>"
        ).encode()
        executor = WebToolExecutor(
            registry=load_config().tool_registry,
            session_policy=web_policy("WebAccessGrant"),
            egress_guard=public_guard(),
            http_transport=FakeHttpTransport(
                (HttpResponse(status=200, headers={"Content-Type": "text/html"}, body=body),)
            ),
        )
        result = await executor.execute(
            ToolCall(id="menu", name="web_fetch", arguments={"url": "https://example.com/page"})
        )

        assert result.status.value == "success"
        assert isinstance(result.data, Mapping)
        content = result.data["content"]
        assert isinstance(content, str)
        # O menu de idiomas some, a âncora para a própria página não vira URL e
        # o destino externo continua disponível junto do texto.
        assert "Idioma 30" not in content
        assert "#nota" not in content
        assert "https://outro.example.com/" in content
        assert "capital federal do Brasil" in content

    asyncio.run(scenario())
