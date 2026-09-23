from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
import unicodedata
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from typing import Any, Protocol, cast
from urllib.parse import SplitResult, urlencode, urljoin, urlsplit

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult
from jsonschema import Draft202012Validator, FormatChecker

from .config import ToolDefinitionConfig, ToolRegistryConfig
from .domain import (
    JsonValue,
    SessionPolicy,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    grant_reason_code,
)
from .ports import ConfirmationPreview, EngineReadiness, ToolBatchPreflight

type SocketAddress = tuple[str, int] | tuple[str, int, int, int]
type AddressInfo = tuple[int, int, int, str, SocketAddress]
type AddressLookup = Callable[[str, int, int, int], Awaitable[Sequence[AddressInfo]]]

_USER_AGENT = "Harness/2.0 (+https://localhost.invalid/harness)"
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_HTML_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_TEXT_CONTENT_TYPES = frozenset(
    {
        "application/json",
        "application/xml",
        "text/markdown",
        "text/plain",
        "text/xml",
    }
)
SUPPRESSED_HTML_TAGS = frozenset({"aside", "footer", "nav", "script", "style"})
_BOUNDARY_HTML_TAGS = frozenset(
    {
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "section",
        "tr",
    }
)


# The keyless fallback: no credential, no signup, so a search never fails just
# because nobody configured a provider. It is a constructor default so a bench can
# serve its own response instead of the corpus reaching the real internet.
DUCKDUCKGO_SEARCH_ENDPOINT = "https://lite.duckduckgo.com/lite/"

# Open-Meteo: no key, no signup, so the weather tool carries no credential path.
GEOCODING_ENDPOINT = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"


class EgressPolicyError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class EgressResolutionError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ResolvedAddress:
    host: str
    port: int
    family: int
    proto: int


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    url: str
    hostname: str
    port: int
    addresses: tuple[ResolvedAddress, ...]


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    truncated: bool = False


class HttpTransport(Protocol):
    async def request(
        self,
        target: ResolvedTarget,
        *,
        headers: Mapping[str, str],
        max_bytes: int,
        timeout_seconds: float,
    ) -> HttpResponse: ...


class _PinnedResolver(AbstractResolver):
    def __init__(self, target: ResolvedTarget) -> None:
        self._target = target

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        del family
        if host.casefold().rstrip(".") != self._target.hostname.casefold().rstrip(".") or (
            port and port != self._target.port
        ):
            raise OSError("Pinned resolver refused an unexpected destination.")
        return [
            ResolveResult(
                hostname=self._target.hostname,
                host=address.host,
                port=address.port,
                family=address.family,
                proto=address.proto,
                flags=0,
            )
            for address in self._target.addresses
        ]

    async def close(self) -> None:
        return None


class AiohttpHttpTransport:
    async def request(
        self,
        target: ResolvedTarget,
        *,
        headers: Mapping[str, str],
        max_bytes: int,
        timeout_seconds: float,
    ) -> HttpResponse:
        connector = aiohttp.TCPConnector(
            resolver=_PinnedResolver(target),
            use_dns_cache=False,
            force_close=True,
            limit=1,
        )
        timeout = aiohttp.ClientTimeout(total=timeout_seconds, connect=timeout_seconds)
        async with (
            aiohttp.ClientSession(
                connector=connector,
                cookie_jar=aiohttp.DummyCookieJar(),
                timeout=timeout,
                trust_env=False,
            ) as session,
            session.get(
                target.url,
                headers=headers,
                allow_redirects=False,
            ) as response,
        ):
            body = bytearray()
            truncated = False
            async for chunk in response.content.iter_chunked(64 * 1024):
                body.extend(chunk)
                if len(body) >= max_bytes:
                    # Página grande demais ainda é página: corta o corpo e segue,
                    # que o resultado sai truncado de qualquer jeito no `limit`.
                    del body[max_bytes:]
                    truncated = True
                    break
            return HttpResponse(
                status=response.status,
                headers=dict(response.headers),
                body=bytes(body),
                truncated=truncated,
            )


@dataclass(frozen=True, slots=True)
class BrowserPage:
    final_url: str
    content_type: str
    body: bytes


class BrowserCapability(Protocol):
    async def fetch(
        self,
        url: str,
        *,
        max_bytes: int,
        timeout_seconds: float,
    ) -> BrowserPage: ...


class BrowserEgressGuard(Protocol):
    async def readiness(self) -> EngineReadiness: ...


@dataclass(frozen=True, slots=True)
class _FetchArtifact:
    content: str
    content_type: str
    final_url: str
    producer: str


class _PreflightIssue(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class GuardedResolver:
    def __init__(
        self,
        *,
        lookup: AddressLookup | None = None,
    ) -> None:
        self._lookup = lookup or _system_lookup
        self._cache: dict[tuple[str, int], tuple[ResolvedAddress, ...]] = {}

    async def resolve(
        self,
        host: str,
        port: int,
        family: int = socket.AF_UNSPEC,
    ) -> tuple[ResolvedAddress, ...]:
        key = (host.casefold().rstrip("."), port)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            records = await self._lookup(host, port, family, socket.SOCK_STREAM)
        except (OSError, UnicodeError) as error:
            raise EgressResolutionError("Host resolution failed.") from error
        if not records:
            raise EgressResolutionError("Host resolution returned no addresses.")

        resolved: list[ResolvedAddress] = []
        seen: set[tuple[str, int, int, int]] = set()
        for address_family, _socket_type, protocol, _canonical_name, socket_address in records:
            address = socket_address[0]
            try:
                parsed = ipaddress.ip_address(address)
            except ValueError as error:
                raise EgressResolutionError(
                    "Host resolution returned an invalid address."
                ) from error
            item = ResolvedAddress(
                host=parsed.compressed,
                port=port,
                family=address_family,
                proto=protocol,
            )
            identity = (item.host, item.port, item.family, item.proto)
            if identity not in seen:
                seen.add(identity)
                resolved.append(item)
        approved = tuple(resolved)
        self._cache[key] = approved
        return approved


class EgressGuard:
    def __init__(
        self,
        resolver: GuardedResolver | None = None,
    ) -> None:
        # Destinos locais e privados são permitidos: o harness tem um único
        # Operator, na própria máquina, e a rede dele (outros containers, o
        # roteador, serviços da LAN) é justamente o que ele quer alcançar.
        self._resolver = resolver or GuardedResolver()

    async def resolve(self, url: str) -> ResolvedTarget:
        parsed = self.validate_url(url)
        hostname = parsed.hostname
        if hostname is None:
            raise EgressPolicyError("url_host_required", "The URL must include a hostname.")
        try:
            explicit_port = parsed.port
        except ValueError as error:
            raise EgressPolicyError("invalid_url_port", "The URL port is invalid.") from error
        port = explicit_port or (443 if parsed.scheme.casefold() == "https" else 80)
        addresses = await self._resolver.resolve(hostname, port)
        return ResolvedTarget(
            url=url,
            hostname=hostname,
            port=port,
            addresses=addresses,
        )

    def validate_url(self, url: str) -> SplitResult:
        try:
            parsed = urlsplit(url)
        except ValueError as error:
            raise EgressPolicyError("invalid_url", "The URL is invalid.") from error
        if parsed.scheme.casefold() not in {"http", "https"}:
            raise EgressPolicyError(
                "disallowed_url_scheme",
                "Only HTTP and HTTPS URLs are allowed.",
            )
        if parsed.username is not None or parsed.password is not None:
            raise EgressPolicyError(
                "url_userinfo_not_allowed",
                "URL userinfo is not allowed.",
            )
        hostname = parsed.hostname
        if hostname is None:
            raise EgressPolicyError("url_host_required", "The URL must include a hostname.")
        try:
            port = parsed.port
        except ValueError as error:
            raise EgressPolicyError("invalid_url_port", "The URL port is invalid.") from error
        if port == 0:
            raise EgressPolicyError("invalid_url_port", "The URL port is invalid.")
        return parsed


class WebToolExecutor:
    def __init__(
        self,
        *,
        registry: ToolRegistryConfig,
        session_policy: SessionPolicy,
        egress_guard: EgressGuard | None = None,
        http_transport: HttpTransport | None = None,
        search_endpoint: str | None = None,
        fallback_search_endpoint: str = DUCKDUCKGO_SEARCH_ENDPOINT,
        browser_capability: BrowserCapability | None = None,
        browser_egress_guard: BrowserEgressGuard | None = None,
        max_response_bytes: int = 2 * 1024 * 1024,
        max_redirects: int = 5,
        timeout_seconds: float = 15.0,
    ) -> None:
        if max_response_bytes < 1 or max_redirects < 0 or timeout_seconds <= 0:
            raise ValueError("Web limits must be positive.")
        self._registry = {
            definition.name: definition
            for definition in registry.model_tools
            if "data_egress" in definition.effects
        }
        self._effective_grants = session_policy.effective_grants
        self._egress_guard = egress_guard or EgressGuard()
        self._http_transport = http_transport or AiohttpHttpTransport()
        self._search_endpoint = search_endpoint
        self._fallback_search_endpoint = fallback_search_endpoint
        self._search_egress_guard = self._egress_guard
        self._browser_capability = browser_capability
        self._browser_egress_guard = browser_egress_guard
        self._max_response_bytes = max_response_bytes
        self._max_redirects = max_redirects
        self._timeout_seconds = timeout_seconds
        self._fetch_cache: dict[str, _FetchArtifact] = {}
        self._search_cache: dict[tuple[str, int], ToolResult] = {}

    async def preflight(self, calls: Sequence[ToolCall]) -> ToolBatchPreflight:
        seen_ids: set[str] = set()
        for raw_call in calls:
            call = self._normalized(raw_call)
            try:
                if not call.id or call.id in seen_ids:
                    raise _PreflightIssue(
                        "duplicate_tool_call_id",
                        "Tool call IDs must be non-empty and unique.",
                    )
                seen_ids.add(call.id)
                self._validate_call(call)
            except _PreflightIssue as issue:
                return ToolBatchPreflight(
                    allowed=False,
                    reason_code=issue.code,
                    detail=issue.detail,
                )
        return ToolBatchPreflight(allowed=True)

    async def execute(self, call: ToolCall) -> ToolResult:
        call = self._normalized(call)
        try:
            self._validate_call(call)
        except _PreflightIssue as issue:
            return _web_error(
                call,
                ToolResultStatus.BLOCKED,
                issue.code,
                issue.detail,
                retryable=False,
            )
        if call.name == "web_fetch":
            return await self._web_fetch(call)
        if call.name == "web_search":
            return await self._web_search(call)
        if call.name == "get_weather":
            return await self._get_weather(call)
        return _web_error(
            call,
            ToolResultStatus.FAILED,
            "tool_not_implemented",
            "Web tool is not implemented.",
            retryable=False,
        )

    async def _web_fetch(self, call: ToolCall) -> ToolResult:
        url = cast(str, call.arguments["url"])
        limit = cast(int, call.arguments.get("max_chars", 12000))
        cached = self._fetch_cache.get(url)
        if cached is not None:
            return _fetch_result(call, cached, limit=limit, cache_hit=True)

        current_url = url
        for redirect_count in range(self._max_redirects + 1):
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    target = await self._egress_guard.resolve(current_url)
            except EgressPolicyError as error:
                return _web_error(
                    call,
                    ToolResultStatus.BLOCKED,
                    error.code,
                    str(error),
                    retryable=False,
                    producer="web_fetch",
                    final_url=current_url,
                )
            except EgressResolutionError:
                return _web_error(
                    call,
                    ToolResultStatus.FAILED,
                    "dns_resolution_failed",
                    "The destination hostname could not be resolved.",
                    retryable=True,
                    producer="web_fetch",
                    final_url=current_url,
                )
            except TimeoutError:
                return _web_error(
                    call,
                    ToolResultStatus.FAILED,
                    "dns_resolution_timeout",
                    "The destination hostname resolution exceeded the time limit.",
                    retryable=True,
                    producer="web_fetch",
                    final_url=current_url,
                )
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    response = await self._http_transport.request(
                        target,
                        headers={
                            "Accept": "text/html, text/plain;q=0.9, */*;q=0.1",
                            "User-Agent": _USER_AGENT,
                        },
                        max_bytes=self._max_response_bytes,
                        timeout_seconds=self._timeout_seconds,
                    )
            except TimeoutError:
                return _web_error(
                    call,
                    ToolResultStatus.FAILED,
                    "response_timeout",
                    "The web response exceeded the time limit.",
                    retryable=True,
                    producer="web_fetch",
                    final_url=current_url,
                )
            except (OSError, aiohttp.ClientError):
                return _web_error(
                    call,
                    ToolResultStatus.FAILED,
                    "transport_unavailable",
                    "The web transport was unavailable.",
                    retryable=True,
                    producer="web_fetch",
                    final_url=current_url,
                )

            if len(response.body) > self._max_response_bytes:
                response = replace(response, body=response.body[: self._max_response_bytes])
            if response.status in _REDIRECT_STATUSES:
                location = _header(response.headers, "location")
                if location is None:
                    return _web_error(
                        call,
                        ToolResultStatus.FAILED,
                        "invalid_redirect",
                        "The redirect response did not include a Location header.",
                        retryable=False,
                        producer="web_fetch",
                        final_url=current_url,
                    )
                if redirect_count >= self._max_redirects:
                    return _web_error(
                        call,
                        ToolResultStatus.BLOCKED,
                        "redirect_limit_exceeded",
                        "The web response exceeded the redirect limit.",
                        retryable=False,
                        producer="web_fetch",
                        final_url=current_url,
                    )
                current_url = urljoin(current_url, location)
                continue
            if response.status in {401, 403}:
                return await self._browser_escalation(call, current_url, limit=limit)
            if response.status < 200 or response.status >= 300:
                return _http_status_error(call, response.status, final_url=current_url)

            artifact_or_error = _response_artifact(response, current_url, producer="web_fetch")
            if isinstance(artifact_or_error, tuple):
                code, message = artifact_or_error
                if code == "empty_extraction":
                    # The declared escalation symptom: HTTP returned a page whose readable
                    # extraction is below the calibrated threshold, which is what a
                    # JavaScript-rendered page looks like to an HTTP client.
                    return await self._browser_escalation(call, current_url, limit=limit)
                return _web_error(
                    call,
                    ToolResultStatus.FAILED,
                    code,
                    message,
                    retryable=False,
                    producer="web_fetch",
                    final_url=current_url,
                )
            self._fetch_cache[url] = artifact_or_error
            return _fetch_result(call, artifact_or_error, limit=limit, cache_hit=False)

        raise AssertionError("redirect loop must return before exhaustion")

    async def _browser_escalation(
        self,
        call: ToolCall,
        url: str,
        *,
        limit: int,
    ) -> ToolResult:
        if self._browser_capability is None or self._browser_egress_guard is None:
            return _browser_unavailable(call, url)
        try:
            async with asyncio.timeout(self._timeout_seconds):
                readiness = await self._browser_egress_guard.readiness()
        except (OSError, TimeoutError):
            return _browser_unavailable(call, url)
        if not readiness.ready:
            return _browser_unavailable(call, url)
        try:
            async with asyncio.timeout(self._timeout_seconds):
                page = await self._browser_capability.fetch(
                    url,
                    max_bytes=self._max_response_bytes,
                    timeout_seconds=self._timeout_seconds,
                )
        except (OSError, TimeoutError):
            return _web_error(
                call,
                ToolResultStatus.FAILED,
                "browser_escalation_failed",
                "The guarded browser escalation failed.",
                retryable=True,
                producer="browser",
                final_url=url,
            )
        if len(page.body) > self._max_response_bytes:
            return _web_error(
                call,
                ToolResultStatus.BLOCKED,
                "response_byte_limit_exceeded",
                "The browser response exceeded the byte limit.",
                retryable=False,
                producer="browser",
                final_url=page.final_url,
            )
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await self._egress_guard.resolve(page.final_url)
        except EgressPolicyError as error:
            return _web_error(
                call,
                ToolResultStatus.BLOCKED,
                error.code,
                str(error),
                retryable=False,
                producer="browser",
                final_url=page.final_url,
            )
        except EgressResolutionError:
            return _web_error(
                call,
                ToolResultStatus.FAILED,
                "dns_resolution_failed",
                "The browser final URL could not be resolved.",
                retryable=True,
                producer="browser",
                final_url=page.final_url,
            )
        except TimeoutError:
            return _web_error(
                call,
                ToolResultStatus.FAILED,
                "dns_resolution_timeout",
                "The browser final URL resolution exceeded the time limit.",
                retryable=True,
                producer="browser",
                final_url=page.final_url,
            )
        response = HttpResponse(
            status=200,
            headers={"Content-Type": page.content_type},
            body=page.body,
        )
        artifact_or_error = _response_artifact(response, page.final_url, producer="browser")
        if isinstance(artifact_or_error, tuple):
            code, message = artifact_or_error
            return _web_error(
                call,
                ToolResultStatus.FAILED,
                code,
                message,
                retryable=False,
                producer="browser",
                final_url=page.final_url,
            )
        source_url = cast(str, call.arguments["url"])
        self._fetch_cache[source_url] = artifact_or_error
        return _fetch_result(call, artifact_or_error, limit=limit, cache_hit=False)

    async def _web_search(self, call: ToolCall) -> ToolResult:
        query = _normalized_query(cast(str, call.arguments["query"]))
        limit = cast(int, call.arguments.get("limit", 8))
        cache_key = (query.casefold(), limit)
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return replace(
                cached,
                tool_call_id=call.id,
                meta={**cached.meta, "cache_hit": True},
            )
        result = None
        if self._search_endpoint is not None:
            result = await self._search_via_searxng(call, query, limit)
        if result is None:
            # SearXNG is optional and public instances go down; DuckDuckGo needs no
            # credential, so a search never fails only for lack of configuration.
            result = await self._search_via_duckduckgo(call, query, limit)
        if result.status in {ToolResultStatus.SUCCESS, ToolResultStatus.EMPTY}:
            self._search_cache[cache_key] = result
        return result

    async def _search_via_searxng(
        self, call: ToolCall, query: str, limit: int
    ) -> ToolResult | None:
        """Returns None when the instance is unusable, so the fallback can answer."""
        endpoint = cast(str, self._search_endpoint)
        request_url = (
            endpoint + ("&" if "?" in endpoint else "?") + urlencode({"q": query, "format": "json"})
        )
        response = await self._search_request(
            request_url,
            accept="application/json",
            guard=self._search_egress_guard,
        )
        if response is None:
            return None
        content_type = _header(response.headers, "content-type")
        if (
            response.status < 200
            or response.status >= 300
            or content_type is None
            or content_type.partition(";")[0].strip().casefold() != "application/json"
        ):
            return None
        try:
            payload: object = json.loads(response.body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        parsed = _parse_searxng_results(payload, limit=limit)
        if parsed is None:
            return None
        results, truncated = parsed
        return _search_result(call, query, results, truncated=truncated, engine="searxng")

    async def _search_via_duckduckgo(self, call: ToolCall, query: str, limit: int) -> ToolResult:
        request_url = (
            self._fallback_search_endpoint
            + ("&" if "?" in self._fallback_search_endpoint else "?")
            + urlencode({"q": query})
        )
        response = await self._search_request(
            request_url,
            accept="text/html",
            guard=self._egress_guard,
        )
        if response is None:
            return _provider_error(
                call,
                "provider_unavailable",
                "No search provider could be reached.",
                retryable=True,
            )
        if response.status == 429:
            return _provider_error(
                call,
                "provider_rate_limited",
                "The search provider rate limited the request.",
                retryable=True,
            )
        if response.status < 200 or response.status >= 300:
            return _provider_error(
                call,
                "provider_unavailable" if response.status >= 500 else "provider_error",
                f"The search provider returned HTTP {response.status}.",
                retryable=response.status >= 500 or response.status == 408,
            )
        results, truncated = _parse_duckduckgo_results(response.body, limit=limit)
        return _search_result(call, query, results, truncated=truncated, engine="duckduckgo")

    async def _search_request(
        self,
        request_url: str,
        *,
        accept: str,
        guard: EgressGuard,
    ) -> HttpResponse | None:
        """One guarded GET. None means "this provider did not answer usefully"."""
        try:
            async with asyncio.timeout(self._timeout_seconds):
                target = await guard.resolve(request_url)
                return await self._http_transport.request(
                    target,
                    headers={"Accept": accept, "User-Agent": _USER_AGENT},
                    max_bytes=self._max_response_bytes,
                    timeout_seconds=self._timeout_seconds,
                )
        except (
            EgressPolicyError,
            EgressResolutionError,
            OSError,
            TimeoutError,
            aiohttp.ClientError,
        ):
            return None

    async def _get_weather(self, call: ToolCall) -> ToolResult:
        location = cast(str, call.arguments["location"])
        days = cast(int, call.arguments.get("days", 3))
        geocoding = await self._json_request(
            GEOCODING_ENDPOINT + "?" + urlencode({"name": location, "count": 1, "format": "json"})
        )
        if geocoding is None:
            return _provider_error(
                call,
                "provider_unavailable",
                "The weather provider could not be reached.",
                retryable=True,
            )
        place = _first_geocoding_match(geocoding)
        if place is None:
            return ToolResult(
                tool_call_id=call.id,
                status=ToolResultStatus.EMPTY,
                retryable=False,
                data={"location": location, "matches": []},
                error=None,
                meta=_weather_meta(),
            )
        forecast = await self._json_request(
            FORECAST_ENDPOINT
            + "?"
            + urlencode(
                {
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                    "forecast_days": days,
                    "timezone": "auto",
                }
            )
        )
        if forecast is None or not isinstance(forecast, Mapping):
            return _provider_error(
                call,
                "provider_invalid_response",
                "The weather provider returned an unusable response.",
                retryable=True,
            )
        payload = cast(Mapping[str, JsonValue], forecast)
        return ToolResult(
            tool_call_id=call.id,
            status=ToolResultStatus.SUCCESS,
            retryable=False,
            data={
                "place": place,
                "current": payload.get("current"),
                "current_units": payload.get("current_units"),
                "daily": payload.get("daily"),
                "daily_units": payload.get("daily_units"),
            },
            error=None,
            meta=_weather_meta(),
        )

    async def _json_request(self, request_url: str) -> object | None:
        response = await self._search_request(
            request_url,
            accept="application/json",
            guard=self._egress_guard,
        )
        if response is None or response.status < 200 or response.status >= 300:
            return None
        try:
            return json.loads(response.body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    async def preview(self, call: ToolCall) -> ConfirmationPreview | None:
        # Nothing here mutates the Workspace, so there is no diff to show.
        del call
        return None

    def _normalized(self, call: ToolCall) -> ToolCall:
        definition = self._registry.get(call.name)
        if definition is None:
            return call
        arguments = definition.normalized_arguments(call.arguments)
        return call if arguments == call.arguments else replace(call, arguments=arguments)

    def _validate_call(self, call: ToolCall) -> ToolDefinitionConfig:
        definition = self._registry.get(call.name)
        if definition is None:
            raise _PreflightIssue("unknown_tool", "Unknown web tool.")
        if definition.status != "enabled":
            raise _PreflightIssue("tool_not_enabled", "The requested tool is not enabled.")
        validator = Draft202012Validator(
            cast(Mapping[str, Any], definition.parameters),
            format_checker=FormatChecker(),
        )
        errors = sorted(
            validator.iter_errors(call.arguments),  # pyright: ignore[reportUnknownMemberType]
            key=lambda error: list(error.path),
        )
        if errors:
            raise _PreflightIssue("invalid_tool_arguments", errors[0].message)
        if call.name == "web_search" and not _normalized_query(cast(str, call.arguments["query"])):
            raise _PreflightIssue("invalid_tool_arguments", "The search query must not be blank.")
        if call.name == "web_fetch":
            try:
                self._egress_guard.validate_url(cast(str, call.arguments["url"]))
            except EgressPolicyError as error:
                raise _PreflightIssue(error.code, str(error)) from error
        # By effect, never by name: the registry says which grants the declared
        # effects demand, and a tool added there is gated without touching this.
        missing = [
            grant for grant in definition.required_grants if grant not in self._effective_grants
        ]
        if missing:
            raise _PreflightIssue(
                grant_reason_code(missing[0]),
                f"The {missing[0]} is required for this effect.",
            )
        return definition


async def _system_lookup(
    host: str,
    port: int,
    family: int,
    socket_type: int,
) -> Sequence[AddressInfo]:
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(host, port, family=family, type=socket_type)
    converted: list[AddressInfo] = []
    for item in records:
        socket_address = item[4]
        if not isinstance(socket_address[0], str):
            raise EgressResolutionError("Host resolution returned an invalid address.")
        converted.append(
            (
                int(item[0]),
                int(item[1]),
                item[2],
                item[3],
                cast(SocketAddress, socket_address),
            )
        )
    return converted


class _ReadableHTMLParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._parts: list[str] = []
        self._suppressed_depth = 0
        self._anchors: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        if self._suppressed_depth:
            if lowered in SUPPRESSED_HTML_TAGS:
                self._suppressed_depth += 1
            return
        if lowered in SUPPRESSED_HTML_TAGS:
            self._suppressed_depth = 1
            return
        if lowered in _BOUNDARY_HTML_TAGS:
            self._parts.append("\n")
        if lowered == "a":
            href = next((value for name, value in attrs if name.casefold() == "href"), None)
            self._anchors.append(_useful_link(self._base_url, href))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if self._suppressed_depth:
            if lowered in SUPPRESSED_HTML_TAGS:
                self._suppressed_depth -= 1
            return
        if lowered == "a" and self._anchors:
            href = self._anchors.pop()
            if href is not None:
                self._parts.append(f" ({href})")
        if lowered in _BOUNDARY_HTML_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._suppressed_depth:
            self._parts.append(data)

    def content(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(without_link_menus(line for line in lines if line)).strip()


def _response_artifact(
    response: HttpResponse,
    final_url: str,
    *,
    producer: str,
) -> _FetchArtifact | tuple[str, str]:
    content_type_header = _header(response.headers, "content-type")
    if content_type_header is None:
        return "content_type_required", "The web response did not include a Content-Type header."
    content_type = content_type_header.partition(";")[0].strip().casefold()
    if content_type not in _HTML_CONTENT_TYPES | _TEXT_CONTENT_TYPES:
        return "unsupported_content_type", "The web response Content-Type is not supported."
    charset_match = re.search(r"charset\s*=\s*[\"']?([^;\s\"']+)", content_type_header, re.I)
    charset = charset_match.group(1) if charset_match is not None else "utf-8"
    try:
        decoded = response.body.decode(charset, errors="replace")
    except LookupError:
        return "unsupported_charset", "The web response charset is not supported."
    if content_type in _HTML_CONTENT_TYPES:
        parser = _ReadableHTMLParser(final_url)
        parser.feed(decoded)
        parser.close()
        content = parser.content()
    else:
        content = decoded.strip()
    if len(content) < 120:
        return "empty_extraction", "The web response did not contain enough readable content."
    return _FetchArtifact(
        content=content,
        content_type=content_type,
        final_url=final_url,
        producer=producer,
    )


def _fetch_result(
    call: ToolCall,
    artifact: _FetchArtifact,
    *,
    limit: int,
    cache_hit: bool,
) -> ToolResult:
    truncated = len(artifact.content) > limit
    meta: dict[str, Any] = {
        "producer": artifact.producer,
        "truncated": truncated,
        "taints": ["UntrustedWebTaint"],
        "final_url": artifact.final_url,
        "cache_hit": cache_hit,
    }
    return ToolResult(
        tool_call_id=call.id,
        status=ToolResultStatus.SUCCESS,
        retryable=False,
        data={
            "url": cast(str, call.arguments["url"]),
            "final_url": artifact.final_url,
            "content_type": artifact.content_type,
            "content": artifact.content[:limit],
        },
        error=None,
        meta=meta,
    )


def _useful_link(base_url: str, href: str | None) -> str | None:
    if href is None:
        return None
    absolute = urljoin(base_url, href.strip())
    split = urlsplit(absolute)
    if split.scheme.casefold() not in {"http", "https"}:
        return None
    # Uma âncora para a própria página (nota de rodapé, índice) não acrescenta
    # destino nenhum e ocupa mais espaço do que o texto que acompanha.
    if split._replace(fragment="").geturl() == urlsplit(base_url)._replace(fragment="").geturl():
        return None
    return absolute


def _is_link_only(line: str) -> bool:
    start = 0 if line.startswith("(http") else line.rfind(" (http") + 1
    if start == 0 and not line.startswith("(http"):
        return False
    return start <= 80 and line.endswith(")") and " " not in line[start:]


def without_link_menus(lines: Iterable[str], *, run: int = 10) -> list[str]:
    # ponytail: menus (lista de idiomas, rodapé, barra lateral) viram longas
    # sequências de linhas que são só rótulo + URL, e comem o orçamento de
    # caracteres antes do texto da página. Heurística de densidade de links no
    # lugar de um extrator readability completo; trocar se ficar imprecisa.
    kept: list[str] = []
    menu: list[str] = []
    for line in lines:
        if _is_link_only(line):
            menu.append(line)
            continue
        if len(menu) < run:
            kept.extend(menu)
        menu = []
        kept.append(line)
    if len(menu) < run:
        kept.extend(menu)
    return kept


def _header(headers: Mapping[str, str], name: str) -> str | None:
    expected = name.casefold()
    return next((value for key, value in headers.items() if key.casefold() == expected), None)


def _http_status_error(call: ToolCall, status: int, *, final_url: str) -> ToolResult:
    retryable = status == 429 or status == 408 or status >= 500
    code = "http_rate_limited" if status == 429 else "http_upstream_error"
    return _web_error(
        call,
        ToolResultStatus.FAILED,
        code,
        f"The web server returned HTTP {status}.",
        retryable=retryable,
        producer="web_fetch",
        final_url=final_url,
    )


def _browser_unavailable(call: ToolCall, final_url: str) -> ToolResult:
    return _web_error(
        call,
        ToolResultStatus.FAILED,
        "browser_escalation_unavailable",
        "Guarded browser escalation is not available.",
        retryable=False,
        producer="web_fetch",
        final_url=final_url,
    )


def _normalized_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).split())


def _search_result(
    call: ToolCall,
    query: str,
    results: list[dict[str, str]],
    *,
    truncated: bool,
    engine: str,
) -> ToolResult:
    meta: dict[str, Any] = {
        "producer": "web_search",
        "truncated": truncated,
        "taints": ["UntrustedWebTaint"],
        "engine": engine,
        "cache_hit": False,
    }
    return ToolResult(
        tool_call_id=call.id,
        status=ToolResultStatus.SUCCESS if results else ToolResultStatus.EMPTY,
        retryable=False,
        data={"query": query, "results": results},
        error=None,
        meta=meta,
    )


def _weather_meta() -> dict[str, Any]:
    return {
        "producer": "open_meteo",
        "truncated": False,
        "taints": ["UntrustedWebTaint"],
    }


def _first_geocoding_match(payload: object) -> dict[str, JsonValue] | None:
    if not isinstance(payload, Mapping):
        return None
    results = cast(Mapping[str, object], payload).get("results")
    if not isinstance(results, list) or not results:
        return None
    first = cast(list[object], results)[0]
    if not isinstance(first, Mapping):
        return None
    entry = cast(Mapping[str, object], first)
    latitude = entry.get("latitude")
    longitude = entry.get("longitude")
    if not isinstance(latitude, int | float) or not isinstance(longitude, int | float):
        return None
    place: dict[str, JsonValue] = {"latitude": latitude, "longitude": longitude}
    for key in ("name", "admin1", "country", "timezone"):
        value = entry.get(key)
        if isinstance(value, str):
            place[key] = value
    return place


def _parse_searxng_results(
    payload: object,
    *,
    limit: int,
) -> tuple[list[dict[str, str]], bool] | None:
    if not isinstance(payload, Mapping):
        return None
    raw_results = cast(Mapping[str, object], payload).get("results", [])
    if not isinstance(raw_results, list):
        return None
    raw_result_items = cast(list[object], raw_results)
    results: list[dict[str, str]] = []
    for item in raw_result_items[:limit]:
        if not isinstance(item, Mapping):
            return None
        item_mapping = cast(Mapping[str, object], item)
        title = item_mapping.get("title")
        url = item_mapping.get("url")
        description = item_mapping.get("content", "")
        if not isinstance(title, str) or not isinstance(url, str):
            return None
        results.append(
            {
                "title": title,
                "url": url,
                "description": description if isinstance(description, str) else "",
            }
        )
    return results, len(raw_result_items) > limit


def _parse_duckduckgo_results(body: bytes, *, limit: int) -> tuple[list[dict[str, str]], bool]:
    parser = _DuckDuckGoResultParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    parser.close()
    found = parser.results
    return found[:limit], len(found) > limit


class _DuckDuckGoResultParser(HTMLParser):
    """Reads the lite/ result table: one anchor per hit, snippet in its own cell."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._in_link = False
        self._in_snippet = False
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.casefold(): value or "" for name, value in attrs}
        classes = values.get("class", "").split()
        if tag.casefold() == "a" and "result-link" in classes:
            self._in_link = True
            self._title = []
            self._url = values.get("href", "")
        elif tag.casefold() == "td" and "result-snippet" in classes:
            self._in_snippet = True
            self._snippet = []

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered == "a" and self._in_link:
            self._in_link = False
            title = "".join(self._title).strip()
            if title and self._url:
                self.results.append({"title": title, "url": self._url, "description": ""})
        elif lowered == "td" and self._in_snippet:
            self._in_snippet = False
            snippet = " ".join("".join(self._snippet).split())
            if snippet and self.results:
                self.results[-1]["description"] = snippet

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._title.append(data)
        elif self._in_snippet:
            self._snippet.append(data)


def _provider_error(
    call: ToolCall,
    code: str,
    message: str,
    *,
    retryable: bool,
) -> ToolResult:
    return _web_error(
        call,
        ToolResultStatus.FAILED,
        code,
        message,
        retryable=retryable,
        producer="web_search",
    )


def _web_error(
    call: ToolCall,
    status: ToolResultStatus,
    code: str,
    message: str,
    *,
    retryable: bool,
    producer: str = "web",
    final_url: str | None = None,
) -> ToolResult:
    meta: dict[str, Any] = {
        "producer": producer,
        "truncated": False,
        "taints": ["UntrustedWebTaint"],
    }
    if final_url is not None:
        meta["final_url"] = final_url
    return ToolResult(
        tool_call_id=call.id,
        status=status,
        retryable=retryable,
        data=None,
        error={"code": code, "message": message},
        meta=meta,
    )
