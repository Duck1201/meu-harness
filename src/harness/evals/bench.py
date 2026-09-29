"""Deterministic loopback bench for web evidence.

The egress policy denies loopback by design, so an experiment that needs a page
with a known symptom cannot point at the production guard. It also cannot point
at a third-party site: the promotion protocol freezes seven digests per run, and
a remote page is none of them — 300 requests to someone else's server would make
the evidence depend on their uptime and their markup.

The exception lives here, inside the evals package, and nothing on the
application path imports this module: ``ApplicationService`` builds
``WebToolExecutor`` with the production :class:`~harness.web_tools.EgressGuard`.
That structural separation is the guarantee; ``config/harness.json#network.eval_bench``
only declares it, and a fixture asserts the production route never gets it.
"""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import dataclass
from types import TracebackType

from ..web_tools import EgressGuard, ResolvedAddress, ResolvedTarget

BENCH_HOSTNAME = "bench.harness.test"

# Well above network.web_fetch.empty_extraction_threshold_chars (120): HTTP alone
# is enough and no browser may start.
HTTP_READABLE_PAGE = (
    "<!doctype html><html><head><title>Readable</title></head><body><main>"
    + "<p>Conteudo legivel servido diretamente por HTTP. </p>" * 3200
    + "</main></body></html>"
).encode()

# Below the threshold until JavaScript runs: only a browser recovers the content.
BROWSER_REQUIRED_PAGE = (
    b"<!doctype html><html><head><title>JavaScript only</title></head><body>"
    b"<main id='app'>carregando</main>"
    b"<script>document.getElementById('app').textContent = "
    b"'Conteudo revelado apenas apos a execucao de JavaScript. '.repeat(400);</script>"
    b"</body></html>"
)

SEARCH_PATH = "/search"

# SearXNG-shaped, so web_search exercises the real parser without leaving the bench.
SEARCH_RESPONSE = json.dumps(
    {
        "query": "bench",
        "results": [
            {
                "title": "Documentacao oficial do modelo",
                "url": "https://bench.harness.test/readable",
                "content": "Pagina de referencia servida pela bancada.",
            },
            {
                "title": "Guia de uso",
                "url": "https://bench.harness.test/js-only",
                "content": "Segundo resultado da bancada.",
            },
        ],
    }
).encode()

GEOCODING_PATH = "/geocoding"
FORECAST_PATH = "/forecast"

# Open-Meteo-shaped: uma cidade só e um tempo fixo, para get_weather passar pelo
# parser real sem a bancada depender do tempo que faz lá fora.
GEOCODING_RESPONSE = json.dumps(
    {
        "results": [
            {
                "name": "Recife",
                "latitude": -8.05,
                "longitude": -34.88,
                "admin1": "Pernambuco",
                "country": "Brasil",
                "timezone": "America/Recife",
            }
        ]
    }
).encode()
FORECAST_RESPONSE = json.dumps(
    {
        "current": {
            "time": "2026-01-01T12:00",
            "temperature_2m": 29.4,
            "relative_humidity_2m": 71,
            "wind_speed_10m": 14.2,
            "weather_code": 2,
        },
        "current_units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "wind_speed_10m": "km/h",
        },
        "daily": {
            "time": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "temperature_2m_max": [31.0, 30.2, 29.8],
            "temperature_2m_min": [24.1, 23.9, 24.0],
            "precipitation_sum": [0.0, 3.4, 12.8],
            "weather_code": [2, 61, 63],
        },
        "daily_units": {"temperature_2m_max": "°C", "precipitation_sum": "mm"},
    }
).encode()

_JSON_ROUTES = {
    SEARCH_PATH: SEARCH_RESPONSE,
    GEOCODING_PATH: GEOCODING_RESPONSE,
    FORECAST_PATH: FORECAST_RESPONSE,
}

BENCH_PAGES: dict[str, bytes] = {
    "/readable": HTTP_READABLE_PAGE,
    "/js-only": BROWSER_REQUIRED_PAGE,
}


@dataclass(frozen=True, slots=True)
class BenchRequest:
    method: str
    path: str


class BenchServer:
    """Serves the fixture pages on an ephemeral loopback port."""

    def __init__(self, pages: dict[str, bytes] | None = None) -> None:
        self._pages = dict(pages if pages is not None else BENCH_PAGES)
        self._server: asyncio.Server | None = None
        self._requests: list[BenchRequest] = []
        self.port = 0

    @property
    def requests(self) -> tuple[BenchRequest, ...]:
        return tuple(self._requests)

    def url(self, path: str) -> str:
        return f"http://{BENCH_HOSTNAME}:{self.port}{path}"

    async def __aenter__(self) -> BenchServer:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            request_line = head.split(b"\r\n", 1)[0].decode("latin-1").split(" ")
            method = request_line[0] if request_line else "GET"
            path = request_line[1] if len(request_line) > 1 else "/"
            self._requests.append(BenchRequest(method=method, path=path))
            route = path.split("?", 1)[0]
            json_body = _JSON_ROUTES.get(route)
            body = json_body if json_body is not None else self._pages.get(route)
            content_type = (
                b"application/json" if json_body is not None else b"text/html; charset=utf-8"
            )
            status = b"200 OK" if body is not None else b"404 Not Found"
            payload = body if body is not None else b"not found"
            writer.write(
                b"HTTP/1.1 "
                + status
                + b"\r\nContent-Type: "
                + content_type
                + b"\r\nContent-Length: "
                + str(len(payload)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + payload
            )
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            return
        finally:
            writer.close()


class BenchEgressGuard(EgressGuard):
    """Pins the bench hostname at the loopback bench server.

    Every other URL still goes through the production checks, so a fixture that
    tries to reach anywhere else is rejected exactly as it would be in production.
    """

    def __init__(self, port: int) -> None:
        super().__init__()
        self._port = port

    async def resolve(self, url: str) -> ResolvedTarget:
        parsed = self.validate_url(url)
        hostname = (parsed.hostname or "").casefold().rstrip(".")
        if hostname != BENCH_HOSTNAME:
            return await super().resolve(url)
        return ResolvedTarget(
            url=url,
            hostname=hostname,
            port=self._port,
            addresses=(
                ResolvedAddress(
                    host="127.0.0.1",
                    port=self._port,
                    family=socket.AF_INET,
                    proto=socket.IPPROTO_TCP,
                ),
            ),
        )
