"""Collects pages into a Corpus, by API when the site has one.

Almost every game wiki is MediaWiki, and MediaWiki answers `/api.php`: the full
page list comes paginated and the text comes as plain extracts, without
navigation, infobox or footer. Following links through the HTML of the same site
costs an order of magnitude more requests and delivers dirtier text, so it is
the fallback, not the plan.

Every request goes through the same EgressGuard as `web_fetch` — scheme, private
ranges, SSRF revalidated on redirect. What the Operator gets to skip is the
grant, not the guard: grants exist to contain the model, and the Operator is the
authority that issues them.
"""

import asyncio
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser
from typing import Protocol, cast
from urllib.parse import SplitResult, urlencode, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from .config import CorpusScraperConfig
from .corpus_ingestion import extract_html
from .web_tools import (
    SUPPRESSED_HTML_TAGS,
    AiohttpHttpTransport,
    EgressGuard,
    EgressPolicyError,
    EgressResolutionError,
    HttpResponse,
    HttpTransport,
)

USER_AGENT = "harness-2-corpus/1.0 (local operator tool)"
_MAX_REDIRECTS = 5
_MAX_PAGE_BYTES = 4 * 1024 * 1024
_TIMEOUT_SECONDS = 30.0
_HTML_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_SITEMAP_LOCATION = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)
# `== Seção ==` até `====== ======`: o nível do heading é a contagem de iguais.
_WIKI_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1$")


class ScrapeError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# Uma página que não veio é uma página a menos, não um motivo para abandonar o
# resto: `aiohttp` levanta TimeoutError puro, e falha de conexão chega como OSError.
_FETCH_FAILURES = (
    ScrapeError,
    EgressPolicyError,
    EgressResolutionError,
    TimeoutError,
    OSError,
)


@dataclass(frozen=True, slots=True)
class ScrapedPage:
    url: str
    title: str
    # HTML para a rota genérica, texto puro para a rota MediaWiki. O ingestor
    # decide o extrator pela extensão que este campo declara.
    filename: str
    data: bytes


@dataclass(frozen=True, slots=True)
class ScrapePlan:
    seed: str
    source: str
    detail: str


class PageRenderer(Protocol):
    """Renderiza uma URL num browser e devolve o HTML final (ADR 0014)."""

    async def render(self, url: str) -> str: ...


class Scraper:
    def __init__(
        self,
        *,
        config: CorpusScraperConfig,
        egress_guard: EgressGuard | None = None,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        renderer: PageRenderer | None = None,
    ) -> None:
        self._config = config
        self._guard = egress_guard or EgressGuard()
        self._transport = transport or AiohttpHttpTransport()
        self._sleep = sleep
        self._renderer = renderer

    async def plan(self, seed: str) -> ScrapePlan:
        """Decides the route before spending anything on it."""
        endpoint = await self._mediawiki_endpoint(seed)
        if endpoint is not None:
            return ScrapePlan(seed=seed, source="mediawiki", detail=endpoint)
        return ScrapePlan(seed=seed, source="html_crawl", detail=urlsplit(seed).netloc)

    async def collect(
        self, plan: ScrapePlan, known_urls: frozenset[str] = frozenset()
    ) -> AsyncIterator[ScrapedPage]:
        """Collects the seed, skipping what the Corpus already carries.

        The skip happens before the request that would fetch the page, not after:
        a wiki that does not fit under the ceiling gets collected across several
        runs, and each run spends its budget on pages the Corpus does not have.

        Only the MediaWiki route resumes. The crawl discovers the next URL inside
        the page it just fetched, so skipping a known page there would also drop
        the links only that page declares — it would resume by shrinking reach.
        """
        if plan.source == "mediawiki":
            async for page in self._collect_mediawiki(plan.detail, known_urls):
                yield page
            return
        async for page in self._collect_html(plan.seed):
            yield page

    async def _mediawiki_endpoint(self, seed: str) -> str | None:
        """Finds `/api.php` for the seed, trying the two shapes wikis actually use."""
        parsed = urlsplit(seed)
        if parsed.scheme.casefold() not in {"http", "https"}:
            raise ScrapeError("disallowed_url_scheme", "A semente precisa ser HTTP ou HTTPS.")
        root = f"{parsed.scheme}://{parsed.netloc}"
        candidates = [f"{root}/api.php", f"{root}/w/api.php"]
        prefix = parsed.path.rsplit("/", 1)[0]
        if prefix and prefix != "/":
            candidates.insert(0, f"{root}{prefix}/api.php")
        for candidate in candidates:
            query = urlencode({"action": "query", "meta": "siteinfo", "format": "json"})
            try:
                response = await self._get(f"{candidate}?{query}")
            except _FETCH_FAILURES:
                continue
            if response.status != 200:
                continue
            payload = _json_object(response.body)
            if payload is not None and "query" in payload:
                return candidate
        return None

    async def _collect_mediawiki(
        self, endpoint: str, known_urls: frozenset[str]
    ) -> AsyncIterator[ScrapedPage]:
        settings = self._config.mediawiki
        continuation: dict[str, str] = {}
        collected = 0
        limit = self._config.html_crawl.max_pages
        while collected < limit:
            listing = await self._api(
                endpoint,
                {
                    "action": "query",
                    "list": "allpages",
                    "aplimit": "500",
                    "apfilterredir": "nonredirects",
                    "format": "json",
                    **continuation,
                },
            )
            pages = _mediawiki_pages(listing)
            if not pages:
                return
            # A listagem é barata e o extrato não é: o que o Corpus já tem sai
            # aqui, antes de virar requisição.
            fresh = tuple(
                (identifier, title, url)
                for identifier, title in pages
                if (url := _mediawiki_page_url(endpoint, identifier)) not in known_urls
            )
            for batch in _batched(fresh, settings.page_batch):
                extracts = await self._api(
                    endpoint,
                    {
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": "1",
                        "exlimit": str(len(batch)),
                        "pageids": "|".join(str(identifier) for identifier, _, _ in batch),
                        "format": "json",
                    },
                )
                for identifier, title, url in batch:
                    page = await self._mediawiki_page(
                        endpoint,
                        identifier=identifier,
                        title=title,
                        url=url,
                        extract=_mediawiki_extract(extracts, identifier),
                    )
                    if page is None:
                        continue
                    collected += 1
                    yield page
                    if collected >= limit:
                        return
                await self._wait()
            following = _mediawiki_continue(listing)
            if not following:
                return
            continuation = following

    async def _mediawiki_page(
        self,
        endpoint: str,
        *,
        identifier: int,
        title: str,
        url: str,
        extract: str,
    ) -> ScrapedPage | None:
        """Prefers the extract, and falls back to the rendered page when it is hollow.

        `extracts` does not render templates. A wiki that keeps its facts inside
        them answers with the navigation around the page and nothing else: an
        ability page of a game wiki came back as twenty characters of "See also".
        The rendered HTML costs one more request and arrives dirtier — infobox,
        edit links, navbox — but the numbers a question is actually about live
        exactly in the table the extract dropped.
        """
        if len(extract) >= self._config.mediawiki.thin_extract_chars:
            return ScrapedPage(
                url=url,
                title=title,
                filename=f"{title}.md",
                data=_mediawiki_markdown(title, extract).encode(),
            )
        rendered = await self._mediawiki_rendered(endpoint, identifier)
        if rendered:
            # `parse` devolve o corpo da página, sem <title>. Sem ele o extrator
            # adota o primeiro heading do fragmento como nome do Documento: a
            # página `Absorb` entrou no Corpus chamada `See also`, e o endereço
            # de todo Chunk dela apontava para o lugar errado. A wiki já disse o
            # nome na listagem, e é ele que manda.
            return ScrapedPage(
                url=url,
                title=title,
                filename=f"{title}.html",
                data=f"<title>{escape(title)}</title>{rendered}".encode(),
            )
        if not extract:
            return None
        return ScrapedPage(
            url=url,
            title=title,
            filename=f"{title}.md",
            data=_mediawiki_markdown(title, extract).encode(),
        )

    async def _mediawiki_rendered(self, endpoint: str, identifier: int) -> str:
        payload = await self._api(
            endpoint,
            {
                "action": "parse",
                "pageid": str(identifier),
                "prop": "text",
                "format": "json",
            },
        )
        parse = _object(payload.get("parse"))
        rendered = _object(parse.get("text")) if parse is not None else None
        html = rendered.get("*") if rendered is not None else None
        return html if isinstance(html, str) else ""

    async def _collect_html(self, seed: str) -> AsyncIterator[ScrapedPage]:
        settings = self._config.html_crawl
        robots = await self._robots(seed) if settings.respect_robots_txt else None
        origin = urlsplit(seed)
        queue: list[tuple[str, int]] = [(seed, 0)]
        if settings.prefer_sitemap:
            queue.extend((url, settings.max_depth) for url in await self._sitemap(seed))
        seen: set[str] = set()
        collected = 0
        total_bytes = 0
        while queue and collected < settings.max_pages and total_bytes < settings.max_total_bytes:
            url, depth = queue.pop(0)
            canonical = _canonical_url(url)
            if canonical in seen or not _same_site(origin, urlsplit(canonical)):
                continue
            seen.add(canonical)
            if robots is not None and not robots.can_fetch(USER_AGENT, canonical):
                continue
            try:
                response = await self._get(canonical)
            except _FETCH_FAILURES:
                continue
            await self._wait()
            if response.status != 200 or not _is_html(response.headers):
                continue
            body = await self._escalated(canonical, response.body)
            total_bytes += len(body)
            collected += 1
            yield ScrapedPage(
                url=canonical,
                title=canonical,
                filename="page.html",
                data=body,
            )
            if depth < settings.max_depth:
                queue.extend(
                    (link, depth + 1)
                    for link in _links(body, canonical)
                    if _canonical_url(link) not in seen
                )

    async def _escalated(self, url: str, body: bytes) -> bytes:
        """O HTML renderizado no browser quando o HTTP trouxe só a casca.

        Escala por sintoma, como o `web_fetch`: a página com texto visível passa
        direto, e só a que chega abaixo de `thin_page_chars` paga um Chromium. Uma
        renderização que falha devolve o que o HTTP trouxe — a página fina ainda é
        uma página, e o resto da coleta não depende dela.
        """
        escalation = self._config.html_crawl.browser_escalation
        if self._renderer is None or escalation.mode != "symptom":
            return body
        if _visible_chars(body) >= escalation.thin_page_chars:
            return body
        try:
            rendered = await self._renderer.render(url)
        except Exception:
            return body
        # Os links que o crawl segue saem deste HTML; os subrecursos que o browser
        # pediu para montá-lo já passaram pelo guard um a um.
        return rendered.encode("utf-8")

    async def _api(self, endpoint: str, parameters: Mapping[str, str]) -> Mapping[str, object]:
        """One API call, with the backoff a shared wiki asks for.

        A 429 or a maxlag is the site saying "slower", not "no": retrying the
        same page immediately is how a collector gets an IP banned and leaves the
        Corpus half full. A read timeout gets the same treatment for the same
        reason — over hours of collecting, one is certain to happen, and letting
        it end the job would throw away everything after it.
        """
        backoff = self._config.mediawiki.backoff_seconds
        for attempt in range(len(backoff) + 1):
            try:
                response = await self._get(f"{endpoint}?{urlencode(dict(parameters))}")
            except (TimeoutError, OSError) as error:
                if attempt >= len(backoff):
                    raise ScrapeError(
                        "mediawiki_unreachable",
                        f"A wiki parou de responder durante a coleta: {error}",
                    ) from error
                await self._wait(backoff[attempt])
                continue
            if response.status in {429, 503} and attempt < len(backoff):
                await self._wait(backoff[attempt])
                continue
            if response.status != 200:
                raise ScrapeError(
                    "mediawiki_http_error",
                    f"A wiki respondeu {response.status} à consulta da API.",
                )
            payload = _json_object(response.body)
            if payload is None:
                raise ScrapeError("mediawiki_invalid_json", "A wiki respondeu algo que não é JSON.")
            return payload
        raise ScrapeError("mediawiki_throttled", "A wiki recusou as tentativas seguidas de coleta.")

    async def _robots(self, seed: str) -> RobotFileParser | None:
        parsed = urlsplit(seed)
        try:
            response = await self._get(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        except _FETCH_FAILURES:
            return None
        if response.status != 200:
            return None
        parser = RobotFileParser()
        parser.parse(response.body.decode("utf-8", errors="replace").splitlines())
        return parser

    async def _sitemap(self, seed: str) -> tuple[str, ...]:
        parsed = urlsplit(seed)
        try:
            response = await self._get(f"{parsed.scheme}://{parsed.netloc}/sitemap.xml")
        except _FETCH_FAILURES:
            return ()
        if response.status != 200:
            return ()
        body = response.body.decode("utf-8", errors="replace")
        return tuple(match.group(1) for match in _SITEMAP_LOCATION.finditer(body))

    async def _get(self, url: str) -> HttpResponse:
        current = url
        for _ in range(_MAX_REDIRECTS + 1):
            # Resolvido de novo a cada salto: o guard vale para o destino final,
            # não só para o que o Operator digitou.
            target = await self._guard.resolve(current)
            response = await self._transport.request(
                target,
                headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
                max_bytes=_MAX_PAGE_BYTES,
                timeout_seconds=_TIMEOUT_SECONDS,
            )
            if response.status not in {301, 302, 303, 307, 308}:
                return response
            location = _header(response.headers, "location")
            if location is None:
                raise ScrapeError("invalid_redirect", "O redirecionamento não trouxe destino.")
            current = urljoin(current, location)
        raise ScrapeError("too_many_redirects", "O destino redirecionou vezes demais.")

    async def _wait(self, seconds: float | None = None) -> None:
        delay = (
            seconds if seconds is not None else self._config.html_crawl.delay_milliseconds / 1000
        )
        if delay <= 0:
            return
        if self._sleep is not None:
            await self._sleep(delay)
            return
        await asyncio.sleep(delay)


def _visible_chars(body: bytes) -> int:
    document = extract_html(body.decode("utf-8", errors="replace"), title_fallback="")
    return sum(len(block.text) for block in document.blocks)


class _LinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self.links: list[str] = []
        self._suppressed = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        if lowered in SUPPRESSED_HTML_TAGS:
            self._suppressed += 1
            return
        if self._suppressed or lowered != "a":
            return
        href = next((value for name, value in attrs if name.casefold() == "href"), None)
        if href is None:
            return
        absolute = urljoin(self._base_url, href.strip())
        if urlsplit(absolute).scheme.casefold() in {"http", "https"}:
            self.links.append(absolute)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in SUPPRESSED_HTML_TAGS and self._suppressed:
            self._suppressed -= 1


def _links(body: bytes, base_url: str) -> tuple[str, ...]:
    parser = _LinkParser(base_url)
    parser.feed(body.decode("utf-8", errors="replace"))
    parser.close()
    return tuple(dict.fromkeys(parser.links))


def _canonical_url(url: str) -> str:
    parsed = urlsplit(url)
    return parsed._replace(fragment="").geturl()


def _same_site(seed: SplitResult, candidate: SplitResult) -> bool:
    """Same registrable domain, so a link out is a link out.

    Compares the last two labels rather than consulting a public-suffix list:
    the crawl already has a page ceiling, and the failure mode of the shortcut is
    refusing a sibling subdomain, not wandering off the site.
    """
    seed_host = (seed.hostname or "").casefold()
    candidate_host = (candidate.hostname or "").casefold()
    if not seed_host or not candidate_host:
        return False
    return seed_host.split(".")[-2:] == candidate_host.split(".")[-2:]


def _is_html(headers: Mapping[str, str]) -> bool:
    value = _header(headers, "content-type")
    if value is None:
        return False
    return value.partition(";")[0].strip().casefold() in _HTML_CONTENT_TYPES


def _header(headers: Mapping[str, str], name: str) -> str | None:
    expected = name.casefold()
    return next((value for key, value in headers.items() if key.casefold() == expected), None)


def _json_object(body: bytes) -> Mapping[str, object] | None:
    try:
        decoded: object = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        return None
    return cast(dict[str, object], decoded) if isinstance(decoded, dict) else None


def _object(value: object) -> Mapping[str, object] | None:
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _mediawiki_pages(payload: Mapping[str, object]) -> tuple[tuple[int, str], ...]:
    query = _object(payload.get("query"))
    if query is None:
        return ()
    listing = query.get("allpages")
    if not isinstance(listing, list):
        return ()
    pages: list[tuple[int, str]] = []
    for raw in cast(list[object], listing):
        item = _object(raw)
        if item is None:
            continue
        identifier = item.get("pageid")
        title = item.get("title")
        if isinstance(identifier, int) and isinstance(title, str):
            pages.append((identifier, title))
    return tuple(pages)


def _mediawiki_continue(payload: Mapping[str, object]) -> dict[str, str]:
    following = _object(payload.get("continue"))
    if following is None:
        return {}
    return {str(key): str(value) for key, value in following.items()}


def _mediawiki_markdown(title: str, extract: str) -> str:
    """Turns the extract into the Markdown the ingestor actually reads.

    `explaintext` answers plain text with wikitext headings and one paragraph
    per line — two conventions the Markdown extractor does not know. Handed over
    as it arrives, the section heading is invisible, so every Chunk of the page
    is addressed by the title alone, and the single line breaks are undone as if
    they were hard wrapping, gluing the whole section into one paragraph that
    blows past the Chunk budget by itself.
    """
    lines = [f"# {title}"]
    for line in extract.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        heading = _WIKI_HEADING.match(stripped)
        lines.append(f"{'#' * len(heading.group(1))} {heading.group(2)}" if heading else stripped)
    return "\n\n".join(lines) + "\n"


def _mediawiki_page_url(endpoint: str, identifier: int) -> str:
    """The address a collected page carries, and the key a later run resumes by."""
    return f"{endpoint.removesuffix('api.php')}index.php?curid={identifier}"


def _mediawiki_extract(payload: Mapping[str, object], identifier: int) -> str:
    query = _object(payload.get("query"))
    pages = _object(query.get("pages")) if query is not None else None
    page = _object(pages.get(str(identifier))) if pages is not None else None
    extract = page.get("extract") if page is not None else None
    return extract.strip() if isinstance(extract, str) else ""


def _batched[T](items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), max(1, size)):
        yield items[start : start + size]


__all__ = ["USER_AGENT", "ScrapeError", "ScrapePlan", "ScrapedPage", "Scraper"]
