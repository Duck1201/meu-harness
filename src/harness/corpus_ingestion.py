"""Turns a file or a page into Chunks, without ever rewriting what it read.

Three stages, the same three for an upload and for the scraper: extract, clean,
chunk. Cleaning is deterministic on purpose — the feature exists because a 4B
model invents facts, and a cleaning stage that paraphrases would be the same
invention, made permanent inside the index and invisible at answer time.

What a Chunk adds to the text is an address, not a summary: the Document title,
the heading path it sits under and the page it came from. That prefix is what
gets embedded and indexed; the passage that gets quoted is the stored text,
byte for byte.
"""

import hashlib
import logging
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from typing import Literal

from .corpus_store import ChunkDraft, DocumentDraft
from .domain import UNTRUSTED_WEB_TAINT
from .ports import TextTokenCounter
from .web_tools import SUPPRESSED_HTML_TAGS, without_link_menus

ACCEPTED_EXTENSIONS = (".txt", ".md", ".html", ".pdf")

_BLOCK_HTML_TAGS = frozenset(
    {
        "p", "div", "section", "article", "li", "tr", "td", "th", "br",
        "pre", "blockquote", "dd", "dt", "figcaption", "caption",
    }
)  # fmt: skip
_HEADING_HTML_TAGS = {f"h{level}": level for level in range(1, 7)}
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*#*$")
# Fim de frase seguido de espaço. Não tenta resolver abreviação nem "1.22": errar
# a fronteira aqui custa um Chunk que começa uma frase adiante, não um fato torto.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
# "recupera-\nção": hífen de quebra de linha, não hífen de palavra composta. Só
# se a próxima linha começa em minúscula, senão "Vice-\nPresidente" perderia o seu.
_LINE_BREAK_HYPHEN = re.compile(r"(\w)-\n([a-zà-öø-ÿ])")
_PAGE_FURNITURE = re.compile(
    r"^(p[áa]g(ina)?\.?\s*)?\d{1,4}(\s*/\s*\d{1,4})?$|^page\s+\d{1,4}(\s+of\s+\d{1,4})?$",
    re.IGNORECASE,
)
# Um bloco com pouca letra é sopa de navegação, base64 ou moldura de tabela. O piso
# é baixo de propósito: "HP: 320" numa wiki é fato, não ruído.
_MINIMUM_LETTER_RATIO = 0.25
_MINIMUM_BLOCK_CHARACTERS = 3
# O que muda de página para página dentro de um cabeçalho: o número, em arábico
# ou romano maiúsculo, e a pontuação que o separa do título. Romano só em caixa
# alta para não comer "civil" nem "mil" de uma linha de texto de verdade.
_FURNITURE_NUMBERING = re.compile(r"\b[IVXLCDM]+\b|\d+|[^\w\s]")
# "Fragmentação de datagramas, 247": termo seguido do número da página. Sumário e
# índice remissivo são feitos disso, e nenhum dos dois responde pergunta alguma.
_LISTING_ENTRY = re.compile(r"[^\W\d_]{3,}[,.]?\s+\d{1,3}\b")
_LISTING_RATIO = 0.08
# Corrida mínima de páginas. Uma página solta cheia de número é figura ou tabela
# — medido: a página de prefixos `200.23.16.0/23` bate a razão e não é listagem.
_LISTING_RUN = 5
# Um cabeçalho ou rodapé de PDF se repete em quase toda página; três páginas é o
# mínimo para a repetição significar alguma coisa.
_FURNITURE_PAGE_FLOOR = 3
# Livro impresso alterna cabeçalho: o verso leva o título da obra e o recto o do
# capítulo, então nenhum dos dois chega perto de toda página — por construção,
# não passa de metade. Medido no Kurose: 43% no título do livro, e limiar de 60%
# não pegava nada. O piso mora abaixo do teto que a alternância impõe.
_FURNITURE_SHARE = 0.4


class UnsupportedSourceError(Exception):
    """The source is not something the harness knows how to read as text."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class SourceBlock:
    text: str
    heading_path: tuple[str, ...]
    page: int | None


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    title: str
    blocks: tuple[SourceBlock, ...]


def accepted_extension(filename: str) -> str | None:
    lowered = filename.lower()
    return next((suffix for suffix in ACCEPTED_EXTENSIONS if lowered.endswith(suffix)), None)


HtmlExtractor = Literal["builtin", "scrapling"]


def extract(
    filename: str, data: bytes, *, html_extractor: HtmlExtractor = "builtin"
) -> ExtractedDocument:
    """Reads a supported file into blocks, or refuses it by name.

    The refusal names the accepted formats because the Operator is the one who
    picked the file, and "unsupported" without a list is a dead end.
    """
    suffix = accepted_extension(filename)
    if suffix is None:
        accepted = ", ".join(ACCEPTED_EXTENSIONS)
        raise UnsupportedSourceError(
            "unsupported_extension",
            f"O harness só extrai texto de {accepted}. Converta o arquivo antes de subir.",
        )
    if suffix == ".pdf":
        return extract_pdf(filename, data)
    text = _decoded(data)
    if suffix == ".html":
        if html_extractor == "scrapling":
            return extract_html_scrapling(text, title_fallback=_stem(filename))
        return extract_html(text, title_fallback=_stem(filename))
    if suffix == ".md":
        return extract_markdown(text, title_fallback=_stem(filename))
    return extract_plain(text, title=_stem(filename))


def extract_plain(text: str, *, title: str) -> ExtractedDocument:
    blocks = tuple(
        SourceBlock(text=paragraph, heading_path=(), page=None)
        for paragraph in _paragraphs(_reflowed(_normalized(text)))
    )
    return ExtractedDocument(title=title, blocks=blocks)


def extract_markdown(text: str, *, title_fallback: str) -> ExtractedDocument:
    normalized = _normalized(text)
    title = title_fallback
    heading_path: list[str] = []
    blocks: list[SourceBlock] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraph = _reflowed("\n".join(buffer))
            for piece in _paragraphs(paragraph):
                blocks.append(SourceBlock(text=piece, heading_path=tuple(heading_path), page=None))
            buffer.clear()

    for line in normalized.split("\n"):
        heading = _MARKDOWN_HEADING.match(line)
        if heading is None:
            buffer.append(line)
            continue
        flush()
        level = len(heading.group(1))
        text_of_heading = heading.group(2).strip()
        if level == 1 and title == title_fallback:
            title = text_of_heading
        del heading_path[level - 1 :]
        heading_path.append(text_of_heading)
    flush()
    return ExtractedDocument(title=title, blocks=tuple(blocks))


def extract_html(html: str, *, title_fallback: str) -> ExtractedDocument:
    parser = _BlockHTMLParser()
    parser.feed(html)
    parser.close()
    return ExtractedDocument(
        title=parser.title or title_fallback,
        blocks=tuple(parser.blocks),
    )


_SETEXT_UNDERLINE = re.compile(r"^(=+|-+)\s*$")
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def extract_html_scrapling(html: str, *, title_fallback: str) -> ExtractedDocument:
    """HTML pela limpeza do Scrapling, depois pelo mesmo leitor de Markdown.

    O Scrapling tira o que o leitor não vê e um modelo leria: elemento com
    `display:none`, `aria-hidden`, `<template>`, comentário e caractere de largura
    zero — o esconderijo clássico de instrução injetada numa página que vira
    acervo. O que sobra chega como Markdown, e daí em diante o caminho é o de um
    upload `.md`: título, trilha de headings e parágrafos. Menus continuam saindo
    pela mesma heurística de densidade de links do extrator nativo.
    """
    from scrapling.engines.toolbelt.custom import Response

    # O Response do Scrapling anuncia cada página como "Fetched (200)"; aqui não
    # houve fetch nenhum, e o log do servidor não é lugar de ruído por Document.
    logging.getLogger("scrapling").setLevel(logging.WARNING)
    page = Response(
        url="",
        content=html,
        status=200,
        reason="OK",
        cookies={},
        headers={},
        request_headers={},
    )
    title = _collapsed(str(page.css("title::text").get() or ""))
    markdown = _atx_headings(_MARKDOWN_LINK.sub(r"\1", page.markdown(main_content_only=True)))
    document = extract_markdown(markdown, title_fallback=title or title_fallback)
    kept = set(without_link_menus(block.text for block in document.blocks))
    return ExtractedDocument(
        title=title or document.title,
        blocks=tuple(block for block in document.blocks if block.text in kept),
    )


def _atx_headings(markdown: str) -> str:
    """`Título\n=====` vira `# Título`: o leitor de Markdown só conhece a forma ATX."""
    lines = markdown.split("\n")
    converted: list[str] = []
    for line in lines:
        underline = _SETEXT_UNDERLINE.match(line)
        if underline and converted and converted[-1].strip() and not converted[-1].startswith("#"):
            level = 1 if underline.group(1).startswith("=") else 2
            converted[-1] = f"{'#' * level} {converted[-1].strip()}"
            continue
        converted.append(line)
    return "\n".join(converted)


def extract_pdf(filename: str, data: bytes) -> ExtractedDocument:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(BytesIO(data))
        # `layout` custa três vezes mais tempo e paga: o modo padrão devolve a
        # página como um bloco corrido, sem a linha em branco que separa um
        # parágrafo do outro, e ainda insere espaço no meio de palavra
        # ("recupera ção"). Medido num livro de 660 páginas: 655 blocos de 550
        # palavras viram 3453 de 104 — parágrafo de verdade, que é a fronteira
        # que o Chunk procura.
        pages = [page.extract_text(extraction_mode="layout") or "" for page in reader.pages]
    except (PdfReadError, ValueError, OSError) as error:
        raise UnsupportedSourceError(
            "pdf_unreadable",
            "Não foi possível ler este PDF: o arquivo parece corrompido ou protegido.",
        ) from error
    if not any(page.strip() for page in pages):
        raise UnsupportedSourceError(
            "pdf_without_text_layer",
            "Este PDF não tem camada de texto — provavelmente é digitalizado. "
            "O harness não faz OCR; passe o arquivo por um OCR antes de subir.",
        )
    cleaned_pages = [_normalized(page) for page in pages]
    furniture = _repeated_page_furniture(cleaned_pages)
    listing = listing_pages(cleaned_pages)
    title = _pdf_title(reader) or _stem(filename)
    blocks: list[SourceBlock] = []
    for number, page in enumerate(cleaned_pages, start=1):
        if number in listing:
            continue
        # A linha vazia é a fronteira de parágrafo, e some se for filtrada junto
        # com a mobília: o que sobrava era a página inteira num bloco só.
        kept: list[str] = []
        for line in page.split("\n"):
            stripped = line.strip()
            if not stripped:
                kept.append("")
                continue
            if furniture_key(stripped) in furniture or _PAGE_FURNITURE.match(stripped):
                continue
            kept.append(stripped)
        for paragraph in _paragraphs(_reflowed("\n".join(kept))):
            blocks.append(SourceBlock(text=paragraph, heading_path=(), page=number))
    return ExtractedDocument(title=title, blocks=tuple(blocks))


def build_document(
    extracted: ExtractedDocument,
    *,
    origin_kind: str,
    origin_ref: str,
    source_digest: str,
    counter: TextTokenCounter,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
    minimum_tokens: int = 8,
) -> DocumentDraft:
    """Cleans the blocks and cuts them into Chunks, offsets included.

    A Chunk never starts in the middle of a paragraph: blocks accumulate until
    the budget is spent, and the overlap is the trailing blocks of the previous
    Chunk, re-included by range instead of copied.

    A section shorter than the floor does not become a Chunk. Its text stays in
    the Document — nothing is thrown away — it just stops competing for one of
    the few passages a Turn can carry, which is what a bare list of links wins
    by matching the question literally while answering nothing.
    """
    blocks = _useful_blocks(extracted.blocks)
    text, spans = _joined(blocks)
    sections = _section_starts(blocks, spans)
    cuts: list[tuple[int, int, int]] = []
    index = 0
    while index < len(blocks):
        # Um bloco sozinho maior que o orçamento é cortado em fim de frase.
        # Medido num livro em PDF: o extrator não marca fim de parágrafo, a
        # limpeza remonta a página inteira como um bloco só, e 92% dos Chunks
        # saíam com o triplo do orçamento — passagem grossa, onde a frase que
        # responde chega diluída e ainda ocupa a vaga de outras duas.
        if counter.count_text(blocks[index].text) > chunk_tokens:
            offset = spans[index][0]
            cuts += [
                (offset + start, offset + stop, index)
                for start, stop, _ in _sentence_pieces(blocks[index].text, counter, chunk_tokens)
            ]
            index += 1
            continue
        used = 0
        end = index
        # Um Chunk pertence a uma seção só: o endereço que ele carrega é o da
        # primeira linha, e atravessar um heading faria a citação apontar para o
        # lugar errado exatamente na metade que interessa conferir.
        section = blocks[index].heading_path
        while end < len(blocks) and blocks[end].heading_path == section:
            block_tokens = counter.count_text(blocks[end].text)
            if used and used + block_tokens > chunk_tokens:
                break
            used += block_tokens
            end += 1
        cuts.append((spans[index][0], spans[end - 1][1], index))
        index = end
    chunks = [
        ChunkDraft(
            start_offset=(
                begin := _overlapped_start(text, sections[owner], start, counter, overlap_tokens)
            ),
            end_offset=stop,
            context_prefix=_context_prefix(extracted.title, blocks[owner]),
            page=blocks[owner].page,
            token_count=counter.count_text(text[begin:stop]),
        )
        for start, stop, owner in cuts
    ]
    # Um Documento que é todo ele mais curto que o piso continua recuperável:
    # a página existe, e o piso está aqui para escolher entre passagens, não
    # para decidir que a página não conta.
    above_floor = [chunk for chunk in chunks if chunk.token_count >= minimum_tokens]
    taints = (UNTRUSTED_WEB_TAINT,) if origin_kind == "scrape" else ()
    return DocumentDraft(
        origin_kind=origin_kind,
        origin_ref=origin_ref,
        title=extracted.title,
        text=text,
        source_digest=source_digest,
        taints=taints,
        chunks=tuple(above_floor or chunks),
    )


def embeddable_texts(draft: DocumentDraft) -> tuple[str, ...]:
    """What actually gets embedded: the address plus the passage."""
    return tuple(
        f"{chunk.context_prefix}\n{draft.text[chunk.start_offset : chunk.end_offset]}"
        for chunk in draft.chunks
    )


def source_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _BlockHTMLParser(HTMLParser):
    """HTML into blocks, keeping the heading path the page already declares.

    The web_fetch extractor flattens a page into lines and inlines link targets,
    which is right for a model reading a page once and wrong for an index: the
    heading structure is exactly the address a citation needs, and `(https://…)`
    inside a passage is noise that gets embedded.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[SourceBlock] = []
        self.title: str = ""
        self._suppressed_depth = 0
        self._heading_level: int | None = None
        self._in_title = False
        self._heading_path: list[str] = []
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.casefold()
        if self._suppressed_depth:
            if lowered in SUPPRESSED_HTML_TAGS:
                self._suppressed_depth += 1
            return
        if lowered in SUPPRESSED_HTML_TAGS:
            self._suppressed_depth = 1
            return
        if lowered == "title":
            self._in_title = True
            return
        if lowered in _HEADING_HTML_TAGS:
            self._flush()
            self._heading_level = _HEADING_HTML_TAGS[lowered]
            return
        if lowered in _BLOCK_HTML_TAGS:
            self._flush()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if self._suppressed_depth:
            if lowered in SUPPRESSED_HTML_TAGS:
                self._suppressed_depth -= 1
            return
        if lowered == "title":
            self._in_title = False
            self.title = self.title.strip()
            return
        if lowered in _HEADING_HTML_TAGS:
            self._close_heading(_HEADING_HTML_TAGS[lowered])
            return
        if lowered in _BLOCK_HTML_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._suppressed_depth:
            return
        if self._in_title:
            self.title += data
            return
        self._buffer.append(data)

    def close(self) -> None:
        super().close()
        if self._heading_level is not None:
            self._close_heading(self._heading_level)
        self._flush()
        self._drop_link_menus()

    def _close_heading(self, level: int) -> None:
        heading = _collapsed("".join(self._buffer))
        self._buffer.clear()
        self._heading_level = None
        if not heading:
            return
        del self._heading_path[level - 1 :]
        self._heading_path.append(heading)
        if not self.title:
            self.title = heading

    def _flush(self) -> None:
        if self._heading_level is not None:
            return
        text = _collapsed("".join(self._buffer))
        self._buffer.clear()
        if text:
            self.blocks.append(
                SourceBlock(text=text, heading_path=tuple(self._heading_path), page=None)
            )

    def _drop_link_menus(self) -> None:
        # A heurística de densidade de links do web_fetch vale igual aqui: uma
        # sequência longa de linhas curtas sem pontuação é menu, não conteúdo.
        kept = set(without_link_menus(block.text for block in self.blocks))
        if len(kept) == len(self.blocks):
            return
        self.blocks = [block for block in self.blocks if block.text in kept]


def _sentence_pieces(
    text: str,
    counter: TextTokenCounter,
    budget: int,
) -> tuple[tuple[int, int, int], ...]:
    """Cuts an oversized block into pieces of whole sentences.

    The rule that a Chunk never starts in the middle of a paragraph exists so a
    citation quotes something that reads on its own. A block that is an entire
    page was never a paragraph — the extractor just could not tell where the
    paragraphs were — and cutting it at the end of a sentence keeps the promise
    the rule was making. A single sentence past the budget is left whole: there
    is nowhere honest left to cut.
    """
    pieces: list[tuple[int, int, int]] = []
    for start, stop in _sentence_spans(text):
        tokens = counter.count_text(text[start:stop])
        if pieces and pieces[-1][2] + tokens <= budget:
            first, _, used = pieces[-1]
            pieces[-1] = (first, stop, used + tokens)
            continue
        pieces.append((start, stop, tokens))
    return tuple(pieces)


def _sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in _SENTENCE_BOUNDARY.finditer(text):
        if boundary.start() > start:
            spans.append((start, boundary.start()))
        start = boundary.end()
    if start < len(text):
        spans.append((start, len(text)))
    return tuple(spans)


def _section_starts(
    blocks: Sequence[SourceBlock],
    spans: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Onde começa a seção de cada bloco — o limite que o overlap não atravessa."""
    starts: list[int] = []
    current = 0
    for position, block in enumerate(blocks):
        if position == 0 or block.heading_path != blocks[position - 1].heading_path:
            current = spans[position][0]
        starts.append(current)
    return tuple(starts)


def _overlapped_start(
    text: str,
    section_start: int,
    start: int,
    counter: TextTokenCounter,
    overlap_tokens: int,
) -> int:
    """Recua o começo do Chunk por frases inteiras, sem sair da seção.

    O overlap existe porque a frase que responde raramente é a primeira: cortada
    do parágrafo anterior, ela chega sem o sujeito de quem se fala. Medido antes,
    só 30% dos Chunks de um livro tinham sobreposição — o recuo era por bloco
    inteiro, e um Chunk feito de um parágrafo só não tinha por onde recuar.
    Frase é a unidade que sempre existe.

    O limite é a seção: atravessá-la faria a passagem começar sob um endereço
    que não é o dela.
    """
    if overlap_tokens <= 0 or start <= section_start:
        return start
    before = text[section_start:start]
    budget = overlap_tokens
    begin = start
    for sentence_start, sentence_end in reversed(_sentence_spans(before)):
        tokens = counter.count_text(before[sentence_start:sentence_end])
        if tokens > budget:
            break
        budget -= tokens
        begin = section_start + sentence_start
    return begin


def _useful_blocks(blocks: Sequence[SourceBlock]) -> tuple[SourceBlock, ...]:
    kept: list[SourceBlock] = []
    seen: set[str] = set()
    for block in blocks:
        text = block.text.strip()
        if len(text) < _MINIMUM_BLOCK_CHARACTERS or _letter_ratio(text) < _MINIMUM_LETTER_RATIO:
            continue
        key = f"{block.heading_path}\x00{text}"
        if key in seen:
            continue
        seen.add(key)
        kept.append(SourceBlock(text=text, heading_path=block.heading_path, page=block.page))
    return tuple(kept)


def _joined(blocks: Sequence[SourceBlock]) -> tuple[str, tuple[tuple[int, int], ...]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    offset = 0
    for block in blocks:
        if parts:
            parts.append("\n\n")
            offset += 2
        parts.append(block.text)
        spans.append((offset, offset + len(block.text)))
        offset += len(block.text)
    return "".join(parts), tuple(spans)


def _context_prefix(title: str, block: SourceBlock) -> str:
    # O H1 de um documento costuma ser o próprio título: repeti-lo no endereço
    # gastaria contexto para dizer duas vezes a mesma coisa.
    path = [heading for heading in block.heading_path if heading != title]
    prefix = " > ".join(part for part in [title, *path] if part)
    if block.page is not None:
        return f"{prefix} — p. {block.page}"
    return prefix


def _repeated_page_furniture(pages: Sequence[str]) -> frozenset[str]:
    """The header and footer that repeat, recognised without their page number.

    A running header carries the number of the page it sits on, so comparing the
    whole line finds nothing: every page has its own. Measured on a textbook,
    `XIV : REDES DE COMPUTADORES E A INTERNET` and `36 • REDES DE COMPUTADORES E
    A INTERNET` are the same furniture, and both survived into the passages.
    """
    if len(pages) < _FURNITURE_PAGE_FLOOR:
        return frozenset()
    counts: dict[str, int] = {}
    for page in pages:
        lines = [line.strip() for line in page.split("\n") if line.strip()]
        for line in {*lines[:2], *lines[-2:]}:
            key = furniture_key(line)
            if key:
                counts[key] = counts.get(key, 0) + 1
    threshold = max(_FURNITURE_PAGE_FLOOR - 1, int(len(pages) * _FURNITURE_SHARE))
    return frozenset(key for key, count in counts.items() if count >= threshold)


def listing_pages(pages: Sequence[str]) -> frozenset[int]:
    """As páginas de sumário e de índice remissivo, por número de página (1-based).

    Elas são texto e passam por qualquer limpeza, mas não respondem nada: o que
    têm é termo e número. Medido num livro, o índice remissivo ocupou uma das
    seis vagas de passagem de um Turn. O que separa listagem de figura cheia de
    número é a corrida: sumário e índice ocupam páginas seguidas, a figura é uma.
    """
    dense = [
        number
        for number, page in enumerate(pages, start=1)
        if len(_LISTING_ENTRY.findall(page)) / max(len(page.split()), 1) >= _LISTING_RATIO
    ]
    listing: set[int] = set()
    run: list[int] = []
    for number in [*dense, 0]:
        if run and number == run[-1] + 1:
            run.append(number)
            continue
        if len(run) >= _LISTING_RUN:
            listing.update(run)
        run = [number]
    return frozenset(listing)


def furniture_key(line: str) -> str:
    """What a header looks like once the page number is taken out of it."""
    words = _FURNITURE_NUMBERING.sub(" ", line).split()
    # O OCR lê o número da página como letra solta — "o" e "g" no lugar de "6" e
    # "9" —, e sobra um token de uma letra numa das pontas, que não é palavra.
    if words and len(words[0]) == 1:
        words = words[1:]
    if words and len(words[-1]) == 1:
        words = words[:-1]
    return " ".join(words).casefold()


def _pdf_title(reader: object) -> str:
    metadata = getattr(reader, "metadata", None)
    title = getattr(metadata, "title", None)
    return str(title).strip() if isinstance(title, str) else ""


def _decoded(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _normalized(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = _CONTROL_CHARACTERS.sub("", normalized)
    lines = "\n".join(_HORIZONTAL_SPACE.sub(" ", line).strip() for line in normalized.split("\n"))
    # Hífen suave no fim da linha é hifenização de quebra, igual ao hífen comum
    # que `_reflowed` já junta. Apagá-lo antes deixava a quebra órfã: o livro em
    # PDF entregava "pode\xad\nmos" e o Corpus indexava "pode mos".
    return lines.replace("­\n", "").replace("­", "")


def _reflowed(text: str) -> str:
    """Undoes hard wrapping so a paragraph is one line again."""
    joined = _LINE_BREAK_HYPHEN.sub(r"\1\2", text)
    lines = joined.split("\n")
    out: list[str] = []
    for line in lines:
        if line.strip() and out and out[-1].strip():
            out[-1] = f"{out[-1]} {line.strip()}"
            continue
        out.append(line)
    return "\n".join(out)


def _paragraphs(text: str) -> Iterable[str]:
    for piece in text.split("\n"):
        stripped = piece.strip()
        if stripped:
            yield stripped


def _collapsed(text: str) -> str:
    return " ".join(text.split())


def _letter_ratio(text: str) -> float:
    letters = sum(1 for character in text if character.isalpha())
    return letters / len(text) if text else 0.0


def _stem(filename: str) -> str:
    name = filename.rsplit("/", 1)[-1]
    for suffix in ACCEPTED_EXTENSIONS:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


__all__ = [
    "ACCEPTED_EXTENSIONS",
    "ExtractedDocument",
    "SourceBlock",
    "UnsupportedSourceError",
    "accepted_extension",
    "build_document",
    "embeddable_texts",
    "extract",
    "extract_html",
    "extract_markdown",
    "extract_pdf",
    "extract_plain",
    "furniture_key",
    "listing_pages",
    "source_digest",
]
