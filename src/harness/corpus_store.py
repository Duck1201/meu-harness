"""One SQLite file per Corpus, and the hybrid search over it.

The directory is the index: each file carries its own meta row, so listing is
reading the files and deleting a Corpus is deleting one file. A registry beside
them would be a second place to disagree with the disk.

Retrieval fuses two rankings that fail in opposite ways. The dense leg finds a
passage that says the same thing in another language or another wording; the
lexical leg finds the exact identifier, error string or number that an embedding
smooths away. Reciprocal rank fusion is what merges them without inventing a
weight nobody measured.
"""

import asyncio
import json
import re
import sqlite3
import struct
from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Concatenate, cast
from uuid import uuid4

# `sqlite-vec` distribui a extensão como wheel e não publica stubs; o pyright em
# modo strict recusaria o import sem esta linha, e escrever um stub para uma função
# só (`load`) seria manutenção sem leitor.
import sqlite_vec  # pyright: ignore[reportMissingTypeStubs]

from .domain import Corpus, Document, RetrievedChunk

_NAME_LIMIT = 120
_DESCRIPTION_LIMIT = 2000
# Um id de Corpus vira nome de arquivo, então ele não pode ser texto livre.
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
# FTS5 lê a query como sintaxe própria: aspas, `*`, `NEAR`, `-` e `:` mudariam o
# sentido do que o Operator escreveu, e um parêntese solto derruba a consulta.
_FTS_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


class CorpusStoreError(Exception):
    pass


class CorpusNotFoundError(CorpusStoreError):
    pass


class EmbeddingMismatchError(CorpusStoreError):
    """The Corpus was built with another embedding model or width.

    Never recoverable by writing anyway: vectors from two models share a space
    only by coincidence, and a mixed index ranks by nothing at all.
    """


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A Chunk before it exists: offsets into the Document, plus its address."""

    start_offset: int
    end_offset: int
    context_prefix: str
    page: int | None
    token_count: int


@dataclass(frozen=True, slots=True)
class DocumentDraft:
    origin_kind: str
    origin_ref: str
    title: str
    text: str
    source_digest: str
    taints: tuple[str, ...]
    chunks: tuple[ChunkDraft, ...]


def _offload[**P, R](
    method: Callable[Concatenate["CorpusStore", P], R],
) -> Callable[Concatenate["CorpusStore", P], Coroutine[None, None, R]]:
    @wraps(method)
    async def offloaded(self: "CorpusStore", *args: P.args, **kwargs: P.kwargs) -> R:
        return await asyncio.to_thread(method, self, *args, **kwargs)

    return offloaded


class CorpusStore:
    """The store of a single Corpus, addressed by its file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        # Por id de Documento, que nunca é reaproveitado: reingerir cria outro id.
        self._exercise_cache: dict[str, tuple[tuple[int, int], ...]] = {}

    @property
    def path(self) -> Path:
        return self._path

    @property
    def corpus_id(self) -> str:
        return self._path.stem

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.enable_load_extension(True)
        try:
            sqlite_vec.load(connection)
        finally:
            connection.enable_load_extension(False)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @_offload
    def create(
        self,
        *,
        name: str,
        description: str,
        embedding_model: str,
        embedding_dimensions: int,
    ) -> Corpus:
        now = _now()
        with self._connect() as connection:
            self._create_schema(connection, embedding_dimensions)
            connection.execute(
                """
                INSERT INTO corpus_meta(
                    id, name, description, embedding_model, embedding_dimensions,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.corpus_id,
                    _validated_name(name),
                    _validated_description(description),
                    embedding_model,
                    embedding_dimensions,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        return self._read_meta_or_raise()

    @_offload
    def read(self) -> Corpus:
        return self._read_meta_or_raise()

    @_offload
    def rename(self, *, name: str | None, description: str | None) -> Corpus:
        with self._connect() as connection:
            if name is not None:
                connection.execute(
                    "UPDATE corpus_meta SET name = ?, updated_at = ?",
                    (_validated_name(name), _now().isoformat()),
                )
            if description is not None:
                connection.execute(
                    "UPDATE corpus_meta SET description = ?, updated_at = ?",
                    (_validated_description(description), _now().isoformat()),
                )
        return self._read_meta_or_raise()

    @_offload
    def list_documents(self, *, limit: int = 200, offset: int = 0) -> tuple[Document, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.*, COUNT(c.id) AS chunk_count
                FROM documents AS d
                LEFT JOIN chunks AS c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.ingested_at DESC, d.id
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return tuple(_document(row) for row in rows)

    @_offload
    def indexed_digests(self) -> frozenset[str]:
        """What the store already has, so a redispatched job can skip it."""
        with self._connect() as connection:
            rows = connection.execute("SELECT source_digest FROM documents").fetchall()
        return frozenset(str(row["source_digest"]) for row in rows)

    @_offload
    def indexed_origins(self) -> Mapping[str, str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT origin_ref, source_digest FROM documents").fetchall()
        return {str(row["origin_ref"]): str(row["source_digest"]) for row in rows}

    @_offload
    def delete_document(self, document_id: str) -> None:
        with self._connect() as connection:
            self._delete_document(connection, document_id)

    @_offload
    def add_document(
        self,
        draft: DocumentDraft,
        embeddings: Sequence[Sequence[float]],
    ) -> Document:
        """Writes one Document and its Chunks in a single transaction.

        The commit unit is the Document on purpose: a crawl of ten thousand pages
        that dies at page nine thousand keeps nine thousand, and redispatching it
        skips them by digest instead of starting over.
        """
        if len(embeddings) != len(draft.chunks):
            raise CorpusStoreError("every chunk needs exactly one embedding")
        now = _now()
        document_id = uuid4().hex
        with self._connect() as connection:
            meta = _meta_row(connection)
            if meta is None:
                raise CorpusNotFoundError(f"corpus {self.corpus_id} does not exist")
            dimensions = int(meta["embedding_dimensions"])
            for vector in embeddings:
                if len(vector) != dimensions:
                    raise EmbeddingMismatchError(
                        f"corpus {self.corpus_id} indexes {dimensions} dimensions, "
                        f"received {len(vector)}"
                    )
            existing = connection.execute(
                "SELECT id FROM documents WHERE source_digest = ?",
                (draft.source_digest,),
            ).fetchone()
            if existing is not None:
                self._delete_document(connection, str(existing["id"]))
            connection.execute(
                """
                INSERT INTO documents(
                    id, origin_kind, origin_ref, title, text, source_digest, taints, ingested_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    draft.origin_kind,
                    draft.origin_ref,
                    draft.title,
                    draft.text,
                    draft.source_digest,
                    json.dumps(list(draft.taints)),
                    now.isoformat(),
                ),
            )
            for ordinal, (chunk, vector) in enumerate(zip(draft.chunks, embeddings, strict=True)):
                chunk_id = uuid4().hex
                text = draft.text[chunk.start_offset : chunk.end_offset]
                connection.execute(
                    """
                    INSERT INTO chunks(
                        id, document_id, ordinal, start_offset, end_offset,
                        context_prefix, page, token_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        document_id,
                        ordinal,
                        chunk.start_offset,
                        chunk.end_offset,
                        chunk.context_prefix,
                        chunk.page,
                        chunk.token_count,
                    ),
                )
                connection.execute(
                    "INSERT INTO chunks_fts(chunk_id, indexed_text) VALUES (?, ?)",
                    (chunk_id, f"{chunk.context_prefix}\n{text}"),
                )
                connection.execute(
                    "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, _packed(vector)),
                )
            connection.execute(
                "UPDATE corpus_meta SET updated_at = ?",
                (now.isoformat(),),
            )
            row = connection.execute(
                """
                SELECT d.*, COUNT(c.id) AS chunk_count
                FROM documents AS d
                LEFT JOIN chunks AS c ON c.document_id = d.id
                WHERE d.id = ?
                """,
                (document_id,),
            ).fetchone()
        return _document(row)

    @_offload
    def search(
        self,
        *,
        dense_query: Sequence[float] | None,
        lexical_queries: Sequence[str],
        limit: int,
        dense_candidates: int = 40,
        lexical_candidates: int = 40,
        rank_constant: int = 60,
        similarity_floor: float = 0.0,
    ) -> tuple[RetrievedChunk, ...]:
        """Fuses the two rankings, but only after something clears the floor.

        Two different numbers, doing two different jobs. Reciprocal rank fusion
        *orders*: it rewards a Chunk that both legs surfaced, and it says nothing
        at all about relevance — the first place always scores 1/(k+1), whether
        the passage answers the question or is the least bad thing in the store.

        Cosine similarity *measures*, so the floor is read there — e por passagem,
        não pela busca inteira. Medido contra um livro: bastava um Chunk passar
        do piso para toda a fusão entrar, e três das seis vagas voltavam com
        referência bibliográfica que a perna lexical achou e ninguém mediu. O
        piso vale para quem for citado, inclusive quem só o BM25 encontrou.
        """
        with self._connect() as connection:
            similarities: dict[str, float] = {}
            rankings: list[list[str]] = []
            if dense_query is not None:
                dense = self._dense_ranking(connection, dense_query, dense_candidates)
                similarities = dict(dense)
                rankings.append([chunk_id for chunk_id, _ in dense])
            for query in lexical_queries:
                ranking = self._lexical_ranking(connection, query, lexical_candidates)
                if ranking:
                    rankings.append(ranking)
            scores: dict[str, float] = {}
            for ranking in rankings:
                for position, chunk_id in enumerate(ranking):
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                        rank_constant + position + 1
                    )
            scores = {
                chunk_id: score
                for chunk_id, score in scores.items()
                if chunk_id not in self._exercise_chunks(connection, list(scores))
            }
            if dense_query is not None:
                similarities |= self._similarities(
                    connection,
                    dense_query,
                    [chunk_id for chunk_id in scores if chunk_id not in similarities],
                )
                scores = {
                    chunk_id: score
                    for chunk_id, score in scores.items()
                    if similarities.get(chunk_id, 0.0) >= similarity_floor
                }
            ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
            return tuple(self._hydrate(connection, ordered, similarities))

    def _exercise_chunks(
        self, connection: sqlite3.Connection, chunk_ids: Sequence[str]
    ) -> frozenset[str]:
        """Os candidatos que caem numa seção de exercícios do seu documento.

        Medido no Kurose (2026-09-29): as atividades do Operator eram as próprias
        Questões de revisão do livro, então a página que lista R1 a R9 ganhava a
        fusão por conter a pergunta palavra por palavra — e o juiz dava 1,0 a
        ela — em 3 de 5 perguntas, sem nunca responder nenhuma. Uma lista de
        perguntas não é passagem que sustente resposta.
        """
        if not chunk_ids:
            return frozenset()
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = connection.execute(
            f"SELECT id, document_id, start_offset, end_offset FROM chunks "
            f"WHERE id IN ({placeholders})",
            tuple(chunk_ids),
        ).fetchall()
        excluded: set[str] = set()
        for row in rows:
            regions = self._exercise_regions_for(connection, str(row["document_id"]))
            start, end = int(row["start_offset"]), int(row["end_offset"])
            # Maioria, não início: o Chunk que cruza a fronteira carrega a abertura
            # do capítulo seguinte, e é conteúdo quando a maior parte está fora.
            inside = sum(max(0, min(end, b) - max(start, a)) for a, b in regions)
            if inside * 2 > end - start:
                excluded.add(str(row["id"]))
        return frozenset(excluded)

    def _exercise_regions_for(
        self, connection: sqlite3.Connection, document_id: str
    ) -> tuple[tuple[int, int], ...]:
        cached = self._exercise_cache.get(document_id)
        if cached is not None:
            return cached
        row = connection.execute(
            "SELECT text FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        regions = exercise_regions(str(row["text"])) if row is not None else ()
        self._exercise_cache[document_id] = regions
        return regions

    def exists(self) -> bool:
        return self._path.is_file()

    def delete_file(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = self._path.with_name(self._path.name + suffix)
            candidate.unlink(missing_ok=True)

    def _dense_ranking(
        self,
        connection: sqlite3.Connection,
        query: Sequence[float],
        candidates: int,
    ) -> list[tuple[str, float]]:
        rows = connection.execute(
            """
            SELECT chunk_id, distance FROM chunks_vec
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (_packed(query), candidates),
        ).fetchall()
        # A coluna é declarada com distance_metric=cosine, então a distância é
        # 1 - similaridade e volta para a escala em que o piso foi medido.
        return [(str(row["chunk_id"]), 1.0 - float(row["distance"])) for row in rows]

    def _similarities(
        self,
        connection: sqlite3.Connection,
        query: Sequence[float],
        chunk_ids: Sequence[str],
    ) -> dict[str, float]:
        """A similaridade de quem a perna lexical trouxe e a densa não mediu."""
        if not chunk_ids:
            return {}
        # O único texto interpolado é a sequência de "?": os ids viajam ligados.
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = connection.execute(
            f"""
            SELECT chunk_id, vec_distance_cosine(embedding, ?) AS distance
            FROM chunks_vec WHERE chunk_id IN ({placeholders})
            """,
            (_packed(query), *chunk_ids),
        ).fetchall()
        return {str(row["chunk_id"]): 1.0 - float(row["distance"]) for row in rows}

    def _lexical_ranking(
        self,
        connection: sqlite3.Connection,
        query: str,
        candidates: int,
    ) -> list[str]:
        expression = _fts_expression(query)
        if not expression:
            return []
        rows = connection.execute(
            """
            SELECT chunk_id FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            (expression, candidates),
        ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    def _hydrate(
        self,
        connection: sqlite3.Connection,
        selected: Sequence[tuple[str, float]],
        similarities: Mapping[str, float],
    ) -> Iterable[RetrievedChunk]:
        for chunk_id, score in selected:
            row = connection.execute(
                """
                SELECT c.id, c.document_id, c.start_offset, c.end_offset,
                       c.context_prefix, c.page,
                       d.title, d.origin_kind, d.origin_ref, d.taints,
                       substr(d.text, c.start_offset + 1, c.end_offset - c.start_offset) AS text
                FROM chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                WHERE c.id = ?
                """,
                (chunk_id,),
            ).fetchone()
            if row is None:
                continue
            yield RetrievedChunk(
                id=str(row["id"]),
                document_id=str(row["document_id"]),
                document_title=str(row["title"]),
                origin_kind=str(row["origin_kind"]),
                origin_ref=str(row["origin_ref"]),
                location=str(row["context_prefix"]),
                text=str(row["text"]),
                score=score,
                taints=_taints(row["taints"]),
                similarity=similarities.get(chunk_id),
            )

    def _delete_document(self, connection: sqlite3.Connection, document_id: str) -> None:
        # FTS5 e vec0 são tabelas virtuais: nenhuma delas participa de foreign key,
        # então o cascade tem que ser escrito à mão ou o índice fica apontando para
        # Chunks que não existem mais.
        chunk_ids = [
            str(row["id"])
            for row in connection.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchall()
        ]
        for chunk_id in chunk_ids:
            connection.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
            connection.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (chunk_id,))
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def _create_schema(self, connection: sqlite3.Connection, dimensions: int) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS corpus_meta (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dimensions INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                origin_kind TEXT NOT NULL CHECK (origin_kind IN ('upload', 'scrape')),
                origin_ref TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                source_digest TEXT NOT NULL UNIQUE,
                taints TEXT NOT NULL,
                ingested_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id),
                ordinal INTEGER NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                context_prefix TEXT NOT NULL,
                page INTEGER,
                token_count INTEGER NOT NULL,
                UNIQUE (document_id, ordinal)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                indexed_text,
                chunk_id UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            """
        )
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
            f"chunk_id TEXT PRIMARY KEY, embedding float[{dimensions}] distance_metric=cosine)"
        )

    def _read_meta_or_raise(self) -> Corpus:
        with self._connect() as connection:
            row = _meta_row(connection)
            if row is None:
                raise CorpusNotFoundError(f"corpus {self.corpus_id} does not exist")
            counts = connection.execute(
                "SELECT (SELECT COUNT(*) FROM documents) AS documents, "
                "(SELECT COUNT(*) FROM chunks) AS chunks"
            ).fetchone()
        return Corpus(
            id=str(row["id"]),
            name=str(row["name"]),
            description=str(row["description"]),
            embedding_model=str(row["embedding_model"]),
            embedding_dimensions=int(row["embedding_dimensions"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            document_count=int(counts["documents"]),
            chunk_count=int(counts["chunks"]),
        )


def valid_corpus_id(value: str) -> bool:
    return bool(_ID_PATTERN.fullmatch(value))


def corpus_id_from_name(name: str, taken: Iterable[str] = ()) -> str:
    """A filesystem-safe id derived from the name the Operator typed."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")[:40]
    base = slug or "corpus"
    if not _ID_PATTERN.fullmatch(base):
        base = f"c-{base}"[:63]
    existing = set(taken)
    if base not in existing:
        return base
    for suffix in range(2, 1000):
        candidate = f"{base}-{suffix}"[:63]
        if candidate not in existing:
            return candidate
    return f"{base[:50]}-{uuid4().hex[:8]}"


def _meta_row(connection: sqlite3.Connection) -> sqlite3.Row | None:
    try:
        return connection.execute("SELECT * FROM corpus_meta LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return None


def _document(row: sqlite3.Row) -> Document:
    return Document(
        id=str(row["id"]),
        origin_kind=str(row["origin_kind"]),
        origin_ref=str(row["origin_ref"]),
        title=str(row["title"]),
        source_digest=str(row["source_digest"]),
        taints=_taints(row["taints"]),
        ingested_at=datetime.fromisoformat(str(row["ingested_at"])),
        chunk_count=int(row["chunk_count"]),
    )


def _taints(raw: object) -> tuple[str, ...]:
    decoded: object = json.loads(str(raw))
    if not isinstance(decoded, list):
        return ()
    items = cast(list[object], decoded)
    return tuple(str(item) for item in items)


def _packed(vector: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


# Onde começa o bloco de fim de capítulo e onde começa o capítulo seguinte. As
# duas grafias de cada: o acervo é do Operator e pode ser em português ou inglês.
_EXERCISE_HEADING = re.compile(
    r"^(?:quest[õo]es de revis[ãa]o|exerc[íi]cios de fixa[çc][ãa]o|exerc[íi]cios|"
    r"review questions|homework problems|exercises)\b[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_CHAPTER_OPENING = re.compile(
    r"^(?:\d{1,2}\.1\s+\S[^\n]*|(?:cap[íi]tulo|chapter)\s+\d{1,2}\b[^\n]*)$",
    re.IGNORECASE | re.MULTILINE,
)
# Uma região de exercícios pergunta; prosa quase não. Sem esse mínimo, um título
# "Exercícios" solto no meio de um manual esconderia o manual.
_EXERCISE_MIN_QUESTIONS = 5


def exercise_regions(text: str) -> tuple[tuple[int, int], ...]:
    """Trechos de fim de capítulo que só perguntam: exercícios, revisão, problemas.

    Vão do título até o parágrafo da última pergunta antes do capítulo seguinte.
    Sem abertura de capítulo depois, a
    região não é marcada: fechar "até o fim do documento" apagaria da busca tudo
    o que viesse depois de um título mal reconhecido.
    """
    regions: list[tuple[int, int]] = []
    for heading in _EXERCISE_HEADING.finditer(text):
        if regions and heading.start() < regions[-1][1]:
            continue
        opening = _CHAPTER_OPENING.search(text, heading.end())
        if opening is None:
            continue
        if text.count("?", heading.start(), opening.start()) < _EXERCISE_MIN_QUESTIONS:
            continue
        # A região acaba no parágrafo da última pergunta, não na abertura do
        # capítulo seguinte: entre os dois fica a introdução desse capítulo, que é
        # conteúdo. No Kurose, fechar na abertura escondia a do Capítulo 2.
        last_question = text.rfind("?", heading.start(), opening.start())
        paragraph_end = text.find("\n\n", last_question, opening.start())
        regions.append((heading.start(), opening.start() if paragraph_end < 0 else paragraph_end))
    return tuple(regions)


def _fts_expression(query: str) -> str:
    tokens = [match.group(0) for match in _FTS_TOKEN.finditer(query)]
    return " OR ".join(f'"{token}"' for token in tokens[:32])


def _validated_name(name: str) -> str:
    stripped = name.strip()
    if not stripped:
        raise CorpusStoreError("um Corpus precisa de nome")
    return stripped[:_NAME_LIMIT]


def _validated_description(description: str) -> str:
    return description.strip()[:_DESCRIPTION_LIMIT]


def _now() -> datetime:
    return datetime.now(UTC)
