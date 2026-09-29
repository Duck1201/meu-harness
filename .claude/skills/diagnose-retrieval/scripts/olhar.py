"""O que a recuperação devolve para uma pergunta, com os parâmetros do contrato.

    uv run python .claude/skills/diagnose-retrieval/scripts/olhar.py "pergunta" [corpus] [reescrita]

Sem `corpus`, usa o primeiro acervo do `state_dir`. A terceira posição é a
reescrita em inglês, para reproduzir a perna lexical como ela roda no turno.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from harness.config import load_config
from harness.corpus_store import CorpusStore
from harness.ollama_runtime import OllamaEmbeddingRuntime

HOST = json.loads((Path.home() / ".config/meu-harness/host.json").read_text())
CORPORA = Path(HOST["state_dir"]) / "corpora"

PERGUNTA = sys.argv[1]
CORPUS = sys.argv[2] if len(sys.argv) > 2 else sorted(CORPORA.glob("*.sqlite3"))[0].stem
INGLES = sys.argv[3] if len(sys.argv) > 3 else None


async def main() -> None:
    config = load_config()
    profile = next(p for p in config.model_profiles.runtime_profiles if p.embedding)
    assert profile.embedding is not None
    embedder = OllamaEmbeddingRuntime(
        base_url=HOST["ollama_url"],
        model=profile.embedding.id,
        expected_digest=profile.embedding.digest_sha256,
        dimensions=profile.embedding.dimensions,
    )
    store = CorpusStore(CORPORA / f"{CORPUS}.sqlite3")
    settings = config.corpus.retrieval
    vector = (await embedder.embed([PERGUNTA]))[0]
    lexical = [q for q in (PERGUNTA, INGLES) if q]
    found = await store.search(
        dense_query=vector,
        lexical_queries=lexical,
        limit=settings.injected_passages,
        dense_candidates=settings.dense_candidates,
        lexical_candidates=settings.lexical_candidates,
        rank_constant=settings.reciprocal_rank_constant,
        similarity_floor=settings.dense_similarity_floor,
    )
    print(f"{PERGUNTA!r} -> {len(found)} passagens (piso {settings.dense_similarity_floor})\n")
    for position, chunk in enumerate(found, 1):
        similarity = f"{chunk.similarity:.3f}" if chunk.similarity is not None else "SEM"
        print(f"[{position}] sim={similarity} rrf={chunk.score:.4f} | {chunk.location}")
        print(f"    {chunk.text[:400]}\n")


asyncio.run(main())
