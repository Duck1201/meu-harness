import asyncio
import shutil
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from pypdf import PdfWriter
from test_corpus_store import DIMENSIONS, HashingEmbedder, WordCounter

from harness.config import load_config
from harness.corpus_ocr import PdftoppmRenderer, clean_transcription, transcribe_pdf
from harness.corpus_service import (
    OCR_PASSAGE_INSTRUCTION,
    CorpusIngestionService,
    CorpusLibrary,
    CorpusRetriever,
    IngestionJob,
    IngestionJobStatus,
)
from harness.domain import OCR_TRANSCRIBED_TAINT

_PAGE_TEXT = (
    "Capitulo 1\n\nO roteador encaminha pacotes entre redes e escolhe o proximo salto "
    "pela tabela de encaminhamento, que o protocolo de roteamento preenche."
)


def _scanned_pdf(pages: int = 2) -> bytes:
    """Um PDF só com páginas em branco: sem camada de texto, como um escaneado."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=300, height=400)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class FakeEmbedder:
    model = "fake"
    dimensions = DIMENSIONS

    async def embed(self, texts: object) -> tuple[tuple[float, ...], ...]:
        return HashingEmbedder().embed(cast(list[str], texts))


class FakeRenderer:
    def __init__(self) -> None:
        self.pages: list[int] = []

    async def render(self, pdf: Path, page: int) -> bytes:
        assert pdf.is_file()
        self.pages.append(page)
        return f"png-{page}".encode()


class FakeOcr:
    async def transcribe(self, image: bytes) -> str:
        return f"{_PAGE_TEXT} Pagina {image.decode().split('-')[1]}."


def test_the_transcription_loses_the_fence_and_the_loop_but_keeps_the_page() -> None:
    """Medido na capa do Kurose: 11082 caracteres de laço voltaram ao que a página diz."""
    page = "KUROSE | ROSS\nRedes de computadores e a internet, uma abordagem top-down, 6a edicao"
    looped = f"{page}\nALWAYS LEARNING PEARSON\n```markdown\n" + f"{page}\n" * 30

    cleaned = clean_transcription(looped)
    # O corte acontece na primeira repetição de um bloco de 120 caracteres: uma
    # ou duas voltas do laço ficam, as outras 28 somem.
    assert cleaned.startswith(f"{page}\nALWAYS LEARNING PEARSON")
    assert cleaned.count(page) <= 3
    assert len(cleaned) < len(looped) / 5
    assert "```" not in cleaned
    # Frase curta que o livro repete de verdade não é laço.
    assert clean_transcription("Figura 2.1\n\nTexto.\n\nFigura 2.1") == (
        "Figura 2.1\n\nTexto.\n\nFigura 2.1"
    )


def test_every_page_is_rendered_and_read_in_order() -> None:
    renderer = FakeRenderer()
    seen: list[int] = []

    async def on_page(page: int) -> None:
        seen.append(page)

    texts = asyncio.run(
        transcribe_pdf(_scanned_pdf(3), 3, renderer=renderer, ocr=FakeOcr(), on_page=on_page)
    )

    assert renderer.pages == [1, 2, 3]
    assert seen == [1, 2, 3]
    assert texts[2].endswith("Pagina 3.")


@pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="pdftoppm is not installed")
def test_pdftoppm_renders_a_page_as_png(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(_scanned_pdf(1))
    renderer = PdftoppmRenderer(cast(str, shutil.which("pdftoppm")), dpi=50)

    image = asyncio.run(renderer.render(pdf, 1))

    assert image.startswith(b"\x89PNG\r\n\x1a\n")


def _service(tmp_path: Path, *, with_ocr: bool) -> tuple[CorpusIngestionService, CorpusLibrary]:
    library = CorpusLibrary(
        tmp_path / "corpora", embedding_model="fake", embedding_dimensions=DIMENSIONS
    )
    service = CorpusIngestionService(
        library=library,
        embedder=FakeEmbedder(),
        counter=WordCounter(),
        config=load_config().corpus,
        page_renderer=FakeRenderer() if with_ocr else None,
        page_ocr=FakeOcr() if with_ocr else None,
    )
    return service, library


async def _until_done(service: CorpusIngestionService, job: IngestionJob) -> IngestionJob:
    for _ in range(200):
        current = service.job(job.id)
        assert current is not None
        if current.status in {IngestionJobStatus.COMPLETED, IngestionJobStatus.FAILED}:
            return current
        await asyncio.sleep(0.01)
    raise AssertionError("OCR job did not finish")


def test_a_scanned_pdf_becomes_an_ocr_job_and_its_passages_say_so(tmp_path: Path) -> None:
    async def scenario() -> tuple[IngestionJob, IngestionJob, dict[str, object]]:
        service, library = _service(tmp_path, with_ocr=True)
        corpus = await library.create(name="Livro escaneado")
        started = await service.ingest_upload(corpus.id, "livro.pdf", _scanned_pdf(2))
        finished = await _until_done(service, started)
        config = load_config().corpus
        retriever = CorpusRetriever(
            library=library,
            embedder=FakeEmbedder(),
            counter=WordCounter(),
            config=config.model_copy(
                update={
                    "retrieval": config.retrieval.model_copy(update={"dense_similarity_floor": 0.2})
                }
            ),
        )
        retrieval = await retriever.retrieve(corpus.id, "como o roteador escolhe o proximo salto")
        return started, finished, cast(dict[str, object], retrieval.payload())

    started, finished, payload = asyncio.run(scenario())

    # A subida não espera: um livro escaneado leva horas.
    assert started.kind == "ocr"
    assert started.status is IngestionJobStatus.RUNNING
    assert finished.status is IngestionJobStatus.COMPLETED, finished.detail
    assert finished.seen == 2
    assert finished.chunks >= 1
    passages = cast(list[dict[str, object]], payload["passages"])
    assert passages
    assert all(passage["transcribed_by_ocr"] is True for passage in passages)
    assert OCR_PASSAGE_INSTRUCTION in str(payload["instruction"])


def test_without_ocr_a_scanned_pdf_is_still_refused_with_what_is_missing(tmp_path: Path) -> None:
    async def scenario() -> IngestionJob:
        service, library = _service(tmp_path, with_ocr=False)
        corpus = await library.create(name="Livro escaneado")
        return await service.ingest_upload(corpus.id, "livro.pdf", _scanned_pdf(1))

    job = asyncio.run(scenario())

    assert job.status is IngestionJobStatus.FAILED
    assert job.reason_code == "pdf_without_text_layer"
    assert "pdftoppm" in str(job.detail)


def test_ocr_documents_carry_the_transcription_taint(tmp_path: Path) -> None:
    async def scenario() -> tuple[str, ...]:
        service, library = _service(tmp_path, with_ocr=True)
        corpus = await library.create(name="Livro escaneado")
        await _until_done(service, await service.ingest_upload(corpus.id, "l.pdf", _scanned_pdf(1)))
        documents = await library.store(corpus.id).list_documents()
        return documents[0].taints

    assert asyncio.run(scenario()) == (OCR_TRANSCRIBED_TAINT,)
