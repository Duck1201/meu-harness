from collections.abc import Sequence
from pathlib import Path

from fastapi.testclient import TestClient
from test_corpus_store import DIMENSIONS, HashingEmbedder

from harness import (
    ApplicationService,
    ConversationStore,
    EngineReadiness,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ObservabilityStore,
    ToolSchema,
    create_app,
    load_config,
)

LOOPBACK = ("127.0.0.1", 51000)


class FakeEstimator:
    validated = True
    readiness = EngineReadiness(ready=True)

    def estimate(self, messages: Sequence[ModelMessage], tools: Sequence[ToolSchema]) -> int:
        return len(messages) + len(tools)

    def count_text(self, text: str) -> int:
        return max(1, len(text.split()))


class FakeRuntime:
    async def verify_profile(self) -> EngineReadiness:
        return EngineReadiness(ready=True)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        del request
        return ModelResponse(content="ok")

    async def aclose(self) -> None:
        return None


class FakeEmbedder:
    model = "fake"
    dimensions = DIMENSIONS

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return HashingEmbedder().embed(texts)


def _app(tmp_path: Path, *, with_embedder: bool = True) -> TestClient:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    config = load_config()
    # O perfil declara bge-m3 com 1024 dimensões; o embedder falso tem outra
    # largura, e um Corpus criado com a largura errada recusaria todo vetor.
    profile = config.runtime_profile
    assert profile.embedding is not None
    patched = config.model_copy(
        update={
            "model_profiles": config.model_profiles.model_copy(
                update={
                    "runtime_profiles": (
                        profile.model_copy(
                            update={
                                "embedding": profile.embedding.model_copy(
                                    update={"dimensions": DIMENSIONS}
                                )
                            }
                        ),
                    )
                }
            )
        }
    )
    service = ApplicationService(
        store=ConversationStore(tmp_path / "conversations.sqlite3"),
        observability_store=ObservabilityStore(tmp_path / "observability.sqlite3"),
        config=patched,
        runtime=FakeRuntime(),
        estimator=FakeEstimator(),
        allowed_workspace_roots=(workspace,),
        corpus_directory=tmp_path / "corpora",
        embedder=FakeEmbedder() if with_embedder else None,
    )
    app = create_app(
        service=service,
        allowed_origins=("http://operator.test",),
        static_dir=tmp_path / "missing-dist",
    )
    return TestClient(app, client=LOOPBACK)


def test_a_corpus_is_created_listed_renamed_and_deleted_as_one_file(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        created = client.post("/api/corpora", json={"name": "Wiki do jogo"}).json()["corpus"]
        listed = client.get("/api/corpora").json()["corpora"]
        renamed = client.patch(
            f"/api/corpora/{created['id']}",
            json={"name": "Wiki renomeada"},
        ).json()["corpus"]
        deleted = client.delete(f"/api/corpora/{created['id']}")
        empty = client.get("/api/corpora").json()["corpora"]

    assert created["id"] == "wiki-do-jogo"
    assert created["document_count"] == 0
    assert [item["id"] for item in listed] == ["wiki-do-jogo"]
    assert renamed["name"] == "Wiki renomeada"
    assert deleted.status_code == 204
    assert empty == []
    assert not (tmp_path / "corpora" / "wiki-do-jogo.sqlite3").exists()


def test_an_upload_becomes_documents_and_a_refused_type_answers_in_portuguese(
    tmp_path: Path,
) -> None:
    with _app(tmp_path) as client:
        corpus = client.post("/api/corpora", json={"name": "Manual"}).json()["corpus"]
        accepted = client.post(
            f"/api/corpora/{corpus['id']}/documents",
            files={
                "file": ("manual.md", b"# Manual\n\n## Rede\n\nO proxy escuta na porta 8899.\n")
            },
        ).json()["job"]
        refused = client.post(
            f"/api/corpora/{corpus['id']}/documents",
            files={"file": ("planilha.xlsx", b"qualquer coisa")},
        ).json()["job"]
        documents = client.get(f"/api/corpora/{corpus['id']}/documents").json()["documents"]

    assert accepted["status"] == "completed"
    assert accepted["indexed"] == 1
    assert refused["status"] == "failed"
    assert refused["reason_code"] == "unsupported_extension"
    assert ".pdf" in refused["detail"]
    assert [item["origin_ref"] for item in documents] == ["manual.md"]
    assert documents[0]["taints"] == []


def test_selecting_a_corpus_is_the_grant_and_desligado_revokes_it(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        corpus = client.post("/api/corpora", json={"name": "Manual"}).json()["corpus"]
        workspace = client.get("/api/workspaces").json()["workspaces"][0]["root"]
        conversation = client.post(
            "/api/conversations",
            json={"workspace_root": workspace},
        ).json()["conversation"]

        selected = client.put(
            f"/api/conversations/{conversation['id']}/corpus",
            json={"corpus_id": corpus["id"]},
        ).json()["grant"]
        snapshot = client.get(f"/api/ui/chat?conversation_id={conversation['id']}").json()
        grants = client.get(f"/api/conversations/{conversation['id']}/grants").json()["grants"]
        cleared = client.put(
            f"/api/conversations/{conversation['id']}/corpus",
            json={"corpus_id": None},
        ).json()["grant"]
        after = client.get(f"/api/ui/chat?conversation_id={conversation['id']}").json()

    # A seleção é o grant: o escopo diz qual acervo, e some da lista ao desligar.
    assert selected["permission"] == "CorpusGrant"
    assert selected["scope"] == corpus["id"]
    assert snapshot["corpus_id"] == corpus["id"]
    assert "CorpusGrant" in [grant["permission"] for grant in grants]
    assert cleared is None
    assert after["corpus_id"] is None


def test_selecting_a_corpus_that_does_not_exist_is_a_404(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        workspace = client.get("/api/workspaces").json()["workspaces"][0]["root"]
        conversation = client.post(
            "/api/conversations",
            json={"workspace_root": workspace},
        ).json()["conversation"]

        response = client.put(
            f"/api/conversations/{conversation['id']}/corpus",
            json={"corpus_id": "inexistente"},
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "corpus_not_found"


def test_without_an_embedding_model_the_tab_says_so_instead_of_faking_a_corpus(
    tmp_path: Path,
) -> None:
    with _app(tmp_path, with_embedder=False) as client:
        snapshot = client.get("/api/ui/corpora").json()
        creation = client.post("/api/corpora", json={"name": "Manual"})

    assert snapshot["available"] is False
    assert snapshot["corpora"] == []
    assert creation.status_code == 409
    assert creation.json()["error"]["code"] == "corpus_unavailable"


def test_the_corpora_snapshot_carries_what_the_tab_needs(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        client.post("/api/corpora", json={"name": "Manual", "description": "notas"})
        snapshot = client.get("/api/ui/corpora").json()

    assert snapshot["available"] is True
    assert snapshot["embedding_model"] == "harness-bge-m3:latest"
    assert ".pdf" in snapshot["accepted_extensions"]
    assert snapshot["corpora"][0]["description"] == "notas"
    assert snapshot["jobs"] == []
