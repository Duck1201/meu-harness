from collections.abc import Sequence
from pathlib import Path
from threading import Event
from time import monotonic, sleep

from fastapi.testclient import TestClient

from harness import (
    ApplicationService,
    ConversationStore,
    EngineReadiness,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ObservabilityStore,
    ToolCall,
    ToolSchema,
    create_app,
    load_config,
)
from harness.api import DEFAULT_PORT, configured_origins

# With no Operator password configured the server only answers direct loopback,
# so a test has to say where its request comes from.
LOOPBACK = ("127.0.0.1", 51000)


class FakeEstimator:
    validated = True
    readiness = EngineReadiness(ready=True)

    def estimate(self, messages: Sequence[ModelMessage], tools: Sequence[ToolSchema]) -> int:
        return len(messages) + len(tools)


class FakeRuntime:
    async def verify_profile(self) -> EngineReadiness:
        return EngineReadiness(ready=True)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        del request
        return ModelResponse(content="ok")

    async def aclose(self) -> None:
        return None


def test_default_origins_follow_the_port_the_server_listens_on() -> None:
    # Before setup writes a HostConfig the SPA's own origin is the only one that
    # can reach the API, so a hardcoded port would leave a fresh install unable
    # to submit its own setup form.
    assert configured_origins(None) == (
        f"http://127.0.0.1:{DEFAULT_PORT}",
        f"http://localhost:{DEFAULT_PORT}",
    )
    assert configured_origins(None, 9100) == (
        "http://127.0.0.1:9100",
        "http://localhost:9100",
    )


def _service(
    tmp_path: Path,
    workspace_root: Path,
    *,
    runtime: FakeRuntime | None = None,
) -> ApplicationService:
    return ApplicationService(
        store=ConversationStore(tmp_path / "conversations.sqlite3"),
        observability_store=ObservabilityStore(tmp_path / "observability.sqlite3"),
        config=load_config(),
        runtime=runtime or FakeRuntime(),
        estimator=FakeEstimator(),
        allowed_workspace_roots=(workspace_root,),
    )


def test_health_and_conversation_creation_enforce_server_workspace_allowlist(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    app = create_app(
        service=_service(tmp_path, allowed),
        allowed_origins=("http://operator.test",),
        static_dir=tmp_path / "missing-dist",
    )

    with TestClient(app, client=LOOPBACK) as client:
        health = client.get("/api/health").json()
        workspaces = client.get("/api/workspaces").json()["workspaces"]
        relative = client.post(
            "/api/conversations",
            json={"workspace_root": "allowed"},
        )
        denied = client.post(
            "/api/conversations",
            json={"workspace_root": str(outside)},
        )
        created = client.post(
            "/api/conversations",
            json={"workspace_root": str(allowed)},
        )

    assert health["ready"] is True
    assert health["capabilities"]["settings_mutation"] is False
    assert workspaces == [{"id": workspaces[0]["id"], "root": str(allowed.resolve())}]
    assert relative.status_code == 422
    assert relative.json()["error"]["code"] == "workspace_root_must_be_absolute"
    assert denied.status_code == 403
    assert denied.json()["error"]["code"] == "workspace_root_not_allowed"
    assert created.status_code == 201
    assert created.json()["conversation"]["workspace_id"] == workspaces[0]["id"]


def test_conversations_can_be_listed_renamed_archived_and_deleted(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = create_app(
        service=_service(tmp_path, workspace),
        static_dir=tmp_path / "missing-dist",
    )

    with TestClient(app, client=LOOPBACK) as client:
        created = client.post(
            "/api/conversations",
            json={"workspace_root": str(workspace), "name": "Initial"},
        ).json()["conversation"]
        conversation_id = created["id"]
        listed = client.get("/api/conversations").json()["conversations"]
        renamed = client.patch(
            f"/api/conversations/{conversation_id}",
            json={"name": "Renamed"},
        ).json()["conversation"]
        fetched = client.get(f"/api/conversations/{conversation_id}").json()["conversation"]
        archived = client.patch(
            f"/api/conversations/{conversation_id}",
            json={"archived": True},
        ).json()["conversation"]
        visible = client.get("/api/conversations").json()["conversations"]
        all_conversations = client.get(
            "/api/conversations", params={"include_archived": True}
        ).json()["conversations"]
        deleted = client.delete(f"/api/conversations/{conversation_id}")
        missing = client.get(f"/api/conversations/{conversation_id}")

    assert created["name"] == "Initial"
    assert [item["id"] for item in listed] == [conversation_id]
    assert renamed["name"] == "Renamed"
    assert fetched["name"] == "Renamed"
    assert archived["archived_at"] is not None
    assert visible == []
    assert [item["id"] for item in all_conversations] == [conversation_id]
    assert deleted.status_code == 204
    assert missing.status_code == 404


def test_requests_run_fifo_in_background_with_one_turn_and_editable_queue(
    tmp_path: Path,
) -> None:
    class GateRuntime(FakeRuntime):
        def __init__(self) -> None:
            self.entered = Event()
            self.release = Event()
            self.active = 0
            self.max_active = 0
            self.calls = 0

        async def generate(self, request: ModelRequest) -> ModelResponse:
            import asyncio

            del request
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls += 1
            try:
                if self.calls == 1:
                    self.entered.set()
                    await asyncio.to_thread(self.release.wait)
                return ModelResponse(content=f"response {self.calls}")
            finally:
                self.active -= 1

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = GateRuntime()
    app = create_app(
        service=_service(tmp_path, workspace, runtime=runtime),
        static_dir=tmp_path / "missing-dist",
    )

    with TestClient(app, client=LOOPBACK) as client:
        conversation_id = client.post(
            "/api/conversations", json={"workspace_root": str(workspace)}
        ).json()["conversation"]["id"]
        first = client.post(
            f"/api/conversations/{conversation_id}/requests",
            json={"content": "first"},
        )
        assert runtime.entered.wait(timeout=2)
        second = client.post(
            f"/api/conversations/{conversation_id}/requests",
            json={"content": "second"},
        ).json()["request"]
        third = client.post(
            f"/api/conversations/{conversation_id}/requests",
            json={"content": "third"},
        ).json()["request"]
        edited = client.patch(
            f"/api/conversations/{conversation_id}/requests/{second['id']}",
            json={"content": "second edited"},
        ).json()["request"]
        canceled = client.delete(
            f"/api/conversations/{conversation_id}/requests/{third['id']}"
        ).json()["request"]
        during = client.get("/api/ui/chat", params={"conversation_id": conversation_id}).json()

        runtime.release.set()
        deadline = monotonic() + 3
        while monotonic() < deadline:
            final = client.get("/api/ui/chat", params={"conversation_id": conversation_id}).json()
            if final["active_turn"] is None and not final["pending_requests"]:
                break
            sleep(0.01)
        else:
            raise AssertionError("background worker did not become idle")

    assert first.status_code == 202
    assert edited["content"] == "second edited"
    assert canceled["status"] == "canceled"
    assert during["active_turn"] is not None
    assert [item["content"] for item in during["pending_requests"]] == ["second edited"]
    assert runtime.calls == 2
    assert runtime.max_active == 1
    user_messages = [
        item["payload"]["content"] for item in final["history"] if item["kind"] == "user_message"
    ]
    assert user_messages == [
        "first",
        "second edited",
    ]


def test_revoked_write_grant_is_revalidated_before_the_effect(tmp_path: Path) -> None:
    class GatedToolRuntime(FakeRuntime):
        def __init__(self) -> None:
            self.entered = Event()
            self.release = Event()

        async def generate(self, request: ModelRequest) -> ModelResponse:
            import asyncio

            del request
            self.entered.set()
            await asyncio.to_thread(self.release.wait)
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="write-1",
                        name="write_file",
                        arguments={"file_path": "effect.txt", "content": "must not happen"},
                    ),
                )
            )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = GatedToolRuntime()
    app = create_app(
        service=_service(tmp_path, workspace, runtime=runtime),
        static_dir=tmp_path / "missing-dist",
    )

    with TestClient(app, client=LOOPBACK) as client:
        conversation_id = client.post(
            "/api/conversations", json={"workspace_root": str(workspace)}
        ).json()["conversation"]["id"]
        grant = client.post(
            f"/api/conversations/{conversation_id}/grants",
            json={"permission": "WriteGrant"},
        ).json()["grant"]
        client.post(
            f"/api/conversations/{conversation_id}/requests",
            json={"content": "write a file"},
        )
        assert runtime.entered.wait(timeout=2)
        revoked = client.delete(f"/api/conversations/{conversation_id}/grants/{grant['id']}")
        runtime.release.set()

        # The revocation is revalidated before the effect: the write does not run,
        # it stops on the Operator. Approving would hand the grant back, so the
        # denial below is what keeps a revoked grant revoked.
        deadline = monotonic() + 3
        while monotonic() < deadline:
            pending = client.get(f"/api/conversations/{conversation_id}/confirmation").json()[
                "confirmation"
            ]
            if pending is not None:
                break
            sleep(0.01)
        else:
            raise AssertionError("the revoked write never reached the Operator")
        assert pending["reason_code"] == "write_grant_required"
        client.post(
            f"/api/conversations/{conversation_id}/confirmation/{pending['id']}",
            json={"approved": False},
        )

        deadline = monotonic() + 3
        while monotonic() < deadline:
            snapshot = client.get(
                "/api/ui/chat", params={"conversation_id": conversation_id}
            ).json()
            if snapshot["active_turn"] is None:
                break
            sleep(0.01)
        else:
            raise AssertionError("blocked turn did not finish")

    outcome = snapshot["turns"][0]["terminal_outcome"]
    assert revoked.status_code == 204
    assert outcome["kind"] == "blocked"
    assert outcome["reason_code"] == "write_grant_denied"
    assert not (workspace / "effect.txt").exists()
    assert snapshot["confirmation_waivers"] == []


def test_stop_is_cooperative_and_the_next_pending_request_starts(tmp_path: Path) -> None:
    class StoppableRuntime(FakeRuntime):
        def __init__(self) -> None:
            self.entered = Event()
            self.release = Event()
            self.calls = 0

        async def generate(self, request: ModelRequest) -> ModelResponse:
            import asyncio

            del request
            self.calls += 1
            if self.calls == 1:
                self.entered.set()
                await asyncio.to_thread(self.release.wait)
            return ModelResponse(content=f"response {self.calls}")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = StoppableRuntime()
    app = create_app(
        service=_service(tmp_path, workspace, runtime=runtime),
        static_dir=tmp_path / "missing-dist",
    )

    with TestClient(app, client=LOOPBACK) as client:
        conversation_id = client.post(
            "/api/conversations", json={"workspace_root": str(workspace)}
        ).json()["conversation"]["id"]
        client.post(f"/api/conversations/{conversation_id}/requests", json={"content": "stop me"})
        assert runtime.entered.wait(timeout=2)
        client.post(
            f"/api/conversations/{conversation_id}/requests",
            json={"content": "run after stop"},
        )
        stopped = client.post(f"/api/conversations/{conversation_id}/stop")
        runtime.release.set()

        deadline = monotonic() + 3
        while monotonic() < deadline:
            snapshot = client.get(
                "/api/ui/chat", params={"conversation_id": conversation_id}
            ).json()
            if snapshot["active_turn"] is None and not snapshot["pending_requests"]:
                break
            sleep(0.01)
        else:
            raise AssertionError("conversation did not drain after stop")

    assert stopped.status_code == 202
    assert runtime.calls == 2
    assert [turn["terminal_outcome"]["kind"] for turn in snapshot["turns"]] == [
        "cancelled",
        "completed",
    ]


def test_feedback_and_ui_snapshots_are_query_projections(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = create_app(
        service=_service(tmp_path, workspace),
        static_dir=tmp_path / "missing-dist",
    )

    with TestClient(app, client=LOOPBACK) as client:
        conversation_id = client.post(
            "/api/conversations", json={"workspace_root": str(workspace)}
        ).json()["conversation"]["id"]
        feedback = client.post(
            f"/api/conversations/{conversation_id}/feedback",
            json={"rating": 1, "comment": "useful"},
        )
        chat = client.get("/api/ui/chat", params={"conversation_id": conversation_id}).json()
        evals = client.get("/api/ui/evals").json()
        settings = client.get("/api/ui/settings").json()

    assert feedback.status_code == 201
    assert chat["feedback"][0]["comment"] == "useful"
    assert evals["runs"] == []
    assert evals["reports"] == []
    assert evals["capabilities"] == {
        "eval_runner": True,
        "progress_transport": "polling_json",
    }
    assert evals["experiments"]
    assert settings["mutable"] is False
    assert settings["default_execution_route"] == "local_web_tools"


def test_approving_the_dialog_is_how_the_write_grant_is_given(tmp_path: Path) -> None:
    class WritingRuntime(FakeRuntime):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, request: ModelRequest) -> ModelResponse:
            del request
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="write-1",
                            name="write_file",
                            arguments={"file_path": "nota.md", "content": "aprovado\n"},
                        ),
                    )
                )
            return ModelResponse(content="escrito")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = create_app(
        service=_service(tmp_path, workspace, runtime=WritingRuntime()),
        static_dir=tmp_path / "missing-dist",
    )

    with TestClient(app, client=LOOPBACK) as client:
        conversation_id = client.post(
            "/api/conversations", json={"workspace_root": str(workspace)}
        ).json()["conversation"]["id"]
        # No WriteGrant is given up front: the dialog is where it is asked for.
        client.post(
            f"/api/conversations/{conversation_id}/requests",
            json={"content": "cria nota.md"},
        )

        deadline = monotonic() + 3
        while monotonic() < deadline:
            pending = client.get(f"/api/conversations/{conversation_id}/confirmation").json()[
                "confirmation"
            ]
            if pending is not None:
                break
            sleep(0.01)
        else:
            raise AssertionError("the write never reached the Operator")

        assert pending["reason_code"] == "write_grant_required"
        assert [preview["path"] for preview in pending["previews"]] == ["nota.md"]
        assert "+aprovado" in pending["previews"][0]["diff"]
        client.post(
            f"/api/conversations/{conversation_id}/confirmation/{pending['id']}",
            json={"approved": True},
        )

        deadline = monotonic() + 3
        while monotonic() < deadline:
            snapshot = client.get(
                "/api/ui/chat", params={"conversation_id": conversation_id}
            ).json()
            if snapshot["active_turn"] is None:
                break
            sleep(0.01)
        else:
            raise AssertionError("approved turn did not finish")

    assert snapshot["turns"][0]["terminal_outcome"]["kind"] == "completed"
    assert (workspace / "nota.md").read_text(encoding="utf-8") == "aprovado\n"
    granted = client.get(f"/api/conversations/{conversation_id}/grants").json()["grants"]
    assert [item["permission"] for item in granted if item["permission"] == "WriteGrant"] == [
        "WriteGrant"
    ]


def test_confirmation_is_absent_until_required_and_cannot_be_answered_blindly(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = create_app(
        service=_service(tmp_path, workspace),
        static_dir=tmp_path / "missing-dist",
    )

    with TestClient(app, client=LOOPBACK) as client:
        conversation_id = client.post(
            "/api/conversations", json={"workspace_root": str(workspace)}
        ).json()["conversation"]["id"]
        pending = client.get(f"/api/conversations/{conversation_id}/confirmation")
        chat = client.get("/api/ui/chat", params={"conversation_id": conversation_id}).json()
        blind = client.post(
            f"/api/conversations/{conversation_id}/confirmation/invented-id",
            json={"approved": True},
        )
        unknown_conversation = client.get("/api/conversations/does-not-exist/confirmation")

    assert pending.status_code == 200
    assert pending.json()["confirmation"] is None
    assert chat["pending_confirmation"] is None
    assert blind.status_code == 409
    assert blind.json()["error"]["code"] == "confirmation_not_pending"
    assert unknown_conversation.status_code == 404


def test_origin_body_limit_and_optional_spa_fallback_are_closed_by_default(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    dist = tmp_path / "dist"
    assets = dist / "assets"
    dist.mkdir()
    assets.mkdir()
    (dist / "index.html").write_text("<main>Harness UI</main>", encoding="utf-8")
    (assets / "app.js").write_text("window.harness = true;", encoding="utf-8")
    app = create_app(
        service=_service(tmp_path, workspace),
        allowed_origins=("http://operator.test",),
        static_dir=dist,
        max_body_bytes=64,
    )

    with TestClient(app, client=LOOPBACK) as client:
        denied_origin = client.get("/api/health", headers={"Origin": "http://attacker.test"})
        allowed_origin = client.get("/api/health", headers={"Origin": "http://operator.test"})
        oversized = client.post(
            "/api/conversations",
            content=b"{" + b'"workspace_root":"' + (b"x" * 80) + b'"}',
            headers={"Content-Type": "application/json"},
        )
        asset = client.get("/assets/app.js")
        route = client.get("/conversations/current")
        missing_api = client.get("/api/does-not-exist")

    assert denied_origin.status_code == 403
    assert denied_origin.json()["error"]["code"] == "origin_not_allowed"
    assert allowed_origin.headers["access-control-allow-origin"] == "http://operator.test"
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "request_body_too_large"
    assert asset.text == "window.harness = true;"
    assert route.text == "<main>Harness UI</main>"
    assert missing_api.status_code == 404
    assert "Harness UI" not in missing_api.text


_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def test_an_attached_image_lands_in_the_workspace_where_describe_image_reads_it(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = create_app(service=_service(tmp_path, workspace), static_dir=tmp_path / "missing-dist")

    with TestClient(app, client=LOOPBACK) as client:
        conversation = client.post(
            "/api/conversations", json={"workspace_root": str(workspace)}
        ).json()["conversation"]
        url = f"/api/conversations/{conversation['id']}/attachments"
        accepted = client.post(url, files={"file": ("Print da tela (1).PNG", _PNG)})
        # O formato é o dos bytes: um PNG renomeado para .jpg continua PNG.
        renamed = client.post(url, files={"file": ("foto.jpg", _PNG)})
        refused = client.post(url, files={"file": ("nota.avif", b"\x00\x00\x00\x1cftypavif")})
        missing = client.post(url, data={"nothing": "here"})

    assert accepted.status_code == 201
    path = accepted.json()["path"]
    assert path.startswith("anexos/") and path.endswith("-Print-da-tela-1.png")
    assert (workspace / path).read_bytes() == _PNG
    assert renamed.json()["path"].endswith("-foto.png")
    assert refused.status_code == 415
    assert refused.json()["error"]["code"] == "image_format_not_accepted"
    assert missing.status_code == 422
    assert [
        item.name for item in (workspace / "anexos").iterdir() if item.name.startswith(".")
    ] == []


def test_an_image_over_the_vision_limit_is_refused_before_it_is_written(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = create_app(service=_service(tmp_path, workspace), static_dir=tmp_path / "missing-dist")
    limit = load_config().vision.max_image_bytes

    with TestClient(app, client=LOOPBACK) as client:
        conversation = client.post(
            "/api/conversations", json={"workspace_root": str(workspace)}
        ).json()["conversation"]
        too_large = client.post(
            f"/api/conversations/{conversation['id']}/attachments",
            files={"file": ("grande.png", _PNG + b"\x00" * limit)},
        )

    assert too_large.status_code == 413
    assert not (workspace / "anexos").exists()
