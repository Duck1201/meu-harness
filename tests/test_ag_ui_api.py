import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from jsonschema import Draft202012Validator

from harness import (
    AgentEvent,
    AgentEventKind,
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
from harness.ag_ui import project_agent_event

# With no Operator password configured the server only answers direct loopback,
# so a test has to say where its request comes from.
LOOPBACK = ("127.0.0.1", 51000)


class FakeEstimator:
    validated = True
    readiness = EngineReadiness(ready=True)

    def estimate(self, messages: Sequence[ModelMessage], tools: Sequence[ToolSchema]) -> int:
        return len(messages) + len(tools)


class SequenceRuntime:
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = list(responses)

    async def verify_profile(self) -> EngineReadiness:
        return EngineReadiness(ready=True)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        del request
        return self._responses.pop(0)

    async def aclose(self) -> None:
        return None


def _event_schema(event_type: str, *required: str, role: str | None = None) -> dict[str, object]:
    properties: dict[str, object] = {"type": {"const": event_type}}
    if role is not None:
        properties["role"] = {"const": role}
        required = (*required, "role")
    return {
        "type": "object",
        "required": ["type", *required],
        "properties": properties,
    }


EVENT_SCHEMAS: dict[str, dict[str, object]] = {
    "RUN_STARTED": _event_schema("RUN_STARTED", "threadId", "runId"),
    "RUN_FINISHED": _event_schema("RUN_FINISHED", "threadId", "runId"),
    "RUN_ERROR": _event_schema("RUN_ERROR", "message"),
    "STEP_STARTED": _event_schema("STEP_STARTED", "stepName"),
    "STEP_FINISHED": _event_schema("STEP_FINISHED", "stepName"),
    "REASONING_START": _event_schema("REASONING_START", "messageId"),
    "REASONING_MESSAGE_START": _event_schema(
        "REASONING_MESSAGE_START", "messageId", role="reasoning"
    ),
    "REASONING_MESSAGE_CONTENT": _event_schema("REASONING_MESSAGE_CONTENT", "messageId", "delta"),
    "REASONING_MESSAGE_END": _event_schema("REASONING_MESSAGE_END", "messageId"),
    "REASONING_END": _event_schema("REASONING_END", "messageId"),
    "TEXT_MESSAGE_START": _event_schema("TEXT_MESSAGE_START", "messageId", role="assistant"),
    "TEXT_MESSAGE_CONTENT": _event_schema("TEXT_MESSAGE_CONTENT", "messageId", "delta"),
    "TEXT_MESSAGE_END": _event_schema("TEXT_MESSAGE_END", "messageId"),
    "TOOL_CALL_START": _event_schema("TOOL_CALL_START", "toolCallId", "toolCallName"),
    "TOOL_CALL_ARGS": _event_schema("TOOL_CALL_ARGS", "toolCallId", "delta"),
    "TOOL_CALL_END": _event_schema("TOOL_CALL_END", "toolCallId"),
    "TOOL_CALL_RESULT": _event_schema(
        "TOOL_CALL_RESULT", "messageId", "toolCallId", "content", role="tool"
    ),
    "CUSTOM": _event_schema("CUSTOM", "name", "value"),
}


def _app(tmp_path: Path, responses: Sequence[ModelResponse]) -> FastAPI:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("hello", encoding="utf-8")
    service = ApplicationService(
        store=ConversationStore(tmp_path / "conversations.sqlite3"),
        observability_store=ObservabilityStore(tmp_path / "observability.sqlite3"),
        config=load_config(),
        runtime=SequenceRuntime(responses),
        estimator=FakeEstimator(),
        allowed_workspace_roots=(workspace,),
    )
    return create_app(service=service, static_dir=tmp_path / "missing-dist")


def _create_conversation(client: TestClient, workspace: Path) -> str:
    return str(
        client.post("/api/conversations", json={"workspace_root": str(workspace)}).json()[
            "conversation"
        ]["id"]
    )


def _events(response: Response) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.iter_lines()
        if line.startswith("data: ")
    ]


def test_agent_sse_accepts_run_agent_input_and_projects_documented_events(
    tmp_path: Path,
) -> None:
    app = _app(
        tmp_path,
        (
            ModelResponse(
                reasoning="inspect the file",
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="read_file",
                        arguments={"file_path": "note.txt"},
                    ),
                ),
            ),
            ModelResponse(content="The note says hello."),
        ),
    )

    with TestClient(app, client=LOOPBACK) as client:
        conversation_id = _create_conversation(client, tmp_path / "workspace")
        with client.stream(
            "POST",
            "/api/agent",
            json={
                "threadId": conversation_id,
                "runId": "client-run-1",
                "messages": [
                    {"id": "system-1", "role": "system", "content": "Be concise."},
                    {"id": "user-old", "role": "user", "content": "old request"},
                    {
                        "id": "assistant-1",
                        "role": "assistant",
                        "content": "old response",
                        "reasoning": "must not be replayed",
                    },
                    {"id": "user-new", "role": "user", "content": "read the note"},
                ],
                "state": {"reasoning": "must not be persisted"},
                "context": [{"description": "ignored", "value": "ignored"}],
                "tools": [{"name": "ignored", "parameters": {"type": "object"}}],
            },
        ) as response:
            events = _events(response)
        snapshot = client.get("/api/ui/chat", params={"conversation_id": conversation_id}).json()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["connection"] == "keep-alive"
    assert response.headers["x-accel-buffering"] == "no"
    assert [event["type"] for event in events] == [
        "RUN_STARTED",
        "STEP_STARTED",
        # O consumo da janela sai logo depois do contexto ser montado, uma vez
        # por passo, antes de qualquer token do modelo.
        "CUSTOM",
        "REASONING_START",
        "REASONING_MESSAGE_START",
        "REASONING_MESSAGE_CONTENT",
        "REASONING_MESSAGE_END",
        "REASONING_END",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "STEP_FINISHED",
        "STEP_STARTED",
        "CUSTOM",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "STEP_FINISHED",
        "CUSTOM",
        "RUN_FINISHED",
    ]
    for event in events:
        Draft202012Validator(EVENT_SCHEMAS[str(event["type"])]).validate(  # pyright: ignore[reportUnknownMemberType]
            event
        )

    assert events[0] == {
        "type": "RUN_STARTED",
        "threadId": conversation_id,
        "runId": "client-run-1",
    }
    assert events[-1] == {
        "type": "RUN_FINISHED",
        "threadId": conversation_id,
        "runId": "client-run-1",
    }
    assert sum(event["type"] in {"RUN_FINISHED", "RUN_ERROR"} for event in events) == 1
    assert events[4]["role"] == "reasoning"
    assert {events[index]["messageId"] for index in (4, 5, 6)} == {events[4]["messageId"]}
    assert {events[index]["toolCallId"] for index in (8, 9, 10, 11)} == {"call-1"}
    assert events[9]["delta"] == '{"file_path":"note.txt"}'
    assert json.loads(str(events[11]["content"]))["status"] == "success"
    assert {events[index]["messageId"] for index in (15, 16, 17)} == {events[15]["messageId"]}
    assert events[16]["delta"] == "The note says hello."
    # A janela é a do contrato e o consumo cabe nela; o estimador do teste é
    # falso, então o número exato não diz nada — a relação diz.
    window = load_config().context.initial_budget_tokens
    for index in (2, 14):
        usage = events[index]
        assert usage["name"] == "harness.context_usage"
        value = cast(dict[str, int], usage["value"])
        assert value["context_window"] == window
        assert 0 < value["estimated_input_tokens"] < window
        assert value["dropped_turns"] == 0
    assert events[-2] == {
        "type": "CUSTOM",
        "name": "harness.turn_outcome",
        "value": {
            "outcome_kind": "completed",
            "reason_code": "final_response",
        },
    }
    assert [
        item["payload"]["content"] for item in snapshot["history"] if item["kind"] == "user_message"
    ] == ["read the note"]
    internal_request_id = snapshot["turns"][0]["request_id"]
    assert internal_request_id != "client-run-1"
    assert internal_request_id not in json.dumps(events)
    assert "must not be replayed" not in json.dumps(events)
    assert not any(
        "reasoning" in json.dumps(item, ensure_ascii=False).lower() for item in snapshot["history"]
    )


def test_agent_generates_run_id_and_rejects_inputs_without_a_typed_user_message(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path, (ModelResponse(content="ok"),))
    run_input_schema = app.openapi()["components"]["schemas"]["RunAgentInput"]

    with TestClient(app, client=LOOPBACK) as client:
        conversation_id = _create_conversation(client, tmp_path / "workspace")
        with client.stream(
            "POST",
            "/api/agent",
            json={
                "threadId": conversation_id,
                "messages": [{"id": "user-1", "role": "user", "content": "hello"}],
            },
        ) as response:
            events = _events(response)
        missing_user = client.post(
            "/api/agent",
            json={
                "threadId": conversation_id,
                "messages": [{"id": "assistant-1", "role": "assistant", "content": "hello"}],
            },
        )
        invalid_role = client.post(
            "/api/agent",
            json={
                "threadId": conversation_id,
                "messages": [{"id": "message-1", "role": "invalid", "content": "hello"}],
            },
        )

    assert UUID(str(events[0]["runId"]))
    assert events[-1]["runId"] == events[0]["runId"]
    assert missing_user.status_code == 422
    assert missing_user.json()["error"]["code"] == "ag_ui_user_message_required"
    assert invalid_role.status_code == 422
    assert run_input_schema["required"] == ["threadId", "messages"]
    message_items = run_input_schema["properties"]["messages"]["items"]
    assert len(message_items["oneOf"]) == 5
    assert set(message_items["discriminator"]["mapping"]) == {
        "assistant",
        "developer",
        "system",
        "tool",
        "user",
    }


def test_failed_agent_stream_emits_custom_outcome_then_one_run_error(tmp_path: Path) -> None:
    app = _app(tmp_path, ())

    with TestClient(app, client=LOOPBACK) as client:
        conversation_id = _create_conversation(client, tmp_path / "workspace")
        with client.stream(
            "POST",
            "/api/agent",
            json={
                "threadId": conversation_id,
                "runId": "failed-run",
                "messages": [{"id": "user-1", "role": "user", "content": "trigger failure"}],
            },
        ) as response:
            events = _events(response)

    assert events[-2] == {
        "type": "CUSTOM",
        "name": "harness.turn_outcome",
        "value": {"outcome_kind": "failed", "reason_code": "engine_error"},
    }
    assert events[-1] == {
        "type": "RUN_ERROR",
        "message": "engine_error",
        "code": "engine_error",
    }
    assert sum(event["type"] in {"RUN_FINISHED", "RUN_ERROR"} for event in events) == 1


@pytest.mark.parametrize(
    ("outcome_kind", "terminal_type"),
    (
        ("completed", "RUN_FINISHED"),
        ("limit_reached", "RUN_FINISHED"),
        ("cancelled", "RUN_FINISHED"),
        ("blocked", "RUN_FINISHED"),
        ("abandoned", "RUN_FINISHED"),
        ("failed", "RUN_ERROR"),
    ),
)
def test_terminal_outcome_custom_event_precedes_exactly_one_ag_ui_terminal(
    outcome_kind: str, terminal_type: str
) -> None:
    projected = project_agent_event(
        AgentEvent(
            kind=AgentEventKind.TURN_FINISHED,
            turn_id="turn-1",
            step_sequence=None,
            payload={"outcome_kind": outcome_kind, "reason_code": "reason"},
            conversation_id="thread-1",
            request_id="internal-request-1",
        ),
        run_id="run-1",
    )

    assert projected[0] == {
        "type": "CUSTOM",
        "name": "harness.turn_outcome",
        "value": {"outcome_kind": outcome_kind, "reason_code": "reason"},
    }
    assert projected[1]["type"] == terminal_type
    assert sum(event["type"] in {"RUN_FINISHED", "RUN_ERROR"} for event in projected) == 1
    Draft202012Validator(EVENT_SCHEMAS[terminal_type]).validate(  # pyright: ignore[reportUnknownMemberType]
        projected[1]
    )


def test_generation_stats_become_the_custom_event_the_speed_meter_reads() -> None:
    projected = project_agent_event(
        AgentEvent(
            kind=AgentEventKind.GENERATION_STATS,
            turn_id="turn-1",
            step_sequence=2,
            payload={"output_tokens": 280, "eval_ms": 5000.0, "prompt_tokens": 1200},
            conversation_id="thread-1",
            request_id="internal-request-1",
        ),
        run_id="run-1",
    )

    assert projected == [
        {
            "type": "CUSTOM",
            "name": "harness.generation_stats",
            "value": {"output_tokens": 280, "eval_ms": 5000.0, "prompt_tokens": 1200},
        }
    ]
