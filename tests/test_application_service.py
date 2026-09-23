import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from harness import (
    AgentEvent,
    AgentEventKind,
    ApplicationService,
    ApplicationServiceError,
    BenchmarkLease,
    ConfirmationRequest,
    ConversationStore,
    EngineReadiness,
    JsonValue,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ObservabilityStore,
    OperatorConfirmationGate,
    TerminalOutcomeKind,
    ToolCall,
    ToolSchema,
    TurnStatus,
    load_config,
)
from harness.ports import VisionAnswer


class RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class FakeEstimator:
    validated = True
    readiness = EngineReadiness(ready=True)

    def estimate(self, messages: Sequence[ModelMessage], tools: Sequence[ToolSchema]) -> int:
        return len(messages) + len(tools)


class CapturingRuntime:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def verify_profile(self) -> EngineReadiness:
        return EngineReadiness(ready=True)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(content="ok")

    async def aclose(self) -> None:
        return None


class UnreadyRuntime(CapturingRuntime):
    async def verify_profile(self) -> EngineReadiness:
        return EngineReadiness(ready=False, reason_code="model_digest_mismatch")


async def _wait_until_idle(service: ApplicationService, conversation_id: str) -> None:
    for _ in range(300):
        snapshot = await service.chat_snapshot(conversation_id)
        if snapshot["active_turn"] is None and not snapshot["pending_requests"]:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("conversation worker did not become idle")


def test_service_exposes_effective_tool_schemas_without_disclosing_the_search_endpoint(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runtime = CapturingRuntime()
        secret = "https://searx.internal.example/search-that-must-not-leak"
        service = ApplicationService(
            store=ConversationStore(tmp_path / "conversations.sqlite3"),
            observability_store=ObservabilityStore(tmp_path / "observability.sqlite3"),
            config=load_config(),
            runtime=runtime,
            estimator=FakeEstimator(),
            allowed_workspace_roots=(workspace,),
            search_endpoint=secret,
        )
        await service.initialize()
        try:
            conversation = await service.create_conversation(str(workspace))

            with patch(
                "harness.conversation_store.secrets.randbits",
                side_effect=(101, 202, 303),
            ):
                await service.enqueue_request(conversation.id, "without effect grants")
                await _wait_until_idle(service, conversation.id)

                write_grant = await service.grant(conversation.id, "WriteGrant")
                web_grant = await service.grant(conversation.id, "WebAccessGrant")
                await service.enqueue_request(conversation.id, "with effect grants")
                await _wait_until_idle(service, conversation.id)

                await service.revoke_grant(conversation.id, write_grant.id)
                await service.revoke_grant(conversation.id, web_grant.id)
                await service.enqueue_request(conversation.id, "after revocation")
                await _wait_until_idle(service, conversation.id)

            # The catalogue does not move with the grants: a schema authorizes
            # nothing, and hiding one only makes the model call it from memory,
            # without its required arguments. Preflight is what refuses.
            catalogue = {
                "read_file",
                "write_file",
                "edit",
                "list_directory",
                "glob",
                "grep_search",
                "web_fetch",
                "web_search",
                "calculate",
                "get_weather",
            }
            # corpus_search fica de fora: este serviço não tem modelo de embedding
            # nem diretório de corpora, então não há executor por trás do nome. O
            # mesmo vale para describe_image, que não tem modelo de visão aqui.
            offered = [{schema.name for schema in request.tools} for request in runtime.requests]
            assert offered == [catalogue, catalogue, catalogue]

            snapshot_text = repr(await service.chat_snapshot(conversation.id))
            events_text = json.dumps(
                [event.payload for event in await service.observability_store.list_events()]
            )
            assert secret not in snapshot_text
            assert secret not in events_text

            turns = await service.store.list_turns(conversation.id)
            step_events = [
                event
                for event in await service.observability_store.list_events()
                if event.event_type == "agent.step_started"
            ]
            assert [turn.base_seed for turn in turns] == [101, 202, 303]
            assert [event.payload["base_seed"] for event in step_events] == [
                turn.base_seed for turn in turns
            ]
            assert [event.payload["seed"] for event in step_events] == [
                (await service.store.list_agent_steps(turn.id))[0].seed for turn in turns
            ]
        finally:
            await service.shutdown()

    asyncio.run(scenario())


class _Vision:
    async def describe(self, image: bytes, question: str) -> VisionAnswer:
        del image, question
        return VisionAnswer(text="ok", latency_ms=1.0)


def test_describe_image_is_offered_and_announced_only_with_a_vision_model(
    tmp_path: Path,
) -> None:
    """Oferecer a tool e dizer ao modelo que ele não enxerga seriam fatos opostos."""

    async def offered(vision: _Vision | None) -> tuple[set[str], str]:
        workspace = tmp_path / ("com" if vision else "sem")
        workspace.mkdir()
        runtime = CapturingRuntime()
        service = ApplicationService(
            store=ConversationStore(workspace.with_suffix(".c.sqlite3")),
            observability_store=ObservabilityStore(workspace.with_suffix(".o.sqlite3")),
            config=load_config(),
            runtime=runtime,
            estimator=FakeEstimator(),
            allowed_workspace_roots=(workspace,),
            vision_runtime=vision,
        )
        await service.initialize()
        try:
            conversation = await service.create_conversation(str(workspace))
            await service.enqueue_request(conversation.id, "o que tem neste print?")
            await _wait_until_idle(service, conversation.id)
        finally:
            await service.shutdown()
        request = runtime.requests[0]
        return {schema.name for schema in request.tools}, request.messages[0].content

    with_tools, with_prompt = asyncio.run(offered(_Vision()))
    without_tools, without_prompt = asyncio.run(offered(None))

    assert "describe_image" in with_tools
    assert "describe_image" not in without_tools
    assert "ask describe_image" in with_prompt
    assert "You cannot see images. If a request" in without_prompt
    assert "describe_image" not in without_prompt


def test_web_search_is_always_offered_because_its_fallback_needs_no_credential(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runtime = CapturingRuntime()
        service = ApplicationService(
            store=ConversationStore(tmp_path / "conversations.sqlite3"),
            observability_store=ObservabilityStore(tmp_path / "observability.sqlite3"),
            config=load_config(),
            runtime=runtime,
            estimator=FakeEstimator(),
            allowed_workspace_roots=(workspace,),
        )
        await service.initialize()
        try:
            conversation = await service.create_conversation(str(workspace))
            await service.grant(conversation.id, "WebAccessGrant")
            await service.enqueue_request(conversation.id, "use available web tools")
            await _wait_until_idle(service, conversation.id)

            offered = {schema.name for schema in runtime.requests[0].tools}
            assert "web_fetch" in offered
            assert "web_search" in offered
        finally:
            await service.shutdown()

    asyncio.run(scenario())


def test_initialize_abandons_stale_active_turn_before_recovering_pending_queue(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        store = ConversationStore(tmp_path / "conversations.sqlite3")
        await store.initialize()
        revision = await store.create_workspace(str(workspace.resolve()))
        conversation = await store.create_conversation(revision.workspace_id)
        await store.enqueue_request(conversation.id, "stale in-flight request")
        stale = await store.start_next_turn(conversation.id, base_seed=1)
        assert stale is not None
        await store.enqueue_request(conversation.id, "recover me")

        runtime = CapturingRuntime()
        service = ApplicationService(
            store=store,
            observability_store=ObservabilityStore(tmp_path / "observability.sqlite3"),
            config=load_config(),
            runtime=runtime,
            estimator=FakeEstimator(),
            allowed_workspace_roots=(workspace,),
        )
        await service.initialize()
        try:
            await _wait_until_idle(service, conversation.id)
            turns = await store.list_turns(conversation.id)

            assert len(runtime.requests) == 1
            assert [turn.status for turn in turns] == [
                TurnStatus.FINISHED,
                TurnStatus.FINISHED,
            ]
            assert turns[0].terminal_outcome is not None
            assert turns[0].terminal_outcome.kind is TerminalOutcomeKind.ABANDONED
            assert turns[0].terminal_outcome.reason_code == "recovered_stale_active_turn"
            assert turns[1].terminal_outcome is not None
            assert turns[1].terminal_outcome.kind is TerminalOutcomeKind.COMPLETED
        finally:
            await service.shutdown()

    asyncio.run(scenario())


def test_unready_runtime_profile_fails_closed_without_model_generation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runtime = UnreadyRuntime()
        service = ApplicationService(
            store=ConversationStore(tmp_path / "conversations.sqlite3"),
            observability_store=ObservabilityStore(tmp_path / "observability.sqlite3"),
            config=load_config(),
            runtime=runtime,
            estimator=FakeEstimator(),
            allowed_workspace_roots=(workspace,),
        )
        await service.initialize()
        try:
            assert service.readiness == EngineReadiness(
                ready=False, reason_code="model_digest_mismatch"
            )
            conversation = await service.create_conversation(str(workspace))
            await service.enqueue_request(conversation.id, "must not reach runtime")
            await _wait_until_idle(service, conversation.id)

            turns = await service.store.list_turns(conversation.id)
            assert runtime.requests == []
            assert turns[0].terminal_outcome is not None
            assert turns[0].terminal_outcome.kind is TerminalOutcomeKind.BLOCKED
            assert turns[0].terminal_outcome.reason_code == "model_digest_mismatch"
        finally:
            await service.shutdown()

    asyncio.run(scenario())


async def _gate_conversation(
    tmp_path: Path, sink: RecordingEventSink
) -> tuple[OperatorConfirmationGate, ConversationStore, str]:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    await store.initialize()
    workspace = await store.create_workspace(str(tmp_path))
    conversation = await store.create_conversation(workspace.workspace_id)
    return OperatorConfirmationGate(sink, store), store, conversation.id


async def _until_pending(gate: OperatorConfirmationGate, conversation_id: str) -> None:
    for _ in range(200):
        if gate.pending(conversation_id) is not None:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("confirmation was never announced")


def _confirmation_request(
    conversation_id: str,
    *,
    reason_code: str = "web_taint_confirmation_required",
    arguments: Mapping[str, JsonValue] | None = None,
) -> ConfirmationRequest:
    return ConfirmationRequest(
        id="turn-1-confirmation-2",
        conversation_id=conversation_id,
        turn_id="turn-1",
        request_id="request-1",
        step_sequence=2,
        reason_code=reason_code,
        tool_calls=(
            ToolCall(
                id="write-1",
                name="write_file",
                arguments=arguments or {"file_path": "report.txt", "content": "from the web"},
            ),
        ),
    )


def test_operator_confirmation_gate_announces_then_waits_for_a_matching_decision(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        sink = RecordingEventSink()
        gate, _, conversation_id = await _gate_conversation(tmp_path, sink)
        request = _confirmation_request(conversation_id)

        waiting = asyncio.create_task(gate.confirm(request))
        await _until_pending(gate, conversation_id)

        assert gate.pending(conversation_id) == request
        assert [event.kind for event in sink.events] == [AgentEventKind.CONFIRMATION_REQUIRED]
        assert sink.events[0].payload["confirmation_id"] == request.id
        assert sink.events[0].payload["tool_calls"] == [
            {
                "id": "write-1",
                "name": "write_file",
                "arguments": {"file_path": "report.txt", "content": "from the web"},
            }
        ]

        with pytest.raises(ApplicationServiceError) as unknown:
            await gate.resolve(conversation_id, "another-confirmation", approved=True)
        assert unknown.value.code == "confirmation_not_pending"
        assert not waiting.done()

        await gate.resolve(conversation_id, request.id, approved=True)
        decision = await asyncio.wait_for(waiting, timeout=1)

        assert decision.approved is True
        assert decision.reason_code == "web_taint_confirmation_approved"
        assert gate.pending(conversation_id) is None

    asyncio.run(scenario())


def test_operator_confirmation_gate_denies_and_abandons_without_an_answer(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        gate, _, conversation_id = await _gate_conversation(tmp_path, RecordingEventSink())
        request = _confirmation_request(conversation_id)

        denial = asyncio.create_task(gate.confirm(request))
        await _until_pending(gate, conversation_id)
        await gate.resolve(conversation_id, request.id, approved=False)
        denied = await asyncio.wait_for(denial, timeout=1)

        stopped = asyncio.create_task(gate.confirm(request))
        await _until_pending(gate, conversation_id)
        gate.abandon(conversation_id, "operator_stop")
        abandoned = await asyncio.wait_for(stopped, timeout=1)

        assert denied.approved is False
        assert denied.reason_code == "web_taint_confirmation_denied"
        assert abandoned.approved is False
        assert abandoned.reason_code == "operator_stop"
        # An abandon with nothing pending is how stop() behaves on a normal Turn.
        gate.abandon(conversation_id, "operator_stop")

    asyncio.run(scenario())


def test_a_waived_write_is_approved_without_asking_but_a_tainted_one_still_asks(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        sink = RecordingEventSink()
        gate, store, conversation_id = await _gate_conversation(tmp_path, sink)
        write = _confirmation_request(conversation_id, reason_code="write_confirmation_required")

        # Approving with the box ticked is what records the waiver.
        asking = asyncio.create_task(gate.confirm(write))
        await _until_pending(gate, conversation_id)
        await gate.resolve(conversation_id, write.id, approved=True, waive=True)
        first = await asyncio.wait_for(asking, timeout=1)
        assert first.reason_code == "write_confirmation_approved"
        assert await store.waived_confirmations(conversation_id) == frozenset({"workspace_write"})

        silent = await gate.confirm(write)
        assert silent.approved is True
        assert silent.reason_code == "write_confirmation_waived"
        assert len(sink.events) == 1  # nothing announced the second time

        # The waiver answers "the model wants to write", not "the web wants it to".
        tainted = _confirmation_request(conversation_id)
        waiting = asyncio.create_task(gate.confirm(tainted))
        await _until_pending(gate, conversation_id)
        assert gate.pending(conversation_id) == tainted
        assert len(sink.events) == 2
        await gate.resolve(conversation_id, tainted.id, approved=False)
        assert (await asyncio.wait_for(waiting, timeout=1)).approved is False

        await store.revoke_confirmation_waiver(conversation_id, "workspace_write")
        assert await store.waived_confirmations(conversation_id) == frozenset()

    asyncio.run(scenario())


def test_approving_a_grant_dialog_records_the_grant_the_reason_code_names(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        sink = RecordingEventSink()
        gate, store, conversation_id = await _gate_conversation(tmp_path, sink)

        for reason_code, permission in (
            ("write_grant_required", "WriteGrant"),
            ("web_access_grant_required", "WebAccessGrant"),
        ):
            request = _confirmation_request(conversation_id, reason_code=reason_code)
            asking = asyncio.create_task(gate.confirm(request))
            await _until_pending(gate, conversation_id)
            await gate.resolve(conversation_id, request.id, approved=True)
            decision = await asyncio.wait_for(asking, timeout=1)
            assert decision.approved is True
            policy = await store.get_session_policy(conversation_id)
            assert permission in policy.effective_grants

        # Approving under taint still grants nothing — that decision is per call.
        tainted = _confirmation_request(conversation_id)
        waiting = asyncio.create_task(gate.confirm(tainted))
        await _until_pending(gate, conversation_id)
        await gate.resolve(conversation_id, tainted.id, approved=True)
        await asyncio.wait_for(waiting, timeout=1)
        policy = await store.get_session_policy(conversation_id)
        assert sorted(policy.effective_grants) == ["WebAccessGrant", "WriteGrant"]

    asyncio.run(scenario())


def test_yolo_answers_every_confirmation_including_the_tainted_write(tmp_path: Path) -> None:
    async def scenario() -> None:
        sink = RecordingEventSink()
        gate, store, conversation_id = await _gate_conversation(tmp_path, sink)
        tainted = _confirmation_request(conversation_id)

        # Default off: the tainted write is still announced and still asked.
        asking = asyncio.create_task(gate.confirm(tainted))
        await _until_pending(gate, conversation_id)
        await gate.resolve(conversation_id, tainted.id, approved=False)
        assert (await asyncio.wait_for(asking, timeout=1)).approved is False

        await store.set_yolo_enabled(True)
        silent = await gate.confirm(tainted)
        assert silent.approved is True
        assert silent.reason_code == "web_taint_confirmation_waived"
        assert await gate.will_announce(tainted) is False
        assert len(sink.events) == 1  # the second decision announced nothing

        # Missing grants are answered too, so a Turn never stops to ask for one.
        for reason_code, permission in (
            ("write_grant_required", "WriteGrant"),
            ("web_access_grant_required", "WebAccessGrant"),
        ):
            decision = await gate.confirm(
                _confirmation_request(conversation_id, reason_code=reason_code)
            )
            assert decision.approved is True
            policy = await store.get_session_policy(conversation_id)
            assert permission in policy.effective_grants

        # Opting one Conversation out leaves the host-wide answer alone.
        await store.set_conversation_yolo_disabled(conversation_id, True)
        assert await store.yolo_enabled() is True
        assert await store.yolo_active(conversation_id) is False
        assert await gate.will_announce(tainted) is True

    asyncio.run(scenario())


def test_benchmark_lease_waits_for_active_turn_and_keeps_new_chat_queued(
    tmp_path: Path,
) -> None:
    class GatedRuntime(CapturingRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                self.entered.set()
                await self.release.wait()
            return ModelResponse(content="ok")

    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        lease = BenchmarkLease()
        runtime = GatedRuntime()
        service = ApplicationService(
            store=ConversationStore(tmp_path / "conversations.sqlite3"),
            observability_store=ObservabilityStore(tmp_path / "observability.sqlite3"),
            config=load_config(),
            runtime=runtime,
            estimator=FakeEstimator(),
            allowed_workspace_roots=(workspace,),
            benchmark_lease=lease,
        )
        await service.initialize()
        try:
            active = await service.create_conversation(str(workspace), name="active")
            queued = await service.create_conversation(str(workspace), name="queued")
            await service.enqueue_request(active.id, "active turn")
            await asyncio.wait_for(runtime.entered.wait(), timeout=1)

            benchmark = asyncio.create_task(lease.acquire("benchmark-run"))
            await asyncio.sleep(0)
            await service.enqueue_request(queued.id, "must remain queued")
            await asyncio.sleep(0.02)

            queued_snapshot = await service.chat_snapshot(queued.id)
            assert queued_snapshot["active_turn"] is None
            assert len(cast(list[object], queued_snapshot["pending_requests"])) == 1
            assert not benchmark.done()

            runtime.release.set()
            await asyncio.wait_for(benchmark, timeout=1)
            await _wait_until_idle(service, active.id)
            assert len(runtime.requests) == 1
            queued_after_acquire = await service.chat_snapshot(queued.id)
            assert len(cast(list[object], queued_after_acquire["pending_requests"])) == 1

            await lease.release("benchmark-run")
            await _wait_until_idle(service, queued.id)
            assert len(runtime.requests) == 2
        finally:
            await service.shutdown()

    asyncio.run(scenario())


def test_telemetry_records_context_usage_and_truncation_without_any_content(
    tmp_path: Path,
) -> None:
    """A telemetria passa a responder "isso está sendo cortado?" com número.

    `meta.truncated` já existia em todo ToolResult e morria no caminho; sem ele
    não há como saber se os tetos das tools apertam, e mexer num teto viraria
    palpite. O que sobe é só o booleano: `meta` inteiro carrega `final_url`, que
    é conteúdo, e o store de telemetria existe para não guardar conteúdo.
    """

    class ReadingRuntime(CapturingRuntime):
        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="call-1",
                            name="read_file",
                            arguments={"file_path": "notes.txt", "max_lines": 1},
                        ),
                    )
                )
            return ModelResponse(content="li a primeira linha")

    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "notes.txt").write_text("uma\nduas\ntres\n", encoding="utf-8")
        service = ApplicationService(
            store=ConversationStore(tmp_path / "conversations.sqlite3"),
            observability_store=ObservabilityStore(tmp_path / "observability.sqlite3"),
            config=load_config(),
            runtime=ReadingRuntime(),
            estimator=FakeEstimator(),
            allowed_workspace_roots=(workspace,),
        )
        await service.initialize()
        try:
            conversation = await service.create_conversation(str(workspace))
            await service.enqueue_request(conversation.id, "leia a nota")
            await _wait_until_idle(service, conversation.id)

            events = await service.observability_store.list_events()
            context_events = [
                event for event in events if event.event_type == "agent.context_built"
            ]
            window = load_config().context.initial_budget_tokens
            assert len(context_events) == len(
                [event for event in events if event.event_type == "agent.step_started"]
            )
            for event in context_events:
                assert event.payload["context_window"] == window
                assert event.payload["output_budget"] == 8192
                assert event.payload["dropped_turn_ids"] == []
                assert cast(int, event.payload["estimated_input_tokens"]) > 0

            results = [event for event in events if event.event_type == "agent.tool_result"]
            assert [event.payload["truncated"] for event in results] == [True]

            # Nada de conteúdo: nem a linha lida, nem o caminho, nem o final_url
            # que `meta` carrega nos resultados de web.
            payloads = json.dumps([event.payload for event in events])
            assert "uma" not in payloads
            assert "notes.txt" not in payloads
            assert "final_url" not in payloads
        finally:
            await service.shutdown()

    asyncio.run(scenario())
