import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Self, cast

from .config import QWEN_REASONING_MARKUP, QWEN_TOOL_MARKUP, RuntimeProfileConfig
from .context_builder import (
    MODEL_VIEW_ROOTS,
    ContextBudgetExceeded,
    ContextBuilder,
    ContextTurn,
    ModelContext,
)
from .conversation_store import ConversationStore
from .domain import (
    CanonicalHistoryEntry,
    CanonicalHistoryEntryKind,
    JsonValue,
    PendingRequest,
    TerminalOutcomeKind,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    Turn,
    effective_tool_calls,
    tool_call_signature,
)
from .ports import (
    AgentEvent,
    AgentEventKind,
    ConfirmationDecision,
    ConfirmationGate,
    ConfirmationPreview,
    ConfirmationRequest,
    DenyingConfirmationGate,
    EngineReadiness,
    EventSink,
    MalformedModelResponseError,
    ModelGenerationTimeout,
    ModelRequest,
    ModelResponse,
    ModelRuntime,
    ModelRuntimeError,
    NeverStopSignal,
    StopSignal,
    ToolExecutor,
    ToolExecutorFactory,
    ToolSchema,
    TurnRetrieval,
)

# The runtime also emits tool calls as markup, not only as JSON. A leaked payload
# starts or ends on one of these tags.
#
# The ModelView's own envelope belongs here too. The model reads that markup in
# every step and does echo it back: asked to fetch a page it once answered with
# `<model_attempt><content><null/></content><tool_calls>…`, which is the harness
# describing an attempt, not the model answering. Persisted as a final response,
# that markup reaches the Operator as the answer.
#
# The template's own tags change with the model family and come from the
# RuntimeProfile; the envelope is the harness's and is always checked.
_MODEL_VIEW_TAGS = (
    *(f"<{root}>" for root in MODEL_VIEW_ROOTS),
    *(f"</{root}>" for root in MODEL_VIEW_ROOTS),
)


@dataclass(frozen=True, slots=True)
class ResponseMarkup:
    """The template markup a final body must not contain, per model family.

    Reasoning is transient by contract and never enters CanonicalHistory. A
    reasoning marker in the final body means the transient channel bled into the
    persisted answer, so the body is rejected wherever the marker sits, not only
    at its edges.
    """

    tool_call_tags: tuple[str, ...] = QWEN_TOOL_MARKUP
    reasoning_tags: tuple[str, ...] = QWEN_REASONING_MARKUP

    @classmethod
    def of(cls, profile: RuntimeProfileConfig) -> Self:
        return cls(profile.tool_markup_leak_markers, profile.reasoning_leak_markers)

    @property
    def leak_tags(self) -> tuple[str, ...]:
        return (*self.tool_call_tags, *_MODEL_VIEW_TAGS)


class AgentEngine:
    def __init__(
        self,
        *,
        store: ConversationStore,
        runtime: ModelRuntime,
        tool_executor: ToolExecutor | None = None,
        tool_executor_factory: ToolExecutorFactory | None = None,
        context_builder: ContextBuilder,
        event_sink: EventSink,
        system_prompt: str,
        tool_schemas: Sequence[ToolSchema],
        model_options: Mapping[str, JsonValue],
        seed: int | None,
        max_model_invocations: int = 15,
        max_tool_calls_per_step: int = 4,
        max_tool_calls_per_turn: int = 20,
        max_read_calls_per_turn: int = 40,
        max_malformed_model_attempts: int = 2,
        tool_effects: Mapping[str, Sequence[str]] | None = None,
        max_turn_duration_seconds: float = 900,
        runtime_readiness: EngineReadiness | None = None,
        stop_signal: StopSignal | None = None,
        confirmation_gate: ConfirmationGate | None = None,
        turn_retrieval: TurnRetrieval | None = None,
        think: bool = True,
        response_markup: ResponseMarkup | None = None,
    ) -> None:
        if max_model_invocations < 1:
            raise ValueError("max_model_invocations must be positive")
        if max_tool_calls_per_step < 1 or max_tool_calls_per_turn < 1:
            raise ValueError("tool call limits must be positive")
        if max_read_calls_per_turn < 1:
            raise ValueError("tool call limits must be positive")
        if max_malformed_model_attempts < 1:
            raise ValueError("max_malformed_model_attempts must be positive")
        if max_turn_duration_seconds <= 0:
            raise ValueError("max_turn_duration_seconds must be positive")
        if seed is not None and (seed < 0 or seed + max_model_invocations - 1 > 2**63 - 1):
            raise ValueError("seed range must fit in a non-negative signed 64-bit integer")
        if (tool_executor is None) == (tool_executor_factory is None):
            raise ValueError("provide exactly one tool executor or tool executor factory")
        self._store = store
        self._runtime = runtime
        self._tool_executor_factory = (
            tool_executor_factory
            if tool_executor_factory is not None
            else _StaticToolExecutorFactory(tool_executor, tool_schemas)
        )
        self._context_builder = context_builder
        self._event_sink = event_sink
        self._system_prompt = system_prompt
        self._model_options = model_options
        # Vem de `sampling.thinking` da ExecutionRoute. Fixo em True, um perfil
        # sem modo de raciocínio recebia o pedido mesmo assim, e o braço
        # `thinking_ollama` desligado rodava igual ao ligado.
        self._think = think
        self._response_markup = response_markup or ResponseMarkup()
        self._seed = seed
        self._max_model_invocations = max_model_invocations
        self._max_tool_calls_per_step = max_tool_calls_per_step
        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._max_read_calls_per_turn = max_read_calls_per_turn
        self._max_malformed_model_attempts = max_malformed_model_attempts
        self._tool_effects = dict(tool_effects or {})
        self._max_turn_duration_seconds = max_turn_duration_seconds
        self._runtime_readiness = runtime_readiness or EngineReadiness(ready=True)
        self._stop_signal = stop_signal or NeverStopSignal()
        self._confirmation_gate = confirmation_gate or DenyingConfirmationGate()
        self._turn_retrieval = turn_retrieval

    @property
    def readiness(self) -> EngineReadiness:
        if not self._runtime_readiness.ready:
            return self._runtime_readiness
        return self._context_builder.readiness

    async def enqueue(self, conversation_id: str, content: str) -> PendingRequest:
        return await self._store.enqueue_request(conversation_id, content)

    async def run(self, conversation_id: str, content: str) -> Turn:
        await self.enqueue(conversation_id, content)
        turn = await self.start_next_turn(conversation_id)
        if turn is None:
            raise RuntimeError("queued request did not produce a turn")
        return turn

    async def start_next_turn(self, conversation_id: str) -> Turn | None:
        turn = await self._store.start_next_turn(conversation_id, base_seed=self._seed)
        if turn is None:
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_turn_duration_seconds
        try:
            try:
                async with asyncio.timeout(self._max_turn_duration_seconds):
                    request = await self._store.get_request(turn.request_id)
                    await self._store.append_canonical_history(
                        turn.id,
                        CanonicalHistoryEntryKind.USER_MESSAGE,
                        {"content": request.content},
                    )
                    readiness = self.readiness
                    if not readiness.ready:
                        return await self._finish(
                            turn,
                            TerminalOutcomeKind.BLOCKED,
                            readiness.reason_code or "engine_not_ready",
                        )
                    if self._stop_signal.stop_requested:
                        return await self._finish(
                            turn, TerminalOutcomeKind.CANCELLED, "operator_stop"
                        )
                    await self._retrieve_for_turn(turn, request.content)
                    return await self._run_active_turn(turn, deadline)
            except TimeoutError:
                if not _deadline_reached(deadline):
                    raise
                return await self._finish(
                    turn,
                    TerminalOutcomeKind.LIMIT_REACHED,
                    "turn_time_budget_exhausted",
                )
        except ContextBudgetExceeded as error:
            return await self._finish(
                turn,
                TerminalOutcomeKind.FAILED,
                "context_budget_exceeded",
                detail=str(error),
            )
        except asyncio.CancelledError:
            await self._finish(turn, TerminalOutcomeKind.CANCELLED, "engine_cancelled")
            raise
        except MalformedModelResponseError as error:
            # The loop catches these to spend a retry, so nothing should reach
            # here. If something does, it is still an answer the harness refused
            # to read — not the provider refusing to answer, which is what the
            # clause below means.
            return await self._finish(
                turn,
                TerminalOutcomeKind.FAILED,
                "malformed_model_response",
                detail=str(error),
            )
        except ModelGenerationTimeout as error:
            # Antes da cláusula abaixo de propósito: é subclasse dela, e quem
            # cortou a geração foi o harness. Chamar isso de provedor
            # indisponível mandava o Operator reiniciar um Ollama que estava no
            # ar e ainda gerando.
            return await self._finish(
                turn,
                TerminalOutcomeKind.FAILED,
                "model_generation_timeout",
                detail=_provider_detail(error),
            )
        except ModelRuntimeError as error:
            # The provider refused or was unreachable. That is not the harness
            # crashing, and an Operator reading engine_error for both has no way
            # to tell whether to restart Ollama or report a bug.
            return await self._finish(
                turn,
                TerminalOutcomeKind.FAILED,
                ("model_provider_unavailable" if error.retryable else "model_provider_error"),
                detail=_provider_detail(error),
            )
        except Exception as error:
            return await self._finish(
                turn,
                TerminalOutcomeKind.FAILED,
                "engine_error",
                detail=f"{type(error).__name__}: {error}",
            )

    def _over_budget(
        self,
        calls: Sequence[ToolCall],
        results: Sequence[ToolResult],
        pending: Sequence[ToolCall],
    ) -> bool:
        """Budgets by effect class: a read is not the expense a write or an egress is.

        Reading inside the workspace jail costs a file handle; writing changes the
        Operator's files and an egress leaves the machine. One ceiling for both
        made the cheap thing as scarce as the dangerous one, and the corpus kept
        catching Turns that spent their whole allowance re-reading what they had
        already been told.
        """
        spent = [*effective_tool_calls(calls, results), *pending]
        effecting = sum(1 for call in spent if not self._is_read_only(call))
        reads = len(spent) - effecting
        return effecting > self._max_tool_calls_per_turn or reads > self._max_read_calls_per_turn

    def _is_read_only(self, call: ToolCall) -> bool:
        effects = self._tool_effects.get(call.name)
        # An unknown tool is never read-only: it is about to be refused, and a
        # refusal that costs nothing is an unlimited retry.
        return bool(effects) and all(effect == "workspace_read" for effect in effects)

    async def _run_active_turn(self, turn: Turn, deadline: float) -> Turn:
        rejected_count = 0
        turn_calls: list[ToolCall] = []
        turn_results: list[ToolResult] = []
        seen_signatures: set[str] = set()
        for step_sequence in range(1, self._max_model_invocations + 1):
            step_seed = turn.base_seed + step_sequence - 1
            if _deadline_reached(deadline):
                return await self._finish_time_limit(turn)
            if self._stop_signal.stop_requested:
                return await self._finish(turn, TerminalOutcomeKind.CANCELLED, "operator_stop")
            await self._emit(
                AgentEvent(
                    kind=AgentEventKind.STEP_STARTED,
                    turn_id=turn.id,
                    step_sequence=step_sequence,
                    payload={"base_seed": turn.base_seed, "seed": step_seed},
                    conversation_id=turn.conversation_id,
                    request_id=turn.request_id,
                )
            )
            effective_tools = await self._tool_executor_factory.effective_tool_schemas(
                turn.conversation_id
            )
            if _deadline_reached(deadline):
                return await self._finish_time_limit(turn)
            offered_tools = () if step_sequence == self._max_model_invocations else effective_tools
            context = await self._build_context(turn, offered_tools)
            await self._emit_context_built(turn, step_sequence, context)
            if _deadline_reached(deadline):
                return await self._finish_time_limit(turn)
            request = ModelRequest(
                messages=context.messages,
                tools=tuple(offered_tools),
                options=self._model_options,
                seed=step_seed,
                max_output_tokens=context.output_budget,
                think=self._think,
            )
            try:
                response = await self._runtime.generate(request)
                await self._emit_generation_stats(turn, step_sequence, response)
                _validate_response(response, self._response_markup)
            except MalformedModelResponseError as error:
                rejected_count += 1
                await self._store.append_canonical_history(
                    turn.id,
                    CanonicalHistoryEntryKind.REJECTED_MODEL_ATTEMPT,
                    _malformed_payload(error),
                )
                await self._store.append_agent_step(turn.id, seed=step_seed)
                await self._emit_step_finished(turn, step_sequence)
                if rejected_count >= self._max_malformed_model_attempts:
                    return await self._finish(
                        turn,
                        TerminalOutcomeKind.FAILED,
                        "malformed_model_response_limit",
                    )
                if step_sequence == self._max_model_invocations:
                    return await self._finish(
                        turn,
                        TerminalOutcomeKind.LIMIT_REACHED,
                        "model_invocation_limit",
                    )
                continue

            if self._stop_signal.stop_requested:
                return await self._finish(turn, TerminalOutcomeKind.CANCELLED, "operator_stop")
            if _deadline_reached(deadline):
                await self._emit_step_finished(turn, step_sequence)
                return await self._finish_time_limit(turn)

            await self._emit_reasoning(turn, step_sequence, response)
            if len(response.tool_calls) > self._max_tool_calls_per_step:
                rejected_count += 1
                await self._store.append_canonical_history(
                    turn.id,
                    CanonicalHistoryEntryKind.REJECTED_MODEL_ATTEMPT,
                    _oversized_batch_payload(response, self._max_tool_calls_per_step),
                )
                await self._store.append_agent_step(
                    turn.id, seed=step_seed, tool_calls=response.tool_calls
                )
                await self._emit_step_finished(turn, step_sequence)
                if rejected_count >= self._max_malformed_model_attempts:
                    return await self._finish(
                        turn,
                        TerminalOutcomeKind.FAILED,
                        "rejected_model_attempt_limit",
                    )
                if step_sequence == self._max_model_invocations:
                    return await self._finish(
                        turn,
                        TerminalOutcomeKind.LIMIT_REACHED,
                        "model_invocation_limit",
                    )
                continue

            await self._store.append_canonical_history(
                turn.id,
                CanonicalHistoryEntryKind.MODEL_ATTEMPT,
                _model_attempt_payload(response),
            )
            for call in response.tool_calls:
                await self._emit(
                    AgentEvent(
                        kind=AgentEventKind.TOOL_CALL,
                        turn_id=turn.id,
                        step_sequence=step_sequence,
                        payload=_tool_call_payload(call),
                        conversation_id=turn.conversation_id,
                        request_id=turn.request_id,
                    )
                )

            if step_sequence == self._max_model_invocations:
                await self._store.append_agent_step(
                    turn.id, seed=step_seed, tool_calls=response.tool_calls
                )
                if response.content is not None:
                    await self._append_final_response(turn, step_sequence, response.content)
                await self._emit_step_finished(turn, step_sequence)
                return await self._finish(
                    turn,
                    TerminalOutcomeKind.LIMIT_REACHED,
                    "model_invocation_limit",
                )

            if not response.tool_calls:
                await self._store.append_agent_step(turn.id, seed=step_seed)
                if response.content is None:
                    raise MalformedModelResponseError("model response has no final content")
                await self._append_final_response(turn, step_sequence, response.content)
                await self._emit_step_finished(turn, step_sequence)
                return await self._finish(
                    turn,
                    TerminalOutcomeKind.COMPLETED,
                    "final_response",
                )

            calls = response.tool_calls
            # A call whose name and arguments were already seen this Turn is admitted
            # optimistically: whether it is a repeat depends on the payload, which only
            # exists after it runs. The count is reconciled below, and a Turn can
            # overrun by at most one batch before the next check ends it.
            fresh = tuple(
                call for call in calls if tool_call_signature(call) not in seen_signatures
            )
            if self._over_budget(turn_calls, turn_results, fresh):
                await self._store.append_agent_step(turn.id, seed=step_seed, tool_calls=calls)
                await self._emit_step_finished(turn, step_sequence)
                return await self._finish(
                    turn,
                    TerminalOutcomeKind.LIMIT_REACHED,
                    "tool_calls_per_turn_limit",
                )

            tool_executor = await self._tool_executor_factory.create(turn.conversation_id)
            if self._stop_signal.stop_requested:
                return await self._finish(turn, TerminalOutcomeKind.CANCELLED, "operator_stop")
            if _deadline_reached(deadline):
                return await self._finish_time_limit(turn)
            preflight = await tool_executor.preflight(calls)
            if _deadline_reached(deadline):
                return await self._finish_time_limit(turn)

            # A missing grant used to end the Turn, so the Operator answered the
            # same question twice: once by hunting for a chip before anything had
            # happened, and again in the dialog — and for WebAccessGrant they also
            # had to re-send the prompt. Asking here folds them into one decision,
            # taken with the call in view. The grant is still the thing policy
            # requires — approving is how the Operator gives it. WorkspaceRootGrant
            # stays out: it comes from the server's root allowlist, not from a click.
            granted_by_dialog = False
            if not preflight.allowed and preflight.reason_code in _GRANTABLE_PREFLIGHT_REASONS:
                decision = await self._await_confirmation(
                    turn,
                    step_sequence,
                    calls,
                    reason_code=preflight.reason_code,
                    previews=await _previews(tool_executor, calls),
                )
                if self._stop_signal.stop_requested:
                    return await self._finish(turn, TerminalOutcomeKind.CANCELLED, "operator_stop")
                if _deadline_reached(deadline):
                    return await self._finish_time_limit(turn)
                if not decision.approved:
                    await self._append_blocked_automation(
                        turn,
                        decision.reason_code,
                        automation_id=_CONFIRMATION_AUTOMATION_ID,
                    )
                    await self._store.append_agent_step(turn.id, seed=step_seed, tool_calls=calls)
                    await self._emit_step_finished(turn, step_sequence)
                    return await self._finish(
                        turn, TerminalOutcomeKind.BLOCKED, decision.reason_code
                    )
                # The gate records the grant; the executor is rebuilt so it reads the
                # policy that now exists, and the batch is judged again against it.
                tool_executor = await self._tool_executor_factory.create(turn.conversation_id)
                preflight = await tool_executor.preflight(calls)
                granted_by_dialog = preflight.allowed

            if not preflight.allowed:
                reason_code = preflight.reason_code or "tool_batch_blocked"
                await self._append_blocked_automation(
                    turn,
                    reason_code,
                    detail=preflight.detail,
                )
                await self._store.append_agent_step(turn.id, seed=step_seed, tool_calls=calls)
                await self._emit_step_finished(turn, step_sequence)
                if reason_code in _MODEL_FIXABLE_PREFLIGHT_REASONS:
                    rejected_count += 1
                    if rejected_count >= self._max_malformed_model_attempts:
                        return await self._finish(
                            turn,
                            TerminalOutcomeKind.FAILED,
                            "rejected_model_attempt_limit",
                        )
                    if step_sequence == self._max_model_invocations:
                        return await self._finish(
                            turn,
                            TerminalOutcomeKind.LIMIT_REACHED,
                            "model_invocation_limit",
                        )
                    continue
                return await self._finish(
                    turn,
                    TerminalOutcomeKind.BLOCKED,
                    reason_code,
                    detail=preflight.detail,
                )

            # The Operator is asked only about a batch that already passed schema,
            # grants and path validation: a dialog about a call that could never
            # run teaches them nothing, and the preview below needs a path the
            # executor has already resolved. One decision per batch — an Operator
            # who just approved this same diff to give the grant is not asked to
            # approve it again.
            confirmation_reason = (
                None
                if granted_by_dialog
                else _confirmation_reason_code(calls, context, self._tool_effects)
            )
            if confirmation_reason is not None:
                decision = await self._await_confirmation(
                    turn,
                    step_sequence,
                    calls,
                    reason_code=confirmation_reason,
                    previews=await _previews(tool_executor, calls),
                )
                if self._stop_signal.stop_requested:
                    return await self._finish(turn, TerminalOutcomeKind.CANCELLED, "operator_stop")
                if _deadline_reached(deadline):
                    return await self._finish_time_limit(turn)
                if not decision.approved:
                    await self._append_blocked_automation(
                        turn,
                        decision.reason_code,
                        automation_id=_CONFIRMATION_AUTOMATION_ID,
                    )
                    await self._store.append_agent_step(turn.id, seed=step_seed, tool_calls=calls)
                    await self._emit_step_finished(turn, step_sequence)
                    return await self._finish(
                        turn, TerminalOutcomeKind.BLOCKED, decision.reason_code
                    )

            results: list[ToolResult] = []
            for call in calls:
                if self._stop_signal.stop_requested:
                    return await self._finish(turn, TerminalOutcomeKind.CANCELLED, "operator_stop")
                if _deadline_reached(deadline):
                    return await self._finish_time_limit(turn)
                result = await tool_executor.execute(call)
                if result.tool_call_id != call.id:
                    raise ValueError("tool result does not match its call")
                results.append(result)
                await self._store.append_canonical_history(
                    turn.id,
                    CanonicalHistoryEntryKind.TOOL_RESULT,
                    _tool_result_payload(call, result),
                )
                await self._emit(
                    AgentEvent(
                        kind=AgentEventKind.TOOL_RESULT,
                        turn_id=turn.id,
                        step_sequence=step_sequence,
                        payload={
                            "tool_call_id": result.tool_call_id,
                            "status": result.status.value,
                            "retryable": result.retryable,
                            "data": result.data,
                            "error": result.error,
                            "meta": result.meta,
                        },
                        conversation_id=turn.conversation_id,
                        request_id=turn.request_id,
                    )
                )
            await self._store.append_agent_step(
                turn.id, seed=step_seed, tool_calls=calls, tool_results=results
            )
            await self._emit_step_finished(turn, step_sequence)
            turn_calls.extend(calls)
            turn_results.extend(results)
            # Only a successful call earns the optimistic admission below: a refusal
            # repeated verbatim has to keep costing, or refusing becomes a free retry.
            call_by_id = {call.id: call for call in calls}
            seen_signatures.update(
                tool_call_signature(call_by_id[result.tool_call_id])
                for result in results
                if result.status is ToolResultStatus.SUCCESS and result.tool_call_id in call_by_id
            )
            # A repeat that came back byte-identical bought the Turn nothing, so it
            # does not spend the Turn's budget. The call still ran: nothing is cached.
            if self._over_budget(turn_calls, turn_results, ()):
                # An optimistically admitted repeat came back different, so it was a
                # real second reading after all. The Turn stops here rather than at
                # the next batch.
                return await self._finish(
                    turn,
                    TerminalOutcomeKind.LIMIT_REACHED,
                    "tool_calls_per_turn_limit",
                )
            blocked = next(
                (result for result in results if result.status is ToolResultStatus.BLOCKED),
                None,
            )
            if blocked is not None:
                return await self._finish(
                    turn,
                    TerminalOutcomeKind.BLOCKED,
                    _tool_error_code(blocked) or "tool_result_blocked",
                )

        raise RuntimeError("agent loop exited without a terminal outcome")

    async def _finish_time_limit(self, turn: Turn) -> Turn:
        return await self._finish(
            turn,
            TerminalOutcomeKind.LIMIT_REACHED,
            "turn_time_budget_exhausted",
        )

    async def _retrieve_for_turn(self, turn: Turn, question: str) -> None:
        """Consults the granted Corpus before the model gets its first word.

        A failure here does not end the Turn, and it does not disappear either:
        the entry says the Corpus was not consulted and why, so the model reads
        that instead of assuming the passages simply did not exist — and the
        Operator can tell "the acervo has nothing" from "the embedder is down".
        """
        if self._turn_retrieval is None:
            return
        try:
            payload = await self._turn_retrieval.for_turn(turn.conversation_id, question)
        except Exception as error:
            await self._store.append_canonical_history(
                turn.id,
                CanonicalHistoryEntryKind.INTERNAL_AUTOMATION,
                {
                    "automation_id": _RETRIEVAL_AUTOMATION_ID,
                    "status": "failed",
                    "detail": f"{type(error).__name__}: {error}",
                },
            )
            return
        if payload is None:
            return
        await self._store.append_canonical_history(
            turn.id,
            CanonicalHistoryEntryKind.INTERNAL_AUTOMATION,
            {
                "automation_id": _RETRIEVAL_AUTOMATION_ID,
                "status": "completed",
                **payload,
            },
        )

    async def _append_blocked_automation(
        self,
        turn: Turn,
        reason_code: str,
        *,
        detail: str | None = None,
        automation_id: str = "tool_batch_blocked",
    ) -> None:
        await self._store.append_canonical_history(
            turn.id,
            CanonicalHistoryEntryKind.INTERNAL_AUTOMATION,
            {
                "automation_id": automation_id,
                "status": "blocked",
                "reason_code": reason_code,
                "detail": detail,
                "instruction": BLOCKED_INSTRUCTION,
            },
        )

    async def _await_confirmation(
        self,
        turn: Turn,
        step_sequence: int,
        calls: Sequence[ToolCall],
        *,
        reason_code: str,
        previews: tuple[ConfirmationPreview, ...] = (),
    ) -> ConfirmationDecision:
        request = ConfirmationRequest(
            id=f"{turn.id}-confirmation-{step_sequence}",
            conversation_id=turn.conversation_id,
            turn_id=turn.id,
            request_id=turn.request_id,
            step_sequence=step_sequence,
            reason_code=reason_code,
            tool_calls=tuple(calls),
            previews=previews,
        )
        # The question is recorded before the wait, not after: a Turn that dies
        # parked on a confirmation has this entry and nothing else to explain it.
        # It is only recorded when there is a question — a gate answering from a
        # waiver decides without asking, and the history says so.
        announced = await self._confirmation_gate.will_announce(request)
        if announced:
            await self._store.append_canonical_history(
                turn.id,
                CanonicalHistoryEntryKind.INTERNAL_AUTOMATION,
                {
                    "automation_id": _CONFIRMATION_AUTOMATION_ID,
                    "status": "requested",
                    "confirmation_id": request.id,
                    "reason_code": request.reason_code,
                    "tool_calls": [_tool_call_payload(call) for call in calls],
                },
            )
        decision = await self._confirmation_gate.confirm(request)
        await self._store.append_canonical_history(
            turn.id,
            CanonicalHistoryEntryKind.INTERNAL_AUTOMATION,
            {
                "automation_id": _CONFIRMATION_AUTOMATION_ID,
                "status": _decision_status(decision.approved, announced=announced),
                "confirmation_id": request.id,
                "reason_code": decision.reason_code,
                **({} if announced else {"tool_calls": [_tool_call_payload(c) for c in calls]}),
            },
        )
        await self._emit(
            AgentEvent(
                kind=AgentEventKind.CONFIRMATION_RESOLVED,
                turn_id=turn.id,
                step_sequence=step_sequence,
                payload={
                    "confirmation_id": request.id,
                    "approved": decision.approved,
                    "reason_code": decision.reason_code,
                },
                conversation_id=turn.conversation_id,
                request_id=turn.request_id,
            )
        )
        return decision

    async def _build_context(self, turn: Turn, offered_tools: Sequence[ToolSchema]) -> ModelContext:
        history = await self._store.list_canonical_history(turn.conversation_id)
        grouped: dict[str, list[CanonicalHistoryEntry]] = {}
        for entry in history:
            grouped.setdefault(entry.turn_id, []).append(entry)
        current = ContextTurn(turn_id=turn.id, entries=tuple(grouped.pop(turn.id, ())))
        completed = tuple(
            ContextTurn(turn_id=turn_id, entries=tuple(entries))
            for turn_id, entries in grouped.items()
        )
        return self._context_builder.build(
            system=self._system_prompt,
            tool_schemas=offered_tools,
            completed_turns=completed,
            current_turn=current,
        )

    async def _emit_context_built(
        self, turn: Turn, step_sequence: int, context: ModelContext
    ) -> None:
        """Publica o que a janela custou neste passo.

        Só números e IDs: é o mesmo evento que alimenta a barra do painel e a
        telemetria, e o store de telemetria recusa conteúdo por construção.
        """
        await self._emit(
            AgentEvent(
                kind=AgentEventKind.CONTEXT_BUILT,
                turn_id=turn.id,
                step_sequence=step_sequence,
                payload={
                    "estimated_input_tokens": context.estimated_input_tokens,
                    "context_window": context.context_window,
                    "output_budget": context.output_budget,
                    "dropped_turn_ids": list(context.dropped_turn_ids),
                },
                conversation_id=turn.conversation_id,
                request_id=turn.request_id,
            )
        )

    async def _append_final_response(self, turn: Turn, step_sequence: int, content: str) -> None:
        await self._store.append_canonical_history(
            turn.id,
            CanonicalHistoryEntryKind.FINAL_RESPONSE,
            {"content": content},
        )
        await self._emit(
            AgentEvent(
                kind=AgentEventKind.FINAL_RESPONSE,
                turn_id=turn.id,
                step_sequence=step_sequence,
                payload={"content": content},
                conversation_id=turn.conversation_id,
                request_id=turn.request_id,
            )
        )

    async def _emit_reasoning(
        self, turn: Turn, step_sequence: int, response: ModelResponse
    ) -> None:
        if response.reasoning is None:
            return
        await self._emit(
            AgentEvent(
                kind=AgentEventKind.REASONING,
                turn_id=turn.id,
                step_sequence=step_sequence,
                payload={"content": response.reasoning},
                conversation_id=turn.conversation_id,
                request_id=turn.request_id,
            )
        )

    async def _emit_generation_stats(
        self, turn: Turn, step_sequence: int, response: ModelResponse
    ) -> None:
        """Quantos tokens o passo gerou e em quanto tempo: o medidor de tok/s da UI.

        Só contagens e tempos, como toda telemetria. Um runtime que não mede um
        dos dois não emite nada: velocidade sem denominador seria número inventado.
        """
        usage, durations = response.usage, response.durations
        if usage is None or durations is None or not durations.eval_ns:
            return
        payload: dict[str, JsonValue] = {
            "output_tokens": usage.output_tokens,
            "eval_ms": round(durations.eval_ns / 1_000_000, 1),
            "prompt_tokens": usage.input_tokens,
        }
        if durations.prompt_eval_ns:
            payload["prompt_eval_ms"] = round(durations.prompt_eval_ns / 1_000_000, 1)
        await self._emit(
            AgentEvent(
                kind=AgentEventKind.GENERATION_STATS,
                turn_id=turn.id,
                step_sequence=step_sequence,
                payload=payload,
                conversation_id=turn.conversation_id,
                request_id=turn.request_id,
            )
        )

    async def _emit_step_finished(self, turn: Turn, step_sequence: int) -> None:
        await self._emit(
            AgentEvent(
                kind=AgentEventKind.STEP_FINISHED,
                turn_id=turn.id,
                step_sequence=step_sequence,
                payload={},
                conversation_id=turn.conversation_id,
                request_id=turn.request_id,
            )
        )

    async def _finish(
        self,
        turn: Turn,
        kind: TerminalOutcomeKind,
        reason_code: str,
        *,
        detail: str | None = None,
    ) -> Turn:
        finished = await self._store.finish_turn(
            turn.id,
            kind,
            reason_code=reason_code,
            detail=detail,
        )
        await self._emit(
            AgentEvent(
                kind=AgentEventKind.TURN_FINISHED,
                turn_id=turn.id,
                step_sequence=None,
                payload={"outcome_kind": kind.value, "reason_code": reason_code},
                conversation_id=turn.conversation_id,
                request_id=turn.request_id,
            )
        )
        return finished

    async def _emit(self, event: AgentEvent) -> None:
        with suppress(Exception):
            await self._event_sink.emit(event)


def _validate_response(response: ModelResponse, markup: ResponseMarkup) -> None:
    # Blank counts as absent, and so does a body with no word in it. Four runs
    # answered the Operator with a single "]", a fragment of the wire format that
    # is no more an answer than an empty string is.
    blank = response.content is None or not _WORD.search(response.content)
    if blank and not response.tool_calls:
        raise MalformedModelResponseError("model response has neither content nor tool calls")
    for call in response.tool_calls:
        if not call.id or not call.name:
            raise MalformedModelResponseError("model response contains an invalid tool call")
    if response.content is not None and _leaked_reasoning(response.content, markup):
        raise MalformedModelResponseError(
            "model response body carries reasoning markers, which are never persisted"
        )
    if response.content is not None and _is_serialized_tool_call(response.content, markup):
        raise MalformedModelResponseError(
            "model response body is a serialized tool call, not a final answer"
        )


# The status field alone did not carry. Denied a write after reading the web, the
# model still answered "the summary was saved to resumo-web.md" — to the very
# Operator who had just refused it. The entry said status blocked and reason
# web_taint_confirmation_denied, and the model read it and wrote past it. Corpus
# retrieval had the same problem and solved it the same way, so this follows
# `corpus_service.CITATION_INSTRUCTION`: say in words what the structure already
# says, because words are what the model acts on.
BLOCKED_INSTRUCTION = (
    "The calls above did not run. Nothing was written, sent or changed by them, "
    "and there is no result to report. Tell the Operator plainly what was stopped "
    "and why, and never state or imply that the action happened."
)

# An answer contains at least one letter or digit; punctuation alone is a leftover
# of the wire format, not a reply.
_WORD = re.compile(r"\w")

# Effects that a Turn carrying UntrustedWebTaint may not spend without the Operator:
# one changes the Operator's files, the other takes their content off the machine.
_TAINT_CONFIRMED_EFFECTS = frozenset({"workspace_write", "data_egress"})

# One id for every Operator decision, whatever prompted it. The registry declares
# it under the same name, so the CanonicalHistory entry, the contract and the
# dialog are talking about one automation instead of three.
_CONFIRMATION_AUTOMATION_ID = "operator_confirmation"
_RETRIEVAL_AUTOMATION_ID = "corpus_retrieval"

# A preflight refusal the model itself can act on: it named a tool that does not
# exist, or filled its arguments wrong. The blocked automation entry already tells
# it what was wrong, so the Turn gives it the same second attempt an oversized
# batch gets instead of ending. Every other refusal — grant, policy, path, taint —
# stays terminal: no argument the model emits next can lift one.
# Preflight refusals the Operator can lift on the spot, by granting in the dialog.
# workspace_root_grant_required is absent by design: that root comes from the
# server's allowlist, so the answer is in the Settings tab, not in a dialog.
_GRANTABLE_PREFLIGHT_REASONS = frozenset({"write_grant_required", "web_access_grant_required"})
_MODEL_FIXABLE_PREFLIGHT_REASONS = frozenset(
    {
        "invalid_tool_arguments",
        "expected_sha256_required",
        "unknown_tool",
        "tool_not_enabled",
        "tool_not_available",
    }
)


def _leaked_reasoning(content: str, markup: ResponseMarkup) -> bool:
    return any(tag in content for tag in markup.reasoning_tags)


def _is_serialized_tool_call(content: str, markup: ResponseMarkup) -> bool:
    """Detects a tool call emitted as prose instead of through the tool channel.

    The model sometimes answers with the wire payload it should have sent as a
    tool call. Left alone that JSON is persisted as the final response and
    delivered to the Operator as the answer. Rejecting it here costs one step and
    gives the model a chance to correct itself.

    Deliberately narrow: only a body that is *entirely* such a payload counts, so
    an answer that merely quotes JSON is untouched. The same narrowness applies to
    the model's other wire format, which is markup rather than JSON: a body that
    opens or closes on a tool-call tag is the payload, a body that mentions one in
    passing is an answer.
    """
    stripped = content.strip()
    tags = markup.leak_tags
    if stripped.startswith(tags) or stripped.endswith(tags):
        return True
    if stripped.startswith("```"):
        without_fence = stripped[3:].partition("\n")[2]
        stripped = without_fence.rpartition("```")[0].strip() or without_fence.strip()
    if not stripped.startswith("{"):
        return False
    try:
        parsed = json.loads(stripped)
    except ValueError:
        # A truncated payload never closes its braces, and it is still not an answer.
        return '"tool_calls"' in stripped or '"tool_name"' in stripped
    if not isinstance(parsed, Mapping):
        return False
    keys = cast(Mapping[str, object], parsed).keys()
    return "tool_calls" in keys or {"name", "arguments"} <= set(keys)


def _model_attempt_payload(response: ModelResponse) -> Mapping[str, JsonValue]:
    return {
        "content": response.content,
        "tool_calls": [_tool_call_payload(call) for call in response.tool_calls],
    }


def _tool_call_payload(call: ToolCall) -> Mapping[str, JsonValue]:
    return {
        "id": call.id,
        "name": call.name,
        "arguments": call.arguments,
        "idempotency_key": call.idempotency_key,
    }


def _tool_result_payload(call: ToolCall, result: ToolResult) -> Mapping[str, JsonValue]:
    return {
        "tool_call_id": result.tool_call_id,
        "tool_name": call.name,
        "status": result.status.value,
        "retryable": result.retryable,
        "data": result.data,
        "error": result.error,
        "meta": result.meta,
    }


def _malformed_payload(error: MalformedModelResponseError) -> Mapping[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "content": None,
        "error": {"code": "malformed_model_response", "message": str(error)},
    }
    if error.raw is not None:
        payload["raw"] = _without_reasoning(error.raw)
    return payload


def _oversized_batch_payload(response: ModelResponse, limit: int) -> Mapping[str, JsonValue]:
    return {
        **_model_attempt_payload(response),
        "error": {
            "code": "tool_calls_per_step_limit",
            "message": f"A model response may request at most {limit} tool calls.",
        },
    }


def _without_reasoning(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {
            key: _without_reasoning(child)
            for key, child in value.items()
            if not _is_reasoning_key(key)
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_without_reasoning(child) for child in value]
    return value


def _is_reasoning_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in {"reasoning", "thinking"} or normalized.endswith(
        ("_reasoning", "_thinking")
    )


def _decision_status(approved: bool, *, announced: bool) -> str:
    if not approved:
        return "denied"
    return "approved" if announced else "waived"


def _provider_detail(error: ModelRuntimeError) -> str:
    """The provider's own class and message, with its status when it gave one."""
    payload = error.error or {}
    code = payload.get("code")
    status = payload.get("status_code", error.status_code)
    parts = [str(code) if isinstance(code, str) else type(error).__name__, str(error)]
    detail = ": ".join(part for part in parts if part)
    return f"{detail} (HTTP {status})" if isinstance(status, int) else detail


def _tool_error_code(result: ToolResult) -> str | None:
    if result.error is None:
        return None
    code = result.error.get("code")
    return code if isinstance(code, str) else None


def _confirmation_reason_code(
    calls: Sequence[ToolCall],
    context: ModelContext,
    tool_effects: Mapping[str, Sequence[str]],
) -> str | None:
    """Why this batch needs the Operator, or None if it does not.

    Gates by effect, never by tool name: a name list only covers the tools that
    existed when it was written. Two reasons, kept apart because they are not the
    same event and the Operator reads them differently — a write is a write, and
    a write the web asked for is a write the web asked for.
    """
    tainted = "UntrustedWebTaint" in context.taints
    if tainted and any(_confirmable_under_taint(call, tool_effects) for call in calls):
        return "web_taint_confirmation_required"
    if any("workspace_write" in tool_effects.get(call.name, ()) for call in calls):
        return "write_confirmation_required"
    return None


async def _previews(
    executor: ToolExecutor, calls: Sequence[ToolCall]
) -> tuple[ConfirmationPreview, ...]:
    previews = [await executor.preview(call) for call in calls]
    return tuple(preview for preview in previews if preview is not None)


def _confirmable_under_taint(call: ToolCall, tool_effects: Mapping[str, Sequence[str]]) -> bool:
    effects = tool_effects.get(call.name)
    # An unknown tool is never assumed harmless: the same conservative reading that
    # keeps it out of the read budget keeps it inside the gate.
    if not effects:
        return True
    return any(effect in _TAINT_CONFIRMED_EFFECTS for effect in effects)


def _deadline_reached(deadline: float) -> bool:
    return asyncio.get_running_loop().time() >= deadline


class _StaticToolExecutorFactory:
    def __init__(self, executor: ToolExecutor | None, tool_schemas: Sequence[ToolSchema]) -> None:
        if executor is None:
            raise ValueError("tool executor is required")
        self._executor = executor
        self._tool_schemas = tuple(tool_schemas)

    async def effective_tool_schemas(self, conversation_id: str) -> tuple[ToolSchema, ...]:
        del conversation_id
        return self._tool_schemas

    async def create(self, conversation_id: str) -> ToolExecutor:
        del conversation_id
        return self._executor
