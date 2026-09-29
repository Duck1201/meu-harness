import json

from .domain import JsonValue
from .ports import AgentEvent, AgentEventKind

_FINISHED_OUTCOMES = frozenset({"completed", "limit_reached", "cancelled", "blocked", "abandoned"})


def project_agent_event(event: AgentEvent, *, run_id: str) -> list[dict[str, JsonValue]]:
    step_name = f"step-{event.step_sequence}" if event.step_sequence is not None else None
    if event.kind is AgentEventKind.STEP_STARTED:
        return [{"type": "STEP_STARTED", "stepName": step_name}]
    if event.kind is AgentEventKind.STEP_FINISHED:
        return [{"type": "STEP_FINISHED", "stepName": step_name}]
    if event.kind is AgentEventKind.GENERATION_STATS:
        value: dict[str, JsonValue] = {
            "output_tokens": _integer(event, "output_tokens"),
            "eval_ms": _number(event, "eval_ms"),
            "prompt_tokens": _integer(event, "prompt_tokens"),
        }
        prompt_eval_ms = event.payload.get("prompt_eval_ms")
        if isinstance(prompt_eval_ms, int | float):
            value["prompt_eval_ms"] = prompt_eval_ms
        return [{"type": "CUSTOM", "name": "harness.generation_stats", "value": value}]
    if event.kind is AgentEventKind.CONTEXT_BUILT:
        dropped = event.payload.get("dropped_turn_ids")
        return [
            {
                "type": "CUSTOM",
                "name": "harness.context_usage",
                "value": {
                    "estimated_input_tokens": _integer(event, "estimated_input_tokens"),
                    "context_window": _integer(event, "context_window"),
                    "output_budget": _integer(event, "output_budget"),
                    # O painel mostra quantos turnos caíram, não quais: o ID
                    # sozinho não diz nada na tela e a lista já está na telemetria.
                    "dropped_turns": len(dropped) if isinstance(dropped, list) else 0,
                },
            }
        ]
    if event.kind is AgentEventKind.REASONING:
        content = _string(event, "content")
        message_id = f"{event.turn_id}-reasoning-{event.step_sequence}"
        return [
            {"type": "REASONING_START", "messageId": message_id},
            {
                "type": "REASONING_MESSAGE_START",
                "messageId": message_id,
                "role": "reasoning",
            },
            {
                "type": "REASONING_MESSAGE_CONTENT",
                "messageId": message_id,
                "delta": content,
            },
            {"type": "REASONING_MESSAGE_END", "messageId": message_id},
            {"type": "REASONING_END", "messageId": message_id},
        ]
    if event.kind is AgentEventKind.TOOL_CALL:
        tool_call_id = _string(event, "id")
        arguments = event.payload.get("arguments")
        serialized = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return [
            {
                "type": "TOOL_CALL_START",
                "toolCallId": tool_call_id,
                "toolCallName": _string(event, "name"),
            },
            {
                "type": "TOOL_CALL_ARGS",
                "toolCallId": tool_call_id,
                "delta": serialized,
            },
            {"type": "TOOL_CALL_END", "toolCallId": tool_call_id},
        ]
    if event.kind is AgentEventKind.TOOL_RESULT:
        tool_call_id = _string(event, "tool_call_id")
        return [
            {
                "type": "TOOL_CALL_RESULT",
                "messageId": f"{event.turn_id}-tool-{tool_call_id}",
                "toolCallId": tool_call_id,
                "role": "tool",
                "content": json.dumps(
                    event.payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        ]
    if event.kind is AgentEventKind.CONFIRMATION_REQUIRED:
        return [
            {
                "type": "CUSTOM",
                "name": "harness.confirmation_required",
                "value": {
                    "confirmation_id": _string(event, "confirmation_id"),
                    "reason_code": _string(event, "reason_code"),
                    "step_sequence": event.step_sequence,
                    "tool_calls": event.payload.get("tool_calls"),
                },
            }
        ]
    if event.kind is AgentEventKind.CONFIRMATION_RESOLVED:
        approved = event.payload.get("approved")
        if not isinstance(approved, bool):
            raise ValueError("agent event field must be a boolean: approved")
        return [
            {
                "type": "CUSTOM",
                "name": "harness.confirmation_resolved",
                "value": {
                    "confirmation_id": _string(event, "confirmation_id"),
                    "reason_code": _string(event, "reason_code"),
                    "approved": approved,
                },
            }
        ]
    if event.kind is AgentEventKind.FINAL_RESPONSE:
        message_id = f"{event.turn_id}-assistant"
        return [
            {
                "type": "TEXT_MESSAGE_START",
                "messageId": message_id,
                "role": "assistant",
            },
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": message_id,
                "delta": _string(event, "content"),
            },
            {"type": "TEXT_MESSAGE_END", "messageId": message_id},
        ]
    if event.kind is AgentEventKind.TURN_FINISHED:
        outcome_kind = _string(event, "outcome_kind")
        reason_code = _string(event, "reason_code")
        terminal: dict[str, JsonValue]
        if outcome_kind == "failed":
            terminal = {
                "type": "RUN_ERROR",
                "message": reason_code,
                "code": reason_code,
            }
        elif outcome_kind in _FINISHED_OUTCOMES:
            terminal = {
                "type": "RUN_FINISHED",
                "threadId": event.conversation_id,
                "runId": run_id,
            }
        else:
            raise ValueError(f"unsupported terminal outcome kind: {outcome_kind}")
        return [
            {
                "type": "CUSTOM",
                "name": "harness.turn_outcome",
                "value": {
                    "outcome_kind": outcome_kind,
                    "reason_code": reason_code,
                },
            },
            terminal,
        ]
    raise ValueError(f"unsupported agent event kind: {event.kind}")


def _number(event: AgentEvent, key: str) -> float:
    value = event.payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"agent event field must be a number: {key}")
    return float(value)


def run_started_event(*, conversation_id: str, run_id: str) -> dict[str, JsonValue]:
    return {
        "type": "RUN_STARTED",
        "threadId": conversation_id,
        "runId": run_id,
    }


def run_error_event(code: str) -> dict[str, JsonValue]:
    return {"type": "RUN_ERROR", "message": code, "code": code}


def encode_sse(event: dict[str, JsonValue]) -> str:
    return (
        "data: "
        + json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n\n"
    )


def _string(event: AgentEvent, key: str) -> str:
    value = event.payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"agent event field must be a string: {key}")
    return value


def _integer(event: AgentEvent, key: str) -> int:
    value = event.payload.get(key)
    # bool é int em Python e um booleano aqui seria um campo trocado, não um número.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"agent event field must be an integer: {key}")
    return value
