import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast
from xml.sax.saxutils import escape, quoteattr

from .domain import (
    CanonicalHistoryEntry,
    CanonicalHistoryEntryKind,
    JsonValue,
    ToolCall,
)
from .ports import EngineReadiness, ModelMessage, ModelRole, TokenEstimator, ToolSchema

_XML_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")

# Os elementos que envelopam uma entrada da ModelView. Ficam declarados aqui, e
# não repetidos em quem precisa deles, porque o modelo aprende esse formato lendo
# o próprio contexto e às vezes devolve o envelope como se fosse resposta: quem
# detecta isso tem de conhecer exatamente as raízes que este módulo emite, e uma
# segunda lista envelheceria calada. `tests/test_context_builder.py` falha se o
# builder passar a emitir uma raiz que não esteja aqui.
MODEL_VIEW_ROOTS = (
    "model_attempt",
    "rejected_model_attempt",
    "tool_result",
    "internal_automation",
)

type ModelViewFormat = Literal["xml", "json"]
type PayloadRenderer = Callable[[str, JsonValue], str]


class ContextBuilderError(Exception):
    pass


class ContextBudgetExceeded(ContextBuilderError):
    pass


class TokenEstimatorNotValidated(ContextBuilderError):
    pass


@dataclass(frozen=True, slots=True)
class ContextTurn:
    turn_id: str
    entries: tuple[CanonicalHistoryEntry, ...]

    def __post_init__(self) -> None:
        if any(entry.turn_id != self.turn_id for entry in self.entries):
            raise ValueError("context turn contains an entry from another turn")


@dataclass(frozen=True, slots=True)
class ModelContext:
    messages: tuple[ModelMessage, ...]
    tool_schemas: tuple[ToolSchema, ...]
    estimated_input_tokens: int
    output_budget: int
    # O denominador viaja junto com o numerador: quem mostra ou grava o consumo
    # da janela não precisa buscar a configuração por um caminho paralelo.
    context_window: int
    dropped_turn_ids: tuple[str, ...]
    taints: frozenset[str]


type AutomationRole = Literal["tool", "user"]


class ContextBuilder:
    def __init__(
        self,
        estimator: TokenEstimator,
        *,
        context_window: int,
        output_budget: int = 8192,
        model_view_format: ModelViewFormat = "json",
        automation_role: AutomationRole = "tool",
    ) -> None:
        if context_window <= output_budget:
            raise ValueError("context window must exceed output budget")
        self._estimator = estimator
        self._context_window = context_window
        self._output_budget = output_budget
        # JSON é o formato do harness. O XML entrou por ADR-0010 sem medição e
        # saiu com ela: na promoção de 15/08/2026 perdeu 36/50 contra 41/50 em
        # verdict PASS e subiu rejected_model_attempts de 22 para 30, e o gate
        # declarado do experimento manda reverter quando perde em qualquer um dos
        # dois. O render XML continua construível porque agora é ele o braço
        # candidato: um braço que não pode ser executado não é braço, é lembrança.
        self._render_payload = _xml_document if model_view_format == "xml" else _json_document
        # Em que papel uma automação interna — a recuperação do Corpus, a decisão
        # do Operator — chega ao modelo. É do template, não do harness: o do Gemma4
        # descarta mensagem `tool` que não responde a uma chamada dele, e com a
        # passagem injetada assim ele respondia "forneça o manual" com o manual no
        # contexto (medido, 2026-09-23). Como `user`, o mesmo modelo cita a passagem.
        self._automation_role = automation_role

    @property
    def readiness(self) -> EngineReadiness:
        if not self._estimator.validated:
            return EngineReadiness(ready=False, reason_code="token_estimator_not_validated")
        return EngineReadiness(ready=True)

    def build(
        self,
        *,
        system: str,
        tool_schemas: Sequence[ToolSchema],
        completed_turns: Sequence[ContextTurn],
        current_turn: ContextTurn,
    ) -> ModelContext:
        if not self._estimator.validated:
            raise TokenEstimatorNotValidated("token estimator has not been validated")
        if any(turn.turn_id == current_turn.turn_id for turn in completed_turns):
            raise ValueError("current turn must not also be a completed turn")

        schemas = tuple(tool_schemas)
        remaining = list(completed_turns)
        dropped: list[str] = []
        while True:
            messages = self._render(system, (*remaining, current_turn))
            estimated = self._estimator.estimate(messages, schemas)
            if estimated + self._output_budget <= self._context_window:
                included_turns = (*remaining, current_turn)
                return ModelContext(
                    messages=messages,
                    tool_schemas=schemas,
                    estimated_input_tokens=estimated,
                    output_budget=self._output_budget,
                    context_window=self._context_window,
                    dropped_turn_ids=tuple(dropped),
                    taints=_tool_result_taints(included_turns),
                )
            if not remaining:
                raise ContextBudgetExceeded("system, tool schemas, and current turn exceed budget")
            dropped.append(remaining.pop(0).turn_id)

    def _render(self, system: str, turns: Sequence[ContextTurn]) -> tuple[ModelMessage, ...]:
        messages = [ModelMessage(role=ModelRole.SYSTEM, content=system)]
        seen_payloads: dict[str, CanonicalHistoryEntry] = {}
        current = len(turns) - 1
        for index, turn in enumerate(turns):
            for entry in turn.entries:
                if (
                    entry.kind is CanonicalHistoryEntryKind.REJECTED_MODEL_ATTEMPT
                    and index != current
                ):
                    # Uma recusa de turno encerrado não ensina nada ao próximo passo e,
                    # repetida no contexto, vira exemplo de resposta que o modelo copia.
                    continue
                message = _entry_message(entry, seen_payloads, self._render_payload)
                if (
                    entry.kind is CanonicalHistoryEntryKind.INTERNAL_AUTOMATION
                    and self._automation_role == "user"
                ):
                    message = ModelMessage(
                        role=ModelRole.USER, content=f"[{message.name}] {message.content}"
                    )
                messages.append(message)
        return tuple(messages)


def _entry_message(
    entry: CanonicalHistoryEntry,
    seen_payloads: dict[str, CanonicalHistoryEntry],
    render: PayloadRenderer,
) -> ModelMessage:
    payload = entry.payload
    if entry.kind is CanonicalHistoryEntryKind.USER_MESSAGE:
        return ModelMessage(role=ModelRole.USER, content=_required_string(payload, "content"))
    if entry.kind is CanonicalHistoryEntryKind.REJECTED_MODEL_ATTEMPT:
        # Vai como aviso do harness, não como fala do modelo: renderizada como
        # assistant, a recusa (com o corpo cru do Ollama) era a resposta anterior
        # "dele", e o Gemma passava a responder no mesmo JSON.
        return ModelMessage(role=ModelRole.USER, content=_rejection_notice(payload))
    if entry.kind is CanonicalHistoryEntryKind.MODEL_ATTEMPT:
        content = payload.get("content")
        # Um passo só de tool calls tem conteúdo vazio: as calls seguem no campo
        # nativo. Serializar o payload aqui ensinava o modelo a escrever tool
        # calls como texto na resposta final.
        return ModelMessage(
            role=ModelRole.ASSISTANT,
            content=content if isinstance(content, str) else "",
            tool_calls=_tool_calls(payload.get("tool_calls")),
        )
    if entry.kind is CanonicalHistoryEntryKind.TOOL_RESULT:
        rendered = _model_facing_result(payload)
        data = payload.get("data")
        if data is not None:
            digest = hashlib.sha256(render("data", data).encode()).hexdigest()
            original = seen_payloads.get(digest)
            if original is None:
                seen_payloads[digest] = entry
            else:
                rendered["data"] = {"$ref": {"entry_id": original.id, "sha256": digest}}
        return ModelMessage(
            role=ModelRole.TOOL,
            content=render("tool_result", rendered),
            tool_call_id=_optional_string(payload, "tool_call_id"),
            name=_optional_string(payload, "tool_name"),
        )
    if entry.kind is CanonicalHistoryEntryKind.FINAL_RESPONSE:
        return ModelMessage(role=ModelRole.ASSISTANT, content=_required_string(payload, "content"))
    if entry.kind is CanonicalHistoryEntryKind.INTERNAL_AUTOMATION:
        return ModelMessage(
            role=ModelRole.TOOL,
            content=render("internal_automation", payload),
            name=_optional_string(payload, "automation_id"),
        )
    raise ContextBuilderError(f"unsupported canonical history entry: {entry.kind}")


def _rejection_notice(payload: Mapping[str, JsonValue]) -> str:
    error = payload.get("error")
    code = error.get("code") if isinstance(error, Mapping) else None
    message = error.get("message") if isinstance(error, Mapping) else None
    details = [item for item in (code, message) if isinstance(item, str) and item]
    reason = f" ({': '.join(details)})" if details else ""
    return (
        f"[harness] Your previous reply could not be used{reason}. Reply again: either call "
        "a tool through the tool-calling interface, or write your answer for the Operator as "
        "plain text."
    )


def _model_facing_result(payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """O ToolResult como o modelo precisa dele; o canônico continua inteiro.

    `tool_call_id` e `tool_name` já vão nos campos nativos da mensagem, e
    produtor, engine, latência e listas vazias não mudam decisão nenhuma — eram
    metade dos caracteres de cada resultado. Fica o que muda o próximo passo:
    status, dado ou erro, se vale tentar de novo, corte e procedência não confiável.
    """
    status = payload.get("status")
    rendered: dict[str, JsonValue] = {"status": status}
    if payload.get("data") is not None:
        rendered["data"] = payload.get("data")
    if payload.get("error") is not None:
        rendered["error"] = payload.get("error")
    if status in {"failed", "blocked"}:
        rendered["retryable"] = payload.get("retryable")
    meta = payload.get("meta")
    if isinstance(meta, Mapping):
        if meta.get("truncated") is True:
            rendered["truncated"] = True
        taints = meta.get("taints")
        if isinstance(taints, Sequence) and not isinstance(taints, str) and taints:
            rendered["taints"] = taints
    return rendered


def _tool_calls(value: JsonValue) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ContextBuilderError("model attempt tool_calls must be an array")
    calls: list[ToolCall] = []
    for raw_item in value:
        if not isinstance(raw_item, Mapping):
            raise ContextBuilderError("model attempt tool call must be an object")
        item = cast(Mapping[str, JsonValue], raw_item)
        arguments = item.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise ContextBuilderError("model attempt tool arguments must be an object")
        calls.append(
            ToolCall(
                id=_required_string(item, "id"),
                name=_required_string(item, "name"),
                arguments=cast(Mapping[str, JsonValue], arguments),
                idempotency_key=_optional_string(item, "idempotency_key"),
            )
        )
    return tuple(calls)


def _tool_result_taints(turns: Sequence[ContextTurn]) -> frozenset[str]:
    taints: set[str] = set()
    for turn in turns:
        for entry in turn.entries:
            if entry.kind is not CanonicalHistoryEntryKind.TOOL_RESULT:
                continue
            meta = entry.payload.get("meta")
            if not isinstance(meta, Mapping):
                continue
            values = meta.get("taints")
            if not isinstance(values, Sequence) or isinstance(values, str):
                continue
            taints.update(value for value in values if isinstance(value, str))
    return frozenset(taints)


def _required_string(payload: Mapping[str, JsonValue], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ContextBuilderError(f"canonical history field must be a string: {key}")
    return value


def _optional_string(payload: Mapping[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise ContextBuilderError(f"canonical history field must be a string or null: {key}")
    return value


def _xml_document(root: str, value: JsonValue) -> str:
    return f"<{root}>{_xml_text(value)}</{root}>"


# Conteúdo externo não pode abrir nem fechar turno, tool call ou raciocínio no
# template do runtime: o Ollama tokeniza marcador especial dentro do texto. O
# escape antigo trocava todo `<`, `>` e `&`, e o modelo passou a ler `<title>`
# como `\\u003ctitle\\u003e` e a gravar esses escapes em arquivo. Agora só o
# `<` que começa um marcador: os do Gemma 4 (`<|turn>`, `<turn|>`, `<|"|>`,
# `<eos>`...) e os do Qwen, que motivaram a regra (`<tool_call>`, `</think>`,
# `<|im_start|>`). HTML e código chegam ao modelo como estão no arquivo.
_TEMPLATE_MARKER = re.compile(
    r"<(?=\|)"
    r"|<(?=[A-Za-z_\"]*\|>)"
    r"|<(?=/?(?:tool_call|tool_response|tool|think|im_start|im_end|start_of_turn|end_of_turn"
    r"|pad|eos|bos|unk|mask)>)"
)


def _json_document(root: str, value: JsonValue) -> str:
    """O render anterior ao ADR-0010, mantido como braço de controle."""
    del root
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _TEMPLATE_MARKER.sub("\\\\u003c", serialized)


def _xml_text(value: JsonValue) -> str:
    parts: list[str] = []
    _write_xml(value, parts)
    return "".join(parts)


def _write_xml(value: JsonValue, parts: list[str]) -> None:
    if value is None:
        parts.append("<null/>")
        return
    if isinstance(value, str):
        parts.append(escape(value))
        return
    if isinstance(value, bool | int | float):
        # json.dumps keeps the canonical number/bool spelling and still rejects NaN and Infinity.
        parts.append(json.dumps(value, allow_nan=False))
        return
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, JsonValue], value)
        for key in sorted(mapping):
            _write_element(key, mapping[key], parts)
        return
    for item in value:
        _write_element("item", item, parts)


def _write_element(name: str, value: JsonValue, parts: list[str]) -> None:
    if _XML_NAME.fullmatch(name):
        parts.append(f"<{name}>")
        _write_xml(value, parts)
        parts.append(f"</{name}>")
        return
    # Keys the model may produce — "$ref", "1", "a b" — are not valid XML names.
    parts.append(f"<entry key={quoteattr(name)}>")
    _write_xml(value, parts)
    parts.append("</entry>")
