"""O que o modelo recebeu e emitiu em cada AgentStep, para dataset de fine-tuning.

O veredicto de uma eval diz se o caso passou; um fine-tune precisa da troca
exata com o modelo — as mensagens da ModelView, os schemas das tools e a saída
do passo. Reconstruir isso depois é aproximar: o system prompt e as tools
oferecidas mudam por braço e por fixture. O gravador fica entre o runner e o
runtime e copia cada par request/response como ele passou.

Reasoning nunca é gravado. O invariante de não persistir reasoning vale para
estado canônico, telemetria e replay; um dataset em disco é persistência do
mesmo tipo, e treinar o modelo nas próprias divagações não é o objetivo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain import JsonValue, ToolCall
from ..ports import ModelMessage, ModelRequest, ModelResponse, ModelRuntime, ToolSchema


def _call_record(call: ToolCall) -> dict[str, JsonValue]:
    return {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}


def _message_record(message: ModelMessage) -> dict[str, JsonValue]:
    record: dict[str, JsonValue] = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        record["tool_calls"] = [_call_record(call) for call in message.tool_calls]
    if message.tool_call_id is not None:
        record["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        record["name"] = message.name
    return record


def _tool_record(tool: ToolSchema) -> dict[str, JsonValue]:
    return {"name": tool.name, "description": tool.description, "parameters": dict(tool.parameters)}


@dataclass(frozen=True, slots=True)
class ModelExchange:
    """Um passo: o request inteiro e a saída do modelo, sem reasoning."""

    request: ModelRequest
    response: ModelResponse

    def to_record(self) -> dict[str, JsonValue]:
        usage = self.response.usage
        return {
            "messages": [_message_record(message) for message in self.request.messages],
            "tools": [_tool_record(tool) for tool in self.request.tools],
            "options": dict(self.request.options),
            "seed": self.request.seed,
            "max_output_tokens": self.request.max_output_tokens,
            "think": self.request.think,
            "output": {
                "content": self.response.content,
                "tool_calls": [_call_record(call) for call in self.response.tool_calls],
            },
            "usage": (
                {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
                if usage is not None
                else None
            ),
        }


@dataclass(slots=True)
class RecordingModelRuntime:
    """Implementa `ModelRuntime` repassando ao runtime real e guardando cada troca.

    O coletor roda um caso por vez e chama `take()` no fim de cada um; o que o
    caso gerou sai junto e o gravador volta vazio para o próximo.
    """

    inner: ModelRuntime
    _exchanges: list[ModelExchange] = field(default_factory=list[ModelExchange])

    async def generate(self, request: ModelRequest) -> ModelResponse:
        response = await self.inner.generate(request)
        self._exchanges.append(ModelExchange(request=request, response=response))
        return response

    def take(self) -> tuple[ModelExchange, ...]:
        taken = tuple(self._exchanges)
        self._exchanges.clear()
        return taken
