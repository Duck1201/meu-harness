import asyncio
import importlib.util
import json
from pathlib import Path
from typing import cast

from harness import load_config
from harness.domain import ToolCall
from harness.evals import build_live_model_runner
from harness.evals.traces import RecordingModelRuntime
from harness.ports import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelRuntime,
    ModelUsage,
    TokenEstimator,
    ToolSchema,
)

ROOT = Path(__file__).resolve().parents[2]


class _ScriptedRuntime:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return self.response


def _request() -> ModelRequest:
    return ModelRequest(
        messages=(
            ModelMessage(role=ModelRole.SYSTEM, content="sistema"),
            ModelMessage(role=ModelRole.USER, content='{"pedido":"liste"}'),
        ),
        tools=(
            ToolSchema(name="glob", description="acha caminhos", parameters={"type": "object"}),
        ),
        options={"temperature": 0.3},
        seed=104729,
    )


def test_the_recorder_keeps_the_exchange_and_never_the_reasoning() -> None:
    response = ModelResponse(
        content=None,
        reasoning="pensando alto sobre o pedido",
        tool_calls=(ToolCall(id="c1", name="glob", arguments={"pattern": "*.md"}),),
        usage=ModelUsage(input_tokens=120, output_tokens=9, reasoning_tokens=40),
    )
    recorder = RecordingModelRuntime(_ScriptedRuntime(response))

    returned = asyncio.run(recorder.generate(_request()))
    (exchange,) = recorder.take()
    record = exchange.to_record()

    # O runner recebe a resposta inteira; só o registro deixa o reasoning de fora.
    assert returned is response
    assert "pensando alto" not in json.dumps(record, ensure_ascii=False)
    assert record["messages"] == [
        {"role": "system", "content": "sistema"},
        {"role": "user", "content": '{"pedido":"liste"}'},
    ]
    assert record["tools"] == [
        {"name": "glob", "description": "acha caminhos", "parameters": {"type": "object"}}
    ]
    assert record["output"] == {
        "content": None,
        "tool_calls": [{"id": "c1", "name": "glob", "arguments": {"pattern": "*.md"}}],
    }
    assert record["usage"] == {"input_tokens": 120, "output_tokens": 9}


def test_take_hands_over_one_case_and_starts_the_next_empty() -> None:
    recorder = RecordingModelRuntime(_ScriptedRuntime(ModelResponse(content="ok")))

    asyncio.run(recorder.generate(_request()))
    asyncio.run(recorder.generate(_request()))

    assert len(recorder.take()) == 2
    assert recorder.take() == ()


def test_the_live_model_runner_carries_vision_judge_and_embedding() -> None:
    # O coletor montava o runner sem visão e caía na primeira fixture que a pede.
    config = load_config()
    live = build_live_model_runner(
        config,
        runtime=cast(ModelRuntime, None),
        estimator=cast(TokenEstimator, None),
        operator_notes="",
        ollama_url="http://127.0.0.1:1",
        embedding=config.runtime_profile.embedding,
    )
    try:
        assert live.embedder is not None
        assert live.runner.supports("corpus_answer")
    finally:
        asyncio.run(live.aclose())


def test_the_trace_collector_script_imports() -> None:
    # Scripts ficam fora do pyright; importar pega nome que sumiu do pacote.
    path = ROOT / "scripts/collect-model-traces.py"
    spec = importlib.util.spec_from_file_location("collect_model_traces", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.collect)
