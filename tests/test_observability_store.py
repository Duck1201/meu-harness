import asyncio
from pathlib import Path

import pytest

from harness import ContentPayloadError, ObservabilityStore


def test_observability_accepts_metadata_and_rejects_known_content_fields(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ObservabilityStore(tmp_path / "observability.sqlite3")
        await store.initialize()

        event = await store.record(
            event_type="turn.finished",
            turn_id="turn-1",
            step_sequence=2,
            payload={
                "duration_ms": 125,
                "terminal_outcome_kind": "completed",
                "reason_code": "final_response",
                "tokens": {"input_tokens": 20, "output_tokens": 8},
            },
        )

        assert await store.list_events() == [event]
        with pytest.raises(ContentPayloadError, match="prompt"):
            await store.record(event_type="model.call", payload={"prompt": "secret"})
        with pytest.raises(ContentPayloadError, match="file_content"):
            await store.record(
                event_type="tool.call",
                payload={"tool": "read_file", "meta": {"file_content": "secret"}},
            )
        with pytest.raises(ContentPayloadError, match="responseBody"):
            await store.record(
                event_type="network.call",
                payload={"http": [{"responseBody": "secret"}]},
            )
        with pytest.raises(ContentPayloadError, match="query"):
            await store.record(
                event_type="network.call",
                payload={"batch": ({"query": "secret"},)},
            )
        # describe_image: nem a imagem nem a pergunta sobre ela saem da conversa.
        for field in ("image", "images", "question", "b64"):
            with pytest.raises(ContentPayloadError, match=field):
                await store.record(event_type="tool.call", payload={"vision": {field: "secret"}})

        assert await store.list_events() == [event]

    asyncio.run(scenario())
