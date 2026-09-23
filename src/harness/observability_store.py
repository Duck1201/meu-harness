import asyncio
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from .domain import JsonValue, ObservabilityEvent


class ObservabilityStoreError(Exception):
    """Base error for observability repository operations."""


class ContentPayloadError(ObservabilityStoreError):
    pass


_CONTENT_FIELDS = frozenset(
    {
        "argument",
        "arguments",
        "args",
        "b64",
        "body",
        "bodies",
        "content",
        "contents",
        "data",
        "document",
        "error",
        "exception",
        "html",
        "image",
        "images",
        "input",
        "inputs",
        "message",
        "messages",
        "markdown",
        "output",
        "outputs",
        "prompt",
        "prompts",
        "query",
        "question",
        "raw",
        "reasoning",
        "response",
        "responses",
        "text",
        "thinking",
        "tool_result",
        "traceback",
    }
)
_CONTENT_SUFFIXES = tuple(f"_{field}" for field in _CONTENT_FIELDS if "_" not in field)


class ObservabilityStore:
    def __init__(self, database: str | Path) -> None:
        self._database = str(database)

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize)

    async def record(
        self,
        *,
        event_type: str,
        payload: Mapping[str, JsonValue],
        turn_id: str | None = None,
        step_sequence: int | None = None,
    ) -> ObservabilityEvent:
        return await asyncio.to_thread(self._record, event_type, payload, turn_id, step_sequence)

    async def list_events(
        self, *, after_sequence: int = 0, limit: int = 100
    ) -> list[ObservabilityEvent]:
        return await asyncio.to_thread(self._list_events, after_sequence, limit)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            applied = connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version = 1"
            ).fetchone()
            if applied is None:
                connection.execute(
                    """
                    CREATE TABLE observability_events (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        id TEXT NOT NULL UNIQUE,
                        event_type TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        turn_id TEXT,
                        step_sequence INTEGER CHECK (step_sequence IS NULL OR step_sequence > 0)
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (1, ?)",
                    (_serialize_datetime(_utcnow()),),
                )

    def _record(
        self,
        event_type: str,
        payload: Mapping[str, JsonValue],
        turn_id: str | None,
        step_sequence: int | None,
    ) -> ObservabilityEvent:
        if not event_type:
            raise ValueError("event_type must not be empty")
        if step_sequence is not None and step_sequence < 1:
            raise ValueError("step_sequence must be positive")
        _reject_content_fields(payload)
        serialized_payload = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        now = _utcnow()
        event_id = str(uuid4())
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO observability_events(
                    id, event_type, payload, occurred_at, turn_id, step_sequence
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_type,
                    serialized_payload,
                    _serialize_datetime(now),
                    turn_id,
                    step_sequence,
                ),
            )
            sequence = cursor.lastrowid
        if sequence is None:
            raise ObservabilityStoreError("SQLite did not assign an event sequence")
        return ObservabilityEvent(
            id=event_id,
            sequence=sequence,
            event_type=event_type,
            payload=_decode_payload(serialized_payload),
            occurred_at=now,
            turn_id=turn_id,
            step_sequence=step_sequence,
        )

    def _list_events(self, after_sequence: int, limit: int) -> list[ObservabilityEvent]:
        if after_sequence < 0 or limit < 1:
            raise ValueError("after_sequence must be non-negative and limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM observability_events
                WHERE sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (after_sequence, limit),
            ).fetchall()
        return [_event_from_row(row) for row in rows]


def _reject_content_fields(value: JsonValue, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalize_key(key)
            if normalized in _CONTENT_FIELDS or normalized.endswith(_CONTENT_SUFFIXES):
                location = ".".join((*path, key))
                raise ContentPayloadError(
                    f"observability payload contains content field: {location}"
                )
            _reject_content_fields(child, (*path, key))
    elif isinstance(value, Sequence) and not isinstance(value, str):
        for index, child in enumerate(value):
            _reject_content_fields(child, (*path, str(index)))


def _normalize_key(key: str) -> str:
    with_word_boundaries = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return re.sub(r"[^a-z0-9]+", "_", with_word_boundaries.lower()).strip("_")


def _event_from_row(row: sqlite3.Row) -> ObservabilityEvent:
    turn_id = row["turn_id"]
    step_sequence = row["step_sequence"]
    return ObservabilityEvent(
        id=str(row["id"]),
        sequence=int(row["sequence"]),
        event_type=str(row["event_type"]),
        payload=_decode_payload(str(row["payload"])),
        occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
        turn_id=str(turn_id) if turn_id is not None else None,
        step_sequence=int(step_sequence) if step_sequence is not None else None,
    )


def _decode_payload(payload: str) -> Mapping[str, JsonValue]:
    decoded = cast(object, json.loads(payload))
    if not isinstance(decoded, dict):
        raise ObservabilityStoreError("invalid stored observability payload")
    return cast(dict[str, JsonValue], decoded)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _serialize_datetime(value: datetime) -> str:
    return value.isoformat()
