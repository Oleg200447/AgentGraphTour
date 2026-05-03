from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

import asyncpg
from pydantic import BaseModel

from app.db.postgres import validate_sql_identifier

logger = logging.getLogger(__name__)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _to_jsonable(value.model_dump(mode="json", by_alias=True))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return str(value)


class ThreadStateRepository:
    def __init__(self, pool: asyncpg.Pool, table_name: str = "agent_thread_state") -> None:
        self.pool = pool
        self.table_name = validate_sql_identifier(table_name)

    async def init_table(self) -> None:
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            thread_id TEXT PRIMARY KEY,
            state_json JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query)
        logger.info("Thread state table initialized", extra={"table": self.table_name})

    async def load_thread_state(self, thread_id: str) -> dict[str, Any] | None:
        query = f"SELECT state_json FROM {self.table_name} WHERE thread_id = $1 LIMIT 1"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, thread_id)
        if row is None:
            logger.debug("Thread state not found", extra={"thread_id": thread_id, "table": self.table_name})
            return None

        state_json = row["state_json"]
        if isinstance(state_json, dict):
            logger.debug("Thread state loaded", extra={"thread_id": thread_id, "table": self.table_name})
            return dict(state_json)
        if isinstance(state_json, str):
            try:
                loaded = json.loads(state_json)
                return loaded if isinstance(loaded, dict) else None
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode thread state JSON",
                    extra={"thread_id": thread_id, "table": self.table_name},
                )
                return None

        logger.warning(
            "Thread state has unsupported type",
            extra={"thread_id": thread_id, "table": self.table_name, "state_type": type(state_json).__name__},
        )
        return None

    async def save_thread_state(self, thread_id: str, state: dict[str, Any]) -> None:
        payload = _to_jsonable(state)
        payload_json = json.dumps(payload, ensure_ascii=False)
        query = f"""
        INSERT INTO {self.table_name} (thread_id, state_json, updated_at)
        VALUES ($1, $2::jsonb, NOW())
        ON CONFLICT (thread_id)
        DO UPDATE SET state_json = EXCLUDED.state_json, updated_at = NOW()
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query, thread_id, payload_json)
        logger.debug(
            "Thread state saved",
            extra={"thread_id": thread_id, "table": self.table_name, "payload_size": len(payload_json)},
        )
