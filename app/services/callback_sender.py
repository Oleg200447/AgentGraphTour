from __future__ import annotations

import httpx


class CallbackSender:
    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def send(self, reply_url: str, payload: dict) -> None:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(reply_url, json=payload)
            response.raise_for_status()
