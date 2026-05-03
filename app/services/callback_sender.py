from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class CallbackSender:
    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def send(self, reply_url: str, payload: dict) -> None:
        target_host = httpx.URL(reply_url).host if reply_url else None
        logger.debug("Sending callback", extra={"target_host": target_host})

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            try:
                response = await client.post(reply_url, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Callback returned non-success status",
                    extra={
                        "target_host": target_host,
                        "status_code": exc.response.status_code,
                    },
                    exc_info=True,
                )
                raise
            except Exception:
                logger.exception(
                    "Callback request failed",
                    extra={"target_host": target_host},
                )
                raise

        logger.info("Callback delivered", extra={"target_host": target_host})
