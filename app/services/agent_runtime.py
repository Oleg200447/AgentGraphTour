from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from app.schemas.api import CallbackMessagePayload, IncomingMessageRequest
from app.services.callback_sender import CallbackSender
from app.services.thread_state_repository import ThreadStateRepository

logger = logging.getLogger(__name__)


@dataclass
class AgentRuntimeResult:
    thread_id: str
    response_text: str


class AgentRuntime:
    def __init__(
        self,
        main_graph: Any,
        thread_repo: ThreadStateRepository,
        callback_sender: CallbackSender,
        debug_skip_callback: bool = False,
    ) -> None:
        self.main_graph = main_graph
        self.thread_repo = thread_repo
        self.callback_sender = callback_sender
        self.debug_skip_callback = debug_skip_callback

    @staticmethod
    def _trim_history_to_last_user_messages(
        messages: list[dict[str, str]],
        max_user_messages: int = 10,
    ) -> list[dict[str, str]]:
        user_indices = [idx for idx, msg in enumerate(messages) if msg.get("role") == "user"]
        if len(user_indices) <= max_user_messages:
            return messages

        first_kept_user_index = user_indices[-max_user_messages]
        return messages[first_kept_user_index:]

    async def handle_message(self, payload: IncomingMessageRequest) -> AgentRuntimeResult:
        thread_id = payload.user_id
        logger.info("Handling incoming message", extra={"thread_id": thread_id, "user_id": payload.user_id})

        previous_state = await self.thread_repo.load_thread_state(thread_id) or {}
        logger.debug(
            "Loaded previous thread state",
            extra={"thread_id": thread_id, "has_previous_state": bool(previous_state)},
        )

        messages: list[dict[str, str]] = list(previous_state.get("messages", []))
        messages.append({"role": "user", "content": payload.message})
        messages = self._trim_history_to_last_user_messages(messages)

        input_state = {
            **previous_state,
            "user_id": payload.user_id,
            "thread_id": thread_id,
            "reply_url": payload.reply_url,
            "latest_user_text": payload.message,
            "messages": messages,
        }

        result = await self.main_graph.ainvoke(input_state)
        if not isinstance(result, dict):
            logger.warning("Graph returned non-dict result, substituting empty state", extra={"thread_id": thread_id})
            result = {}

        response_text = str(result.get("assistant_response_text") or "Не удалось сформировать ответ.")
        messages.append({"role": "assistant", "content": response_text})
        messages = self._trim_history_to_last_user_messages(messages)

        result["messages"] = messages
        result["thread_id"] = thread_id
        result["user_id"] = payload.user_id
        result["reply_url"] = payload.reply_url

        await self.thread_repo.save_thread_state(thread_id, result)
        logger.info("Thread state persisted", extra={"thread_id": thread_id, "messages_count": len(messages)})

        callback_payload = CallbackMessagePayload(
            user_id=payload.user_id,
            thread_id=thread_id,
            text=response_text,
            meta={"agent": str(result.get("response_agent") or "offer_agent")},
        )
        if not self.debug_skip_callback:
            await self.callback_sender.send(payload.reply_url, callback_payload.model_dump(mode="json"))
            logger.info("Callback sent", extra={"thread_id": thread_id})
        else:
            logger.debug("Callback skipped due to debug flag", extra={"thread_id": thread_id})

        return AgentRuntimeResult(thread_id=thread_id, response_text=response_text)

    async def handle_message_background(self, payload: IncomingMessageRequest) -> None:
        thread_id = payload.user_id
        try:
            await self.handle_message(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Background message processing failed",
                extra={"thread_id": thread_id, "user_id": payload.user_id},
            )

            error_text = str(exc).strip() or exc.__class__.__name__
            callback_payload = CallbackMessagePayload(
                user_id=payload.user_id,
                thread_id=thread_id,
                status="error",
                error=error_text,
                meta={"agent": "runtime"},
            )

            if self.debug_skip_callback:
                logger.debug("Callback skipped due to debug flag", extra={"thread_id": thread_id})
                return

            try:
                await self.callback_sender.send(payload.reply_url, callback_payload.model_dump(mode="json"))
                logger.info("Error callback sent", extra={"thread_id": thread_id})
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to send error callback",
                    extra={"thread_id": thread_id, "user_id": payload.user_id},
                )
