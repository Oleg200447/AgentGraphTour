import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas.api import IncomingMessageRequest, IncomingMessageResponse
from app.services.agent_runtime import AgentRuntime

router = APIRouter(prefix="/api", tags=["agent"])
logger = logging.getLogger(__name__)


@router.post("/messages/incoming", response_model=IncomingMessageResponse)
async def incoming_message(payload: IncomingMessageRequest, request: Request) -> IncomingMessageResponse:
    logger.debug(
        "Incoming message received",
        extra={"user_id": payload.user_id, "message_length": len(payload.message)},
    )

    runtime: AgentRuntime | None = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        logger.warning("Agent runtime is not initialized")
        raise HTTPException(status_code=500, detail="Agent runtime is not initialized")

    result = await runtime.handle_message(payload)
    logger.info("Incoming message processed", extra={"thread_id": result.thread_id})
    return IncomingMessageResponse(thread_id=result.thread_id, response_text=result.response_text)


@router.post("/debug/callback")
async def debug_callback(payload: dict) -> dict:
    """Local test endpoint to receive callbacks from the agent itself."""
    logger.debug("Debug callback payload received", extra={"keys": list(payload.keys())})
    return {"ok": True, "payload": payload}
