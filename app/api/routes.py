import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.schemas.api import IncomingMessageRequest, IncomingMessageResponse
from app.services.agent_runtime import AgentRuntime

router = APIRouter(prefix="/api", tags=["agent"])
logger = logging.getLogger(__name__)


@router.post("/messages/incoming", response_model=IncomingMessageResponse)
async def incoming_message(
    payload: IncomingMessageRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> IncomingMessageResponse:
    logger.debug(
        "Incoming message received",
        extra={"user_id": payload.user_id, "message_length": len(payload.message)},
    )

    runtime: AgentRuntime | None = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        logger.warning("Agent runtime is not initialized")
        raise HTTPException(status_code=500, detail="Agent runtime is not initialized")

    background_tasks.add_task(runtime.handle_message_background, payload)
    thread_id = payload.user_id
    logger.info("Incoming message accepted", extra={"thread_id": thread_id})
    return IncomingMessageResponse(thread_id=thread_id)


@router.post("/debug/callback")
async def debug_callback(payload: dict) -> dict:
    """Local test endpoint to receive callbacks from the agent itself."""
    logger.debug("Debug callback payload received", extra={"keys": list(payload.keys())})
    return {"ok": True, "payload": payload}
