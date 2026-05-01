from fastapi import APIRouter, HTTPException, Request

from app.schemas.api import IncomingMessageRequest, IncomingMessageResponse
from app.services.agent_runtime import AgentRuntime

router = APIRouter(prefix="/api", tags=["agent"])


@router.post("/messages/incoming", response_model=IncomingMessageResponse)
async def incoming_message(payload: IncomingMessageRequest, request: Request) -> IncomingMessageResponse:
    runtime: AgentRuntime | None = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=500, detail="Agent runtime is not initialized")

    result = await runtime.handle_message(payload)
    return IncomingMessageResponse(thread_id=result.thread_id, response_text=result.response_text)


@router.post("/debug/callback")
async def debug_callback(payload: dict) -> dict:
    """Local test endpoint to receive callbacks from the agent itself."""
    return {"ok": True, "payload": payload}
