from pydantic import BaseModel, Field


class IncomingMessageRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=200)
    reply_url: str = Field(min_length=1, max_length=4000)
    message: str = Field(min_length=1, max_length=4000)


class CallbackMessagePayload(BaseModel):
    user_id: str
    thread_id: str
    status: str = "success"
    text: str | None = None
    error: str | None = None
    meta: dict[str, str] = Field(default_factory=dict)


class IncomingMessageResponse(BaseModel):
    status: str = "accepted"
    thread_id: str
