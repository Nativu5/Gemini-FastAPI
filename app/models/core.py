from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AppToolCallFunction(BaseModel):
    name: str
    arguments: str


class AppToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: AppToolCallFunction


class AppContentItem(BaseModel):
    type: str
    text: str | None = None
    url: str | None = None
    file_data: str | bytes | None = Field(default=None, exclude=True)
    filename: str | None = None
    raw_data: dict[str, Any] | None = None


class AppMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    name: str | None = None
    content: str | list[AppContentItem] | None = None
    tool_calls: list[AppToolCall] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None


class ConversationInStore(BaseModel):
    """Persisted conversation record stored in LMDB."""

    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)
    model: str = Field(..., description="Model used for the conversation")
    client_id: str = Field(..., description="Identifier of the Gemini client")
    metadata: list[str | None] = Field(
        ..., description="Metadata for Gemini API to locate the conversation"
    )
    messages: list[AppMessage] = Field(
        ..., description="Canonical message contents in the conversation"
    )
