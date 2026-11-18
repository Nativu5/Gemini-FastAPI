from __future__ import annotations

import base64
import hashlib
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
    content_digest: str | None = None

    @model_validator(mode="after")
    def populate_content_digest(self) -> AppContentItem:
        """Persist a digest for inline media whose raw bytes are intentionally excluded.

        Only `file_data` and inline data URLs need one. Every other field survives
        serialization, so it can be compared directly instead of through a fingerprint.
        """
        if self.content_digest:
            return self

        encoded = self.file_data
        if encoded is None and self.url and self.url.startswith("data:"):
            encoded = self.url.partition(",")[2]
        if encoded is None:
            return self

        # utf-8, not ascii: a data URL may carry non-Base64 text, and a codec error here
        # would abort model construction.
        raw = encoded.encode("utf-8", "surrogatepass") if isinstance(encoded, str) else encoded
        if raw.startswith(b"data:"):
            raw = raw.partition(b",")[2]
        try:
            digest_input = base64.b64decode(b"".join(raw.split()), validate=True)
        except ValueError:
            digest_input = raw

        self.content_digest = hashlib.sha256(digest_input).hexdigest()
        return self


type AppMessageRole = Literal["system", "user", "assistant", "tool"]


class AppMessage(BaseModel):
    role: AppMessageRole
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
    chat_scope: str | None = Field(
        default=None,
        description=(
            "Identity of the ephemeral window this chat lives in, or None for a normal chat kept "
            "in the account's history. Reusable only while the client still reports this scope"
        ),
    )
