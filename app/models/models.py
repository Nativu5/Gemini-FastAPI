from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ContentItem(BaseModel):
    """Individual content item (text, image, or file) within a message."""

    type: Literal["text", "image_url", "file", "input_audio"]
    text: str | None = None
    image_url: dict[str, str] | None = None
    input_audio: dict[str, Any] | None = None
    file: dict[str, str] | None = None
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class Message(BaseModel):
    """Message model"""

    role: str
    content: str | list[ContentItem] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    refusal: str | None = None
    reasoning_content: str | None = None
    audio: dict[str, Any] | None = None
    annotations: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_role(self) -> Message:
        """Normalize 'developer' role to 'system' for Gemini compatibility."""
        if self.role == "developer":
            self.role = "system"
        return self


class Choice(BaseModel):
    """Choice model"""

    index: int
    message: Message
    finish_reason: str
    logprobs: dict[str, Any] | None = None


class FunctionCall(BaseModel):
    """Function call payload"""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call item"""

    id: str
    type: Literal["function"]
    function: FunctionCall


class ToolFunctionDefinition(BaseModel):
    """Function definition for tool."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    """Tool specification."""

    type: Literal["function"]
    function: ToolFunctionDefinition


class ToolChoiceFunctionDetail(BaseModel):
    """Detail of a tool choice function."""

    name: str


class ToolChoiceFunction(BaseModel):
    """Tool choice forcing a specific function."""

    type: Literal["function"]
    function: ToolChoiceFunctionDetail


class Usage(BaseModel):
    """Usage statistics model"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, int] | None = None
    completion_tokens_details: dict[str, int] | None = None


class ModelData(BaseModel):
    """Model data model"""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "google"


class ChatCompletionRequest(BaseModel):
    """Chat completion request model"""

    model: str
    messages: list[Message]
    stream: bool | None = False
    user: str | None = None
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    max_tokens: int | None = None
    tools: list[Tool] | None = None
    tool_choice: (
        Literal["none"] | Literal["auto"] | Literal["required"] | ToolChoiceFunction | None
    ) = None
    response_format: dict[str, Any] | None = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response model"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class ModelListResponse(BaseModel):
    """Model list model"""

    object: str = "list"
    data: list[ModelData]


class HealthCheckResponse(BaseModel):
    """Health check response model"""

    ok: bool
    storage: dict[str, str | int] | None = None
    clients: dict[str, bool] | None = None
    error: str | None = None


class ConversationInStore(BaseModel):
    """Conversation model for storing in the database."""

    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)

    # Gemini Web API does not support changing models once a conversation is created.
    model: str = Field(..., description="Model used for the conversation")
    client_id: str = Field(..., description="Identifier of the Gemini client")
    metadata: list[str | None] = Field(
        ..., description="Metadata for Gemini API to locate the conversation"
    )
    messages: list[Message] = Field(..., description="Message contents in the conversation")


class ResponseInputContent(BaseModel):
    """Content item for Responses API input."""

    type: Literal["input_text", "input_image", "input_file"]
    text: str | None = None
    image_url: str | None = None
    detail: Literal["auto", "low", "high"] | None = None
    file_url: str | None = None
    file_data: str | None = None
    filename: str | None = None
    annotations: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_output_text(cls, data: Any) -> Any:
        """Allow output_text (from previous turns) to be treated as input_text."""
        if isinstance(data, dict) and data.get("type") == "output_text":
            data["type"] = "input_text"
        return data


class ResponseInputItem(BaseModel):
    """Single input item for Responses API."""

    type: Literal["message"] | None = "message"
    role: Literal["user", "assistant", "system", "developer"]
    content: str | list[ResponseInputContent]


class ResponseToolChoice(BaseModel):
    """Tool choice enforcing a specific tool in Responses API."""

    type: Literal["function", "image_generation"]
    function: ToolChoiceFunctionDetail | None = None


class ResponseImageTool(BaseModel):
    """Image generation tool specification for Responses API."""

    type: Literal["image_generation"]
    model: str | None = None
    output_format: str | None = None


class ResponseCreateRequest(BaseModel):
    """Responses API request payload."""

    model: str
    input: str | list[ResponseInputItem]
    instructions: str | list[ResponseInputItem] | None = None
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    max_output_tokens: int | None = None
    stream: bool | None = False
    tool_choice: str | ResponseToolChoice | None = None
    tools: list[Tool | ResponseImageTool] | None = None
    store: bool | None = None
    user: str | None = None
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ResponseUsage(BaseModel):
    """Usage statistics for Responses API."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponseOutputContent(BaseModel):
    """Content item for Responses API output."""

    type: Literal["output_text"]
    text: str | None = ""
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    """Assistant message returned by Responses API."""

    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    content: list[ResponseOutputContent]


class ResponseImageGenerationCall(BaseModel):
    """Image generation call record emitted in Responses API."""

    id: str
    type: Literal["image_generation_call"] = "image_generation_call"
    status: Literal["completed", "in_progress", "generating", "failed"] = "completed"
    result: str | None = None
    output_format: str | None = None
    size: str | None = None
    revised_prompt: str | None = None


class ResponseToolCall(BaseModel):
    """Tool call record emitted in Responses API."""

    id: str
    type: Literal["tool_call"] = "tool_call"
    status: Literal["in_progress", "completed", "failed", "requires_action"] = "completed"
    function: FunctionCall


class ResponseCreateResponse(BaseModel):
    """Responses API response payload."""

    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    output: list[ResponseOutputMessage | ResponseImageGenerationCall | ResponseToolCall]
    status: Literal[
        "in_progress",
        "completed",
        "failed",
        "incomplete",
        "cancelled",
        "requires_action",
    ] = "completed"
    tool_choice: str | ResponseToolChoice | None = None
    tools: list[Tool | ResponseImageTool] | None = None
    usage: ResponseUsage
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    input: str | list[ResponseInputItem] | None = None


# Rebuild models with forward references
Message.model_rebuild()
ToolCall.model_rebuild()
ChatCompletionRequest.model_rebuild()
