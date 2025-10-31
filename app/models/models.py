from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ContentItem(BaseModel):
    """Content item model"""

    type: Literal["text", "image_url", "file", "input_audio"]
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None
    file: Optional[Dict[str, str]] = None


class Message(BaseModel):
    """Message model"""

    role: str
    content: Union[str, List[ContentItem], None] = None
    name: Optional[str] = None
    tool_calls: Optional[List["ToolCall"]] = None


class Choice(BaseModel):
    """Choice model"""

    index: int
    message: Message
    finish_reason: str


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
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


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


class ModelData(BaseModel):
    """Model data model"""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "google"


class ChatCompletionRequest(BaseModel):
    """Chat completion request model"""

    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    user: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    tools: Optional[List["Tool"]] = None
    tool_choice: Optional[
        Union[Literal["none"], Literal["auto"], Literal["required"], "ToolChoiceFunction"]
    ] = None
    response_format: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response model"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ModelListResponse(BaseModel):
    """Model list model"""

    object: str = "list"
    data: List[ModelData]


class HealthCheckResponse(BaseModel):
    """Health check response model"""

    ok: bool
    storage: Optional[Dict[str, str | int]] = None
    clients: Optional[Dict[str, bool]] = None
    error: Optional[str] = None


class ConversationInStore(BaseModel):
    """Conversation model for storing in the database."""

    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)

    # NOTE: Gemini Web API do not support changing models once a conversation is created.
    model: str = Field(..., description="Model used for the conversation")
    client_id: str = Field(..., description="Identifier of the Gemini client")
    metadata: list[str | None] = Field(
        ..., description="Metadata for Gemini API to locate the conversation"
    )
    messages: list[Message] = Field(..., description="Message contents in the conversation")


class ResponseInputContent(BaseModel):
    """Content item for Responses API input."""

    type: Literal["input_text", "input_image"]
    text: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    mime_type: Optional[str] = None


class ResponseInputItem(BaseModel):
    """Single input item for Responses API."""

    type: Optional[Literal["message"]] = "message"
    role: Literal["user", "assistant", "system", "developer"]
    content: Union[str, List[ResponseInputContent]]


class ResponseToolChoice(BaseModel):
    """Tool choice enforcing a specific tool in Responses API."""

    type: Literal["image_generation"]


class ResponseImageTool(BaseModel):
    """Image generation tool specification for Responses API."""

    type: Literal["image_generation"]
    model: Optional[str] = None
    output_format: Optional[str] = None


class ResponseCreateRequest(BaseModel):
    """Responses API request payload."""

    model: str
    input: Union[str, List[ResponseInputItem]]
    instructions: Optional[Union[str, List[ResponseInputItem]]] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_output_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tool_choice: Optional[ResponseToolChoice] = None
    tools: Optional[List[ResponseImageTool]] = None
    store: Optional[bool] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ResponseUsage(BaseModel):
    """Usage statistics for Responses API."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponseOutputContent(BaseModel):
    """Content item for Responses API output."""

    type: Literal["output_text", "output_image"]
    text: Optional[str] = None
    image_base64: Optional[str] = None
    mime_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class ResponseOutputMessage(BaseModel):
    """Assistant message returned by Responses API."""

    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    content: List[ResponseOutputContent]


class ResponseImageGenerationCall(BaseModel):
    """Image generation call record emitted in Responses API."""

    id: str
    type: Literal["image_generation_call"] = "image_generation_call"
    status: Literal["completed", "in_progress", "generating", "failed"] = "completed"
    result: Optional[str] = None
    output_format: Optional[str] = None
    size: Optional[str] = None
    revised_prompt: Optional[str] = None


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
    created: int
    model: str
    output: List[Union[ResponseOutputMessage, ResponseImageGenerationCall, ResponseToolCall]]
    output_text: Optional[str] = None
    status: Literal[
        "in_progress",
        "completed",
        "failed",
        "incomplete",
        "requires_action",
    ] = "completed"
    usage: ResponseUsage
    metadata: Optional[Dict[str, Any]] = None
    system_fingerprint: Optional[str] = None
    input: Optional[Union[str, List[ResponseInputItem]]] = None


# Rebuild models with forward references
Message.model_rebuild()
ToolCall.model_rebuild()
ChatCompletionRequest.model_rebuild()
