from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class FunctionCall(BaseModel):
    """Executed function call payload."""

    name: str
    arguments: str


class FunctionDefinition(BaseModel):
    """Schema of a callable function exposed to the model."""

    name: str
    description: str | None = Field(default=None)
    parameters: dict[str, Any] | None = Field(default=None)


class ChatCompletionRequestContentItem(BaseModel):
    """Content item for user / system / tool messages."""

    type: Literal["text", "image_url", "file", "input_audio"]
    text: str | None = Field(default=None)
    image_url: dict[str, Any] | None = Field(default=None)
    input_audio: dict[str, Any] | None = Field(default=None)
    file: dict[str, Any] | None = Field(default=None)


class ChatCompletionAssistantContentItem(BaseModel):
    """Content item for assistant messages.

    ``refusal`` is an official OpenAI content part.
    ``reasoning`` is a community extension used to persist chain-of-thought
    text for reusable-session matching.
    """

    type: Literal["text", "refusal", "reasoning"]
    text: str | None = Field(default=None)
    refusal: str | None = Field(default=None)
    annotations: list[dict[str, Any]] = Field(default_factory=list)


ChatCompletionContentItem = ChatCompletionRequestContentItem | ChatCompletionAssistantContentItem


class ChatCompletionMessageToolCall(BaseModel):
    """A single tool call emitted by the assistant."""

    id: str
    type: Literal["function"]
    function: FunctionCall


class ChatCompletionMessage(BaseModel):
    """A single message in a Chat Completions conversation."""

    role: Literal["developer", "system", "user", "assistant", "tool", "function"]
    content: (
        str | list[ChatCompletionRequestContentItem | ChatCompletionAssistantContentItem] | None
    ) = Field(default=None)
    name: str | None = Field(default=None)
    tool_calls: list[ChatCompletionMessageToolCall] | None = Field(default=None)
    tool_call_id: str | None = Field(default=None)
    refusal: str | None = Field(default=None)
    reasoning_content: str | None = Field(default=None)
    audio: dict[str, Any] | None = Field(default=None)
    annotations: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_role(self) -> ChatCompletionMessage:
        """Normalize ``developer`` role to ``system`` for Gemini compatibility."""
        if self.role == "developer":
            self.role = "system"
        return self


class ChatCompletionFunctionTool(BaseModel):
    """A function tool for the Chat Completions API."""

    type: Literal["function"]
    function: FunctionDefinition


class ChatCompletionNamedToolChoiceFunction(BaseModel):
    name: str


class ChatCompletionNamedToolChoice(BaseModel):
    """Forces the model to call a specific named function."""

    type: Literal["function"]
    function: ChatCompletionNamedToolChoiceFunction


class CompletionUsage(BaseModel):
    """Token-usage statistics for a Chat Completions response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, int] | None = Field(default=None)
    completion_tokens_details: dict[str, int] | None = Field(default=None)


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"]
    logprobs: dict[str, Any] | None = Field(default=None)


class ChatCompletionRequest(BaseModel):
    """Request body for POST /v1/chat/completions."""

    model: str
    messages: list[ChatCompletionMessage]
    stream: bool | None = Field(default=False)
    stream_options: dict[str, Any] | None = Field(default=None)
    prompt_cache_key: str | None = Field(default=None)
    temperature: float | None = Field(default=1, ge=0, le=2)
    top_p: float | None = Field(default=1, ge=0, le=1)
    max_completion_tokens: int | None = Field(default=None)
    tools: list[ChatCompletionFunctionTool] | None = Field(default=None)
    tool_choice: Literal["none", "auto", "required"] | ChatCompletionNamedToolChoice | None = Field(
        default=None
    )
    response_format: dict[str, Any] | None = Field(default=None)
    parallel_tool_calls: bool | None = Field(default=True)


class ChatCompletionResponse(BaseModel):
    """Response body for POST /v1/chat/completions."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage
    system_fingerprint: str | None = Field(default=None)


class ResponseInputText(BaseModel):
    """Text content item in a Responses API input message."""

    type: Literal["input_text"]
    text: str | None = Field(default=None)


class ResponseInputImage(BaseModel):
    """Image content item in a Responses API input message."""

    type: Literal["input_image"]
    detail: Literal["auto", "low", "high"] | None = Field(default=None)
    file_id: str | None = Field(default=None)
    image_url: str | None = Field(default=None)


class ResponseInputFile(BaseModel):
    """File content item in a Responses API input message."""

    type: Literal["input_file"]
    file_id: str | None = Field(default=None)
    file_url: str | None = Field(default=None)
    file_data: str | None = Field(default=None)
    filename: str | None = Field(default=None)


class ResponseInputMessageContentList(BaseModel):
    """Normalised content item stored on ``ResponseInputMessage`` server-side.

    Superset of all input content types (text, image, file, reasoning) so they
    can be represented in a single model after round-tripping through the server.
    """

    type: Literal["input_text", "output_text", "reasoning_text", "input_image", "input_file"]
    text: str | None = Field(default=None)
    image_url: str | None = Field(default=None)
    detail: Literal["auto", "low", "high"] | None = Field(default=None)
    file_id: str | None = Field(default=None)
    file_url: str | None = Field(default=None)
    file_data: str | None = Field(default=None)
    filename: str | None = Field(default=None)


class ResponseInputMessage(BaseModel):
    """A single conversation turn in a Responses API input list."""

    type: Literal["message"] | None = Field(default="message")
    role: Literal["user", "system", "developer", "assistant"]
    content: str | list[ResponseInputText | ResponseInputImage | ResponseInputFile]
    status: Literal["in_progress", "completed", "incomplete"] = Field(default="completed")


class ResponseFunctionToolCall(BaseModel):
    """An assistant function-call item replayed as part of the input history."""

    type: Literal["function_call"] | None = Field(default="function_call")
    id: str | None = Field(default=None)
    call_id: str
    name: str
    arguments: str
    status: Literal["in_progress", "completed", "incomplete"] = Field(default="completed")


class FunctionCallOutput(BaseModel):
    """A tool-result item providing function output back to the model."""

    type: Literal["function_call_output"] | None = Field(default="function_call_output")
    id: str | None = Field(default=None)
    call_id: str
    output: str | list[ResponseInputText | ResponseInputImage | ResponseInputFile]
    status: Literal["in_progress", "completed", "incomplete"] = Field(default="completed")


class FunctionTool(BaseModel):
    """A function tool for the Responses API (flat schema)."""

    type: Literal["function"]
    name: str
    description: str | None = Field(default=None)
    parameters: dict[str, Any] | None = Field(default=None)
    strict: bool | None = Field(default=None)


class ImageGeneration(BaseModel):
    """Image-generation built-in tool for the Responses API."""

    type: Literal["image_generation"]
    action: Literal["generate", "edit", "auto"] = Field(default="auto")
    model: str | None = Field(default=None)
    output_format: Literal["png", "webp", "jpeg"] = Field(default="png")
    quality: Literal["low", "medium", "high", "auto"] = Field(default="auto")
    size: str = Field(default="auto")


class ToolChoiceFunction(BaseModel):
    """Forces the model to call a specific named function (Responses API)."""

    type: Literal["function"]
    name: str


class ToolChoiceTypes(BaseModel):
    """Forces the model to use a specific built-in tool type."""

    type: Literal["image_generation"]


class ResponseUsage(BaseModel):
    """Token-usage statistics for a Responses API response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, Any] = Field(default_factory=lambda: {"cached_tokens": 0})
    output_tokens_details: dict[str, Any] = Field(default_factory=lambda: {"reasoning_tokens": 0})


class ResponseOutputText(BaseModel):
    """Text content part inside a Responses API output message."""

    type: Literal["output_text"]
    text: str | None = Field(default=None)
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    logprobs: list[dict[str, Any]] | None = Field(default=None)


class ResponseOutputRefusal(BaseModel):
    """Refusal content part inside a Responses API output message."""

    type: Literal["refusal"]
    refusal: str | None = Field(default=None)


ResponseOutputContent = ResponseOutputText | ResponseOutputRefusal


class ResponseOutputMessage(BaseModel):
    """Assistant message output item in a Responses API response."""

    id: str
    type: Literal["message"]
    status: Literal["in_progress", "completed", "incomplete"] = Field(default="completed")
    role: Literal["assistant"]
    content: list[ResponseOutputText | ResponseOutputRefusal]


class SummaryTextContent(BaseModel):
    """Summary text part inside a reasoning item."""

    type: Literal["summary_text"] = Field(default="summary_text")
    text: str


class ReasoningTextContent(BaseModel):
    """Full reasoning text part inside a reasoning item."""

    type: Literal["reasoning_text"] = Field(default="reasoning_text")
    text: str


class ResponseReasoningItem(BaseModel):
    """A reasoning output item emitted by a thinking model."""

    id: str
    type: Literal["reasoning"] = Field(default="reasoning")
    status: Literal["in_progress", "completed", "incomplete"] | None = Field(default=None)
    summary: list[SummaryTextContent] | None = Field(default=None)
    content: list[ReasoningTextContent] | None = Field(default=None)
    encrypted_content: str | None = Field(default=None)


class ResponseToolCall(BaseModel):
    """A function-call output item emitted by the model."""

    id: str
    type: Literal["function_call"] = Field(default="function_call")
    call_id: str
    name: str
    arguments: str
    status: Literal["in_progress", "completed", "incomplete"] = Field(default="completed")


class ImageGenerationCall(BaseModel):
    """An image-generation output item emitted by the Responses API."""

    id: str
    type: Literal["image_generation_call"] = Field(default="image_generation_call")
    status: Literal["completed", "in_progress", "generating", "failed"] = Field(default="completed")
    result: str | None = Field(default=None)
    output_format: str | None = Field(default=None)
    size: str | None = Field(default=None)
    revised_prompt: str | None = Field(default=None)


class ResponseFormatText(BaseModel):
    """Plain-text output format."""

    type: Literal["text"] = Field(default="text")


class ResponseFormatTextJSONSchemaConfig(BaseModel):
    """JSON-schema-constrained output format."""

    model_config = {"protected_namespaces": (), "arbitrary_types_allowed": True}

    type: Literal["json_schema"] = Field(default="json_schema")
    name: str | None = Field(default=None)
    schema_: dict[str, Any] | None = Field(
        default=None, alias="schema", serialization_alias="schema"
    )
    description: str | None = Field(default=None)


class ResponseTextConfig(BaseModel):
    """Top-level text configuration block in a Responses API response."""

    format: ResponseFormatText | ResponseFormatTextJSONSchemaConfig = Field(
        default_factory=ResponseFormatText
    )


class ResponseCreateRequest(BaseModel):
    """Request body for POST /v1/responses."""

    model: str
    input: (
        str
        | list[
            ResponseInputMessage
            | ResponseOutputMessage
            | ResponseReasoningItem
            | ResponseFunctionToolCall
            | FunctionCallOutput
            | ImageGenerationCall
        ]
    )
    instructions: str | None = Field(default=None)
    temperature: float | None = Field(default=1, ge=0, le=2)
    top_p: float | None = Field(default=1, ge=0, le=1)
    max_output_tokens: int | None = Field(default=None)
    stream: bool | None = Field(default=False)
    stream_options: dict[str, Any] | None = Field(default=None)
    tool_choice: (
        Literal["none", "auto", "required"] | ToolChoiceFunction | ToolChoiceTypes | None
    ) = Field(default=None)
    tools: list[FunctionTool | ImageGeneration] | None = Field(default=None)
    store: bool | None = Field(default=None)
    prompt_cache_key: str | None = Field(default=None)
    response_format: dict[str, Any] | None = Field(default=None)
    metadata: dict[str, Any] | None = Field(default=None)
    parallel_tool_calls: bool | None = Field(default=True)


class ResponseCreateResponse(BaseModel):
    """Response body for POST /v1/responses."""

    id: str
    object: Literal["response"] = Field(default="response")
    created_at: int
    completed_at: int | None = Field(default=None)
    model: str
    output: list[
        ResponseReasoningItem
        | ResponseOutputMessage
        | ResponseFunctionToolCall
        | ImageGenerationCall
    ]
    status: Literal["completed", "failed", "in_progress", "cancelled", "queued", "incomplete"] = (
        Field(default="completed")
    )
    tool_choice: (
        Literal["none", "auto", "required"] | ToolChoiceFunction | ToolChoiceTypes | None
    ) = Field(default=None)
    tools: list[FunctionTool | ImageGeneration] = Field(default_factory=list)
    usage: ResponseUsage | None = Field(default=None)
    error: dict[str, Any] | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    text: ResponseTextConfig | None = Field(default_factory=ResponseTextConfig)


class ModelData(BaseModel):
    """Single model entry in the model list."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "google"


class ModelListResponse(BaseModel):
    """Response body for GET /v1/models."""

    object: str = "list"
    data: list[ModelData]


class HealthCheckResponse(BaseModel):
    """Response body for the health check endpoint."""

    ok: bool
    storage: dict[str, Any] | None = Field(default=None)
    clients: dict[str, bool] | None = Field(default=None)
    error: str | None = Field(default=None)


class ConversationInStore(BaseModel):
    """Persisted conversation record stored in LMDB."""

    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)
    model: str = Field(..., description="Model used for the conversation")
    client_id: str = Field(..., description="Identifier of the Gemini client")
    metadata: list[str | None] = Field(
        ..., description="Metadata for Gemini API to locate the conversation"
    )
    messages: list[ChatCompletionMessage] = Field(
        ..., description="Message contents in the conversation"
    )


ChatCompletionMessage.model_rebuild()
ChatCompletionMessageToolCall.model_rebuild()
ChatCompletionRequest.model_rebuild()
