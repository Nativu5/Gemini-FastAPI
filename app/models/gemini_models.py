# ruff: noqa: N815  # camelCase field names are the Gemini REST wire format
"""Pydantic models for the Google Gemini REST API v1beta."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, AliasGenerator, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_snake


class _GeminiModel(BaseModel):
    # Google accepts every field under its proto name too, and the Python SDK sends some that way.
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda name: AliasChoices(name, to_snake(name))
        )
    )


class GeminiInlineData(_GeminiModel):
    """Embedded binary data, such as an image."""

    mimeType: str
    data: str


class GeminiFileData(_GeminiModel):
    """Reference to a file uploaded through the Files API."""

    mimeType: str
    fileUri: str


class GeminiFunctionCall(_GeminiModel):
    """Function call issued by the model."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class GeminiFunctionResponse(_GeminiModel):
    """Result of a function execution submitted by the user."""

    name: str
    response: dict[str, Any] = Field(default_factory=dict)


class GeminiPart(_GeminiModel):
    """A content part containing text, thoughts, data, or a function call or response."""

    text: str | None = None
    thought: bool | None = None
    inlineData: GeminiInlineData | None = None
    functionCall: GeminiFunctionCall | None = None
    functionResponse: GeminiFunctionResponse | None = None
    fileData: GeminiFileData | None = None


class GeminiContent(_GeminiModel):
    """Message content for a user, model, or function role."""

    role: Literal["user", "model", "function"] | None = None
    parts: list[GeminiPart] = Field(default_factory=list)


class GeminiSystemInstruction(_GeminiModel):
    """Top-level system instruction, separate from contents."""

    parts: list[GeminiPart] = Field(default_factory=list)


class GeminiFunctionDeclaration(_GeminiModel):
    """Function declaration."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    parametersJsonSchema: dict[str, Any] | None = None


class GeminiTool(_GeminiModel):
    """Collection of tools."""

    functionDeclarations: list[GeminiFunctionDeclaration] = Field(default_factory=list)


class GeminiFunctionCallingConfig(_GeminiModel):
    """Function calling configuration."""

    mode: Literal["AUTO", "NONE", "ANY", "VALIDATED"] = "AUTO"
    allowedFunctionNames: list[str] | None = None


class GeminiToolConfig(_GeminiModel):
    """Tool configuration."""

    functionCallingConfig: GeminiFunctionCallingConfig | None = None


class GeminiSafetySetting(_GeminiModel):
    """Safety setting."""

    category: str
    threshold: str


class GeminiThinkingConfig(_GeminiModel):
    """Thinking configuration controlling model reasoning behavior."""

    includeThoughts: bool | None = None
    thinkingBudget: int | None = None
    thinkingLevel: str | None = None


class GeminiGenerationConfig(_GeminiModel):
    """Content generation parameters."""

    temperature: float | None = None
    topP: float | None = None
    topK: int | None = None
    maxOutputTokens: int | None = None
    stopSequences: list[str] | None = None
    responseMimeType: str | None = None
    responseSchema: dict[str, Any] | None = None
    responseJsonSchema: dict[str, Any] | None = None
    candidateCount: int | None = None
    thinkingConfig: GeminiThinkingConfig | None = None
    responseFormat: dict[str, Any] | None = None


class GeminiGenerateContentRequest(_GeminiModel):
    """Request body for generateContent or streamGenerateContent."""

    contents: list[GeminiContent] = Field(default_factory=list)
    systemInstruction: GeminiSystemInstruction | None = None
    tools: list[GeminiTool] | None = None
    toolConfig: GeminiToolConfig | None = None
    safetySettings: list[GeminiSafetySetting] | None = None
    generationConfig: GeminiGenerationConfig | None = None
    cachedContent: str | None = None


class GeminiCandidate(_GeminiModel):
    """Generated candidate."""

    content: GeminiContent | None = None
    finishReason: str | None = None
    index: int = 0


class GeminiUsageMetadata(_GeminiModel):
    """Token usage statistics."""

    promptTokenCount: int = 0
    candidatesTokenCount: int = 0
    totalTokenCount: int = 0
    thoughtsTokenCount: int | None = None


class GeminiGenerateContentResponse(_GeminiModel):
    """Response body for generateContent or streamGenerateContent."""

    candidates: list[GeminiCandidate] = Field(default_factory=list)
    usageMetadata: GeminiUsageMetadata | None = None
    modelVersion: str | None = None


class GeminiModelInfo(_GeminiModel):
    """Information about a single model."""

    name: str
    version: str | None = None
    displayName: str | None = None
    description: str | None = None
    inputTokenLimit: int | None = None
    outputTokenLimit: int | None = None
    supportedGenerationMethods: list[str] = Field(default_factory=list)


class GeminiModelListResponse(_GeminiModel):
    """Response body for models.list."""

    models: list[GeminiModelInfo] = Field(default_factory=list)
    nextPageToken: str | None = None


class GeminiErrorDetail(_GeminiModel):
    """Standard Google API error details."""

    code: int
    message: str
    status: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class GeminiErrorResponse(_GeminiModel):
    """Standard Google API error envelope."""

    error: GeminiErrorDetail
