# ruff: noqa: N815  # camelCase field names are the Gemini REST wire format
"""Pydantic models for the Google Gemini REST API v1beta.

Defines request and response models for the ``generateContent``,
``streamGenerateContent``, ``models.list``, and ``models.get`` endpoints.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GeminiInlineData(BaseModel):
    """Embedded binary data, such as an image."""

    mimeType: str
    data: str


class GeminiFileData(BaseModel):
    """Reference to a file uploaded through the Files API."""

    mimeType: str
    fileUri: str


class GeminiFunctionCall(BaseModel):
    """Function call issued by the model."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class GeminiFunctionResponse(BaseModel):
    """Result of a function execution submitted by the user."""

    name: str
    response: dict[str, Any] = Field(default_factory=dict)


class GeminiPart(BaseModel):
    """A content part containing text, thoughts, data, or a function call or response."""

    text: str | None = None
    thought: bool | None = None
    inlineData: GeminiInlineData | None = None
    functionCall: GeminiFunctionCall | None = None
    functionResponse: GeminiFunctionResponse | None = None
    fileData: GeminiFileData | None = None


class GeminiContent(BaseModel):
    """Message content for a user, model, or function role."""

    role: Literal["user", "model", "function"] | None = None
    parts: list[GeminiPart] = Field(default_factory=list)


class GeminiSystemInstruction(BaseModel):
    """Top-level system instruction, separate from contents."""

    parts: list[GeminiPart] = Field(default_factory=list)


class GeminiFunctionDeclaration(BaseModel):
    """Function declaration."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class GeminiTool(BaseModel):
    """Collection of tools."""

    functionDeclarations: list[GeminiFunctionDeclaration] = Field(default_factory=list)


class GeminiFunctionCallingConfig(BaseModel):
    """Function calling configuration."""

    mode: Literal["AUTO", "NONE", "ANY", "VALIDATED"] = "AUTO"
    allowedFunctionNames: list[str] | None = None


class GeminiToolConfig(BaseModel):
    """Tool configuration."""

    functionCallingConfig: GeminiFunctionCallingConfig | None = None


class GeminiSafetySetting(BaseModel):
    """Safety setting."""

    category: str
    threshold: str


class GeminiThinkingConfig(BaseModel):
    """Thinking configuration controlling model reasoning behavior."""

    includeThoughts: bool | None = None
    thinkingBudget: int | None = None
    thinkingLevel: str | None = None


class GeminiGenerationConfig(BaseModel):
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


class GeminiGenerateContentRequest(BaseModel):
    """Request body for generateContent or streamGenerateContent."""

    contents: list[GeminiContent] = Field(default_factory=list)
    systemInstruction: GeminiSystemInstruction | None = None
    tools: list[GeminiTool] | None = None
    toolConfig: GeminiToolConfig | None = None
    safetySettings: list[GeminiSafetySetting] | None = None
    generationConfig: GeminiGenerationConfig | None = None
    cachedContent: str | None = None


class GeminiSafetyRating(BaseModel):
    """Safety rating."""

    category: str
    probability: str


class GeminiCitationSource(BaseModel):
    """Individual citation source."""

    startIndex: int | None = None
    endIndex: int | None = None
    uri: str | None = None
    license: str | None = None


class GeminiCitationMetadata(BaseModel):
    """Citation metadata."""

    citationSources: list[GeminiCitationSource] = Field(default_factory=list)


class GeminiGroundingChunkWeb(BaseModel):
    """Web information for a grounding source."""

    uri: str | None = None
    title: str | None = None


class GeminiGroundingChunk(BaseModel):
    """Grounding source chunk."""

    web: GeminiGroundingChunkWeb | None = None


class GeminiSearchEntryPoint(BaseModel):
    """Search entry point."""

    renderedContent: str | None = None
    sdkBlob: str | None = None


class GeminiGroundingSupport(BaseModel):
    """Grounding support segment."""

    segment: dict[str, Any] | None = None
    groundingChunkIndices: list[int] = Field(default_factory=list)
    confidenceScores: list[float] = Field(default_factory=list)


class GeminiGroundingMetadata(BaseModel):
    """Grounding metadata, such as Google Search results."""

    webSearchQueries: list[str] = Field(default_factory=list)
    groundingChunks: list[GeminiGroundingChunk] = Field(default_factory=list)
    searchEntryPoint: GeminiSearchEntryPoint | None = None
    groundingSupports: list[GeminiGroundingSupport] = Field(default_factory=list)


class GeminiCandidate(BaseModel):
    """Generated candidate."""

    content: GeminiContent | None = None
    finishReason: str | None = None
    index: int = 0
    safetyRatings: list[GeminiSafetyRating] = Field(default_factory=list)
    citationMetadata: GeminiCitationMetadata | None = None
    groundingMetadata: GeminiGroundingMetadata | None = None
    tokenCount: int | None = None
    avgLogprobs: float | None = None


class GeminiUsageMetadata(BaseModel):
    """Token usage statistics."""

    promptTokenCount: int = 0
    candidatesTokenCount: int = 0
    totalTokenCount: int = 0
    thoughtsTokenCount: int | None = None
    cachedContentTokenCount: int | None = None
    toolUsePromptTokenCount: int | None = None


class GeminiGenerateContentResponse(BaseModel):
    """Response body for generateContent or streamGenerateContent."""

    candidates: list[GeminiCandidate] = Field(default_factory=list)
    usageMetadata: GeminiUsageMetadata | None = None
    modelVersion: str | None = None


class GeminiModelInfo(BaseModel):
    """Information about a single model."""

    name: str
    version: str | None = None
    displayName: str | None = None
    description: str | None = None
    inputTokenLimit: int | None = None
    outputTokenLimit: int | None = None
    supportedGenerationMethods: list[str] = Field(default_factory=list)


class GeminiModelListResponse(BaseModel):
    """Response body for models.list."""

    models: list[GeminiModelInfo] = Field(default_factory=list)
    nextPageToken: str | None = None


class GeminiErrorDetail(BaseModel):
    """Standard Google API error details."""

    code: int
    message: str
    status: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class GeminiErrorResponse(BaseModel):
    """Standard Google API error envelope."""

    error: GeminiErrorDetail
