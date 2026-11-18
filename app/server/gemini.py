"""Native Gemini REST API v1beta endpoints.

- GET  /v1beta/models                                — list models
- GET  /v1beta/models/{model}                        — get one model
- POST /v1beta/models/{model}:generateContent        — non-streaming generation
- POST /v1beta/models/{model}:streamGenerateContent  — streaming (SSE)

Requests are translated into the same internal AppMessage/FunctionTool pipeline the
OpenAI-shaped routes use, so this module owns only the Gemini wire format: it converts
`contents` in, converts candidates back out, and renders errors in Google's envelope.

Two Gemini features have no local equivalent and are refused rather than dropped, because
silently ignoring them would change what the model is answering: `fileData` (a Files API URI
this process cannot fetch) and `cachedContent`. Generation controls Gemini Web does not expose
are accepted and logged instead - see `_log_ignored_gemini_options`.
"""

from __future__ import annotations

import mimetypes
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import orjson
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from gemini_webapi import ModelOutput
from gemini_webapi.client import ChatSession
from loguru import logger
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.models import (
    AppContentItem,
    AppMessage,
    AppMessageRole,
    AppToolCall,
    AppToolCallFunction,
    FunctionTool,
    StructuredOutputRequirement,
    ToolChoiceFunction,
)
from app.models.gemini_models import (
    GeminiCandidate,
    GeminiContent,
    GeminiErrorDetail,
    GeminiErrorResponse,
    GeminiFunctionCall,
    GeminiGenerateContentRequest,
    GeminiGenerateContentResponse,
    GeminiGenerationConfig,
    GeminiInlineData,
    GeminiModelInfo,
    GeminiModelListResponse,
    GeminiPart,
    GeminiSystemInstruction,
    GeminiUsageMetadata,
)

# The request pipeline is the one the OpenAI routes use; this module adds the Gemini wire format.
from app.server.chat import (
    ProcessedImageData,
    ProcessedMediaData,
    StreamingOutputFilter,
    _build_structured_requirement,
    _Downloads,
    _media_markdown,
    _persist_conversation,
    _prepare_messages_for_model,
    _resolve_model_name,
    _send_request,
    _tool_choice_failure,
    available_models,
)
from app.server.middleware import (
    get_temp_dir,
    openai_error,
    verify_gemini_api_key,
)
from app.services import GeminiClientPool, GeminiClientWrapper, LMDBConversationStore
from app.utils.helper import (
    StructuredOutputValidationError,
    calculate_usage,
    dump_model,
    guess_extension_for_mime,
    normalize_llm_text,
    normalize_openapi_schema,
    process_llm_output,
    validate_json_schema,
)

router = APIRouter()


_GRPC_STATUS = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    429: "RESOURCE_EXHAUSTED",
    503: "UNAVAILABLE",
}


def add_exception_handlers(app: FastAPI) -> None:
    """Answer every error in the envelope of the API the request was made to."""

    def error(request: Request, status_code: int, message: str) -> JSONResponse:
        if request.url.path.startswith("/v1beta/"):
            return _error_response(status_code, message)
        return openai_error(status_code, message)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        response = error(request, exc.status_code, str(exc.detail))
        response.headers.update(exc.headers or {})  # e.g. `Allow` on a 405
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        """A malformed request is a 400 in both APIs, not FastAPI's 422."""
        return error(request, 400, f"Invalid request: {exc.errors() or exc}")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        return error(request, 500, str(exc))


def _inline_data_item(inline: GeminiInlineData) -> AppContentItem:
    """An `inlineData` part as a file attached to a message."""
    return AppContentItem(
        type="file",
        file_data=inline.data,
        filename=f"inline{guess_extension_for_mime(inline.mimeType)}",
    )


def _gemini_contents_to_messages(
    contents: list[GeminiContent],
    system_instruction: GeminiSystemInstruction | None = None,
) -> list[AppMessage]:
    """Convert Gemini contents and systemInstruction into internal messages.

    - a `model` turn with functionCall parts becomes an assistant message with tool calls
    - every functionResponse part becomes a tool message, whatever role carried it
    - text and inline data keep the order they were sent in
    """
    messages: list[AppMessage] = []
    if system_instruction and (texts := [p.text for p in system_instruction.parts if p.text]):
        messages.append(AppMessage(role="system", content="\n".join(texts)))

    roles: dict[str | None, AppMessageRole] = {"model": "assistant", "function": "tool"}
    for content in contents:
        role = roles.get(content.role, "user")
        texts = []
        items: list[AppContentItem] = []  # text and files, in order
        has_file = False
        tool_calls: list[AppToolCall] = []
        tool_results: list[AppMessage] = []

        for part in content.parts:
            if part.thought:
                # A thought this server returned, replayed by the client: not part of the answer.
                continue
            if part.text is not None:
                texts.append(part.text)
                items.append(AppContentItem(type="text", text=part.text))
            if part.inlineData:
                items.append(_inline_data_item(part.inlineData))
                has_file = True
            if part.functionCall:
                tool_calls.append(
                    AppToolCall(
                        # The native API pairs a call and its response by function name. An id
                        # invented here must not vary between requests, or the same history
                        # would stop matching the chat stored for it.
                        id=f"call_{len(tool_calls)}",
                        function=AppToolCallFunction(
                            name=part.functionCall.name,
                            arguments=orjson.dumps(part.functionCall.args or {}).decode(),
                        ),
                    )
                )
            if part.functionResponse:
                tool_results.append(
                    AppMessage(
                        role="tool",
                        content=orjson.dumps(part.functionResponse.response).decode(),
                        name=part.functionResponse.name,
                    )
                )

        text = "\n".join(texts)
        if tool_results:
            messages += tool_results
            if texts and role != "tool":
                messages.append(AppMessage(role=role, content=text))
        elif tool_calls:
            messages.append(
                AppMessage(role="assistant", content=text if texts else None, tool_calls=tool_calls)
            )
        else:
            # Parts with nothing in them still make a message, to keep the turn structure.
            messages.append(AppMessage(role=role, content=items if has_file else text))

    return messages


def _gemini_tools_to_internal(
    tools: list[Any] | None,
    tool_config: Any | None = None,
) -> tuple[
    list[FunctionTool] | None,
    Literal["none", "auto", "required"] | ToolChoiceFunction | None,
]:
    """Convert Gemini tools + toolConfig into internal FunctionTool list and tool_choice."""
    if not tools:
        return None, None

    internal_tools: list[FunctionTool] = []
    for tool in tools:
        internal_tools.extend(
            FunctionTool(
                type="function",
                name=decl.name,
                description=decl.description,
                parameters=decl.parameters,
            )
            for decl in tool.functionDeclarations or []
        )
    tool_choice: Literal["none", "auto", "required"] | ToolChoiceFunction | None = None
    if tool_config and tool_config.functionCallingConfig:
        call_config = tool_config.functionCallingConfig
        mode = call_config.mode

        # Google: "This should only be set when the Mode is ANY or VALIDATED." Acting on it
        # under AUTO or NONE would hide tools the upstream would still have offered.
        allowed_names = (
            set(call_config.allowedFunctionNames or [])
            if mode in {"ANY", "VALIDATED"}
            else set[str]()
        )
        if allowed_names:
            internal_tools = [tool for tool in internal_tools if tool.name in allowed_names]

        if mode == "NONE":
            tool_choice = cast(Literal["none", "auto", "required"], "none")
        elif mode == "ANY":
            tool_choice = (
                ToolChoiceFunction(type="function", name=next(iter(allowed_names)))
                if len(allowed_names) == 1
                else cast(Literal["none", "auto", "required"], "required")
            )
        else:
            # AUTO, and VALIDATED - which also permits a natural-language answer.
            tool_choice = "auto"

    return internal_tools or None, tool_choice


def _function_call_parts(tool_calls: list[AppToolCall]) -> list[GeminiPart]:
    return [
        GeminiPart(
            functionCall=GeminiFunctionCall(
                name=call.function.name, args=orjson.loads(call.function.arguments)
            )
        )
        for call in tool_calls
    ]


def _usage_metadata(usage: tuple[int, int, int, int]) -> GeminiUsageMetadata:
    prompt, completion, total, reasoning = usage
    return GeminiUsageMetadata(
        promptTokenCount=prompt,
        candidatesTokenCount=completion - reasoning,
        totalTokenCount=total,
        thoughtsTokenCount=reasoning or None,
    )


def _sse(model: BaseModel) -> str:
    """One SSE frame; fields that are unset are left out, as Google does."""
    return f"data: {orjson.dumps(dump_model(model)).decode()}\n\n"


def _part_frame(part: GeminiPart) -> str:
    """An SSE frame that carries one part of the reply."""
    return _sse(
        GeminiGenerateContentResponse(
            candidates=[GeminiCandidate(content=GeminiContent(role="model", parts=[part]), index=0)]
        )
    )


def _gemini_error(status_code: int, message: str) -> GeminiErrorResponse:
    """An error in Google's envelope."""
    return GeminiErrorResponse(
        error=GeminiErrorDetail(
            code=status_code,
            message=message,
            status=_GRPC_STATUS.get(status_code, "INTERNAL"),
        )
    )


def _error_response(status_code: int, message: str) -> JSONResponse:
    """An HTTP error in Google's envelope."""
    content = _gemini_error(status_code, message).model_dump(mode="json")
    return JSONResponse(status_code=status_code, content=content)


def _log_ignored_gemini_options(
    request: GeminiGenerateContentRequest, response_schema: dict[str, Any] | None
) -> None:
    """Debug-log valid Gemini options that the Gemini Web client cannot forward.

    `response_schema` is the already-translated result of `_gemini_response_schema`, passed in
    rather than recomputed so the OpenAPI translation and its validation run once per request.
    """
    ignored: set[str] = set()
    if "safetySettings" in request.model_fields_set:
        ignored.add("safetySettings")

    if gen_cfg := request.generationConfig:
        supported_structured_fields: set[str] = set()
        if gen_cfg.responseMimeType == "application/json":
            supported_structured_fields.add("responseMimeType")
        # The schema fields count as honored only if one survived translation. An empty schema
        # is valid and still counts; `None` alone means translation failed or none was supplied.
        if response_schema is not None:
            supported_structured_fields.update(("responseSchema", "responseJsonSchema"))
        ignored.update(
            name for name in gen_cfg.model_fields_set if name not in supported_structured_fields
        )

    if request.toolConfig and (call_config := request.toolConfig.functionCallingConfig):
        mode = call_config.mode
        if mode == "VALIDATED":
            ignored.add("toolConfig.functionCallingConfig.mode")
        if call_config.allowedFunctionNames and mode not in {"ANY", "VALIDATED"}:
            ignored.add("toolConfig.functionCallingConfig.allowedFunctionNames")

    if ignored:
        logger.debug(
            "[Gemini API] Ignoring option(s) unsupported by the Gemini Web upstream: "
            f"{', '.join(sorted(ignored))}"
        )


def _validate_gemini_request(request: GeminiGenerateContentRequest) -> str | None:
    """Reject malformed or unrepresentable inputs, not optional generation controls."""
    if not request.contents:
        return "contents is required and cannot be empty."

    for content in request.contents:
        if not content.parts:
            return "Each content entry must contain at least one part."
        if any(part.fileData is not None for part in content.parts):
            return "fileData is not supported; provide the data using inlineData instead."
    if request.systemInstruction and any(
        part.fileData is not None for part in request.systemInstruction.parts
    ):
        return "fileData is not supported; provide the data using inlineData instead."

    if request.cachedContent is not None:
        return "cachedContent is not supported by the Gemini Web upstream."

    if request.toolConfig and request.toolConfig.functionCallingConfig:
        call_config = request.toolConfig.functionCallingConfig
        # Only meaningful in the modes that act on it; elsewhere it is ignored, not invalid.
        if call_config.mode in {"ANY", "VALIDATED"}:
            declared_names = {
                declaration.name
                for tool in request.tools or []
                for declaration in tool.functionDeclarations
            }
            allowed_names = set(call_config.allowedFunctionNames or [])
            if unknown_names := allowed_names - declared_names:
                return (
                    f"allowedFunctionNames contains undeclared functions: {sorted(unknown_names)}"
                )

    if request.generationConfig:
        gen_cfg = request.generationConfig
        # Only `responseJsonSchema` is JSON Schema, so only it can be judged as such.
        # `responseSchema` is the OpenAPI subset, translated at use time; a gap in that
        # translation must not turn a valid Gemini request into a 400.
        if (
            gen_cfg.responseMimeType == "application/json"
            and gen_cfg.responseJsonSchema is not None
        ):
            try:
                validate_json_schema(gen_cfg.responseJsonSchema)
            except ValueError as exc:
                return str(exc)
    return None


def _gemini_response_schema(gen_cfg: GeminiGenerationConfig) -> dict[str, Any] | None:
    """Return the requested response schema as JSON Schema, or None if it cannot be used."""
    if gen_cfg.responseMimeType != "application/json":
        return None

    if gen_cfg.responseJsonSchema is not None:
        return gen_cfg.responseJsonSchema

    if gen_cfg.responseSchema is None:
        return None

    schema = normalize_openapi_schema(gen_cfg.responseSchema)
    try:
        validate_json_schema(schema)
    except ValueError as exc:
        # Enforcing a schema we could not translate would reject good answers.
        logger.debug(f"[Gemini API] Ignoring responseSchema that is not representable: {exc}")
        return None
    return schema


def _gemini_structured_requirement(
    request: GeminiGenerateContentRequest,
) -> tuple[dict[str, Any] | None, StructuredOutputRequirement | None]:
    """Translate the requested response schema once, returning it with its requirement.

    The requirement is deliberately non-strict. Google's own API guarantees conformance through
    constrained decoding; Gemini Web offers no such control, so the schema can only be asked for
    in the prompt. Failing the request on a near-miss would throw away an answer the caller can
    still use, so a violation degrades to the raw text and is logged instead.
    """
    if not request.generationConfig:
        return None, None

    gen_cfg = request.generationConfig
    if gen_cfg.responseMimeType != "application/json":
        return None, None

    schema = _gemini_response_schema(gen_cfg)
    response_format = (
        {"type": "json_object"}
        if schema is None
        else {"type": "json_schema", "json_schema": {"schema": schema, "strict": False}}
    )
    requirement = _build_structured_requirement(response_format)
    return schema, requirement


def _model_info(model_id: str) -> GeminiModelInfo:
    return GeminiModelInfo(
        name=f"models/{model_id}",
        displayName=model_id,
        description=f"Gemini model: {model_id}",
        supportedGenerationMethods=["generateContent", "streamGenerateContent"],
    )


@router.get("/v1beta/models")
async def gemini_list_models(api_key: str = Depends(verify_gemini_api_key)):
    """List available models (Gemini API format)."""
    models = available_models(GeminiClientPool())

    logger.info(f"[Gemini API] Retrieved {len(models)} models")
    if not models:
        logger.warning("[Gemini API] Model list is empty")

    return GeminiModelListResponse(models=[_model_info(m.id) for m in models])


@router.get("/v1beta/models/{model:path}")
async def gemini_get_model(model: str, api_key: str = Depends(verify_gemini_api_key)):
    """Get one model's info (Gemini API format)."""
    model_name = model.removeprefix("models/")
    try:
        _resolve_model_name(GeminiClientPool(), model_name)
    except ValueError as exc:
        return _error_response(404, str(exc))
    # Reported under the name it was asked for, which may be an alias the list does not show.
    return _model_info(model_name)


class _SentRequest(NamedTuple):
    """A generation request that has been sent upstream, with what finishing it needs."""

    output: AsyncGenerator[ModelOutput] | ModelOutput
    session: ChatSession
    client: GeminiClientWrapper
    db: LMDBConversationStore
    model_name: str
    resolved_model: str
    messages: list[AppMessage]  # as the caller sent them; usage is counted on these
    prepared: list[AppMessage]  # with tool and schema instructions added; this is what is stored
    tool_choice: Literal["none", "auto", "required"] | ToolChoiceFunction | None
    structured_requirement: StructuredOutputRequirement | None


async def _send_gemini_request(
    model: str, request: GeminiGenerateContentRequest, tmp_dir: Path, *, stream: bool
) -> _SentRequest:
    """Validate a generation request and send it upstream.

    Raises `HTTPException`, which the app answers in Google's envelope.
    """
    model_name = model.removeprefix("models/")
    pool, db = GeminiClientPool(), LMDBConversationStore()
    try:
        resolved_model = _resolve_model_name(pool, model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if validation_error := _validate_gemini_request(request):
        raise HTTPException(status_code=400, detail=validation_error)
    response_schema, structured_requirement = _gemini_structured_requirement(request)
    _log_ignored_gemini_options(request, response_schema)

    messages = _gemini_contents_to_messages(request.contents, request.systemInstruction)
    tools, tool_choice = _gemini_tools_to_internal(request.tools, request.toolConfig)
    instructions = [structured_requirement.instruction] if structured_requirement else None
    prepared = _prepare_messages_for_model(messages, tools, tool_choice, instructions)

    output, session, client = await _send_request(
        pool=pool,
        db=db,
        resolved_model=resolved_model,
        history=[],  # gems are found from the history; this surface never uses them
        prepared=prepared,
        tools=tools,
        tool_choice=tool_choice,
        instructions=instructions,
        repeat_instructions=True,
        tmp_dir=tmp_dir,
        stream=stream,
    )
    return _SentRequest(
        output,
        session,
        client,
        db,
        model_name,
        resolved_model,
        messages,
        prepared,
        tool_choice,
        structured_requirement,
    )


async def _generated_files(
    downloads: _Downloads, base_url: str
) -> tuple[list[GeminiInlineData], str]:
    """The reply's generated images as inline data, and markdown links to its videos and audio.

    Videos and audio are linked from the text, as on the OpenAI routes: the native format has no
    part for them, and inlining a video would make the response enormous.
    """
    images: list[GeminiInlineData] = []
    media_markdown = ""
    for kind, item, data in await downloads.all():
        if kind == "media":
            files = cast(ProcessedMediaData, data)
            if links := _media_markdown(item, files, downloads.kept_media, base_url):
                media_markdown += f"\n\n{links}"
        elif name := downloads.new_image(image := cast(ProcessedImageData, data)):
            mime_type = mimetypes.guess_type(name)[0] or "image/png"
            images.append(GeminiInlineData(mimeType=mime_type, data=image[0]))
    return images, media_markdown


def _stored_reply(text: str, images: list[GeminiInlineData]) -> str | list[AppContentItem]:
    """A reply as it is stored: what `_gemini_contents_to_messages` makes of it when a client
    sends it back, images included, so that the history still matches this chat."""
    if not images:
        return text
    text_items = [AppContentItem(type="text", text=text)] if text else []
    return [*text_items, *(_inline_data_item(image) for image in images)]


@router.post("/v1beta/models/{model:path}:generateContent")
async def gemini_generate_content(
    model: str,
    request: GeminiGenerateContentRequest,
    raw_request: Request,
    api_key: str = Depends(verify_gemini_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    """Non-streaming content generation (Gemini API format)."""
    sent = await _send_gemini_request(model, request, tmp_dir, stream=False)
    resp = sent.output
    assert isinstance(resp, ModelOutput)

    try:
        thoughts, visible_output, storage_output, tool_calls = process_llm_output(
            normalize_llm_text(resp.thoughts or ""),
            normalize_llm_text(resp.text or ""),
            sent.structured_requirement,
        )
    except StructuredOutputValidationError as exc:
        return _error_response(502, str(exc))
    if choice_error := _tool_choice_failure(sent.tool_choice, tool_calls):
        return _error_response(502, choice_error)

    downloads = _Downloads()
    downloads.add(resp)
    images, media_markdown = await _generated_files(downloads, str(raw_request.base_url))
    visible_output += media_markdown

    usage = calculate_usage(sent.messages, visible_output, tool_calls, thoughts)
    _persist_conversation(
        sent.db,
        sent.resolved_model,
        sent.client,
        sent.session.metadata,
        sent.prepared,
        _stored_reply(storage_output + media_markdown, images),
        tool_calls,
    )

    parts = [GeminiPart(text=thoughts, thought=True)] if thoughts else []
    if visible_output:
        parts.append(GeminiPart(text=visible_output))
    parts += [GeminiPart(inlineData=image) for image in images]
    parts += _function_call_parts(tool_calls)
    return dump_model(
        GeminiGenerateContentResponse(
            candidates=[
                GeminiCandidate(
                    content=GeminiContent(role="model", parts=parts), finishReason="STOP", index=0
                )
            ],
            usageMetadata=_usage_metadata(usage),
            modelVersion=sent.model_name,
        )
    )


@router.post("/v1beta/models/{model:path}:streamGenerateContent")
async def gemini_stream_generate_content(
    model: str,
    request: GeminiGenerateContentRequest,
    raw_request: Request,
    api_key: str = Depends(verify_gemini_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    """Streaming content generation (Gemini API format, SSE)."""
    sent = await _send_gemini_request(model, request, tmp_dir, stream=True)
    return StreamingResponse(
        _stream_reply(sent, str(raw_request.base_url)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_reply(sent: _SentRequest, base_url: str) -> AsyncGenerator[str]:
    """The SSE frames of a streamed reply; the conversation is stored once it is complete."""
    downloads = _Downloads()
    try:
        async for frame in _reply_frames(sent, base_url, downloads):
            yield frame
    finally:
        # Whatever ended the stream early - an error, or the client going away.
        await downloads.cancel()


async def _reply_frames(
    sent: _SentRequest, base_url: str, downloads: _Downloads
) -> AsyncGenerator[str]:
    generator = sent.output
    assert not isinstance(generator, ModelOutput)
    full_thoughts, full_text = "", ""
    last_chunk: ModelOutput | None = None
    suppressor = StreamingOutputFilter()
    # Structured output is withheld until it has been validated.
    forward_text = not sent.structured_requirement

    try:
        async for chunk in generator:
            last_chunk = chunk
            downloads.add(chunk)

            if chunk.thoughts_delta:
                full_thoughts += chunk.thoughts_delta
                yield _part_frame(GeminiPart(text=chunk.thoughts_delta, thought=True))
            if chunk.text_delta:
                full_text += chunk.text_delta
                if forward_text and (visible := suppressor.process(chunk.text_delta)):
                    yield _part_frame(GeminiPart(text=visible))
    except Exception as e:
        logger.exception(f"[Gemini API] Streaming error: {e}")
        yield _sse(_gemini_error(500, "Streaming error occurred."))
        return

    # The final chunk can carry more than its deltas added up to. The rest is sent on, so that
    # what gets validated and stored is what the client received.
    if last_chunk is not None:
        final_thoughts, final_text = last_chunk.thoughts or "", last_chunk.text or ""
        if len(final_thoughts) > len(full_thoughts) and final_thoughts.startswith(full_thoughts):
            yield _part_frame(GeminiPart(text=final_thoughts[len(full_thoughts) :], thought=True))
            full_thoughts = final_thoughts
        if len(final_text) > len(full_text) and final_text.startswith(full_text):
            tail, full_text = final_text[len(full_text) :], final_text
            if forward_text and (visible := suppressor.process(tail)):
                yield _part_frame(GeminiPart(text=visible))
    if forward_text and (held_back := suppressor.flush()):
        yield _part_frame(GeminiPart(text=held_back))

    # The two expected failures are reported with their own message and status, matching the
    # non-streaming route; only unexpected errors fall through to the catch-all below, which
    # cannot say anything more useful than that something broke.
    try:
        thoughts, visible_output, storage_output, tool_calls = process_llm_output(
            normalize_llm_text(full_thoughts),
            normalize_llm_text(full_text),
            sent.structured_requirement,
        )
    except StructuredOutputValidationError as exc:
        logger.warning(f"[Gemini API] Structured output rejected mid-stream: {exc}")
        yield _sse(_gemini_error(502, str(exc)))
        return
    if choice_error := _tool_choice_failure(sent.tool_choice, tool_calls):
        logger.warning(f"[Gemini API] Forced tool choice unmet mid-stream: {choice_error}")
        yield _sse(_gemini_error(502, choice_error))
        return

    try:
        if not forward_text and visible_output:
            yield _part_frame(GeminiPart(text=visible_output))

        # ponytail: no keep-alive is sent while the downloads finish; the OpenAI streams send
        # a comment line every 5 s, which Google's clients are not known to accept.
        images, media_markdown = await _generated_files(downloads, base_url)
        if media_markdown:
            yield _part_frame(GeminiPart(text=media_markdown))
            visible_output += media_markdown
            storage_output += media_markdown
        for image in images:
            yield _part_frame(GeminiPart(inlineData=image))

        usage = calculate_usage(sent.messages, visible_output, tool_calls, thoughts)
        call_parts = _function_call_parts(tool_calls)
        yield _sse(
            GeminiGenerateContentResponse(
                candidates=[
                    GeminiCandidate(
                        content=GeminiContent(role="model", parts=call_parts)
                        if call_parts
                        else None,
                        finishReason="STOP",
                        index=0,
                    )
                ],
                usageMetadata=_usage_metadata(usage),
                modelVersion=sent.model_name,
            )
        )
        _persist_conversation(
            sent.db,
            sent.resolved_model,
            sent.client,
            sent.session.metadata,
            sent.prepared,
            _stored_reply(storage_output, images),
            tool_calls,
        )
    except Exception as exc:
        logger.exception(f"[Gemini API] Post-processing error: {exc}")
        yield _sse(_gemini_error(500, "Post-processing error."))
