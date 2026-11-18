import asyncio
import base64
import hashlib
import io
import reprlib
import uuid
from collections.abc import AsyncGenerator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from gemini_webapi import ModelOutput
from gemini_webapi.client import ChatSession
from gemini_webapi.exceptions import ModelInvalidError
from gemini_webapi.types.image import GeneratedImage, Image
from gemini_webapi.types.video import GeneratedMedia, GeneratedVideo
from loguru import logger

from app.models import (
    AppContentItem,
    AppMessage,
    AppToolCall,
    AppToolCallFunction,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionNamedToolChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    ConversationInStore,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    ImageGeneration,
    ImageGenerationCall,
    ModelData,
    ModelListResponse,
    ResponseCreateRequest,
    ResponseCreateResponse,
    ResponseFormatJSONObject,
    ResponseFormatText,
    ResponseFormatTextJSONSchemaConfig,
    ResponseFunctionToolCall,
    ResponseInputMessage,
    ResponseOutputContent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextConfig,
    ResponseUsage,
    StructuredOutputRequirement,
    SummaryTextContent,
    ToolChoiceFunction,
    ToolChoiceTypes,
)
from app.server.middleware import (
    get_media_store_dir,
    get_media_token,
    get_temp_dir,
    verify_api_key,
)
from app.services import GeminiClientPool, GeminiClientWrapper, LMDBConversationStore
from app.utils import g_config
from app.utils.config import ChatMode
from app.utils.helper import (
    SCHEMA_ADHERENCE_PROMPT,
    STREAM_FLUSH_TAIL_RE,
    STREAM_MASTER_RE,
    STREAM_TAIL_RE,
    STRICT_SCHEMA_ADHERENCE_PROMPT,
    STRUCTURED_JSON_WRAP_HINT,
    StructuredOutputValidationError,
    append_tool_hint_to_last_user_message,
    build_image_generation_instruction,
    build_tool_prompt,
    calculate_usage,
    convert_to_app_messages,
    detect_image_extension,
    dump_model,
    extract_image_dimensions,
    normalize_app_message_role,
    normalize_llm_text,
    process_llm_output,
    serialize_tool_choice_for_response,
    serialize_tools_for_response,
    strip_system_hints,
    validate_json_schema,
)

MAX_CHARS_PER_REQUEST = int(g_config.gemini.max_chars_per_request * 0.9)
# Google's temporary chat mode accepts a smaller payload than a normal chat, so tighten
# the guardrail further on top of the standard 10% safety margin.
TEMPORARY_MAX_CHARS_PER_REQUEST = int(MAX_CHARS_PER_REQUEST * 0.9)

router = APIRouter()
_AVAILABLE_MODELS_CACHE: list[ModelData] | None = None
_AVAILABLE_MODELS_CACHE_LOCK = asyncio.Lock()


type ProcessedImageData = tuple[str, int | None, int | None, str, str]
type ProcessedMediaData = dict[str, tuple[str, str]]
type ProcessedImageResult = tuple[Literal["image"], Image, ProcessedImageData]
type ProcessedMediaResult = tuple[
    Literal["media"], GeneratedVideo | GeneratedMedia, ProcessedMediaData
]


# --- Helper Functions ---


async def _image_to_base64(
    image: Image, temp_dir: Path
) -> tuple[str, int | None, int | None, str, str]:
    """Persist an image provided by gemini_webapi and return base64 plus dimensions, filename, and hash."""
    if isinstance(image, GeneratedImage):
        try:
            saved_path = await image.save(path=str(temp_dir), full_size=True)
        except Exception as e:
            logger.warning(
                f"Failed to download full-size GeneratedImage, retrying with default size: {e}"
            )
            saved_path = await image.save(path=str(temp_dir), full_size=False)
    else:
        saved_path = await image.save(path=str(temp_dir))

    if not saved_path:
        raise ValueError("Failed to save generated image")

    original_path = Path(saved_path)
    data = original_path.read_bytes()
    suffix = original_path.suffix

    if not suffix:
        detected_ext = detect_image_extension(data)
        suffix = detected_ext or (".png" if isinstance(image, GeneratedImage) else ".jpg")

    random_name = f"img_{uuid.uuid4().hex}{suffix}"
    new_path = temp_dir / random_name
    original_path.rename(new_path)

    width, height = extract_image_dimensions(data)
    filename = random_name
    file_hash = hashlib.sha256(data).hexdigest()
    return base64.b64encode(data).decode("ascii"), width, height, filename, file_hash


async def _media_to_local_file(
    media: GeneratedVideo | GeneratedMedia, temp_dir: Path
) -> dict[str, tuple[str, str]]:
    """Persist media and return dict mapping type to (filename, hash)"""
    try:
        saved_paths = await media.save(path=str(temp_dir))
        if not saved_paths:
            logger.warning("No files saved from media object.")
            return {}
    except Exception as e:
        logger.error(f"Failed to save media: {e}")
        return {}

    default_extensions = {
        "video": ".mp4",
        "audio": ".mp3",
        "video_thumbnail": ".jpg",
        "audio_thumbnail": ".jpg",
    }

    results = {}
    path_map = {}

    for mtype, spath in saved_paths.items():
        if not spath:
            continue
        try:
            original_path = Path(spath)
            if not original_path.exists():
                if spath in path_map:
                    results[mtype] = path_map[spath]
                continue

            if spath in path_map:
                results[mtype] = path_map[spath]
                continue

            data = original_path.read_bytes()
            suffix = original_path.suffix or (
                default_extensions.get(mtype) or (".mp4" if "video" in mtype else ".mp3")
            )

            random_name = f"media_{uuid.uuid4().hex}{suffix}"
            new_path = temp_dir / random_name
            original_path.rename(new_path)

            fhash = hashlib.sha256(data).hexdigest()
            results[mtype] = (random_name, fhash)
            path_map[spath] = (random_name, fhash)
        except Exception as e:
            logger.warning(f"Error processing {mtype} at {spath}: {e}")

    return results


def _create_responses_standard_payload(
    response_id: str,
    created_time: int,
    model_name: str,
    detected_tool_calls: list[AppToolCall] | None,
    image_call_items: list[ImageGenerationCall],
    response_contents: list[ResponseOutputContent],
    usage: ResponseUsage,
    request: ResponseCreateRequest,
    structured_requirement: StructuredOutputRequirement | None = None,
    full_thoughts: str | None = None,
    message_id: str | None = None,
    reason_id: str | None = None,
) -> ResponseCreateResponse:
    """Unified factory for building ResponseCreateResponse objects."""
    message_id = message_id or f"msg_{uuid.uuid4().hex[:24]}"
    reason_id = reason_id or f"rs_{uuid.uuid4().hex[:24]}"
    now_ts = int(datetime.now(tz=UTC).timestamp())

    output_items: list[Any] = []
    if full_thoughts:
        output_items.append(
            ResponseReasoningItem(
                id=reason_id,
                type="reasoning",
                status="completed",
                summary=[SummaryTextContent(type="summary_text", text=full_thoughts)],
            )
        )

    if response_contents or not (detected_tool_calls or image_call_items):
        output_items.append(
            ResponseOutputMessage(
                id=message_id,
                type="message",
                status="completed",
                role="assistant",
                content=response_contents,
            )
        )

    if detected_tool_calls:
        output_items.extend(
            [
                ResponseFunctionToolCall(
                    id=call.id,
                    call_id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                    status="completed",
                )
                for call in detected_tool_calls
            ]
        )

    output_items.extend(image_call_items)

    text_config = request.text.model_copy(deep=True) if request.text else ResponseTextConfig()
    if request.response_format and request.response_format.get("type") == "json_schema":
        legacy_config = request.response_format.get("json_schema") or {}
        text_config.format = ResponseFormatTextJSONSchemaConfig(
            name=legacy_config.get("name"),
            schema=legacy_config.get("schema"),
            description=legacy_config.get("description"),
        )

    # Report the enforcement applied, not the value the request carried: a client that reads
    # `strict: true` back may skip its own validation.
    if isinstance(text_config.format, ResponseFormatTextJSONSchemaConfig):
        text_config.format.strict = bool(structured_requirement and structured_requirement.strict)

    return ResponseCreateResponse(
        id=response_id,
        object="response",
        created_at=created_time,
        completed_at=now_ts,
        model=model_name,
        output=output_items,
        status="completed",
        usage=usage,
        metadata=request.metadata or {},
        tools=serialize_tools_for_response(request.tools),
        tool_choice=serialize_tool_choice_for_response(request.tool_choice),
        text=text_config,
    )


def _create_chat_completion_standard_payload(
    completion_id: str,
    created_time: int,
    model_name: str,
    visible_output: str | None,
    tool_calls: list[AppToolCall] | None,
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"],
    usage: dict,
    reasoning_content: str | None = None,
) -> ChatCompletionResponse:
    """Unified factory for building Chat Completion response objects."""
    tc_converted = None
    if tool_calls:
        tc_converted = [
            ChatCompletionMessageToolCall(
                id=tc.id,
                type="function",
                function=FunctionCall(name=tc.function.name, arguments=tc.function.arguments),
            )
            for tc in tool_calls
        ]

    message = ChatCompletionMessage(
        role="assistant",
        content=visible_output or None,
        tool_calls=tc_converted,
        reasoning_content=reasoning_content or None,
    )

    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created_time,
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason=finish_reason,
            )
        ],
        usage=CompletionUsage(**usage),
    )


def _persist_conversation(
    db: LMDBConversationStore,
    model_name: str,
    client: GeminiClientWrapper,
    metadata: list[str | None],
    messages: list[AppMessage],
    storage_output: str | None,
    tool_calls: list[AppToolCall] | None,
) -> str | None:
    """Unified logic to save conversation history to LMDB."""
    # This turn is now the last chat this client opened; any window it replaced is closed. Set
    # before the store, so a persistence failure cannot leave a closed window looking reusable,
    # and in every mode, since an expired cookie can turn a client ephemeral without warning.
    client.latest_chat_cid = _cid_of(metadata)
    chat_scope = client.chat_scope(_use_temporary_chat_mode())

    try:
        current_assistant_message = AppMessage(
            role="assistant",
            content=storage_output or None,
            tool_calls=tool_calls or None,
            reasoning_content=None,
        )
        full_history = [*messages, current_assistant_message]

        # An ephemeral chat is worth storing like any other while its window is live, tagged with
        # that window so a later session cannot mistake it for one of its own.
        db.store(
            client_id=client.id,
            model=model_name,
            messages=full_history,
            metadata=metadata,
            chat_scope=chat_scope,
        )
        logger.debug("Conversation saved to LMDB.")
        return "success"
    except Exception as e:
        logger.warning(f"Failed to save {len(messages) + 1} messages to LMDB: {e}")
        return None


def _build_structured_requirement(
    response_format: dict[str, Any] | None,
) -> StructuredOutputRequirement | None:
    """Translate an OpenAI-style response_format into fenced-JSON prompt instructions.

    `json_schema` becomes an enforced requirement, `json_object` a best-effort one carrying an
    empty schema, and every other type - including the `text` default - is dropped, since a mode
    this wrapper cannot represent must not block the rest of the request.

    `strict` is read from the payload and defaults to False, matching OpenAI's own default for
    Chat Completions. It has teeth here: Gemini Web has no constrained decoding, so a strict
    requirement that the model misses costs the caller the whole answer. Callers that ask for
    `strict: true` have opted into that; callers that say nothing get the reply as text.
    A schema that is well-formed JSON but not valid JSON Schema is asked for without being
    enforced, rather than failing the request; only a malformed `response_format` is rejected.
    """
    if not response_format or not isinstance(response_format, dict):
        return None

    format_type = response_format.get("type")
    if format_type == "json_object":
        return StructuredOutputRequirement(
            schema_name="response",
            schema={},
            instruction=STRUCTURED_JSON_WRAP_HINT,
            raw_format=response_format,
            strict=False,
        )

    if format_type != "json_schema":
        logger.debug(f"Ignoring response_format type unsupported by Gemini Web: {format_type!r}")
        return None

    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        raise ValueError("response format must contain a json_schema object")

    schema = json_schema.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("json_schema must contain a schema object")

    schema_name = json_schema.get("name") or "response"
    requested_strict = json_schema.get("strict", False)
    if not isinstance(requested_strict, bool):
        raise ValueError("json_schema.strict must be a boolean")

    enforced_schema, strict = schema, requested_strict
    try:
        validate_json_schema(schema)
    except ValueError as exc:
        # Same stance as the Gemini surface: a schema we cannot evaluate is still shown to the
        # model, it just cannot be used to judge the reply. Failing the request instead would
        # lose a usable answer over a gap on our side.
        logger.warning(
            f"Asking for schema {schema_name!r} without enforcing it, it is not representable "
            f"as JSON Schema: {exc}"
        )
        enforced_schema, strict = {}, False

    pretty_schema = orjson.dumps(schema, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    instruction_parts = [
        STRUCTURED_JSON_WRAP_HINT,
        SCHEMA_ADHERENCE_PROMPT,
        f'Schema name: "{schema_name}"',
        "JSON Schema:",
        pretty_schema,
    ]
    # Prompted from what the caller asked for, enforced from what we can actually check.
    if requested_strict:
        instruction_parts.insert(1, STRICT_SCHEMA_ADHERENCE_PROMPT)

    instruction = "\n\n".join(instruction_parts)
    return StructuredOutputRequirement(
        schema_name=schema_name,
        schema=enforced_schema,
        instruction=instruction,
        raw_format=response_format,
        strict=strict,
    )


def _responses_response_format(request: ResponseCreateRequest) -> dict[str, Any] | None:
    """Reduce Responses `text.format` and the legacy `response_format` to one format dict."""
    text_format = request.text.format if request.text is not None else None

    if request.response_format is not None:
        # The legacy project extension only loses to a `text.format` that actually asks for a
        # format; a default or plain-text block alongside it is not a conflict.
        if isinstance(text_format, ResponseFormatText) or text_format is None:
            return request.response_format
        raise ValueError("Use either text.format or response_format, not both")

    if isinstance(text_format, ResponseFormatJSONObject):
        return {"type": "json_object"}
    if not isinstance(text_format, ResponseFormatTextJSONSchemaConfig):
        return None
    if not isinstance(text_format.schema_, dict):
        raise ValueError("text.format.schema is required for json_schema output")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": text_format.name or "response",
            "schema": text_format.schema_,
            "description": text_format.description,
            # Same default as Chat Completions and as OpenAI: an omitted `strict` is best-effort,
            # so one schema cannot hard-fail on one surface and degrade on the other.
            "strict": text_format.strict if text_format.strict is not None else False,
        },
    }


def _log_ignored_openai_options(
    request: ChatCompletionRequest | ResponseCreateRequest,
) -> None:
    """Debug-log generation controls that were accepted but cannot reach Gemini Web."""
    control_fields = (
        ("temperature", "top_p", "max_completion_tokens", "parallel_tool_calls")
        if isinstance(request, ChatCompletionRequest)
        else ("temperature", "top_p", "max_output_tokens", "parallel_tool_calls")
    )
    if ignored := {name for name in control_fields if name in request.model_fields_set}:
        logger.debug(
            "Ignoring option(s) unsupported by the Gemini Web upstream: "
            f"{', '.join(sorted(ignored))}"
        )


def _tool_choice_failure(
    tool_choice: Any,
    tool_calls: list[AppToolCall],
    *,
    has_images: bool = False,
    has_image_tool: bool = False,
) -> str | None:
    """Describe how the model failed a forced tool_choice, or None if it honored it.

    Gemini Web has no constrained decoding, so a forced choice is only ever a prompt
    instruction; this is the check that the instruction actually took.

    An image only discharges `required` when an image-generation tool was declared, so that an
    image Gemini volunteers on its own cannot stand in for the function call that was forced.
    """
    if tool_choice == "required" and not tool_calls and not (has_images and has_image_tool):
        return "The model did not return a required tool result"
    if (
        target_name := (
            tool_choice.function.name
            if isinstance(tool_choice, ChatCompletionNamedToolChoice)
            else (tool_choice.name if isinstance(tool_choice, ToolChoiceFunction) else None)
        )
    ) and all(call.function.name != target_name for call in tool_calls):
        return f"The model did not call the required function {target_name!r}"
    if isinstance(tool_choice, ToolChoiceTypes) and not has_images:
        return "The model did not return a required image generation result"
    return None


def _tool_choice_declaration_error(
    function_names: set[str],
    has_image_tool: bool,
    tool_choice: Any,
) -> str | None:
    """Reject forced choices that do not name a declared compatible tool."""
    if tool_choice == "required" and not function_names and not has_image_tool:
        return "tool_choice='required' requires at least one tool"
    if isinstance(tool_choice, ChatCompletionNamedToolChoice):
        target_name = tool_choice.function.name
        if target_name not in function_names:
            return f"tool_choice names undeclared function {target_name!r}"
    if isinstance(tool_choice, ToolChoiceFunction) and tool_choice.name not in function_names:
        return f"tool_choice names undeclared function {tool_choice.name!r}"
    if isinstance(tool_choice, ToolChoiceTypes) and not has_image_tool:
        return "tool_choice='image_generation' requires an image_generation tool"
    return None


def _validate_responses_input(items: Any) -> str | None:
    """Describe why an input item is unusable, or None if every part can be represented.

    Only content is judged here: a reference this wrapper cannot resolve, or a media part whose
    source is missing or ambiguous. Dropping either silently would change what the model sees.
    """
    if isinstance(items, str):
        return None
    for item in items:
        parts = (
            item.output if isinstance(item, FunctionCallOutput) else getattr(item, "content", None)
        )
        if not isinstance(parts, list):
            continue
        for part in parts:
            if getattr(part, "file_id", None):
                return "file_id inputs are not supported; use file_url or inline Base64 data"
            if getattr(part, "type", None) == "input_file":
                sources = [
                    getattr(part, "file_url", None),
                    getattr(part, "file_data", None),
                ]
                if sum(value is not None for value in sources) != 1:
                    return "input_file must contain exactly one of file_url or file_data"
            if getattr(part, "type", None) == "input_image" and not getattr(
                part, "image_url", None
            ):
                return "input_image must contain image_url"
    return None


def _prepare_messages_for_model(
    source_messages: list[AppMessage],
    tools: Sequence[Any] | None,
    tool_choice: Literal["none", "auto", "required"]
    | ChatCompletionNamedToolChoice
    | ToolChoiceFunction
    | ToolChoiceTypes
    | None,
    extra_instructions: list[str] | None = None,
    inject_system_defaults: bool = True,
) -> list[AppMessage]:
    """Return a copy of messages enriched with tool instructions when needed."""
    prepared = [msg.model_copy(deep=True) for msg in source_messages]

    # Resolve tool names for 'tool' messages by looking back at previous assistant tool calls
    tool_id_to_name = {}
    for msg in prepared:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_id_to_name[tc.id] = tc.function.name

    for msg in prepared:
        if msg.role == "tool" and not msg.name and msg.tool_call_id:
            msg.name = tool_id_to_name.get(msg.tool_call_id)

    instructions: list[str] = []
    tool_prompt_injected = False
    if inject_system_defaults and tools and (tool_prompt := build_tool_prompt(tools, tool_choice)):
        instructions.append(tool_prompt)
        tool_prompt_injected = True

    if extra_instructions:
        instructions.extend(instr for instr in extra_instructions if instr)
        logger.debug(
            f"Applied {len(extra_instructions)} extra instructions for tool/structured output."
        )

    if not instructions:
        if tools and tool_choice != "none" and not tool_prompt_injected:
            append_tool_hint_to_last_user_message(prepared)
        return prepared

    combined_instructions = "\n\n".join(instructions)
    if prepared and prepared[0].role == "system" and isinstance(prepared[0].content, str):
        existing = prepared[0].content or ""
        if combined_instructions not in existing:
            separator = "\n\n" if existing else ""
            prepared[0].content = f"{existing}{separator}{combined_instructions}"
    else:
        prepared.insert(0, AppMessage(role="system", content=combined_instructions))

    if tools and tool_choice != "none" and not tool_prompt_injected:
        append_tool_hint_to_last_user_message(prepared)

    return prepared


def _convert_responses_to_app_messages(
    items: Any,
) -> list[AppMessage]:
    """Convert Responses API input items into internal AppMessage objects, skipping incomplete tool calls."""
    messages: list[AppMessage] = []

    if isinstance(items, str):
        messages.append(AppMessage(role="user", content=items))
        logger.debug("Normalized Responses input: single string message.")
        return messages

    for item in items:
        if isinstance(item, (ResponseInputMessage, ResponseOutputMessage)):
            raw_role = getattr(item, "role", "user")
            role = normalize_app_message_role(raw_role)

            content = item.content
            if isinstance(content, str):
                messages.append(AppMessage(role=role, content=content))
            else:
                converted: list[AppContentItem] = []
                reasoning_parts: list[str] = []
                for part in content:
                    if part.type in ("input_text", "output_text"):
                        if text_value := getattr(part, "text", "") or "":
                            converted.append(AppContentItem(type="text", text=text_value))
                    elif part.type == "reasoning_text":
                        if text_value := getattr(part, "text", "") or "":
                            reasoning_parts.append(text_value)
                    elif part.type == "input_image":
                        if image_url := getattr(part, "image_url", None):
                            converted.append(AppContentItem(type="image_url", url=image_url))
                    elif part.type == "input_file":
                        file_url = getattr(part, "file_url", None)
                        file_data = getattr(part, "file_data", None)
                        if file_url or file_data:
                            converted.append(
                                AppContentItem(
                                    type="file",
                                    url=file_url,
                                    file_data=file_data,
                                    filename=getattr(part, "filename", None),
                                )
                            )
                reasoning_val = "\n\n".join(reasoning_parts) if reasoning_parts else None
                messages.append(
                    AppMessage(
                        role=role,
                        content=converted or None,
                        reasoning_content=reasoning_val,
                    )
                )

        elif isinstance(item, ResponseFunctionToolCall):
            call_id = item.call_id or item.id
            if not call_id or not item.name or item.arguments is None:
                logger.warning(
                    f"Skipping incomplete function_call input item: {reprlib.repr(item.model_dump(mode='json'))}"
                )
                continue
            messages.append(
                AppMessage(
                    role="assistant",
                    tool_calls=[
                        AppToolCall(
                            id=call_id,
                            type="function",
                            function=AppToolCallFunction(name=item.name, arguments=item.arguments),
                        )
                    ],
                )
            )
        elif isinstance(item, FunctionCallOutput):
            output_content: str | list[AppContentItem] | None
            if isinstance(item.output, list):
                converted_output: list[AppContentItem] = []
                for part in item.output:
                    if part.type == "input_text":
                        converted_output.append(AppContentItem(type="text", text=part.text or ""))
                    elif part.type == "input_image" and part.image_url:
                        converted_output.append(
                            AppContentItem(type="image_url", url=part.image_url)
                        )
                    elif part.type == "input_file":
                        converted_output.append(
                            AppContentItem(
                                type="file",
                                url=part.file_url,
                                file_data=part.file_data,
                                filename=part.filename,
                            )
                        )
                output_content = converted_output or None
            else:
                output_content = item.output
            messages.append(
                AppMessage(
                    role="tool",
                    tool_call_id=item.call_id,
                    content=output_content,
                )
            )
        elif isinstance(item, ResponseReasoningItem):
            reasoning_val = None
            if item.content:
                reasoning_val = "\n\n".join(x.text for x in item.content if x.text)
            messages.append(
                AppMessage(
                    role="assistant",
                    reasoning_content=reasoning_val,
                )
            )
        elif isinstance(item, ImageGenerationCall):
            messages.append(
                AppMessage(
                    role="assistant",
                    content=item.result or None,
                )
            )

        else:
            if hasattr(item, "role"):
                raw_role = getattr(item, "role", "user")
                role = normalize_app_message_role(raw_role)
                messages.append(
                    AppMessage(
                        role=role,
                        content=str(getattr(item, "content", "")),
                    )
                )

    compacted_messages: list[AppMessage] = []
    for msg in messages:
        if not compacted_messages:
            compacted_messages.append(msg)
            continue

        last_msg = compacted_messages[-1]
        if last_msg.role == "assistant" and msg.role == "assistant":
            reasoning_parts = []
            if last_msg.reasoning_content:
                reasoning_parts.append(last_msg.reasoning_content)
            if msg.reasoning_content:
                reasoning_parts.append(msg.reasoning_content)

            merged_content = []
            if isinstance(last_msg.content, str):
                merged_content.append(AppContentItem(type="text", text=last_msg.content))
            elif isinstance(last_msg.content, list):
                merged_content.extend(last_msg.content)

            if isinstance(msg.content, str):
                merged_content.append(AppContentItem(type="text", text=msg.content))
            elif isinstance(msg.content, list):
                merged_content.extend(msg.content)

            merged_tools = []
            if last_msg.tool_calls:
                merged_tools.extend(last_msg.tool_calls)
            if msg.tool_calls:
                merged_tools.extend(msg.tool_calls)

            last_msg.reasoning_content = "\n\n".join(reasoning_parts) if reasoning_parts else None
            last_msg.content = merged_content or None
            last_msg.tool_calls = merged_tools or None
        else:
            compacted_messages.append(msg)

    logger.debug(f"Normalized Responses input: {len(compacted_messages)} message items.")
    return compacted_messages


def _convert_instructions_to_app_messages(
    instructions: str | list[ResponseInputMessage] | None,
) -> list[AppMessage]:
    """Normalize instructions payload into AppMessage objects."""
    if not instructions:
        return []

    if isinstance(instructions, str):
        return [AppMessage(role="system", content=instructions)]

    instruction_messages: list[AppMessage] = []
    for instruction in instructions:
        if instruction.type and instruction.type != "message":
            continue

        role = normalize_app_message_role(instruction.role)

        content = instruction.content
        if isinstance(content, str):
            instruction_messages.append(AppMessage(role=role, content=content))
        else:
            converted: list[AppContentItem] = []
            for part in content:
                if part.type in ("input_text", "output_text"):
                    if text_value := getattr(part, "text", "") or "":
                        converted.append(AppContentItem(type="text", text=text_value))
                elif part.type == "input_image":
                    if image_url := getattr(part, "image_url", None):
                        converted.append(AppContentItem(type="image_url", url=image_url))
                elif part.type == "input_file":
                    file_url = getattr(part, "file_url", None)
                    file_data = getattr(part, "file_data", None)
                    if file_url or file_data:
                        converted.append(
                            AppContentItem(
                                type="file",
                                url=file_url,
                                file_data=file_data,
                                filename=getattr(part, "filename", None),
                            )
                        )
            instruction_messages.append(AppMessage(role=role, content=converted or None))

    return instruction_messages


def _resolve_model_name(pool: GeminiClientPool, name: str) -> str:
    """Canonical name of the model a request asked for, resolved against a client's registry.

    Names, aliases and hex ids all resolve, and every client discovers its own models, so nothing
    here has to be configured or kept up to date. Resolution is canonical on purpose: two aliases
    of one model must reach the same conversation records.

    The name, not the resolved object, is what callers pass on - each client re-resolves it in its
    own registry at send time, where the model header carries that account's tier. Preferring an
    authenticated client keeps a guest, whose registry marks everything but the default
    unavailable, from narrowing what the whole pool accepts.
    """
    # A registry outlives an auto-close, so an idle client still resolves; one that never
    # initialized has nothing to offer.
    clients = sorted(pool.clients, key=lambda c: (c.is_guest(), not c.running()))
    for client in clients:
        if not client.list_models():
            continue

        try:
            return client.resolve_model(name).model_name
        except ValueError:
            continue

    raise ValueError(f"Model '{name}' is not available on any Gemini client.")


async def _build_available_models(pool: GeminiClientPool) -> list[ModelData]:
    """Build the available model list from the models the clients discovered."""
    now = int(datetime.now(tz=UTC).timestamp())
    models_data = []
    seen_model_ids = set()

    for client in pool.clients:
        if client_models := client.list_models():
            for model in client_models:
                # A guest session registers the models it can see but may only use the
                # default one; advertising the rest would promise what it cannot serve.
                if not model.is_available:
                    continue

                model_id = model.model_name or model.model_id
                if model_id and model_id not in seen_model_ids:
                    models_data.append(
                        ModelData(
                            id=model_id,
                            created=now,
                            owned_by="google",
                        )
                    )
                    seen_model_ids.add(model_id)

    return models_data


async def refresh_available_models_cache(pool: GeminiClientPool) -> list[ModelData]:
    """Refresh and return the cached model list while clients are available."""
    global _AVAILABLE_MODELS_CACHE

    async with _AVAILABLE_MODELS_CACHE_LOCK:
        models = await _build_available_models(pool)
        _AVAILABLE_MODELS_CACHE = models
        logger.info(f"Cached {len(models)} available model(s).")
        return list(models)


async def _get_available_models(pool: GeminiClientPool) -> list[ModelData]:
    """Return cached available models, populating the cache if it has not been warmed yet."""
    if _AVAILABLE_MODELS_CACHE is not None:
        return list(_AVAILABLE_MODELS_CACHE)

    return await refresh_available_models_cache(pool)


def _cid_of(metadata: list[str | None] | None) -> str | None:
    """Chat id from a metadata list, which stores it at index 0."""
    return metadata[0] if metadata else None


def _is_reusable_chat(
    conv: ConversationInStore, client: GeminiClientWrapper, temporary: bool
) -> bool:
    """Whether a stored chat can still be continued on this client.

    A normal chat lives in the account's history and stays continuable indefinitely, so only the
    ephemeral records need checking: temporary-mode chats, and everything a guest session opened.
    Those survive only as the one open window of the session that created them, and replaying a
    closed one makes Google answer from a fresh chat without the earlier context - no error, just
    silent loss. So an ephemeral record must still carry the client's current scope, which no
    longer matches once that session is gone (reinitialized, or downgraded to guest by expired
    cookies, or authenticated again afterwards), and its cid must be the last one this client
    opened, since a newer conversation has closed anything older.

    A `latest_chat_cid` match is necessary but not proof: the cid's kind is never verified.
    """
    scope = client.chat_scope(temporary)
    if conv.chat_scope is None and scope is None:
        return True

    if conv.chat_scope != scope:
        logger.debug(
            f"Stored chat scope {conv.chat_scope!r} no longer matches client {client.id} "
            f"({scope!r}); the window behind it is gone, so replaying the full history in a "
            "fresh conversation."
        )
        return False

    latest = client.latest_chat_cid
    if not latest:
        logger.debug(f"Client {client.id} has no chat on record; starting a fresh conversation.")
        return False

    cid = _cid_of(conv.metadata)
    if cid and cid == latest:
        return True

    logger.debug(
        f"Stored chat {cid!r} is not the latest ({latest!r}) on client {client.id}; a newer "
        "conversation has closed it, so replaying the full history in a fresh conversation."
    )
    return False


async def _find_reusable_session(
    db: LMDBConversationStore,
    pool: GeminiClientPool,
    resolved_model: str,
    messages: list[AppMessage],
    temporary: bool = False,
    require_account: bool = False,
) -> tuple[
    ChatSession | None, GeminiClientWrapper | None, list[AppMessage], ConversationInStore | None
]:
    """Find an existing chat session matching the longest suitable history prefix."""
    if len(messages) < 2:
        return None, None, messages, None

    search_end = len(messages)
    while search_end >= 2:
        search_history = messages[:search_end]
        if search_history[-1].role in {"assistant", "system", "tool"}:
            try:
                if conv := db.find(resolved_model, search_history):
                    client = await pool.acquire(conv.client_id)
                    if require_account and client.is_guest():
                        # Continuing here would fail on the upload; a fresh chat on an
                        # authenticated client can still serve the request.
                        logger.debug(
                            f"Client {client.id} owns the match but is a guest session and this "
                            "request needs an upload; starting a fresh conversation."
                        )
                        break
                    # Checked after acquiring: acquire may restart a closed client, which rerolls
                    # the scope and clears the tracked cid, invalidating any ephemeral window.
                    if not _is_reusable_chat(conv, client, temporary):
                        # Every prefix of one conversation carries the same cid and scope, so if
                        # the longest match is not the live window, no shorter one will be either.
                        break
                    session = client.start_chat(
                        metadata=conv.metadata, model=client.usable_model(resolved_model)
                    )
                    remain = messages[search_end:]
                    logger.debug(
                        f"Match found at prefix length {search_end}/{len(messages)}. Client: {conv.client_id}"
                    )
                    return session, client, remain, conv
            except Exception as e:
                logger.warning(
                    f"Error checking LMDB for reusable session at length {search_end}: {e}"
                )
                break
        search_end -= 1

    logger.debug(f"No reusable session found for {len(messages)} messages.")
    return None, None, messages, None


def _use_temporary_chat_mode() -> bool:
    """Whether requests should be sent through Google's temporary chat mode."""
    return g_config.gemini.chat_mode == ChatMode.TEMPORARY


def _effective_max_chars_per_request(temporary: bool) -> int:
    """Return the payload guardrail for the active chat mode."""
    return TEMPORARY_MAX_CHARS_PER_REQUEST if temporary else MAX_CHARS_PER_REQUEST


def _requires_upload(messages: list[AppMessage], temporary: bool) -> bool:
    """Whether serving these messages needs a file upload, which a guest session cannot do.

    Attachments do; so does input long enough to be sent as `message.txt`. The length is measured
    before the prompt is assembled, so it slightly underestimates and only steers client choice -
    `_send_with_split` makes the real call.
    """
    total = 0
    for message in messages:
        if isinstance(message.content, str):
            total += len(message.content)
        elif isinstance(message.content, list):
            for item in message.content:
                if item.type != "text":
                    return True
                total += len(item.text or "")

    return total > _effective_max_chars_per_request(temporary)


def _can_upload(session: ChatSession) -> bool:
    """Whether the client behind this session may attach files."""
    client = session.geminiclient
    return client.can_upload() if isinstance(client, GeminiClientWrapper) else True


async def _send_with_split(
    session: ChatSession,
    text: str,
    files: list[Any] | None = None,
    stream: bool = False,
    temporary: bool = False,
) -> AsyncGenerator[ModelOutput] | ModelOutput:
    """Send text to Gemini with configured generation options, using an attachment if too long."""
    limit = _effective_max_chars_per_request(temporary)
    if len(text) <= limit:
        try:
            if stream:
                return session.send_message_stream(
                    text,
                    files=files,
                    temporary=temporary,
                    extended_thinking=g_config.gemini.extended_thinking,
                )
            return await session.send_message(
                text,
                files=files,
                temporary=temporary,
                extended_thinking=g_config.gemini.extended_thinking,
            )
        except Exception as e:
            logger.error(f"Error sending message to Gemini: {e}")
            raise

    if not _can_upload(session):
        # Only reachable once every client is a guest, since routing prefers an authenticated one.
        raise RuntimeError(
            f"Message length ({len(text)}) exceeds limit ({limit}) and would have to be sent as "
            "an attachment, which a guest session cannot upload. Refresh the client cookies or "
            "shorten the request."
        )

    logger.info(
        f"Message length ({len(text)}) exceeds limit ({limit}). Converting text to file attachment."
    )
    file_obj = io.BytesIO(text.encode("utf-8"))
    file_obj.name = "message.txt"
    try:
        final_files: list[Any] = list(files) if files else []
        final_files.insert(0, file_obj)
        instruction = (
            "The user's input exceeds the character limit and is provided in the attached file `message.txt`.\n\n"
            "**System Instruction:**\n"
            "1. Read the content of `message.txt`.\n"
            "2. Treat that content as the **primary** user prompt for this turn.\n"
            "3. Execute the instructions or answer the questions found *inside* that file immediately.\n"
        )
        if stream:
            return session.send_message_stream(
                instruction,
                files=final_files,
                temporary=temporary,
                extended_thinking=g_config.gemini.extended_thinking,
            )
        return await session.send_message(
            instruction,
            files=final_files,
            temporary=temporary,
            extended_thinking=g_config.gemini.extended_thinking,
        )
    except Exception as e:
        logger.error(f"Error sending large text as file to Gemini: {e}")
        raise


async def _restream(
    first: ModelOutput, rest: AsyncGenerator[ModelOutput]
) -> AsyncGenerator[ModelOutput]:
    """Re-emit an already-consumed first chunk, then delegate to the remainder."""
    yield first
    async for chunk in rest:
        yield chunk


async def _send_and_await_first_chunk(
    session: ChatSession,
    text: str,
    *,
    files: list[Any],
    stream: bool,
    temporary: bool,
) -> AsyncGenerator[ModelOutput] | ModelOutput:
    """Send to Gemini, pulling the first streamed chunk so start-of-stream errors surface here.

    `send_message_stream` is an async generator function: calling it runs none of its body, so
    without this the request would not reach Google until the caller iterates - by which point
    the HTTP response has already been committed and a failure can no longer be recovered.
    """
    output = await _send_with_split(session, text, files=files, stream=stream, temporary=temporary)
    if not stream:
        return output

    generator = cast(AsyncGenerator[ModelOutput], output)
    try:
        first = await anext(generator)
    except StopAsyncIteration:
        return generator  # already exhausted, so iterating it again simply yields nothing
    except BaseException:
        await generator.aclose()
        raise
    return _restream(first, generator)


async def _send_with_internal_fallback(
    *,
    pool: GeminiClientPool,
    db: LMDBConversationStore,
    resolved_model: str,
    session: ChatSession,
    client: GeminiClientWrapper,
    current_input: str,
    files: list[Any],
    full_prepared_messages: list[AppMessage],
    stored_conversation: ConversationInStore | None,
    tmp_dir: Path,
    stream: bool,
    temporary: bool,
) -> tuple[AsyncGenerator[ModelOutput] | ModelOutput, ChatSession, GeminiClientWrapper]:
    """Send the request, replaying the full history in a fresh chat if reused metadata is dead.

    Streaming is recovered as well as non-streaming: the first chunk is pulled here, which is
    where Google reports a rejected chat, so the retry happens before any response is committed
    to the client. The cost is that response headers wait for Google's first chunk.
    """
    try:
        output = await _send_and_await_first_chunk(
            session, current_input, files=files, stream=stream, temporary=temporary
        )
        return output, session, client
    except ModelInvalidError:
        if stored_conversation is None:
            raise

        # Drop the dead metadata so the next request does not rediscover and re-fail on it.
        try:
            if db.evict(stored_conversation):
                logger.info("Evicted stale conversation metadata after Google rejected it.")
        except Exception as evict_exc:
            logger.warning(f"Failed to evict stale conversation metadata: {evict_exc}")

        logger.warning(
            "Metadata-backed chat reuse failed; retrying with internal history replay in a fresh chat."
        )
        fallback_input, fallback_files = await GeminiClientWrapper.process_conversation(
            full_prepared_messages, tmp_dir
        )
        # Built before acquiring, so a replay that needs an upload is not handed to a guest.
        fallback_client = await pool.acquire(
            require_account=bool(fallback_files)
            or len(fallback_input) > _effective_max_chars_per_request(temporary)
        )
        fallback_session = fallback_client.start_chat(
            model=fallback_client.usable_model(resolved_model)
        )
        # Keep the caller's streaming mode: the endpoints reject a ModelOutput when the client
        # asked for a stream, so downgrading here would turn a recovery into a 502.
        output = await _send_and_await_first_chunk(
            fallback_session,
            fallback_input,
            files=list(fallback_files),
            stream=stream,
            temporary=temporary,
        )
        return output, fallback_session, fallback_client


class StreamingOutputFilter:
    """
    Filter to suppress technical protocol markers, tool calls, and system hints from the stream.
    Uses a stack-based state machine to handle nested fragmented markers.
    """

    def __init__(self):
        self.buffer = ""
        self.stack = ["NORMAL"]
        self.current_role = ""

    @property
    def state(self):
        return self.stack[-1]

    def _is_outputting(self) -> bool:
        """Determines if the current state allows yielding text to the stream."""
        if self.state == "POST_BLOCK":
            return False
        return self.state == "NORMAL" or (self.state == "IN_BLOCK" and self.current_role != "tool")

    def process(self, chunk: str) -> str:
        self.buffer += chunk
        output = []

        while self.buffer:
            if self.state == "IN_TAG_HEADER":
                nl_idx = self.buffer.find("\n")
                if nl_idx == -1:
                    break

                self.current_role = self.buffer[:nl_idx].strip().lower()
                self.buffer = self.buffer[nl_idx + 1 :]
                self.stack[-1] = "IN_BLOCK"
                continue
            if self.state == "POST_BLOCK":
                stripped = self.buffer.lstrip()
                if not stripped:
                    break
                self.buffer = stripped
                self.stack[-1] = "NORMAL"

            match = STREAM_MASTER_RE.search(self.buffer)
            if not match:
                if tail_match := STREAM_TAIL_RE.search(self.buffer):
                    yield_len = len(self.buffer) - len(tail_match.group(0))
                    if yield_len > 0:
                        if self._is_outputting():
                            output.append(self.buffer[:yield_len])
                        self.buffer = self.buffer[yield_len:]
                else:
                    if self._is_outputting():
                        output.append(self.buffer)
                    self.buffer = ""
                break

            start, end = match.span()
            matched_group = match.lastgroup
            pre_text = self.buffer[:start]

            if self._is_outputting():
                output.append(pre_text)

            if matched_group and matched_group.endswith("_START"):
                m_type = matched_group.split("_")[0]
                if m_type == "TAG":
                    self.stack.append("IN_TAG_HEADER")
                else:
                    self.stack.append(f"IN_{m_type}")
            elif matched_group in ("PROTOCOL_EXIT", "TAG_EXIT", "HINT_EXIT"):
                if len(self.stack) > 1:
                    self.stack.pop()
                else:
                    self.stack = ["NORMAL"]

                if self.state == "NORMAL" and matched_group in (
                    "PROTOCOL_EXIT",
                    "HINT_EXIT",
                ):
                    self.stack[-1] = "POST_BLOCK"

                if self.state in ("NORMAL", "POST_BLOCK"):
                    self.current_role = ""

            self.buffer = self.buffer[end:]

        return "".join(output)

    def flush(self) -> str:
        """Release remaining buffer content and perform final cleanup at stream end."""
        res = ""
        if self._is_outputting():
            res = self.buffer
            if tail_match := STREAM_FLUSH_TAIL_RE.search(res):
                res = res[: -len(tail_match.group(0))]

        self.buffer = ""
        self.stack = ["NORMAL"]
        self.current_role = ""
        return strip_system_hints(res)


# --- Media Processing Helpers ---


async def _process_image_item(image: Image) -> ProcessedImageResult | None:
    """Process an image item by converting it to base64 and returning a typed image result tuple."""
    try:
        media_store = get_media_store_dir()
        return "image", image, await _image_to_base64(image, media_store)
    except Exception as exc:
        logger.warning(f"Background image processing failed: {exc}")
        return None


async def _process_media_item(
    media_item: GeneratedVideo | GeneratedMedia,
) -> ProcessedMediaResult | None:
    """Process a media item by saving it to local files and returning a typed media result tuple."""
    try:
        media_store = get_media_store_dir()
        return "media", media_item, await _media_to_local_file(media_item, media_store)
    except Exception as exc:
        logger.warning(f"Background media processing failed: {exc}")
        return None


# --- Response Builders & Streaming ---


def _sse_error(
    message: str, error_type: str, param: str | None = None, code: str | None = None
) -> str:
    """Render a Chat Completions SSE error frame followed by the stream terminator.

    The status line is already committed by the time these fire, so the terminator is the only
    way left to tell a client the stream ended on purpose rather than being cut off.
    """
    payload = {"error": {"message": message, "type": error_type, "param": param, "code": code}}
    return f"data: {orjson.dumps(payload).decode('utf-8')}\n\ndata: [DONE]\n\n"


def _create_real_streaming_response(
    resp_or_stream: AsyncGenerator[ModelOutput] | ModelOutput,
    completion_id: str,
    created_time: int,
    model_name: str,
    messages: list[AppMessage],
    db: LMDBConversationStore,
    resolved_model: str,
    client_wrapper: GeminiClientWrapper,
    session: ChatSession,
    base_url: str,
    structured_requirement: StructuredOutputRequirement | None = None,
    tool_choice: Any = None,
) -> StreamingResponse:
    """
    Create a real-time streaming response.
    Reconciles manual delta accumulation with the model's final authoritative state.
    Emits typed image and media results as incremental markdown deltas.
    """

    async def generate_stream():
        full_text = ""
        full_thoughts = ""
        has_started = False
        last_output: ModelOutput | None = None
        suppressor = StreamingOutputFilter()

        media_tasks = []
        seen_media_urls = set()
        seen_image_urls = set()

        async def _make_async_gen(item: ModelOutput) -> AsyncGenerator[ModelOutput]:
            yield item

        async def discard_media_tasks() -> None:
            """Cancel and reap media downloads on a path that will never publish their results.

            These tasks are spawned eagerly while chunks arrive. Abandoning them on an early
            return leaves them writing files nothing references and, on failure, raises
            `Task exception was never retrieved` once they are garbage collected.
            """
            if not media_tasks:
                return
            for task in media_tasks:
                task.cancel()
            await asyncio.gather(*media_tasks, return_exceptions=True)
            media_tasks.clear()

        def make_chunk(delta_content: dict) -> str:
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [{"index": 0, **delta_content}],
            }
            return f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        try:
            try:
                if hasattr(resp_or_stream, "__aiter__"):
                    generator = cast(AsyncGenerator[ModelOutput], resp_or_stream)
                else:
                    generator = _make_async_gen(cast(ModelOutput, resp_or_stream))

                async for chunk in generator:
                    last_output = chunk
                    if not has_started:
                        yield make_chunk(
                            {"delta": {"role": "assistant", "content": ""}, "finish_reason": None}
                        )
                        has_started = True

                    if t_delta := chunk.thoughts_delta:
                        full_thoughts += t_delta
                        yield make_chunk(
                            {"delta": {"reasoning_content": t_delta}, "finish_reason": None}
                        )

                    if text_delta := chunk.text_delta:
                        full_text += text_delta
                        if not structured_requirement and (
                            visible_delta := suppressor.process(text_delta)
                        ):
                            yield make_chunk(
                                {"delta": {"content": visible_delta}, "finish_reason": None}
                            )

                    for img in chunk.images or []:
                        if img.url and img.url not in seen_image_urls:
                            seen_image_urls.add(img.url)
                            media_tasks.append(asyncio.create_task(_process_image_item(img)))

                    m_list = (chunk.videos or []) + (chunk.media or [])
                    for m in m_list:
                        p_url = getattr(m, "url", None) or getattr(m, "mp3_url", None)
                        if p_url and p_url not in seen_media_urls:
                            seen_media_urls.add(p_url)
                            media_tasks.append(asyncio.create_task(_process_media_item(m)))
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                await discard_media_tasks()
                yield _sse_error(f"Streaming error occurred: {e}", "server_error")
                return

            if last_output is not None:
                final_chunk = last_output
                if final_chunk.thoughts:
                    f_thoughts = final_chunk.thoughts
                    ft_len, ct_len = len(f_thoughts), len(full_thoughts)
                    if ft_len > ct_len and f_thoughts.startswith(full_thoughts):
                        drift_t = f_thoughts[ct_len:]
                        full_thoughts = f_thoughts
                        yield make_chunk(
                            {"delta": {"reasoning_content": drift_t}, "finish_reason": None}
                        )

                if final_chunk.text:
                    f_text = final_chunk.text
                    f_len, c_len = len(f_text), len(full_text)
                    if f_len > c_len and f_text.startswith(full_text):
                        drift = f_text[c_len:]
                        full_text = f_text
                        if not structured_requirement and (
                            visible_drift := suppressor.process(drift)
                        ):
                            yield make_chunk(
                                {"delta": {"content": visible_drift}, "finish_reason": None}
                            )

            if not structured_requirement and (remaining_text := suppressor.flush()):
                yield make_chunk({"delta": {"content": remaining_text}, "finish_reason": None})

            try:
                _, visible_output, storage_output, detected_tool_calls = process_llm_output(
                    normalize_llm_text(full_thoughts or ""),
                    normalize_llm_text(full_text or ""),
                    structured_requirement,
                )
            except StructuredOutputValidationError as exc:
                await discard_media_tasks()
                yield _sse_error(
                    str(exc), "invalid_model_output", "response_format", "schema_validation_failed"
                )
                return
            # No `has_images` escape hatch here: Chat Completions has no image-generation tool, so an
            # image Gemini volunteers on its own cannot stand in for a function call that was forced.
            if choice_error := _tool_choice_failure(tool_choice, detected_tool_calls):
                await discard_media_tasks()
                yield _sse_error(
                    choice_error, "invalid_model_output", "tool_choice", "required_tool_missing"
                )
                return
            if structured_requirement and visible_output:
                yield make_chunk({"delta": {"content": visible_output}, "finish_reason": None})

            seen_hashes = {}
            seen_media_hashes = {}
            media_store = get_media_store_dir()

            if media_tasks:
                logger.debug(
                    f"Waiting for {len(media_tasks)} background media tasks with heartbeat..."
                )
                while media_tasks:
                    done, pending = await asyncio.wait(
                        media_tasks, timeout=5.0, return_when=asyncio.FIRST_COMPLETED
                    )
                    media_tasks = list(pending)

                    if not done:
                        yield ": ping\n\n"
                        continue

                    for task in done:
                        res = task.result()
                        if not res:
                            continue

                        rtype, original_item, media_data = res
                        if rtype == "image":
                            _, _, _, fname, fhash = media_data
                            if fhash in seen_hashes:
                                (media_store / fname).unlink(missing_ok=True)
                                fname = seen_hashes[fhash]
                            else:
                                seen_hashes[fhash] = fname

                            img_url = f"{base_url}media/{fname}?token={get_media_token(fname)}"
                            title = getattr(original_item, "title", "Image")
                            md = f"![{title}]({img_url})"
                            storage_output += f"\n\n{md}"
                            yield make_chunk(
                                {"delta": {"content": f"\n\n{md}"}, "finish_reason": None}
                            )

                        elif rtype == "media":
                            m_dict = cast(ProcessedMediaData, media_data)
                            if not m_dict:
                                continue

                            m_urls = {}
                            for mtype, (random_name, fhash) in m_dict.items():
                                if fhash in seen_media_hashes:
                                    existing_name = seen_media_hashes[fhash]
                                    if random_name != existing_name:
                                        (media_store / random_name).unlink(missing_ok=True)
                                    m_urls[mtype] = (
                                        f"{base_url}media/{existing_name}?token={get_media_token(existing_name)}"
                                    )
                                else:
                                    seen_media_hashes[fhash] = random_name
                                    m_urls[mtype] = (
                                        f"{base_url}media/{random_name}?token={get_media_token(random_name)}"
                                    )

                            title = getattr(original_item, "title", "Media")
                            video_url = m_urls.get("video")
                            audio_url = m_urls.get("audio")
                            current_thumb = m_urls.get("video_thumbnail") or m_urls.get(
                                "audio_thumbnail"
                            )

                            md_parts = []
                            if video_url:
                                md_parts.append(
                                    f"[![{title}]({current_thumb})]({video_url})"
                                    if current_thumb
                                    else f"[{title}]({video_url})"
                                )
                            if audio_url:
                                md_parts.append(
                                    f"[![{title} - Audio]({current_thumb})]({audio_url})"
                                    if current_thumb
                                    else f"[{title} - Audio]({audio_url})"
                                )

                            if md_parts:
                                md = "\n\n".join(md_parts)
                                storage_output += f"\n\n{md}"
                                yield make_chunk(
                                    {"delta": {"content": f"\n\n{md}"}, "finish_reason": None}
                                )

            if detected_tool_calls:
                for idx, call in enumerate(detected_tool_calls):
                    tc_dict = {
                        "index": idx,
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }

                    yield make_chunk(
                        {
                            "delta": {
                                "tool_calls": [tc_dict],
                            },
                            "finish_reason": None,
                        }
                    )

            p_tok, c_tok, t_tok, r_tok = calculate_usage(
                messages, storage_output, detected_tool_calls, full_thoughts
            )
            usage = CompletionUsage(
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=t_tok,
                completion_tokens_details={"reasoning_tokens": r_tok},
            )
            _persist_conversation(
                db,
                resolved_model,
                client_wrapper,
                session.metadata,
                messages,
                storage_output,
                detected_tool_calls,
            )
            yield make_chunk(
                {
                    "delta": {},
                    "finish_reason": "tool_calls" if detected_tool_calls else "stop",
                    "usage": dump_model(usage),
                }
            )
            yield "data: [DONE]\n\n"
        finally:
            await discard_media_tasks()

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _create_responses_real_streaming_response(
    resp_or_stream: AsyncGenerator[ModelOutput] | ModelOutput,
    response_id: str,
    created_time: int,
    model_name: str,
    messages: list[AppMessage],
    db: LMDBConversationStore,
    resolved_model: str,
    client_wrapper: GeminiClientWrapper,
    session: ChatSession,
    request: ResponseCreateRequest,
    base_url: str,
    structured_requirement: StructuredOutputRequirement | None = None,
    has_image_tool: bool = False,
) -> StreamingResponse:
    """
    Create a real-time streaming response for the Responses API.
    Ensures final accumulated text and thoughts are synchronized and follow the formal event stream spec.
    Emits typed image and media results as incremental response output text events.
    """
    base_event = {
        "id": response_id,
        "object": "response",
        "created_at": created_time,
        "model": model_name,
    }

    async def generate_stream():
        seq = 0

        def make_event(etype: str, data: dict) -> str:
            nonlocal seq
            data["sequence_number"] = seq
            seq += 1
            return f"event: {etype}\ndata: {orjson.dumps(data).decode()}\n\n"

        media_tasks = []
        seen_media_urls = set()
        seen_image_urls = set()

        async def discard_media_tasks() -> None:
            """Cancel downloads whose results cannot be published after a stream failure."""
            if not media_tasks:
                return
            for task in media_tasks:
                task.cancel()
            await asyncio.gather(*media_tasks, return_exceptions=True)
            media_tasks.clear()

        try:
            yield make_event(
                "response.created",
                {
                    **base_event,
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created_time,
                        "model": model_name,
                        "status": "in_progress",
                        "metadata": request.metadata or {},
                        "input": None,
                        "tools": serialize_tools_for_response(request.tools),
                        "tool_choice": serialize_tool_choice_for_response(request.tool_choice),
                        "output": [],
                        "usage": None,
                    },
                },
            )
            yield make_event(
                "response.in_progress",
                {
                    **base_event,
                    "type": "response.in_progress",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created_time,
                        "model": model_name,
                        "status": "in_progress",
                        "metadata": request.metadata or {},
                        "output": [],
                    },
                },
            )

            full_text = ""
            full_thoughts = ""
            last_output: ModelOutput | None = None

            thought_item_id = f"rs_{uuid.uuid4().hex[:24]}"
            message_item_id = f"msg_{uuid.uuid4().hex[:24]}"

            thought_open, message_open = False, False
            next_output_index = 0
            thought_index = 0
            message_index = 0
            suppressor = StreamingOutputFilter()

            try:
                if hasattr(resp_or_stream, "__aiter__"):
                    generator = cast(AsyncGenerator[ModelOutput], resp_or_stream)
                else:

                    async def _make_async_gen(item: ModelOutput) -> AsyncGenerator[ModelOutput]:
                        yield item

                    generator = _make_async_gen(cast(ModelOutput, resp_or_stream))

                async for chunk in generator:
                    last_output = chunk

                    if chunk.thoughts_delta:
                        if not thought_open:
                            thought_index = next_output_index
                            next_output_index += 1
                            yield make_event(
                                "response.output_item.added",
                                {
                                    **base_event,
                                    "type": "response.output_item.added",
                                    "output_index": thought_index,
                                    "item": dump_model(
                                        ResponseReasoningItem(
                                            id=thought_item_id,
                                            type="reasoning",
                                            status="in_progress",
                                            summary=[],
                                        )
                                    ),
                                },
                            )

                            yield make_event(
                                "response.reasoning_summary_part.added",
                                {
                                    **base_event,
                                    "type": "response.reasoning_summary_part.added",
                                    "item_id": thought_item_id,
                                    "output_index": thought_index,
                                    "summary_index": 0,
                                    "part": dump_model(SummaryTextContent(text="")),
                                },
                            )
                            thought_open = True

                        full_thoughts += chunk.thoughts_delta
                        yield make_event(
                            "response.reasoning_summary_text.delta",
                            {
                                **base_event,
                                "type": "response.reasoning_summary_text.delta",
                                "item_id": thought_item_id,
                                "output_index": thought_index,
                                "summary_index": 0,
                                "delta": chunk.thoughts_delta,
                            },
                        )

                    if chunk.text_delta:
                        full_text += chunk.text_delta
                        if thought_open:
                            yield make_event(
                                "response.reasoning_summary_text.done",
                                {
                                    **base_event,
                                    "type": "response.reasoning_summary_text.done",
                                    "item_id": thought_item_id,
                                    "output_index": thought_index,
                                    "summary_index": 0,
                                    "text": full_thoughts,
                                },
                            )
                            yield make_event(
                                "response.reasoning_summary_part.done",
                                {
                                    **base_event,
                                    "type": "response.reasoning_summary_part.done",
                                    "item_id": thought_item_id,
                                    "output_index": thought_index,
                                    "summary_index": 0,
                                    "part": dump_model(SummaryTextContent(text=full_thoughts)),
                                },
                            )
                            yield make_event(
                                "response.output_item.done",
                                {
                                    **base_event,
                                    "type": "response.output_item.done",
                                    "output_index": thought_index,
                                    "item": dump_model(
                                        ResponseReasoningItem(
                                            id=thought_item_id,
                                            type="reasoning",
                                            status="completed",
                                            summary=[SummaryTextContent(text=full_thoughts)],
                                        )
                                    ),
                                },
                            )
                            thought_open = False

                        if not structured_requirement:
                            if not message_open:
                                message_index = next_output_index
                                next_output_index += 1
                                yield make_event(
                                    "response.output_item.added",
                                    {
                                        **base_event,
                                        "type": "response.output_item.added",
                                        "output_index": message_index,
                                        "item": dump_model(
                                            ResponseOutputMessage(
                                                id=message_item_id,
                                                type="message",
                                                status="in_progress",
                                                role="assistant",
                                                content=[],
                                            )
                                        ),
                                    },
                                )

                                yield make_event(
                                    "response.content_part.added",
                                    {
                                        **base_event,
                                        "type": "response.content_part.added",
                                        "item_id": message_item_id,
                                        "output_index": message_index,
                                        "content_index": 0,
                                        "part": dump_model(
                                            ResponseOutputText(type="output_text", text="")
                                        ),
                                    },
                                )
                                message_open = True

                            if visible := suppressor.process(chunk.text_delta):
                                yield make_event(
                                    "response.output_text.delta",
                                    {
                                        **base_event,
                                        "type": "response.output_text.delta",
                                        "item_id": message_item_id,
                                        "output_index": message_index,
                                        "content_index": 0,
                                        "delta": visible,
                                        "logprobs": [],
                                    },
                                )

                    for img in chunk.images or []:
                        if img.url and img.url not in seen_image_urls:
                            seen_image_urls.add(img.url)
                            media_tasks.append(asyncio.create_task(_process_image_item(img)))

                    m_list = (chunk.videos or []) + (chunk.media or [])
                    for m in m_list:
                        p_url = getattr(m, "url", None) or getattr(m, "mp3_url", None)
                        if p_url and p_url not in seen_media_urls:
                            seen_media_urls.add(p_url)
                            media_tasks.append(asyncio.create_task(_process_media_item(m)))

            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                await discard_media_tasks()
                yield make_event(
                    "error",
                    {
                        **base_event,
                        "type": "error",
                        "error": {"message": f"Streaming error occurred: {e}"},
                    },
                )
                return

            if last_output is not None:
                last = last_output
                if last.thoughts:
                    l_thoughts = last.thoughts
                    lt_len, ct_len = len(l_thoughts), len(full_thoughts)
                    if lt_len > ct_len and l_thoughts.startswith(full_thoughts):
                        drift_t = l_thoughts[ct_len:]
                        full_thoughts = l_thoughts
                        if not thought_open:
                            thought_index = next_output_index
                            next_output_index += 1
                            yield make_event(
                                "response.output_item.added",
                                {
                                    **base_event,
                                    "type": "response.output_item.added",
                                    "output_index": thought_index,
                                    "item": dump_model(
                                        ResponseReasoningItem(
                                            id=thought_item_id,
                                            type="reasoning",
                                            status="in_progress",
                                            summary=[],
                                        )
                                    ),
                                },
                            )
                            yield make_event(
                                "response.reasoning_summary_part.added",
                                {
                                    **base_event,
                                    "type": "response.reasoning_summary_part.added",
                                    "item_id": thought_item_id,
                                    "output_index": thought_index,
                                    "summary_index": 0,
                                    "part": dump_model(SummaryTextContent(text="")),
                                },
                            )
                            thought_open = True

                        yield make_event(
                            "response.reasoning_summary_text.delta",
                            {
                                **base_event,
                                "type": "response.reasoning_summary_text.delta",
                                "item_id": thought_item_id,
                                "output_index": thought_index,
                                "summary_index": 0,
                                "delta": drift_t,
                            },
                        )

                if last.text:
                    l_text = last.text
                    l_len, c_len = len(l_text), len(full_text)
                    if l_len > c_len and l_text.startswith(full_text):
                        drift = l_text[c_len:]
                        full_text = l_text
                        if not structured_requirement and (visible := suppressor.process(drift)):
                            if not message_open:
                                message_index = next_output_index
                                next_output_index += 1
                                yield make_event(
                                    "response.output_item.added",
                                    {
                                        **base_event,
                                        "type": "response.output_item.added",
                                        "output_index": message_index,
                                        "item": dump_model(
                                            ResponseOutputMessage(
                                                id=message_item_id,
                                                type="message",
                                                status="in_progress",
                                                role="assistant",
                                                content=[],
                                            )
                                        ),
                                    },
                                )
                                yield make_event(
                                    "response.content_part.added",
                                    {
                                        **base_event,
                                        "type": "response.content_part.added",
                                        "item_id": message_item_id,
                                        "output_index": message_index,
                                        "content_index": 0,
                                        "part": dump_model(
                                            ResponseOutputText(type="output_text", text="")
                                        ),
                                    },
                                )
                                message_open = True

                            yield make_event(
                                "response.output_text.delta",
                                {
                                    **base_event,
                                    "type": "response.output_text.delta",
                                    "item_id": message_item_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "delta": visible,
                                    "logprobs": [],
                                },
                            )

            remaining = "" if structured_requirement else suppressor.flush()
            if remaining and message_open:
                yield make_event(
                    "response.output_text.delta",
                    {
                        **base_event,
                        "type": "response.output_text.delta",
                        "item_id": message_item_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "delta": remaining,
                        "logprobs": [],
                    },
                )

            if thought_open:
                yield make_event(
                    "response.reasoning_summary_text.done",
                    {
                        **base_event,
                        "type": "response.reasoning_summary_text.done",
                        "item_id": thought_item_id,
                        "output_index": thought_index,
                        "summary_index": 0,
                        "text": full_thoughts,
                    },
                )
                yield make_event(
                    "response.reasoning_summary_part.done",
                    {
                        **base_event,
                        "type": "response.reasoning_summary_part.done",
                        "item_id": thought_item_id,
                        "output_index": thought_index,
                        "summary_index": 0,
                        "part": dump_model(SummaryTextContent(text=full_thoughts)),
                    },
                )
                yield make_event(
                    "response.output_item.done",
                    {
                        **base_event,
                        "type": "response.output_item.done",
                        "output_index": thought_index,
                        "item": dump_model(
                            ResponseReasoningItem(
                                id=thought_item_id,
                                type="reasoning",
                                status="completed",
                                summary=[SummaryTextContent(text=full_thoughts)],
                            )
                        ),
                    },
                )

            try:
                _, assistant_text, storage_output, detected_tool_calls = process_llm_output(
                    normalize_llm_text(full_thoughts or ""),
                    normalize_llm_text(full_text or ""),
                    structured_requirement,
                )
            except StructuredOutputValidationError as exc:
                await discard_media_tasks()
                yield make_event(
                    "error",
                    {
                        **base_event,
                        "type": "error",
                        "error": {
                            "message": str(exc),
                            "type": "invalid_model_output",
                            "param": "text.format",
                            "code": "schema_validation_failed",
                        },
                    },
                )
                return

            if structured_requirement and assistant_text and not message_open:
                message_index = next_output_index
                next_output_index += 1
                yield make_event(
                    "response.output_item.added",
                    {
                        **base_event,
                        "type": "response.output_item.added",
                        "output_index": message_index,
                        "item": dump_model(
                            ResponseOutputMessage(
                                id=message_item_id,
                                type="message",
                                status="in_progress",
                                role="assistant",
                                content=[],
                            )
                        ),
                    },
                )
                yield make_event(
                    "response.content_part.added",
                    {
                        **base_event,
                        "type": "response.content_part.added",
                        "item_id": message_item_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "part": dump_model(ResponseOutputText(type="output_text", text="")),
                    },
                )
                message_open = True
                yield make_event(
                    "response.output_text.delta",
                    {
                        **base_event,
                        "type": "response.output_text.delta",
                        "item_id": message_item_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "delta": assistant_text,
                        "logprobs": [],
                    },
                )

            image_items = []
            seen_hashes = {}
            seen_media_hashes = {}
            media_store = get_media_store_dir()

            if media_tasks:
                logger.debug(
                    f"Waiting for {len(media_tasks)} background media tasks in Responses with heartbeat..."
                )
                while media_tasks:
                    done, pending = await asyncio.wait(
                        media_tasks, timeout=5.0, return_when=asyncio.FIRST_COMPLETED
                    )
                    media_tasks = list(pending)

                    if not done:
                        yield ": ping\n\n"
                        continue

                    for task in done:
                        res = task.result()
                        if not res:
                            continue

                        rtype, original_item, media_data = res
                        if rtype == "image":
                            b64, w, h, fname, fhash = media_data
                            if fhash in seen_hashes:
                                (media_store / fname).unlink(missing_ok=True)
                                b64, w, h, fname = seen_hashes[fhash]
                            else:
                                seen_hashes[fhash] = (b64, w, h, fname)

                            parts = fname.rsplit(".", 1)
                            img_id = parts[0]
                            fmt = parts[1] if len(parts) > 1 else "png"

                            img_item = ImageGenerationCall(
                                id=img_id,
                                result=b64,
                                output_format=fmt,
                                size=f"{w}x{h}" if w and h else None,
                            )

                            img_link = f"![{fname}]({base_url}media/{fname}?token={get_media_token(fname)})"
                            md_to_add = f"\n\n{img_link}"

                            img_index = next_output_index
                            next_output_index += 1
                            yield make_event(
                                "response.output_item.added",
                                {
                                    **base_event,
                                    "type": "response.output_item.added",
                                    "output_index": img_index,
                                    "item": dump_model(img_item),
                                },
                            )
                            yield make_event(
                                "response.output_item.done",
                                {
                                    **base_event,
                                    "type": "response.output_item.done",
                                    "output_index": img_index,
                                    "item": dump_model(img_item),
                                },
                            )

                            if not message_open:
                                message_index = next_output_index
                                next_output_index += 1
                                yield make_event(
                                    "response.output_item.added",
                                    {
                                        **base_event,
                                        "type": "response.output_item.added",
                                        "output_index": message_index,
                                        "item": dump_model(
                                            ResponseOutputMessage(
                                                id=message_item_id,
                                                type="message",
                                                status="in_progress",
                                                role="assistant",
                                                content=[],
                                            )
                                        ),
                                    },
                                )
                                yield make_event(
                                    "response.content_part.added",
                                    {
                                        **base_event,
                                        "type": "response.content_part.added",
                                        "item_id": message_item_id,
                                        "output_index": message_index,
                                        "content_index": 0,
                                        "part": dump_model(
                                            ResponseOutputText(type="output_text", text="")
                                        ),
                                    },
                                )
                                message_open = True

                            yield make_event(
                                "response.output_text.delta",
                                {
                                    **base_event,
                                    "type": "response.output_text.delta",
                                    "item_id": message_item_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "delta": md_to_add,
                                    "logprobs": [],
                                },
                            )
                            assistant_text += md_to_add
                            storage_output += md_to_add
                            image_items.append(img_item)

                        elif rtype == "media":
                            m_dict = cast(ProcessedMediaData, media_data)
                            if not m_dict:
                                continue

                            m_urls = {}
                            for mtype, (random_name, fhash) in m_dict.items():
                                if fhash in seen_media_hashes:
                                    existing_name = seen_media_hashes[fhash]
                                    if random_name != existing_name:
                                        (media_store / random_name).unlink(missing_ok=True)
                                    m_urls[mtype] = (
                                        f"{base_url}media/{existing_name}?token={get_media_token(existing_name)}"
                                    )
                                else:
                                    seen_media_hashes[fhash] = random_name
                                    m_urls[mtype] = (
                                        f"{base_url}media/{random_name}?token={get_media_token(random_name)}"
                                    )

                            title = getattr(original_item, "title", "Media")
                            video_url = m_urls.get("video")
                            audio_url = m_urls.get("audio")
                            current_thumb = m_urls.get("video_thumbnail") or m_urls.get(
                                "audio_thumbnail"
                            )

                            md_parts = []
                            if video_url:
                                md_parts.append(
                                    f"[![{title}]({current_thumb})]({video_url})"
                                    if current_thumb
                                    else f"[{title}]({video_url})"
                                )
                            if audio_url:
                                md_parts.append(
                                    f"[![{title} - Audio]({current_thumb})]({audio_url})"
                                    if current_thumb
                                    else f"[{title} - Audio]({audio_url})"
                                )

                            if md_parts:
                                media_md = "\n\n".join(md_parts)
                                md_to_add = f"\n\n{media_md}"

                                if not message_open:
                                    message_index = next_output_index
                                    next_output_index += 1
                                    yield make_event(
                                        "response.output_item.added",
                                        {
                                            **base_event,
                                            "type": "response.output_item.added",
                                            "output_index": message_index,
                                            "item": dump_model(
                                                ResponseOutputMessage(
                                                    id=message_item_id,
                                                    type="message",
                                                    status="in_progress",
                                                    role="assistant",
                                                    content=[],
                                                )
                                            ),
                                        },
                                    )
                                    yield make_event(
                                        "response.content_part.added",
                                        {
                                            **base_event,
                                            "type": "response.content_part.added",
                                            "item_id": message_item_id,
                                            "output_index": message_index,
                                            "content_index": 0,
                                            "part": dump_model(
                                                ResponseOutputText(type="output_text", text="")
                                            ),
                                        },
                                    )
                                    message_open = True

                                yield make_event(
                                    "response.output_text.delta",
                                    {
                                        **base_event,
                                        "type": "response.output_text.delta",
                                        "item_id": message_item_id,
                                        "output_index": message_index,
                                        "content_index": 0,
                                        "delta": md_to_add,
                                        "logprobs": [],
                                    },
                                )
                                assistant_text += md_to_add
                                storage_output += md_to_add

            final_response_contents: list[ResponseOutputContent] = []
            if choice_error := _tool_choice_failure(
                request.tool_choice,
                detected_tool_calls,
                has_images=bool(image_items),
                has_image_tool=has_image_tool,
            ):
                yield make_event(
                    "error",
                    {
                        **base_event,
                        "type": "error",
                        "error": {
                            "message": choice_error,
                            "type": "invalid_model_output",
                            "param": "tool_choice",
                            "code": "required_tool_missing",
                        },
                    },
                )
                return
            if message_open:
                if assistant_text:
                    final_response_contents = [
                        ResponseOutputText(type="output_text", text=assistant_text)
                    ]
                else:
                    final_response_contents = [ResponseOutputText(type="output_text", text="")]

                yield make_event(
                    "response.output_text.done",
                    {
                        **base_event,
                        "type": "response.output_text.done",
                        "item_id": message_item_id,
                        "output_index": message_index,
                        "content_index": 0,
                    },
                )
                yield make_event(
                    "response.content_part.done",
                    {
                        **base_event,
                        "type": "response.content_part.done",
                        "item_id": message_item_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "part": dump_model(
                            ResponseOutputText(type="output_text", text=assistant_text)
                        ),
                    },
                )

                yield make_event(
                    "response.output_item.done",
                    {
                        **base_event,
                        "type": "response.output_item.done",
                        "output_index": message_index,
                        "item": dump_model(
                            ResponseOutputMessage(
                                id=message_item_id,
                                type="message",
                                status="completed",
                                role="assistant",
                                content=final_response_contents,
                            )
                        ),
                    },
                )

            for call in detected_tool_calls:
                tc_index = next_output_index
                next_output_index += 1
                tc_item = ResponseFunctionToolCall(
                    id=call.id,
                    call_id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                    status="completed",
                )
                yield make_event(
                    "response.output_item.added",
                    {
                        **base_event,
                        "type": "response.output_item.added",
                        "output_index": tc_index,
                        "item": dump_model(tc_item),
                    },
                )
                yield make_event(
                    "response.output_item.done",
                    {
                        **base_event,
                        "type": "response.output_item.done",
                        "output_index": tc_index,
                        "item": dump_model(tc_item),
                    },
                )

            p_tok, c_tok, t_tok, r_tok = calculate_usage(
                messages, storage_output, detected_tool_calls, full_thoughts
            )
            usage = ResponseUsage(
                input_tokens=p_tok,
                output_tokens=c_tok,
                total_tokens=t_tok,
                output_tokens_details={"reasoning_tokens": r_tok},
            )
            payload = _create_responses_standard_payload(
                response_id,
                created_time,
                model_name,
                detected_tool_calls,
                image_items,
                final_response_contents,
                usage,
                request,
                structured_requirement,
                full_thoughts,
                message_item_id,
                thought_item_id,
            )
            _persist_conversation(
                db,
                resolved_model,
                client_wrapper,
                session.metadata,
                messages,
                storage_output,
                detected_tool_calls,
            )

            yield make_event(
                "response.completed",
                {
                    **base_event,
                    "type": "response.completed",
                    "response": dump_model(payload),
                },
            )

            yield "data: [DONE]\n\n"
        finally:
            await discard_media_tasks()

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# --- Main Router Endpoints ---


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    pool = GeminiClientPool()
    models = await _get_available_models(pool)
    return ModelListResponse(data=models)


@router.post("/v1/chat/completions", response_model_exclude_none=True)
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    base_url = str(raw_request.base_url)
    pool, db = GeminiClientPool(), LMDBConversationStore()
    try:
        resolved_model = _resolve_model_name(pool, request.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if not request.messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Messages required.")

    _log_ignored_openai_options(request)
    function_names = {tool.function.name for tool in request.tools or []}
    if choice_error := _tool_choice_declaration_error(function_names, False, request.tool_choice):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=choice_error)

    try:
        structured_requirement = _build_structured_requirement(request.response_format)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    extra_instr = [structured_requirement.instruction] if structured_requirement else None

    app_messages = convert_to_app_messages(request.messages)

    msgs = _prepare_messages_for_model(
        app_messages,
        request.tools,
        request.tool_choice,
        extra_instr,
    )

    use_temporary = _use_temporary_chat_mode()
    needs_upload = _requires_upload(msgs, use_temporary)
    session, client, remain, stored_conv = await _find_reusable_session(
        db, pool, resolved_model, msgs, temporary=use_temporary, require_account=needs_upload
    )

    if session:
        if not remain:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No new messages.")

        input_msgs = _prepare_messages_for_model(
            remain,
            request.tools,
            request.tool_choice,
            extra_instr,
            False,
        )
        m_input, files = await GeminiClientWrapper.process_conversation(input_msgs, tmp_dir)

        logger.debug(
            f"Reused session {reprlib.repr(session.metadata)} - sending {len(input_msgs)} prepared messages."
        )
    else:
        try:
            client = await pool.acquire(require_account=needs_upload)
            session = client.start_chat(model=client.usable_model(resolved_model))
            m_input, files = await GeminiClientWrapper.process_conversation(msgs, tmp_dir)
        except Exception as e:
            logger.error(f"Error in preparing conversation: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            ) from e

    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(datetime.now(tz=UTC).timestamp())

    if session is None or client is None:
        logger.error("No Gemini session or client available after preparing conversation.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No available Gemini client."
        )

    try:
        logger.debug(
            f"Client ID: {client.id}, Input length: {len(m_input)}, files count: {len(files)}"
        )
        resp_or_stream, session, client = await _send_with_internal_fallback(
            pool=pool,
            db=db,
            resolved_model=resolved_model,
            session=session,
            client=client,
            current_input=m_input,
            files=files,
            full_prepared_messages=msgs,
            stored_conversation=stored_conv,
            tmp_dir=tmp_dir,
            stream=bool(request.stream),
            temporary=use_temporary,
        )
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    if request.stream:
        if isinstance(resp_or_stream, ModelOutput):
            logger.error("Expected a streaming response from Gemini but got a complete output.")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="Streaming response unavailable."
            )
        return _create_real_streaming_response(
            resp_or_stream,
            completion_id,
            created_time,
            request.model,
            msgs,
            db,
            resolved_model,
            client,
            session,
            base_url,
            structured_requirement,
            request.tool_choice,
        )

    if not isinstance(resp_or_stream, ModelOutput):
        logger.error("Expected a complete output from Gemini but got a stream.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Unexpected streaming response."
        )

    try:
        thoughts, visible_output, storage_output, tool_calls = process_llm_output(
            normalize_llm_text(resp_or_stream.thoughts or ""),
            normalize_llm_text(resp_or_stream.text or ""),
            structured_requirement,
        )
    except StructuredOutputValidationError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    images = resp_or_stream.images or []
    # No `has_images` escape hatch here: Chat Completions has no image-generation tool, so an
    # image Gemini volunteers on its own cannot stand in for a function call that was forced.
    if choice_error := _tool_choice_failure(request.tool_choice, tool_calls):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=choice_error)

    media_items: list[GeneratedVideo | GeneratedMedia] = (resp_or_stream.videos or []) + (
        resp_or_stream.media or []
    )
    unique_media = []
    seen_urls = set()
    for m in media_items:
        v_url = getattr(m, "url", None)
        a_url = getattr(m, "mp3_url", None)
        primary_url = v_url or a_url
        if primary_url and primary_url not in seen_urls:
            unique_media.append(m)
            seen_urls.add(primary_url)

    tasks = [_process_image_item(img) for img in images] + [
        _process_media_item(m) for m in unique_media
    ]
    results = await asyncio.gather(*tasks)

    image_markdown = ""
    media_markdown = ""
    seen_hashes = {}
    seen_media_hashes = {}
    media_store = get_media_store_dir()

    for res in results:
        if not res:
            continue

        if res[0] == "image":
            original_item = res[1]
            media_data = res[2]
            _, _, _, fname, fhash = media_data
            if fhash in seen_hashes:
                (media_store / fname).unlink(missing_ok=True)
                fname = seen_hashes[fhash]
            else:
                seen_hashes[fhash] = fname

            img_url = f"{base_url}media/{fname}?token={get_media_token(fname)}"
            title = getattr(original_item, "title", "Image")
            image_markdown += f"\n\n![{title}]({img_url})"

        elif res[0] == "media":
            original_item = res[1]
            media_data = res[2]
            m_dict = media_data
            if not m_dict:
                continue

            m_urls = {}
            for mtype, (random_name, fhash) in m_dict.items():
                if fhash in seen_media_hashes:
                    existing_name = seen_media_hashes[fhash]
                    if random_name != existing_name:
                        (media_store / random_name).unlink(missing_ok=True)
                    m_urls[mtype] = (
                        f"{base_url}media/{existing_name}?token={get_media_token(existing_name)}"
                    )
                else:
                    seen_media_hashes[fhash] = random_name
                    m_urls[mtype] = (
                        f"{base_url}media/{random_name}?token={get_media_token(random_name)}"
                    )

            title = getattr(original_item, "title", "Media")
            video_url = m_urls.get("video")
            audio_url = m_urls.get("audio")
            current_thumb = m_urls.get("video_thumbnail") or m_urls.get("audio_thumbnail")

            md_parts = []
            if video_url:
                md_parts.append(
                    f"[![{title}]({current_thumb})]({video_url})"
                    if current_thumb
                    else f"[{title}]({video_url})"
                )
            if audio_url:
                md_parts.append(
                    f"[![{title} - Audio]({current_thumb})]({audio_url})"
                    if current_thumb
                    else f"[{title} - Audio]({audio_url})"
                )

            if md_parts:
                media_markdown += f"\n\n{'\n\n'.join(md_parts)}"

    if image_markdown:
        visible_output += image_markdown
        storage_output += image_markdown

    if media_markdown:
        visible_output += media_markdown
        storage_output += media_markdown

    p_tok, c_tok, t_tok, r_tok = calculate_usage(app_messages, storage_output, tool_calls, thoughts)
    usage = {
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "total_tokens": t_tok,
        "completion_tokens_details": {"reasoning_tokens": r_tok},
    }
    payload = _create_chat_completion_standard_payload(
        completion_id,
        created_time,
        request.model,
        visible_output,
        tool_calls or None,
        "tool_calls" if tool_calls else "stop",
        usage,
        thoughts,
    )
    _persist_conversation(
        db,
        resolved_model,
        client,
        session.metadata,
        msgs,
        storage_output,
        tool_calls,
    )
    return payload


@router.post("/v1/responses", response_model_exclude_none=True)
async def create_response(
    request: ResponseCreateRequest,
    raw_request: Request,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    base_url = str(raw_request.base_url)
    _log_ignored_openai_options(request)
    if input_error := _validate_responses_input(request.input):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=input_error)
    base_messages = _convert_responses_to_app_messages(request.input)
    try:
        structured_requirement = _build_structured_requirement(_responses_response_format(request))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    extra_instr = [structured_requirement.instruction] if structured_requirement else []

    standard_tools, image_tools = [], []
    if request.tools:
        for t in request.tools:
            if isinstance(t, FunctionTool):
                standard_tools.append(t)
            elif isinstance(t, ImageGeneration):
                image_tools.append(t)
            elif isinstance(t, dict):
                if t.get("type") == "function":
                    standard_tools.append(FunctionTool.model_validate(t))
                elif t.get("type") == "image_generation":
                    image_tools.append(ImageGeneration.model_validate(t))

    if ignored_image_options := {
        name for image_tool in image_tools for name in image_tool.model_fields_set if name != "type"
    }:
        logger.debug(
            "Ignoring image-generation option(s) unsupported by the Gemini Web upstream: "
            f"{', '.join(sorted(ignored_image_options))}"
        )

    if choice_error := _tool_choice_declaration_error(
        {tool.name for tool in standard_tools},
        bool(image_tools),
        request.tool_choice,
    ):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=choice_error)

    img_instr = build_image_generation_instruction(
        image_tools,
        request.tool_choice if isinstance(request.tool_choice, ToolChoiceTypes) else None,
    )
    if img_instr:
        extra_instr.append(img_instr)
    preface = _convert_instructions_to_app_messages(request.instructions)
    conv_messages = [*preface, *base_messages] if preface else base_messages
    model_tool_choice = (
        request.tool_choice if isinstance(request.tool_choice, (str, ToolChoiceFunction)) else None
    )

    messages = _prepare_messages_for_model(
        conv_messages,
        standard_tools or None,
        model_tool_choice,
        extra_instr or None,
    )
    pool, db = GeminiClientPool(), LMDBConversationStore()
    try:
        resolved_model = _resolve_model_name(pool, request.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    use_temporary = _use_temporary_chat_mode()
    needs_upload = _requires_upload(messages, use_temporary)
    session, client, remain, stored_conv = await _find_reusable_session(
        db, pool, resolved_model, messages, temporary=use_temporary, require_account=needs_upload
    )
    if session:
        msgs = _prepare_messages_for_model(
            remain,
            standard_tools or None,
            model_tool_choice,
            None,
            False,
        )
        if not msgs:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No new messages.")
        m_input, files = await GeminiClientWrapper.process_conversation(msgs, tmp_dir)
        logger.debug(
            f"Reused session {reprlib.repr(session.metadata)} - sending {len(msgs)} prepared messages."
        )
    else:
        try:
            client = await pool.acquire(require_account=needs_upload)
            session = client.start_chat(model=client.usable_model(resolved_model))
            m_input, files = await GeminiClientWrapper.process_conversation(messages, tmp_dir)
        except Exception as e:
            logger.error(f"Error in preparing conversation: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            ) from e

    response_id = f"resp_{uuid.uuid4().hex}"
    created_time = int(datetime.now(tz=UTC).timestamp())

    if session is None or client is None:
        logger.error("No Gemini session or client available after preparing conversation.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No available Gemini client."
        )

    try:
        logger.debug(
            f"Client ID: {client.id}, Input length: {len(m_input)}, files count: {len(files)}"
        )
        resp_or_stream, session, client = await _send_with_internal_fallback(
            pool=pool,
            db=db,
            resolved_model=resolved_model,
            session=session,
            client=client,
            current_input=m_input,
            files=files,
            full_prepared_messages=messages,
            stored_conversation=stored_conv,
            tmp_dir=tmp_dir,
            stream=bool(request.stream),
            temporary=use_temporary,
        )
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    if request.stream:
        if isinstance(resp_or_stream, ModelOutput):
            logger.error("Expected a streaming response from Gemini but got a complete output.")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="Streaming response unavailable."
            )
        return _create_responses_real_streaming_response(
            resp_or_stream,
            response_id,
            created_time,
            request.model,
            messages,
            db,
            resolved_model,
            client,
            session,
            request,
            base_url,
            structured_requirement,
            bool(image_tools),
        )

    if not isinstance(resp_or_stream, ModelOutput):
        logger.error("Expected a complete output from Gemini but got a stream.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Unexpected streaming response."
        )

    try:
        thoughts, assistant_text, storage_output, tool_calls = process_llm_output(
            normalize_llm_text(resp_or_stream.thoughts or ""),
            normalize_llm_text(resp_or_stream.text or ""),
            structured_requirement,
        )
    except StructuredOutputValidationError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    images = resp_or_stream.images or []
    if choice_error := _tool_choice_failure(
        request.tool_choice,
        tool_calls,
        has_images=bool(images),
        has_image_tool=bool(image_tools),
    ):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=choice_error)

    unique_media = []
    seen_urls = set()
    for m in (resp_or_stream.videos or []) + (resp_or_stream.media or []):
        p_url = getattr(m, "url", None) or getattr(m, "mp3_url", None)
        if p_url and p_url not in seen_urls:
            unique_media.append(m)
            seen_urls.add(p_url)

    tasks = [_process_image_item(img) for img in images] + [
        _process_media_item(m) for m in unique_media
    ]
    results = await asyncio.gather(*tasks)

    contents, img_calls = [], []
    seen_hashes = {}
    seen_media_hashes = {}
    media_markdown = ""
    media_store = get_media_store_dir()

    for res in results:
        if not res:
            continue

        if res[0] == "image":
            media_data = res[2]
            b64, w, h, fname, fhash = media_data
            if fhash in seen_hashes:
                (media_store / fname).unlink(missing_ok=True)
                b64, w, h, fname = seen_hashes[fhash]
            else:
                seen_hashes[fhash] = (b64, w, h, fname)

            parts = fname.rsplit(".", 1)
            img_id = parts[0]
            fmt = parts[1] if len(parts) > 1 else "png"
            img_calls.append(
                ImageGenerationCall(
                    id=img_id, result=b64, output_format=fmt, size=f"{w}x{h}" if w and h else None
                )
            )

        elif res[0] == "media":
            original_item = res[1]
            media_data = res[2]
            m_dict = media_data
            if not m_dict:
                continue

            m_urls = {}
            for mtype, (random_name, fhash) in m_dict.items():
                if fhash in seen_media_hashes:
                    existing_name = seen_media_hashes[fhash]
                    if random_name != existing_name:
                        (media_store / random_name).unlink(missing_ok=True)
                    m_urls[mtype] = (
                        f"{base_url}media/{existing_name}?token={get_media_token(existing_name)}"
                    )
                else:
                    seen_media_hashes[fhash] = random_name
                    m_urls[mtype] = (
                        f"{base_url}media/{random_name}?token={get_media_token(random_name)}"
                    )

            title = getattr(original_item, "title", "Media")
            video_url = m_urls.get("video")
            audio_url = m_urls.get("audio")
            current_thumb = m_urls.get("video_thumbnail") or m_urls.get("audio_thumbnail")

            md_parts = []
            if video_url:
                md_parts.append(
                    f"[![{title}]({current_thumb})]({video_url})"
                    if current_thumb
                    else f"[{title}]({video_url})"
                )
            if audio_url:
                md_parts.append(
                    f"[![{title} - Audio]({current_thumb})]({audio_url})"
                    if current_thumb
                    else f"[{title} - Audio]({audio_url})"
                )

            if md_parts:
                media_markdown += f"\n\n{'\n\n'.join(md_parts)}"

    if assistant_text:
        contents.append(ResponseOutputText(type="output_text", text=assistant_text))

    image_markdown = ""
    for ic in img_calls:
        img_url = f"{base_url}media/{ic.id}.{ic.output_format}?token={get_media_token(f'{ic.id}.{ic.output_format}')}"
        image_markdown += f"\n\n![{ic.id}]({img_url})"

    if image_markdown:
        storage_output += image_markdown
        contents.append(ResponseOutputText(type="output_text", text=image_markdown))

    if media_markdown:
        storage_output += media_markdown
        contents.append(ResponseOutputText(type="output_text", text=media_markdown))

    if not contents:
        contents.append(ResponseOutputText(type="output_text", text=""))

    p_tok, c_tok, t_tok, r_tok = calculate_usage(messages, storage_output, tool_calls, thoughts)
    usage = ResponseUsage(
        input_tokens=p_tok,
        output_tokens=c_tok,
        total_tokens=t_tok,
        output_tokens_details={"reasoning_tokens": r_tok},
    )
    payload = _create_responses_standard_payload(
        response_id,
        created_time,
        request.model,
        tool_calls,
        img_calls,
        contents,
        usage,
        request,
        structured_requirement,
        thoughts,
    )
    _persist_conversation(
        db,
        resolved_model,
        client,
        session.metadata,
        messages,
        storage_output,
        tool_calls,
    )
    return payload
