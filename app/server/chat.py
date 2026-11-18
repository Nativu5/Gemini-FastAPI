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
    ChatCompletionMessageToolCall,
    ChatCompletionNamedToolChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
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
    ResponseInputFile,
    ResponseInputImage,
    ResponseInputMessage,
    ResponseInputText,
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
from app.services.lmdb import cid_of
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
    dump_model,
    extract_image_dimensions,
    extract_named_tool_choice,
    normalize_app_message_role,
    normalize_llm_text,
    process_llm_output,
    strip_system_hints,
    text_from_message,
    validate_json_schema,
)

MAX_CHARS_PER_REQUEST = int(g_config.gemini.max_chars_per_request * 0.9)
# Google's temporary chat mode accepts a smaller payload than a normal chat, so tighten
# the guardrail further on top of the standard 10% safety margin.
TEMPORARY_MAX_CHARS_PER_REQUEST = int(MAX_CHARS_PER_REQUEST * 0.9)

router = APIRouter()
_AVAILABLE_MODELS_CACHE: list[ModelData] | None = None


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
    # The library names the file after its content type, so it always has an extension.
    random_name = f"img_{uuid.uuid4().hex}{original_path.suffix}"
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


def _echoed_tools(request: ResponseCreateRequest) -> dict[str, Any]:
    """The tools and tool choice of a request, as its response reports them back."""
    choice = request.tool_choice
    return {
        "tools": [dump_model(tool) for tool in request.tools or []],
        "tool_choice": choice or "auto" if isinstance(choice, str | None) else dump_model(choice),
    }


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
        **_echoed_tools(request),
        text=text_config,
    )


def _persist_conversation(
    db: LMDBConversationStore,
    model_name: str,
    client: GeminiClientWrapper,
    metadata: list[str | None],
    messages: list[AppMessage],
    storage_output: str | list[AppContentItem] | None,
    tool_calls: list[AppToolCall] | None,
) -> None:
    """Record this turn as the client's latest chat and store the history so a later request
    can continue it. A storage failure is logged, not raised: the reply is already made."""
    # This turn is now the last chat this client opened; any window it replaced is closed. Set
    # before the store, so a persistence failure cannot leave a closed window looking reusable,
    # and in every mode, since an expired cookie can turn a client ephemeral without warning.
    client.latest_chat_cid = cid_of(metadata)
    chat_scope = client.chat_scope(_use_temporary_chat_mode())

    try:
        reply = AppMessage(
            role="assistant", content=storage_output or None, tool_calls=tool_calls or None
        )

        # An ephemeral chat is worth storing like any other while its window is live, tagged with
        # that window so a later session cannot mistake it for one of its own.
        db.store(
            client_id=client.id,
            model=model_name,
            messages=[*messages, reply],
            metadata=metadata,
            chat_scope=chat_scope,
            chat_mode=g_config.gemini.chat_mode,
        )
        logger.debug("Conversation saved to LMDB.")
    except Exception as e:
        logger.warning(f"Failed to save {len(messages) + 1} messages to LMDB: {e}")


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
    if not response_format:
        return None

    format_type = response_format.get("type")
    if format_type == "json_object":
        return StructuredOutputRequirement(
            schema_name="response",
            schema={},
            instruction=STRUCTURED_JSON_WRAP_HINT,
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
    requested_strict = json_schema.get("strict")
    if requested_strict is None:  # absent or null: OpenAI's default
        requested_strict = False
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
    control_fields = {
        "temperature",
        "top_p",
        "max_completion_tokens",
        "max_output_tokens",
        "parallel_tool_calls",
    }
    if ignored := control_fields & request.model_fields_set:
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
    if (target_name := extract_named_tool_choice(tool_choice)) and all(
        call.function.name != target_name for call in tool_calls
    ):
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
    if (
        target_name := extract_named_tool_choice(tool_choice)
    ) and target_name not in function_names:
        return f"tool_choice names undeclared function {target_name!r}"
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

    instructions = (
        [build_tool_prompt(tools, tool_choice)] if inject_system_defaults and tools else []
    )
    instructions += [instruction for instruction in extra_instructions or [] if instruction]
    if instructions:
        combined_instructions = "\n\n".join(instructions)
        if prepared and prepared[0].role == "system" and isinstance(prepared[0].content, str):
            existing = prepared[0].content or ""
            if combined_instructions not in existing:
                separator = "\n\n" if existing else ""
                prepared[0].content = f"{existing}{separator}{combined_instructions}"
        else:
            prepared.insert(0, AppMessage(role="system", content=combined_instructions))

    # A continued chat was given the tool prompt when it was opened; it is only reminded.
    if tools and tool_choice != "none" and not inject_system_defaults:
        append_tool_hint_to_last_user_message(prepared)

    return prepared


def _extract_leading_system_prompt(
    messages: list[AppMessage],
) -> tuple[str | None, list[AppMessage]]:
    """Extract and remove leading text-only system messages, returning joined system text."""
    if not messages:
        return None, messages

    idx = 0
    system_parts: list[str] = []
    while idx < len(messages) and messages[idx].role == "system":
        msg = messages[idx]
        if isinstance(msg.content, list) and any(item.type != "text" for item in msg.content):
            break
        text = text_from_message(msg).strip()
        if text:
            system_parts.append(text)
        idx += 1

    if not system_parts:
        return None, messages

    return "\n\n".join(system_parts), messages[idx:]


async def _resolve_system_prompt_gem(
    client: GeminiClientWrapper, system_prompt: str | None
) -> str | None:
    """Id of the gem that carries this system prompt, or None to send the prompt inline."""
    if not system_prompt:
        return None
    try:
        return await client.system_prompt_gem_id_or_create(system_prompt)
    except Exception as exc:
        logger.warning(
            f"Failed to resolve managed system-prompt gem on client {client.id}; "
            f"falling back to inline system prompt: {exc}"
        )
        return None


def _convert_responses_to_app_messages(
    items: str
    | list[
        ResponseInputMessage
        | ResponseOutputMessage
        | ResponseReasoningItem
        | ResponseFunctionToolCall
        | FunctionCallOutput
        | ImageGenerationCall
    ],
) -> list[AppMessage]:
    """Convert Responses API input items into internal messages.

    Consecutive assistant items - a message, its reasoning, its function calls - are one turn of
    the model and become one message.
    """
    if isinstance(items, str):
        return [AppMessage(role="user", content=items)]

    messages: list[AppMessage] = []
    for item in items:
        if isinstance(item, (ResponseInputMessage, ResponseOutputMessage)):
            role = normalize_app_message_role(item.role)
            if isinstance(item.content, str):
                messages.append(AppMessage(role=role, content=item.content))
                continue
            converted: list[AppContentItem] = []
            for part in item.content:
                if isinstance(part, (ResponseInputText, ResponseOutputText)):
                    if part.text:
                        converted.append(AppContentItem(type="text", text=part.text))
                elif isinstance(part, ResponseInputImage):
                    if part.image_url:
                        converted.append(AppContentItem(type="image_url", url=part.image_url))
                elif isinstance(part, ResponseInputFile) and (part.file_url or part.file_data):
                    converted.append(
                        AppContentItem(
                            type="file",
                            url=part.file_url,
                            file_data=part.file_data,
                            filename=part.filename,
                        )
                    )
            messages.append(AppMessage(role=role, content=converted or None))

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
                    if isinstance(part, ResponseInputText):
                        converted_output.append(AppContentItem(type="text", text=part.text or ""))
                    elif isinstance(part, ResponseInputImage):
                        if part.image_url:
                            converted_output.append(
                                AppContentItem(type="image_url", url=part.image_url)
                            )
                    else:
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
            # The reasoning itself is not sent back to the model, but the item still marks a
            # turn of the model: a reply that was only reasoning is stored as an empty one.
            messages.append(AppMessage(role="assistant"))
        # An image_generation_call is skipped: the reply text already links the image, and its
        # base64 is no use to the model as text.

    def as_items(content: str | list[AppContentItem] | None) -> list[AppContentItem]:
        return (
            [AppContentItem(type="text", text=content)]
            if isinstance(content, str)
            else content or []
        )

    turns: list[AppMessage] = []
    for msg in messages:
        last = turns[-1] if turns else None
        if last is None or last.role != "assistant" or msg.role != "assistant":
            turns.append(msg)
            continue
        last.content = [*as_items(last.content), *as_items(msg.content)] or None
        last.tool_calls = [*(last.tool_calls or []), *(msg.tool_calls or [])] or None
    return turns


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


def _build_available_models(pool: GeminiClientPool) -> list[ModelData]:
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


def available_models(pool: GeminiClientPool) -> list[ModelData]:
    """The models the clients discovered, listed once and kept: `created` is when that was."""
    global _AVAILABLE_MODELS_CACHE
    if _AVAILABLE_MODELS_CACHE is None:
        _AVAILABLE_MODELS_CACHE = _build_available_models(pool)
        logger.info(f"Cached {len(_AVAILABLE_MODELS_CACHE)} available model(s).")
    return list(_AVAILABLE_MODELS_CACHE)


def _is_reusable_chat(conv: ConversationInStore, client: GeminiClientWrapper) -> bool:
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

    A chat is also never continued across the `auto_delete` boundary. Records are tagged with the
    mode active when they are written, and that tag alone decides whether the chat is later
    removed from the Google account. Continuing a kept chat under `auto_delete` would tag it for
    deletion; continuing an `auto_delete` chat under another mode would keep it alive for good.
    Replaying into a fresh chat keeps every Google chat under the one mode that created it.
    """
    stored_auto_delete = conv.chat_mode == ChatMode.AUTO_DELETE
    if stored_auto_delete != (g_config.gemini.chat_mode == ChatMode.AUTO_DELETE):
        logger.debug(
            f"Stored chat was written under chat mode {conv.chat_mode or 'normal'!r} but "
            f"{g_config.gemini.chat_mode.value!r} is active; replaying the full history in a "
            "fresh conversation."
        )
        return False

    scope = client.chat_scope(_use_temporary_chat_mode())
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

    cid = cid_of(conv.metadata)
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
                    if not _is_reusable_chat(conv, client):
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


def _max_chars_per_request() -> int:
    """Longest prompt sent as text in the active chat mode; anything longer goes as a file."""
    if _use_temporary_chat_mode():
        return TEMPORARY_MAX_CHARS_PER_REQUEST
    return MAX_CHARS_PER_REQUEST


def _requires_upload(messages: list[AppMessage]) -> bool:
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

    return total > _max_chars_per_request()


async def _send_with_split(
    session: ChatSession, text: str, files: list[Any], stream: bool
) -> AsyncGenerator[ModelOutput] | ModelOutput:
    """Send text to Gemini, as an attachment when it is too long for a prompt."""
    limit = _max_chars_per_request()
    if len(text) > limit:
        if cast(GeminiClientWrapper, session.geminiclient).is_guest():
            # Routing prefers an authenticated client, so this is hit only when the request
            # still landed on a guest: none other was available, or the chat belongs to one.
            raise RuntimeError(
                f"Message length ({len(text)}) exceeds limit ({limit}) and would have to be "
                "sent as an attachment, which a guest session cannot upload. Refresh the client "
                "cookies or shorten the request."
            )
        logger.info(
            f"Message length ({len(text)}) exceeds limit ({limit}). Converting text to file attachment."
        )
        file_obj = io.BytesIO(text.encode("utf-8"))
        file_obj.name = "message.txt"
        files = [file_obj, *files]
        text = (
            "The user's input exceeds the character limit and is provided in the attached file `message.txt`.\n\n"
            "**System Instruction:**\n"
            "1. Read the content of `message.txt`.\n"
            "2. Treat that content as the **primary** user prompt for this turn.\n"
            "3. Execute the instructions or answer the questions found *inside* that file immediately.\n"
        )

    send = session.send_message_stream if stream else session.send_message
    result = send(
        text,
        files=files,
        temporary=_use_temporary_chat_mode(),
        extended_thinking=g_config.gemini.extended_thinking,
    )
    return result if isinstance(result, AsyncGenerator) else await result


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
) -> AsyncGenerator[ModelOutput] | ModelOutput:
    """Send to Gemini, pulling the first streamed chunk so start-of-stream errors surface here.

    `send_message_stream` is an async generator function: calling it runs none of its body, so
    without this the request would not reach Google until the caller iterates - by which point
    the HTTP response has already been committed and a failure can no longer be recovered.
    """
    output = await _send_with_split(session, text, files, stream)
    if isinstance(output, ModelOutput):
        return output

    generator = output
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
) -> tuple[AsyncGenerator[ModelOutput] | ModelOutput, ChatSession, GeminiClientWrapper]:
    """Send the request, replaying the full history in a fresh chat if reused metadata is dead.

    Streaming is recovered as well as non-streaming: the first chunk is pulled here, which is
    where Google reports a rejected chat, so the retry happens before any response is committed
    to the client. The cost is that response headers wait for Google's first chunk.
    """
    try:
        output = await _send_and_await_first_chunk(
            session, current_input, files=files, stream=stream
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
            require_account=bool(fallback_files) or len(fallback_input) > _max_chars_per_request()
        )
        fallback_session = fallback_client.start_chat(
            model=fallback_client.usable_model(resolved_model)
        )
        output = await _send_and_await_first_chunk(
            fallback_session, fallback_input, files=list(fallback_files), stream=stream
        )
        return output, fallback_session, fallback_client


async def _send_request(
    *,
    pool: GeminiClientPool,
    db: LMDBConversationStore,
    resolved_model: str,
    history: list[AppMessage],
    prepared: list[AppMessage],
    tools: Sequence[Any] | None,
    tool_choice: Literal["none", "auto", "required"]
    | ChatCompletionNamedToolChoice
    | ToolChoiceFunction
    | ToolChoiceTypes
    | None,
    instructions: list[str] | None,
    repeat_instructions: bool,
    tmp_dir: Path,
    stream: bool,
) -> tuple[AsyncGenerator[ModelOutput] | ModelOutput, ChatSession, GeminiClientWrapper]:
    """Continue the stored chat that `history` extends, or open a new one, and send the rest.

    `prepared` is `history` with the tool and output instructions added. It is what reuse is
    matched on and what is stored, system prompt included, so changing or removing a system
    prompt never continues a chat opened under another one.

    Raises `HTTPException`: 400 when there is nothing new to send, 503 when no client could take
    the request, 502 when Gemini fails.
    """
    system_prompt, without_system = _extract_leading_system_prompt(history)
    if not without_system:
        system_prompt = None  # a request that is only a system prompt keeps it as its message
    needs_upload = _requires_upload(prepared)
    session, client, remain, stored_conv = await _find_reusable_session(
        db, pool, resolved_model, prepared, require_account=needs_upload
    )

    if session and client:
        if not remain:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No new messages.")
        if gem_id := await _resolve_system_prompt_gem(client, system_prompt):
            session.gem = gem_id
        to_send = _prepare_messages_for_model(
            remain, tools, tool_choice, instructions if repeat_instructions else None, False
        )
        m_input, files = await GeminiClientWrapper.process_conversation(to_send, tmp_dir)
        logger.debug(
            f"Reused session {reprlib.repr(session.metadata)} - sending {len(to_send)} prepared messages."
        )
    else:
        try:
            client = await pool.acquire(require_account=needs_upload)
            gem_id = await _resolve_system_prompt_gem(client, system_prompt)
            # A system prompt carried by a gem is left out of the message itself.
            to_send = (
                _prepare_messages_for_model(without_system, tools, tool_choice, instructions)
                if gem_id
                else prepared
            )
            session = client.start_chat(model=client.usable_model(resolved_model), gem=gem_id)
            m_input, files = await GeminiClientWrapper.process_conversation(to_send, tmp_dir)
        except Exception as e:
            logger.error(f"Error in preparing conversation: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            ) from e

    try:
        logger.debug(
            f"Client ID: {client.id}, Input length: {len(m_input)}, files count: {len(files)}"
        )
        return await _send_with_internal_fallback(
            pool=pool,
            db=db,
            resolved_model=resolved_model,
            session=session,
            client=client,
            current_input=m_input,
            files=files,
            full_prepared_messages=prepared,
            stored_conversation=stored_conv,
            tmp_dir=tmp_dir,
            stream=stream,
        )
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e


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


async def _process_image_item(image: Image) -> ProcessedImageResult | None:
    """Download one generated image into the media store; None if that fails."""
    try:
        return "image", image, await _image_to_base64(image, get_media_store_dir())
    except Exception as exc:
        logger.warning(f"Background image processing failed: {exc}")
        return None


async def _process_media_item(
    media_item: GeneratedVideo | GeneratedMedia,
) -> ProcessedMediaResult | None:
    """Download one generated video or audio item into the media store; None if that fails."""
    try:
        return "media", media_item, await _media_to_local_file(media_item, get_media_store_dir())
    except Exception as exc:
        logger.warning(f"Background media processing failed: {exc}")
        return None


class _Downloads:
    """The images, videos and audio generated for one reply, downloaded in the background.

    A download starts as soon as a chunk mentions the file, so most are finished by the time the
    text has been sent. Gemini can list one file under several urls; it is used once.
    """

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task[ProcessedImageResult | ProcessedMediaResult | None]] = []
        self._seen_urls: set[str] = set()
        self._image_hashes: set[str] = set()
        self.kept_media: dict[
            str, str
        ] = {}  # content hash -> file kept for it, for `_media_markdown`

    def add(self, output: ModelOutput) -> None:
        """Start downloading what this chunk mentions for the first time."""
        for image in output.images or []:
            if image.url and image.url not in self._seen_urls:
                self._seen_urls.add(image.url)
                self._tasks.append(asyncio.create_task(_process_image_item(image)))
        for item in (output.videos or []) + (output.media or []):
            url = getattr(item, "url", None) or getattr(item, "mp3_url", None)
            if url and url not in self._seen_urls:
                self._seen_urls.add(url)
                self._tasks.append(asyncio.create_task(_process_media_item(item)))

    async def cancel(self) -> None:
        """Stop the downloads still running, on a path that will not use their results.

        Left alone they keep writing files nothing refers to and, if they fail, raise
        `Task exception was never retrieved` once they are garbage collected.
        """
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def as_completed(
        self,
    ) -> AsyncGenerator[ProcessedImageResult | ProcessedMediaResult | None]:
        """Each download as it finishes, and None after every 5 seconds without one, so that a
        stream can send a keep-alive. Failed downloads are skipped."""
        while self._tasks:
            done, _ = await asyncio.wait(
                self._tasks, timeout=5.0, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                yield None
                continue
            finished = [task for task in self._tasks if task in done]
            self._tasks = [task for task in self._tasks if task not in done]
            for task in finished:
                if result := task.result():
                    yield result

    async def all(self) -> list[ProcessedImageResult | ProcessedMediaResult]:
        """Every download that succeeded, in the order the files were listed."""
        tasks, self._tasks = self._tasks, []
        return [result for result in await asyncio.gather(*tasks) if result]

    def new_image(self, data: ProcessedImageData) -> str | None:
        """File name of a downloaded image, or None for a second copy of a picture this reply
        already has; the copy is deleted."""
        *_, name, content_hash = data
        if content_hash in self._image_hashes:
            (get_media_store_dir() / name).unlink(missing_ok=True)
            return None
        self._image_hashes.add(content_hash)
        return name


def _media_url(name: str, base_url: str) -> str:
    """The signed link under which a file in the media store is served."""
    return f"{base_url}media/{name}?token={get_media_token(name)}"


def _media_markdown(
    item: object,  # only its title is used
    files: ProcessedMediaData,
    seen_hashes: dict[str, str],
    base_url: str,
) -> str:
    """Markdown links to one generated video or audio item, or "" if it has nothing new to show.

    `seen_hashes` maps content hash to the file this response already keeps for it. A second copy
    of the same content is deleted; a video or audio track that is such a copy is not linked
    again, and a thumbnail that is one reuses the file kept.
    """
    media_store = get_media_store_dir()
    urls: dict[str, str] = {}
    for kind, (name, content_hash) in files.items():
        kept = seen_hashes.setdefault(content_hash, name)
        if kept != name:
            (media_store / name).unlink(missing_ok=True)
            if kind in ("video", "audio"):
                continue
        urls[kind] = _media_url(kept, base_url)

    title = getattr(item, "title", "Media")
    thumbnail = urls.get("video_thumbnail") or urls.get("audio_thumbnail")
    links = []
    for kind, label in (("video", title), ("audio", f"{title} - Audio")):
        if url := urls.get(kind):
            links.append(f"[![{label}]({thumbnail})]({url})" if thumbnail else f"[{label}]({url})")
    return "\n\n".join(links)


def _chat_markdown(
    result: ProcessedImageResult | ProcessedMediaResult, downloads: _Downloads, base_url: str
) -> str:
    """Markdown that shows one downloaded file in a Chat Completions reply; "" for a repeat."""
    kind, item, data = result
    if kind == "media":
        files = cast(ProcessedMediaData, data)
        return _media_markdown(item, files, downloads.kept_media, base_url)
    if name := downloads.new_image(cast(ProcessedImageData, data)):
        return f"![{getattr(item, 'title', 'Image')}]({_media_url(name, base_url)})"
    return ""


def _image_generation_call(data: ProcessedImageData) -> ImageGenerationCall:
    """A downloaded image as a Responses output item."""
    b64, width, height, name, _ = data
    return ImageGenerationCall(
        id=Path(name).stem,
        result=b64,
        output_format=Path(name).suffix.lstrip("."),
        size=f"{width}x{height}" if width and height else None,
    )


# --- Response Builders & Streaming ---


def _unsent_tail(final: str | None, sent: str) -> str:
    """What the final chunk carries beyond the deltas added up so far.

    The rest is sent on, so that what gets validated and stored is what the client received.
    "" when there is nothing more, or when the final text does not continue what was sent.
    """
    if final and len(final) > len(sent) and final.startswith(sent):
        return final[len(sent) :]
    return ""


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
    generator: AsyncGenerator[ModelOutput],
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

        downloads = _Downloads()

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

                    downloads.add(chunk)
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                yield _sse_error(f"Streaming error occurred: {e}", "server_error")
                return

            if last_output is not None:
                if tail := _unsent_tail(last_output.thoughts, full_thoughts):
                    full_thoughts += tail
                    yield make_chunk({"delta": {"reasoning_content": tail}, "finish_reason": None})
                if tail := _unsent_tail(last_output.text, full_text):
                    full_text += tail
                    if not structured_requirement and (visible_tail := suppressor.process(tail)):
                        yield make_chunk(
                            {"delta": {"content": visible_tail}, "finish_reason": None}
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
                yield _sse_error(
                    str(exc), "invalid_model_output", "response_format", "schema_validation_failed"
                )
                return
            # No `has_images` escape hatch here: Chat Completions has no image-generation tool, so an
            # image Gemini volunteers on its own cannot stand in for a function call that was forced.
            if choice_error := _tool_choice_failure(tool_choice, detected_tool_calls):
                yield _sse_error(
                    choice_error, "invalid_model_output", "tool_choice", "required_tool_missing"
                )
                return
            if structured_requirement and visible_output:
                yield make_chunk({"delta": {"content": visible_output}, "finish_reason": None})

            async for result in downloads.as_completed():
                if result is None:
                    yield ": ping\n\n"
                elif md := _chat_markdown(result, downloads, base_url):
                    storage_output += f"\n\n{md}"
                    yield make_chunk({"delta": {"content": f"\n\n{md}"}, "finish_reason": None})

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
            await downloads.cancel()

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _create_responses_real_streaming_response(
    generator: AsyncGenerator[ModelOutput],
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

        def make_event(etype: str, fields: dict) -> str:
            nonlocal seq
            data = {**base_event, "type": etype, **fields, "sequence_number": seq}
            seq += 1
            return f"event: {etype}\ndata: {orjson.dumps(data).decode()}\n\n"

        downloads = _Downloads()

        try:
            yield make_event(
                "response.created",
                {
                    "response": {
                        **base_event,
                        "status": "in_progress",
                        "metadata": request.metadata or {},
                        "input": None,
                        **_echoed_tools(request),
                        "output": [],
                        "usage": None,
                    },
                },
            )
            yield make_event(
                "response.in_progress",
                {
                    "response": {
                        **base_event,
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

            def open_thought() -> list[str]:
                """Events that start the reasoning item, unless one is open."""
                nonlocal thought_open, thought_index, next_output_index
                if thought_open:
                    return []
                thought_open = True
                thought_index = next_output_index
                next_output_index += 1
                return [
                    make_event(
                        "response.output_item.added",
                        {
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
                    ),
                    make_event(
                        "response.reasoning_summary_part.added",
                        {
                            "item_id": thought_item_id,
                            "output_index": thought_index,
                            "summary_index": 0,
                            "part": dump_model(SummaryTextContent(text="")),
                        },
                    ),
                ]

            def thought_delta(delta: str) -> str:
                return make_event(
                    "response.reasoning_summary_text.delta",
                    {
                        "item_id": thought_item_id,
                        "output_index": thought_index,
                        "summary_index": 0,
                        "delta": delta,
                    },
                )

            def close_thought() -> list[str]:
                """Events that finish the reasoning item, if one is open."""
                nonlocal thought_open
                if not thought_open:
                    return []
                thought_open = False
                return [
                    make_event(
                        "response.reasoning_summary_text.done",
                        {
                            "item_id": thought_item_id,
                            "output_index": thought_index,
                            "summary_index": 0,
                            "text": full_thoughts,
                        },
                    ),
                    make_event(
                        "response.reasoning_summary_part.done",
                        {
                            "item_id": thought_item_id,
                            "output_index": thought_index,
                            "summary_index": 0,
                            "part": dump_model(SummaryTextContent(text=full_thoughts)),
                        },
                    ),
                    make_event(
                        "response.output_item.done",
                        {
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
                    ),
                ]

            def open_message() -> list[str]:
                """Events that start the assistant message, unless it is open."""
                nonlocal message_open, message_index, next_output_index
                if message_open:
                    return []
                message_open = True
                message_index = next_output_index
                next_output_index += 1
                return [
                    make_event(
                        "response.output_item.added",
                        {
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
                    ),
                    make_event(
                        "response.content_part.added",
                        {
                            "item_id": message_item_id,
                            "output_index": message_index,
                            "content_index": 0,
                            "part": dump_model(ResponseOutputText(type="output_text", text="")),
                        },
                    ),
                ]

            def text_delta(delta: str) -> str:
                return make_event(
                    "response.output_text.delta",
                    {
                        "item_id": message_item_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "delta": delta,
                        "logprobs": [],
                    },
                )

            try:
                async for chunk in generator:
                    last_output = chunk

                    if chunk.thoughts_delta:
                        for event in open_thought():
                            yield event

                        full_thoughts += chunk.thoughts_delta
                        yield thought_delta(chunk.thoughts_delta)

                    if chunk.text_delta:
                        full_text += chunk.text_delta
                        for event in close_thought():
                            yield event

                        if not structured_requirement:
                            for event in open_message():
                                yield event

                            if visible := suppressor.process(chunk.text_delta):
                                yield text_delta(visible)

                    downloads.add(chunk)

            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                yield make_event(
                    "error",
                    {
                        "error": {"message": f"Streaming error occurred: {e}"},
                    },
                )
                return

            if last_output is not None:
                if tail := _unsent_tail(last_output.thoughts, full_thoughts):
                    full_thoughts += tail
                    for event in open_thought():
                        yield event
                    yield thought_delta(tail)
                if tail := _unsent_tail(last_output.text, full_text):
                    full_text += tail
                    if not structured_requirement and (visible := suppressor.process(tail)):
                        for event in open_message():
                            yield event
                        yield text_delta(visible)

            if not structured_requirement and (remaining := suppressor.flush()):
                # Text that arrived only with the final chunk can still be held back here.
                for event in open_message():
                    yield event
                yield text_delta(remaining)

            for event in close_thought():
                yield event

            try:
                _, assistant_text, storage_output, detected_tool_calls = process_llm_output(
                    normalize_llm_text(full_thoughts or ""),
                    normalize_llm_text(full_text or ""),
                    structured_requirement,
                )
            except StructuredOutputValidationError as exc:
                yield make_event(
                    "error",
                    {
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
                for event in open_message():
                    yield event
                yield text_delta(assistant_text)

            image_items = []
            async for result in downloads.as_completed():
                if result is None:
                    yield ": ping\n\n"
                    continue
                kind, item, data = result
                if kind == "media":
                    files = cast(ProcessedMediaData, data)
                    link = _media_markdown(item, files, downloads.kept_media, base_url)
                elif name := downloads.new_image(cast(ProcessedImageData, data)):
                    img_item = _image_generation_call(cast(ProcessedImageData, data))
                    img_index = next_output_index
                    next_output_index += 1
                    for etype in ("response.output_item.added", "response.output_item.done"):
                        yield make_event(
                            etype, {"output_index": img_index, "item": dump_model(img_item)}
                        )
                    image_items.append(img_item)
                    link = f"![{name}]({_media_url(name, base_url)})"
                else:
                    continue
                if link:
                    for event in open_message():
                        yield event
                    yield text_delta(f"\n\n{link}")
                    assistant_text += f"\n\n{link}"
                    storage_output += f"\n\n{link}"

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
                text_part = ResponseOutputText(type="output_text", text=assistant_text)
                final_response_contents = [text_part]

                yield make_event(
                    "response.output_text.done",
                    {
                        "item_id": message_item_id,
                        "output_index": message_index,
                        "content_index": 0,
                    },
                )
                yield make_event(
                    "response.content_part.done",
                    {
                        "item_id": message_item_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "part": dump_model(text_part),
                    },
                )

                yield make_event(
                    "response.output_item.done",
                    {
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
                        "output_index": tc_index,
                        "item": dump_model(tc_item),
                    },
                )
                yield make_event(
                    "response.output_item.done",
                    {
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
                    "response": dump_model(payload),
                },
            )

            yield "data: [DONE]\n\n"
        finally:
            await downloads.cancel()

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# --- Main Router Endpoints ---


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    pool = GeminiClientPool()
    models = available_models(pool)
    return ModelListResponse(data=models)


@router.post("/v1/chat/completions")
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

    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(datetime.now(tz=UTC).timestamp())
    resp_or_stream, session, client = await _send_request(
        pool=pool,
        db=db,
        resolved_model=resolved_model,
        history=app_messages,
        prepared=msgs,
        tools=request.tools,
        tool_choice=request.tool_choice,
        instructions=extra_instr,
        repeat_instructions=True,
        tmp_dir=tmp_dir,
        stream=bool(request.stream),
    )

    if not isinstance(resp_or_stream, ModelOutput):  # which is what `stream` asked for
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

    try:
        thoughts, visible_output, storage_output, tool_calls = process_llm_output(
            normalize_llm_text(resp_or_stream.thoughts or ""),
            normalize_llm_text(resp_or_stream.text or ""),
            structured_requirement,
        )
    except StructuredOutputValidationError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    # No `has_images` escape hatch here: Chat Completions has no image-generation tool, so an
    # image Gemini volunteers on its own cannot stand in for a function call that was forced.
    if choice_error := _tool_choice_failure(request.tool_choice, tool_calls):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=choice_error)

    downloads = _Downloads()
    downloads.add(resp_or_stream)
    for result in await downloads.all():
        if md := _chat_markdown(result, downloads, base_url):
            visible_output += f"\n\n{md}"
            storage_output += f"\n\n{md}"

    p_tok, c_tok, t_tok, r_tok = calculate_usage(app_messages, storage_output, tool_calls, thoughts)
    payload = ChatCompletionResponse(
        id=completion_id,
        created=created_time,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    content=visible_output or None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id=call.id,
                            type="function",
                            function=FunctionCall(
                                name=call.function.name, arguments=call.function.arguments
                            ),
                        )
                        for call in tool_calls
                    ]
                    or None,
                    reasoning_content=thoughts or None,
                ),
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
            completion_tokens_details={"reasoning_tokens": r_tok},
        ),
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


@router.post("/v1/responses")
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

    standard_tools = [t for t in request.tools or [] if isinstance(t, FunctionTool)]
    image_tools = [t for t in request.tools or [] if isinstance(t, ImageGeneration)]

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
    conv_messages = base_messages
    if request.instructions:
        conv_messages = [AppMessage(role="system", content=request.instructions), *base_messages]
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

    response_id = f"resp_{uuid.uuid4().hex}"
    created_time = int(datetime.now(tz=UTC).timestamp())
    resp_or_stream, session, client = await _send_request(
        pool=pool,
        db=db,
        resolved_model=resolved_model,
        history=conv_messages,
        prepared=messages,
        tools=standard_tools or None,
        tool_choice=model_tool_choice,
        instructions=extra_instr or None,
        # Unlike Chat Completions, a continued chat is not sent the instructions again.
        repeat_instructions=False,
        tmp_dir=tmp_dir,
        stream=bool(request.stream),
    )

    if not isinstance(resp_or_stream, ModelOutput):  # which is what `stream` asked for
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

    try:
        thoughts, assistant_text, storage_output, tool_calls = process_llm_output(
            normalize_llm_text(resp_or_stream.thoughts or ""),
            normalize_llm_text(resp_or_stream.text or ""),
            structured_requirement,
        )
    except StructuredOutputValidationError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    if choice_error := _tool_choice_failure(
        request.tool_choice,
        tool_calls,
        has_images=bool(resp_or_stream.images),
        has_image_tool=bool(image_tools),
    ):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=choice_error)

    downloads = _Downloads()
    downloads.add(resp_or_stream)
    img_calls: list[ImageGenerationCall] = []
    image_markdown = media_markdown = ""
    for kind, item, data in await downloads.all():
        if kind == "media":
            files = cast(ProcessedMediaData, data)
            if md := _media_markdown(item, files, downloads.kept_media, base_url):
                media_markdown += f"\n\n{md}"
        elif name := downloads.new_image(cast(ProcessedImageData, data)):
            img_calls.append(_image_generation_call(cast(ProcessedImageData, data)))
            image_markdown += f"\n\n![{img_calls[-1].id}]({_media_url(name, base_url)})"

    # Text, image links and media links are three parts here, and one in a stream.
    contents: list[ResponseOutputContent] = [
        ResponseOutputText(type="output_text", text=text)
        for text in (assistant_text, image_markdown, media_markdown)
        if text
    ] or [ResponseOutputText(type="output_text", text="")]
    storage_output += image_markdown + media_markdown

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
