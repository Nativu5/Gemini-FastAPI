import asyncio
import base64
import hashlib
import io
import reprlib
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from gemini_webapi import ModelOutput
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from gemini_webapi.types.image import GeneratedImage, Image
from gemini_webapi.types.video import GeneratedMedia, GeneratedVideo
from loguru import logger

from app.models import (
    AppContentItem,
    AppMessage,
    AppToolCall,
    AppToolCallFunction,
    ChatCompletionChoice,
    ChatCompletionFunctionTool,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionNamedToolChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    ImageGeneration,
    ImageGenerationCall,
    ModelData,
    ModelListResponse,
    ResponseCreateRequest,
    ResponseCreateResponse,
    ResponseFormatTextJSONSchemaConfig,
    ResponseFunctionToolCall,
    ResponseInputMessage,
    ResponseOutputContent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextConfig,
    ResponseUsage,
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
from app.utils.helper import (
    STREAM_MASTER_RE,
    STREAM_TAIL_RE,
    TOOL_HINT_STRIPPED,
    TOOL_WRAP_HINT,
    detect_image_extension,
    estimate_tokens,
    extract_image_dimensions,
    extract_tool_calls,
    normalize_llm_text,
    strip_system_hints,
    text_from_message,
)

MAX_CHARS_PER_REQUEST = int(g_config.gemini.max_chars_per_request * 0.9)

router = APIRouter()


@dataclass
class StructuredOutputRequirement:
    """Represents a structured response request from the client."""

    schema_name: str
    schema: dict[str, Any]
    instruction: str
    raw_format: dict[str, Any]


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
            suffix = original_path.suffix
            if not suffix:
                suffix = default_extensions.get(mtype) or (".mp4" if "video" in mtype else ".mp3")

            random_name = f"media_{uuid.uuid4().hex}{suffix}"
            new_path = temp_dir / random_name
            original_path.rename(new_path)

            fhash = hashlib.sha256(data).hexdigest()
            results[mtype] = (random_name, fhash)
            path_map[spath] = (random_name, fhash)
        except Exception as e:
            logger.warning(f"Error processing {mtype} at {spath}: {e}")

    return results


def _calculate_usage(
    messages: list[AppMessage],
    assistant_text: str | None,
    tool_calls: list[AppToolCall] | None,
    thoughts: str | None = None,
) -> tuple[int, int, int, int]:
    """Calculate prompt, completion, total and reasoning tokens consistently."""
    prompt_tokens = sum(estimate_tokens(text_from_message(msg)) for msg in messages)
    tool_args_text = ""
    if tool_calls:
        for call in tool_calls:
            tool_args_text += call.function.arguments or ""

    completion_basis = assistant_text or ""
    if tool_args_text:
        completion_basis = (
            f"{completion_basis}\n{tool_args_text}" if completion_basis else tool_args_text
        )

    completion_tokens = estimate_tokens(completion_basis)
    reasoning_tokens = estimate_tokens(thoughts) if thoughts else 0
    total_completion_tokens = completion_tokens + reasoning_tokens

    return (
        prompt_tokens,
        total_completion_tokens,
        prompt_tokens + total_completion_tokens,
        reasoning_tokens,
    )


def _create_responses_standard_payload(
    response_id: str,
    created_time: int,
    model_name: str,
    detected_tool_calls: list[AppToolCall] | None,
    image_call_items: list[ImageGenerationCall],
    response_contents: list[ResponseOutputContent],
    usage: ResponseUsage,
    request: ResponseCreateRequest,
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

    text_config = ResponseTextConfig()
    if request.response_format and request.response_format.get("type") == "json_schema":
        text_config.format = ResponseFormatTextJSONSchemaConfig()

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
        tools=request.tools or [],
        tool_choice=request.tool_choice if request.tool_choice is not None else "auto",
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


def _process_llm_output(
    thoughts: str | None,
    raw_text: str,
    structured_requirement: StructuredOutputRequirement | None,
) -> tuple[str | None, str, str, list[AppToolCall]]:
    """
    Post-process Gemini output to extract tool calls and prepare clean text for display and storage.
    Returns: (thoughts, visible_text, storage_output, tool_calls)
    """
    if thoughts:
        thoughts = thoughts.strip()

    visible_output, tool_calls = extract_tool_calls(raw_text)
    if tool_calls:
        logger.debug(f"Detected {len(tool_calls)} tool call(s) in model output.")

    visible_output = visible_output.strip()
    storage_output = visible_output

    if structured_requirement and visible_output:
        try:
            structured_payload = orjson.loads(visible_output)
            canonical_output = orjson.dumps(structured_payload).decode("utf-8")
            visible_output = canonical_output
            storage_output = canonical_output
            logger.debug(
                f"Structured response fulfilled (schema={structured_requirement.schema_name})."
            )
        except orjson.JSONDecodeError:
            logger.warning(
                f"Failed to decode JSON for structured response (schema={structured_requirement.schema_name})."
            )

    return thoughts, visible_output, storage_output, tool_calls


def _convert_to_app_messages(messages: list[ChatCompletionMessage]) -> list[AppMessage]:
    """Convert ChatCompletionMessage (OpenAI format) to generic internal AppMessage."""
    app_messages = []
    for msg in messages:
        app_content = None
        if isinstance(msg.content, str):
            app_content = msg.content
        elif isinstance(msg.content, list):
            app_content = []
            for item in msg.content:
                if item.type == "text":
                    app_content.append(AppContentItem(type="text", text=item.text))
                elif item.type == "image_url":
                    media_dict = getattr(item, "image_url", None)
                    url = media_dict.get("url") if media_dict else None
                    if url and url.startswith("data:"):
                        # image_url can be either a regular url or base64 data url
                        app_content.append(AppContentItem(type="image_url", url=url))
                    else:
                        app_content.append(AppContentItem(type="image_url", url=url))
                elif item.type == "file":
                    file_dict = getattr(item, "file", None)
                    filename = file_dict.get("filename") if file_dict else None
                    file_data = file_dict.get("file_data") if file_dict else None
                    app_content.append(
                        AppContentItem(type="file", filename=filename, file_data=file_data)
                    )
                elif item.type == "input_audio":
                    audio_dict = getattr(item, "input_audio", None)
                    audio_data = audio_dict.get("data") if audio_dict else None
                    app_content.append(
                        AppContentItem(
                            type="input_audio",
                            file_data=audio_data,
                            raw_data=audio_dict,
                        )
                    )
                elif item.type in ("refusal", "reasoning"):
                    text_val = getattr(item, "text", None) or getattr(item, item.type, None)
                    app_content.append(AppContentItem(type=item.type, text=text_val))

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                AppToolCall(
                    id=tc.id,
                    type="function",
                    function=AppToolCallFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in msg.tool_calls
            ]

        role = {"developer": "system", "function": "tool"}.get(msg.role, msg.role)
        if role not in ("system", "user", "assistant", "tool"):
            role = "system"

        app_messages.append(
            AppMessage(
                role=role,  # type: ignore
                content=app_content,
                tool_calls=tool_calls,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
                reasoning_content=getattr(msg, "reasoning_content", None),
            )
        )
    return app_messages


def _persist_conversation(
    db: LMDBConversationStore,
    model_name: str,
    client_id: str,
    metadata: list[str | None],
    messages: list[AppMessage],
    storage_output: str | None,
    tool_calls: list[AppToolCall] | None,
) -> str | None:
    """Unified logic to save conversation history to LMDB."""
    try:
        current_assistant_message = AppMessage(
            role="assistant",
            content=storage_output or None,
            tool_calls=tool_calls or None,
            reasoning_content=None,
        )
        full_history = [*messages, current_assistant_message]

        db.store(
            client_id=client_id,
            model=model_name,
            messages=full_history,
            metadata=metadata,
        )
        logger.debug("Conversation saved to LMDB.")
        return "success"
    except Exception as e:
        logger.warning(f"Failed to save {len(messages) + 1} messages to LMDB: {e}")
        return None


def _build_structured_requirement(
    response_format: dict[str, Any] | None,
) -> StructuredOutputRequirement | None:
    """Translate OpenAI-style response_format into internal instructions."""
    if not response_format or not isinstance(response_format, dict):
        return None

    if response_format.get("type") != "json_schema":
        logger.warning(
            f"Unsupported response_format type requested: {reprlib.repr(response_format)}"
        )
        return None

    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        logger.warning(
            f"Invalid json_schema payload in response_format: {reprlib.repr(response_format)}"
        )
        return None

    schema = json_schema.get("schema")
    if not isinstance(schema, dict):
        logger.warning(
            f"Missing `schema` object in response_format payload: {reprlib.repr(response_format)}"
        )
        return None

    schema_name = json_schema.get("name") or "response"
    strict = json_schema.get("strict", True)

    pretty_schema = orjson.dumps(schema, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    instruction_parts = [
        "You must respond with a single valid JSON document that conforms to the schema shown below.",
        "Do not include explanations, comments, or any text before or after the JSON.",
        f'Schema name: "{schema_name}"',
        "JSON Schema:",
        pretty_schema,
    ]
    if not strict:
        instruction_parts.insert(
            1,
            "The schema allows unspecified fields, but include only what is necessary to satisfy the user's request.",
        )

    instruction = "\n\n".join(instruction_parts)
    return StructuredOutputRequirement(
        schema_name=schema_name,
        schema=schema,
        instruction=instruction,
        raw_format=response_format,
    )


def _build_tool_prompt(
    tools: list[ChatCompletionFunctionTool],
    tool_choice: (
        Literal["none", "auto", "required"]
        | ChatCompletionNamedToolChoice
        | ToolChoiceFunction
        | ToolChoiceTypes
        | None
    ),
) -> str:
    """Generate a system prompt describing available tools and the PascalCase protocol."""
    if not tools:
        return ""

    lines: list[str] = [
        "SYSTEM INTERFACE: You have access to the following technical tools. You MUST invoke them when necessary to fulfill the request, strictly adhering to the provided JSON schemas."
    ]

    for tool in tools:
        function = tool.function
        description = function.description or "No description provided."
        lines.append(f"Tool `{function.name}`: {description}")
        if function.parameters:
            schema_text = orjson.dumps(function.parameters, option=orjson.OPT_SORT_KEYS).decode(
                "utf-8"
            )
            lines.append("Arguments JSON schema:")
            lines.append(schema_text)
        else:
            lines.append("Arguments JSON schema: {}")

    if tool_choice == "none":
        lines.append(
            "For this request you must not call any tool. Provide the best possible natural language answer."
        )
    elif tool_choice == "required":
        lines.append(
            "You must call at least one tool before responding to the user. Do not provide a final user-facing answer until a tool call has been issued."
        )
    elif isinstance(tool_choice, ChatCompletionNamedToolChoice):
        target = tool_choice.function.name
        lines.append(
            f"You are required to call the tool named `{target}`. Do not call any other tool."
        )

    lines.append(TOOL_WRAP_HINT)

    return "\n".join(lines)


def _build_image_generation_instruction(
    tools: list[ImageGeneration] | None,
    tool_choice: ToolChoiceFunction | None,
) -> str | None:
    """Construct explicit guidance so Gemini emits images when requested."""
    has_forced_choice = tool_choice is not None and tool_choice.type == "image_generation"
    primary = tools[0] if tools else None

    if not has_forced_choice and primary is None:
        return None

    instructions: list[str] = [
        "IMAGE GENERATION ENABLED: When an image is requested, you MUST return a real generated image directly.",
        "1. For new requests, generate new images matching the description immediately.",
        "2. For edits to existing images, apply changes and return a new generated version.",
        "3. CRITICAL: Provide ZERO text explanation, prologue, or apologies. Do not describe the creation process.",
        "4. NEVER send placeholder text or descriptions like 'Generating image...' without an actual image attachment.",
    ]

    if has_forced_choice:
        instructions.append(
            "Image generation was explicitly requested. You MUST return at least one generated image. Any response without an image will be treated as a failure."
        )

    return "\n\n".join(instructions)


def _append_tool_hint_to_last_user_message(messages: list[AppMessage]) -> None:
    """Ensure the last user message carries the tool wrap hint."""
    for msg in reversed(messages):
        if msg.role != "user" or msg.content is None:
            continue

        if isinstance(msg.content, str):
            if TOOL_HINT_STRIPPED not in msg.content:
                msg.content = f"{msg.content}\n{TOOL_WRAP_HINT}"
            return

        if isinstance(msg.content, list):
            for part in reversed(msg.content):
                if getattr(part, "type", None) != "text":
                    continue
                text_value = getattr(part, "text", "") or ""
                if TOOL_HINT_STRIPPED in text_value:
                    return
                part.text = f"{text_value}\n{TOOL_WRAP_HINT}"
                return

            messages_text = TOOL_WRAP_HINT.strip()
            msg.content.append(AppContentItem(type="text", text=messages_text))
            return


def _prepare_messages_for_model(
    source_messages: list[AppMessage],
    tools: list[ChatCompletionFunctionTool] | None,
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
    if inject_system_defaults:
        if tools:
            tool_prompt = _build_tool_prompt(tools, tool_choice)
            if tool_prompt:
                instructions.append(tool_prompt)
                tool_prompt_injected = True

        if extra_instructions:
            instructions.extend(instr for instr in extra_instructions if instr)
            logger.debug(
                f"Applied {len(extra_instructions)} extra instructions for tool/structured output."
            )

    if not instructions:
        if tools and tool_choice != "none" and not tool_prompt_injected:
            _append_tool_hint_to_last_user_message(prepared)
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
        _append_tool_hint_to_last_user_message(prepared)

    return prepared


def _convert_responses_to_app_messages(
    items: Any,
) -> list[AppMessage]:
    """Convert Responses API input items into internal AppMessage objects."""
    messages: list[AppMessage] = []

    if isinstance(items, str):
        messages.append(AppMessage(role="user", content=items))
        logger.debug("Normalized Responses input: single string message.")
        return messages

    for item in items:
        if isinstance(item, (ResponseInputMessage, ResponseOutputMessage)):
            raw_role = getattr(item, "role", "user")
            normalized_role = {"developer": "system", "function": "tool"}.get(raw_role, raw_role)
            if normalized_role not in ("system", "user", "assistant", "tool"):
                normalized_role = "system"
            role = cast(Literal["system", "user", "assistant", "tool"], normalized_role)

            content = item.content
            if isinstance(content, str):
                messages.append(AppMessage(role=role, content=content))
            else:
                converted: list[AppContentItem] = []
                reasoning_parts: list[str] = []
                for part in content:
                    if part.type in ("input_text", "output_text"):
                        text_value = getattr(part, "text", "") or ""
                        if text_value:
                            converted.append(AppContentItem(type="text", text=text_value))
                    elif part.type == "reasoning_text":
                        text_value = getattr(part, "text", "") or ""
                        if text_value:
                            reasoning_parts.append(text_value)
                    elif part.type == "input_image":
                        image_url = getattr(part, "image_url", None)
                        if image_url:
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
            messages.append(
                AppMessage(
                    role="assistant",
                    tool_calls=[
                        AppToolCall(
                            id=item.call_id,
                            type="function",
                            function=AppToolCallFunction(name=item.name, arguments=item.arguments),
                        )
                    ],
                )
            )
        elif isinstance(item, FunctionCallOutput):
            output_content = str(item.output) if isinstance(item.output, list) else item.output
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
                normalized_role = {"developer": "system", "function": "tool"}.get(
                    raw_role, raw_role
                )
                if normalized_role not in ("system", "user", "assistant", "tool"):
                    normalized_role = "system"
                role = cast(Literal["system", "user", "assistant", "tool"], normalized_role)
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
            last_msg.content = merged_content if merged_content else None
            last_msg.tool_calls = merged_tools if merged_tools else None
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

        raw_role = instruction.role
        normalized_role = {"developer": "system", "function": "tool"}.get(raw_role, raw_role)
        if normalized_role not in ("system", "user", "assistant", "tool"):
            normalized_role = "system"
        role = cast(Literal["system", "user", "assistant", "tool"], normalized_role)

        content = instruction.content
        if isinstance(content, str):
            instruction_messages.append(AppMessage(role=role, content=content))
        else:
            converted: list[AppContentItem] = []
            for part in content:
                if part.type in ("input_text", "output_text"):
                    text_value = getattr(part, "text", "") or ""
                    if text_value:
                        converted.append(AppContentItem(type="text", text=text_value))
                elif part.type == "input_image":
                    image_url = getattr(part, "image_url", None)
                    if image_url:
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


def _get_model_by_name(name: str) -> Model:
    """Retrieve a Model instance by name."""
    strategy = g_config.gemini.model_strategy
    custom_models = {m.model_name: m for m in g_config.gemini.models if m.model_name}

    if name in custom_models:
        return Model.from_dict(custom_models[name].model_dump())

    if strategy == "overwrite":
        raise ValueError(f"Model '{name}' not found in custom models (strategy='overwrite').")

    return Model.from_name(name)


async def _get_available_models(pool: GeminiClientPool) -> list[ModelData]:
    """Return a list of available models based on the configuration strategy and per-client accounts."""
    now = int(datetime.now(tz=UTC).timestamp())
    strategy = g_config.gemini.model_strategy
    models_data = []
    seen_model_ids = set()

    for m in g_config.gemini.models:
        if m.model_name and m.model_name not in seen_model_ids:
            models_data.append(
                ModelData(
                    id=m.model_name,
                    created=now,
                    owned_by="custom",
                )
            )
            seen_model_ids.add(m.model_name)

    if strategy == "append":
        for client in pool.clients:
            if not client.running():
                continue

            client_models = client.list_models()
            if client_models:
                for model in client_models:
                    model_id = model.model_name if model.model_name else model.model_id
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


async def _find_reusable_session(
    db: LMDBConversationStore,
    pool: GeminiClientPool,
    model: Model,
    messages: list[AppMessage],
) -> tuple[ChatSession | None, GeminiClientWrapper | None, list[AppMessage]]:
    """Find an existing chat session matching the longest suitable history prefix."""
    if len(messages) < 2:
        return None, None, messages

    search_end = len(messages)
    while search_end >= 2:
        search_history = messages[:search_end]
        if search_history[-1].role in {"assistant", "system", "tool"}:
            try:
                if conv := db.find(model.model_name, search_history):
                    client = await pool.acquire(conv.client_id)
                    session = client.start_chat(metadata=conv.metadata, model=model)
                    remain = messages[search_end:]
                    logger.debug(
                        f"Match found at prefix length {search_end}/{len(messages)}. Client: {conv.client_id}"
                    )
                    return session, client, remain
            except Exception as e:
                logger.warning(
                    f"Error checking LMDB for reusable session at length {search_end}: {e}"
                )
                break
        search_end -= 1

    logger.debug(f"No reusable session found for {len(messages)} messages.")
    return None, None, messages


async def _send_with_split(
    session: ChatSession,
    text: str,
    files: list[Any] | None = None,
    stream: bool = False,
) -> AsyncGenerator[ModelOutput] | ModelOutput:
    """Send text to Gemini, splitting or converting to attachment if too long."""
    if len(text) <= MAX_CHARS_PER_REQUEST:
        try:
            if stream:
                return session.send_message_stream(text, files=files)
            return await session.send_message(text, files=files)
        except Exception as e:
            logger.error(f"Error sending message to Gemini: {e}")
            raise

    logger.info(
        f"Message length ({len(text)}) exceeds limit ({MAX_CHARS_PER_REQUEST}). Converting text to file attachment."
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
            return session.send_message_stream(instruction, files=final_files)
        return await session.send_message(instruction, files=final_files)
    except Exception as e:
        logger.error(f"Error sending large text as file to Gemini: {e}")
        raise


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
                if nl_idx != -1:
                    self.current_role = self.buffer[:nl_idx].strip().lower()
                    self.buffer = self.buffer[nl_idx + 1 :]
                    self.stack[-1] = "IN_BLOCK"
                    continue
                else:
                    break

            if self.state == "POST_BLOCK":
                stripped = self.buffer.lstrip()
                if not stripped:
                    break
                self.buffer = stripped
                self.stack[-1] = "NORMAL"

            match = STREAM_MASTER_RE.search(self.buffer)
            if not match:
                tail_match = STREAM_TAIL_RE.search(self.buffer)
                if tail_match:
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
            tail_match = STREAM_TAIL_RE.search(res)
            if tail_match:
                res = res[: -len(tail_match.group(0))]

        self.buffer = ""
        self.stack = ["NORMAL"]
        self.current_role = ""
        return strip_system_hints(res)


# --- Media Processing Helpers ---


async def _process_image_item(image: Image):
    """Process an image item by converting it to base64 and returning a standard result tuple."""
    try:
        media_store = get_media_store_dir()
        return "image", image, await _image_to_base64(image, media_store)
    except Exception as exc:
        logger.warning(f"Background image processing failed: {exc}")
        return None


async def _process_media_item(media_item: GeneratedVideo | GeneratedMedia):
    """Process a media item by saving it to a local file and returning a standard result tuple."""
    try:
        media_store = get_media_store_dir()
        return "media", media_item, await _media_to_local_file(media_item, media_store)
    except Exception as exc:
        logger.warning(f"Background media processing failed: {exc}")
        return None


# --- Response Builders & Streaming ---


def _create_real_streaming_response(
    resp_or_stream: AsyncGenerator[ModelOutput] | ModelOutput,
    completion_id: str,
    created_time: int,
    model_name: str,
    messages: list[AppMessage],
    db: LMDBConversationStore,
    model: Model,
    client_wrapper: GeminiClientWrapper,
    session: ChatSession,
    base_url: str,
    structured_requirement: StructuredOutputRequirement | None = None,
) -> StreamingResponse:
    """
    Create a real-time streaming response.
    Reconciles manual delta accumulation with the model's final authoritative state.
    """

    async def generate_stream():
        full_text = ""
        full_thoughts = ""
        has_started = False
        all_outputs: list[ModelOutput] = []
        suppressor = StreamingOutputFilter()

        media_tasks = []
        seen_media_urls = set()
        seen_image_urls = set()

        async def _make_async_gen(item: ModelOutput) -> AsyncGenerator[ModelOutput]:
            yield item

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
            if hasattr(resp_or_stream, "__aiter__"):
                generator = cast(AsyncGenerator[ModelOutput], resp_or_stream)
            else:
                generator = _make_async_gen(cast(ModelOutput, resp_or_stream))

            async for chunk in generator:
                all_outputs.append(chunk)
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
                    if visible_delta := suppressor.process(text_delta):
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
            yield f"data: {orjson.dumps({'error': {'message': f'Streaming error occurred: {e}', 'type': 'server_error', 'param': None, 'code': None}}).decode('utf-8')}\n\n"
            return

        if all_outputs:
            final_chunk = all_outputs[-1]
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
                    if visible_drift := suppressor.process(drift):
                        yield make_chunk(
                            {"delta": {"content": visible_drift}, "finish_reason": None}
                        )

        if remaining_text := suppressor.flush():
            yield make_chunk({"delta": {"content": remaining_text}, "finish_reason": None})

        _, _, storage_output, detected_tool_calls = _process_llm_output(
            normalize_llm_text(full_thoughts or ""),
            normalize_llm_text(full_text or ""),
            structured_requirement,
        )

        seen_hashes = {}
        seen_media_hashes = {}
        media_store = get_media_store_dir()

        if media_tasks:
            logger.debug(f"Waiting for {len(media_tasks)} background media tasks with heartbeat...")
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
                        yield make_chunk({"delta": {"content": f"\n\n{md}"}, "finish_reason": None})

                    elif rtype == "media":
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
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }

                yield make_chunk(
                    {
                        "delta": {
                            "tool_calls": [tc_dict],
                        },
                        "finish_reason": None,
                    }
                )

        p_tok, c_tok, t_tok, r_tok = _calculate_usage(
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
            model.model_name,
            client_wrapper.id,
            session.metadata,
            messages,
            storage_output,
            detected_tool_calls,
        )
        yield make_chunk(
            {
                "delta": {},
                "finish_reason": "tool_calls" if detected_tool_calls else "stop",
                "usage": usage.model_dump(mode="json"),
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _create_responses_real_streaming_response(
    resp_or_stream: AsyncGenerator[ModelOutput] | ModelOutput,
    response_id: str,
    created_time: int,
    model_name: str,
    messages: list[AppMessage],
    db: LMDBConversationStore,
    model: Model,
    client_wrapper: GeminiClientWrapper,
    session: ChatSession,
    request: ResponseCreateRequest,
    base_url: str,
    structured_requirement: StructuredOutputRequirement | None = None,
) -> StreamingResponse:
    """
    Create a real-time streaming response for the Responses API.
    Ensures final accumulated text and thoughts are synchronized and follow the formal event stream spec.
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
                    "tools": request.tools or [],
                    "tool_choice": request.tool_choice or "auto",
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
        media_tasks = []
        seen_media_urls = set()
        seen_image_urls = set()

        all_outputs: list[ModelOutput] = []

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
                all_outputs.append(chunk)

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
                                "item": ResponseReasoningItem(
                                    id=thought_item_id,
                                    type="reasoning",
                                    status="in_progress",
                                    summary=[],
                                ).model_dump(mode="json"),
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
                                "part": SummaryTextContent(text="").model_dump(mode="json"),
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
                                "part": SummaryTextContent(text=full_thoughts).model_dump(
                                    mode="json"
                                ),
                            },
                        )
                        yield make_event(
                            "response.output_item.done",
                            {
                                **base_event,
                                "type": "response.output_item.done",
                                "output_index": thought_index,
                                "item": ResponseReasoningItem(
                                    id=thought_item_id,
                                    type="reasoning",
                                    status="completed",
                                    summary=[SummaryTextContent(text=full_thoughts)],
                                ).model_dump(mode="json"),
                            },
                        )
                        thought_open = False

                    if not message_open:
                        message_index = next_output_index
                        next_output_index += 1
                        yield make_event(
                            "response.output_item.added",
                            {
                                **base_event,
                                "type": "response.output_item.added",
                                "output_index": message_index,
                                "item": ResponseOutputMessage(
                                    id=message_item_id,
                                    type="message",
                                    status="in_progress",
                                    role="assistant",
                                    content=[],
                                ).model_dump(mode="json"),
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
                                "part": ResponseOutputText(type="output_text", text="").model_dump(
                                    mode="json"
                                ),
                            },
                        )
                        message_open = True

                    full_text += chunk.text_delta
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
            yield make_event(
                "error",
                {
                    **base_event,
                    "type": "error",
                    "error": {"message": f"Streaming error occurred: {e}"},
                },
            )
            return

        if all_outputs:
            last = all_outputs[-1]
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
                                "item": ResponseReasoningItem(
                                    id=thought_item_id,
                                    type="reasoning",
                                    status="in_progress",
                                    summary=[],
                                ).model_dump(mode="json"),
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
                                "part": SummaryTextContent(text="").model_dump(mode="json"),
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
                    if visible := suppressor.process(drift):
                        if not message_open:
                            message_index = next_output_index
                            next_output_index += 1
                            yield make_event(
                                "response.output_item.added",
                                {
                                    **base_event,
                                    "type": "response.output_item.added",
                                    "output_index": message_index,
                                    "item": ResponseOutputMessage(
                                        id=message_item_id,
                                        type="message",
                                        status="in_progress",
                                        role="assistant",
                                        content=[],
                                    ).model_dump(mode="json"),
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
                                    "part": ResponseOutputText(
                                        type="output_text", text=""
                                    ).model_dump(mode="json"),
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

        remaining = suppressor.flush()
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
                    "part": SummaryTextContent(text=full_thoughts).model_dump(mode="json"),
                },
            )
            yield make_event(
                "response.output_item.done",
                {
                    **base_event,
                    "type": "response.output_item.done",
                    "output_index": thought_index,
                    "item": ResponseReasoningItem(
                        id=thought_item_id,
                        type="reasoning",
                        status="completed",
                        summary=[SummaryTextContent(text=full_thoughts)],
                    ).model_dump(mode="json"),
                },
            )

        _, assistant_text, storage_output, detected_tool_calls = _process_llm_output(
            normalize_llm_text(full_thoughts or ""),
            normalize_llm_text(full_text or ""),
            structured_requirement,
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

                        img_link = (
                            f"![{fname}]({base_url}media/{fname}?token={get_media_token(fname)})"
                        )
                        md_to_add = f"\n\n{img_link}"

                        img_index = next_output_index
                        next_output_index += 1
                        yield make_event(
                            "response.output_item.added",
                            {
                                **base_event,
                                "type": "response.output_item.added",
                                "output_index": img_index,
                                "item": img_item.model_dump(mode="json"),
                            },
                        )
                        yield make_event(
                            "response.output_item.done",
                            {
                                **base_event,
                                "type": "response.output_item.done",
                                "output_index": img_index,
                                "item": img_item.model_dump(mode="json"),
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
                                    "item": ResponseOutputMessage(
                                        id=message_item_id,
                                        type="message",
                                        status="in_progress",
                                        role="assistant",
                                        content=[],
                                    ).model_dump(mode="json"),
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
                                    "part": ResponseOutputText(
                                        type="output_text", text=""
                                    ).model_dump(mode="json"),
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
                                        "item": ResponseOutputMessage(
                                            id=message_item_id,
                                            type="message",
                                            status="in_progress",
                                            role="assistant",
                                            content=[],
                                        ).model_dump(mode="json"),
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
                                        "part": ResponseOutputText(
                                            type="output_text", text=""
                                        ).model_dump(mode="json"),
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
                    "part": ResponseOutputText(type="output_text", text=assistant_text).model_dump(
                        mode="json"
                    ),
                },
            )

            yield make_event(
                "response.output_item.done",
                {
                    **base_event,
                    "type": "response.output_item.done",
                    "output_index": message_index,
                    "item": ResponseOutputMessage(
                        id=message_item_id,
                        type="message",
                        status="completed",
                        role="assistant",
                        content=final_response_contents,
                    ).model_dump(mode="json"),
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
                    "item": tc_item.model_dump(mode="json"),
                },
            )
            yield make_event(
                "response.output_item.done",
                {
                    **base_event,
                    "type": "response.output_item.done",
                    "output_index": tc_index,
                    "item": tc_item.model_dump(mode="json"),
                },
            )

        p_tok, c_tok, t_tok, r_tok = _calculate_usage(
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
            full_thoughts,
            message_item_id,
            thought_item_id,
        )
        _persist_conversation(
            db,
            model.model_name,
            client_wrapper.id,
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
                "response": payload.model_dump(mode="json"),
            },
        )

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# --- Main Router Endpoints ---


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    pool = GeminiClientPool()
    models = await _get_available_models(pool)
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
        model = _get_model_by_name(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if not request.messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Messages required.")

    structured_requirement = _build_structured_requirement(request.response_format)
    extra_instr = [structured_requirement.instruction] if structured_requirement else None

    app_messages = _convert_to_app_messages(request.messages)

    msgs = _prepare_messages_for_model(
        app_messages,
        request.tools,
        request.tool_choice,
        extra_instr,
    )

    session, client, remain = await _find_reusable_session(db, pool, model, msgs)

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
            client = await pool.acquire()
            session = client.start_chat(model=model)
            m_input, files = await GeminiClientWrapper.process_conversation(msgs, tmp_dir)
        except Exception as e:
            logger.error(f"Error in preparing conversation: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            ) from e

    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(datetime.now(tz=UTC).timestamp())

    try:
        assert session and client
        logger.debug(
            f"Client ID: {client.id}, Input length: {len(m_input)}, files count: {len(files)}"
        )
        resp_or_stream = await _send_with_split(
            session, m_input, files=files, stream=bool(request.stream)
        )
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    if request.stream:
        assert not isinstance(resp_or_stream, ModelOutput)
        return _create_real_streaming_response(
            resp_or_stream,
            completion_id,
            created_time,
            request.model,
            msgs,
            db,
            model,
            client,
            session,
            base_url,
            structured_requirement,
        )

    assert isinstance(resp_or_stream, ModelOutput)

    thoughts, visible_output, storage_output, tool_calls = _process_llm_output(
        normalize_llm_text(resp_or_stream.thoughts or ""),
        normalize_llm_text(resp_or_stream.text or ""),
        structured_requirement,
    )

    images = resp_or_stream.images or []
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
            image_markdown += f"\n\n![{title}]({img_url})"

        elif rtype == "media":
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

    p_tok, c_tok, t_tok, r_tok = _calculate_usage(
        app_messages, storage_output, tool_calls, thoughts
    )
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
        model.model_name,
        client.id,
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
    base_messages = _convert_responses_to_app_messages(request.input)
    structured_requirement = _build_structured_requirement(request.response_format)
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

    img_instr = _build_image_generation_instruction(
        image_tools,
        request.tool_choice if isinstance(request.tool_choice, ToolChoiceFunction) else None,
    )
    if img_instr:
        extra_instr.append(img_instr)
    preface = _convert_instructions_to_app_messages(request.instructions)
    conv_messages = [*preface, *base_messages] if preface else base_messages
    model_tool_choice = (
        request.tool_choice
        if isinstance(request.tool_choice, (str, ChatCompletionNamedToolChoice))
        else None
    )

    messages = _prepare_messages_for_model(
        conv_messages,
        standard_tools or None,
        model_tool_choice,
        extra_instr or None,
    )
    pool, db = GeminiClientPool(), LMDBConversationStore()
    try:
        model = _get_model_by_name(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    session, client, remain = await _find_reusable_session(db, pool, model, messages)
    if session:
        msgs = _prepare_messages_for_model(
            remain,
            request.tools,  # type: ignore
            request.tool_choice,
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
            client = await pool.acquire()
            session = client.start_chat(model=model)
            m_input, files = await GeminiClientWrapper.process_conversation(messages, tmp_dir)
        except Exception as e:
            logger.error(f"Error in preparing conversation: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            ) from e

    response_id = f"resp_{uuid.uuid4().hex}"
    created_time = int(datetime.now(tz=UTC).timestamp())

    try:
        assert session and client
        logger.debug(
            f"Client ID: {client.id}, Input length: {len(m_input)}, files count: {len(files)}"
        )
        resp_or_stream = await _send_with_split(
            session, m_input, files=files, stream=bool(request.stream)
        )
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    if request.stream:
        assert not isinstance(resp_or_stream, ModelOutput)
        return _create_responses_real_streaming_response(
            resp_or_stream,
            response_id,
            created_time,
            request.model,
            messages,
            db,
            model,
            client,
            session,
            request,
            base_url,
            structured_requirement,
        )

    assert isinstance(resp_or_stream, ModelOutput)

    thoughts, assistant_text, storage_output, tool_calls = _process_llm_output(
        normalize_llm_text(resp_or_stream.thoughts or ""),
        normalize_llm_text(resp_or_stream.text or ""),
        structured_requirement,
    )
    images = resp_or_stream.images or []
    if (
        request.tool_choice is not None
        and hasattr(request.tool_choice, "type")
        and request.tool_choice.type == "image_generation"
    ) and not images:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No images returned.")

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
            img_calls.append(
                ImageGenerationCall(
                    id=img_id, result=b64, output_format=fmt, size=f"{w}x{h}" if w and h else None
                )
            )

        elif rtype == "media":
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

    p_tok, c_tok, t_tok, r_tok = _calculate_usage(messages, storage_output, tool_calls, thoughts)
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
        thoughts,
    )
    _persist_conversation(
        db,
        model.model_name,
        client.id,
        session.metadata,
        messages,
        storage_output,
        tool_calls,
    )
    return payload
