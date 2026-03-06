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
    remove_tool_call_blocks,
    strip_system_hints,
    text_from_message,
)

MAX_CHARS_PER_REQUEST = int(g_config.gemini.max_chars_per_request * 0.9)
METADATA_TTL_MINUTES = 15

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
    except Exception as e:
        logger.warning(f"Failed to save media: {e}")
        return {}

    results = {}
    for mtype, spath in saved_paths.items():
        if not spath:
            continue

        original_path = Path(spath)
        data = original_path.read_bytes()
        suffix = original_path.suffix

        if not suffix:
            suffix = ".mp4" if "video" in mtype else ".mp3"

        random_name = f"media_{uuid.uuid4().hex}{suffix}"
        new_path = temp_dir / random_name
        original_path.rename(new_path)

        fhash = hashlib.sha256(data).hexdigest()
        results[mtype] = (random_name, fhash)

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
) -> ResponseCreateResponse:
    """Unified factory for building ResponseCreateResponse objects."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    reason_id = f"rs_{uuid.uuid4().hex[:24]}"
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

    storage_output = remove_tool_call_blocks(raw_text)
    storage_output = storage_output.strip()

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

    custom_models = [m for m in g_config.gemini.models if m.model_name]
    for m in custom_models:
        models_data.append(
            ModelData(
                id=m.model_name or "",
                created=now,
                owned_by="custom",
            )
        )

    if strategy == "append":
        custom_ids = {m.id for m in models_data}
        seen_model_ids = set()

        for client in pool.clients:
            if not client.running():
                continue

            client_models = client.list_models()
            if client_models:
                for am in client_models:
                    if am.id not in custom_ids and am.id not in seen_model_ids:
                        models_data.append(
                            ModelData(
                                id=am.id,
                                created=now,
                                owned_by="gemini-web",
                            )
                        )
                        seen_model_ids.add(am.id)

        for model in Model:
            m_name = model.model_name
            if not m_name or m_name == "unspecified":
                continue
            if m_name in custom_ids or m_name in seen_model_ids:
                continue

            models_data.append(
                ModelData(
                    id=m_name,
                    created=now,
                    owned_by="gemini-web",
                )
            )

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
                    now = datetime.now()
                    updated_at = conv.updated_at or conv.created_at or now
                    age_minutes = (now - updated_at).total_seconds() / 60
                    if age_minutes <= METADATA_TTL_MINUTES:
                        client = await pool.acquire(conv.client_id)
                        session = client.start_chat(metadata=conv.metadata, model=model)
                        remain = messages[search_end:]
                        logger.debug(
                            f"Match found at prefix length {search_end}/{len(messages)}. Client: {conv.client_id}"
                        )
                        return session, client, remain
                    else:
                        logger.debug(
                            f"Matched conversation at length {search_end} is too old ({age_minutes:.1f}m), skipping reuse."
                        )
                else:
                    # Log that we tried this prefix but failed
                    pass
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
            logger.exception(f"Error sending message to Gemini: {e}")
            raise

    logger.info(
        f"Message length ({len(text)}) exceeds limit ({MAX_CHARS_PER_REQUEST}). Converting text to file attachment."
    )
    file_obj = io.BytesIO(text.encode("utf-8"))
    file_obj.name = "message.txt"
    try:
        final_files: list[Any] = list(files) if files else []
        final_files.append(file_obj)
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
        logger.exception(f"Error sending large text as file to Gemini: {e}")
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

            match = STREAM_MASTER_RE.search(self.buffer)
            if not match:
                tail_match = STREAM_TAIL_RE.search(self.buffer)
                keep_len = len(tail_match.group(0)) if tail_match else 0
                yield_len = len(self.buffer) - keep_len
                if yield_len > 0:
                    if self._is_outputting():
                        output.append(self.buffer[:yield_len])
                    self.buffer = self.buffer[yield_len:]
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

                if self.state == "NORMAL":
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
        full_thoughts, full_text = "", ""
        has_started = False
        all_outputs: list[ModelOutput] = []
        suppressor = StreamingOutputFilter()

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
        except Exception as e:
            logger.exception(f"Error during OpenAI streaming: {e}")
            yield f"data: {orjson.dumps({'error': {'message': 'Streaming error occurred.', 'type': 'server_error', 'param': None, 'code': None}}).decode('utf-8')}\n\n"
            return

        if all_outputs:
            final_chunk = all_outputs[-1]
            if final_chunk.text:
                full_text = final_chunk.text
            if final_chunk.thoughts:
                full_thoughts = final_chunk.thoughts

        if remaining_text := suppressor.flush():
            yield make_chunk({"delta": {"content": remaining_text}, "finish_reason": None})

        _thoughts, assistant_text, storage_output, detected_tool_calls = _process_llm_output(
            full_thoughts, full_text, structured_requirement
        )

        images = []
        seen_image_urls = set()
        media_items: list[GeneratedVideo | GeneratedMedia] = []
        seen_media_urls = set()

        for out in all_outputs:
            if out.images:
                for img in out.images:
                    if img.url not in seen_image_urls:
                        images.append(img)
                        seen_image_urls.add(img.url)

            m_list = (out.videos or []) + (out.media or [])
            for m in m_list:
                m_url = getattr(m, "url", None) or getattr(m, "mp3_url", None)
                if m_url and m_url not in seen_media_urls:
                    media_items.append(m)
                    seen_media_urls.add(m_url)

        image_results = []
        seen_hashes = set()
        for image in images:
            try:
                media_store = get_media_store_dir()
                _, _, _, fname, fhash = await _image_to_base64(image, media_store)
                if fhash in seen_hashes:
                    (media_store / fname).unlink(missing_ok=True)
                    continue
                seen_hashes.add(fhash)
                img_url = f"{base_url}media/{fname}?token={get_media_token(fname)}"
                title = getattr(image, "title", "Image")
                image_results.append(f"![{title}]({img_url})")
            except Exception as exc:
                logger.warning(f"Failed to process image in OpenAI stream: {exc}")

        media_results = []
        seen_media_hashes = set()
        for media_item in media_items:
            try:
                media_store = get_media_store_dir()
                m_dict = await _media_to_local_file(media_item, media_store)

                m_urls = {}
                for mtype, (random_name, fhash) in m_dict.items():
                    if fhash in seen_media_hashes:
                        (media_store / random_name).unlink(missing_ok=True)
                        continue
                    seen_media_hashes.add(fhash)
                    m_urls[mtype] = (
                        f"{base_url}media/{random_name}?token={get_media_token(random_name)}"
                    )

                media_url = m_urls.get("video") or m_urls.get("audio")
                thumb_url = m_urls.get("video_thumbnail") or m_urls.get("audio_thumbnail")

                title = getattr(media_item, "title", "Media")
                if thumb_url and media_url:
                    media_results.append(f"[![{title}]({thumb_url})]({media_url})")
                elif media_url:
                    media_results.append(f"[{title}]({media_url})")
                elif thumb_url:
                    media_results.append(f"![{title}]({thumb_url})")
            except Exception as exc:
                logger.warning(f"Failed to process media in OpenAI stream: {exc}")

        for image_url in image_results:
            yield make_chunk(
                {
                    "delta": {"content": f"\n\n{image_url}"},
                    "finish_reason": None,
                }
            )

        for media_md in media_results:
            yield make_chunk(
                {
                    "delta": {"content": f"\n\n{media_md}"},
                    "finish_reason": None,
                }
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
            messages, assistant_text, detected_tool_calls, full_thoughts
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
    media_store: Path,
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

        full_thoughts, full_text = "", ""
        all_outputs: list[ModelOutput] = []

        thought_item_id = f"rs_{uuid.uuid4().hex[:24]}"
        message_item_id = f"msg_{uuid.uuid4().hex[:24]}"

        thought_open, message_open = False, False
        current_index = 0
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
                        yield make_event(
                            "response.output_item.added",
                            {
                                **base_event,
                                "type": "response.output_item.added",
                                "output_index": current_index,
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
                                "output_index": current_index,
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
                            "output_index": current_index,
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
                                "output_index": current_index,
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
                                "output_index": current_index,
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
                                "output_index": current_index,
                                "item": ResponseReasoningItem(
                                    id=thought_item_id,
                                    type="reasoning",
                                    status="completed",
                                    summary=[SummaryTextContent(text=full_thoughts)],
                                ).model_dump(mode="json"),
                            },
                        )
                        current_index += 1
                        thought_open = False

                    if not message_open:
                        yield make_event(
                            "response.output_item.added",
                            {
                                **base_event,
                                "type": "response.output_item.added",
                                "output_index": current_index,
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
                                "output_index": current_index,
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
                                "output_index": current_index,
                                "content_index": 0,
                                "delta": visible,
                                "logprobs": [],
                            },
                        )

        except Exception:
            logger.exception("Responses streaming error")
            yield make_event(
                "error",
                {**base_event, "type": "error", "error": {"message": "Streaming error."}},
            )
            return

        if all_outputs:
            last = all_outputs[-1]
            if last.text:
                full_text = last.text
            if last.thoughts:
                full_thoughts = last.thoughts

        remaining = suppressor.flush()
        if remaining and message_open:
            yield make_event(
                "response.output_text.delta",
                {
                    **base_event,
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": current_index,
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
                    "output_index": current_index,
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
                    "output_index": current_index,
                    "summary_index": 0,
                    "part": SummaryTextContent(text=full_thoughts).model_dump(mode="json"),
                },
            )
            yield make_event(
                "response.output_item.done",
                {
                    **base_event,
                    "type": "response.output_item.done",
                    "output_index": current_index,
                    "item": ResponseReasoningItem(
                        id=thought_item_id,
                        type="reasoning",
                        status="completed",
                        summary=[SummaryTextContent(text=full_thoughts)],
                    ).model_dump(mode="json"),
                },
            )
            current_index += 1

        _thoughts, assistant_text, storage_output, detected_tool_calls = _process_llm_output(
            full_thoughts, full_text, structured_requirement
        )

        if message_open:
            yield make_event(
                "response.output_text.done",
                {
                    **base_event,
                    "type": "response.output_text.done",
                    "item_id": message_item_id,
                    "output_index": current_index,
                    "content_index": 0,
                },
            )
            yield make_event(
                "response.content_part.done",
                {
                    **base_event,
                    "type": "response.content_part.done",
                    "item_id": message_item_id,
                    "output_index": current_index,
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
                    "output_index": current_index,
                    "item": ResponseOutputMessage(
                        id=message_item_id,
                        type="message",
                        status="completed",
                        role="assistant",
                        content=[ResponseOutputText(type="output_text", text=assistant_text)],
                    ).model_dump(mode="json"),
                },
            )
            current_index += 1

        image_items: list[ImageGenerationCall] = []
        final_response_contents: list[ResponseOutputContent] = []
        seen_hashes = set()

        images = []
        seen_image_urls = set()
        media_items: list[GeneratedVideo | GeneratedMedia] = []
        seen_media_urls = set()

        for out in all_outputs:
            if out.images:
                for img in out.images:
                    if img.url not in seen_image_urls:
                        images.append(img)
                        seen_image_urls.add(img.url)

            m_list = (out.videos or []) + (out.media or [])
            for m in m_list:
                m_url = getattr(m, "url", None) or getattr(m, "mp3_url", None)
                if m_url and m_url not in seen_media_urls:
                    media_items.append(m)
                    seen_media_urls.add(m_url)

        for image in images:
            try:
                b64, w, h, fname, fhash = await _image_to_base64(image, media_store)
                if fhash in seen_hashes:
                    continue
                seen_hashes.add(fhash)

                parts = fname.rsplit(".", 1)
                img_id = parts[0]
                fmt = parts[1] if len(parts) > 1 else "png"

                img_item = ImageGenerationCall(
                    id=img_id,
                    result=b64,
                    output_format=fmt,
                    size=f"{w}x{h}" if w and h else None,
                )

                image_url = f"![{fname}]({base_url}media/{fname}?token={get_media_token(fname)})"
                final_response_contents.append(
                    ResponseOutputText(type="output_text", text=image_url)
                )

                yield make_event(
                    "response.output_item.added",
                    {
                        **base_event,
                        "type": "response.output_item.added",
                        "output_index": current_index,
                        "item": img_item.model_dump(mode="json"),
                    },
                )

                yield make_event(
                    "response.output_item.done",
                    {
                        **base_event,
                        "type": "response.output_item.done",
                        "output_index": current_index,
                        "item": img_item.model_dump(mode="json"),
                    },
                )
                current_index += 1
                image_items.append(img_item)
                storage_output += f"\n\n{image_url}"
            except Exception:
                logger.warning("Image processing failed in stream")

        seen_media_hashes = set()
        for media_item in media_items:
            try:
                m_dict = await _media_to_local_file(media_item, media_store)

                m_urls = {}
                for mtype, (random_name, fhash) in m_dict.items():
                    if fhash in seen_media_hashes:
                        (media_store / random_name).unlink(missing_ok=True)
                        continue
                    seen_media_hashes.add(fhash)
                    m_urls[mtype] = (
                        f"{base_url}media/{random_name}?token={get_media_token(random_name)}"
                    )

                media_url = m_urls.get("video") or m_urls.get("audio")
                thumb_url = m_urls.get("video_thumbnail") or m_urls.get("audio_thumbnail")

                title = getattr(media_item, "title", "Media")
                media_md = ""
                if thumb_url and media_url:
                    media_md = f"[![{title}]({thumb_url})]({media_url})"
                elif media_url:
                    media_md = f"[{title}]({media_url})"
                elif thumb_url:
                    media_md = f"![{title}]({thumb_url})"

                if media_md:
                    final_response_contents.append(
                        ResponseOutputText(type="output_text", text=media_md)
                    )
                    storage_output += f"\n\n{media_md}"
            except Exception:
                logger.warning("Media processing failed in stream")

        for call in detected_tool_calls:
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
                    "output_index": current_index,
                    "item": tc_item.model_dump(mode="json"),
                },
            )
            yield make_event(
                "response.output_item.done",
                {
                    **base_event,
                    "type": "response.output_item.done",
                    "output_index": current_index,
                    "item": tc_item.model_dump(mode="json"),
                },
            )
            current_index += 1

        if assistant_text:
            final_response_contents.insert(
                0, ResponseOutputText(type="output_text", text=assistant_text)
            )

        p_tok, c_tok, t_tok, r_tok = _calculate_usage(
            messages, assistant_text, detected_tool_calls, full_thoughts
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
    media_store: Path = Depends(get_media_store_dir),
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
            logger.exception("Error in preparing conversation")
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
        logger.exception("Gemini API error")
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

    try:
        thoughts = resp_or_stream.thoughts
        raw_clean = GeminiClientWrapper.extract_output(resp_or_stream, include_thoughts=False)
    except Exception as exc:
        logger.exception("Gemini output parsing failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Malformed response."
        ) from exc

    thoughts, visible_output, storage_output, tool_calls = _process_llm_output(
        thoughts, raw_clean, structured_requirement
    )

    # Process images for OpenAI non-streaming flow
    images = resp_or_stream.images or []
    image_markdown = ""
    seen_hashes = set()
    for image in images:
        try:
            _, _, _, fname, fhash = await _image_to_base64(image, media_store)
            if fhash in seen_hashes:
                (media_store / fname).unlink(missing_ok=True)
                continue
            seen_hashes.add(fhash)

            img_url = f"![{fname}]({base_url}media/{fname}?token={get_media_token(fname)})"
            image_markdown += f"\n\n{img_url}"
        except Exception as exc:
            logger.warning(f"Failed to process image in OpenAI response: {exc}")

    if image_markdown:
        visible_output += image_markdown
        storage_output += image_markdown

    media_items: list[GeneratedVideo | GeneratedMedia] = (resp_or_stream.videos or []) + (
        resp_or_stream.media or []
    )
    media_markdown = ""
    seen_media_hashes = set()
    for m_item in media_items:
        try:
            m_dict = await _media_to_local_file(m_item, media_store)

            m_urls = {}
            for mtype, (random_name, fhash) in m_dict.items():
                if fhash in seen_media_hashes:
                    (media_store / random_name).unlink(missing_ok=True)
                    continue
                seen_media_hashes.add(fhash)
                m_urls[mtype] = (
                    f"{base_url}media/{random_name}?token={get_media_token(random_name)}"
                )

            media_url = m_urls.get("video") or m_urls.get("audio")
            thumb_url = m_urls.get("video_thumbnail") or m_urls.get("audio_thumbnail")

            title = getattr(m_item, "title", "Media")
            if thumb_url and media_url:
                media_markdown += f"\n\n[![{title}]({thumb_url})]({media_url})"
            elif media_url:
                media_markdown += f"\n\n[{title}]({media_url})"
            elif thumb_url:
                media_markdown += f"\n\n![{title}]({thumb_url})"
        except Exception as exc:
            logger.warning(f"Failed to process media in OpenAI response: {exc}")

    if media_markdown:
        visible_output += media_markdown
        storage_output += media_markdown

    if tool_calls:
        logger.debug(
            f"Detected tool calls: {reprlib.repr([tc.model_dump(mode='json') for tc in tool_calls])}"
        )

    p_tok, c_tok, t_tok, r_tok = _calculate_usage(
        app_messages, visible_output, tool_calls, thoughts
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
    media_store: Path = Depends(get_media_store_dir),
):
    base_url = str(raw_request.base_url)
    base_messages = _convert_responses_to_app_messages(request.input)
    struct_req = _build_structured_requirement(request.response_format)
    extra_instr = [struct_req.instruction] if struct_req else []

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
            logger.exception("Error in preparing conversation")
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
        logger.exception("Gemini API error")
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
            media_store,
            base_url,
            struct_req,
        )

    assert isinstance(resp_or_stream, ModelOutput)

    try:
        thoughts = resp_or_stream.thoughts
        raw_clean = GeminiClientWrapper.extract_output(resp_or_stream, include_thoughts=False)
    except Exception as exc:
        logger.exception("Gemini parsing failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Malformed response."
        ) from exc

    thoughts, assistant_text, storage_output, tool_calls = _process_llm_output(
        thoughts, raw_clean, struct_req
    )
    images = resp_or_stream.images or []
    if (
        request.tool_choice is not None
        and hasattr(request.tool_choice, "type")
        and request.tool_choice.type == "image_generation"
    ) and not images:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No images returned.")

    contents, img_calls = [], []
    seen_hashes = set()
    for img in images:
        try:
            b64, w, h, fname, fhash = await _image_to_base64(img, media_store)
            if fhash in seen_hashes:
                (media_store / fname).unlink(missing_ok=True)
                continue
            seen_hashes.add(fhash)

            parts = fname.rsplit(".", 1)
            img_id = parts[0]
            img_format = (
                parts[1]
                if len(parts) > 1
                else ("png" if isinstance(img, GeneratedImage) else "jpeg")
            )

            img_calls.append(
                ImageGenerationCall(
                    id=img_id,
                    result=b64,
                    output_format=img_format,
                    size=f"{w}x{h}" if w and h else None,
                )
            )
        except Exception as e:
            logger.warning(f"Image error: {e}")

    if assistant_text:
        contents.append(ResponseOutputText(type="output_text", text=assistant_text))
    if not contents:
        contents.append(ResponseOutputText(type="output_text", text=""))

    image_markdown = ""
    for img_call in img_calls:
        fname = f"{img_call.id}.{img_call.output_format}"
        img_url = f"![{fname}]({base_url}media/{fname}?token={get_media_token(fname)})"
        image_markdown += f"\n\n{img_url}"

    if image_markdown:
        storage_output += image_markdown
        contents.append(ResponseOutputText(type="output_text", text=image_markdown.strip()))

    media_items: list[GeneratedVideo | GeneratedMedia] = (resp_or_stream.videos or []) + (
        resp_or_stream.media or []
    )
    seen_media_hashes = set()
    media_markdown = ""
    for m_item in media_items:
        try:
            m_dict = await _media_to_local_file(m_item, media_store)

            m_urls = {}
            for mtype, (random_name, fhash) in m_dict.items():
                if fhash in seen_media_hashes:
                    (media_store / random_name).unlink(missing_ok=True)
                    continue
                seen_media_hashes.add(fhash)
                m_urls[mtype] = (
                    f"{base_url}media/{random_name}?token={get_media_token(random_name)}"
                )

            media_url = m_urls.get("video") or m_urls.get("audio")
            thumb_url = m_urls.get("video_thumbnail") or m_urls.get("audio_thumbnail")

            title = getattr(m_item, "title", "Media")
            m_md = ""
            if thumb_url and media_url:
                m_md = f"[![{title}]({thumb_url})]({media_url})"
            elif media_url:
                m_md = f"[{title}]({media_url})"
            elif thumb_url:
                m_md = f"![{title}]({thumb_url})"

            if m_md:
                media_markdown += f"\n\n{m_md}"
        except Exception as exc:
            logger.warning(f"Failed to process media in OpenAI response: {exc}")

    if media_markdown:
        storage_output += media_markdown
        contents.append(ResponseOutputText(type="output_text", text=media_markdown.strip()))

    p_tok, c_tok, t_tok, r_tok = _calculate_usage(messages, assistant_text, tool_calls, thoughts)
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
