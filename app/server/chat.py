import base64
import hashlib
import io
import reprlib
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from gemini_webapi import ModelOutput
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from gemini_webapi.types.image import GeneratedImage, Image
from loguru import logger

from app.models import (
    ChatCompletionRequest,
    ContentItem,
    ConversationInStore,
    Message,
    ModelData,
    ModelListResponse,
    ResponseCreateRequest,
    ResponseCreateResponse,
    ResponseImageGenerationCall,
    ResponseImageTool,
    ResponseInputContent,
    ResponseInputItem,
    ResponseOutputContent,
    ResponseOutputMessage,
    ResponseReasoning,
    ResponseReasoningContentPart,
    ResponseToolCall,
    ResponseToolChoice,
    ResponseUsage,
    Tool,
    ToolChoiceFunction,
)
from app.server.middleware import (
    get_image_store_dir,
    get_image_token,
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


def _calculate_usage(
    messages: list[Message],
    assistant_text: str | None,
    tool_calls: list[Any] | None,
    thoughts: str | None = None,
) -> tuple[int, int, int, int]:
    """Calculate prompt, completion, total and reasoning tokens consistently."""
    prompt_tokens = sum(estimate_tokens(text_from_message(msg)) for msg in messages)
    tool_args_text = ""
    if tool_calls:
        for call in tool_calls:
            if hasattr(call, "function"):
                tool_args_text += call.function.arguments or ""
            elif isinstance(call, dict):
                tool_args_text += call.get("function", {}).get("arguments", "")

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
    detected_tool_calls: list[Any] | None,
    image_call_items: list[ResponseImageGenerationCall],
    response_contents: list[ResponseOutputContent],
    usage: ResponseUsage,
    request: ResponseCreateRequest,
    normalized_input: Any,
    full_thoughts: str | None = None,
) -> ResponseCreateResponse:
    """Unified factory for building ResponseCreateResponse objects."""
    message_id = f"msg_{uuid.uuid4().hex}"
    reason_id = f"reason_{uuid.uuid4().hex}"

    output_items: list[Any] = []
    if full_thoughts:
        output_items.append(
            ResponseReasoning(
                id=reason_id,
                status="completed",
                content=[ResponseReasoningContentPart(text=full_thoughts)],
            )
        )

    output_items.append(
        ResponseOutputMessage(
            id=message_id,
            type="message",
            role="assistant",
            content=response_contents,
        )
    )

    if detected_tool_calls:
        output_items.extend(
            [
                ResponseToolCall(
                    id=call.id if hasattr(call, "id") else call["id"],
                    status="completed",
                    function=call.function if hasattr(call, "function") else call["function"],
                )
                for call in detected_tool_calls
            ]
        )

    output_items.extend(image_call_items)

    return ResponseCreateResponse(
        id=response_id,
        created_at=created_time,
        model=model_name,
        output=output_items,
        status="completed",
        usage=usage,
        input=normalized_input or None,
        metadata=request.metadata or None,
        tools=request.tools,
        tool_choice=request.tool_choice,
    )


def _create_chat_completion_standard_payload(
    completion_id: str,
    created_time: int,
    model_name: str,
    visible_output: str | None,
    tool_calls_payload: list[dict] | None,
    finish_reason: str,
    usage: dict,
    reasoning_content: str | None = None,
) -> dict:
    """Unified factory for building Chat Completion response dictionaries."""
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": visible_output or None,
                    "tool_calls": tool_calls_payload or None,
                    "reasoning_content": reasoning_content or None,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def _process_llm_output(
    thoughts: str | None,
    raw_text: str,
    structured_requirement: StructuredOutputRequirement | None,
) -> tuple[str | None, str, str, list[Any]]:
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


def _persist_conversation(
    db: LMDBConversationStore,
    model_name: str,
    client_id: str,
    metadata: list[str | None],
    messages: list[Message],
    storage_output: str | None,
    tool_calls: list[Any] | None,
    thoughts: str | None = None,
) -> str | None:
    """Unified logic to save conversation history to LMDB."""
    try:
        current_assistant_message = Message(
            role="assistant",
            content=storage_output or None,
            tool_calls=tool_calls or None,
            reasoning_content=thoughts or None,
        )
        full_history = [*messages, current_assistant_message]
        cleaned_history = db.sanitize_messages(full_history)

        conv = ConversationInStore(
            model=model_name,
            client_id=client_id,
            metadata=metadata,
            messages=cleaned_history,
        )
        key = db.store(conv)
        logger.debug(f"Conversation saved to LMDB with key: {key[:12]}")
        return key
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
    tools: list[Tool],
    tool_choice: str | ToolChoiceFunction | None,
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
    elif isinstance(tool_choice, ToolChoiceFunction):
        target = tool_choice.function.name
        lines.append(
            f"You are required to call the tool named `{target}`. Do not call any other tool."
        )

    lines.append(TOOL_WRAP_HINT)

    return "\n".join(lines)


def _build_image_generation_instruction(
    tools: list[ResponseImageTool] | None,
    tool_choice: ResponseToolChoice | None,
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


def _append_tool_hint_to_last_user_message(messages: list[Message]) -> None:
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
                text_value = part.text or ""
                if TOOL_HINT_STRIPPED in text_value:
                    return
                part.text = f"{text_value}\n{TOOL_WRAP_HINT}"
                return

            messages_text = TOOL_WRAP_HINT.strip()
            msg.content.append(ContentItem(type="text", text=messages_text))
            return


def _prepare_messages_for_model(
    source_messages: list[Message],
    tools: list[Tool] | None,
    tool_choice: str | ToolChoiceFunction | None,
    extra_instructions: list[str] | None = None,
    inject_system_defaults: bool = True,
) -> list[Message]:
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
        prepared.insert(0, Message(role="system", content=combined_instructions))

    if tools and tool_choice != "none" and not tool_prompt_injected:
        _append_tool_hint_to_last_user_message(prepared)

    return prepared


def _response_items_to_messages(
    items: str | list[ResponseInputItem],
) -> tuple[list[Message], str | list[ResponseInputItem]]:
    """Convert Responses API input items into internal Message objects and normalized input."""
    messages: list[Message] = []

    if isinstance(items, str):
        messages.append(Message(role="user", content=items))
        logger.debug("Normalized Responses input: single string message.")
        return messages, items

    normalized_input: list[ResponseInputItem] = []
    for item in items:
        role = item.role
        content = item.content
        normalized_contents: list[ResponseInputContent] = []
        if isinstance(content, str):
            normalized_contents.append(ResponseInputContent(type="input_text", text=content))
            messages.append(Message(role=role, content=content))
        else:
            converted: list[ContentItem] = []
            reasoning_parts: list[str] = []
            for part in content:
                if part.type in ("input_text", "output_text"):
                    text_value = part.text or ""
                    normalized_contents.append(
                        ResponseInputContent(type=part.type, text=text_value)
                    )
                    if text_value:
                        converted.append(ContentItem(type="text", text=text_value))
                elif part.type == "reasoning_text":
                    text_value = part.text or ""
                    normalized_contents.append(
                        ResponseInputContent(type="reasoning_text", text=text_value)
                    )
                    if text_value:
                        reasoning_parts.append(text_value)
                elif part.type == "input_image":
                    image_url = part.image_url
                    if image_url:
                        normalized_contents.append(
                            ResponseInputContent(
                                type="input_image",
                                image_url=image_url,
                                detail=part.detail if part.detail else "auto",
                            )
                        )
                        converted.append(
                            ContentItem(
                                type="image_url",
                                image_url={
                                    "url": image_url,
                                    "detail": part.detail if part.detail else "auto",
                                },
                            )
                        )
                elif part.type == "input_file":
                    if part.file_url or part.file_data:
                        normalized_contents.append(part)
                        file_info = {}
                        if part.file_data:
                            file_info["file_data"] = part.file_data
                            file_info["filename"] = part.filename
                        if part.file_url:
                            file_info["url"] = part.file_url
                        converted.append(ContentItem(type="file", file=file_info))
            messages.append(Message(role=role, content=converted or None))

        normalized_input.append(
            ResponseInputItem(type="message", role=item.role, content=normalized_contents or [])
        )

    logger.debug(f"Normalized Responses input: {len(normalized_input)} message items.")
    return messages, normalized_input


def _instructions_to_messages(
    instructions: str | list[ResponseInputItem] | None,
) -> list[Message]:
    """Normalize instructions payload into Message objects."""
    if not instructions:
        return []

    if isinstance(instructions, str):
        return [Message(role="system", content=instructions)]

    instruction_messages: list[Message] = []
    for item in instructions:
        if item.type and item.type != "message":
            continue

        role = item.role
        content = item.content
        if isinstance(content, str):
            instruction_messages.append(Message(role=role, content=content))
        else:
            converted: list[ContentItem] = []
            reasoning_parts: list[str] = []
            for part in content:
                if part.type in ("input_text", "output_text"):
                    text_value = part.text or ""
                    if text_value:
                        converted.append(ContentItem(type="text", text=text_value))
                elif part.type == "reasoning_text":
                    text_value = part.text or ""
                    if text_value:
                        reasoning_parts.append(text_value)
                elif part.type == "input_image":
                    image_url = part.image_url
                    if image_url:
                        converted.append(
                            ContentItem(
                                type="image_url",
                                image_url={
                                    "url": image_url,
                                    "detail": part.detail if part.detail else "auto",
                                },
                            )
                        )
                elif part.type == "input_file":
                    file_info = {}
                    if part.file_data:
                        file_info["file_data"] = part.file_data
                        file_info["filename"] = part.filename
                    if part.file_url:
                        file_info["url"] = part.file_url
                    if file_info:
                        converted.append(ContentItem(type="file", file=file_info))
            instruction_messages.append(
                Message(
                    role=role,
                    content=converted or None,
                    reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
                )
            )

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


def _get_available_models() -> list[ModelData]:
    """Return a list of available models based on configuration strategy."""
    now = int(datetime.now(tz=UTC).timestamp())
    strategy = g_config.gemini.model_strategy
    models_data = []

    custom_models = [m for m in g_config.gemini.models if m.model_name]
    for m in custom_models:
        models_data.append(
            ModelData(
                id=m.model_name,
                created=now,
                owned_by="custom",
            )
        )

    if strategy == "append":
        custom_ids = {m.model_name for m in custom_models}
        for model in Model:
            m_name = model.model_name
            if not m_name or m_name == "unspecified":
                continue
            if m_name in custom_ids:
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
    messages: list[Message],
) -> tuple[ChatSession | None, GeminiClientWrapper | None, list[Message]]:
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
    files: list[Path | str | io.BytesIO] | None = None,
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
        final_files = list(files) if files else []
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

            if matched_group.endswith("_START"):
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
    generator: AsyncGenerator[ModelOutput],
    completion_id: str,
    created_time: int,
    model_name: str,
    messages: list[Message],
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
        try:
            async for chunk in generator:
                all_outputs.append(chunk)
                if not has_started:
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"
                    has_started = True

                if t_delta := chunk.thoughts_delta:
                    full_thoughts += t_delta
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": t_delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

                if text_delta := chunk.text_delta:
                    full_text += text_delta
                    if visible_delta := suppressor.process(text_delta):
                        data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": visible_delta},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"
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
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"content": remaining_text}, "finish_reason": None}
                ],
            }
            yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        _thoughts, assistant_text, storage_output, tool_calls = _process_llm_output(
            full_thoughts, full_text, structured_requirement
        )

        images = []
        seen_urls = set()
        for out in all_outputs:
            if out.images:
                for img in out.images:
                    # Use the image URL as a stable identifier across chunks
                    if img.url not in seen_urls:
                        images.append(img)
                        seen_urls.add(img.url)

        image_markdown = ""
        seen_hashes = set()
        for image in images:
            try:
                image_store = get_image_store_dir()
                _, _, _, fname, fhash = await _image_to_base64(image, image_store)
                if fhash in seen_hashes:
                    # Duplicate content, delete the file and skip
                    (image_store / fname).unlink(missing_ok=True)
                    continue
                seen_hashes.add(fhash)

                img_url = f"![{fname}]({base_url}images/{fname}?token={get_image_token(fname)})"
                image_markdown += f"\n\n{img_url}"
            except Exception as exc:
                logger.warning(f"Failed to process image in OpenAI stream: {exc}")

        if image_markdown:
            assistant_text += image_markdown
            storage_output += image_markdown
            # Send the image Markdown as a final text chunk before usage
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"content": image_markdown}, "finish_reason": None}
                ],
            }
            yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        tool_calls_payload = [call.model_dump(mode="json") for call in tool_calls]
        if tool_calls_payload:
            tool_calls_delta = [
                {**call, "index": idx} for idx, call in enumerate(tool_calls_payload)
            ]
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"tool_calls": tool_calls_delta}, "finish_reason": None}
                ],
            }
            yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        p_tok, c_tok, t_tok, r_tok = _calculate_usage(
            messages, assistant_text, tool_calls, full_thoughts
        )
        usage = {
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "completion_tokens_details": {"reasoning_tokens": r_tok},
        }
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}
            ],
            "usage": usage,
        }
        _persist_conversation(
            db,
            model.model_name,
            client_wrapper.id,
            session.metadata,
            messages,
            storage_output,
            tool_calls,
            full_thoughts,
        )
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _create_responses_real_streaming_response(
    generator: AsyncGenerator[ModelOutput],
    response_id: str,
    created_time: int,
    model_name: str,
    messages: list[Message],
    db: LMDBConversationStore,
    model: Model,
    client_wrapper: GeminiClientWrapper,
    session: ChatSession,
    request: ResponseCreateRequest,
    image_store: Path,
    base_url: str,
    structured_requirement: StructuredOutputRequirement | None = None,
) -> StreamingResponse:
    """
    Create a real-time streaming response for the Responses API.
    Ensures final accumulated text and thoughts are synchronized.
    """
    base_event = {
        "id": response_id,
        "object": "response",
        "created_at": created_time,
        "model": model_name,
    }

    async def generate_stream():
        yield f"data: {orjson.dumps({**base_event, 'type': 'response.created', 'response': {'id': response_id, 'object': 'response', 'created_at': created_time, 'model': model_name, 'status': 'in_progress', 'metadata': request.metadata, 'input': None, 'tools': request.tools, 'tool_choice': request.tool_choice}}).decode('utf-8')}\n\n"

        full_thoughts, full_text = "", ""
        thought_item_id = f"reason_{uuid.uuid4().hex}"
        message_item_id = f"msg_{uuid.uuid4().hex}"
        thought_item_added = False
        message_item_added = False
        last_chunk_was_thought = False
        current_idx = 0

        all_outputs: list[ModelOutput] = []
        suppressor = StreamingOutputFilter()

        try:
            async for chunk in generator:
                all_outputs.append(chunk)
                if t_delta := chunk.thoughts_delta:
                    if not thought_item_added:
                        yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.added', 'output_index': current_idx, 'item': {'id': thought_item_id, 'type': 'reasoning', 'status': 'in_progress', 'content': []}}).decode('utf-8')}\n\n"
                        yield f"data: {orjson.dumps({**base_event, 'type': 'response.content_part.added', 'output_index': current_idx, 'part_index': 0, 'part': {'type': 'reasoning_text', 'text': ''}}).decode('utf-8')}\n\n"
                        thought_item_added = True

                    full_thoughts += t_delta
                    yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_text.delta', 'output_index': current_idx, 'part_index': 0, 'delta': t_delta}).decode('utf-8')}\n\n"
                    last_chunk_was_thought = True

                if text_delta := chunk.text_delta:
                    if last_chunk_was_thought:
                        yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_text.done', 'output_index': current_idx, 'part_index': 0}).decode('utf-8')}\n\n"
                        yield f"data: {orjson.dumps({**base_event, 'type': 'response.content_part.done', 'output_index': current_idx, 'part_index': 0}).decode('utf-8')}\n\n"
                        yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.done', 'output_index': current_idx, 'item': {'id': thought_item_id, 'type': 'reasoning', 'status': 'completed', 'content': [{'type': 'reasoning_text', 'text': full_thoughts}]}}).decode('utf-8')}\n\n"
                        current_idx += 1
                        last_chunk_was_thought = False

                    if not message_item_added:
                        yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.added', 'output_index': current_idx, 'item': {'id': message_item_id, 'type': 'message', 'role': 'assistant', 'content': []}}).decode('utf-8')}\n\n"
                        yield f"data: {orjson.dumps({**base_event, 'type': 'response.content_part.added', 'output_index': current_idx, 'part_index': 0, 'part': {'type': 'output_text', 'text': ''}}).decode('utf-8')}\n\n"
                        message_item_added = True

                    full_text += text_delta
                    if visible_delta := suppressor.process(text_delta):
                        yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_text.delta', 'output_index': current_idx, 'part_index': 0, 'delta': visible_delta}).decode('utf-8')}\n\n"
        except Exception as e:
            logger.exception(f"Error during Responses API streaming: {e}")
            yield f"data: {orjson.dumps({**base_event, 'type': 'error', 'error': {'message': 'Streaming error.'}}).decode('utf-8')}\n\n"
            return

        if all_outputs:
            final_chunk = all_outputs[-1]
            if final_chunk.text:
                full_text = final_chunk.text
            if final_chunk.thoughts:
                full_thoughts = final_chunk.thoughts

        if last_chunk_was_thought:
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.content_part.done', 'output_index': current_idx, 'part_index': 0}).decode('utf-8')}\n\n"
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.done', 'output_index': current_idx, 'item': {'id': thought_item_id, 'type': 'reasoning', 'status': 'completed', 'content': [{'type': 'reasoning_text', 'text': full_thoughts}]}}).decode('utf-8')}\n\n"
            current_idx += 1

        remaining_from_suppressor = suppressor.flush()
        if remaining_from_suppressor:
            if not message_item_added:
                yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.added', 'output_index': current_idx, 'item': {'id': message_item_id, 'type': 'message', 'role': 'assistant', 'content': []}}).decode('utf-8')}\n\n"
                yield f"data: {orjson.dumps({**base_event, 'type': 'response.content_part.added', 'output_index': current_idx, 'part_index': 0, 'part': {'type': 'output_text', 'text': ''}}).decode('utf-8')}\n\n"
                message_item_added = True
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_text.delta', 'output_index': current_idx, 'part_index': 0, 'delta': remaining_from_suppressor}).decode('utf-8')}\n\n"

        # IMPORTANT: Process output now to get the final assistant_text
        _thoughts, assistant_text, storage_output, detected_tool_calls = _process_llm_output(
            full_thoughts, full_text, structured_requirement
        )

        response_contents: list[ResponseOutputContent] = []
        if message_item_added:
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_text.done', 'output_index': current_idx, 'part_index': 0}).decode('utf-8')}\n\n"
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.content_part.done', 'output_index': current_idx, 'part_index': 0}).decode('utf-8')}\n\n"

            msg_content = [{"type": "output_text", "text": assistant_text}]
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.done', 'output_index': current_idx, 'item': {'id': message_item_id, 'type': 'message', 'role': 'assistant', 'content': msg_content}}).decode('utf-8')}\n\n"
            response_contents.append(ResponseOutputContent(type="output_text", text=assistant_text))
            current_idx += 1

        images = []
        seen_urls = set()
        for out in all_outputs:
            if out.images:
                for img in out.images:
                    if img.url not in seen_urls:
                        images.append(img)
                        seen_urls.add(img.url)

        image_call_items: list[ResponseImageGenerationCall] = []
        seen_hashes = set()
        for image in images:
            try:
                b64, w, h, fname, fhash = await _image_to_base64(image, image_store)
                if fhash in seen_hashes:
                    (image_store / fname).unlink(missing_ok=True)
                    continue
                seen_hashes.add(fhash)

                if "." in fname:
                    img_id, img_format = fname.rsplit(".", 1)
                else:
                    img_id = fname
                    img_format = "png" if isinstance(image, GeneratedImage) else "jpeg"

                img_item = ResponseImageGenerationCall(
                    id=img_id,
                    result=b64,
                    output_format=img_format,
                    size=f"{w}x{h}" if w and h else None,
                )
                image_call_items.append(img_item)

                yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.added', 'output_index': current_idx, 'item': img_item.model_dump(mode='json')}).decode('utf-8')}\n\n"
                yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.done', 'output_index': current_idx, 'item': img_item.model_dump(mode='json')}).decode('utf-8')}\n\n"
                current_idx += 1
            except Exception as exc:
                logger.warning(f"Failed to process image in stream: {exc}")

        image_markdown = ""
        for img_call in image_call_items:
            fname = f"{img_call.id}.{img_call.output_format}"
            img_url = f"![{fname}]({base_url}images/{fname}?token={get_image_token(fname)})"
            image_markdown += f"\n\n{img_url}"

        if image_markdown:
            storage_output += image_markdown

        for call in detected_tool_calls:
            tc_item = ResponseToolCall(id=call.id, status="completed", function=call.function)
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.added', 'output_index': current_idx, 'item': tc_item.model_dump(mode='json')}).decode('utf-8')}\n\n"
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.done', 'output_index': current_idx, 'item': tc_item.model_dump(mode='json')}).decode('utf-8')}\n\n"
            current_idx += 1

        p_tok, c_tok, t_tok, r_tok = _calculate_usage(
            messages, assistant_text, detected_tool_calls, full_thoughts
        )
        usage = ResponseUsage(
            input_tokens=p_tok,
            output_tokens=c_tok,
            total_tokens=t_tok,
            output_tokens_details={"reasoning_tokens": r_tok},
        )

        # Ensure we have at least one content item if none was created
        if not response_contents:
            response_contents.append(
                ResponseOutputContent(type="output_text", text=assistant_text or "")
            )

        payload = _create_responses_standard_payload(
            response_id,
            created_time,
            model_name,
            detected_tool_calls,
            image_call_items,
            response_contents,
            usage,
            request,
            None,
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
            full_thoughts,
        )
        yield f"data: {orjson.dumps({**base_event, 'type': 'response.completed', 'response': payload.model_dump(mode='json')}).decode('utf-8')}\n\n"
        yield f"data: {orjson.dumps({**base_event, 'type': 'response.done'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# --- Main Router Endpoints ---


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    models = _get_available_models()
    return ModelListResponse(data=models)


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
    image_store: Path = Depends(get_image_store_dir),
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

    # This ensures that server-injected system instructions are part of the history
    msgs = _prepare_messages_for_model(
        request.messages,
        request.tools,
        request.tool_choice,
        extra_instr,
    )

    session, client, remain = await _find_reusable_session(db, pool, model, msgs)

    if session:
        if not remain:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No new messages.")

        # For reused sessions, we only need to process the remaining messages.
        # We don't re-inject system defaults to avoid duplicating instructions already in history.
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
            # Use the already prepared 'msgs' for a fresh session
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
            session, m_input, files=files, stream=request.stream
        )
    except Exception as e:
        logger.exception("Gemini API error")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    if request.stream:
        return _create_real_streaming_response(
            resp_or_stream,
            completion_id,
            created_time,
            request.model,
            msgs,  # Use prepared 'msgs'
            db,
            model,
            client,
            session,
            base_url,
            structured_requirement,
        )

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
            _, _, _, fname, fhash = await _image_to_base64(image, image_store)
            if fhash in seen_hashes:
                (image_store / fname).unlink(missing_ok=True)
                continue
            seen_hashes.add(fhash)

            img_url = f"![{fname}]({base_url}images/{fname}?token={get_image_token(fname)})"
            image_markdown += f"\n\n{img_url}"
        except Exception as exc:
            logger.warning(f"Failed to process image in OpenAI response: {exc}")

    if image_markdown:
        visible_output += image_markdown
        storage_output += image_markdown

    tool_calls_payload = [call.model_dump(mode="json") for call in tool_calls]
    if tool_calls_payload:
        logger.debug(f"Detected tool calls: {reprlib.repr(tool_calls_payload)}")

    p_tok, c_tok, t_tok, r_tok = _calculate_usage(
        request.messages, visible_output, tool_calls, thoughts
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
        tool_calls_payload,
        "tool_calls" if tool_calls else "stop",
        usage,
        thoughts,
    )
    _persist_conversation(
        db,
        model.model_name,
        client.id,
        session.metadata,
        msgs,  # Use prepared messages 'msgs'
        storage_output,
        tool_calls,
        thoughts,
    )
    return payload


@router.post("/v1/responses")
async def create_response(
    request: ResponseCreateRequest,
    raw_request: Request,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
    image_store: Path = Depends(get_image_store_dir),
):
    base_url = str(raw_request.base_url)
    base_messages, norm_input = _response_items_to_messages(request.input)
    struct_req = _build_structured_requirement(request.response_format)
    extra_instr = [struct_req.instruction] if struct_req else []

    standard_tools, image_tools = [], []
    if request.tools:
        for t in request.tools:
            if isinstance(t, Tool):
                standard_tools.append(t)
            elif isinstance(t, ResponseImageTool):
                image_tools.append(t)
            elif isinstance(t, dict):
                if t.get("type") == "function":
                    standard_tools.append(Tool.model_validate(t))
                elif t.get("type") == "image_generation":
                    image_tools.append(ResponseImageTool.model_validate(t))

    img_instr = _build_image_generation_instruction(
        image_tools,
        request.tool_choice if isinstance(request.tool_choice, ResponseToolChoice) else None,
    )
    if img_instr:
        extra_instr.append(img_instr)
    preface = _instructions_to_messages(request.instructions)
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
        model = _get_model_by_name(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    session, client, remain = await _find_reusable_session(db, pool, model, messages)
    if session:
        msgs = _prepare_messages_for_model(
            remain,
            request.tools,
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
            session, m_input, files=files, stream=request.stream
        )
    except Exception as e:
        logger.exception("Gemini API error")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    if request.stream:
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
            image_store,
            base_url,
            struct_req,
        )

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
        request.tool_choice is not None and request.tool_choice.type == "image_generation"
    ) and not images:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No images returned.")

    contents, img_calls = [], []
    seen_hashes = set()
    for img in images:
        try:
            b64, w, h, fname, fhash = await _image_to_base64(img, image_store)
            if fhash in seen_hashes:
                (image_store / fname).unlink(missing_ok=True)
                continue
            seen_hashes.add(fhash)

            if "." in fname:
                img_id, img_format = fname.rsplit(".", 1)
            else:
                img_id = fname
                img_format = "png" if isinstance(img, GeneratedImage) else "jpeg"

            contents.append(
                ResponseOutputContent(
                    type="output_text",
                    text=f"![{fname}]({base_url}images/{fname}?token={get_image_token(fname)})",
                )
            )
            img_calls.append(
                ResponseImageGenerationCall(
                    id=img_id,
                    result=b64,
                    output_format=img_format,
                    size=f"{w}x{h}" if w and h else None,
                )
            )
        except Exception as e:
            logger.warning(f"Image error: {e}")

    if assistant_text:
        contents.append(ResponseOutputContent(type="output_text", text=assistant_text))
    if not contents:
        contents.append(ResponseOutputContent(type="output_text", text=""))

    # Aggregate images for storage
    image_markdown = ""
    for img_call in img_calls:
        fname = f"{img_call.id}.{img_call.output_format}"
        img_url = f"![{fname}]({base_url}images/{fname}?token={get_image_token(fname)})"
        image_markdown += f"\n\n{img_url}"

    if image_markdown:
        storage_output += image_markdown

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
        norm_input,
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
        thoughts,
    )
    return payload
