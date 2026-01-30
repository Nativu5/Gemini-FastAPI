import base64
import io
import reprlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import orjson
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from gemini_webapi import ModelOutput
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from gemini_webapi.types.image import GeneratedImage, Image
from loguru import logger

from ..models import (
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
    ResponseToolCall,
    ResponseToolChoice,
    ResponseUsage,
    Tool,
    ToolChoiceFunction,
)
from ..services import GeminiClientPool, GeminiClientWrapper, LMDBConversationStore
from ..utils import g_config
from ..utils.helper import (
    CODE_BLOCK_HINT,
    CODE_HINT_STRIPPED,
    CONTROL_TOKEN_RE,
    XML_HINT_STRIPPED,
    XML_WRAP_HINT,
    estimate_tokens,
    extract_image_dimensions,
    extract_tool_calls,
    strip_code_fence,
    text_from_message,
)
from .middleware import get_image_store_dir, get_image_token, get_temp_dir, verify_api_key

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


async def _image_to_base64(image: Image, temp_dir: Path) -> tuple[str, int | None, int | None, str]:
    """Persist an image provided by gemini_webapi and return base64 plus dimensions and filename."""
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
    random_name = f"img_{uuid.uuid4().hex}{original_path.suffix}"
    new_path = temp_dir / random_name
    original_path.rename(new_path)

    data = new_path.read_bytes()
    width, height = extract_image_dimensions(data)
    filename = random_name
    return base64.b64encode(data).decode("ascii"), width, height, filename


def _calculate_usage(
    messages: list[Message],
    assistant_text: str | None,
    tool_calls: list[Any] | None,
) -> tuple[int, int, int]:
    """Calculate prompt, completion and total tokens consistently."""
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
    return prompt_tokens, completion_tokens, prompt_tokens + completion_tokens


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
) -> ResponseCreateResponse:
    """Unified factory for building ResponseCreateResponse objects."""
    message_id = f"msg_{uuid.uuid4().hex}"
    tool_call_items: list[ResponseToolCall] = []
    if detected_tool_calls:
        tool_call_items = [
            ResponseToolCall(
                id=call.id if hasattr(call, "id") else call["id"],
                status="completed",
                function=call.function if hasattr(call, "function") else call["function"],
            )
            for call in detected_tool_calls
        ]

    return ResponseCreateResponse(
        id=response_id,
        created_at=created_time,
        model=model_name,
        output=[
            ResponseOutputMessage(
                id=message_id,
                type="message",
                role="assistant",
                content=response_contents,
            ),
            *tool_call_items,
            *image_call_items,
        ],
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
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def _process_llm_output(
    raw_output_with_think: str,
    raw_output_clean: str,
    structured_requirement: StructuredOutputRequirement | None,
) -> tuple[str, str, list[Any]]:
    """
    Common post-processing logic for Gemini output.
    Returns: (visible_text, storage_output, tool_calls)
    """
    visible_with_think, tool_calls = extract_tool_calls(raw_output_with_think)
    if tool_calls:
        logger.debug(f"Detected {len(tool_calls)} tool call(s) in model output.")

    visible_output = visible_with_think.strip()

    storage_output, _ = extract_tool_calls(raw_output_clean)
    storage_output = storage_output.strip()

    if structured_requirement:
        cleaned_for_json = LMDBConversationStore.remove_think_tags(visible_output)
        json_text = strip_code_fence(cleaned_for_json or "")
        if json_text:
            try:
                structured_payload = orjson.loads(json_text)
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

    return visible_output, storage_output, tool_calls


def _persist_conversation(
    db: LMDBConversationStore,
    model_name: str,
    client_id: str,
    metadata: list[str | None],
    messages: list[Message],
    storage_output: str | None,
    tool_calls: list[Any] | None,
) -> str | None:
    """Unified logic to save conversation history to LMDB."""
    try:
        current_assistant_message = Message(
            role="assistant",
            content=storage_output or None,
            tool_calls=tool_calls or None,
        )
        full_history = [*messages, current_assistant_message]
        cleaned_history = db.sanitize_assistant_messages(full_history)

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
    """Generate a system prompt chunk describing available tools."""
    if not tools:
        return ""

    lines: list[str] = [
        "You can invoke the following developer tools. Call a tool only when it is required and follow the JSON schema exactly when providing arguments."
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

    lines.append(
        "When you decide to call a tool you MUST respond with nothing except a single fenced block exactly like the template below."
    )
    lines.append(
        "The fenced block MUST use ```xml as the opening fence and ``` as the closing fence. Do not add text before or after it."
    )
    lines.append("```xml")
    lines.append('<tool_call name="tool_name">{"argument": "value"}</tool_call>')
    lines.append("```")
    lines.append(
        "Use double quotes for JSON keys and values. If you omit the fenced block or include any extra text, the system will assume you are NOT calling a tool and your request will fail."
    )
    lines.append(
        "If multiple tool calls are required, include multiple <tool_call> entries inside the same fenced block. Without a tool call, reply normally and do NOT emit any ```xml fence."
    )

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
        "Image generation is enabled. When the user requests an image, you must return an actual generated image, not a text description.",
        "For new image requests, generate at least one new image matching the description.",
        "If the user provides an image and asks for edits or variations, return a newly generated image with the requested changes.",
        "Avoid all text replies unless a short caption is explicitly requested. Do not explain, apologize, or describe image creation steps.",
        "Never send placeholder text like 'Here is your image' or any other response without an actual image attachment.",
    ]

    if primary:
        if primary.model:
            instructions.append(
                f"Where styles differ, favor the `{primary.model}` image model when rendering the scene."
            )
        if primary.output_format:
            instructions.append(
                f"Encode the image using the `{primary.output_format}` format whenever possible."
            )

    if has_forced_choice:
        instructions.append(
            "Image generation was explicitly requested. You must return at least one generated image. Any response without an image will be treated as a failure."
        )

    return "\n\n".join(instructions)


def _append_xml_hint_to_last_user_message(messages: list[Message]) -> None:
    """Ensure the last user message carries the XML wrap hint."""
    for msg in reversed(messages):
        if msg.role != "user" or msg.content is None:
            continue

        if isinstance(msg.content, str):
            if XML_HINT_STRIPPED not in msg.content:
                msg.content = f"{msg.content}\n{XML_WRAP_HINT}"
            return

        if isinstance(msg.content, list):
            for part in reversed(msg.content):
                if getattr(part, "type", None) != "text":
                    continue
                text_value = part.text or ""
                if XML_HINT_STRIPPED in text_value:
                    return
                part.text = f"{text_value}\n{XML_WRAP_HINT}"
                return

            messages_text = XML_WRAP_HINT.strip()
            msg.content.append(ContentItem(type="text", text=messages_text))
            return


def _conversation_has_code_hint(messages: list[Message]) -> bool:
    """Return True if any system message already includes the code block hint."""
    for msg in messages:
        if msg.role != "system" or msg.content is None:
            continue

        if isinstance(msg.content, str):
            if CODE_HINT_STRIPPED in msg.content:
                return True
            continue

        if isinstance(msg.content, list):
            for part in msg.content:
                if getattr(part, "type", None) != "text":
                    continue
                if part.text and CODE_HINT_STRIPPED in part.text:
                    return True

    return False


def _prepare_messages_for_model(
    source_messages: list[Message],
    tools: list[Tool] | None,
    tool_choice: str | ToolChoiceFunction | None,
    extra_instructions: list[str] | None = None,
    inject_system_defaults: bool = True,
) -> list[Message]:
    """Return a copy of messages enriched with tool instructions when needed."""
    prepared = [msg.model_copy(deep=True) for msg in source_messages]

    instructions: list[str] = []
    if inject_system_defaults:
        if tools:
            tool_prompt = _build_tool_prompt(tools, tool_choice)
            if tool_prompt:
                instructions.append(tool_prompt)

        if extra_instructions:
            instructions.extend(instr for instr in extra_instructions if instr)
            logger.debug(
                f"Applied {len(extra_instructions)} extra instructions for tool/structured output."
            )

        if not _conversation_has_code_hint(prepared):
            instructions.append(CODE_BLOCK_HINT)
            logger.debug("Injected default code block hint for Gemini conversation.")

    if not instructions:
        if tools and tool_choice != "none":
            _append_xml_hint_to_last_user_message(prepared)
        return prepared

    combined_instructions = "\n\n".join(instructions)
    if prepared and prepared[0].role == "system" and isinstance(prepared[0].content, str):
        existing = prepared[0].content or ""
        separator = "\n\n" if existing else ""
        prepared[0].content = f"{existing}{separator}{combined_instructions}"
    else:
        prepared.insert(0, Message(role="system", content=combined_instructions))

    if tools and tool_choice != "none":
        _append_xml_hint_to_last_user_message(prepared)

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
            for part in content:
                if part.type == "input_text":
                    text_value = part.text or ""
                    normalized_contents.append(
                        ResponseInputContent(type="input_text", text=text_value)
                    )
                    if text_value:
                        converted.append(ContentItem(type="text", text=text_value))
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
            for part in content:
                if part.type == "input_text":
                    text_value = part.text or ""
                    if text_value:
                        converted.append(ContentItem(type="text", text=text_value))
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
            instruction_messages.append(Message(role=role, content=converted or None))

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
    now = int(datetime.now(tz=timezone.utc).timestamp())
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
) -> AsyncGenerator[ModelOutput, None] | ModelOutput:
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
    Enhanced streaming filter that suppresses:
    1. XML tool call blocks: ```xml ... ```
    2. ChatML tool blocks: <|im_start|>tool\n...<|im_end|>
    3. ChatML role headers: <|im_start|>role\n (only suppresses the header, keeps content)
    4. Control tokens: <|im_start|>, <|im_end|>
    5. System instructions/hints: XML_WRAP_HINT, CODE_BLOCK_HINT, etc.
    """

    def __init__(self):
        self.buffer = ""
        self.in_xml_tool = False
        self.in_tagged_block = False
        self.in_role_header = False
        self.current_role = ""

        self.XML_START = "```xml"
        self.XML_END = "```"
        self.TAG_START = "<|im_start|>"
        self.TAG_END = "<|im_end|>"
        self.SYSTEM_HINTS = [
            XML_WRAP_HINT,
            XML_HINT_STRIPPED,
            CODE_BLOCK_HINT,
            CODE_HINT_STRIPPED,
        ]

    def process(self, chunk: str) -> str:
        self.buffer += chunk
        to_yield = ""

        while self.buffer:
            if self.in_xml_tool:
                end_idx = self.buffer.find(self.XML_END)
                if end_idx != -1:
                    self.buffer = self.buffer[end_idx + len(self.XML_END) :]
                    self.in_xml_tool = False
                else:
                    break
            elif self.in_role_header:
                nl_idx = self.buffer.find("\n")
                if nl_idx != -1:
                    role_text = self.buffer[:nl_idx].strip().lower()
                    self.current_role = role_text
                    self.buffer = self.buffer[nl_idx + 1 :]
                    self.in_role_header = False
                    self.in_tagged_block = True
                else:
                    break
            elif self.in_tagged_block:
                end_idx = self.buffer.find(self.TAG_END)
                if end_idx != -1:
                    content = self.buffer[:end_idx]
                    if self.current_role != "tool":
                        to_yield += content
                    self.buffer = self.buffer[end_idx + len(self.TAG_END) :]
                    self.in_tagged_block = False
                    self.current_role = ""
                else:
                    if self.current_role == "tool":
                        break
                    else:
                        yield_len = len(self.buffer) - (len(self.TAG_END) - 1)
                        if yield_len > 0:
                            to_yield += self.buffer[:yield_len]
                            self.buffer = self.buffer[yield_len:]
                        break
            else:
                # Outside any special block. Look for starts.
                earliest_idx = -1
                match_type = ""

                xml_idx = self.buffer.find(self.XML_START)
                if xml_idx != -1:
                    earliest_idx = xml_idx
                    match_type = "xml"

                tag_s_idx = self.buffer.find(self.TAG_START)
                if tag_s_idx != -1:
                    if earliest_idx == -1 or tag_s_idx < earliest_idx:
                        earliest_idx = tag_s_idx
                        match_type = "tag_start"

                tag_e_idx = self.buffer.find(self.TAG_END)
                if tag_e_idx != -1:
                    if earliest_idx == -1 or tag_e_idx < earliest_idx:
                        earliest_idx = tag_e_idx
                        match_type = "tag_end"

                if earliest_idx != -1:
                    # Yield text before the match
                    to_yield += self.buffer[:earliest_idx]
                    self.buffer = self.buffer[earliest_idx:]

                    if match_type == "xml":
                        self.in_xml_tool = True
                        self.buffer = self.buffer[len(self.XML_START) :]
                    elif match_type == "tag_start":
                        self.in_role_header = True
                        self.buffer = self.buffer[len(self.TAG_START) :]
                    elif match_type == "tag_end":
                        # Orphaned end tag, just skip it
                        self.buffer = self.buffer[len(self.TAG_END) :]
                    continue
                else:
                    # Check for prefixes
                    prefixes = [self.XML_START, self.TAG_START, self.TAG_END]
                    max_keep = 0
                    for p in prefixes:
                        for i in range(len(p) - 1, 0, -1):
                            if self.buffer.endswith(p[:i]):
                                max_keep = max(max_keep, i)
                                break

                    yield_len = len(self.buffer) - max_keep
                    if yield_len > 0:
                        to_yield += self.buffer[:yield_len]
                        self.buffer = self.buffer[yield_len:]
                    break

        # Final pass: filter out system hints from the text to be yielded
        for hint in self.SYSTEM_HINTS:
            if hint in to_yield:
                to_yield = to_yield.replace(hint, "")

        return to_yield

    def flush(self) -> str:
        # If we are stuck in a tool block or role header at the end,
        # it usually means malformed output.
        if self.in_xml_tool or (self.in_tagged_block and self.current_role == "tool"):
            return ""

        final_text = self.buffer
        self.buffer = ""

        # Filter out any orphaned/partial control tokens or hints
        final_text = CONTROL_TOKEN_RE.sub("", final_text)
        for hint in self.SYSTEM_HINTS:
            final_text = final_text.replace(hint, "")

        return final_text.strip()


# --- Response Builders & Streaming ---


def _create_real_streaming_response(
    generator: AsyncGenerator[ModelOutput, None],
    completion_id: str,
    created_time: int,
    model_name: str,
    messages: list[Message],
    db: LMDBConversationStore,
    model: Model,
    client_wrapper: GeminiClientWrapper,
    session: ChatSession,
    structured_requirement: StructuredOutputRequirement | None = None,
) -> StreamingResponse:
    """
    Create a real-time streaming response.
    Reconciles manual delta accumulation with the model's final authoritative state.
    """

    async def generate_stream():
        full_thoughts, full_text = "", ""
        has_started = False
        last_chunk_was_thought = False
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
                    if not last_chunk_was_thought and not full_thoughts:
                        yield f"data: {orjson.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': '<think>'}, 'finish_reason': None}]}).decode('utf-8')}\n\n"
                    full_thoughts += t_delta
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {"content": t_delta}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"
                    last_chunk_was_thought = True

                if text_delta := chunk.text_delta:
                    if last_chunk_was_thought:
                        yield f"data: {orjson.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': '</think>\n'}, 'finish_reason': None}]}).decode('utf-8')}\n\n"
                        last_chunk_was_thought = False
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

        if last_chunk_was_thought:
            yield f"data: {orjson.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': '</think>\n'}, 'finish_reason': None}]}).decode('utf-8')}\n\n"

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

        raw_output_with_think = f"<think>{full_thoughts}</think>\n" if full_thoughts else ""
        raw_output_with_think += full_text
        assistant_text, storage_output, tool_calls = _process_llm_output(
            raw_output_with_think, full_text, structured_requirement
        )

        images = []
        for out in all_outputs:
            if out.images:
                images.extend(out.images)

        image_markdown = ""
        for image in images:
            try:
                image_store = get_image_store_dir()
                _, _, _, filename = await _image_to_base64(image, image_store)
                img_url = f"![{filename}](images/{filename}?token={get_image_token(filename)})"
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

        p_tok, c_tok, t_tok = _calculate_usage(messages, assistant_text, tool_calls)
        usage = {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": t_tok}
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
            messages,  # This should be the prepared messages
            storage_output,
            tool_calls,
        )
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _create_responses_real_streaming_response(
    generator: AsyncGenerator[ModelOutput, None],
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
        message_id = f"msg_{uuid.uuid4().hex}"
        yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.added', 'output_index': 0, 'item': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': []}}).decode('utf-8')}\n\n"

        full_thoughts, full_text = "", ""
        last_chunk_was_thought = False
        all_outputs: list[ModelOutput] = []
        suppressor = StreamingOutputFilter()

        try:
            async for chunk in generator:
                all_outputs.append(chunk)
                if t_delta := chunk.thoughts_delta:
                    if not last_chunk_was_thought and not full_thoughts:
                        yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_text.delta', 'output_index': 0, 'delta': '<think>'}).decode('utf-8')}\n\n"
                    full_thoughts += t_delta
                    yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_text.delta', 'output_index': 0, 'delta': t_delta}).decode('utf-8')}\n\n"
                    last_chunk_was_thought = True
                if text_delta := chunk.text_delta:
                    if last_chunk_was_thought:
                        yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_text.delta', 'output_index': 0, 'delta': '</think>\n'}).decode('utf-8')}\n\n"
                        last_chunk_was_thought = False
                    full_text += text_delta
                    if visible_delta := suppressor.process(text_delta):
                        yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_text.delta', 'output_index': 0, 'delta': visible_delta}).decode('utf-8')}\n\n"
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
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_text.delta', 'output_index': 0, 'delta': '</think>\n'}).decode('utf-8')}\n\n"
        if remaining_text := suppressor.flush():
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_text.delta', 'output_index': 0, 'delta': remaining_text}).decode('utf-8')}\n\n"
        yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_text.done', 'output_index': 0}).decode('utf-8')}\n\n"

        raw_output_with_think = f"<think>{full_thoughts}</think>\n" if full_thoughts else ""
        raw_output_with_think += full_text
        assistant_text, storage_output, detected_tool_calls = _process_llm_output(
            raw_output_with_think, full_text, structured_requirement
        )

        images = []
        for out in all_outputs:
            if out.images:
                images.extend(out.images)

        response_contents, image_call_items = [], []
        for image in images:
            try:
                image_base64, width, height, filename = await _image_to_base64(image, image_store)
                img_format = "png" if isinstance(image, GeneratedImage) else "jpeg"
                image_url = f"![{filename}](images/{filename}?token={get_image_token(filename)})"
                image_call_items.append(
                    ResponseImageGenerationCall(
                        id=filename.rsplit(".", 1)[0],
                        result=image_base64,
                        output_format=img_format,
                        size=f"{width}x{height}" if width and height else None,
                    )
                )
                response_contents.append(ResponseOutputContent(type="output_text", text=image_url))
            except Exception as exc:
                logger.warning(f"Failed to process image in stream: {exc}")

        if assistant_text:
            response_contents.append(ResponseOutputContent(type="output_text", text=assistant_text))
        if not response_contents:
            response_contents.append(ResponseOutputContent(type="output_text", text=""))

        # Aggregate images for storage
        image_markdown = ""
        for img_call in image_call_items:
            fname = f"{img_call.id}.{img_call.output_format}"
            img_url = f"![{fname}](images/{fname}?token={get_image_token(fname)})"
            image_markdown += f"\n\n{img_url}"

        if image_markdown:
            storage_output += image_markdown

        yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.done', 'output_index': 0, 'item': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': [c.model_dump(mode='json') for c in response_contents]}}).decode('utf-8')}\n\n"

        current_idx = 1
        for call in detected_tool_calls:
            tc_item = ResponseToolCall(id=call.id, status="completed", function=call.function)
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.added', 'output_index': current_idx, 'item': tc_item.model_dump(mode='json')}).decode('utf-8')}\n\n"
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.done', 'output_index': current_idx, 'item': tc_item.model_dump(mode='json')}).decode('utf-8')}\n\n"
            current_idx += 1
        for img_call in image_call_items:
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.added', 'output_index': current_idx, 'item': img_call.model_dump(mode='json')}).decode('utf-8')}\n\n"
            yield f"data: {orjson.dumps({**base_event, 'type': 'response.output_item.done', 'output_index': current_idx, 'item': img_call.model_dump(mode='json')}).decode('utf-8')}\n\n"
            current_idx += 1

        p_tok, c_tok, t_tok = _calculate_usage(messages, assistant_text, detected_tool_calls)
        usage = ResponseUsage(input_tokens=p_tok, output_tokens=c_tok, total_tokens=t_tok)
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
        yield f"data: {orjson.dumps({**base_event, 'type': 'response.completed', 'response': payload.model_dump(mode='json')}).decode('utf-8')}\n\n"
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
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
    image_store: Path = Depends(get_image_store_dir),
):
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
        request.messages, request.tools, request.tool_choice, extra_instr
    )

    session, client, remain = await _find_reusable_session(db, pool, model, msgs)

    if session:
        if not remain:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No new messages.")

        # For reused sessions, we only need to process the remaining messages.
        # We don't re-inject system defaults to avoid duplicating instructions already in history.
        input_msgs = _prepare_messages_for_model(
            remain, request.tools, request.tool_choice, extra_instr, False
        )
        if len(input_msgs) == 1:
            m_input, files = await GeminiClientWrapper.process_message(
                input_msgs[0], tmp_dir, tagged=False
            )
        else:
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
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(datetime.now(tz=timezone.utc).timestamp())

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
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))

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
            structured_requirement,
        )

    try:
        raw_with_t = GeminiClientWrapper.extract_output(resp_or_stream, include_thoughts=True)
        raw_clean = GeminiClientWrapper.extract_output(resp_or_stream, include_thoughts=False)
    except Exception as exc:
        logger.exception("Gemini output parsing failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Malformed response."
        ) from exc

    visible_output, storage_output, tool_calls = _process_llm_output(
        raw_with_t, raw_clean, structured_requirement
    )

    # Process images for OpenAI non-streaming flow
    images = resp_or_stream.images or []
    image_markdown = ""
    for image in images:
        try:
            _, _, _, filename = await _image_to_base64(image, image_store)
            img_url = f"![{filename}](images/{filename}?token={get_image_token(filename)})"
            image_markdown += f"\n\n{img_url}"
        except Exception as exc:
            logger.warning(f"Failed to process image in OpenAI response: {exc}")

    if image_markdown:
        visible_output += image_markdown
        storage_output += image_markdown

    tool_calls_payload = [call.model_dump(mode="json") for call in tool_calls]
    if tool_calls_payload:
        logger.debug(f"Detected tool calls: {reprlib.repr(tool_calls_payload)}")

    p_tok, c_tok, t_tok = _calculate_usage(request.messages, visible_output, tool_calls)
    usage = {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": t_tok}
    payload = _create_chat_completion_standard_payload(
        completion_id,
        created_time,
        request.model,
        visible_output,
        tool_calls_payload,
        "tool_calls" if tool_calls else "stop",
        usage,
    )
    _persist_conversation(
        db,
        model.model_name,
        client.id,
        session.metadata,
        msgs,  # Use prepared messages 'msgs'
        storage_output,
        tool_calls,
    )
    return payload


@router.post("/v1/responses")
async def create_response(
    request: ResponseCreateRequest,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
    image_store: Path = Depends(get_image_store_dir),
):
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
        conv_messages, standard_tools or None, model_tool_choice, extra_instr or None
    )
    pool, db = GeminiClientPool(), LMDBConversationStore()
    try:
        model = _get_model_by_name(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    session, client, remain = await _find_reusable_session(db, pool, model, messages)
    if session:
        msgs = _prepare_messages_for_model(remain, request.tools, request.tool_choice, None, False)
        if not msgs:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No new messages.")
        m_input, files = (
            await GeminiClientWrapper.process_message(msgs[0], tmp_dir, tagged=False)
            if len(msgs) == 1
            else await GeminiClientWrapper.process_conversation(msgs, tmp_dir)
        )
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
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    response_id = f"resp_{uuid.uuid4().hex}"
    created_time = int(datetime.now(tz=timezone.utc).timestamp())

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
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))

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
            struct_req,
        )

    try:
        raw_t = GeminiClientWrapper.extract_output(resp_or_stream, include_thoughts=True)
        raw_c = GeminiClientWrapper.extract_output(resp_or_stream, include_thoughts=False)
    except Exception as exc:
        logger.exception("Gemini parsing failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Malformed response."
        ) from exc

    assistant_text, storage_output, tool_calls = _process_llm_output(raw_t, raw_c, struct_req)
    images = resp_or_stream.images or []
    if (
        request.tool_choice is not None and request.tool_choice.type == "image_generation"
    ) and not images:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No images returned.")

    contents, img_calls = [], []
    for img in images:
        try:
            b64, w, h, fname = await _image_to_base64(img, image_store)
            contents.append(
                ResponseOutputContent(
                    type="output_text",
                    text=f"![{fname}](images/{fname}?token={get_image_token(fname)})",
                )
            )
            img_calls.append(
                ResponseImageGenerationCall(
                    id=fname.rsplit(".", 1)[0],
                    result=b64,
                    output_format="png" if isinstance(img, GeneratedImage) else "jpeg",
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
        img_url = f"![{fname}](images/{fname}?token={get_image_token(fname)})"
        image_markdown += f"\n\n{img_url}"

    if image_markdown:
        storage_output += image_markdown

    p_tok, c_tok, t_tok = _calculate_usage(messages, assistant_text, tool_calls)
    usage = ResponseUsage(input_tokens=p_tok, output_tokens=c_tok, total_tokens=t_tok)
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
    )
    _persist_conversation(
        db, model.model_name, client.id, session.metadata, messages, storage_output, tool_calls
    )
    return payload
