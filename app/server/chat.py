import base64
import json
import re
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from gemini_webapi.types.image import GeneratedImage, Image
from loguru import logger

from .middleware import get_temp_dir, verify_api_key
from ..models import (
    ContentItem,
    ChatCompletionRequest,
    ConversationInStore,
    FunctionCall,
    Message,
    ModelData,
    ModelListResponse,
    ResponseCreateRequest,
    ResponseCreateResponse,
    ResponseImageGenerationCall,
    ResponseInputItem,
    ResponseOutputContent,
    ResponseOutputMessage,
    Tool,
    ToolCall,
    ToolChoiceFunction,
    Usage,
)
from ..services import GeminiClientPool, GeminiClientWrapper, LMDBConversationStore
from ..services.client import XML_WRAP_HINT
from ..utils import g_config
from ..utils.helper import estimate_tokens

# Maximum characters Gemini Web can accept in a single request (configurable)
MAX_CHARS_PER_REQUEST = int(g_config.gemini.max_chars_per_request * 0.9)

CONTINUATION_HINT = "\n(More messages to come, please reply with just 'ok.')"

TOOL_BLOCK_RE = re.compile(r"```xml\s*(.*?)```", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(
    r"<tool_call\s+name=\"([^\"]+)\">(.*?)</tool_call>", re.DOTALL | re.IGNORECASE
)
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
XML_HINT_STRIPPED = XML_WRAP_HINT.strip()

router = APIRouter()


@dataclass
class StructuredOutputRequirement:
    """Represents a structured response request from the client."""

    schema_name: str
    schema: dict[str, Any]
    instruction: str
    raw_format: dict[str, Any]


def _build_structured_requirement(
    response_format: dict[str, Any] | None,
) -> StructuredOutputRequirement | None:
    """Translate OpenAI-style response_format into internal instructions."""
    if not response_format or not isinstance(response_format, dict):
        return None

    if response_format.get("type") != "json_schema":
        logger.warning(f"Unsupported response_format type requested: {response_format}")
        return None

    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        logger.warning(f"Invalid json_schema payload in response_format: {response_format}")
        return None

    schema = json_schema.get("schema")
    if not isinstance(schema, dict):
        logger.warning(f"Missing `schema` object in response_format payload: {response_format}")
        return None

    schema_name = json_schema.get("name") or "response"
    strict = json_schema.get("strict", True)

    pretty_schema = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)
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


def _strip_code_fence(text: str) -> str:
    """Remove surrounding ```json fences if present."""
    match = JSON_FENCE_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


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
            schema_text = json.dumps(function.parameters, ensure_ascii=False, indent=2)
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
    # `auto` or None fall back to default instructions.

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


def _append_xml_hint_to_last_user_message(messages: list[Message]) -> None:
    """Ensure the last user message carries the XML wrap hint."""
    for msg in reversed(messages):
        if msg.role != "user" or msg.content is None:
            continue

        if isinstance(msg.content, str):
            if XML_HINT_STRIPPED not in msg.content:
                msg.content = f"{msg.content}{XML_WRAP_HINT}"
            return

        if isinstance(msg.content, list):
            for part in reversed(msg.content):
                if getattr(part, "type", None) != "text":
                    continue
                text_value = part.text or ""
                if XML_HINT_STRIPPED in text_value:
                    return
                part.text = f"{text_value}{XML_WRAP_HINT}"
                return

            messages_text = XML_WRAP_HINT.strip()
            msg.content.append(ContentItem(type="text", text=messages_text))
            return

    # No user message to annotate; nothing to do.


def _prepare_messages_for_model(
    source_messages: list[Message],
    tools: list[Tool] | None,
    tool_choice: str | ToolChoiceFunction | None,
    extra_instructions: list[str] | None = None,
) -> list[Message]:
    """Return a copy of messages enriched with tool instructions when needed."""
    prepared = [msg.model_copy(deep=True) for msg in source_messages]

    instructions: list[str] = []
    if tools:
        tool_prompt = _build_tool_prompt(tools, tool_choice)
        if tool_prompt:
            instructions.append(tool_prompt)

    if extra_instructions:
        instructions.extend(instr for instr in extra_instructions if instr)

    if not instructions:
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


def _strip_xml_hint(text: str) -> str:
    """Remove the XML wrap hint text from a given string."""
    if not text:
        return text
    cleaned = text.replace(XML_WRAP_HINT, "").replace(XML_HINT_STRIPPED, "")
    return cleaned.strip()


def _response_items_to_messages(items: list[ResponseInputItem]) -> list[Message]:
    """Convert Responses API input items into internal Message objects."""
    messages: list[Message] = []

    for item in items:
        if item.type != "message":
            continue

        role = item.role
        if role == "developer":
            role = "system"

        content = item.content
        if isinstance(content, list):
            converted: list[ContentItem] = []
            for part in content:
                if part.type == "input_text":
                    if part.text:
                        converted.append(ContentItem(type="text", text=part.text))
                elif part.type == "input_image":
                    image_url = part.image_url
                    if not image_url and part.image_base64:
                        mime_type = part.mime_type or "image/png"
                        image_url = f"data:{mime_type};base64,{part.image_base64}"
                    if image_url:
                        converted.append(
                            ContentItem(type="image_url", image_url={"url": image_url})
                        )
            messages.append(Message(role=role, content=converted or None))
        else:
            messages.append(Message(role=role, content=content))

    return messages


def _remove_tool_call_blocks(text: str) -> str:
    """Strip tool call code blocks from text."""
    if not text:
        return text
    cleaned = TOOL_BLOCK_RE.sub("", text)
    return _strip_xml_hint(cleaned)


def _extract_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Extract tool call definitions and return cleaned text."""
    if not text:
        return text, []

    tool_calls: list[ToolCall] = []

    def _replace(match: re.Match[str]) -> str:
        block_content = match.group(1)
        if not block_content:
            return ""

        for call_match in TOOL_CALL_RE.finditer(block_content):
            name = (call_match.group(1) or "").strip()
            raw_args = (call_match.group(2) or "").strip()
            if not name:
                logger.warning(
                    f"Encountered tool_call block without a function name: {block_content}"
                )
                continue

            arguments = raw_args
            try:
                parsed_args = json.loads(raw_args)
                arguments = json.dumps(parsed_args, ensure_ascii=False)
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse tool call arguments for '{name}'. Passing raw string."
                )

            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex}",
                    type="function",
                    function=FunctionCall(name=name, arguments=arguments),
                )
            )

        return ""

    cleaned = TOOL_BLOCK_RE.sub(_replace, text)
    cleaned = _strip_xml_hint(cleaned)
    return cleaned, tool_calls


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    now = int(datetime.now(tz=timezone.utc).timestamp())

    models = []
    for model in Model:
        m_name = model.model_name
        if not m_name or m_name == "unspecified":
            continue

        models.append(
            ModelData(
                id=m_name,
                created=now,
                owned_by="gemini-web",
            )
        )

    return ModelListResponse(data=models)


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    pool = GeminiClientPool()
    db = LMDBConversationStore()
    model = Model.from_name(request.model)

    if len(request.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one message is required in the conversation.",
        )

    structured_requirement = _build_structured_requirement(request.response_format)
    if structured_requirement and request.stream:
        logger.debug(
            "Structured response requested with streaming enabled; will stream canonical JSON once ready."
        )

    extra_instructions = [structured_requirement.instruction] if structured_requirement else None

    # Check if conversation is reusable
    session, client, remaining_messages = _find_reusable_session(db, pool, model, request.messages)

    if session:
        messages_to_send = _prepare_messages_for_model(
            remaining_messages, request.tools, request.tool_choice, extra_instructions
        )
        if not messages_to_send:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No new messages to send for the existing session.",
            )
        if len(messages_to_send) == 1:
            model_input, files = await GeminiClientWrapper.process_message(
                messages_to_send[0], tmp_dir, tagged=False
            )
        else:
            model_input, files = await GeminiClientWrapper.process_conversation(
                messages_to_send, tmp_dir
            )
        logger.debug(
            f"Reused session {session.metadata} - sending {len(messages_to_send)} prepared messages."
        )
    else:
        # Start a new session and concat messages into a single string
        try:
            client = pool.acquire()
            session = client.start_chat(model=model)
            messages_to_send = _prepare_messages_for_model(
                request.messages, request.tools, request.tool_choice, extra_instructions
            )
            model_input, files = await GeminiClientWrapper.process_conversation(
                messages_to_send, tmp_dir
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            logger.exception(f"Error in preparing conversation: {e}")
            raise
        logger.debug("New session started.")

    # Generate response
    try:
        assert session and client, "Session and client not available"
        logger.debug(
            f"Client ID: {client.id}, Input length: {len(model_input)}, files count: {len(files)}"
        )
        response = await _send_with_split(session, model_input, files=files)
    except Exception as e:
        logger.exception(f"Error generating content from Gemini API: {e}")
        raise

    # Format the response from API
    raw_output_with_think = GeminiClientWrapper.extract_output(response, include_thoughts=True)
    raw_output_clean = GeminiClientWrapper.extract_output(response, include_thoughts=False)

    visible_output, tool_calls = _extract_tool_calls(raw_output_with_think)
    storage_output = _remove_tool_call_blocks(raw_output_clean).strip()
    tool_calls_payload = [call.model_dump(mode="json") for call in tool_calls]

    if structured_requirement:
        cleaned_visible = _strip_code_fence(visible_output or "")
        if not cleaned_visible:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM returned an empty response while JSON schema output was requested.",
            )
        try:
            structured_payload = json.loads(cleaned_visible)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"Failed to decode JSON for structured response (schema={structured_requirement.schema_name}): "
                f"{cleaned_visible}"
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM returned invalid JSON for the requested response_format.",
            ) from exc

        canonical_output = json.dumps(structured_payload, ensure_ascii=False)
        visible_output = canonical_output
        storage_output = canonical_output

    if tool_calls_payload:
        logger.debug(f"Detected tool calls: {tool_calls_payload}")

    # After formatting, persist the conversation to LMDB
    try:
        last_message = Message(
            role="assistant",
            content=storage_output or None,
            tool_calls=tool_calls or None,
        )
        cleaned_history = db.sanitize_assistant_messages(request.messages)
        conv = ConversationInStore(
            model=model.model_name,
            client_id=client.id,
            metadata=session.metadata,
            messages=[*cleaned_history, last_message],
        )
        key = db.store(conv)
        logger.debug(f"Conversation saved to LMDB with key: {key}")
    except Exception as e:
        # We can still return the response even if saving fails
        logger.warning(f"Failed to save conversation to LMDB: {e}")

    # Return with streaming or standard response
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    timestamp = int(datetime.now(tz=timezone.utc).timestamp())
    if request.stream:
        return _create_streaming_response(
            visible_output,
            tool_calls_payload,
            completion_id,
            timestamp,
            request.model,
            request.messages,
        )
    else:
        return _create_standard_response(
            visible_output,
            tool_calls_payload,
            completion_id,
            timestamp,
            request.model,
            request.messages,
        )


@router.post("/v1/responses")
async def create_response(
    request: ResponseCreateRequest,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming responses are not supported for this endpoint.",
        )

    messages = _response_items_to_messages(request.input)
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No message input provided."
        )

    pool = GeminiClientPool()
    db = LMDBConversationStore()

    try:
        model = Model.from_name(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    session, client, remaining_messages = _find_reusable_session(db, pool, model, messages)

    if session:
        messages_to_send = remaining_messages
        if not messages_to_send:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No new messages to send for the existing session.",
            )
        if len(messages_to_send) == 1:
            model_input, files = await GeminiClientWrapper.process_message(
                messages_to_send[0], tmp_dir, tagged=False
            )
        else:
            model_input, files = await GeminiClientWrapper.process_conversation(
                messages_to_send, tmp_dir
            )
        logger.debug(
            f"Reused session {session.metadata} - sending {len(messages_to_send)} prepared messages."
        )
    else:
        try:
            client = pool.acquire()
            session = client.start_chat(model=model)
            model_input, files = await GeminiClientWrapper.process_conversation(messages, tmp_dir)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            logger.exception(f"Error in preparing conversation for responses API: {e}")
            raise
        logger.debug("New session started for /v1/responses request.")

    try:
        assert session and client, "Session and client not available"
        logger.debug(
            f"Client ID: {client.id}, Input length: {len(model_input)}, files count: {len(files)}"
        )
        model_output = await _send_with_split(session, model_input, files=files)
    except Exception as e:
        logger.exception(f"Error generating content from Gemini API for responses: {e}")
        raise

    text_with_think = GeminiClientWrapper.extract_output(model_output, include_thoughts=True)
    text_without_think = GeminiClientWrapper.extract_output(model_output, include_thoughts=False)

    storage_output = _remove_tool_call_blocks(text_without_think).strip()
    visible_text = _remove_tool_call_blocks(text_with_think).strip()
    assistant_text = LMDBConversationStore.remove_think_tags(visible_text)

    expects_image = (
        request.tool_choice is not None and request.tool_choice.type == "image_generation"
    )
    if expects_image and not model_output.images:
        summary = assistant_text.strip() if assistant_text else ""
        if summary:
            summary = re.sub(r"\s+", " ", summary)
            if len(summary) > 200:
                summary = f"{summary[:197]}..."
        logger.warning(
            "Image generation was requested via tool_choice but Gemini returned no images."
        )
        detail = "LLM returned no images for the requested image_generation tool."
        if summary:
            detail = f"{detail} Assistant response: {summary}"
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)

    image_contents: list[ResponseOutputContent] = []
    image_call_items: list[ResponseImageGenerationCall] = []
    for image in model_output.images:
        try:
            image_base64, width, height = await _image_to_base64(image, tmp_dir)
        except Exception as exc:
            logger.warning(f"Failed to download generated image: {exc}")
            continue
        mime_type = "image/png" if isinstance(image, GeneratedImage) else "image/jpeg"
        image_contents.append(
            ResponseOutputContent(
                type="output_image",
                image_base64=image_base64,
                mime_type=mime_type,
                width=width,
                height=height,
            )
        )
        image_call_items.append(
            ResponseImageGenerationCall(
                id=f"img_{uuid.uuid4().hex}",
                status="completed",
                result=image_base64,
                output_format="png" if isinstance(image, GeneratedImage) else "jpeg",
                size=f"{width}x{height}" if width and height else None,
            )
        )

    response_contents: list[ResponseOutputContent] = []
    if assistant_text:
        response_contents.append(ResponseOutputContent(type="output_text", text=assistant_text))
    response_contents.extend(image_contents)

    if not response_contents:
        response_contents.append(ResponseOutputContent(type="output_text", text=""))

    created_time = int(datetime.now(tz=timezone.utc).timestamp())
    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"

    prompt_tokens = sum(estimate_tokens(_text_from_message(msg)) for msg in messages)
    completion_tokens = estimate_tokens(assistant_text)
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    response_payload = ResponseCreateResponse(
        id=response_id,
        created=created_time,
        model=request.model,
        output=[
            ResponseOutputMessage(
                id=message_id,
                type="message",
                role="assistant",
                content=response_contents,
            ),
            *image_call_items,
        ],
        output_text=assistant_text or None,
        usage=usage,
    )

    try:
        last_message = Message(role="assistant", content=storage_output or None)
        cleaned_history = db.sanitize_assistant_messages(messages)
        conv = ConversationInStore(
            model=model.model_name,
            client_id=client.id,
            metadata=session.metadata,
            messages=[*cleaned_history, last_message],
        )
        db.store(conv)
    except Exception as exc:
        logger.warning(f"Failed to save Responses conversation to LMDB: {exc}")

    return response_payload


def _text_from_message(message: Message) -> str:
    """Return text content from a message for token estimation."""
    base_text = ""
    if isinstance(message.content, str):
        base_text = message.content
    elif isinstance(message.content, list):
        base_text = "\n".join(
            item.text or "" for item in message.content if getattr(item, "type", "") == "text"
        )
    elif message.content is None:
        base_text = ""

    if message.tool_calls:
        tool_arg_text = "".join(call.function.arguments or "" for call in message.tool_calls)
        base_text = f"{base_text}\n{tool_arg_text}" if base_text else tool_arg_text

    return base_text


def _find_reusable_session(
    db: LMDBConversationStore,
    pool: GeminiClientPool,
    model: Model,
    messages: list[Message],
) -> tuple[ChatSession | None, GeminiClientWrapper | None, list[Message]]:
    """Find an existing chat session that matches the *longest* prefix of
    ``messages`` **whose last element is an assistant/system reply**.

    Rationale
    ---------
    When a reply was generated by *another* server instance, the local LMDB may
    only contain an older part of the conversation.  However, as long as we can
    line up **any** earlier assistant/system response, we can restore the
    corresponding Gemini session and replay the *remaining* turns locally
    (including that missing assistant reply and the subsequent user prompts).

    The algorithm therefore walks backwards through the history **one message at
    a time**, each time requiring the current tail to be assistant/system before
    querying LMDB.  As soon as a match is found we recreate the session and
    return the untouched suffix as ``remaining_messages``.
    """

    if len(messages) < 2:
        return None, None, messages

    # Start with the full history and iteratively trim from the end.
    search_end = len(messages)
    while search_end >= 2:
        search_history = messages[:search_end]

        # Only try to match if the last stored message would be assistant/system.
        if search_history[-1].role in {"assistant", "system"}:
            try:
                if conv := db.find(model.model_name, search_history):
                    client = pool.acquire(conv.client_id)
                    session = client.start_chat(metadata=conv.metadata, model=model)
                    remain = messages[search_end:]
                    return session, client, remain
            except Exception as e:
                logger.warning(f"Error checking LMDB for reusable session: {e}")
                break

        # Trim one message and try again.
        search_end -= 1

    return None, None, messages


async def _send_with_split(session: ChatSession, text: str, files: list[Path | str] | None = None):
    """Send text to Gemini, automatically splitting into multiple batches if it is
    longer than ``MAX_CHARS_PER_REQUEST``.

    Every intermediate batch (that is **not** the last one) is suffixed with a hint
    telling Gemini that more content will come, and it should simply reply with
    "ok". The final batch carries any file uploads and the real user prompt so
    that Gemini can produce the actual answer.
    """
    if len(text) <= MAX_CHARS_PER_REQUEST:
        # No need to split - a single request is fine.
        return await session.send_message(text, files=files)
    hint_len = len(CONTINUATION_HINT)
    chunk_size = MAX_CHARS_PER_REQUEST - hint_len

    chunks: list[str] = []
    pos = 0
    total = len(text)
    while pos < total:
        end = min(pos + chunk_size, total)
        chunk = text[pos:end]
        pos = end

        # If this is NOT the last chunk, add the continuation hint.
        if end < total:
            chunk += CONTINUATION_HINT
        chunks.append(chunk)

    # Fire off all but the last chunk, discarding the interim "ok" replies.
    for chk in chunks[:-1]:
        try:
            await session.send_message(chk)
        except Exception as e:
            logger.exception(f"Error sending chunk to Gemini: {e}")
            raise

    # The last chunk carries the files (if any) and we return its response.
    return await session.send_message(chunks[-1], files=files)


def _create_streaming_response(
    model_output: str,
    tool_calls: list[dict],
    completion_id: str,
    created_time: int,
    model: str,
    messages: list[Message],
) -> StreamingResponse:
    """Create streaming response with `usage` calculation included in the final chunk."""

    # Calculate token usage
    prompt_tokens = sum(estimate_tokens(_text_from_message(msg)) for msg in messages)
    tool_args = "".join(call.get("function", {}).get("arguments", "") for call in tool_calls or [])
    completion_tokens = estimate_tokens(model_output + tool_args)
    total_tokens = prompt_tokens + completion_tokens
    finish_reason = "tool_calls" if tool_calls else "stop"

    async def generate_stream():
        # Send start event
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        # Stream output text in chunks for efficiency
        chunk_size = 32
        if model_output:
            for i in range(0, len(model_output), chunk_size):
                chunk = model_output[i : i + chunk_size]
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        if tool_calls:
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": tool_calls},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        # Send end event
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _create_standard_response(
    model_output: str,
    tool_calls: list[dict],
    completion_id: str,
    created_time: int,
    model: str,
    messages: list[Message],
) -> dict:
    """Create standard response"""
    # Calculate token usage
    prompt_tokens = sum(estimate_tokens(_text_from_message(msg)) for msg in messages)
    tool_args = "".join(call.get("function", {}).get("arguments", "") for call in tool_calls or [])
    completion_tokens = estimate_tokens(model_output + tool_args)
    total_tokens = prompt_tokens + completion_tokens
    finish_reason = "tool_calls" if tool_calls else "stop"

    message_payload: dict = {"role": "assistant", "content": model_output or None}
    if tool_calls:
        message_payload["tool_calls"] = tool_calls

    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message_payload,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    logger.debug(f"Response created with {total_tokens} total tokens")
    return result


def _extract_image_dimensions(data: bytes) -> tuple[int | None, int | None]:
    """Return image dimensions (width, height) if PNG or JPEG headers are present."""
    # PNG: dimensions stored in bytes 16..24 of the IHDR chunk
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        try:
            width, height = struct.unpack(">II", data[16:24])
            return int(width), int(height)
        except struct.error:
            return None, None

    # JPEG: dimensions stored in SOF segment; iterate through markers to locate it
    if len(data) >= 4 and data[0:2] == b"\xff\xd8":
        idx = 2
        length = len(data)
        sof_markers = {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
        while idx < length:
            # Find marker alignment (markers are prefixed with 0xFF bytes)
            if data[idx] != 0xFF:
                idx += 1
                continue
            while idx < length and data[idx] == 0xFF:
                idx += 1
            if idx >= length:
                break
            marker = data[idx]
            idx += 1

            if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
                continue

            if idx + 1 >= length:
                break
            segment_length = (data[idx] << 8) + data[idx + 1]
            idx += 2
            if segment_length < 2:
                break

            if marker in sof_markers:
                if idx + 4 < length:
                    # Skip precision byte at idx, then read height/width (big-endian)
                    height = (data[idx + 1] << 8) + data[idx + 2]
                    width = (data[idx + 3] << 8) + data[idx + 4]
                    return int(width), int(height)
                break

            idx += segment_length - 2

    return None, None


async def _image_to_base64(image: Image, temp_dir: Path) -> tuple[str, int | None, int | None]:
    """Persist an image provided by gemini_webapi and return base64 plus dimensions."""
    if isinstance(image, GeneratedImage):
        saved_path = await image.save(path=str(temp_dir), full_size=True)
    else:
        saved_path = await image.save(path=str(temp_dir))

    if not saved_path:
        raise ValueError("Failed to save generated image")

    data = Path(saved_path).read_bytes()
    width, height = _extract_image_dimensions(data)
    return base64.b64encode(data).decode("utf-8"), width, height
