import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import orjson
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from loguru import logger

from ..models import (
    ChatCompletionRequest,
    ConversationInStore,
    FunctionCall,
    Message,
    ModelData,
    ModelListResponse,
    Tool,
    ToolCall,
    ToolChoiceFunction,
)
from ..services import GeminiClientPool, GeminiClientWrapper, LMDBConversationStore
from ..services.client import XML_WRAP_HINT
from ..utils import g_config
from ..utils.helper import estimate_tokens
from .middleware import get_temp_dir, verify_api_key

# Maximum characters Gemini Web can accept in a single request (configurable)
MAX_CHARS_PER_REQUEST = int(g_config.gemini.max_chars_per_request * 0.9)

CONTINUATION_HINT = "\n(More messages to come, please reply with just 'ok.')"

TOOL_BLOCK_RE = re.compile(r"```xml\s*(.*?)```", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(
    r"<tool_call\s+name=\"([^\"]+)\">(.*?)</tool_call>", re.DOTALL | re.IGNORECASE
)
XML_HINT_STRIPPED = XML_WRAP_HINT.strip()


router = APIRouter()


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
        "When you decide to call a tool, respond with NOTHING except the following format wrapped inside a ```xml``` fenced block:"
    )
    lines.append("```xml")
    lines.append('<tool_call name="tool_name">{"argument": "value"}</tool_call>')
    lines.append("```")
    lines.append(
        "Use double quotes for JSON keys and values. If multiple tool calls are required, include multiple <tool_call> entries inside the same fenced block. Without a tool call, reply normally without any XML block."
    )

    return "\n".join(lines)


def _prepare_messages_for_model(
    source_messages: list[Message],
    tools: list[Tool] | None,
    tool_choice: str | ToolChoiceFunction | None,
) -> list[Message]:
    """Return a copy of messages enriched with tool instructions when needed."""
    prepared = [msg.model_copy(deep=True) for msg in source_messages]

    if not tools:
        return prepared

    instructions = _build_tool_prompt(tools, tool_choice)
    if not instructions:
        return prepared

    if prepared and prepared[0].role == "system" and isinstance(prepared[0].content, str):
        existing = prepared[0].content or ""
        separator = "\n\n" if existing else ""
        prepared[0].content = f"{existing}{separator}{instructions}"
    else:
        prepared.insert(0, Message(role="system", content=instructions))

    return prepared


def _strip_xml_hint(text: str) -> str:
    """Remove the XML wrap hint text from a given string."""
    if not text:
        return text
    cleaned = text.replace(XML_WRAP_HINT, "").replace(XML_HINT_STRIPPED, "")
    return cleaned.strip()


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
                logger.warning("Encountered tool_call block without a function name: %s", block_content)
                continue

            arguments = raw_args
            try:
                parsed_args = json.loads(raw_args)
                arguments = json.dumps(parsed_args, ensure_ascii=False)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse tool call arguments for '%s'. Passing raw string.", name
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

    # Check if conversation is reusable
    session, client, remaining_messages = _find_reusable_session(db, pool, model, request.messages)

    if session:
        messages_to_send = _prepare_messages_for_model(
            remaining_messages, request.tools, request.tool_choice
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
                request.messages, request.tools, request.tool_choice
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

    if tool_calls_payload:
        logger.debug("Detected tool calls: %s", tool_calls_payload)

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
    line-up **any** earlier assistant/system response, we can restore the
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
    tool_args = "".join(
        call.get("function", {}).get("arguments", "") for call in tool_calls or []
    )
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
                "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}],
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
    tool_args = "".join(
        call.get("function", {}).get("arguments", "") for call in tool_calls or []
    )
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
