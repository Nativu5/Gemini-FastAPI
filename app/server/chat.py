import base64
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from gemini_webapi.exceptions import APIError
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
    XML_HINT_STRIPPED,
    XML_WRAP_HINT,
    estimate_tokens,
    extract_image_dimensions,
    extract_tool_calls,
    iter_stream_segments,
    remove_tool_call_blocks,
    strip_code_fence,
    text_from_message,
)
from .middleware import get_image_store_dir, get_image_token, get_temp_dir, verify_api_key

# Maximum characters Gemini Web can accept in a single request (configurable)
MAX_CHARS_PER_REQUEST = int(g_config.gemini.max_chars_per_request * 0.9)
CONTINUATION_HINT = "\n(More messages to come, please reply with just 'ok.')"

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
        logger.debug(
            f"Applied {len(extra_instructions)} extra instructions for tool/structured output."
        )

    if not _conversation_has_code_hint(prepared):
        instructions.append(CODE_BLOCK_HINT)
        logger.debug("Injected default code block hint for Gemini conversation.")

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
        if role == "developer":
            role = "system"

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

    logger.debug(
        f"Normalized Responses input: {len(normalized_input)} message items (developer roles mapped to system)."
    )
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
        if role == "developer":
            role = "system"

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
    """
    Retrieve a Model instance by name, considering custom models from config
    and the update strategy (append or overwrite).
    """
    strategy = g_config.gemini.model_strategy
    custom_models = {m.model_name: m for m in g_config.gemini.models if m.model_name}

    if name in custom_models:
        return Model.from_dict(custom_models[name].model_dump())

    if strategy == "overwrite":
        raise ValueError(f"Model '{name}' not found in custom models (strategy='overwrite').")

    return Model.from_name(name)


def _get_available_models() -> list[ModelData]:
    """
    Return a list of available models based on configuration strategy.
    """
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
    pool = GeminiClientPool()
    db = LMDBConversationStore()

    try:
        model = _get_model_by_name(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

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
    if structured_requirement:
        logger.debug(
            f"Structured response requested for /v1/chat/completions (schema={structured_requirement.schema_name})."
        )

    extra_instructions = [structured_requirement.instruction] if structured_requirement else None

    # Check if conversation is reusable
    session, client, remaining_messages = await _find_reusable_session(
        db, pool, model, request.messages
    )

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
            client = await pool.acquire()
            session = client.start_chat(model=model)
            messages_to_send = _prepare_messages_for_model(
                request.messages, request.tools, request.tool_choice, extra_instructions
            )
            model_input, files = await GeminiClientWrapper.process_conversation(
                messages_to_send, tmp_dir
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
        except Exception as e:
            logger.exception(f"Error in preparing conversation: {e}")
            raise
        logger.debug("New session started.")

    # Generate response
    try:
        assert session and client, "Session and client not available"
        client_id = client.id
        logger.debug(
            f"Client ID: {client_id}, Input length: {len(model_input)}, files count: {len(files)}"
        )
        response = await _send_with_split(session, model_input, files=files)
    except APIError as exc:
        client_id = client.id if client else "unknown"
        logger.warning(f"Gemini API returned invalid response for client {client_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini temporarily returned an invalid response. Please retry.",
        ) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error generating content from Gemini API: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned an unexpected error.",
        ) from e

    # Format the response from API
    try:
        raw_output_with_think = GeminiClientWrapper.extract_output(response, include_thoughts=True)
        raw_output_clean = GeminiClientWrapper.extract_output(response, include_thoughts=False)
    except IndexError as exc:
        logger.exception("Gemini output parsing failed (IndexError).")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned malformed response content.",
        ) from exc
    except Exception as exc:
        logger.exception("Gemini output parsing failed unexpectedly.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini output parsing failed unexpectedly.",
        ) from exc

    visible_output, tool_calls = extract_tool_calls(raw_output_with_think)
    storage_output = remove_tool_call_blocks(raw_output_clean).strip()
    tool_calls_payload = [call.model_dump(mode="json") for call in tool_calls]

    if structured_requirement:
        cleaned_visible = strip_code_fence(visible_output or "")
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
    request_data: ResponseCreateRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
    image_store: Path = Depends(get_image_store_dir),
):
    base_messages, normalized_input = _response_items_to_messages(request_data.input)
    structured_requirement = _build_structured_requirement(request_data.response_format)
    if structured_requirement and request_data.stream:
        logger.debug(
            "Structured response requested with streaming enabled; streaming not supported for Responses."
        )

    extra_instructions: list[str] = []
    if structured_requirement:
        extra_instructions.append(structured_requirement.instruction)
        logger.debug(
            f"Structured response requested for /v1/responses (schema={structured_requirement.schema_name})."
        )

    # Separate standard tools from image generation tools
    standard_tools: list[Tool] = []
    image_tools: list[ResponseImageTool] = []

    if request_data.tools:
        for t in request_data.tools:
            if isinstance(t, Tool):
                standard_tools.append(t)
            elif isinstance(t, ResponseImageTool):
                image_tools.append(t)
            # Handle dicts if Pydantic didn't convert them fully (fallback)
            elif isinstance(t, dict):
                t_type = t.get("type")
                if t_type == "function":
                    standard_tools.append(Tool.model_validate(t))
                elif t_type == "image_generation":
                    image_tools.append(ResponseImageTool.model_validate(t))

    image_instruction = _build_image_generation_instruction(
        image_tools,
        request_data.tool_choice
        if isinstance(request_data.tool_choice, ResponseToolChoice)
        else None,
    )
    if image_instruction:
        extra_instructions.append(image_instruction)
        logger.debug("Image generation support enabled for /v1/responses request.")

    preface_messages = _instructions_to_messages(request_data.instructions)
    conversation_messages = base_messages
    if preface_messages:
        conversation_messages = [*preface_messages, *base_messages]
        logger.debug(
            f"Injected {len(preface_messages)} instruction messages before sending to Gemini."
        )

    # Pass standard tools to the prompt builder
    # Determine tool_choice for standard tools (ignore image_generation choice here as it is handled via instruction)
    model_tool_choice = None
    if isinstance(request_data.tool_choice, str):
        model_tool_choice = request_data.tool_choice
    elif isinstance(request_data.tool_choice, ToolChoiceFunction):
        model_tool_choice = request_data.tool_choice
    # If tool_choice is ResponseToolChoice (image_generation), we don't pass it as a function tool choice.

    messages = _prepare_messages_for_model(
        conversation_messages,
        tools=standard_tools or None,
        tool_choice=model_tool_choice,
        extra_instructions=extra_instructions or None,
    )

    pool = GeminiClientPool()
    db = LMDBConversationStore()

    try:
        model = _get_model_by_name(request_data.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    session, client, remaining_messages = await _find_reusable_session(db, pool, model, messages)

    async def _build_payload(
        _payload_messages: list[Message], _reuse_session: bool
    ) -> tuple[str, list[Path | str]]:
        if _reuse_session and len(_payload_messages) == 1:
            return await GeminiClientWrapper.process_message(
                _payload_messages[0], tmp_dir, tagged=False
            )
        return await GeminiClientWrapper.process_conversation(_payload_messages, tmp_dir)

    reuse_session = session is not None
    if reuse_session:
        messages_to_send = _prepare_messages_for_model(
            remaining_messages,
            tools=None,
            tool_choice=None,
            extra_instructions=extra_instructions or None,
        )
        if not messages_to_send:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No new messages to send for the existing session.",
            )
        payload_messages = messages_to_send
        model_input, files = await _build_payload(payload_messages, _reuse_session=True)
        logger.debug(
            f"Reused session {session.metadata} - sending {len(payload_messages)} prepared messages."
        )
    else:
        try:
            client = await pool.acquire()
            session = client.start_chat(model=model)
            payload_messages = messages
            model_input, files = await _build_payload(payload_messages, _reuse_session=False)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
        except Exception as e:
            logger.exception(f"Error in preparing conversation for responses API: {e}")
            raise
        logger.debug("New session started for /v1/responses request.")

    try:
        assert session and client, "Session and client not available"
        client_id = client.id
        logger.debug(
            f"Client ID: {client_id}, Input length: {len(model_input)}, files count: {len(files)}"
        )
        model_output = await _send_with_split(session, model_input, files=files)
    except APIError as exc:
        client_id = client.id if client else "unknown"
        logger.warning(f"Gemini API returned invalid response for client {client_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini temporarily returned an invalid response. Please retry.",
        ) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error generating content from Gemini API for responses: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned an unexpected error.",
        ) from e

    try:
        text_with_think = GeminiClientWrapper.extract_output(model_output, include_thoughts=True)
        text_without_think = GeminiClientWrapper.extract_output(
            model_output, include_thoughts=False
        )
    except IndexError as exc:
        logger.exception("Gemini output parsing failed (IndexError).")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned malformed response content.",
        ) from exc
    except Exception as exc:
        logger.exception("Gemini output parsing failed unexpectedly.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini output parsing failed unexpectedly.",
        ) from exc

    visible_text, detected_tool_calls = extract_tool_calls(text_with_think)
    storage_output = remove_tool_call_blocks(text_without_think).strip()
    assistant_text = LMDBConversationStore.remove_think_tags(visible_text.strip())

    if structured_requirement:
        cleaned_visible = strip_code_fence(assistant_text or "")
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
        assistant_text = canonical_output
        storage_output = canonical_output
        logger.debug(
            f"Structured response fulfilled for /v1/responses (schema={structured_requirement.schema_name})."
        )

    expects_image = (
        request_data.tool_choice is not None and request_data.tool_choice.type == "image_generation"
    )
    images = model_output.images or []
    logger.debug(
        f"Gemini returned {len(images)} image(s) for /v1/responses "
        f"(expects_image={expects_image}, instruction_applied={bool(image_instruction)})."
    )
    if expects_image and not images:
        summary = assistant_text.strip() if assistant_text else ""
        if summary:
            summary = re.sub(r"\s+", " ", summary)
            if len(summary) > 200:
                summary = f"{summary[:197]}..."
        logger.warning(
            "Image generation requested but Gemini produced no images. "
            f"client_id={client_id}, forced_tool_choice={request_data.tool_choice is not None}, "
            f"instruction_applied={bool(image_instruction)}, assistant_preview='{summary}'"
        )
        detail = "LLM returned no images for the requested image_generation tool."
        if summary:
            detail = f"{detail} Assistant response: {summary}"
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)

    response_contents: list[ResponseOutputContent] = []
    image_call_items: list[ResponseImageGenerationCall] = []
    for image in images:
        try:
            image_base64, width, height, filename = await _image_to_base64(image, image_store)
        except Exception as exc:
            logger.warning(f"Failed to download generated image: {exc}")
            continue

        img_format = "png" if isinstance(image, GeneratedImage) else "jpeg"

        # Use static URL for compatibility
        image_url = (
            f"![{filename}]({request.base_url}images/{filename}?token={get_image_token(filename)})"
        )

        image_call_items.append(
            ResponseImageGenerationCall(
                id=filename.split(".")[0],
                status="completed",
                result=image_base64,
                output_format=img_format,
                size=f"{width}x{height}" if width and height else None,
            )
        )
        # Add as output_text content for compatibility
        response_contents.append(
            ResponseOutputContent(type="output_text", text=image_url, annotations=[])
        )

    tool_call_items: list[ResponseToolCall] = []
    if detected_tool_calls:
        tool_call_items = [
            ResponseToolCall(
                id=call.id,
                status="completed",
                function=call.function,
            )
            for call in detected_tool_calls
        ]

    if assistant_text:
        response_contents.append(
            ResponseOutputContent(type="output_text", text=assistant_text, annotations=[])
        )

    if not response_contents:
        response_contents.append(ResponseOutputContent(type="output_text", text="", annotations=[]))

    created_time = int(datetime.now(tz=timezone.utc).timestamp())
    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"

    input_tokens = sum(estimate_tokens(text_from_message(msg)) for msg in messages)
    tool_arg_text = "".join(call.function.arguments or "" for call in detected_tool_calls)
    completion_basis = assistant_text or ""
    if tool_arg_text:
        completion_basis = (
            f"{completion_basis}\n{tool_arg_text}" if completion_basis else tool_arg_text
        )
    output_tokens = estimate_tokens(completion_basis)
    usage = ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )

    response_payload = ResponseCreateResponse(
        id=response_id,
        created_at=created_time,
        model=request_data.model,
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
        metadata=request_data.metadata or None,
        tools=request_data.tools,
        tool_choice=request_data.tool_choice,
    )

    try:
        last_message = Message(
            role="assistant",
            content=storage_output or None,
            tool_calls=detected_tool_calls or None,
        )
        cleaned_history = db.sanitize_assistant_messages(messages)
        conv = ConversationInStore(
            model=model.model_name,
            client_id=client.id,
            metadata=session.metadata,
            messages=[*cleaned_history, last_message],
        )
        key = db.store(conv)
        logger.debug(f"Conversation saved to LMDB with key: {key}")
    except Exception as exc:
        logger.warning(f"Failed to save Responses conversation to LMDB: {exc}")

    if request_data.stream:
        logger.debug(
            f"Streaming Responses API payload (response_id={response_payload.id}, text_chunks={bool(assistant_text)})."
        )
        return _create_responses_streaming_response(response_payload, assistant_text or "")

    return response_payload


async def _find_reusable_session(
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
                    client = await pool.acquire(conv.client_id)
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
        try:
            return await session.send_message(text, files=files)
        except Exception as e:
            logger.exception(f"Error sending message to Gemini: {e}")
            raise
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
    try:
        return await session.send_message(chunks[-1], files=files)
    except Exception as e:
        logger.exception(f"Error sending final chunk to Gemini: {e}")
        raise


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
    prompt_tokens = sum(estimate_tokens(text_from_message(msg)) for msg in messages)
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
        for chunk in iter_stream_segments(model_output):
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
            yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        if tool_calls:
            tool_calls_delta = [{**call, "index": idx} for idx, call in enumerate(tool_calls)]
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": tool_calls_delta},
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


def _create_responses_streaming_response(
    response_payload: ResponseCreateResponse,
    assistant_text: str | None,
) -> StreamingResponse:
    """Create streaming response for Responses API using event types defined by OpenAI."""

    response_dict = response_payload.model_dump(mode="json")
    response_id = response_payload.id
    created_time = response_payload.created_at
    model = response_payload.model

    logger.debug(
        f"Preparing streaming envelope for /v1/responses (response_id={response_id}, model={model})."
    )

    base_event = {
        "id": response_id,
        "object": "response",
        "created_at": created_time,
        "model": model,
    }

    created_snapshot: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_time,
        "model": model,
        "status": "in_progress",
    }
    if response_dict.get("metadata") is not None:
        created_snapshot["metadata"] = response_dict["metadata"]
    if response_dict.get("input") is not None:
        created_snapshot["input"] = response_dict["input"]
    if response_dict.get("tools") is not None:
        created_snapshot["tools"] = response_dict["tools"]
    if response_dict.get("tool_choice") is not None:
        created_snapshot["tool_choice"] = response_dict["tool_choice"]

    async def generate_stream():
        # Emit creation event
        data = {
            **base_event,
            "type": "response.created",
            "response": created_snapshot,
        }
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        # Stream output items (Message/Text, Tool Calls, Images)
        for i, item in enumerate(response_payload.output):
            item_json = item.model_dump(mode="json", exclude_none=True)

            added_event = {
                **base_event,
                "type": "response.output_item.added",
                "output_index": i,
                "item": item_json,
            }
            yield f"data: {orjson.dumps(added_event).decode('utf-8')}\n\n"

            # 2. Stream content if it's a message (text)
            if item.type == "message":
                content_text = ""
                # Aggregate text content to stream
                for c in item.content:
                    if c.type == "output_text" and c.text:
                        content_text += c.text

                if content_text:
                    for chunk in iter_stream_segments(content_text):
                        delta_event = {
                            **base_event,
                            "type": "response.output_text.delta",
                            "output_index": i,
                            "delta": chunk,
                        }
                        yield f"data: {orjson.dumps(delta_event).decode('utf-8')}\n\n"

                    # Text done
                    done_event = {
                        **base_event,
                        "type": "response.output_text.done",
                        "output_index": i,
                    }
                    yield f"data: {orjson.dumps(done_event).decode('utf-8')}\n\n"

            # 3. Emit output_item.done for all types
            # This confirms the item is fully transferred.
            item_done_event = {
                **base_event,
                "type": "response.output_item.done",
                "output_index": i,
                "item": item_json,
            }
            yield f"data: {orjson.dumps(item_done_event).decode('utf-8')}\n\n"

        # Emit completed event with full payload
        completed_event = {
            **base_event,
            "type": "response.completed",
            "response": response_dict,
        }
        yield f"data: {orjson.dumps(completed_event).decode('utf-8')}\n\n"
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
    prompt_tokens = sum(estimate_tokens(text_from_message(msg)) for msg in messages)
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


async def _image_to_base64(image: Image, temp_dir: Path) -> tuple[str, int | None, int | None, str]:
    """Persist an image provided by gemini_webapi and return base64 plus dimensions and filename."""
    if isinstance(image, GeneratedImage):
        try:
            saved_path = await image.save(path=str(temp_dir), full_size=True)
        except Exception as e:
            logger.warning(f"Failed to download full-size image, retrying with default size: {e}")
            saved_path = await image.save(path=str(temp_dir), full_size=False)
    else:
        saved_path = await image.save(path=str(temp_dir))

    if not saved_path:
        raise ValueError("Failed to save generated image")

    # Rename file to a random UUID to ensure uniqueness and unpredictability
    original_path = Path(saved_path)
    random_name = f"img_{uuid.uuid4().hex}{original_path.suffix}"
    new_path = temp_dir / random_name
    original_path.rename(new_path)

    data = new_path.read_bytes()
    width, height = extract_image_dimensions(data)
    filename = random_name
    return base64.b64encode(data).decode("ascii"), width, height, filename
