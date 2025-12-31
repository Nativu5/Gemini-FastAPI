import base64
import json
import mimetypes
import re
import struct
import tempfile
import uuid
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import httpx
from loguru import logger

from ..models import FunctionCall, Message, ToolCall

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
}
VALID_TAG_ROLES = {"user", "assistant", "system", "tool"}
XML_WRAP_HINT = (
    "\nYou MUST wrap every tool call response inside a single fenced block exactly like:\n"
    '```xml\n<tool_call name="tool_name">{"arg": "value"}</tool_call>\n```\n'
    "Do not surround the fence with any other text or whitespace; otherwise the call will be ignored.\n"
)
CODE_BLOCK_HINT = (
    "\nWhenever you include code, markup, or shell snippets, wrap each snippet in a Markdown fenced "
    "block and supply the correct language label (for example, ```python ... ``` or ```html ... ```).\n"
    "Fence ONLY the actual code/markup; keep all narrative or explanatory text outside the fences.\n"
)
TOOL_BLOCK_RE = re.compile(r"```xml\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(
    r"<tool_call\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE
)
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
CONTROL_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>")
XML_HINT_STRIPPED = XML_WRAP_HINT.strip()
CODE_HINT_STRIPPED = CODE_BLOCK_HINT.strip()


def add_tag(role: str, content: str, unclose: bool = False) -> str:
    """Surround content with role tags"""
    if role not in VALID_TAG_ROLES:
        logger.warning(f"Unknown role: {role}, returning content without tags")
        return content

    return f"<|im_start|>{role}\n{content}" + ("\n<|im_end|>" if not unclose else "")


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens heuristically based on character count"""
    return int(len(text) / 3)


async def save_file_to_tempfile(
    file_in_base64: str, file_name: str = "", tempdir: Path | None = None
) -> Path:
    data = base64.b64decode(file_in_base64)
    suffix = Path(file_name).suffix if file_name else ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempdir) as tmp:
        tmp.write(data)
        path = Path(tmp.name)

    return path


async def save_url_to_tempfile(url: str, tempdir: Path | None = None) -> Path:
    data: bytes | None = None
    suffix: str | None = None
    if url.startswith("data:image/"):
        # Base64 encoded image
        metadata_part = url.split(",")[0]
        mime_type = metadata_part.split(":")[1].split(";")[0]

        base64_data = url.split(",")[1]
        data = base64.b64decode(base64_data)

        suffix = mimetypes.guess_extension(mime_type)
        if not suffix:
            suffix = f".{mime_type.split('/')[1]}"
    else:
        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
            content_type = resp.headers.get("content-type")

            if content_type:
                mime_type = content_type.split(";")[0].strip()
                suffix = mimetypes.guess_extension(mime_type)

            if not suffix:
                path_url = urlparse(url).path
                suffix = Path(path_url).suffix

            if not suffix:
                suffix = ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempdir) as tmp:
        tmp.write(data)
        path = Path(tmp.name)

    return path


def strip_code_fence(text: str) -> str:
    """Remove surrounding ```json fences if present."""
    match = JSON_FENCE_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def strip_tagged_blocks(text: str) -> str:
    """Remove <|im_start|>role ... <|im_end|> sections, dropping tool blocks entirely.
    - tool blocks are removed entirely (if missing end marker, drop to EOF).
    - other roles: remove markers and role, keep inner content (if missing end marker, keep to EOF).
    """
    if not text:
        return text

    result: list[str] = []
    idx = 0
    length = len(text)
    start_marker = "<|im_start|>"
    end_marker = "<|im_end|>"

    while idx < length:
        start = text.find(start_marker, idx)
        if start == -1:
            result.append(text[idx:])
            break

        # append any content before this block
        result.append(text[idx:start])

        role_start = start + len(start_marker)
        newline = text.find("\n", role_start)
        if newline == -1:
            # malformed block; keep remainder as-is (safe behavior)
            result.append(text[start:])
            break

        role = text[role_start:newline].strip().lower()

        end = text.find(end_marker, newline + 1)
        if end == -1:
            # missing end marker
            if role == "tool":
                # drop from start marker to EOF (skip remainder)
                break
            else:
                # keep inner content from after the role newline to EOF
                result.append(text[newline + 1 :])
                break

        block_end = end + len(end_marker)

        if role == "tool":
            # drop whole block
            idx = block_end
            continue

        # keep the content without role markers
        content = text[newline + 1 : end]
        result.append(content)
        idx = block_end

    return "".join(result)


def strip_system_hints(text: str) -> str:
    """Remove system-level hint text from a given string."""
    if not text:
        return text
    cleaned = strip_tagged_blocks(text)
    cleaned = cleaned.replace(XML_WRAP_HINT, "").replace(XML_HINT_STRIPPED, "")
    cleaned = cleaned.replace(CODE_BLOCK_HINT, "").replace(CODE_HINT_STRIPPED, "")
    cleaned = CONTROL_TOKEN_RE.sub("", cleaned)
    return cleaned.strip()


def remove_tool_call_blocks(text: str) -> str:
    """Strip tool call code blocks from text."""
    if not text:
        return text

    # 1. Remove fenced blocks ONLY if they contain tool calls
    def _replace_block(match: re.Match[str]) -> str:
        block_content = match.group(1)
        if not block_content:
            return match.group(0)

        # Check if the block contains any tool call tag
        if TOOL_CALL_RE.search(block_content):
            return ""

        # Preserve the block if no tool call found
        return match.group(0)

    cleaned = TOOL_BLOCK_RE.sub(_replace_block, text)

    # 2. Remove orphaned tool calls
    cleaned = TOOL_CALL_RE.sub("", cleaned)

    return strip_system_hints(cleaned)


def extract_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Extract tool call definitions and return cleaned text."""
    if not text:
        return text, []

    tool_calls: list[ToolCall] = []

    def _create_tool_call(name: str, raw_args: str) -> None:
        """Helper to parse args and append to tool_calls list."""
        if not name:
            logger.warning("Encountered tool_call without a function name.")
            return

        arguments = raw_args
        try:
            parsed_args = json.loads(raw_args)
            arguments = json.dumps(parsed_args, ensure_ascii=False)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call arguments for '{name}'. Passing raw string.")

        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex}",
                type="function",
                function=FunctionCall(name=name, arguments=arguments),
            )
        )

    def _replace_block(match: re.Match[str]) -> str:
        block_content = match.group(1)
        if not block_content:
            return match.group(0)

        found_in_block = False
        for call_match in TOOL_CALL_RE.finditer(block_content):
            found_in_block = True
            name = (call_match.group(1) or "").strip()
            raw_args = (call_match.group(2) or "").strip()
            _create_tool_call(name, raw_args)

        if found_in_block:
            return ""
        else:
            return match.group(0)

    cleaned = TOOL_BLOCK_RE.sub(_replace_block, text)

    def _replace_orphan(match: re.Match[str]) -> str:
        name = (match.group(1) or "").strip()
        raw_args = (match.group(2) or "").strip()
        _create_tool_call(name, raw_args)
        return ""

    cleaned = TOOL_CALL_RE.sub(_replace_orphan, cleaned)

    cleaned = strip_system_hints(cleaned)
    return cleaned, tool_calls


def iter_stream_segments(model_output: str, chunk_size: int = 64) -> Iterator[str]:
    """Yield stream segments while keeping <think> markers and words intact."""
    if not model_output:
        return

    token_pattern = re.compile(r"\s+|\S+\s*")
    pending = ""

    def _flush_pending() -> Iterator[str]:
        nonlocal pending
        if pending:
            yield pending
            pending = ""

    # Split on <think> boundaries so the markers are never fragmented.
    parts = re.split(r"(</?think>)", model_output)
    for part in parts:
        if not part:
            continue
        if part in {"<think>", "</think>"}:
            yield from _flush_pending()
            yield part
            continue

        for match in token_pattern.finditer(part):
            token = match.group(0)

            if len(token) > chunk_size:
                yield from _flush_pending()
                for idx in range(0, len(token), chunk_size):
                    yield token[idx : idx + chunk_size]
                continue

            if pending and len(pending) + len(token) > chunk_size:
                yield from _flush_pending()

            pending += token

    yield from _flush_pending()


def text_from_message(message: Message) -> str:
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


def extract_image_dimensions(data: bytes) -> tuple[int | None, int | None]:
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
