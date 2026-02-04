import base64
import hashlib
import mimetypes
import re
import struct
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
import orjson
from loguru import logger

from ..models import FunctionCall, Message, ToolCall

VALID_TAG_ROLES = {"user", "assistant", "system", "tool"}
XML_WRAP_HINT = (
    "\nYou MUST wrap every tool call response inside a single fenced block exactly like:\n"
    '```xml\n<tool_call name="tool_name">{"argument": "value"}</tool_call>\n```\n'
    "Do not surround the fence with any other text or whitespace; otherwise the call will be ignored.\n"
)
TOOL_BLOCK_RE = re.compile(r"```xml\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(
    r"<tool_call\s+name=\"([^\"]+)\"\s*>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE
)
CONTROL_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>")
XML_HINT_STRIPPED = XML_WRAP_HINT.strip()
_hint_lines = [line.strip() for line in XML_WRAP_HINT.split("\n") if line.strip()]
XML_HINT_LINE_START = _hint_lines[0] if _hint_lines else ""
XML_HINT_LINE_END = _hint_lines[-1] if _hint_lines else ""


def add_tag(role: str, content: str, unclose: bool = False) -> str:
    """Surround content with role tags"""
    if role not in VALID_TAG_ROLES:
        logger.warning(f"Unknown role: {role}, returning content without tags")
        return content

    return f"<|im_start|>{role}\n{content}" + ("\n<|im_end|>" if not unclose else "")


def estimate_tokens(text: str | None) -> int:
    """Estimate the number of tokens heuristically based on character count"""
    if not text:
        return 0
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
        metadata_part = url.split(",")[0]
        mime_type = metadata_part.split(":")[1].split(";")[0]

        base64_data = url.split(",")[1]
        data = base64.b64decode(base64_data)

        suffix = mimetypes.guess_extension(mime_type)
        if not suffix:
            suffix = f".{mime_type.split('/')[1]}"
    else:
        async with httpx.AsyncClient(follow_redirects=True) as client:
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


def strip_tagged_blocks(text: str) -> str:
    """Remove <|im_start|>role ... <|im_end|> sections.
    - tool blocks are removed entirely (including content).
    - other roles: remove markers and role, keep inner content.
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

        result.append(text[idx:start])

        role_start = start + len(start_marker)
        newline = text.find("\n", role_start)
        if newline == -1:
            result.append(text[start:])
            break

        role = text[role_start:newline].strip().lower()

        end = text.find(end_marker, newline + 1)
        if end == -1:
            if role == "tool":
                break
            else:
                result.append(text[newline + 1 :])
                break

        block_end = end + len(end_marker)

        if role == "tool":
            idx = block_end
            continue

        content = text[newline + 1 : end]
        result.append(content)
        idx = block_end

    return "".join(result)


def strip_system_hints(text: str) -> str:
    """Remove system-level hint text from a given string."""
    if not text:
        return text

    # Remove the full hints first
    cleaned = text.replace(XML_WRAP_HINT, "").replace(XML_HINT_STRIPPED, "")

    # Remove fragments using derived constants
    if XML_HINT_LINE_START:
        cleaned = re.sub(rf"\n?{re.escape(XML_HINT_LINE_START)}:?\s*", "", cleaned)
    if XML_HINT_LINE_END:
        cleaned = re.sub(rf"\s*{re.escape(XML_HINT_LINE_END)}\.?\n?", "", cleaned)

    cleaned = strip_tagged_blocks(cleaned)
    cleaned = CONTROL_TOKEN_RE.sub("", cleaned)
    return cleaned


def _process_tools_internal(text: str, extract: bool = True) -> tuple[str, list[ToolCall]]:
    """
    Unified engine for stripping tool call blocks and extracting tool metadata.
    If extract=True, parses JSON arguments and assigns deterministic call IDs.
    """
    if not text:
        return text, []

    tool_calls: list[ToolCall] = []

    def _create_tool_call(name: str, raw_args: str) -> None:
        if not extract:
            return
        if not name:
            logger.warning("Encountered tool_call without a function name.")
            return

        arguments = raw_args
        try:
            parsed_args = orjson.loads(raw_args)
            arguments = orjson.dumps(parsed_args, option=orjson.OPT_SORT_KEYS).decode("utf-8")
        except orjson.JSONDecodeError:
            logger.warning(f"Failed to parse tool call arguments for '{name}'. Passing raw string.")

        index = len(tool_calls)
        seed = f"{name}:{arguments}:{index}".encode("utf-8")
        call_id = f"call_{hashlib.sha256(seed).hexdigest()[:24]}"

        tool_calls.append(
            ToolCall(
                id=call_id,
                type="function",
                function=FunctionCall(name=name, arguments=arguments),
            )
        )

    def _replace_block(match: re.Match[str]) -> str:
        block_content = match.group(1)
        if not block_content:
            return match.group(0)

        is_tool_block = bool(TOOL_CALL_RE.search(block_content))

        if is_tool_block:
            if extract:
                for call_match in TOOL_CALL_RE.finditer(block_content):
                    name = (call_match.group(1) or "").strip()
                    raw_args = (call_match.group(2) or "").strip()
                    _create_tool_call(name, raw_args)
            return ""
        else:
            return match.group(0)

    cleaned = TOOL_BLOCK_RE.sub(_replace_block, text)

    def _replace_orphan(match: re.Match[str]) -> str:
        if extract:
            name = (match.group(1) or "").strip()
            raw_args = (match.group(2) or "").strip()
            _create_tool_call(name, raw_args)
        return ""

    cleaned = TOOL_CALL_RE.sub(_replace_orphan, cleaned)
    cleaned = strip_system_hints(cleaned)
    return cleaned, tool_calls


def remove_tool_call_blocks(text: str) -> str:
    """Strip tool call code blocks from text."""
    cleaned, _ = _process_tools_internal(text, extract=False)
    return cleaned


def extract_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Extract tool call definitions and return cleaned text."""
    return _process_tools_internal(text, extract=True)


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
