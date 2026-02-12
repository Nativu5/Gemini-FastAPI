import base64
import hashlib
import html
import mimetypes
import re
import reprlib
import struct
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import httpx
import orjson
from loguru import logger

from ..models import FunctionCall, Message, ToolCall

VALID_TAG_ROLES = {"user", "assistant", "system", "tool"}
TOOL_WRAP_HINT = (
    "\nWhen you decide to call tools, you MUST respond ONLY with a single [function_calls] block using this EXACT syntax:\n"
    "[function_calls]\n"
    "[call:tool_name]\n"
    "@args\n"
    "\n<<<ARG:arg_name>>>\n"
    "value\n"
    "<<<END:arg_name>>>\n"
    "[/call]\n"
    "[/function_calls]\n"
    "CRITICAL: Arguments MUST use <<<ARG:name>>>...<<<END:name>>> tags. Content inside tags can be any format.\n"
)
TOOL_BLOCK_RE = re.compile(
    r"\\?\[function_calls\\?]\s*(.*?)\s*\\?\[/function_calls\\?]", re.DOTALL | re.IGNORECASE
)
TOOL_CALL_RE = re.compile(
    r"\\?\[call:([^]\\]+)\\?]\s*(.*?)\s*\\?\[/call\\?]", re.DOTALL | re.IGNORECASE
)
RESPONSE_BLOCK_RE = re.compile(
    r"\\?\[function_responses\\?]\s*(.*?)\s*\\?\[/function_responses\\?]",
    re.DOTALL | re.IGNORECASE,
)
RESPONSE_ITEM_RE = re.compile(
    r"\\?\[response:([^]\\]+)\\?]\s*(.*?)\s*\\?\[/response\\?]", re.DOTALL | re.IGNORECASE
)
TAGGED_ARG_RE = re.compile(
    r"(?:\\?<){3}ARG:([^>\\]+)(?:\\?>){3}\s*(.*?)\s*(?:\\?<){3}END:\1(?:\\?>){3}",
    re.DOTALL | re.IGNORECASE,
)
TAGGED_RESULT_RE = re.compile(
    r"(?:\\?<){3}RESULT(?:\\?>){3}\s*(.*?)\s*(?:\\?<){3}END:RESULT(?:\\?>){3}",
    re.DOTALL | re.IGNORECASE,
)
CONTROL_TOKEN_RE = re.compile(r"\\?<\|im_(?:start|end)\|\\?>")
COMMONMARK_UNESCAPE_RE = re.compile(
    r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])"
)  # See: https://spec.commonmark.org/current/#backslash-escapes
TOOL_HINT_STRIPPED = TOOL_WRAP_HINT.strip()
_hint_lines = [line.strip() for line in TOOL_WRAP_HINT.split("\n") if line.strip()]
TOOL_HINT_LINE_START = _hint_lines[0] if _hint_lines else ""
TOOL_HINT_LINE_END = _hint_lines[-1] if _hint_lines else ""


def add_tag(role: str, content: str, unclose: bool = False) -> str:
    """Surround content with role tags"""
    if role not in VALID_TAG_ROLES:
        logger.warning(f"Unknown role: {role}, returning content without tags")
        return content

    return f"<|im_start|>{role}\n{content}" + ("\n<|im_end|>" if not unclose else "")


def normalize_llm_text(s: str) -> str:
    """
    Safely normalize LLM-generated text for both display and hashing.
    Includes: HTML unescaping, NFC normalization, and line ending standardization.
    """
    if not s:
        return ""

    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    return s


def unescape_llm_text(s: str) -> str:
    r"""Unescape characters escaped by Gemini Web's post-processing."""
    return COMMONMARK_UNESCAPE_RE.sub(r"\1", s)


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
    cleaned = text.replace(TOOL_WRAP_HINT, "").replace(TOOL_HINT_STRIPPED, "")

    # Remove fragments or multi-line blocks using derived constants
    if TOOL_HINT_LINE_START and TOOL_HINT_LINE_END:
        # Match from the start line to the end line, inclusive, handling internal modifications
        pattern = rf"\n?{re.escape(TOOL_HINT_LINE_START)}.*?{re.escape(TOOL_HINT_LINE_END)}\.?\n?"
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

    if TOOL_HINT_LINE_START:
        cleaned = re.sub(rf"\n?{re.escape(TOOL_HINT_LINE_START)}:?\s*", "", cleaned)
    if TOOL_HINT_LINE_END:
        cleaned = re.sub(rf"\s*{re.escape(TOOL_HINT_LINE_END)}\.?\n?", "", cleaned)

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

    cleaned = strip_system_hints(text)

    tool_calls: list[ToolCall] = []

    def _create_tool_call(name: str, raw_args: str) -> None:
        if not extract:
            return
        if not name:
            logger.warning("Encountered tool_call without a function name.")
            return

        name = unescape_llm_text(name.strip())
        raw_args = unescape_llm_text(raw_args)

        arg_matches = TAGGED_ARG_RE.findall(raw_args)
        if arg_matches:
            args_dict = {arg_name.strip(): arg_value.strip() for arg_name, arg_value in arg_matches}
            arguments = orjson.dumps(args_dict).decode("utf-8")
            logger.debug(f"Successfully parsed {len(args_dict)} tagged arguments for tool: {name}")
        else:
            cleaned_raw = raw_args.replace("@args", "").strip()
            if not cleaned_raw:
                logger.debug(f"Tool '{name}' called without arguments.")
            else:
                logger.warning(
                    f"Malformed arguments for tool '{name}'. Text found but no valid tags: {reprlib.repr(cleaned_raw)}"
                )
            arguments = "{}"

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

    all_calls = []
    for match in TOOL_CALL_RE.finditer(cleaned):
        all_calls.append(
            {
                "start": match.start(),
                "name": unescape_llm_text((match.group(1) or "").strip()),
                "args": (match.group(2) or "").strip(),
            }
        )

    all_calls.sort(key=lambda x: x["start"])

    if extract:
        for call in all_calls:
            _create_tool_call(call["name"], call["args"])

    cleaned = TOOL_BLOCK_RE.sub("", cleaned)
    cleaned = TOOL_CALL_RE.sub("", cleaned)
    cleaned = RESPONSE_BLOCK_RE.sub("", cleaned)
    cleaned = RESPONSE_ITEM_RE.sub("", cleaned)
    cleaned = TAGGED_ARG_RE.sub("", cleaned)
    cleaned = TAGGED_RESULT_RE.sub("", cleaned)

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


def detect_image_extension(data: bytes) -> str | None:
    """Detect image extension from magic bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    if data.startswith(b"GIF8"):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return None
