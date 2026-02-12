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
    "\nWhen you decide to call tools, you MUST respond ONLY with a single [ToolCalls] block using this EXACT syntax:\n"
    "[ToolCalls]\n"
    "[Call:tool_name]\n"
    "@args\n"
    "<<<CallParameter:arg_name>>>\n"
    "value\n"
    "<<<EndCallParameter>>>\n"
    "[/Call]\n"
    "[/ToolCalls]\n"
    "CRITICAL: Every argument MUST be enclosed in <<<CallParameter:arg_name>>>...<<<EndCallParameter>>>. Output as RAW text. Content inside tags can be any format.\n"
)
TOOL_BLOCK_RE = re.compile(
    r"\\?\[ToolCalls\\?]\s*(.*?)\s*\\?\[/ToolCalls\\?]", re.DOTALL | re.IGNORECASE
)
TOOL_CALL_RE = re.compile(
    r"\\?\[Call\\?:((?:[^]\\]|\\.)+)\\?]\s*(.*?)\s*\\?\[/Call\\?]", re.DOTALL | re.IGNORECASE
)
RESPONSE_BLOCK_RE = re.compile(
    r"\\?\[ToolResults\\?]\s*(.*?)\s*\\?\[/ToolResults\\?]",
    re.DOTALL | re.IGNORECASE,
)
RESPONSE_ITEM_RE = re.compile(
    r"\\?\[Result\\?:((?:[^]\\]|\\.)+)\\?]\s*(.*?)\s*\\?\[/Result\\?]",
    re.DOTALL | re.IGNORECASE,
)
TAGGED_ARG_RE = re.compile(
    r"(?:\\?<){3}CallParameter\\?:((?:[^>\\]|\\.)+)(?:\\?>){3}\s*(.*?)\s*(?:\\?<){3}EndCallParameter(?:\\?>){3}",
    re.DOTALL | re.IGNORECASE,
)
TAGGED_RESULT_RE = re.compile(
    r"(?:\\?<){3}ToolResult(?:\\?>){3}\s*(.*?)\s*(?:\\?<){3}EndToolResult(?:\\?>){3}",
    re.DOTALL | re.IGNORECASE,
)
CONTROL_TOKEN_RE = re.compile(r"\\?<\|im\\?_(?:start|end)\|\\?>", re.IGNORECASE)
CHATML_START_RE = re.compile(r"\\?<\|im\\?_start\|\\?>\s*(\w+)\s*\n?", re.IGNORECASE)
CHATML_END_RE = re.compile(r"\\?<\|im\\?_end\|\\?>", re.IGNORECASE)
COMMONMARK_UNESCAPE_RE = re.compile(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")
TOOL_HINT_STRIPPED = TOOL_WRAP_HINT.strip()
_hint_lines = [line.strip() for line in TOOL_WRAP_HINT.split("\n") if line.strip()]
TOOL_HINT_LINE_START = _hint_lines[0] if _hint_lines else ""
TOOL_HINT_LINE_END = _hint_lines[-1] if _hint_lines else ""


def add_tag(role: str, content: str, unclose: bool = False) -> str:
    """Surround content with ChatML role tags."""
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
    """Unescape characters escaped by Gemini Web's post-processing (e.g., \\_ to _)."""
    return COMMONMARK_UNESCAPE_RE.sub(r"\1", s)


def estimate_tokens(text: str | None) -> int:
    """Estimate the number of tokens heuristically based on character count."""
    if not text:
        return 0
    return int(len(text) / 3)


async def save_file_to_tempfile(
    file_in_base64: str, file_name: str = "", tempdir: Path | None = None
) -> Path:
    """Decode base64 file data and save to a temporary file."""
    data = base64.b64decode(file_in_base64)
    suffix = Path(file_name).suffix if file_name else ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempdir) as tmp:
        tmp.write(data)
        path = Path(tmp.name)

    return path


async def save_url_to_tempfile(url: str, tempdir: Path | None = None) -> Path:
    """Download content from a URL and save to a temporary file."""
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
    """
    Remove ChatML role blocks (<|im_start|>role...<|im_end|>).
    Role 'tool' blocks are removed entirely; others have markers stripped but content preserved.
    Handles both raw and escaped markers consistently.
    """
    if not text:
        return text

    result = []
    idx = 0
    while idx < len(text):
        match_start = CHATML_START_RE.search(text, idx)
        if not match_start:
            result.append(text[idx:])
            break

        result.append(text[idx : match_start.start()])
        role = match_start.group(1).lower()
        content_start = match_start.end()

        match_end = CHATML_END_RE.search(text, content_start)
        if not match_end:
            if role != "tool":
                result.append(text[content_start:])
            break

        if role != "tool":
            result.append(text[content_start : match_end.start()])

        idx = match_end.end()

    return "".join(result)


def strip_system_hints(text: str) -> str:
    """Remove system hints, ChatML tags, and technical protocol markers from text."""
    if not text:
        return text

    cleaned = text.replace(TOOL_WRAP_HINT, "").replace(TOOL_HINT_STRIPPED, "")

    if TOOL_HINT_LINE_START and TOOL_HINT_LINE_END:
        pattern = rf"\n?{re.escape(TOOL_HINT_LINE_START)}.*?{re.escape(TOOL_HINT_LINE_END)}\.?\n?"
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

    if TOOL_HINT_LINE_START:
        cleaned = re.sub(rf"\n?{re.escape(TOOL_HINT_LINE_START)}:?\s*", "", cleaned)
    if TOOL_HINT_LINE_END:
        cleaned = re.sub(rf"\s*{re.escape(TOOL_HINT_LINE_END)}\.?\n?", "", cleaned)

    cleaned = strip_tagged_blocks(cleaned)

    cleaned = CONTROL_TOKEN_RE.sub("", cleaned)
    cleaned = TOOL_BLOCK_RE.sub("", cleaned)
    cleaned = TOOL_CALL_RE.sub("", cleaned)
    cleaned = RESPONSE_BLOCK_RE.sub("", cleaned)
    cleaned = RESPONSE_ITEM_RE.sub("", cleaned)
    cleaned = TAGGED_ARG_RE.sub("", cleaned)
    cleaned = TAGGED_RESULT_RE.sub("", cleaned)

    return cleaned


def _process_tools_internal(text: str, extract: bool = True) -> tuple[str, list[ToolCall]]:
    """
    Extract tool metadata and return text stripped of technical markers.
    Arguments are parsed into JSON and assigned deterministic call IDs.
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

    for match in TOOL_CALL_RE.finditer(text):
        _create_tool_call(match.group(1), match.group(2))

    cleaned = strip_system_hints(text)

    return cleaned, tool_calls


def remove_tool_call_blocks(text: str) -> str:
    """Strip tool call blocks from text for display."""
    cleaned, _ = _process_tools_internal(text, extract=False)
    return cleaned


def extract_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Extract tool calls and return cleaned text."""
    return _process_tools_internal(text, extract=True)


def text_from_message(message: Message) -> str:
    """Concatenate text and tool arguments from a message for token estimation."""
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
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        try:
            width, height = struct.unpack(">II", data[16:24])
            return int(width), int(height)
        except struct.error:
            return None, None

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
