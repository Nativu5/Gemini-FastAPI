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
from urllib.parse import unquote, urlparse

import httpx
import orjson
from loguru import logger

from ..models import FunctionCall, Message, ToolCall

VALID_TAG_ROLES = {"user", "assistant", "system", "tool"}
TOOL_WRAP_HINT = (
    "\n\nSYSTEM INTERFACE: Tool calling protocol. You MUST follow these MANDATORY rules:\n\n"
    "1. Respond ONLY with a single [tool_calls] block. NO conversational text, NO explanations, NO filler.\n"
    "2. For ALL parameters, the value MUST be entirely enclosed in a single markdown code block (start/end with backticks) inside the tags. NO text allowed outside this block.\n"
    "3. Use a markdown fence longer than any backtick sequence in the value (e.g., use ```` if value has ```).\n\n"
    "EXACT SYNTAX TEMPLATE:\n"
    "[tool_calls]\n"
    "[call:tool_name]\n"
    "[call_parameter:parameter_name]\n"
    "```\n"
    "value\n"
    "```\n"
    "[/call_parameter]\n"
    "[/call]\n"
    "[/tool_calls]\n\n"
    "CRITICAL: Every tag MUST be opened and closed accurately.\n\n"
    "Multiple tools: List them sequentially inside one [tool_calls] block. No tool: respond naturally, NEVER use protocol tags.\n"
)
TOOL_BLOCK_RE = re.compile(r"\[tool_calls]\s*(.*?)\s*\[/tool_calls]", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(
    r"\[call:((?:[^]\\]|\\.)+)]\s*(.*?)\s*\[/call]", re.DOTALL | re.IGNORECASE
)
RESPONSE_BLOCK_RE = re.compile(
    r"\[tool_results]\s*(.*?)\s*\[/tool_results]",
    re.DOTALL | re.IGNORECASE,
)
RESPONSE_ITEM_RE = re.compile(
    r"\[result:((?:[^]\\]|\\.)+)]\s*(.*?)\s*\[/result]",
    re.DOTALL | re.IGNORECASE,
)
TAGGED_PARAM_RE = re.compile(
    r"\[call_parameter:((?:[^]\\]|\\.)+)]\s*(.*?)\s*\[/call_parameter]",
    re.DOTALL | re.IGNORECASE,
)
TAGGED_RESULT_RE = re.compile(
    r"\[tool_result]\s*(.*?)\s*\[/tool_result]",
    re.DOTALL | re.IGNORECASE,
)
CONTROL_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>", re.IGNORECASE)
CHATML_START_RE = re.compile(r"<\|im_start\|>\s*(\w+)\s*\n?", re.IGNORECASE)
CHATML_END_RE = re.compile(r"<\|im_end\|>", re.IGNORECASE)
FILE_PATH_PATTERN = re.compile(
    r"^(?=.*[./\\]|.*:\d+|^(?:Dockerfile|Makefile|Jenkinsfile|Procfile|Rakefile|Gemfile|Vagrantfile|Caddyfile|Justfile|LICENSE|README|CONTRIBUTING|CODEOWNERS|AUTHORS|NOTICE|CHANGELOG)$)([a-zA-Z0-9_./\\-]+(?::\d+)?)$",
    re.IGNORECASE,
)
GOOGLE_SEARCH_PATTERN = re.compile(
    r"(?P<md_start>`?\[`?)?"
    r"(?P<text>[^]]+)?"
    r"(?(md_start)`?]\()?"
    r"https://www\.google\.com/search\?q=(?P<query>[^&\s\"'<>)]+)"
    r"(?(md_start)\)?`?)",
    re.IGNORECASE,
)
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


def _strip_google_search(match: re.Match) -> str:
    """Extract raw text from Google Search links if it looks like a file path."""
    text_to_check = match.group("text") if match.group("text") else unquote(match.group("query"))
    text_to_check = unquote(text_to_check.strip())

    if FILE_PATH_PATTERN.match(text_to_check):
        return text_to_check
    return match.group(0)


def _strip_param_fences(s: str) -> str:
    """
    Remove one layer of outermost Markdown code fences,
    supporting nested blocks by detecting variable fence lengths.
    """
    s = s.strip()
    if not s:
        return ""

    match = re.match(r"^(?P<fence>`{3,})", s)
    if not match or not s.endswith(match.group("fence")):
        return s

    lines = s.splitlines()
    if len(lines) >= 2:
        return "\n".join(lines[1:-1])

    n = len(match.group("fence"))
    return s[n:-n].strip()


def _repair_param_value(s: str) -> str:
    """
    Standardize and repair LLM-generated parameter values
    to ensure compatibility with specialized clients like Roo Code.
    """
    if not s:
        return ""

    s = GOOGLE_SEARCH_PATTERN.sub(_strip_google_search, s)

    return s


def estimate_tokens(text: str | None) -> int:
    """Estimate the number of tokens heuristically based on character count."""
    if not text:
        return 0
    return int(len(text) / 3)


async def save_file_to_tempfile(
    file_in_base64: str, file_name: str = "", tempdir: Path | None = None
) -> Path:
    """Decode base64 file data and save to a temporary file."""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file_name).suffix if file_name else ".bin", dir=tempdir
    ) as tmp:
        tmp.write(base64.b64decode(file_in_base64))
        path = Path(tmp.name)
    return path


async def save_url_to_tempfile(url: str, tempdir: Path | None = None) -> Path:
    """Download content from a URL and save to a temporary file."""
    data: bytes | None = None
    suffix: str | None = None
    if url.startswith("data:image/"):
        metadata_part = url.split(",")[0]
        mime_type = metadata_part.split(":")[1].split(";")[0]
        data = base64.b64decode(url.split(",")[1])
        suffix = mimetypes.guess_extension(mime_type) or f".{mime_type.split('/')[1]}"
    else:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
            content_type = resp.headers.get("content-type")
            if content_type:
                suffix = mimetypes.guess_extension(content_type.split(";")[0].strip())
            if not suffix:
                suffix = Path(urlparse(url).path).suffix or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempdir) as tmp:
        tmp.write(data)
        path = Path(tmp.name)
    return path


def strip_tagged_blocks(text: str) -> str:
    """
    Remove ChatML role blocks (<|im_start|>role...<|im_end|>).
    Role 'tool' blocks are removed entirely; others have markers stripped but content preserved.
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
    cleaned = TAGGED_PARAM_RE.sub("", cleaned)
    cleaned = TAGGED_RESULT_RE.sub("", cleaned)

    return cleaned


def _process_tools_internal(text: str, extract: bool = True) -> tuple[str, list[ToolCall]]:
    """
    Extract tool metadata and return text stripped of technical markers.
    Parameters are parsed into JSON and assigned deterministic call IDs.
    """
    if not text:
        return text, []

    tool_calls: list[ToolCall] = []

    def _create_tool_call(name: str, raw_params: str) -> None:
        if not extract:
            return

        name = name.strip()
        if not name:
            logger.warning("Encountered tool_call without a function name.")
            return

        param_matches = TAGGED_PARAM_RE.findall(raw_params)
        if param_matches:
            params_dict = {
                param_name.strip(): _repair_param_value(_strip_param_fences(param_value))
                for param_name, param_value in param_matches
            }
            arguments = orjson.dumps(params_dict).decode("utf-8")
            logger.debug(f"Successfully parsed {len(params_dict)} parameters for tool: {name}")
        else:
            cleaned_raw = raw_params.strip()
            if not cleaned_raw:
                logger.debug(f"Successfully parsed 0 parameters for tool: {name}")
            else:
                logger.warning(
                    f"Malformed parameters for tool '{name}'. Text found but no valid tags: {reprlib.repr(cleaned_raw)}"
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
    """Concatenate text and tool parameters from a message for token estimation."""
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
        tool_param_text = "".join(call.function.arguments or "" for call in message.tool_calls)
        base_text = f"{base_text}\n{tool_param_text}" if base_text else tool_param_text

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
        sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
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
