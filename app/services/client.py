import html
import re
from pathlib import Path
from typing import Any, cast

import orjson
from gemini_webapi import GeminiClient, ModelOutput
from loguru import logger

from ..models import Message
from ..utils import g_config
from ..utils.helper import (
    add_tag,
    save_file_to_tempfile,
    save_url_to_tempfile,
)

HTML_ESCAPE_RE = re.compile(r"&(?:lt|gt|amp|quot|apos|#[0-9]+|#x[0-9a-fA-F]+);")
MARKDOWN_ESCAPE_RE = re.compile(r"\\(?=[-\\`*_{}\[\]()#+.!<>])")
CODE_FENCE_RE = re.compile(r"(```.*?```|`[^`\n]+?`)", re.DOTALL)
FILE_PATH_PATTERN = re.compile(
    r"^(?=.*[./\\]|.*:\d+|^(?:Dockerfile|Makefile|Jenkinsfile|Procfile|Rakefile|Gemfile|Vagrantfile|Caddyfile|Justfile|LICENSE|README|CONTRIBUTING|CODEOWNERS|AUTHORS|NOTICE|CHANGELOG)$)([a-zA-Z0-9_./\\-]+(?::\d+)?)$",
    re.IGNORECASE,
)
GOOGLE_SEARCH_LINK_PATTERN = re.compile(
    r"`?\[`?(.+?)`?`?]\((https://www\.google\.com/search\?q=)([^)]*)\)`?"
)
_UNSET = object()


def _resolve(value: Any, fallback: Any):
    return fallback if value is _UNSET else value


class GeminiClientWrapper(GeminiClient):
    """Gemini client with helper methods."""

    def __init__(self, client_id: str, **kwargs):
        super().__init__(**kwargs)
        self.id = client_id

    async def init(
        self,
        timeout: float = cast(float, _UNSET),
        auto_close: bool = False,
        close_delay: float = 300,
        auto_refresh: bool = cast(bool, _UNSET),
        refresh_interval: float = cast(float, _UNSET),
        verbose: bool = cast(bool, _UNSET),
    ) -> None:
        """
        Inject default configuration values.
        """
        config = g_config.gemini
        timeout = cast(float, _resolve(timeout, config.timeout))
        auto_refresh = cast(bool, _resolve(auto_refresh, config.auto_refresh))
        refresh_interval = cast(float, _resolve(refresh_interval, config.refresh_interval))
        verbose = cast(bool, _resolve(verbose, config.verbose))

        try:
            await super().init(
                timeout=timeout,
                auto_close=auto_close,
                close_delay=close_delay,
                auto_refresh=auto_refresh,
                refresh_interval=refresh_interval,
                verbose=verbose,
            )
        except Exception:
            logger.exception(f"Failed to initialize GeminiClient {self.id}")
            raise

    def running(self) -> bool:
        return self._running

    @staticmethod
    async def process_message(
        message: Message, tempdir: Path | None = None, tagged: bool = True
    ) -> tuple[str, list[Path | str]]:
        """
        Process a single Message object into a format suitable for the Gemini API.
        Extracts text fragments, handles images and files, and appends tool call blocks if present.
        """
        files: list[Path | str] = []
        text_fragments: list[str] = []

        if isinstance(message.content, str):
            if message.content or message.role == "tool":
                text_fragments.append(message.content or "{}")
        elif isinstance(message.content, list):
            for item in message.content:
                if item.type == "text":
                    if item.text or message.role == "tool":
                        text_fragments.append(item.text or "{}")
                elif item.type == "image_url":
                    if not item.image_url:
                        raise ValueError("Image URL cannot be empty")
                    if url := item.image_url.get("url", None):
                        files.append(await save_url_to_tempfile(url, tempdir))
                    else:
                        raise ValueError("Image URL must contain 'url' key")
                elif item.type == "file":
                    if not item.file:
                        raise ValueError("File cannot be empty")
                    if file_data := item.file.get("file_data", None):
                        filename = item.file.get("filename", "")
                        files.append(await save_file_to_tempfile(file_data, filename, tempdir))
                    elif url := item.file.get("url", None):
                        files.append(await save_url_to_tempfile(url, tempdir))
                    else:
                        raise ValueError("File must contain 'file_data' or 'url' key")
        elif message.content is None and message.role == "tool":
            text_fragments.append("{}")
        elif message.content is not None:
            raise ValueError("Unsupported message content type.")

        if message.role == "tool":
            tool_name = message.name or "unknown"
            combined_content = "\n".join(text_fragments).strip() or "{}"
            text_fragments = [
                f'<tool_response name="{tool_name}">{combined_content}</tool_response>'
            ]

        if message.tool_calls:
            tool_blocks: list[str] = []
            for call in message.tool_calls:
                args_text = call.function.arguments.strip()
                try:
                    parsed_args = orjson.loads(args_text)
                    args_text = orjson.dumps(parsed_args).decode("utf-8")
                except orjson.JSONDecodeError:
                    pass
                tool_blocks.append(
                    f'<tool_call name="{call.function.name}">{args_text}</tool_call>'
                )

            if tool_blocks:
                tool_section = "```xml\n" + "".join(tool_blocks) + "\n```"
                text_fragments.append(tool_section)

        model_input = "\n".join(fragment for fragment in text_fragments if fragment is not None)

        if model_input or message.role == "tool":
            if tagged:
                model_input = add_tag(message.role, model_input)

        return model_input, files

    @staticmethod
    async def process_conversation(
        messages: list[Message], tempdir: Path | None = None
    ) -> tuple[str, list[Path | str]]:
        need_tag = any(m.role != "user" for m in messages)
        conversation: list[str] = []
        files: list[Path | str] = []
        for msg in messages:
            input_part, files_part = await GeminiClientWrapper.process_message(
                msg, tempdir, tagged=need_tag
            )
            conversation.append(input_part)
            files.extend(files_part)
        if need_tag:
            conversation.append(add_tag("assistant", "", unclose=True))
        return "\n".join(conversation), files

    @staticmethod
    def extract_output(response: ModelOutput, include_thoughts: bool = True) -> str:
        text = ""
        if include_thoughts and response.thoughts:
            text += f"<think>{response.thoughts}</think>\n"
        if response.text:
            text += response.text
        else:
            text += str(response)

        def _unescape_html(text_content: str) -> str:
            parts: list[str] = []
            last_index = 0
            for match in CODE_FENCE_RE.finditer(text_content):
                non_code = text_content[last_index : match.start()]
                if non_code:
                    parts.append(HTML_ESCAPE_RE.sub(lambda m: html.unescape(m.group(0)), non_code))
                parts.append(match.group(0))
                last_index = match.end()
            tail = text_content[last_index:]
            if tail:
                parts.append(HTML_ESCAPE_RE.sub(lambda m: html.unescape(m.group(0)), tail))
            return "".join(parts)

        def _unescape_markdown(text_content: str) -> str:
            parts: list[str] = []
            last_index = 0
            for match in CODE_FENCE_RE.finditer(text_content):
                non_code = text_content[last_index : match.start()]
                if non_code:
                    parts.append(MARKDOWN_ESCAPE_RE.sub("", non_code))
                parts.append(match.group(0))
                last_index = match.end()
            tail = text_content[last_index:]
            if tail:
                parts.append(MARKDOWN_ESCAPE_RE.sub("", tail))
            return "".join(parts)

        text = _unescape_html(text)
        text = _unescape_markdown(text)

        def extract_file_path_from_display_text(text_content: str) -> str | None:
            match = re.match(FILE_PATH_PATTERN, text_content)
            if match:
                return match.group(1)
            return None

        def replacer(match: re.Match) -> str:
            display_text = str(match.group(1)).strip()
            google_search_prefix = match.group(2)
            query_part = match.group(3)

            file_path = extract_file_path_from_display_text(display_text)

            if file_path:
                # If it's a file path, transform it into a self-referencing Markdown link
                return f"[`{file_path}`]({file_path})"
            else:
                # Otherwise, reconstruct the original Google search link with the display_text
                original_google_search_url = f"{google_search_prefix}{query_part}"
                return f"[`{display_text}`]({original_google_search_url})"

        return re.sub(GOOGLE_SEARCH_LINK_PATTERN, replacer, text)
