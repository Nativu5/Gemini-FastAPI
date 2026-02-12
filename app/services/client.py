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
    normalize_llm_text,
    save_file_to_tempfile,
    save_url_to_tempfile,
)

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
        watchdog_timeout: float = cast(float, _UNSET),
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
        watchdog_timeout = cast(float, _resolve(watchdog_timeout, config.watchdog_timeout))
        auto_refresh = cast(bool, _resolve(auto_refresh, config.auto_refresh))
        refresh_interval = cast(float, _resolve(refresh_interval, config.refresh_interval))
        verbose = cast(bool, _resolve(verbose, config.verbose))

        try:
            await super().init(
                timeout=timeout,
                watchdog_timeout=watchdog_timeout,
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
        message: Message, tempdir: Path | None = None, tagged: bool = True, wrap_tool: bool = True
    ) -> tuple[str, list[Path | str]]:
        """
        Process a Message into Gemini API format using the PascalCase technical protocol.
        Extracts text, handles files, and appends ToolCalls/ToolResults blocks.
        """
        files: list[Path | str] = []
        text_fragments: list[str] = []

        if isinstance(message.content, str):
            if message.content or message.role == "tool":
                text_fragments.append(message.content or "")
        elif isinstance(message.content, list):
            for item in message.content:
                if item.type == "text":
                    if item.text or message.role == "tool":
                        text_fragments.append(item.text or "")
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
            text_fragments.append("")
        elif message.content is not None:
            raise ValueError("Unsupported message content type.")

        if message.role == "tool":
            tool_name = message.name or "unknown"
            combined_content = "\n".join(text_fragments).strip()
            res_block = (
                f"[Result:{tool_name}]\n"
                f"@results\n"
                f"<<<ToolResult>>>\n{combined_content}\n<<<EndToolResult>>>\n"
                f"[/Result]"
            )
            if wrap_tool:
                text_fragments = [f"[ToolResults]\n{res_block}\n[/ToolResults]"]
            else:
                text_fragments = [res_block]

        if message.tool_calls:
            tool_blocks: list[str] = []
            for call in message.tool_calls:
                args_text = call.function.arguments.strip()
                formatted_args = "@args\n"
                try:
                    parsed_args = orjson.loads(args_text)
                    if isinstance(parsed_args, dict):
                        for k, v in parsed_args.items():
                            val_str = v if isinstance(v, str) else orjson.dumps(v).decode("utf-8")
                            formatted_args += (
                                f"<<<CallParameter:{k}>>>\n{val_str}\n<<<EndCallParameter>>>\n"
                            )
                    else:
                        formatted_args += args_text
                except orjson.JSONDecodeError:
                    formatted_args += args_text

                tool_blocks.append(f"[Call:{call.function.name}]\n{formatted_args}[/Call]")

            if tool_blocks:
                tool_section = "[ToolCalls]\n" + "\n".join(tool_blocks) + "\n[/ToolCalls]"
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
        conversation: list[str] = []
        files: list[Path | str] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == "tool":
                # Group consecutive tool messages
                tool_blocks: list[str] = []
                while i < len(messages) and messages[i].role == "tool":
                    part, part_files = await GeminiClientWrapper.process_message(
                        messages[i], tempdir, tagged=False, wrap_tool=False
                    )
                    tool_blocks.append(part)
                    files.extend(part_files)
                    i += 1

                combined_tool_content = "\n".join(tool_blocks)
                wrapped_content = (
                    f"[function_responses]\n{combined_tool_content}\n[/function_responses]"
                )
                conversation.append(add_tag("tool", wrapped_content))
            else:
                input_part, files_part = await GeminiClientWrapper.process_message(
                    msg, tempdir, tagged=True
                )
                conversation.append(input_part)
                files.extend(files_part)
                i += 1

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

        text = normalize_llm_text(text)

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
