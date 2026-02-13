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
        Process a Message into Gemini API format using the snake_case technical protocol.
        Extracts text, handles files, and appends tool_calls/tool_results blocks.
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
            res_block = f"[result:{tool_name}]\n[tool_result]\n{combined_content}\n[/tool_result]\n[/result]"
            if wrap_tool:
                text_fragments = [f"[tool_results]\n{res_block}\n[/tool_results]"]
            else:
                text_fragments = [res_block]

        if message.tool_calls:
            tool_blocks: list[str] = []
            for call in message.tool_calls:
                params_text = call.function.arguments.strip()
                formatted_params = ""
                try:
                    parsed_params = orjson.loads(params_text)
                    if isinstance(parsed_params, dict):
                        for k, v in parsed_params.items():
                            val_str = v if isinstance(v, str) else orjson.dumps(v).decode("utf-8")
                            formatted_params += f"[call_parameter:{k}]{val_str}[/call_parameter]\n"
                    else:
                        formatted_params += params_text
                except orjson.JSONDecodeError:
                    formatted_params += params_text

                tool_blocks.append(f"[call:{call.function.name}]\n{formatted_params}[/call]")

            if tool_blocks:
                tool_section = "[tool_calls]\n" + "\n".join(tool_blocks) + "\n[/tool_calls]"
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
                tool_blocks: list[str] = []
                while i < len(messages) and messages[i].role == "tool":
                    part, part_files = await GeminiClientWrapper.process_message(
                        messages[i], tempdir, tagged=False, wrap_tool=False
                    )
                    tool_blocks.append(part)
                    files.extend(part_files)
                    i += 1

                combined_tool_content = "\n".join(tool_blocks)
                wrapped_content = f"[tool_results]\n{combined_tool_content}\n[/tool_results]"
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

        return normalize_llm_text(text)
