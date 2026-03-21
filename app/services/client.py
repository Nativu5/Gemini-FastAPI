import io
from pathlib import Path
from typing import Any

import orjson
from gemini_webapi import GeminiClient
from loguru import logger

from app.models import AppMessage
from app.utils import g_config
from app.utils.helper import (
    add_tag,
    save_file_to_tempfile,
    save_url_to_tempfile,
)


class GeminiClientWrapper(GeminiClient):
    """Gemini client with helper methods."""

    def __init__(self, client_id: str, **kwargs):
        super().__init__(**kwargs)
        self.id = client_id

    async def init(self, *args: Any, **kwargs: Any) -> None:
        """
        Inject default configuration values from global settings.
        """
        config = g_config.gemini
        try:
            await super().init(
                timeout=config.timeout,
                watchdog_timeout=config.watchdog_timeout,
                auto_refresh=config.auto_refresh,
                refresh_interval=config.refresh_interval,
                verbose=config.verbose,
            )
        except Exception:
            logger.exception(f"Failed to initialize GeminiClient {self.id}")
            raise

    def running(self) -> bool:
        return self._running

    @staticmethod
    async def process_message(
        message: AppMessage,
        tempdir: Path | None = None,
        tagged: bool = True,
        wrap_tool: bool = True,
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
                    item_text = getattr(item, "text", "") or ""
                    if item_text or message.role == "tool":
                        text_fragments.append(item_text)
                elif item.type == "image_url":
                    item_media_url = getattr(item, "url", None)
                    if not item_media_url:
                        raise ValueError(f"{item.type} cannot be empty")
                    files.append(await save_url_to_tempfile(item_media_url, tempdir))
                elif item.type == "file":
                    file_data = getattr(item, "file_data", None)
                    if file_data:
                        filename = getattr(item, "filename", "") or ""
                        files.append(await save_file_to_tempfile(file_data, filename, tempdir))
                    else:
                        raise ValueError("File must contain 'file_data'")
                elif item.type == "input_audio":
                    file_data = getattr(item, "file_data", None)
                    if file_data:
                        files.append(await save_file_to_tempfile(file_data, "audio.wav", tempdir))
                    else:
                        raise ValueError("input_audio must contain 'file_data' key")
        elif message.content is None and message.role == "tool":
            text_fragments.append("")
        elif message.content is not None:
            raise ValueError(f"Unsupported message content type: {type(message.content)}")

        if message.role == "tool":
            tool_name = message.name or "unknown"
            combined_content = "\n".join(text_fragments).strip()
            res_block = (
                f"[Result:{tool_name}]\n[ToolResult]\n{combined_content}\n[/ToolResult]\n[/Result]"
            )
            if wrap_tool:
                text_fragments = [f"[ToolResults]\n{res_block}\n[/ToolResults]"]
            else:
                text_fragments = [res_block]

        if message.tool_calls:
            tool_blocks: list[str] = []
            for call in message.tool_calls:
                params_text = call.function.arguments.strip()
                formatted_params = ""
                if params_text:
                    try:
                        parsed_params = orjson.loads(params_text)
                        if isinstance(parsed_params, dict):
                            for k, v in parsed_params.items():
                                val_str = (
                                    v if isinstance(v, str) else orjson.dumps(v).decode("utf-8")
                                )
                                formatted_params += (
                                    f"[CallParameter:{k}]\n```\n{val_str}\n```\n[/CallParameter]\n"
                                )
                        else:
                            formatted_params += f"```\n{params_text}\n```\n"
                    except orjson.JSONDecodeError:
                        formatted_params += f"```\n{params_text}\n```\n"

                tool_blocks.append(f"[Call:{call.function.name}]\n{formatted_params}[/Call]")

            if tool_blocks:
                tool_section = "[ToolCalls]\n" + "\n".join(tool_blocks) + "\n[/ToolCalls]"
                text_fragments.append(tool_section)

        model_input = "\n".join(fragment for fragment in text_fragments if fragment is not None)

        if (model_input or message.role == "tool") and tagged:
            model_input = add_tag(message.role, model_input)

        return model_input, files

    @staticmethod
    async def process_conversation(
        messages: list[AppMessage], tempdir: Path | None = None
    ) -> tuple[str, list[str | Path | bytes | io.BytesIO]]:
        conversation: list[str] = []
        files: list[str | Path | bytes | io.BytesIO] = []

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
                wrapped_content = f"[ToolResults]\n{combined_tool_content}\n[/ToolResults]"
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
