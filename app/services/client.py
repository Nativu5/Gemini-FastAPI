import asyncio
import json
import re
from pathlib import Path
from typing import Any, cast

from gemini_webapi import GeminiClient, ModelOutput
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from gemini_webapi.exceptions import ModelInvalid
from gemini_webapi.types import Gem

from ..models import Message
from ..utils import g_config
from ..utils.helper import add_tag, save_file_to_tempfile, save_url_to_tempfile

XML_WRAP_HINT = (
    "\nYou MUST wrap every tool call response inside a single fenced block exactly like:\n"
    '```xml\n<tool_call name="tool_name">{"arg": "value"}</tool_call>\n```\n'
    "No other text may appear before or after the fence; otherwise the call will be ignored.\n"
)

MARKDOWN_ESCAPE_RE = re.compile(r"\\(?=\s*[\\`*_{}\[\]()#+\-.!])")
CODE_FENCE_RE = re.compile(r"(```.*?```|`[^`]*`)", re.DOTALL)


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

        await super().init(
            timeout=timeout,
            auto_close=auto_close,
            close_delay=close_delay,
            auto_refresh=auto_refresh,
            refresh_interval=refresh_interval,
            verbose=verbose,
        )

    async def generate_content(
        self,
        prompt: str,
        files: list[str | Path] | None = None,
        model: Model | str = Model.UNSPECIFIED,
        gem: Gem | str | None = None,
        chat: ChatSession | None = None,
        **kwargs,
    ) -> ModelOutput:
        cnt = 2  # Try 2 times before giving up
        last_exception: ModelInvalid | None = None
        while cnt:
            cnt -= 1
            try:
                return await super().generate_content(prompt, files, model, gem, chat, **kwargs)
            except ModelInvalid as e:
                # This is not always caused by model selection. Instead, it can be solved by retrying.
                # So we catch it and retry as a workaround.
                await asyncio.sleep(1)
                last_exception = e

        # If retrying failed, re-raise ModelInvalid
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("generate_content failed without receiving a ModelInvalid error.")

    @staticmethod
    async def process_message(
        message: Message, tempdir: Path | None = None, tagged: bool = True
    ) -> tuple[str, list[Path | str]]:
        """
        Process a single message and return model input.
        """
        files: list[Path | str] = []
        text_fragments: list[str] = []

        if isinstance(message.content, str):
            # Pure text content
            if message.content:
                text_fragments.append(message.content)
        elif isinstance(message.content, list):
            # Mixed content
            # TODO: Use Pydantic to enforce the value checking
            for item in message.content:
                if item.type == "text":
                    # Append multiple text fragments
                    if item.text:
                        text_fragments.append(item.text)

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
                    else:
                        raise ValueError("File must contain 'file_data' key")
        elif message.content is not None:
            raise ValueError("Unsupported message content type.")

        if message.tool_calls:
            tool_blocks: list[str] = []
            for call in message.tool_calls:
                args_text = call.function.arguments.strip()
                try:
                    parsed_args = json.loads(args_text)
                    args_text = json.dumps(parsed_args, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    # Leave args_text as is if it is not valid JSON
                    pass
                tool_blocks.append(
                    f'<tool_call name="{call.function.name}">{args_text}</tool_call>'
                )

            if tool_blocks:
                tool_section = "```xml\n" + "\n".join(tool_blocks) + "\n```"
                text_fragments.append(tool_section)

        model_input = "\n".join(fragment for fragment in text_fragments if fragment)

        # This is a workaround for Gemini Web's displaying issues with XML blocks.
        # Add this for tool calling
        if re.search(r"<\s*[^>]+>", model_input):
            hint = XML_WRAP_HINT
        else:
            hint = ""

        # Add role tag if needed
        if model_input:
            if tagged:
                model_input = add_tag(message.role, model_input + hint)
            else:
                model_input += hint

        return model_input, files

    @staticmethod
    async def process_conversation(
        messages: list[Message], tempdir: Path | None = None
    ) -> tuple[str, list[Path | str]]:
        """
        Process the entire conversation and return a formatted string and list of
        files. The last message is assumed to be the assistant's response.
        """
        # Determine once whether we need to wrap messages with role tags: only required
        # if the history already contains assistant/system messages. When every message
        # so far is from the user, we can skip tagging entirely.
        need_tag = any(m.role != "user" for m in messages)

        conversation: list[str] = []
        files: list[Path | str] = []

        for msg in messages:
            input_part, files_part = await GeminiClientWrapper.process_message(
                msg, tempdir, tagged=need_tag
            )
            conversation.append(input_part)
            files.extend(files_part)

        # Append an opening assistant tag only when we used tags above so that Gemini
        # knows where to start its reply.
        if need_tag:
            conversation.append(add_tag("assistant", "", unclose=True))

        return "\n".join(conversation), files

    @staticmethod
    def extract_output(response: ModelOutput, include_thoughts: bool = True) -> str:
        """
        Extract and format the output text from the Gemini response.
        """
        text = ""

        if include_thoughts and response.thoughts:
            text += f"<think>{response.thoughts}</think>\n"

        if response.text:
            text += response.text
        else:
            text += str(response)

        # Fix some escaped characters
        text = text.replace("&lt;", "<").replace("\\<", "<").replace("\\_", "_").replace("\\>", ">")

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

        text = _unescape_markdown(text)

        def simplify_link_target(text_content: str) -> str:
            match_colon_num = re.match(r"([^:]+:\d+)", text_content)
            if match_colon_num:
                return match_colon_num.group(1)
            return text_content

        def replacer(match: re.Match) -> str:
            outer_open_paren = match.group(1)
            display_text = match.group(2)

            new_target_url = simplify_link_target(display_text)
            new_link_segment = f"[`{display_text}`]({new_target_url})"

            if outer_open_paren:
                return f"{outer_open_paren}{new_link_segment})"
            else:
                return new_link_segment

        # Replace Google search links with simplified Markdown links
        pattern = r"(\()?\[`([^`]+?)`\]\((https://www.google.com/search\?q=)(.*?)(?<!\\)\)\)*(\))?"
        text = re.sub(pattern, replacer, text)

        # Fix inline code blocks
        pattern = r"`(\[[^\]]+\]\([^\)]+\))`"
        return re.sub(pattern, r"\1", text)
