import io
from pathlib import Path
from typing import Any

import orjson
from gemini_webapi import GeminiClient
from gemini_webapi.constants import AccountStatus
from gemini_webapi.types import AvailableModel
from loguru import logger

from app.models import AppMessage
from app.utils import g_config
from app.utils.helper import (
    add_tag,
    save_file_to_tempfile,
    save_url_to_tempfile,
)

# How a model is addressed: by name, or as one of the models a client discovered. `None` leaves
# the choice to Google.
type ModelSpec = str | AvailableModel | None


class GeminiClientWrapper(GeminiClient):
    """Gemini client with helper methods."""

    def __init__(self, client_id: str, **kwargs):
        self._cfg_impersonate: str | None = kwargs.pop("impersonate", None)
        super().__init__(**kwargs)
        self.id = client_id
        self._initialized = False
        # Chat id of the last conversation this client opened, its kind unverified. Google closes
        # an ephemeral window as soon as another conversation is created, so for those chats this
        # is the only cid still continuable. In memory and cleared on every (re)initialization:
        # once the session that opened a window is gone, nothing can vouch for it.
        self.latest_chat_cid: str | None = None

    async def init(self, *args: Any, **kwargs: Any) -> None:
        """
        Inject default configuration values from global settings.
        """
        config = g_config.gemini
        init_kwargs: dict[str, Any] = {
            "timeout": config.timeout,
            "watchdog_timeout": config.watchdog_timeout,
            "auto_refresh": config.auto_refresh,
            "refresh_interval": config.refresh_interval,
            "auto_close": config.auto_close,
            "close_delay": config.close_delay,
            "verbose": config.verbose,
        }
        if self._cfg_impersonate is not None:
            init_kwargs["impersonate"] = self._cfg_impersonate
        try:
            await super().init(**init_kwargs)
            self._initialized = True
            self.latest_chat_cid = None
        except Exception:
            self._initialized = False
            logger.exception(f"Failed to initialize GeminiClient {self.id}")
            raise

    def running(self) -> bool:
        return self._running

    def is_healthy(self) -> bool:
        """
        Check if the client is healthy.

        A client is healthy if it is active (running or initialized with auto-close)
        and the account status is available.
        """
        is_active = self._running or (self.auto_close and self._initialized)
        return is_active and self.account_status == AccountStatus.AVAILABLE

    def is_guest(self) -> bool:
        """Whether this client talks to Google without an account.

        Set when initialization falls back to a guest session, or when an authenticated one is
        rejected mid-flight because its cookies expired. Text generation keeps working, so this
        has to be asked rather than assumed away: the session just has no history, no uploads
        and no model choice.
        """
        return self.account_status == AccountStatus.UNAUTHENTICATED

    def can_upload(self) -> bool:
        """Whether files may be attached; Google rejects uploads from a guest session."""
        return not self.is_guest()

    def usable_model(self, model: ModelSpec) -> ModelSpec:
        """The requested model, or the only one a guest session may use.

        Google offers a guest no model choice, so honouring the request is impossible; serving
        its default beats failing outright. Callers keep addressing the request by its original
        model name, which is what conversation storage stays keyed on.
        """
        if not self.is_guest():
            return model

        # None when the guest registry is empty, which leaves the model unspecified and lets
        # Google answer with whatever it gives a signed-out visitor.
        default = next((m for m in self._model_registry.values() if m.is_available), None)
        requested = model if isinstance(model, str) else getattr(model, "model_name", None)
        if requested != getattr(default, "model_name", None):
            logger.warning(
                f"Client {self.id} is a guest session; serving "
                f"{default or 'the default model'} instead of the requested {requested}."
            )
        return default

    def chat_scope(self, temporary: bool) -> str | None:
        """Identity of the ephemeral window chats opened now belong to, else None.

        `None` is a normal chat, which Google keeps in the account's history and stays reusable
        across restarts. Any other value names a window only that exact session can continue, so
        a stored scope that no longer matches this client's proves the window behind it is gone.

        Keyed on the parent's per-session id, rerolled on every successful initialization, plus
        guest state. That covers both ways a window dies silently: a reinitialization, and a
        mid-session downgrade to guest, which keeps the session id but loses the account's chats.
        """
        if self.is_guest():
            return f"guest:{self._sessionid}"
        return f"temporary:{self._sessionid}" if temporary else None

    @staticmethod
    async def _process_content_item(
        item: Any, role: str, tempdir: Path | None
    ) -> tuple[str | None, Path | str | None]:
        """
        Process a single content item (text, image_url, file, input_audio).
        Returns a tuple of (text_fragment, file_path).
        """
        if item.type == "text":
            item_text = getattr(item, "text", "") or ""
            if item_text or role == "tool":
                return item_text, None
        elif item.type == "image_url":
            if item_media_url := getattr(item, "url", None):
                return None, await save_url_to_tempfile(item_media_url, tempdir)
            raise ValueError(f"{item.type} cannot be empty")
        elif item.type == "file":
            if file_url := getattr(item, "url", None):
                return None, await save_url_to_tempfile(file_url, tempdir)
            if not (file_data := getattr(item, "file_data", None)):
                raise ValueError("File must contain 'file_data' or 'url'")
            filename = getattr(item, "filename", "") or ""
            return None, await save_file_to_tempfile(file_data, filename, tempdir)
        elif item.type == "input_audio":
            if file_data := getattr(item, "file_data", None):
                return None, await save_file_to_tempfile(file_data, "audio.wav", tempdir)
            raise ValueError("input_audio must contain 'file_data' key")
        return None, None

    @staticmethod
    async def _extract_content_and_files(
        message: AppMessage, tempdir: Path | None
    ) -> tuple[list[str], list[Path | str]]:
        """
        Extract text fragments and files from message content.
        """
        files: list[Path | str] = []
        text_fragments: list[str] = []

        if isinstance(message.content, str):
            if message.content or message.role == "tool":
                text_fragments.append(message.content or "")
        elif isinstance(message.content, list):
            for item in message.content:
                text, file = await GeminiClientWrapper._process_content_item(
                    item, message.role, tempdir
                )
                if text is not None:
                    text_fragments.append(text)
                if file is not None:
                    files.append(file)
        elif message.content is None and message.role == "tool":
            text_fragments.append("")
        elif message.content is not None:
            raise ValueError(f"Unsupported message content type: {type(message.content)}")

        return text_fragments, files

    @staticmethod
    def _format_tool_results(
        text_fragments: list[str], tool_name: str | None, wrap_tool: bool
    ) -> list[str]:
        """
        Format tool results into the PascalCase technical protocol blocks.
        """
        tool_name = tool_name or "unknown"
        combined_content = "\n".join(text_fragments).strip()
        res_block = (
            f"[Result:{tool_name}]\n[ToolResult]\n{combined_content}\n[/ToolResult]\n[/Result]"
        )
        return [f"[ToolResults]\n{res_block}\n[/ToolResults]"] if wrap_tool else [res_block]

    @staticmethod
    def _format_tool_calls(message: AppMessage) -> str | None:
        """
        Format tool calls into the PascalCase technical protocol blocks.
        """
        if not message.tool_calls:
            return None

        tool_blocks: list[str] = []
        for call in message.tool_calls:
            params_text = call.function.arguments.strip()
            formatted_params = ""
            if params_text:
                try:
                    parsed_params = orjson.loads(params_text)
                    if isinstance(parsed_params, dict):
                        for k, v in parsed_params.items():
                            val_str = v if isinstance(v, str) else orjson.dumps(v).decode("utf-8")
                            formatted_params += (
                                f"[CallParameter:{k}]\n```\n{val_str}\n```\n[/CallParameter]\n"
                            )
                    else:
                        formatted_params += f"```\n{params_text}\n```\n"
                except orjson.JSONDecodeError:
                    formatted_params += f"```\n{params_text}\n```\n"

            tool_blocks.append(f"[Call:{call.function.name}]\n{formatted_params}[/Call]")

        return "[ToolCalls]\n" + "\n".join(tool_blocks) + "\n[/ToolCalls]" if tool_blocks else None

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
        text_fragments, files = await GeminiClientWrapper._extract_content_and_files(
            message, tempdir
        )

        if message.role == "tool":
            text_fragments = GeminiClientWrapper._format_tool_results(
                text_fragments, message.name, wrap_tool
            )

        if tool_section := GeminiClientWrapper._format_tool_calls(message):
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
