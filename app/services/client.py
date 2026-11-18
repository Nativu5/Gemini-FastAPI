import asyncio
import hashlib
import io
import time
from collections import deque
from pathlib import Path
from typing import Any

import orjson
from gemini_webapi import GeminiClient
from gemini_webapi.constants import AccountStatus
from gemini_webapi.types import AvailableModel, Gem
from loguru import logger

from app.models import AppMessage
from app.utils import g_config
from app.utils.config import GeminiGemPolicyMode
from app.utils.helper import (
    add_tag,
    save_file_to_tempfile,
    save_url_to_tempfile,
)

from .policy_gems import (
    delete_unused_gems,
    extract_managed_last_used_at,
    touch_managed_description,
)

# How a model is addressed: by name, or as one of the models a client discovered. `None` leaves
# the choice to Google.
type ModelSpec = str | AvailableModel | None


class GeminiClientWrapper(GeminiClient):
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
        self._gem_lock = asyncio.Lock()
        self._system_prompt_gems: dict[str, Gem] = {}  # sha256 of a system prompt -> its gem
        self._gems_created_at: deque[float] = deque()  # creations within the last minute
        self._gem_use_recorded_at: dict[str, float] = {}  # gem id -> time of the last update

    async def init(self, *args: Any, **kwargs: Any) -> None:
        """Initialize with the settings in `g_config.gemini`. Arguments are ignored: the
        library passes its own when it restarts a client, and the configuration decides."""
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

    async def delete_unused_gems(self) -> None:
        """Delete the managed gems that went unused for longer than the configured retention.

        Never call this from `init()`: when Google rejects a gem request the library restarts
        the client inside the call that holds `_gem_lock`, and `init()` would wait for it forever.
        """
        gem_cfg = g_config.gemini.gems
        if (
            gem_cfg.policy == GeminiGemPolicyMode.OFF
            or not gem_cfg.cleanup.enabled
            or self.is_guest()
        ):
            return

        async with self._gem_lock:
            deleted = await delete_unused_gems(
                self,
                prefix=gem_cfg.policies.prefix,
                unused_days=gem_cfg.cleanup.unused_days,
                dry_run=gem_cfg.cleanup.dry_run,
                max_deletes=gem_cfg.cleanup.max_deletes_per_run,
            )
        self._system_prompt_gems = {
            key: gem for key, gem in self._system_prompt_gems.items() if gem.id not in deleted
        }

    async def system_prompt_gem_id_or_create(self, system_prompt: str) -> str | None:
        """Id of the managed gem that holds this system prompt, or None to send it inline.

        None when gems are off, the session is a guest, no gem exists and the policy forbids
        creating one, or a creation limit is reached.
        """
        prompt = (system_prompt or "").strip()
        if not prompt or self.is_guest() or g_config.gemini.gems.policy == GeminiGemPolicyMode.OFF:
            return None

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        gem = self._system_prompt_gems.get(prompt_hash)
        if gem is None:
            async with self._gem_lock:
                # Looked up again: another request may have created it while this one waited.
                gem = self._system_prompt_gems.get(prompt_hash) or await self._find_or_create_gem(
                    prompt, prompt_hash
                )
                if gem is None:
                    return None
                self._system_prompt_gems[prompt_hash] = gem
        await self._record_gem_use(gem, prompt)
        return gem.id

    async def _find_or_create_gem(self, prompt: str, prompt_hash: str) -> Gem | None:
        """The account's gem for this prompt, created if the policy and the limits allow."""
        gem_cfg = g_config.gemini.gems
        name = f"{gem_cfg.policies.prefix}sys_{prompt_hash[:24]}"
        gems = [gem for gem in await self.fetch_gems(include_hidden=True) if not gem.predefined]
        if existing := next((gem for gem in gems if gem.name == name), None):
            return existing
        if gem_cfg.policy != GeminiGemPolicyMode.CREATE_ON_DEMAND:
            return None

        while self._gems_created_at and self._gems_created_at[0] < time.monotonic() - 60:
            self._gems_created_at.popleft()
        managed = sum(gem.name.startswith(gem_cfg.policies.prefix) for gem in gems)
        if (
            len(self._gems_created_at) >= gem_cfg.create_rate_limit_per_minute
            or managed >= gem_cfg.managed_gems_max_total
        ):
            return None

        created = await self.create_gem(
            name=name,
            prompt=prompt,
            description=touch_managed_description(
                "Managed system prompt gem created on-demand from API system message.",
                now_ts=time.time(),
            ),
        )
        self._gems_created_at.append(time.monotonic())
        return created

    async def _record_gem_use(self, gem: Gem, prompt: str) -> None:
        """Write the time of this use into the gem's description, where cleanup reads it.

        At most once per `touch_interval_minutes` for a gem, counted from the time already in its
        description. A failed update waits for the next interval too: while Google rejects gem
        requests, retrying on every request would hold each one up for the library's retries.
        """
        cleanup = g_config.gemini.gems.cleanup
        now = time.time()
        recorded_at = (
            self._gem_use_recorded_at.get(gem.id)
            or extract_managed_last_used_at(gem.description)
            or 0
        )
        if not cleanup.enabled or now - recorded_at < cleanup.touch_interval_minutes * 60:
            return
        self._gem_use_recorded_at[gem.id] = now
        try:
            await self.update_gem(
                gem=gem,
                name=gem.name,
                prompt=gem.prompt or prompt,
                description=touch_managed_description(gem.description, now_ts=now),
            )
        except Exception as exc:
            logger.warning(f"Failed to record the last use of gem {gem.name!r}: {exc}")

    def is_healthy(self) -> bool:
        """Running, or closed for inactivity after a good start, on an account in good standing."""
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
        """One content item as (text, file); at most one of the two is set."""
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
        """The text pieces and the files of a message, each in the order they appear."""
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
    def _format_tool_calls(message: AppMessage) -> str | None:
        """A message's tool calls as a `[ToolCalls]` protocol block, or None if it has none."""
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
        message: AppMessage, tempdir: Path | None = None
    ) -> tuple[str, list[Path | str]]:
        """One message as prompt text plus the files it attaches.

        A tool message becomes a `[Result:name]` block, which `process_conversation` groups with
        its neighbours; any other message is wrapped in its role's ChatML tags, with its tool
        calls appended as a `[ToolCalls]` block.
        """
        text_fragments, files = await GeminiClientWrapper._extract_content_and_files(
            message, tempdir
        )
        if message.role == "tool":
            result = "\n".join(text_fragments).strip()
            name = message.name or "unknown"
            return f"[Result:{name}]\n[ToolResult]\n{result}\n[/ToolResult]\n[/Result]", files

        if tool_section := GeminiClientWrapper._format_tool_calls(message):
            text_fragments.append(tool_section)
        model_input = "\n".join(text_fragments)
        return add_tag(message.role, model_input) if model_input else "", files

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
                        messages[i], tempdir
                    )
                    tool_blocks.append(part)
                    files.extend(part_files)
                    i += 1

                combined_tool_content = "\n".join(tool_blocks)
                wrapped_content = f"[ToolResults]\n{combined_tool_content}\n[/ToolResults]"
                conversation.append(add_tag("tool", wrapped_content))
            else:
                input_part, files_part = await GeminiClientWrapper.process_message(msg, tempdir)
                conversation.append(input_part)
                files.extend(files_part)
                i += 1

        conversation.append(add_tag("assistant", "", unclose=True))
        return "\n".join(conversation), files
