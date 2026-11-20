import asyncio
import hashlib
import io
import itertools
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

import gemini_webapi.client as webapi_client
import orjson
from gemini_webapi import GeminiClient
from gemini_webapi.constants import AccountStatus
from gemini_webapi.types import AvailableModel, Gem
from loguru import logger

from app.models import AppMessage, AppToolCall
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

# gemini-webapi 2.1.1 removes the link Google leaves where generated media goes only up to its
# first number, so ".../image_generation_content/0_974" left "_974" in the reply. This pattern
# takes the whole link; it can go once the library's own does.
webapi_client.ARTIFACTS_RE = re.compile(r"https?://googleusercontent\.com/(?:\w+/)+\d+(?:_\d+)*\n*")

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


async def _texts_and_files(
    message: AppMessage, tempdir: Path | None
) -> tuple[list[str], list[Path | str]]:
    """The text pieces and the files of a message, each in the order they appear. Empty pieces
    are kept only in a tool result."""
    if not isinstance(message.content, list):
        return [message.content or ""], []
    texts: list[str] = []
    files: list[Path | str] = []
    for item in message.content:
        if item.type == "text":
            if item.text or message.role == "tool":
                texts.append(item.text or "")
        elif item.type in ("image_url", "file") and item.url:
            files.append(await save_url_to_tempfile(item.url, tempdir))
        elif item.type in ("file", "input_audio") and item.file_data:
            filename = "audio.wav" if item.type == "input_audio" else item.filename or ""
            files.append(await save_file_to_tempfile(item.file_data, filename, tempdir))
        elif item.type in ("image_url", "file", "input_audio"):
            raise ValueError(f"{item.type} has neither a url nor data")
    return texts, files


def _tool_calls_block(calls: list[AppToolCall]) -> str:
    """Tool calls as the `[ToolCalls]` protocol block the model is taught."""
    blocks = []
    for call in calls:
        params = call.function.arguments.strip()
        try:
            parsed = orjson.loads(params) if params else {}
        except orjson.JSONDecodeError:
            parsed = params
        if isinstance(parsed, dict):
            body = "".join(
                f"[CallParameter:{key}]\n```\n"
                f"{value if isinstance(value, str) else orjson.dumps(value).decode()}"
                "\n```\n[/CallParameter]\n"
                for key, value in parsed.items()
            )
        else:
            body = f"```\n{params}\n```\n"
        blocks.append(f"[Call:{call.function.name}]\n{body}[/Call]")
    return "[ToolCalls]\n" + "\n".join(blocks) + "\n[/ToolCalls]"


async def process_conversation(
    messages: list[AppMessage], tempdir: Path | None = None
) -> tuple[str, list[str | Path | bytes | io.BytesIO]]:
    """Messages as one prompt, each in its role's ChatML tags and ending with an open assistant
    turn, plus the files they attach.

    A run of tool messages shares one `tool` turn, a `[Result:name]` block for each; another
    message's tool calls follow its text as a `[ToolCalls]` block.
    """
    turns: list[str] = []
    files: list[str | Path | bytes | io.BytesIO] = []
    for is_tool, group in itertools.groupby(messages, key=lambda message: message.role == "tool"):
        prompts: list[str] = []
        for message in group:
            texts, message_files = await _texts_and_files(message, tempdir)
            files += message_files
            if is_tool:
                result = "\n".join(texts).strip()
                name = message.name or "unknown"
                prompts.append(f"[Result:{name}]\n[ToolResult]\n{result}\n[/ToolResult]\n[/Result]")
                continue
            calls = _tool_calls_block(message.tool_calls) if message.tool_calls else ""
            text = "\n".join(part for part in ("\n".join(texts), calls) if part)
            prompts.append(add_tag(message.role, text) if text else "")
        if is_tool:
            joined = "\n".join(prompts)
            turns.append(add_tag("tool", f"[ToolResults]\n{joined}\n[/ToolResults]"))
        else:
            turns += prompts
    turns.append(add_tag("assistant", "", unclose=True))
    return "\n".join(turns), files
