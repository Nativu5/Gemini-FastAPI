import asyncio
import hashlib
import importlib
import io
import itertools
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import gemini_webapi.client as webapi_client
import orjson
from curl_cffi.requests import Cookies, Response
from gemini_webapi import GeminiClient
from gemini_webapi.constants import GRPC, AccountStatus
from gemini_webapi.exceptions import AuthError
from gemini_webapi.types import AvailableModel, Gem, RPCData
from loguru import logger

from app.models import AppMessage, AppToolCall
from app.utils import g_config
from app.utils.config import GeminiGemPolicyMode
from app.utils.helper import (
    add_tag,
    save_file_to_tempfile,
    save_url_to_tempfile,
)

from .policy_gems import extract_managed_last_used_at, touch_managed_description

# gemini-webapi 2.1.1 removes the link Google leaves where generated media goes only up to its
# first number, so ".../image_generation_content/0_974" left "_974" in the reply. This pattern
# takes the whole link; it can go once the library's own does.
webapi_client.ARTIFACTS_RE = re.compile(r"https?://googleusercontent\.com/(?:\w+/)+\d+(?:_\d+)*\n*")

# Since about 2026-11-19 Google sends cookies it no longer accepts to its sign-in page, which
# carries the same init values as Gemini, so gemini-webapi 2.1.1 takes it for a signed-in session.
# Read as a failed attempt, it lets the library try the next cookie source: the configured
# cookies after a stale cache. HanaokaYuzu/Gemini-API#382 does the same; this goes with it.
_token_module = vars(importlib.import_module("gemini_webapi.utils.get_access_token"))
_extract_init_values = _token_module["_extract_payload"]


def _gemini_init_values(response: Response):
    if urlparse(str(response.url)).hostname != "gemini.google.com":
        return None
    return _extract_init_values(response)


_token_module["_extract_payload"] = _gemini_init_values

# Without a __Secure-1PSID, gemini-webapi 2.1.1 signs in with the most recently cached cookies,
# which belong to whichever account was cached last. A client configured without cookies is a
# guest, so that lookup finds nothing.
_token_module["_get_cookie_cache_dir"] = lambda: Path(os.devnull)

# A Cookie header names no domains. These are the cookies of gemini.google.com alone: sent to
# lh3.google.com as well, they turn the download of a generated image into a 403
# (HanaokaYuzu/Gemini-API#381). Every other cookie belongs to all of google.com.
_GEMINI_HOST_COOKIES = {"OSID", "__Secure-OSID"}


def _cookie_jar(header: str) -> Cookies:
    """The cookies of a Cookie request header sent to gemini.google.com, on their domains."""
    jar = Cookies()
    for pair in header.split(";"):
        name, sep, value = pair.strip().partition("=")
        if sep:
            domain = "gemini.google.com" if name in _GEMINI_HOST_COOKIES else ".google.com"
            jar.set(name, value, domain=domain, secure=True)
    return jar


# How a model is addressed: by name, or as one of the models a client discovered. `None` leaves
# the choice to Google.
type ModelSpec = str | AvailableModel | None


class GeminiClientWrapper(GeminiClient):
    def __init__(self, client_id: str, cookies: str | None = None, **kwargs):
        self._cfg_impersonate: str | None = kwargs.pop("impersonate", None)
        super().__init__(**kwargs)
        # Configured with cookies, a client must stay signed in: run as a guest, it would serve
        # every request with less, unnoticed. One configured without them is a guest on purpose.
        self.has_account = cookies is not None
        if cookies is not None:
            self.cookies = _cookie_jar(cookies)
        self.id = client_id
        self._initialized = False
        # Chat id of the last conversation this client opened, its kind unverified. Google closes
        # an ephemeral window as soon as another conversation is created, so for those chats this
        # is the only cid still continuable. A cid from before a restart cannot pass for one: the
        # restart changes `chat_scope`, which is compared first.
        self.latest_chat_cid: str | None = None
        self._gem_lock = asyncio.Lock()
        self._system_prompt_gems: dict[str, Gem] = {}  # sha256 of a system prompt -> its gem
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
        except Exception:
            self._initialized = False
            logger.exception(f"Failed to initialize GeminiClient {self.id}")
            raise

    async def _init_rpc(self) -> None:
        # Signed in with a browser's full cookies, Gemini answers in the account's language, and
        # the library names models after what Gemini calls them: "gemini-自动", not "gemini-auto".
        # Every request of the session is then sent in English.
        self.language = "en"
        await super()._init_rpc()

    async def _sync_activity(self) -> None:
        """The library's heartbeat, which it also sends at startup and before every message.

        Once Google no longer accepts the cookies it refuses the heartbeat, and the library would
        go on as a guest. Raising here stops the startup, or the message before it is sent.
        """
        await super()._sync_activity()
        if self.signed_out():
            raise AuthError(
                f"Google no longer accepts the cookies of client {self.id}: copy a new Cookie "
                "header from a signed-in browser."
            )

    async def _batch_execute(self, payloads: list[RPCData], *args: Any, **kwargs: Any) -> Response:
        # The web app asks for the account status with "[1]". gemini-webapi 2.1.1 sends "[]", which
        # Google has been seen to answer with 400 since about 2026-11-19 (HanaokaYuzu/Gemini-API#381).
        payloads = [
            p.model_copy(update={"payload": "[1]"}) if p.rpcid == GRPC.GET_USER_STATUS else p
            for p in payloads
        ]
        response = await super()._batch_execute(payloads, *args, **kwargs)
        if any(p.rpcid == GRPC.READ_USER_PREFERENCES for p in payloads):
            # gemini-webapi 2.1.1 ignores the answer to its heartbeat. Read, a refusal marks the
            # session unauthenticated, as a refused answer to any other request does.
            list(self._parse_rpc_results(response.text, GRPC.READ_USER_PREFERENCES))
        return response

    def running(self) -> bool:
        return self._running

    async def delete_unused_gems(self) -> None:
        """Delete the managed gems last used more than `cleanup.unused_days` ago.

        Only a gem that carries this server's last-use record is ever deleted: one a person made
        under the same name prefix has none, and is left alone.

        Never call this from `init()`: when Google rejects a gem request the library restarts
        the client inside the call that holds `_gem_lock`, and `init()` would wait for it forever.
        """
        if self.is_guest():
            return
        gem_cfg = g_config.gemini.gems
        cutoff_ts = int(time.time() - gem_cfg.cleanup.unused_days * 24 * 60 * 60)
        deleted: list[str] = []
        would_delete = 0
        async with self._gem_lock:
            # Hidden predefined gems keep the list from ever being empty, which gemini-webapi
            # treats as a broken response: it raises and closes the client.
            for gem in await self.fetch_gems(include_hidden=True):
                last_used_at = extract_managed_last_used_at(gem.description)
                if (
                    gem.predefined
                    or not gem.name.startswith(gem_cfg.policies.prefix)
                    or last_used_at is None
                    or last_used_at >= cutoff_ts
                    or len(deleted) >= gem_cfg.cleanup.max_deletes_per_run
                ):
                    continue
                if gem_cfg.cleanup.dry_run:
                    would_delete += 1
                    continue
                try:
                    await self.delete_gem(gem)
                    deleted.append(gem.id)
                except Exception as exc:
                    # Still stale, so the next cleanup tries again.
                    logger.warning(f"Failed to delete unused managed gem {gem.name!r}: {exc}")
        logger.info(f"Managed gem cleanup: deleted={len(deleted)}, dry_run={would_delete}")
        self._system_prompt_gems = {
            key: gem for key, gem in self._system_prompt_gems.items() if gem.id not in deleted
        }

    async def system_prompt_gem_id(self, system_prompt: str) -> str | None:
        """Id of the managed gem that holds this system prompt, or None to send it inline.

        None when gems are off, the session is a guest, or the account has no such gem.
        """
        if self.is_guest() or g_config.gemini.gems.policy == GeminiGemPolicyMode.OFF:
            return None

        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        gem = self._system_prompt_gems.get(prompt_hash)
        if gem is None:
            async with self._gem_lock:
                # Looked up again: another request may have found it while this one waited.
                gem = self._system_prompt_gems.get(prompt_hash) or await self._find_gem(prompt_hash)
                if gem is None:
                    return None
                self._system_prompt_gems[prompt_hash] = gem
        await self._record_gem_use(gem, system_prompt)
        return gem.id

    async def _find_gem(self, prompt_hash: str) -> Gem | None:
        """The account's managed gem for this prompt, if an earlier version of the server made it."""
        name = f"{g_config.gemini.gems.policies.prefix}sys_{prompt_hash[:24]}"
        gems = await self.fetch_gems(include_hidden=True)
        return next((gem for gem in gems if not gem.predefined and gem.name == name), None)

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
        """Running, or closed for inactivity after a good start, as what it was configured to be:
        signed in to an account in good standing, or a guest."""
        is_active = self._running or (self.auto_close and self._initialized)
        expected = AccountStatus.AVAILABLE if self.has_account else AccountStatus.UNAUTHENTICATED
        return is_active and self.account_status == expected

    def is_guest(self) -> bool:
        """Whether this client talks to Google without an account: it has no history, no uploads
        and no model choice."""
        return self.account_status == AccountStatus.UNAUTHENTICATED

    def signed_out(self) -> bool:
        """Configured with cookies that Google no longer accepts."""
        return self.has_account and self.is_guest()

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

        Keyed on the parent's per-session id, rerolled on every successful initialization, which is
        how a window dies silently. A guest's chats are ephemeral in every mode.
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
