import asyncio
import hashlib
import time
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Any, cast

import orjson
from gemini_webapi import GeminiClient, ModelOutput
from gemini_webapi.types import Gem
from loguru import logger

from app.models import Message
from app.utils import g_config
from app.utils.helper import (
    add_tag,
    normalize_llm_text,
    save_file_to_tempfile,
    save_url_to_tempfile,
)

from .policy_gems import PolicySyncResult, sync_policy_gems, touch_managed_description

_UNSET = object()


def _resolve(value: Any, fallback: Any):
    return fallback if value is _UNSET else value


class GeminiClientWrapper(GeminiClient):
    """Gemini client with helper methods."""

    @dataclass
    class _ManagedGemRetry:
        op: str
        gem_id: str
        attempt: int
        next_retry_at: float

    def __init__(self, client_id: str, **kwargs):
        super().__init__(**kwargs)
        self.id = client_id
        self._gem_lock = asyncio.Lock()
        self._policy_gem_ids: dict[str, str] = {}
        self._system_prompt_gem_ids: dict[str, str] = {}
        self._managed_gem_create_timestamps: deque[float] = deque()
        self._managed_gem_last_touch_timestamps: dict[str, float] = {}
        self._managed_gem_retry_queue: list[GeminiClientWrapper._ManagedGemRetry] = []
        self._managed_gem_metrics: dict[str, int] = {
            "managed_gems_created": 0,
            "managed_gems_updated": 0,
            "managed_gems_deleted": 0,
            "managed_gems_delete_dry_run": 0,
            "managed_gems_skipped_missing_marker": 0,
            "managed_gems_skipped_cap": 0,
            "managed_gems_rate_limit_skips": 0,
            "managed_gems_retry_enqueued": 0,
            "managed_gems_retry_success": 0,
            "managed_gems_retry_failed": 0,
            "managed_gems_touch_updated": 0,
        }

    def _acquire_managed_gem_create_budget(self, per_minute: int) -> int:
        """Return remaining create budget in the current 60-second window."""
        now = time.monotonic()
        window_start = now - 60.0
        while self._managed_gem_create_timestamps and self._managed_gem_create_timestamps[0] < window_start:
            self._managed_gem_create_timestamps.popleft()

        used = len(self._managed_gem_create_timestamps)
        return max(0, per_minute - used)

    def _consume_managed_gem_create_budget(self, count: int) -> None:
        """Record managed gem create usage for rate limiting."""
        if count <= 0:
            return
        now = time.monotonic()
        for _ in range(count):
            self._managed_gem_create_timestamps.append(now)

    def _enqueue_retry(self, op: str, gem_id: str, attempt: int = 1) -> None:
        delay_sec = min(300.0, float(2**attempt))
        self._managed_gem_retry_queue.append(
            self._ManagedGemRetry(
                op=op,
                gem_id=gem_id,
                attempt=attempt,
                next_retry_at=time.time() + delay_sec,
            )
        )
        self._managed_gem_metrics["managed_gems_retry_enqueued"] += 1

    async def _process_managed_retry_queue(self) -> None:
        """Process due retry operations (delete/touch) with backoff."""
        if not self._managed_gem_retry_queue:
            return

        now = time.time()
        due = [op for op in self._managed_gem_retry_queue if op.next_retry_at <= now]
        self._managed_gem_retry_queue = [op for op in self._managed_gem_retry_queue if op.next_retry_at > now]
        if not due:
            return

        async with self._gem_lock:
            gems = list(await self.fetch_gems(include_hidden=True))
            by_id = {gem.id: gem for gem in gems}
            for retry in due:
                try:
                    target = by_id.get(retry.gem_id)
                    if retry.op == "delete":
                        if target is not None and not target.predefined and target.name.startswith(g_config.gemini.gems.policies.prefix):
                            await self.delete_gem(target)
                        self._managed_gem_metrics["managed_gems_retry_success"] += 1
                    elif retry.op == "touch":
                        if target is None:
                            self._managed_gem_metrics["managed_gems_retry_success"] += 1
                            continue
                        if target.predefined or not target.name.startswith(g_config.gemini.gems.policies.prefix) or target.prompt is None:
                            self._managed_gem_metrics["managed_gems_retry_success"] += 1
                            continue

                        updated_description = touch_managed_description(target.description, now_ts=time.time())
                        await self.update_gem(
                            gem=target,
                            name=target.name,
                            description=updated_description,
                            prompt=target.prompt,
                        )
                        self._managed_gem_last_touch_timestamps[target.id] = time.time()
                        self._managed_gem_metrics["managed_gems_touch_updated"] += 1
                        self._managed_gem_metrics["managed_gems_retry_success"] += 1
                except Exception:
                    self._managed_gem_metrics["managed_gems_retry_failed"] += 1
                    self._enqueue_retry(retry.op, retry.gem_id, retry.attempt + 1)

    def _apply_policy_sync_result(self, sync_result: PolicySyncResult) -> None:
        """Apply sync result into cache, metrics, and retry queue."""
        self._policy_gem_ids = sync_result.gem_ids
        self._managed_gem_metrics["managed_gems_created"] += sync_result.created_count
        self._managed_gem_metrics["managed_gems_updated"] += sync_result.updated_count
        self._managed_gem_metrics["managed_gems_deleted"] += sync_result.deleted_count
        self._managed_gem_metrics["managed_gems_delete_dry_run"] += sync_result.dry_run_delete_count
        self._managed_gem_metrics["managed_gems_skipped_missing_marker"] += (
            sync_result.skipped_missing_marker_count
        )
        self._managed_gem_metrics["managed_gems_skipped_cap"] += (
            sync_result.skipped_due_to_cap_count
        )
        for failed_id in sync_result.failed_delete_ids:
            self._enqueue_retry("delete", failed_id)

    async def init(
        self,
        timeout: float = cast(float, _UNSET),
        watchdog_timeout: float = cast(float, _UNSET),
        auto_close: bool = False,
        close_delay: float = cast(float, _UNSET),
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
        close_delay = timeout
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

            # Keep gem cache and server-managed policy gems in a known-good state.
            await self._initialize_gems()
        except Exception:
            logger.exception(f"Failed to initialize GeminiClient {self.id}")
            raise

    def running(self) -> bool:
        return self._running

    async def _initialize_gems(self) -> None:
        """Initialize gem cache and built-in policy gems based on server config."""
        gem_cfg = g_config.gemini.gems
        if not gem_cfg.enabled:
            return

        async with self._gem_lock:
            include_hidden = gem_cfg.include_hidden_on_fetch

            if gem_cfg.fetch_on_init:
                await self.fetch_gems(include_hidden=include_hidden)

            policy_mode = gem_cfg.policy
            if policy_mode == "off":
                return

            if policy_mode == "privacy":
                logger.warning(
                    "gemini.gems.policy='privacy' is intended for request-time ephemeral flow; "
                    "startup policy sync is skipped"
                )
                return

            # Force include_hidden=True during managed-policy sync so hidden
            # server-managed gems are discovered.
            if policy_mode in ("fetch_only", "create_on_demand"):
                default_prompt = None
                policy_dp = getattr(gem_cfg.policies, "default_policy", None)
                if policy_dp and getattr(policy_dp, "enabled", False):
                    default_prompt = getattr(policy_dp, "prompt", None)

                create_budget = None
                if policy_mode == "create_on_demand":
                    create_budget = self._acquire_managed_gem_create_budget(
                        gem_cfg.create_rate_limit_per_minute
                    )
                    if create_budget <= 0:
                        self._managed_gem_metrics["managed_gems_rate_limit_skips"] += 1

                cleanup_days = None
                if gem_cfg.cleanup.enabled:
                    cleanup_days = gem_cfg.cleanup.unused_days

                sync_result: PolicySyncResult = await sync_policy_gems(
                    self,
                    prefix=gem_cfg.policies.prefix,
                    include_hidden=True,
                    default_prompt=default_prompt,
                    mode=policy_mode,
                    create_budget=create_budget,
                    cleanup_unused_days=cleanup_days,
                    cleanup_dry_run=gem_cfg.cleanup.dry_run,
                    cleanup_max_deletes_per_run=gem_cfg.cleanup.max_deletes_per_run,
                    cleanup_require_managed_marker=gem_cfg.cleanup.require_managed_marker,
                    managed_max_total=gem_cfg.managed_gems_max_total,
                )
                self._apply_policy_sync_result(sync_result)

                if policy_mode == "create_on_demand":
                    self._consume_managed_gem_create_budget(sync_result.created_count)

                logger.info(
                    "Managed gem sync stats client='{}': created={}, updated={}, deleted={}, "
                    "dry_run_deletes={}, retries_queued={}, managed_total={}",
                    self.id,
                    sync_result.created_count,
                    sync_result.updated_count,
                    sync_result.deleted_count,
                    sync_result.dry_run_delete_count,
                    len(sync_result.failed_delete_ids),
                    sync_result.managed_total_count,
                )

                # Refresh once more so callers can immediately read the final state.
                await self.fetch_gems(include_hidden=include_hidden)

        await self._process_managed_retry_queue()

    def policy_gem_id(self, key: str) -> str | None:
        """Return a synced policy gem id for a logical key, or None when unavailable."""
        gem_id = self._policy_gem_ids.get(key)
        if gem_id:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._touch_managed_policy_gem_usage(gem_id))
            except RuntimeError:
                # No running loop in this context.
                pass
        return gem_id

    async def policy_gem_id_or_create(self, key: str) -> str | None:
        """Return policy gem id, creating a missing managed gem on-demand when allowed.

        On-demand creation is only attempted when:
        - gem management is enabled,
        - `gemini.gems.policy` is `create_on_demand`, and
        - create-rate/cap limits allow creating more managed gems.
        """
        existing = self.policy_gem_id(key)
        if existing is not None:
            return existing

        gem_cfg = g_config.gemini.gems
        if not gem_cfg.enabled:
            return None
        if gem_cfg.policy != "create_on_demand":
            return None

        async with self._gem_lock:
            # Re-check after acquiring lock in case another coroutine just synced.
            existing = self._policy_gem_ids.get(key)
            if existing is not None:
                return existing

            create_budget = self._acquire_managed_gem_create_budget(
                gem_cfg.create_rate_limit_per_minute
            )
            if create_budget <= 0:
                self._managed_gem_metrics["managed_gems_rate_limit_skips"] += 1
                return None

            default_prompt = None
            policy_dp = getattr(gem_cfg.policies, "default_policy", None)
            if policy_dp and getattr(policy_dp, "enabled", False):
                default_prompt = getattr(policy_dp, "prompt", None)

            cleanup_days = gem_cfg.cleanup.unused_days if gem_cfg.cleanup.enabled else None
            sync_result = await sync_policy_gems(
                self,
                prefix=gem_cfg.policies.prefix,
                include_hidden=True,
                default_prompt=default_prompt,
                mode="create_on_demand",
                create_budget=create_budget,
                cleanup_unused_days=cleanup_days,
                cleanup_dry_run=gem_cfg.cleanup.dry_run,
                cleanup_max_deletes_per_run=gem_cfg.cleanup.max_deletes_per_run,
                cleanup_require_managed_marker=gem_cfg.cleanup.require_managed_marker,
                managed_max_total=gem_cfg.managed_gems_max_total,
            )
            self._apply_policy_sync_result(sync_result)
            self._consume_managed_gem_create_budget(sync_result.created_count)

            created_or_found = self._policy_gem_ids.get(key)
            if created_or_found is not None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._touch_managed_policy_gem_usage(created_or_found))
                except RuntimeError:
                    pass
            return created_or_found

    async def system_prompt_gem_id_or_create(self, system_prompt: str) -> str | None:
        """Return/create a managed gem id for a raw system prompt text.

        This supports request-time prompt de-duplication: same system prompt will
        map to the same managed gem name (hash-based) and be cached in memory.
        """
        prompt = (system_prompt or "").strip()
        if not prompt:
            return None

        gem_cfg = g_config.gemini.gems
        if not gem_cfg.enabled:
            return None

        policy_mode = gem_cfg.policy
        if policy_mode not in {"fetch_only", "create_on_demand"}:
            return None

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_key = f"sys:{prompt_hash}"
        cached = self._system_prompt_gem_ids.get(cache_key)
        if cached is not None:
            return cached

        name = f"{gem_cfg.policies.prefix}sys_{prompt_hash[:24]}"

        async with self._gem_lock:
            cached = self._system_prompt_gem_ids.get(cache_key)
            if cached is not None:
                return cached

            gems = list(await self.fetch_gems(include_hidden=True))
            custom_gems = [gem for gem in gems if not gem.predefined]

            existing = next((gem for gem in custom_gems if gem.name == name), None)
            if existing is not None:
                self._system_prompt_gem_ids[cache_key] = existing.id
                return existing.id

            if policy_mode != "create_on_demand":
                return None

            create_budget = self._acquire_managed_gem_create_budget(
                gem_cfg.create_rate_limit_per_minute
            )
            if create_budget <= 0:
                self._managed_gem_metrics["managed_gems_rate_limit_skips"] += 1
                return None

            managed_total = len(
                [gem for gem in custom_gems if gem.name.startswith(gem_cfg.policies.prefix)]
            )
            if managed_total >= gem_cfg.managed_gems_max_total:
                self._managed_gem_metrics["managed_gems_skipped_cap"] += 1
                return None

            description = "Managed system prompt gem created on-demand from API system message."
            created = await self.create_gem(name=name, prompt=prompt, description=description)
            self._consume_managed_gem_create_budget(1)
            self._managed_gem_metrics["managed_gems_created"] += 1
            self._system_prompt_gem_ids[cache_key] = created.id
            return created.id

    async def _touch_managed_policy_gem_usage(self, gem_id: str) -> None:
        """Refresh managed metadata last-used timestamp with write-throttling."""
        gem_cfg = g_config.gemini.gems
        if not gem_cfg.cleanup.enabled:
            return

        now_ts = time.time()
        min_interval_sec = gem_cfg.cleanup.touch_interval_minutes * 60
        last_touch = self._managed_gem_last_touch_timestamps.get(gem_id)
        if last_touch is not None and now_ts - last_touch < min_interval_sec:
            return

        async with self._gem_lock:
            gems = list(await self.fetch_gems(include_hidden=True))
            target: Gem | None = next((gem for gem in gems if gem.id == gem_id), None)
            if target is None:
                return

            if target.predefined:
                return

            prefix = gem_cfg.policies.prefix
            if not target.name.startswith(prefix):
                return

            if target.prompt is None:
                return

            updated_description = touch_managed_description(target.description, now_ts=now_ts)
            if (target.description or "") == updated_description:
                self._managed_gem_last_touch_timestamps[gem_id] = now_ts
                return

            try:
                await self.update_gem(
                    gem=target,
                    name=target.name,
                    description=updated_description,
                    prompt=target.prompt,
                )
                self._managed_gem_last_touch_timestamps[gem_id] = now_ts
                self._managed_gem_metrics["managed_gems_touch_updated"] += 1
            except Exception:
                self._enqueue_retry("touch", gem_id)
        
    def managed_gem_metrics(self) -> dict[str, int]:
        """Return a copy of managed gem lifecycle counters."""
        return dict(self._managed_gem_metrics)

    async def refresh_gems(self, include_hidden: bool | None = None) -> list[Gem]:
        """Fetch gems from Gemini and return a plain list for API responses."""
        gem_cfg = g_config.gemini.gems
        use_hidden = gem_cfg.include_hidden_on_fetch if include_hidden is None else include_hidden

        async with self._gem_lock:
            gem_jar = await self.fetch_gems(include_hidden=use_hidden)
            return list(gem_jar)

    def list_cached_gems(self) -> list[Gem]:
        """Return cached gems, or an empty list when cache is not initialized yet."""
        try:
            return list(self.gems)
        except RuntimeError:
            return []

    @staticmethod
    def _find_gem_in_list(gems: list[Gem], gem_ref: str) -> Gem | None:
        """Find a gem in a list by id or case-insensitive name."""
        ref_stripped = (gem_ref or "").strip()
        normalized = ref_stripped.lower()
        for gem in gems:
            gem_id = (gem.id or "").strip()
            if gem_id == ref_stripped or gem.name.lower() == normalized:
                return gem
        return None

    async def get_gem(self, gem_ref: str, include_hidden: bool | None = None) -> Gem:
        """Find a gem by id or name. Name matching is case-insensitive."""
        gems = self.list_cached_gems()
        if not gems:
            gems = await self.refresh_gems(include_hidden=include_hidden)

        found = self._find_gem_in_list(gems, gem_ref)
        if found is not None:
            return found

        raise ValueError(f"Gem '{gem_ref}' not found")

    async def create_custom_gem(self, name: str, prompt: str, description: str = "") -> Gem:
        """Create a custom gem and refresh local cache."""
        async with self._gem_lock:
            created = await self.create_gem(name=name, prompt=prompt, description=description)
            await self.fetch_gems(include_hidden=g_config.gemini.gems.include_hidden_on_fetch)
            return created

    async def update_custom_gem(
        self, gem_ref: str, name: str, prompt: str, description: str = ""
    ) -> Gem:
        """Update a custom gem identified by id or name and refresh local cache."""
        async with self._gem_lock:
            gems = self.list_cached_gems()
            if not gems:
                gems = list(
                    await self.fetch_gems(
                        include_hidden=g_config.gemini.gems.include_hidden_on_fetch,
                    )
                )
            target = self._find_gem_in_list(gems, gem_ref)
            if target is None:
                raise ValueError(f"Gem '{gem_ref}' not found")

            updated = await self.update_gem(
                gem=target,
                name=name,
                prompt=prompt,
                description=description,
            )
            await self.fetch_gems(include_hidden=g_config.gemini.gems.include_hidden_on_fetch)
            return updated

    async def delete_custom_gem(self, gem_ref: str) -> None:
        """Delete a custom gem identified by id or name and refresh local cache."""
        async with self._gem_lock:
            gems = self.list_cached_gems()
            if not gems:
                gems = list(
                    await self.fetch_gems(
                        include_hidden=g_config.gemini.gems.include_hidden_on_fetch,
                    )
                )
            target = self._find_gem_in_list(gems, gem_ref)
            if target is None:
                raise ValueError(f"Gem '{gem_ref}' not found")

            await self.delete_gem(target)
            await self.fetch_gems(include_hidden=g_config.gemini.gems.include_hidden_on_fetch)

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
