import asyncio
import contextlib
import hashlib
import io
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

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
    PolicySyncResult,
    extract_managed_last_used_at,
    sync_policy_gems,
    touch_managed_description,
)

# How a model is addressed: by name, or as one of the models a client discovered. `None` leaves
# the choice to Google.
type ModelSpec = str | AvailableModel | None

_MAX_MANAGED_GEM_RETRY_ATTEMPTS = 5


@dataclass
class _ManagedGemRetry:
    op: str
    gem_id: str
    attempt: int
    next_retry_at: float


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
        self._gem_lock = asyncio.Lock()
        self._policy_gem_ids: dict[str, str] = {}
        self._system_prompt_gem_ids: dict[str, str] = {}
        self._managed_gem_create_timestamps: deque[float] = deque()
        self._managed_gem_last_touch_timestamps: dict[str, float] = {}
        self._managed_gem_pending_touch_ids: set[str] = set()
        self._managed_gem_touch_worker_task: asyncio.Task[None] | None = None
        self._managed_gem_retry_queue: list[_ManagedGemRetry] = []
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
        while (
            self._managed_gem_create_timestamps
            and self._managed_gem_create_timestamps[0] < window_start
        ):
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

    def _prune_deleted_system_gem_ids(self, deleted_ids: list[str]) -> None:
        """Remove deleted gem ids from in-memory system prompt cache."""
        if not deleted_ids:
            return
        deleted_set = set(deleted_ids)
        self._system_prompt_gem_ids = {
            k: v for k, v in self._system_prompt_gem_ids.items() if v not in deleted_set
        }
        for gem_id in deleted_set:
            self._managed_gem_last_touch_timestamps.pop(gem_id, None)
            self._managed_gem_pending_touch_ids.discard(gem_id)

    def _enqueue_retry(self, op: str, gem_id: str, attempt: int = 1) -> None:
        if attempt > _MAX_MANAGED_GEM_RETRY_ATTEMPTS:
            return
        delay_sec = min(300.0, float(2**attempt))
        self._managed_gem_retry_queue.append(
            _ManagedGemRetry(
                op=op,
                gem_id=gem_id,
                attempt=attempt,
                next_retry_at=time.time() + delay_sec,
            )
        )
        self._managed_gem_metrics["managed_gems_retry_enqueued"] += 1

    async def _process_managed_retry_queue(self) -> None:
        """Process due retry operations (delete/touch) with backoff."""
        if not self._managed_gem_retry_queue or not self.running() or self.is_guest():
            return

        now = time.time()
        due = [op for op in self._managed_gem_retry_queue if op.next_retry_at <= now]
        self._managed_gem_retry_queue = [
            op for op in self._managed_gem_retry_queue if op.next_retry_at > now
        ]
        if not due:
            return

        prefix = g_config.gemini.gems.policies.prefix
        async with self._gem_lock:
            if not self.running() or self.is_guest():
                return
            gems = list(await self.fetch_gems(include_hidden=True))
            by_id = {gem.id: gem for gem in gems}
            for retry in due:
                try:
                    target = by_id.get(retry.gem_id)
                    if retry.op == "delete":
                        if (
                            target is not None
                            and not target.predefined
                            and target.name.startswith(prefix)
                        ):
                            await self.delete_gem(target)
                            self._prune_deleted_system_gem_ids([target.id])
                        self._managed_gem_metrics["managed_gems_retry_success"] += 1
                    elif retry.op == "touch":
                        if target is None:
                            self._managed_gem_metrics["managed_gems_retry_success"] += 1
                            continue
                        if (
                            target.predefined
                            or not target.name.startswith(prefix)
                            or target.prompt is None
                        ):
                            self._managed_gem_metrics["managed_gems_retry_success"] += 1
                            continue

                        updated_description = touch_managed_description(
                            target.description, now_ts=time.time()
                        )
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

    def _schedule_managed_policy_touch(self, gem_id: str) -> None:
        """Queue gem usage touch updates and ensure a background worker exists."""
        gem_cfg = g_config.gemini.gems
        if not gem_cfg.enabled or not gem_cfg.cleanup.enabled or self.is_guest():
            return

        self._managed_gem_pending_touch_ids.add(gem_id)
        if (
            self._managed_gem_touch_worker_task is None
            or self._managed_gem_touch_worker_task.done()
        ):
            self._managed_gem_touch_worker_task = asyncio.create_task(
                self._managed_policy_touch_worker()
            )

    async def _managed_policy_touch_worker(self) -> None:
        """Batch and flush pending managed gem touches in the background."""
        try:
            while self._managed_gem_pending_touch_ids and self.running() and not self.is_guest():
                await asyncio.sleep(0.5)
                await self._flush_managed_policy_touches()
                await self._process_managed_retry_queue()
        except asyncio.CancelledError:
            pass
        finally:
            self._managed_gem_touch_worker_task = None

    async def _flush_managed_policy_touches(self) -> None:
        """Flush queued managed gem touch updates in a single fetch/update pass."""
        gem_cfg = g_config.gemini.gems
        if (
            not gem_cfg.enabled
            or not gem_cfg.cleanup.enabled
            or not self.running()
            or self.is_guest()
        ):
            self._managed_gem_pending_touch_ids.clear()
            return

        pending = list(self._managed_gem_pending_touch_ids)
        if not pending:
            return

        self._managed_gem_pending_touch_ids.clear()

        now_ts = time.time()
        min_interval_sec = gem_cfg.cleanup.touch_interval_minutes * 60

        async with self._gem_lock:
            if not self.running() or self.is_guest():
                return
            gems = list(await self.fetch_gems(include_hidden=True))
            by_id = {gem.id: gem for gem in gems}

            for gem_id in pending:
                last_touch = self._managed_gem_last_touch_timestamps.get(gem_id)
                if last_touch is not None and now_ts - last_touch < min_interval_sec:
                    continue

                target = by_id.get(gem_id)
                if target is None or target.predefined:
                    continue
                if not target.name.startswith(gem_cfg.policies.prefix):
                    continue
                if target.prompt is None:
                    continue

                updated_description = touch_managed_description(target.description, now_ts=now_ts)
                if (target.description or "") == updated_description:
                    self._managed_gem_last_touch_timestamps[gem_id] = now_ts
                    continue

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
        self._prune_deleted_system_gem_ids(sync_result.deleted_ids)
        for failed_id in sync_result.failed_delete_ids:
            self._enqueue_retry("delete", failed_id)

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
            await self._initialize_gems()
        except Exception:
            self._initialized = False
            logger.exception(f"Failed to initialize GeminiClient {self.id}")
            raise

    async def close(self, *args: Any, **kwargs: Any) -> None:
        """Cancel background gem workers and close the underlying Gemini client."""
        task = self._managed_gem_touch_worker_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._managed_gem_touch_worker_task = None
        await super().close(*args, **kwargs)

    def running(self) -> bool:
        return self._running

    async def _initialize_gems(self) -> None:
        """Initialize gem cache and built-in policy gems based on server config."""
        gem_cfg = g_config.gemini.gems
        if not gem_cfg.enabled or self.is_guest():
            return

        async with self._gem_lock:
            if self.is_guest():
                return

            include_hidden = gem_cfg.include_hidden_on_fetch
            policy_mode = gem_cfg.policy

            if policy_mode == GeminiGemPolicyMode.OFF:
                if gem_cfg.fetch_on_init:
                    await self.fetch_gems(include_hidden=include_hidden)
                return

            if policy_mode in (
                GeminiGemPolicyMode.FETCH_ONLY,
                GeminiGemPolicyMode.CREATE_ON_DEMAND,
            ):
                sync_mode = cast(Literal["fetch_only", "create_on_demand"], str(policy_mode))
                default_policy = gem_cfg.policies.default_policy

                create_budget = None
                if policy_mode == GeminiGemPolicyMode.CREATE_ON_DEMAND:
                    create_budget = self._acquire_managed_gem_create_budget(
                        gem_cfg.create_rate_limit_per_minute
                    )
                    if create_budget <= 0:
                        self._managed_gem_metrics["managed_gems_rate_limit_skips"] += 1

                cleanup_days = gem_cfg.cleanup.unused_days if gem_cfg.cleanup.enabled else None

                sync_result: PolicySyncResult = await sync_policy_gems(
                    self,
                    prefix=gem_cfg.policies.prefix,
                    include_hidden=True,
                    default_policy_enabled=default_policy.enabled,
                    default_policy_key=default_policy.key,
                    default_prompt=default_policy.prompt,
                    mode=sync_mode,
                    create_budget=create_budget,
                    cleanup_unused_days=cleanup_days,
                    cleanup_dry_run=gem_cfg.cleanup.dry_run,
                    cleanup_max_deletes_per_run=gem_cfg.cleanup.max_deletes_per_run,
                    cleanup_require_managed_marker=gem_cfg.cleanup.require_managed_marker,
                    managed_max_total=gem_cfg.managed_gems_max_total,
                )
                self._apply_policy_sync_result(sync_result)

                if policy_mode == GeminiGemPolicyMode.CREATE_ON_DEMAND:
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

                # Refresh once more so callers observe the configured hidden visibility.
                await self.fetch_gems(include_hidden=include_hidden)

        await self._process_managed_retry_queue()

    def policy_gem_id(self, key: str) -> str | None:
        """Return a synced policy gem id for a logical key, or None when unavailable."""
        if self.is_guest():
            return None
        gem_id = self._policy_gem_ids.get(key)
        if gem_id:
            try:
                asyncio.get_running_loop()
                self._schedule_managed_policy_touch(gem_id)
            except RuntimeError:
                pass
        return gem_id

    async def policy_gem_id_or_create(self, key: str) -> str | None:
        """Return policy gem id, creating a missing managed gem on-demand when allowed."""
        if self.is_guest():
            return None

        existing = self.policy_gem_id(key)
        if existing is not None:
            return existing

        gem_cfg = g_config.gemini.gems
        if not gem_cfg.enabled or gem_cfg.policy != GeminiGemPolicyMode.CREATE_ON_DEMAND:
            return None

        async with self._gem_lock:
            if self.is_guest():
                return None

            existing = self._policy_gem_ids.get(key)
            if existing is not None:
                return existing

            create_budget = self._acquire_managed_gem_create_budget(
                gem_cfg.create_rate_limit_per_minute
            )
            if create_budget <= 0:
                self._managed_gem_metrics["managed_gems_rate_limit_skips"] += 1
                return None

            default_policy = gem_cfg.policies.default_policy
            cleanup_days = gem_cfg.cleanup.unused_days if gem_cfg.cleanup.enabled else None
            sync_result = await sync_policy_gems(
                self,
                prefix=gem_cfg.policies.prefix,
                include_hidden=True,
                default_policy_enabled=default_policy.enabled,
                default_policy_key=default_policy.key,
                default_prompt=default_policy.prompt,
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
                self._schedule_managed_policy_touch(created_or_found)
            return created_or_found

    async def system_prompt_gem_id_or_create(self, system_prompt: str) -> str | None:
        """Return or create a managed gem id for a raw system prompt text."""
        prompt = (system_prompt or "").strip()
        if not prompt or self.is_guest():
            return None

        gem_cfg = g_config.gemini.gems
        if not gem_cfg.enabled:
            return None

        policy_mode = gem_cfg.policy
        if policy_mode not in (
            GeminiGemPolicyMode.FETCH_ONLY,
            GeminiGemPolicyMode.CREATE_ON_DEMAND,
        ):
            return None

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_key = f"sys:{prompt_hash}"
        cached = self._system_prompt_gem_ids.get(cache_key)
        if cached is not None:
            self._schedule_managed_policy_touch(cached)
            return cached

        name = f"{gem_cfg.policies.prefix}sys_{prompt_hash[:24]}"
        base_description = "Managed system prompt gem created on-demand from API system message."

        async with self._gem_lock:
            if self.is_guest():
                return None

            cached = self._system_prompt_gem_ids.get(cache_key)
            if cached is not None:
                self._schedule_managed_policy_touch(cached)
                return cached

            gems = list(await self.fetch_gems(include_hidden=True))
            custom_gems = [gem for gem in gems if not gem.predefined]

            existing = next((gem for gem in custom_gems if gem.name == name), None)
            if existing is not None:
                if (
                    policy_mode == GeminiGemPolicyMode.CREATE_ON_DEMAND
                    and extract_managed_last_used_at(existing.description) is None
                ):
                    now_ts = time.time()
                    await self.update_gem(
                        gem=existing,
                        name=existing.name,
                        prompt=existing.prompt or prompt,
                        description=touch_managed_description(
                            existing.description or base_description,
                            now_ts=now_ts,
                        ),
                    )
                    self._managed_gem_last_touch_timestamps[existing.id] = now_ts
                    self._managed_gem_metrics["managed_gems_updated"] += 1
                else:
                    self._schedule_managed_policy_touch(existing.id)
                self._system_prompt_gem_ids[cache_key] = existing.id
                return existing.id

            if policy_mode != GeminiGemPolicyMode.CREATE_ON_DEMAND:
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

            now_ts = time.time()
            description = touch_managed_description(base_description, now_ts=now_ts)
            created = await self.create_gem(name=name, prompt=prompt, description=description)
            self._consume_managed_gem_create_budget(1)
            self._managed_gem_last_touch_timestamps[created.id] = now_ts
            self._managed_gem_metrics["managed_gems_created"] += 1
            self._system_prompt_gem_ids[cache_key] = created.id
            return created.id

    def managed_gem_metrics(self) -> dict[str, int]:
        """Return a copy of managed gem lifecycle counters."""
        return dict(self._managed_gem_metrics)

    def _require_authenticated_for_gems(self) -> None:
        if self.is_guest():
            raise RuntimeError("Gem operations require an authenticated Gemini client.")

    async def refresh_gems(self, include_hidden: bool | None = None) -> list[Gem]:
        """Fetch gems from Gemini and return a plain list for API responses."""
        self._require_authenticated_for_gems()
        gem_cfg = g_config.gemini.gems
        use_hidden = gem_cfg.include_hidden_on_fetch if include_hidden is None else include_hidden

        async with self._gem_lock:
            self._require_authenticated_for_gems()
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
        self._require_authenticated_for_gems()
        gems = self.list_cached_gems()
        if not gems:
            gems = await self.refresh_gems(include_hidden=include_hidden)

        found = self._find_gem_in_list(gems, gem_ref)
        if found is not None:
            return found

        raise ValueError(f"Gem '{gem_ref}' not found")

    async def create_custom_gem(self, name: str, prompt: str, description: str = "") -> Gem:
        """Create a custom gem and refresh local cache."""
        self._require_authenticated_for_gems()
        async with self._gem_lock:
            self._require_authenticated_for_gems()
            created = await self.create_gem(name=name, prompt=prompt, description=description)
            await self.fetch_gems(include_hidden=g_config.gemini.gems.include_hidden_on_fetch)
            return created

    async def update_custom_gem(
        self, gem_ref: str, name: str, prompt: str, description: str = ""
    ) -> Gem:
        """Update a custom gem identified by id or name and refresh local cache."""
        self._require_authenticated_for_gems()
        async with self._gem_lock:
            self._require_authenticated_for_gems()
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
        self._require_authenticated_for_gems()
        async with self._gem_lock:
            self._require_authenticated_for_gems()
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
            self._prune_deleted_system_gem_ids([target.id])
            await self.fetch_gems(include_hidden=g_config.gemini.gems.include_hidden_on_fetch)

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
