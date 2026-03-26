from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal

from gemini_webapi import GeminiClient
from gemini_webapi.types import Gem

from app.utils import g_config


@dataclass(frozen=True)
class PolicyGemSpec:
    """Declarative definition of a server-managed policy gem."""

    key: str
    name: str
    description: str
    prompt: str


@dataclass(frozen=True)
class PolicySyncResult:
    """Result payload for managed policy gem synchronization."""

    gem_ids: dict[str, str]
    created_count: int
    updated_count: int
    deleted_count: int
    dry_run_delete_count: int
    failed_delete_ids: list[str]
    skipped_missing_marker_count: int
    skipped_due_to_cap_count: int
    managed_total_count: int


_META_MARKER = "\n\n[gemini_fastapi_meta]"


def _split_description_meta(description: str | None) -> tuple[str, dict[str, Any]]:
    """Split managed metadata suffix from gem description."""
    text = description or ""
    marker_index = text.rfind(_META_MARKER)
    if marker_index == -1:
        return text, {}

    base = text[:marker_index]
    raw_meta = text[marker_index + len(_META_MARKER) :].strip()
    if not raw_meta:
        return base, {}

    try:
        parsed = json.loads(raw_meta)
        if isinstance(parsed, dict):
            return base, parsed
    except json.JSONDecodeError:
        pass
    return base, {}


def _compose_description_with_meta(base_description: str, last_used_at: float) -> str:
    """Compose description with stable managed metadata suffix."""
    meta = {
        "managed_by": "gemini_fastapi",
        "last_used_at": int(last_used_at),
    }
    return f"{base_description}{_META_MARKER}{json.dumps(meta, separators=(',', ':'))}"


def touch_managed_description(description: str | None, now_ts: float) -> str:
    """Return description with refreshed managed last_used timestamp."""
    base_description, _meta = _split_description_meta(description)
    return _compose_description_with_meta(base_description, last_used_at=now_ts)


def extract_managed_last_used_at(description: str | None) -> int | None:
    """Return managed `last_used_at` unix timestamp from description metadata."""
    _base, meta = _split_description_meta(description)
    value = meta.get("last_used_at")
    if isinstance(value, int) and value > 0:
        return value
    return None


def has_managed_marker(description: str | None) -> bool:
    """Return whether a description contains Gemini-FastAPI managed metadata."""
    _base, meta = _split_description_meta(description)
    return meta.get("managed_by") == "gemini_fastapi"


def _build_specs(prefix: str, default_prompt: str | None = None) -> list[PolicyGemSpec]:
    """Return built-in policy gems that should exist for every configured client.

    `default_prompt` may be supplied (from config) to override the built-in prompt.
    """
    # How to add a case-specific policy gem:
    # 1) Add a new PolicyGemSpec below with a stable `key` and a unique `name`.
    # 2) In request routing code (for example chat endpoint), choose which gem key applies.
    # 3) Resolve the gem id via `await client.policy_gem_id_or_create("your_key")`
    #    when using `create_on_demand`, or `client.policy_gem_id("your_key")`
    #    for fetch-only behavior, then pass that id only
    #    when the request matches your condition.
    # Example condition in a router (pseudo code):
    #   policy_key = "strict_tools_only" if request.tools else "general_capability_guardrail"
    #   policy_id = await client.policy_gem_id_or_create(policy_key)
    #   if policy_id:
    #       await session.send_message(..., gemini_options={"gem_id": policy_id})
    if default_prompt is None:
        general_guardrail_prompt = (
            "You are operating behind an OpenAI-compatible Gemini wrapper.\n"
        )
    else:
        general_guardrail_prompt = default_prompt

    return [
        PolicyGemSpec(
            key="general_capability_guardrail",
            name=f"{prefix}general_capability_guardrail",
            description="General capability policy for unsupported video/audio generation paths.",
            prompt=general_guardrail_prompt,
        )
    ]


async def _upsert_gem(
    client: GeminiClient,
    spec: PolicyGemSpec,
    existing: Gem | None,
) -> tuple[Gem, bool, bool]:
    """Create the policy gem if missing, or update it when the content changed."""
    now_ts = time.time()
    desired_description = _compose_description_with_meta(spec.description, last_used_at=now_ts)

    if existing is None:
        created = await client.create_gem(
            name=spec.name,
            description=desired_description,
            prompt=spec.prompt,
        )

        return created, True, False

    existing_base_description, _existing_meta = _split_description_meta(existing.description)
    if existing_base_description != spec.description or (existing.prompt or "") != spec.prompt:
        updated = await client.update_gem(
            gem=existing,
            name=spec.name,
            description=desired_description,
            prompt=spec.prompt,
        )
        return updated, False, True

    # Backfill metadata on older managed gems that predate managed suffix.
    if extract_managed_last_used_at(existing.description) is None:
        updated = await client.update_gem(
            gem=existing,
            name=spec.name,
            description=desired_description,
            prompt=spec.prompt,
        )
        return updated, False, True

    return existing, False, False


async def sync_policy_gems(
    client: GeminiClient,
    prefix: str = "fastapi_policy_",
    include_hidden: bool | None = None,
    default_prompt: str | None = None,
    mode: Literal["fetch_only", "create_on_demand"] = "fetch_only",
    create_budget: int | None = None,
    cleanup_unused_days: int | None = None,
    cleanup_dry_run: bool = False,
    cleanup_max_deletes_per_run: int | None = None,
    cleanup_require_managed_marker: bool = True,
    managed_max_total: int | None = None,
) -> PolicySyncResult:
    """Synchronize built-in policy gems and return a map from policy key to gem id.

    By default the runtime config `g_config.gemini.gems.include_hidden_on_fetch` is used
    unless `include_hidden` is explicitly provided. Callers may pass `include_hidden=True`
    to ensure hidden gems are included during the sync (recommended when reconciling
    hidden policy gems).

    Modes:
    - `fetch_only`: read existing managed gems and build id mapping only.
    - `create_on_demand`: create/update only the managed gem specs, without deleting extras.

    `create_budget` limits how many new managed gems can be created during this run.
    `cleanup_unused_days` removes managed prefixed gems whose last-used metadata
    is older than the configured threshold.
    `cleanup_dry_run` logs stale candidates without deleting.
    `cleanup_max_deletes_per_run` caps deletions for each sync pass.
    `cleanup_require_managed_marker` restricts deletion to managed-marker gems.
    `managed_max_total` caps total server-managed gems with this prefix.
    """

    prefix = (prefix or "fastapi_policy_").strip() or "fastapi_policy_"
    # Default include_hidden to the runtime config when not explicitly provided.
    use_hidden = include_hidden if include_hidden is not None else g_config.gemini.gems.include_hidden_on_fetch
    specs = _build_specs(prefix, default_prompt=default_prompt)
    await client.fetch_gems(include_hidden=use_hidden)
    custom_gems = [gem for gem in client.gems if not gem.predefined]

    deleted_count = 0
    dry_run_delete_count = 0
    failed_delete_ids: list[str] = []
    skipped_missing_marker_count = 0

    max_deletes_left = cleanup_max_deletes_per_run

    if cleanup_unused_days is not None and cleanup_unused_days > 0:
        cutoff_ts = int(time.time() - cleanup_unused_days * 24 * 60 * 60)
        for gem in custom_gems:
            if not gem.name.startswith(prefix):
                continue
            if cleanup_require_managed_marker and not has_managed_marker(gem.description):
                skipped_missing_marker_count += 1
                continue
            last_used_at = extract_managed_last_used_at(gem.description)
            if last_used_at is None:
                continue
            if last_used_at < cutoff_ts:
                if max_deletes_left is not None and max_deletes_left <= 0:
                    continue
                if cleanup_dry_run:
                    dry_run_delete_count += 1
                    continue
                try:
                    await client.delete_gem(gem)
                    deleted_count += 1
                    if max_deletes_left is not None:
                        max_deletes_left -= 1
                except Exception:
                    failed_delete_ids.append(gem.id)

        # Refresh inventory after cleanup deletions.
        if deleted_count > 0:
            await client.fetch_gems(include_hidden=use_hidden)
            custom_gems = [gem for gem in client.gems if not gem.predefined]

    managed_gems = [gem for gem in custom_gems if gem.name.startswith(prefix)]
    single_by_name = {gem.name: gem for gem in managed_gems}
    managed_total_count = len(managed_gems)

    result: dict[str, str] = {}
    created_count = 0
    updated_count = 0
    skipped_due_to_cap_count = 0
    for spec in specs:
        existing = single_by_name.get(spec.name)
        if mode == "fetch_only":
            if existing is None:
                continue
            gem = existing
        else:
            if existing is None and managed_max_total is not None and managed_total_count >= managed_max_total:
                skipped_due_to_cap_count += 1
                continue
            if existing is None and create_budget is not None and create_budget <= 0:
                continue
            gem, created, updated = await _upsert_gem(client, spec=spec, existing=existing)
            if created:
                managed_total_count += 1
                created_count += 1
                if create_budget is not None:
                    create_budget -= 1
            if updated:
                updated_count += 1
        result[spec.key] = gem.id

    return PolicySyncResult(
        gem_ids=result,
        created_count=created_count,
        updated_count=updated_count,
        deleted_count=deleted_count,
        dry_run_delete_count=dry_run_delete_count,
        failed_delete_ids=failed_delete_ids,
        skipped_missing_marker_count=skipped_missing_marker_count,
        skipped_due_to_cap_count=skipped_due_to_cap_count,
        managed_total_count=managed_total_count,
    )
