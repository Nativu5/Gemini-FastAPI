"""Bookkeeping for the gems this server creates in a Google account.

A managed gem carries a short JSON suffix in its description, saying who owns it and when it was
last used. That is what lets unused ones be found and deleted without touching gems the account
owner made.
"""

import json
import time
from typing import Any

from gemini_webapi import GeminiClient
from loguru import logger

_META_MARKER = "\n\n[gemini_fastapi_meta]"


def _split_description_meta(description: str | None) -> tuple[str, dict[str, Any]]:
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
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


async def delete_unused_gems(
    client: GeminiClient, prefix: str, unused_days: int, *, dry_run: bool, max_deletes: int
) -> list[str]:
    """Delete managed gems last used more than `unused_days` ago and return their ids.

    Only a gem that carries this server's last-use record is ever deleted: one a person made
    under the same name prefix has none, and is left alone.
    """
    cutoff_ts = int(time.time() - unused_days * 24 * 60 * 60)
    deleted: list[str] = []
    would_delete = 0

    for gem in await client.fetch_gems(include_hidden=True):
        if gem.predefined or not gem.name.startswith(prefix):
            continue
        last_used_at = extract_managed_last_used_at(gem.description)
        if last_used_at is None or last_used_at >= cutoff_ts or len(deleted) >= max_deletes:
            continue
        if dry_run:
            would_delete += 1
            continue
        try:
            await client.delete_gem(gem)
            deleted.append(gem.id)
        except Exception as exc:
            # Still stale, so the next cleanup tries again.
            logger.warning(f"Failed to delete unused managed gem {gem.name!r}: {exc}")

    logger.info(f"Managed gem cleanup: deleted={len(deleted)}, dry_run={would_delete}")
    return deleted
