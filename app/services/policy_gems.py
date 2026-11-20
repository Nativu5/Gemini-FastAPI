"""Bookkeeping for the gems this server creates in a Google account.

A managed gem carries a short JSON suffix in its description, saying who owns it and when it was
last used. That is what lets unused ones be found and deleted without touching gems the account
owner made.
"""

import json
from typing import Any

_META_MARKER = "\n\n[gemini_fastapi_meta]"


def _split_description_meta(description: str | None) -> tuple[str, dict[str, Any]]:
    text = description or ""
    marker_index = text.rfind(_META_MARKER)
    if marker_index == -1:
        return text, {}

    base = text[:marker_index]
    try:
        parsed = json.loads(text[marker_index + len(_META_MARKER) :])
        if isinstance(parsed, dict):
            return base, parsed
    except json.JSONDecodeError:
        pass
    return base, {}


def touch_managed_description(description: str | None, now_ts: float) -> str:
    """Return description with refreshed managed last_used timestamp."""
    base_description, _meta = _split_description_meta(description)
    meta = {"managed_by": "gemini_fastapi", "last_used_at": int(now_ts)}
    return f"{base_description}{_META_MARKER}{json.dumps(meta, separators=(',', ':'))}"


def extract_managed_last_used_at(description: str | None) -> int | None:
    """Return managed `last_used_at` unix timestamp from description metadata."""
    _base, meta = _split_description_meta(description)
    value = meta.get("last_used_at")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None
