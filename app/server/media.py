import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.server.middleware import get_media_store_dir, verify_media_token

router = APIRouter()
# The extension is whatever the upstream save produced, so it has to allow more than plain
# alphanumerics (`.tar.gz`, `.x-m4a`) while still excluding every path separator and `..`.
MEDIA_FILENAME_RE = re.compile(r"(?:img|media)_[0-9a-f]{32}\.[A-Za-z0-9][A-Za-z0-9.\-_]{0,15}\Z")


def _resolve_media_file(media_store: Path, filename: str) -> Path | None:
    """Return an existing media file inside the store, or None if the name is not one of ours.

    The name has to match the pattern this server generates, which excludes separators and
    traversal outright; the containment check then covers a store reached through a symlink.
    """
    if not MEDIA_FILENAME_RE.fullmatch(filename):
        return None

    root = media_store.resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


@router.get("/media/{filename}", tags=["Media"])
async def get_media(filename: str, token: str | None = Query(default=None)):
    if not verify_media_token(filename, token):
        raise HTTPException(status_code=403, detail="Invalid token")

    file_path = _resolve_media_file(get_media_store_dir(), filename)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(file_path)
