import asyncio
import base64
import hashlib
import hmac
import re
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from gemini_webapi import ModelOutput
from gemini_webapi.types.image import GeneratedImage, Image
from gemini_webapi.types.video import GeneratedMedia, GeneratedVideo
from loguru import logger

from app.utils import g_config
from app.utils.helper import extract_image_dimensions

MEDIA_STORE_DIR = Path(g_config.storage.media_path)
MEDIA_STORE_DIR.mkdir(parents=True, exist_ok=True)

# The extension is whatever the upstream save produced, so it has to allow more than plain
# alphanumerics (`.tar.gz`, `.x-m4a`) while still excluding every path separator, and with
# it any way to step out of the store.
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


def get_media_token(filename: str) -> str:
    """Generate a HMAC-SHA256 token for a filename using the API key."""
    secret = g_config.server.api_key
    if not secret:
        return ""

    return hmac.new(secret.encode(), filename.encode(), hashlib.sha256).hexdigest()


def verify_media_token(filename: str, token: str | None) -> bool:
    """True if the token is the one signed for this file, or if no API key is configured:
    media is then open, like the rest of the API."""
    if expected := get_media_token(filename):
        return hmac.compare_digest(token, expected) if token else False
    return True


def cleanup_expired_media(ttl_seconds: int) -> int:
    """Delete media files last written more than `ttl_seconds` ago; 0 keeps them for good."""
    if ttl_seconds <= 0:
        return 0

    cutoff = time.time() - ttl_seconds
    count = 0
    for file_path in MEDIA_STORE_DIR.iterdir():
        if not file_path.is_file():
            continue
        try:
            if file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                count += 1
        except Exception as e:
            logger.warning(f"Failed to delete expired media {file_path}: {e}")

    if count > 0:
        logger.info(f"Cleaned up {count} expired media files.")
    return count


type ProcessedImageData = tuple[str, int | None, int | None, str, str]
type ProcessedMediaData = dict[str, tuple[str, str]]
type GeneratedFile = Image | GeneratedVideo | GeneratedMedia
type DownloadedData = ProcessedImageData | ProcessedMediaData
# A downloaded file as a reply shows it: the markdown that links it, and its data if an image.
type ShownFile = tuple[str, ProcessedImageData | None]


async def _image_to_base64(image: Image, temp_dir: Path) -> ProcessedImageData:
    """Persist an image provided by gemini_webapi and return base64 plus dimensions, filename, and hash."""
    if isinstance(image, GeneratedImage):
        try:
            saved_path = await image.save(path=str(temp_dir), full_size=True)
        except Exception as e:
            logger.warning(
                f"Failed to download full-size GeneratedImage, retrying with default size: {e}"
            )
            saved_path = await image.save(path=str(temp_dir), full_size=False)
    else:
        saved_path = await image.save(path=str(temp_dir))

    original_path = Path(saved_path)
    data = original_path.read_bytes()
    # The library names the file after its content type, so it always has an extension.
    random_name = f"img_{uuid.uuid4().hex}{original_path.suffix}"
    new_path = temp_dir / random_name
    original_path.rename(new_path)

    width, height = extract_image_dimensions(data)
    file_hash = hashlib.sha256(data).hexdigest()
    return base64.b64encode(data).decode("ascii"), width, height, random_name, file_hash


async def _media_to_local_file(
    media: GeneratedVideo | GeneratedMedia, temp_dir: Path
) -> ProcessedMediaData:
    """Persist media and return dict mapping type to (filename, hash)"""
    saved_paths = await media.save(path=str(temp_dir))
    results = {}
    path_map = {}

    for mtype, spath in saved_paths.items():
        if not spath:
            continue
        try:
            if spath in path_map:
                # One saved file listed under two roles; it was renamed the first time round.
                results[mtype] = path_map[spath]
                continue
            original_path = Path(spath)
            data = original_path.read_bytes()
            # The library names the file after its content type, so it always has an extension.
            random_name = f"media_{uuid.uuid4().hex}{original_path.suffix}"
            new_path = temp_dir / random_name
            original_path.rename(new_path)

            fhash = hashlib.sha256(data).hexdigest()
            results[mtype] = (random_name, fhash)
            path_map[spath] = (random_name, fhash)
        except Exception as e:
            logger.warning(f"Error processing {mtype} at {spath}: {e}")

    return results


async def _download(item: GeneratedFile) -> tuple[GeneratedFile, DownloadedData] | None:
    """Download one generated file into the media store; None if that fails."""
    try:
        if isinstance(item, Image):
            return item, await _image_to_base64(item, MEDIA_STORE_DIR)
        return item, await _media_to_local_file(item, MEDIA_STORE_DIR)
    except Exception as exc:
        logger.warning(f"Background download failed: {exc}")
        return None


class _Downloads:
    """The images, videos and audio generated for one reply, downloaded in the background.

    A download starts as soon as a chunk mentions the file, so most are finished by the time the
    text has been sent. Gemini can list one file under several urls, or send one picture twice;
    the reply shows it once, and a second copy is deleted.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._tasks: list[asyncio.Task[tuple[GeneratedFile, DownloadedData] | None]] = []
        self._seen_urls: set[str] = set()
        self._image_hashes: set[str] = set()
        self._kept_media: dict[str, str] = {}  # content hash -> file kept for it

    def add(self, output: ModelOutput) -> None:
        """Start downloading what this chunk mentions for the first time."""
        for item in [*output.images, *output.videos, *output.media]:
            url = getattr(item, "url", None) or getattr(item, "mp3_url", None)
            if url and url not in self._seen_urls:
                self._seen_urls.add(url)
                self._tasks.append(asyncio.create_task(_download(item)))

    async def cancel(self) -> None:
        """Stop the downloads still running, on a path that will not use their results.

        Left alone they keep writing files nothing refers to and, if they fail, raise
        `Task exception was never retrieved` once they are garbage collected.
        """
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def as_completed(self) -> AsyncGenerator[ShownFile | None]:
        """Each file to show as its download finishes, and None after every 5 seconds without
        one, so that a stream can send a keep-alive."""
        while self._tasks:
            done, _ = await asyncio.wait(
                self._tasks, timeout=5.0, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                yield None
                continue
            finished = [task for task in self._tasks if task in done]
            self._tasks = [task for task in self._tasks if task not in done]
            for task in finished:
                if shown := self._shown(task.result()):
                    yield shown

    async def all(self) -> list[ShownFile]:
        """Every file to show, in the order the files were listed."""
        tasks, self._tasks = self._tasks, []
        return [shown for done in await asyncio.gather(*tasks) if (shown := self._shown(done))]

    def _shown(self, download: tuple[GeneratedFile, DownloadedData] | None) -> ShownFile | None:
        """A finished download as the reply shows it; None if it failed or shows nothing new."""
        if download is None:
            return None
        item, data = download
        if isinstance(data, dict):
            markdown = _media_markdown(item.title, data, self._kept_media, self._base_url)
            return (markdown, None) if markdown else None
        *_, name, content_hash = data
        if content_hash in self._image_hashes:
            (MEDIA_STORE_DIR / name).unlink(missing_ok=True)
            return None
        self._image_hashes.add(content_hash)
        return f"![{item.title}]({_media_url(name, self._base_url)})", data


def _media_url(name: str, base_url: str) -> str:
    """The signed link under which a file in the media store is served."""
    return f"{base_url}media/{name}?token={get_media_token(name)}"


def _media_markdown(
    title: str, files: ProcessedMediaData, seen_hashes: dict[str, str], base_url: str
) -> str:
    """Markdown links to one generated video or audio item, or "" if it has nothing new to show.

    `seen_hashes` maps content hash to the file this response already keeps for it. A second copy
    of the same content is deleted; a video or audio track that is such a copy is not linked
    again, and a thumbnail that is one reuses the file kept.
    """
    urls: dict[str, str] = {}
    for kind, (name, content_hash) in files.items():
        kept = seen_hashes.setdefault(content_hash, name)
        if kept != name:
            (MEDIA_STORE_DIR / name).unlink(missing_ok=True)
            if kind in ("video", "audio"):
                continue
        urls[kind] = _media_url(kept, base_url)

    thumbnail = urls.get("video_thumbnail") or urls.get("audio_thumbnail")
    links = []
    for kind, label in (("video", title), ("audio", f"{title} - Audio")):
        if url := urls.get(kind):
            links.append(f"[![{label}]({thumbnail})]({url})" if thumbnail else f"[{label}]({url})")
    return "\n\n".join(links)


router = APIRouter()


@router.get("/media/{filename}")
async def get_media(filename: str, token: str | None = Query(default=None)):
    if not verify_media_token(filename, token):
        raise HTTPException(status_code=403, detail="Invalid token")

    file_path = _resolve_media_file(MEDIA_STORE_DIR, filename)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(file_path)
