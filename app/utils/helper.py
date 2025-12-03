import base64
import mimetypes
import tempfile
from pathlib import Path

import httpx
from loguru import logger

VALID_TAG_ROLES = {"user", "assistant", "system", "tool"}


def add_tag(role: str, content: str, unclose: bool = False) -> str:
    """Surround content with role tags"""
    if role not in VALID_TAG_ROLES:
        logger.warning(f"Unknown role: {role}, returning content without tags")
        return content

    return f"<|im_start|>{role}\n{content}" + ("\n<|im_end|>" if not unclose else "")


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens heuristically based on character count"""
    return int(len(text) / 3)


async def save_file_to_tempfile(
    file_in_base64: str, file_name: str = "", tempdir: Path | None = None
) -> Path:
    data = base64.b64decode(file_in_base64)
    suffix = Path(file_name).suffix if file_name else ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempdir) as tmp:
        tmp.write(data)
        path = Path(tmp.name)

    return path


async def save_url_to_tempfile(url: str, tempdir: Path | None = None):
    data: bytes | None = None
    suffix: str | None = None
    if url.startswith("data:image/"):
        # Base64 encoded image
        metadata_part = url.split(",")[0]
        mime_type = metadata_part.split(":")[1].split(";")[0]

        base64_data = url.split(",")[1]
        data = base64.b64decode(base64_data)

        # Guess extension from mime type, default to the subtype if not found
        suffix = mimetypes.guess_extension(mime_type)
        if not suffix:
            suffix = f".{mime_type.split('/')[1]}"
    else:
        # http files
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
            suffix = Path(url).suffix or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempdir) as tmp:
        tmp.write(data)
        path = Path(tmp.name)

    return path
