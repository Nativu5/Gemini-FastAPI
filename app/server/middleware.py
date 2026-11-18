import hashlib
import hmac
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.utils import g_config

MEDIA_STORE_DIR = Path(g_config.storage.media_path)
MEDIA_STORE_DIR.mkdir(parents=True, exist_ok=True)


class RequestBodyLimitMiddleware:
    """Reject request bodies past a local ceiling, before the app buffers them.

    A declared `content-length` is refused up front; a chunked body is measured as it arrives
    and cut off once it crosses the limit. The ceiling exists to bound this process's memory,
    not to describe what Gemini Web will accept - the upstream decides that for itself.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max(0, max_body_bytes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.max_body_bytes == 0:
            await self.app(scope, receive, send)
            return

        content_length = next(
            (
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = 0
            if declared_size > self.max_body_bytes:
                await self._send_too_large(scope, receive, send)
                return

        received = 0
        overflowed = False

        async def limited_receive() -> Message:
            nonlocal overflowed, received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    overflowed = True
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=self._detail(),
                    )
            return message

        async def limited_send(message: Message) -> None:
            # FastAPI turns the receive error into its own generic envelope; drop that so the
            # surface-appropriate body below is what the client actually gets.
            if not overflowed:
                await send(message)

        await self.app(scope, limited_receive, limited_send)
        if overflowed:
            await self._send_too_large(scope, receive, send)

    def _detail(self) -> str:
        return (
            f"Request body exceeds the wrapper safety ceiling of {self.max_body_bytes} bytes. "
            "This is a local resource guard, not Gemini Web's upstream input limit."
        )

    async def _send_too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        detail = self._detail()
        if str(scope.get("path", "")).startswith("/v1beta/"):
            content: dict[str, Any] = {
                "error": {
                    "code": status.HTTP_413_CONTENT_TOO_LARGE,
                    "message": detail,
                    "status": "RESOURCE_EXHAUSTED",
                }
            }
        else:
            content = _openai_error_body(status.HTTP_413_CONTENT_TOO_LARGE, detail)
        response = JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content=content,
        )
        await response(scope, receive, send)


def get_media_store_dir() -> Path:
    return MEDIA_STORE_DIR


def get_media_token(filename: str) -> str:
    """Generate a HMAC-SHA256 token for a filename using the API key."""
    secret = g_config.server.api_key
    if not secret:
        return ""

    msg = filename.encode("utf-8")
    secret_bytes = secret.encode("utf-8")
    return hmac.new(secret_bytes, msg, hashlib.sha256).hexdigest()


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


_OPENAI_ERROR_TYPES = {
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
}


def _openai_error_body(status_code: int, message: str) -> dict[str, Any]:
    error_type = _OPENAI_ERROR_TYPES.get(
        status_code, "invalid_request_error" if status_code < 500 else "server_error"
    )
    return {"error": {"message": message, "type": error_type, "param": None, "code": None}}


def openai_error(status_code: int, message: str) -> JSONResponse:
    """An HTTP error in the envelope the OpenAI clients read."""
    return JSONResponse(status_code=status_code, content=_openai_error_body(status_code, message))


def get_temp_dir():
    temp_dir = tempfile.TemporaryDirectory()
    try:
        yield Path(temp_dir.name)
    finally:
        temp_dir.cleanup()


def _check_api_key(api_key: str | None) -> None:
    """Let a request through if it carries the configured key, or if none is configured."""
    expected = g_config.server.api_key
    if not expected:
        return
    if api_key is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    if not hmac.compare_digest(api_key.encode(), expected.encode()):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Wrong API key")


def _bearer_token(request: Request) -> str | None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token if scheme.lower() == "bearer" else None


def verify_api_key(
    # Read through HTTPBearer so that /docs shows the scheme and offers to send the key.
    credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
) -> None:
    """OpenAI-style auth: `Authorization: Bearer <key>`."""
    _check_api_key(credentials.credentials if credentials else None)


def verify_gemini_api_key(request: Request) -> None:
    """Gemini-style auth: the `x-goog-api-key` header, else the `key` query parameter, else a
    bearer token. The first one present decides, so a wrong key is not rescued by a right one."""
    _check_api_key(
        request.headers.get("x-goog-api-key")
        or request.query_params.get("key")
        or _bearer_token(request)
    )
