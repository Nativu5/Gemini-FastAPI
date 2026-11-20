import hmac
import tempfile
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.datastructures import Headers, QueryParams
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.utils import g_config


class GoogleClientsOnV1Middleware:
    """Serve `/v1/models...` from the native API when the key comes the way Google's API takes it.

    Google's SDKs call `/v1` unless told otherwise, and `/v1/models` is also the OpenAI model list;
    OpenAI clients send the key as a bearer token instead.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope["path"] if scope["type"] == "http" else ""
        if (path == "/v1/models" or path.startswith("/v1/models/")) and (
            "x-goog-api-key" in Headers(scope=scope) or "key" in QueryParams(scope["query_string"])
        ):
            # In place: Starlette answers an unhandled error outside this middleware, from this scope.
            scope["path"] = "/v1beta" + path.removeprefix("/v1")
        await self.app(scope, receive, send)


class RequestBodyLimitMiddleware:
    """Reject request bodies past a local ceiling, before the app buffers them.

    A declared `content-length` is refused up front; a chunked body is measured as it arrives
    and cut off once it crosses the limit. The ceiling exists to bound this process's memory,
    not to describe what Gemini Web will accept - the upstream decides that for itself.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.detail = (
            f"Request body exceeds the wrapper safety ceiling of {max_body_bytes} bytes. "
            "This is a local resource guard, not Gemini Web's upstream input limit."
        )

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
                response = error_response(
                    scope["path"], status.HTTP_413_CONTENT_TOO_LARGE, self.detail
                )
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    # Raised in the route, whose error handler answers it.
                    raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail=self.detail)
            return message

        await self.app(scope, limited_receive, send)


_OPENAI_ERROR_TYPES = {
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
}

_GRPC_STATUS = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    413: "RESOURCE_EXHAUSTED",
    429: "RESOURCE_EXHAUSTED",
    503: "UNAVAILABLE",
}


def google_error_body(status_code: int, message: str) -> dict[str, Any]:
    """An error in the envelope of Google's API."""
    grpc_status = _GRPC_STATUS.get(status_code, "INTERNAL")
    return {"error": {"code": status_code, "message": message, "status": grpc_status}}


def error_response(path: str, status_code: int, message: str) -> JSONResponse:
    """An HTTP error in the envelope of the API that `path` belongs to."""
    if path.startswith("/v1beta/"):
        return JSONResponse(
            status_code=status_code, content=google_error_body(status_code, message)
        )
    error_type = _OPENAI_ERROR_TYPES.get(
        status_code, "invalid_request_error" if status_code < 500 else "server_error"
    )
    body = {"error": {"message": message, "type": error_type, "param": None, "code": None}}
    return JSONResponse(status_code=status_code, content=body)


def add_exception_handlers(app: FastAPI) -> None:
    """Answer every error in the envelope of the API the request was made to."""

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        response = error_response(request.url.path, exc.status_code, str(exc.detail))
        response.headers.update(exc.headers or {})  # e.g. `Allow` on a 405
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        """A malformed request is a 400 in both APIs, not FastAPI's 422."""
        return error_response(request.url.path, 400, f"Invalid request: {exc.errors() or exc}")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        return error_response(request.url.path, 500, str(exc))


def get_temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


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
