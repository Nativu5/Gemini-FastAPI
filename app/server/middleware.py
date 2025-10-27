import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..utils import g_config


def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return ORJSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail}},
        )

    return ORJSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"message": str(exc)}},
    )


def get_temp_dir():
    temp_dir = tempfile.TemporaryDirectory()
    try:
        yield Path(temp_dir.name)
    finally:
        temp_dir.cleanup()


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    if not g_config.server.api_key:
        return ""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token"
        )

    api_key = credentials.credentials
    if api_key != g_config.server.api_key:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Wrong API key")

    return api_key


def add_exception_handler(app: FastAPI):
    app.add_exception_handler(Exception, global_exception_handler)


def add_cors_middleware(app: FastAPI):
    if g_config.cors.enabled:
        cors = g_config.cors
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors.allow_origins,
            allow_credentials=cors.allow_credentials,
            allow_methods=cors.allow_methods,
            allow_headers=cors.allow_headers,
        )
