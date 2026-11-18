"""Transport-layer behaviour: the request body ceiling, and serving generated media.

The limit is enforced twice over: once from a declared `content-length`, and again by counting
a chunked body as it arrives. Both have to produce the error shape of the surface they were
addressed to, and both have to travel back out through CORS.

Media resolution is the other half: the name has to be one this server generated - which rules
out traversal - without being so narrow that a legitimate extension becomes an unreachable file.
"""

import os

import pytest
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.server.media import _resolve_media_file
from app.server.middleware import RequestBodyLimitMiddleware

LIMIT = 100


def _build_app(max_body_bytes: int = LIMIT, *, with_cors: bool = False) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/echo")
    async def echo(request: Request):
        return {"received": len(await request.body())}

    @app.post("/v1beta/models/x:generateContent")
    async def gemini_echo(request: Request):
        return {"received": len(await request.body())}

    # Registration order is reversed at runtime, so this mirrors app.main.create_app: the
    # limiter goes on first precisely so CORS ends up wrapping it.
    app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=max_body_bytes)
    if with_cors:
        app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
        )
    return app


def _chunks(total: int, size: int = 40):
    """Send without a content-length, so only the running count can catch the overflow."""
    sent = 0
    while sent < total:
        step = min(size, total - sent)
        yield b"x" * step
        sent += step


def test_a_body_within_the_ceiling_reaches_the_route():
    with TestClient(_build_app()) as client:
        response = client.post("/v1/echo", content=b"x" * (LIMIT - 1))
    assert response.status_code == 200
    assert response.json() == {"received": LIMIT - 1}


def test_a_body_exactly_at_the_ceiling_is_allowed():
    with TestClient(_build_app()) as client:
        response = client.post("/v1/echo", content=b"x" * LIMIT)
    assert response.status_code == 200


def test_a_declared_oversize_body_is_refused():
    with TestClient(_build_app()) as client:
        response = client.post("/v1/echo", content=b"x" * (LIMIT + 1))
    assert response.status_code == 413
    assert "safety ceiling" in response.json()["error"]["message"]


def test_a_chunked_oversize_body_is_refused():
    with TestClient(_build_app()) as client:
        response = client.post("/v1/echo", content=_chunks(LIMIT * 5))
    assert response.status_code == 413
    assert "safety ceiling" in response.json()["error"]["message"]


@pytest.mark.parametrize("content", [b"x" * (LIMIT + 1), None], ids=["declared", "chunked"])
def test_the_gemini_surface_gets_googles_error_envelope(content):
    body = content if content is not None else _chunks(LIMIT * 5)
    with TestClient(_build_app()) as client:
        response = client.post("/v1beta/models/x:generateContent", content=body)
    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == 413
    assert error["status"] == "RESOURCE_EXHAUSTED"


def test_a_zero_ceiling_disables_the_guard():
    with TestClient(_build_app(0)) as client:
        response = client.post("/v1/echo", content=b"x" * (LIMIT * 100))
    assert response.status_code == 200


def test_a_refusal_still_carries_cors_headers():
    """Without this the browser reports an opaque CORS failure instead of the 413."""
    with TestClient(_build_app(with_cors=True)) as client:
        allowed = client.post("/v1/echo", content=b"x", headers={"Origin": "https://example.com"})
        refused = client.post(
            "/v1/echo", content=b"x" * (LIMIT + 1), headers={"Origin": "https://example.com"}
        )
    assert allowed.headers["access-control-allow-origin"] == "*"
    assert refused.status_code == 413
    assert refused.headers["access-control-allow-origin"] == "*"


def test_an_unparsable_content_length_falls_back_to_counting():
    """A bogus header must not be trusted as 0 and wave an oversize body through."""
    app = _build_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/echo",
            content=b"x" * (LIMIT + 1),
            headers={"Content-Length": str(LIMIT + 1)},
        )
    assert response.status_code == 413


def test_requests_without_a_body_are_untouched():
    app = _build_app()

    @app.get("/v1/ping")
    async def ping():
        return {"ok": True}

    with TestClient(app) as client:
        assert client.get("/v1/ping").status_code == 200


STEM = "img_" + "0" * 32


@pytest.mark.parametrize("suffix", [".png", ".mp4", ".m4a", ".3gp", ".x-m4a", ".tar.gz", ".JPG"])
def test_every_extension_this_server_can_produce_is_servable(tmp_path, suffix):
    """The extension comes from whatever the upstream saved, not from a fixed list.

    Rejecting an unusual one would 404 a file whose token verifies, so the pattern has to be
    permissive about the extension while still admitting no path separator.
    """
    target = tmp_path / f"{STEM}{suffix}"
    target.write_bytes(b"data")
    assert _resolve_media_file(tmp_path, target.name) == target.resolve()


@pytest.mark.parametrize(
    "filename",
    [
        "../config/config.yaml",
        f"..{os.sep}{STEM}.png",
        f"{STEM}.png{os.sep}..{os.sep}secret",
        "secret.png",
        "img_notahexdigest.png",
        f"{STEM}.",
        f"{STEM}.png/../../etc/passwd",
    ],
)
def test_names_this_server_never_generates_are_refused(tmp_path, filename):
    assert _resolve_media_file(tmp_path, filename) is None


def test_a_matching_name_with_no_file_behind_it_is_refused(tmp_path):
    assert _resolve_media_file(tmp_path, f"{STEM}.png") is None
