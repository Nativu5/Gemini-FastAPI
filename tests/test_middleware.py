"""Transport-layer behaviour: the request body ceiling, and serving generated media.

The limit is enforced twice over: once from a declared `content-length`, and again by counting
a chunked body as it arrives. Both have to produce the error shape of the surface they were
addressed to, and both have to travel back out through CORS.

Media resolution is the other half: the name has to be one this server generated - which rules
out traversal - without being so narrow that a legitimate extension becomes an unreachable file.
"""

import os
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.main import create_app
from app.server.media import _resolve_media_file
from app.server.middleware import RequestBodyLimitMiddleware
from app.utils import g_config
from app.utils.config import CORSConfig

LIMIT = 100


def _build_app(max_body_bytes: int = LIMIT) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/echo")
    async def echo(request: Request):
        return {"received": len(await request.body())}

    @app.post("/v1beta/models/x:generateContent")
    async def gemini_echo(request: Request):
        return {"received": len(await request.body())}

    app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=max_body_bytes)
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


def test_a_refusal_still_carries_cors_headers(monkeypatch):
    """Without this the browser reports an opaque CORS failure instead of the 413.

    It depends on the order in which `create_app` adds its middleware, so it runs the real app.
    """
    monkeypatch.setattr(g_config.server, "max_request_body_bytes", LIMIT)
    monkeypatch.setattr(g_config, "cors", CORSConfig(allow_origins=["https://example.com"]))
    client = TestClient(create_app())  # no `with`: the lifespan would log in to Google

    for body in (b"x" * (LIMIT + 1), _chunks(LIMIT * 5)):  # declared, then counted
        refused = client.post(
            "/v1/chat/completions", content=body, headers={"Origin": "https://example.com"}
        )

        assert refused.status_code == 413
        assert refused.json()["error"]["type"] == "invalid_request_error"
        assert refused.headers["access-control-allow-origin"] == "https://example.com"


def test_an_unparsable_content_length_falls_back_to_counting():
    """A bogus header must not be trusted as 0 and wave an oversize body through."""
    with TestClient(_build_app()) as client:
        response = client.post(
            "/v1/echo", content=b"x" * (LIMIT + 1), headers={"Content-Length": "abc"}
        )
    assert response.status_code == 413


HEX = "0" * 32
STEM = f"img_{HEX}"


@pytest.mark.parametrize(
    "name",
    [
        f"img_{HEX}.png",
        f"media_{HEX}.mp4",  # videos and audio are stored under this prefix
        f"media_{HEX}.x-m4a",
        f"media_{HEX}.3gp",
        f"img_{HEX}.tar.gz",
        f"img_{HEX}.JPG",
    ],
)
def test_every_name_this_server_can_produce_is_servable(tmp_path, name):
    """The extension comes from whatever the upstream saved, not from a fixed list.

    Rejecting an unusual one would 404 a file whose token verifies, so the pattern has to be
    permissive about the extension while still admitting no path separator.
    """
    target = tmp_path / name
    target.write_bytes(b"data")
    assert _resolve_media_file(tmp_path, name) == target.resolve()


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
    """Each name leads to a file that exists, so only the check stands in the way."""
    store = tmp_path / "store"
    (store / f"{STEM}.png").mkdir(parents=True)  # lets a name step through it with `..`
    target = Path(os.path.normpath(store / filename))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not yours")

    assert _resolve_media_file(store, filename) is None


def test_a_link_that_leads_out_of_the_store_is_not_followed(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (tmp_path / "outside.png").write_bytes(b"not yours")
    (store / f"{STEM}.png").symlink_to(tmp_path / "outside.png")

    assert _resolve_media_file(store, f"{STEM}.png") is None


def test_a_matching_name_with_no_file_behind_it_is_refused(tmp_path):
    assert _resolve_media_file(tmp_path, f"{STEM}.png") is None
