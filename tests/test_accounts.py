"""How a configured account becomes a client: its cookies, and what happens when Google
rejects them."""

import asyncio
import contextlib
import sys
from collections import deque
from types import SimpleNamespace

import pytest
from conftest import MODEL, reply
from gemini_webapi import GeminiClient
from gemini_webapi.constants import AccountStatus
from gemini_webapi.exceptions import AuthError, GeminiError
from gemini_webapi.types import RPCData
from loguru import logger

from app.services.client import GeminiClientWrapper
from app.services.pool import GeminiClientPool
from app.utils import setup_logging
from app.utils.config import initialize_config

USER = {"role": "user", "content": "Hello"}


def test_the_cookies_of_gemini_alone_stay_on_its_host():
    header = "__Secure-1PSID=a; NID=511=b; OSID=c; __Secure-OSID=d"

    cookies = GeminiClientWrapper(client_id="a", cookies=header).cookies

    assert {c.name: (c.domain, c.value) for c in cookies.jar} == {
        "__Secure-1PSID": (".google.com", "a"),
        "NID": (".google.com", "511=b"),
        "OSID": ("gemini.google.com", "c"),
        "__Secure-OSID": ("gemini.google.com", "d"),
    }


def _client(
    cookies: str | None, status: AccountStatus, client_id: str = "a"
) -> GeminiClientWrapper:
    client = GeminiClientWrapper(client_id=client_id, cookies=cookies)
    client._running = True
    client.account_status = status
    return client


@pytest.mark.parametrize(("cookies", "fails"), [("__Secure-1PSID=psid", True), (None, False)])
def test_a_refused_heartbeat_stops_a_client_with_cookies_but_not_a_guest(
    monkeypatch, cookies, fails
):
    """The library sends its heartbeat at startup and before every message, and ignores the
    answer, which is where Google first refuses cookies it no longer accepts."""
    refused = ')]}\'\n\n[["wrb.fr","ESY5D",null,null,null,[7],"generic"]]'

    async def _batch_execute(self: GeminiClient, payloads: list[RPCData], **kwargs: object):
        return SimpleNamespace(text=refused)

    monkeypatch.setattr(GeminiClient, "_batch_execute", _batch_execute)
    client = _client(cookies, AccountStatus.AVAILABLE)

    with pytest.raises(AuthError) if fails else contextlib.nullcontext():
        asyncio.run(client._sync_activity())
    assert client.account_status == AccountStatus.UNAUTHENTICATED


def _pool(*clients: GeminiClientWrapper) -> GeminiClientPool:
    pool = GeminiClientPool.__new__(GeminiClientPool)  # skips the singleton and the real config
    pool.clients = list(clients)
    pool._id_map = {client.id: client for client in clients}
    pool._round_robin = deque(clients)
    return pool


def test_a_client_google_signed_out_is_restarted_and_not_used_as_a_guest(monkeypatch):
    client = _client("__Secure-1PSID=psid", AccountStatus.UNAUTHENTICATED)
    restarts = _rejected_on_restart(monkeypatch, client)

    with pytest.raises(RuntimeError):
        asyncio.run(_pool(client).acquire())
    assert restarts == ["a"]
    assert not client.running()


def _rejected_on_restart(monkeypatch, client: GeminiClientWrapper) -> list[str]:
    restarts: list[str] = []

    async def _rejected_again() -> None:
        restarts.append(client.id)
        raise AuthError("rejected")

    monkeypatch.setattr(client, "init", _rejected_again)
    return restarts


def test_a_client_with_an_account_is_preferred_and_an_upload_never_goes_to_a_guest(monkeypatch):
    guest = _client(None, AccountStatus.UNAUTHENTICATED, "guest")
    account = _client("__Secure-1PSID=psid", AccountStatus.AVAILABLE, "account")
    pool = _pool(guest, account)

    assert asyncio.run(pool.acquire()) is account

    account.account_status = AccountStatus.UNAUTHENTICATED
    _rejected_on_restart(monkeypatch, account)
    with pytest.raises(RuntimeError):
        asyncio.run(pool.acquire(require_account=True))
    assert asyncio.run(pool.acquire()) is guest


def test_a_client_configured_as_a_guest_serves_and_is_healthy():
    client = _client(None, AccountStatus.UNAUTHENTICATED)

    assert asyncio.run(_pool(client).acquire()) is client
    assert client.is_healthy()
    assert not _client("__Secure-1PSID=psid", AccountStatus.UNAUTHENTICATED).is_healthy()


def test_a_restart_in_progress_is_waited_for_not_closed_under(monkeypatch):
    """Two requests reach a signed-out client together: the second must not close the restart the
    first started, whose request would then never finish."""
    client = _client("__Secure-1PSID=psid", AccountStatus.UNAUTHENTICATED)
    restarting = False
    closed_while_restarting: list[bool] = []

    async def _restart() -> None:
        nonlocal restarting
        async with client._lock:  # as the library's `init` holds it
            restarting = True
            await asyncio.sleep(0.01)
            restarting = False
            client._running = True
            client.account_status = AccountStatus.AVAILABLE

    async def _close(*args: object) -> None:
        closed_while_restarting.append(restarting)
        client._running = False

    monkeypatch.setattr(client, "init", _restart)
    monkeypatch.setattr(client, "close", _close)
    pool = _pool(client)

    async def _two_requests() -> list[GeminiClientWrapper]:
        return list(await asyncio.gather(pool.acquire(), pool.acquire()))

    assert asyncio.run(_two_requests()) == [client, client]
    assert closed_while_restarting == [False]


@pytest.mark.parametrize("gemini", [reply("Hello, visitor."), GeminiError("Permission denied")])
def test_a_message_on_a_client_signed_out_meanwhile_is_a_503(api, gemini):
    """Another request's heartbeat, or the library retrying without one, can find the cookies
    rejected while this message is under way."""
    api.upstream.signed_out = True
    api.upstream.replies.append(gemini)

    response = api.post("/v1/chat/completions", json={"model": MODEL, "messages": [USER]})

    assert response.status_code == 503


def test_a_message_google_refuses_for_rejected_cookies_is_a_503(api):
    api.upstream.replies.append(AuthError("Google no longer accepts the cookies of client a"))

    response = api.post("/v1/chat/completions", json={"model": MODEL, "messages": [USER]})

    assert response.status_code == 503


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("SECURE_1PSID", "removed-key-secret"),
        ("COOKIES", "SID=no-1psid-secret"),
        ("COOKIES", "Cookie: __Secure-1PSID=with-its-name-secret"),
    ],
)
def test_cookies_the_server_cannot_use_stop_it_without_being_logged(monkeypatch, key, value):
    """A removed key would leave the client without cookies, and a header without
    __Secure-1PSID signs no one in: either way the client would run as a guest."""
    monkeypatch.setenv(f"CONFIG_GEMINI__CLIENTS__0__{key}", value)
    logged: list[str] = []
    sink = logger.add(logged.append)
    try:
        with pytest.raises(SystemExit):
            initialize_config()
    finally:
        logger.remove(sink)

    assert key.lower() in "".join(logged)
    assert value not in "".join(logged)


def test_a_logged_traceback_does_not_show_the_cookies_it_passed_through(capfd):
    """A failed start is logged with its traceback, and the cookie jar is a local of its frames."""
    setup_logging("ERROR")
    try:

        def _start(cookies: str) -> None:
            raise RuntimeError("Google refused")

        cookies = "__Secure-1PSID=" + "-".join(["frame", "local", "secret"])  # not in the source
        try:
            _start(cookies)
        except RuntimeError:
            logger.exception("Failed to initialize GeminiClient a")
        logger.complete()
    finally:
        logger.remove()
        logger.add(sys.stderr)

    logged = capfd.readouterr().err
    assert "Google refused" in logged
    assert "frame-local-secret" not in logged
