"""Corrections that `app.services.client` applies to gemini-webapi when it is imported."""

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any

import pytest
from curl_cffi.requests import Cookies
from gemini_webapi import GeminiClient
from gemini_webapi.constants import GRPC
from gemini_webapi.exceptions import AuthError
from gemini_webapi.types import RPCData

from app.services.client import GeminiClientWrapper

token_module = importlib.import_module("gemini_webapi.utils.get_access_token")


class _Offline:
    """The session the library's init opens, with no network behind it."""

    def __init__(self, **kwargs: Any) -> None:
        self.cookies = Cookies()

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(status_code=503)  # no preflight cookies, so no guest attempt

    async def close(self) -> None:
        pass


def test_the_link_google_leaves_for_a_generated_image_is_removed_whole():
    """Parsed by the library itself, so this also fails if it stops reading the pattern that
    is replaced."""
    raw = "Here it is:\n\nhttp://googleusercontent.com/image_generation_content/0_974\n\nEnjoy."

    text = GeminiClient("psid", "psidts")._parse_candidate([None, [raw]], "c", "r", "rc")[0]

    assert text == "Here it is:\n\nEnjoy."


def test_the_sign_in_page_is_not_taken_for_a_signed_in_session(tmp_path, monkeypatch):
    """Google sends cookies it no longer accepts to its sign-in page, which carries the same init
    values as Gemini. Driven through the library's own init, with every request answered here."""

    def init_landing_on(url: str) -> Any:
        page = SimpleNamespace(url=url, text='"SNlM0e":"token"')

        async def _send_request(*args: Any, **kwargs: Any) -> Any:
            return page

        monkeypatch.setattr(token_module, "_send_request", _send_request)
        return asyncio.run(token_module.get_access_token({"__Secure-1PSID": "psid"}))

    monkeypatch.setenv("GEMINI_COOKIE_PATH", str(tmp_path))  # no cached cookies
    monkeypatch.setattr(token_module, "AsyncSession", _Offline)

    assert init_landing_on("https://gemini.google.com/app").access_token == "token"
    with pytest.raises(AuthError):
        init_landing_on("https://accounts.google.com/v3/signin/identifier")


def test_the_account_status_is_asked_for_the_way_the_web_app_asks(monkeypatch):
    sent: list[RPCData] = []

    async def _batch_execute(self: GeminiClient, payloads: list[RPCData], **kwargs: Any) -> Any:
        sent.extend(payloads)
        return SimpleNamespace(text="")

    monkeypatch.setattr(GeminiClient, "_batch_execute", _batch_execute)

    asyncio.run(GeminiClientWrapper(client_id="a")._fetch_user_status())

    assert [(p.rpcid, p.payload) for p in sent] == [(GRPC.GET_USER_STATUS, "[1]")]


def test_models_are_discovered_under_their_english_names(monkeypatch):
    """Gemini names models in the language of the session, and the library takes the names it is
    given. A full set of browser cookies carries the account's language."""
    asked_in: list[str] = []

    async def _init_rpc(self: GeminiClient) -> None:
        asked_in.append(self.language)

    monkeypatch.setattr(GeminiClient, "_init_rpc", _init_rpc)
    client = GeminiClientWrapper(client_id="a")
    client.language = "zh-CN"  # what the page of a Chinese-language account reports

    asyncio.run(client._init_rpc())

    assert asked_in == ["en"]


def test_a_client_without_cookies_does_not_sign_in_with_another_accounts_cache(
    tmp_path, monkeypatch
):
    """With no __Secure-1PSID the library would use the most recently cached cookies."""
    (tmp_path / ".cached_cookies_other.json").write_text(
        '[{"name": "__Secure-1PSID", "value": "other", "domain": ".google.com", "path": "/"}]'
    )
    monkeypatch.setenv("GEMINI_COOKIE_PATH", str(tmp_path))
    sent: list[list[str]] = []

    async def _send_request(client: Any, jar: Cookies, **kwargs: Any) -> Any:
        sent.append([cookie.name for cookie in jar.jar])
        return SimpleNamespace(url="https://gemini.google.com/app", text='"SNlM0e":"token"')

    monkeypatch.setattr(token_module, "AsyncSession", _Offline)
    monkeypatch.setattr(token_module, "_send_request", _send_request)

    with pytest.raises(AuthError):
        asyncio.run(token_module.get_access_token({}))
    assert sent == []
