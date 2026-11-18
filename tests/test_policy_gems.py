"""Tests for managed gems: configuration, cleanup, and system-prompt routing."""

import asyncio
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import gemini_webapi.client as gemini_client_module
import gemini_webapi.utils.decorators as gemini_decorators
import pytest
from fastapi import Request
from gemini_webapi import GeminiClient, ModelOutput
from gemini_webapi.constants import AccountStatus
from gemini_webapi.exceptions import APIError
from gemini_webapi.types import Gem, GemJar
from gemini_webapi.types.candidate import Candidate

from app import main as main_module
from app.models import AppContentItem, AppMessage, ChatCompletionMessage, ChatCompletionRequest
from app.server import chat
from app.services import client as client_module
from app.services import policy_gems as policy_gems_module
from app.services.client import GeminiClientWrapper
from app.services.lmdb import LMDBConversationStore
from app.services.policy_gems import (
    extract_managed_last_used_at,
    touch_managed_description,
)
from app.services.pool import GeminiClientPool
from app.utils.config import (
    GeminiGemCleanupConfig,
    GeminiGemPolicyMode,
    GeminiGemsConfig,
)


def test_wrapper_guest_session_skips_gem_operations(monkeypatch):
    wrapper = GeminiClientWrapper(client_id="guest-1")
    wrapper._running = True
    monkeypatch.setattr(wrapper, "is_guest", lambda: True)

    gems_cfg = GeminiGemsConfig(
        policy=GeminiGemPolicyMode.CREATE_ON_DEMAND,
        cleanup=GeminiGemCleanupConfig(enabled=True),
    )
    monkeypatch.setattr(client_module.g_config.gemini, "gems", gems_cfg)

    async def _fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Guest session must not call gem RPCs")

    monkeypatch.setattr(wrapper, "fetch_gems", _fail)
    monkeypatch.setattr(wrapper, "create_gem", _fail)

    asyncio.run(wrapper.delete_unused_gems())
    assert asyncio.run(wrapper.system_prompt_gem_id_or_create("You are helpful.")) is None


@pytest.fixture
def account(monkeypatch):
    """A logged-in client whose gem requests go to an in-memory account, on a clock the test moves.

    `account.requests` lists every gem request made, `account.gems` is what the account holds.
    """
    wrapper = GeminiClientWrapper(client_id="auth-1")
    wrapper._running = True
    wrapper.account_status = AccountStatus.AVAILABLE
    state = SimpleNamespace(
        client=wrapper, gems={}, requests=[], reject_updates=False, now=time.time()
    )

    async def fetch_gems(include_hidden: bool = False, **kwargs: Any) -> GemJar:
        state.requests.append("fetch")
        return GemJar(state.gems)

    async def create_gem(name: str, prompt: str, description: str = "") -> Gem:
        state.requests.append("create")
        gem = Gem(
            id=f"g{state.requests.count('create')}",
            name=name,
            prompt=prompt,
            description=description,
            predefined=False,
        )
        state.gems[gem.id] = gem
        return gem

    async def update_gem(gem: Gem, name: str, prompt: str, description: str = "") -> Gem:
        state.requests.append("update")
        if state.reject_updates:
            raise APIError("Google said no")
        state.gems[gem.id] = Gem(
            id=gem.id, name=name, prompt=prompt, description=description, predefined=False
        )
        return state.gems[gem.id]

    async def delete_gem(gem: Gem, **kwargs: Any) -> None:
        state.requests.append("delete")
        del state.gems[gem.id]

    for name, fake in (
        ("fetch_gems", fetch_gems),
        ("create_gem", create_gem),
        ("update_gem", update_gem),
        ("delete_gem", delete_gem),
    ):
        monkeypatch.setattr(wrapper, name, fake)
    clock = SimpleNamespace(time=lambda: state.now, monotonic=lambda: state.now)
    monkeypatch.setattr(client_module, "time", clock)
    monkeypatch.setattr(policy_gems_module, "time", clock)
    return state


def _use_gems(monkeypatch, **settings: Any) -> None:
    gems_cfg = GeminiGemsConfig(policy=GeminiGemPolicyMode.CREATE_ON_DEMAND, **settings)
    monkeypatch.setattr(client_module.g_config.gemini, "gems", gems_cfg)


def test_one_gem_serves_every_request_with_a_prompt_and_creation_is_rate_limited(
    account, monkeypatch
):
    _use_gems(monkeypatch, create_rate_limit_per_minute=1)
    lookup = account.client.system_prompt_gem_id_or_create

    async def _run() -> None:
        assert await lookup("System prompt A") == "g1"
        assert await lookup("System prompt A") == "g1"
        assert account.requests == ["fetch", "create"]  # the second lookup asked Google nothing
        assert extract_managed_last_used_at(account.gems["g1"].description) == int(account.now)

        assert await lookup("System prompt B") is None  # one creation a minute, already used
        account.now += 61
        assert await lookup("System prompt B") == "g2"

    asyncio.run(_run())


def test_a_gem_in_use_has_its_last_use_recorded_once_per_interval(account, monkeypatch):
    _use_gems(monkeypatch, cleanup=GeminiGemCleanupConfig(enabled=True, touch_interval_minutes=60))
    lookup = account.client.system_prompt_gem_id_or_create

    async def _run() -> None:
        await lookup("System prompt A")
        await lookup("System prompt A")
        assert "update" not in account.requests  # created a moment ago: nothing to record yet

        account.now += 61 * 60
        await lookup("System prompt A")
        await lookup("System prompt A")
        assert account.requests.count("update") == 1
        assert extract_managed_last_used_at(account.gems["g1"].description) == int(account.now)

    asyncio.run(_run())


def test_a_rejected_last_use_update_neither_fails_the_request_nor_repeats_at_once(
    account, monkeypatch
):
    _use_gems(monkeypatch, cleanup=GeminiGemCleanupConfig(enabled=True, touch_interval_minutes=60))
    lookup = account.client.system_prompt_gem_id_or_create

    async def _run() -> None:
        await lookup("System prompt A")
        account.reject_updates = True
        account.now += 61 * 60
        assert await lookup("System prompt A") == "g1"
        assert await lookup("System prompt A") == "g1"
        assert account.requests.count("update") == 1  # not retried on the very next request

        account.reject_updates = False
        account.now += 61 * 60
        await lookup("System prompt A")
        assert extract_managed_last_used_at(account.gems["g1"].description) == int(account.now)

    asyncio.run(_run())


def _gem(gem_id: str, name: str, description: str) -> Gem:
    return Gem(id=gem_id, name=name, description=description, prompt="p", predefined=False)


def test_cleanup_deletes_only_gems_this_server_recorded_as_unused(account, monkeypatch):
    days = 24 * 60 * 60
    used_at = {
        "stale": account.now - 10 * days,
        "stale_too": account.now - 9 * days,
        "fresh": account.now - days,
    }
    account.gems = {
        gem_id: _gem(
            gem_id,
            f"fastapi_policy_sys_{gem_id}",
            touch_managed_description("Managed", now_ts=when),
        )
        for gem_id, when in used_at.items()
    }
    # Made by a person under the same prefix, or outside it: never this server's to delete.
    account.gems["manual"] = _gem("manual", "fastapi_policy_manual", "Mine, keep it")
    account.gems["other"] = _gem("other", "my_gem", touch_managed_description("x", now_ts=1))

    _use_gems(monkeypatch, cleanup=GeminiGemCleanupConfig(enabled=True, dry_run=True))
    asyncio.run(account.client.delete_unused_gems())
    assert len(account.gems) == 5  # a dry run only reports

    _use_gems(monkeypatch, cleanup=GeminiGemCleanupConfig(enabled=True, max_deletes_per_run=1))
    asyncio.run(account.client.delete_unused_gems())
    assert sorted(account.gems) == ["fresh", "manual", "other", "stale_too"]  # one per run

    # With room to delete more, the two that are not this server's are still left alone.
    _use_gems(monkeypatch, cleanup=GeminiGemCleanupConfig(enabled=True))
    asyncio.run(account.client.delete_unused_gems())
    assert sorted(account.gems) == ["fresh", "manual", "other"]


def test_a_gem_that_cleanup_deleted_is_not_handed_out_again(account, monkeypatch):
    _use_gems(monkeypatch, cleanup=GeminiGemCleanupConfig(enabled=True, unused_days=7))
    lookup = account.client.system_prompt_gem_id_or_create

    async def _run() -> None:
        assert await lookup("System prompt A") == "g1"
        account.now += 8 * 24 * 60 * 60
        await account.client.delete_unused_gems()
        assert account.gems == {}

        assert await lookup("System prompt A") == "g2"

    asyncio.run(_run())


def test_a_gem_request_that_google_rejects_fails_instead_of_hanging(monkeypatch):
    """When Google rejects a request the library closes the client and starts it again before it
    retries, all inside the call that holds the gem lock. Starting must not wait for that lock."""
    wrapper = GeminiClientWrapper(client_id="auth-1")
    wrapper.account_status = AccountStatus.AVAILABLE
    gems_cfg = GeminiGemsConfig(
        policy=GeminiGemPolicyMode.CREATE_ON_DEMAND,
        cleanup=GeminiGemCleanupConfig(enabled=True),
    )
    monkeypatch.setattr(client_module.g_config.gemini, "gems", gems_cfg)

    class _RejectingSession:
        cookies = MappingProxyType({})  # read by the library when it closes

        async def post(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(status_code=500, http_version=None)

        async def close(self) -> None:
            pass

    async def _library_init(self: GeminiClient, **kwargs: Any) -> None:
        self.client = cast(Any, _RejectingSession())
        self._running = True

    monkeypatch.setattr(GeminiClient, "init", _library_init)
    monkeypatch.setattr(gemini_client_module, "save_cookies", lambda *args, **kwargs: None)
    monkeypatch.setattr(gemini_decorators, "_DELAY_FACTOR", 0)  # no pause between retries

    async def _run() -> None:
        await asyncio.wait_for(wrapper.init(), timeout=2)
        for call in (
            wrapper.system_prompt_gem_id_or_create("You are helpful."),
            wrapper.delete_unused_gems(),
        ):
            with pytest.raises(APIError):
                await asyncio.wait_for(call, timeout=2)

    asyncio.run(_run())


def test_gem_cleanup_reaches_every_account_even_after_one_fails(monkeypatch):
    cleaned: list[str] = []

    async def _cleanup(client_id: str) -> None:
        if client_id == "broken":
            raise APIError("rejected")
        cleaned.append(client_id)

    pool = GeminiClientPool.__new__(GeminiClientPool)  # skips the singleton and the real config
    pool.clients = [
        cast(
            Any,
            SimpleNamespace(
                id=client_id,
                running=lambda: True,
                delete_unused_gems=lambda client_id=client_id: _cleanup(client_id),
            ),
        )
        for client_id in ("first", "broken", "last")
    ]

    asyncio.run(pool.delete_unused_gems())
    assert cleaned == ["first", "last"]


def test_gem_cleanup_alone_keeps_the_periodic_task_alive_and_runs_on_its_own_clock(monkeypatch):
    passes = gem_cleanups = 0

    class _Store:
        retention_days = 0

        def get_pending_deletions(self) -> list[Any]:
            return []

        def has_auto_delete_records(self) -> bool:
            return False

        def cleanup_expired(self) -> tuple[int, list[Any]]:
            nonlocal passes
            passes += 1
            return 0, []

    class _Pool:
        async def delete_unused_gems(self) -> None:
            nonlocal gem_cleanups
            gem_cleanups += 1

    gems_cfg = GeminiGemsConfig(
        policy=GeminiGemPolicyMode.FETCH_ONLY,
        cleanup=GeminiGemCleanupConfig(enabled=True),
    )
    monkeypatch.setattr(main_module.g_config.gemini, "gems", gems_cfg)
    monkeypatch.setattr(main_module, "LMDBConversationStore", _Store)
    monkeypatch.setattr(main_module, "GeminiClientPool", _Pool)
    monkeypatch.setattr(main_module, "_get_cleanup_interval", lambda: 0.001)

    async def _run() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(main_module._run_retention_cleanup(stop))
        while passes < 3 and not task.done():
            await asyncio.sleep(0.001)
        stop.set()
        await task

    asyncio.run(_run())
    assert passes >= 3  # nothing else is switched on, so the task would have exited at once
    assert gem_cleanups == 1  # at startup, then not again until six hours have passed


def test_leading_system_messages_are_joined_and_a_later_one_stays_in_place():
    messages = [
        AppMessage(role="system", content="First"),
        AppMessage(role="system", content="Second"),
        AppMessage(role="user", content="Hi"),
        AppMessage(role="system", content="Late system"),
    ]

    extracted, rest = chat._extract_leading_system_prompt(messages)

    assert extracted == "First\n\nSecond"
    assert [m.role for m in rest] == ["user", "system"]


def test_a_system_message_with_an_attachment_is_not_moved_into_a_gem():
    """A gem holds text only, so the attachment would be lost."""
    messages = [
        AppMessage(
            role="system",
            content=[
                AppContentItem(type="text", text="Describe"),
                AppContentItem(type="image_url", url="https://example.com/a.png"),
            ],
        ),
        AppMessage(role="user", content="Go"),
    ]

    extracted, rest = chat._extract_leading_system_prompt(messages)

    assert extracted is None
    assert rest == messages


def test_chat_completion_persists_canonical_history_and_isolates_changed_system_prompt(
    tmp_path, monkeypatch
):
    store = LMDBConversationStore.open_isolated(db_path=str(tmp_path / "lmdb"))
    try:
        gems_cfg = GeminiGemsConfig(
            policy=GeminiGemPolicyMode.CREATE_ON_DEMAND,
        )
        monkeypatch.setattr(chat.g_config.gemini, "gems", gems_cfg)
        monkeypatch.setattr(chat.g_config.gemini, "chat_mode", chat.ChatMode.NORMAL)

        sent_calls: list[dict[str, Any]] = []

        class _FakeSession:
            def __init__(self, gem: str | None = None, metadata: list[str | None] | None = None):
                self.gem = gem
                self.metadata = metadata or ["cid-1", "rid-1", "rcid-1"]
                self.geminiclient = fake_client

            async def send_message(self, prompt: str, **kwargs: Any) -> ModelOutput:
                sent_calls.append(
                    {"prompt": prompt, "gem": self.gem, "metadata": list(self.metadata)}
                )
                output = ModelOutput(
                    metadata=["cid-1", "rid-2", "rcid-2"],
                    candidates=[Candidate(rcid="rcid-2", text="ok")],
                )
                self.metadata = list(output.metadata)
                return output

        class _FakeClient:
            def __init__(self):
                self.id = "client-id-1"
                self.latest_chat_cid = None
                self.gems_by_prompt: dict[str, str] = {}

            def is_guest(self) -> bool:
                return False

            def usable_model(self, model: Any) -> Any:
                return model

            def chat_scope(self, temporary: bool) -> str | None:
                return None

            async def system_prompt_gem_id_or_create(self, prompt: str) -> str:
                return self.gems_by_prompt.setdefault(prompt, f"gem_{len(self.gems_by_prompt) + 1}")

            def start_chat(self, **kwargs: Any) -> _FakeSession:
                return _FakeSession(gem=kwargs.get("gem"), metadata=kwargs.get("metadata"))

        fake_client = _FakeClient()

        class _FakePool:
            async def acquire(
                self, client_id: str | None = None, require_account: bool = False
            ) -> _FakeClient:
                return fake_client

        monkeypatch.setattr(chat, "GeminiClientPool", _FakePool)
        monkeypatch.setattr(chat, "LMDBConversationStore", lambda: store)
        monkeypatch.setattr(chat, "_resolve_model_name", lambda pool, name: "gemini-3-flash")

        raw_request = cast(Request, SimpleNamespace(base_url="http://localhost:8000/", headers={}))

        # Turn 1: system prompt A is sent via gem_1, stripped from the prompt payload.
        req1 = ChatCompletionRequest(
            model="gemini-3-flash",
            messages=[
                ChatCompletionMessage(role="system", content="Be a pirate"),
                ChatCompletionMessage(role="user", content="Hello"),
            ],
        )
        asyncio.run(chat.create_chat_completion(req1, raw_request, api_key="", tmp_dir=tmp_path))
        assert len(sent_calls) == 1
        assert sent_calls[0]["gem"] == "gem_1"
        assert "Be a pirate" not in sent_calls[0]["prompt"]

        # Turn 2 with same system prompt A reuses the LMDB session and sends only the new user turn.
        req2 = ChatCompletionRequest(
            model="gemini-3-flash",
            messages=[
                ChatCompletionMessage(role="system", content="Be a pirate"),
                ChatCompletionMessage(role="user", content="Hello"),
                ChatCompletionMessage(role="assistant", content="ok"),
                ChatCompletionMessage(role="user", content="Next question"),
            ],
        )
        asyncio.run(chat.create_chat_completion(req2, raw_request, api_key="", tmp_dir=tmp_path))
        assert len(sent_calls) == 2
        assert sent_calls[1]["gem"] == "gem_1"
        assert sent_calls[1]["metadata"] == ["cid-1", "rid-2", "rcid-2"]
        assert "Hello" not in sent_calls[1]["prompt"]

        # Turn 2 with changed system prompt B must NOT reuse Turn 1's gem_1 session.
        req3 = ChatCompletionRequest(
            model="gemini-3-flash",
            messages=[
                ChatCompletionMessage(role="system", content="Be a formal lawyer"),
                ChatCompletionMessage(role="user", content="Hello"),
                ChatCompletionMessage(role="assistant", content="ok"),
                ChatCompletionMessage(role="user", content="Next question"),
            ],
        )
        asyncio.run(chat.create_chat_completion(req3, raw_request, api_key="", tmp_dir=tmp_path))
        assert len(sent_calls) == 3
        assert sent_calls[2]["gem"] == "gem_2"
        assert "Hello" in sent_calls[2]["prompt"]
        assert "Be a formal lawyer" not in sent_calls[2]["prompt"]

        # Turn 2 with system prompt removed must also start a fresh session without gem.
        req4 = ChatCompletionRequest(
            model="gemini-3-flash",
            messages=[
                ChatCompletionMessage(role="user", content="Hello"),
                ChatCompletionMessage(role="assistant", content="ok"),
                ChatCompletionMessage(role="user", content="No system now"),
            ],
        )
        asyncio.run(chat.create_chat_completion(req4, raw_request, api_key="", tmp_dir=tmp_path))
        assert len(sent_calls) == 4
        assert sent_calls[3]["gem"] is None
        assert "Hello" in sent_calls[3]["prompt"]
    finally:
        store.close()
