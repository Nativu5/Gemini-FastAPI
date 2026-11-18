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
from app.services.client import GeminiClientWrapper
from app.services.lmdb import LMDBConversationStore
from app.services.policy_gems import (
    delete_unused_gems,
    extract_managed_last_used_at,
    has_managed_marker,
    touch_managed_description,
)
from app.services.pool import GeminiClientPool
from app.utils.config import (
    GeminiConfig,
    GeminiGemCleanupConfig,
    GeminiGemPolicyMode,
    GeminiGemsConfig,
)


class _FakeGemClient:
    """In-memory stand-in for the gem RPCs the cleanup uses."""

    def __init__(self, gems: list[Gem]):
        self._store = {gem.id: gem for gem in gems}

    async def fetch_gems(self, include_hidden: bool = False, **kwargs: Any) -> GemJar:
        return GemJar(self._store)

    async def delete_gem(self, gem: Gem | str, **kwargs: Any) -> None:
        self._store.pop(gem.id if isinstance(gem, Gem) else gem, None)


def test_gem_config_defaults_are_safe_and_consistent():
    config = GeminiConfig(clients=[], auto_refresh=True, verbose=True)
    assert config.gems.policy == GeminiGemPolicyMode.OFF
    assert config.gems.create_rate_limit_per_minute == 4
    assert config.gems.managed_gems_max_total == 200
    assert config.gems.cleanup.enabled is False


def test_cleanup_deletes_only_stale_gems_that_carry_our_marker():
    old_ts = int(time.time() - 10 * 24 * 60 * 60)
    stale_marked = Gem(
        id="gem_stale",
        name="fastapi_policy_sys_old",
        description=touch_managed_description("Managed", now_ts=old_ts),
        prompt="old",
        predefined=False,
    )
    unmarked = Gem(
        id="gem_unmarked",
        name="fastapi_policy_manual",
        description="User-created gem sharing prefix",
        prompt="keep me",
        predefined=False,
    )
    fresh_marked = Gem(
        id="gem_fresh",
        name="fastapi_policy_sys_new",
        description=touch_managed_description("Managed", now_ts=time.time()),
        prompt="new",
        predefined=False,
    )
    fake = _FakeGemClient([stale_marked, unmarked, fresh_marked])

    deleted = asyncio.run(
        delete_unused_gems(cast(Any, fake), prefix="fastapi_policy_", unused_days=7)
    )

    assert deleted == ["gem_stale"]
    assert set(fake._store) == {"gem_unmarked", "gem_fresh"}


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


def test_wrapper_system_prompt_gem_dedup_rate_limit_and_cleanup_prune(monkeypatch):
    wrapper = GeminiClientWrapper(client_id="auth-1")
    wrapper._running = True
    wrapper.account_status = AccountStatus.AVAILABLE

    store: dict[str, Gem] = {}
    seq = 0

    async def _fetch_gems(include_hidden: bool = False, **kwargs: Any) -> GemJar:
        wrapper._gems = GemJar(store)
        return wrapper._gems

    async def _create_gem(name: str, prompt: str, description: str = "") -> Gem:
        nonlocal seq
        seq += 1
        gem = Gem(id=f"g{seq}", name=name, prompt=prompt, description=description, predefined=False)
        store[gem.id] = gem
        return gem

    monkeypatch.setattr(wrapper, "fetch_gems", _fetch_gems)
    monkeypatch.setattr(wrapper, "create_gem", _create_gem)

    gems_cfg = GeminiGemsConfig(
        policy=GeminiGemPolicyMode.CREATE_ON_DEMAND,
        create_rate_limit_per_minute=1,
        managed_gems_max_total=10,
    )
    monkeypatch.setattr(client_module.g_config.gemini, "gems", gems_cfg)

    async def _run() -> None:
        id1 = await wrapper.system_prompt_gem_id_or_create("System prompt A")
        id1_again = await wrapper.system_prompt_gem_id_or_create("System prompt A")
        # Rate limit of 1/min blocks a second distinct prompt in the same minute.
        id2 = await wrapper.system_prompt_gem_id_or_create("System prompt B")
        assert id1 == "g1"
        assert id1_again == "g1"
        assert id2 is None
        assert has_managed_marker(store["g1"].description)

        wrapper._prune_deleted_system_gem_ids(["g1"])
        assert wrapper._system_prompt_gem_ids == {}
        await wrapper.close()

    asyncio.run(_run())


def test_a_failed_last_use_update_is_tried_again_on_the_next_use(monkeypatch):
    wrapper = GeminiClientWrapper(client_id="auth-1")
    wrapper._running = True
    wrapper.account_status = AccountStatus.AVAILABLE

    store = {
        "g1": Gem(
            id="g1",
            name="fastapi_policy_sys_abc",
            prompt="p",
            description=touch_managed_description("Managed", now_ts=1),
            predefined=False,
        )
    }
    attempts: list[str] = []

    async def _fetch_gems(include_hidden: bool = False, **kwargs: Any) -> GemJar:
        return GemJar(store)

    async def _update_gem(gem: Gem | str, name: str, prompt: str, description: str = "") -> Gem:
        attempts.append(description)
        if len(attempts) == 1:
            raise RuntimeError("Google said no")
        store["g1"] = Gem(
            id="g1", name=name, prompt=prompt, description=description, predefined=False
        )
        return store["g1"]

    monkeypatch.setattr(wrapper, "fetch_gems", _fetch_gems)
    monkeypatch.setattr(wrapper, "update_gem", _update_gem)
    gems_cfg = GeminiGemsConfig(
        policy=GeminiGemPolicyMode.CREATE_ON_DEMAND,
        cleanup=GeminiGemCleanupConfig(enabled=True),
    )
    monkeypatch.setattr(client_module.g_config.gemini, "gems", gems_cfg)

    async def _use_the_gem_twice() -> None:
        for _ in range(2):
            wrapper._managed_gem_pending_touch_ids.add("g1")
            await wrapper._flush_managed_policy_touches()

    asyncio.run(_use_the_gem_twice())

    assert len(attempts) == 2
    assert (extract_managed_last_used_at(store["g1"].description) or 0) > 1


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

        # The background last-use update takes the same lock; it logs the failure and ends.
        await wrapper.init()
        wrapper._schedule_managed_policy_touch("gem_used")
        worker = wrapper._managed_gem_touch_worker_task
        assert worker is not None
        await asyncio.wait_for(worker, timeout=2)

    asyncio.run(_run())


def test_gem_cleanup_reaches_every_account_even_after_one_fails(monkeypatch):
    cleaned: list[str] = []

    async def _cleanup(client_id: str) -> None:
        if client_id == "broken":
            raise APIError("rejected")
        cleaned.append(client_id)

    pool = GeminiClientPool.__new__(GeminiClientPool)  # skips the singleton and the real config
    pool._clients = [
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
    pool._restart_locks = {}

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


def test_extract_leading_system_prompt_handles_edge_cases():
    msgs = [
        AppMessage(role="system", content="First"),
        AppMessage(role="system", content="Second"),
        AppMessage(role="user", content="Hi"),
        AppMessage(role="system", content="Late system"),
    ]
    extracted, rest = chat._extract_leading_system_prompt(msgs)
    assert extracted == "First\n\nSecond"
    assert [m.role for m in rest] == ["user", "system"]

    # Multimodal leading system message must not be stripped.
    multimodal = [
        AppMessage(
            role="system",
            content=[
                AppContentItem(type="text", text="Describe"),
                AppContentItem(type="image_url", url="https://example.com/a.png"),
            ],
        ),
        AppMessage(role="user", content="Go"),
    ]
    extracted_mm, rest_mm = chat._extract_leading_system_prompt(multimodal)
    assert extracted_mm is None
    assert len(rest_mm) == 2


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

            def can_upload(self) -> bool:
                return True

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
