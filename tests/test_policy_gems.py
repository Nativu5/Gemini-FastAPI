"""Tests for managed policy gems configuration, lifecycle, and system-prompt routing."""

import asyncio
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import Request
from gemini_webapi import ModelOutput
from gemini_webapi.constants import AccountStatus
from gemini_webapi.types import Gem, GemJar
from gemini_webapi.types.candidate import Candidate

from app.models import AppContentItem, AppMessage, ChatCompletionMessage, ChatCompletionRequest
from app.server import chat
from app.services import client as client_module
from app.services.client import GeminiClientWrapper
from app.services.lmdb import LMDBConversationStore
from app.services.policy_gems import (
    extract_managed_last_used_at,
    has_managed_marker,
    sync_policy_gems,
    touch_managed_description,
)
from app.utils.config import GeminiConfig, GeminiGemPolicyMode, GeminiGemsConfig


class _FakeGemClient:
    """In-memory stub implementing GeminiClient gem RPCs for unit tests."""

    def __init__(self, initial_gems: list[Gem] | None = None):
        self._store: dict[str, Gem] = {}
        for gem in initial_gems or []:
            self._store[gem.id] = gem
        self._gems = GemJar(self._store)
        self._seq = 0
        self.created: list[str] = []
        self.updated: list[str] = []
        self.deleted: list[str] = []
        self.fetch_calls: list[bool] = []

    @property
    def gems(self) -> GemJar:
        return self._gems

    async def fetch_gems(self, include_hidden: bool = False, **kwargs: Any) -> GemJar:
        self.fetch_calls.append(include_hidden)
        self._gems = GemJar(self._store)
        return self._gems

    async def create_gem(self, name: str, prompt: str, description: str = "") -> Gem:
        self._seq += 1
        gem_id = f"gem_{self._seq}"
        gem = Gem(id=gem_id, name=name, prompt=prompt, description=description, predefined=False)
        self._store[gem_id] = gem
        self.created.append(name)
        return gem

    async def update_gem(
        self, gem: Gem | str, name: str, prompt: str, description: str = ""
    ) -> Gem:
        gem_id = gem.id if isinstance(gem, Gem) else gem
        updated = Gem(
            id=gem_id, name=name, prompt=prompt, description=description, predefined=False
        )
        self._store[gem_id] = updated
        self.updated.append(gem_id)
        return updated

    async def delete_gem(self, gem: Gem | str, **kwargs: Any) -> None:
        gem_id = gem.id if isinstance(gem, Gem) else gem
        self._store.pop(gem_id, None)
        self.deleted.append(gem_id)


def test_gem_config_defaults_are_safe_and_consistent():
    config = GeminiConfig(clients=[], auto_refresh=True, verbose=True)
    assert config.gems.enabled is False
    assert config.gems.policy == GeminiGemPolicyMode.OFF
    assert config.gems.create_rate_limit_per_minute == 4
    assert config.gems.managed_gems_max_total == 200
    assert config.gems.policies.default_policy.enabled is False
    assert config.gems.cleanup.enabled is False


def test_sync_policy_gems_skips_default_policy_when_disabled():
    fake = _FakeGemClient()
    result = asyncio.run(
        sync_policy_gems(
            cast(Any, fake),
            default_policy_enabled=False,
            mode="create_on_demand",
        )
    )
    assert result.gem_ids == {}
    assert result.created_count == 0
    assert fake.created == []


def test_sync_policy_gems_creates_and_updates_when_enabled():
    fake = _FakeGemClient()
    res1 = asyncio.run(
        sync_policy_gems(
            cast(Any, fake),
            prefix="fastapi_policy_",
            default_policy_enabled=True,
            default_policy_key="custom_guard",
            default_prompt="Always answer concisely.",
            mode="create_on_demand",
        )
    )
    assert res1.created_count == 1
    assert "custom_guard" in res1.gem_ids
    gem_id = res1.gem_ids["custom_guard"]
    assert has_managed_marker(fake._store[gem_id].description)
    assert extract_managed_last_used_at(fake._store[gem_id].description) is not None

    # Second run with identical prompt makes no edits.
    res2 = asyncio.run(
        sync_policy_gems(
            cast(Any, fake),
            prefix="fastapi_policy_",
            default_policy_enabled=True,
            default_policy_key="custom_guard",
            default_prompt="Always answer concisely.",
            mode="create_on_demand",
        )
    )
    assert res2.created_count == 0
    assert res2.updated_count == 0

    # Changing prompt updates the existing gem in place.
    res3 = asyncio.run(
        sync_policy_gems(
            cast(Any, fake),
            prefix="fastapi_policy_",
            default_policy_enabled=True,
            default_policy_key="custom_guard",
            default_prompt="Updated prompt.",
            mode="create_on_demand",
        )
    )
    assert res3.updated_count == 1
    assert fake._store[gem_id].prompt == "Updated prompt."


def test_sync_policy_gems_cleanup_respects_marker_and_returns_deleted_ids():
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
    fake = _FakeGemClient([stale_marked, unmarked])

    res = asyncio.run(
        sync_policy_gems(
            cast(Any, fake),
            prefix="fastapi_policy_",
            default_policy_enabled=False,
            mode="fetch_only",
            cleanup_unused_days=7,
            cleanup_require_managed_marker=True,
        )
    )
    assert res.deleted_count == 1
    assert res.deleted_ids == ["gem_stale"]
    assert res.skipped_missing_marker_count == 1
    assert "gem_unmarked" in fake._store
    assert "gem_stale" not in fake._store


def test_wrapper_guest_session_skips_gem_operations(monkeypatch):
    wrapper = GeminiClientWrapper(client_id="guest-1")
    wrapper._running = True
    monkeypatch.setattr(wrapper, "is_guest", lambda: True)

    gems_cfg = GeminiGemsConfig(enabled=True, policy=GeminiGemPolicyMode.CREATE_ON_DEMAND)
    monkeypatch.setattr(client_module.g_config.gemini, "gems", gems_cfg)

    async def _fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Guest session must not call gem RPCs")

    monkeypatch.setattr(wrapper, "fetch_gems", _fail)
    monkeypatch.setattr(wrapper, "create_gem", _fail)

    asyncio.run(wrapper._initialize_gems())
    assert wrapper.policy_gem_id("general_capability_guardrail") is None
    assert asyncio.run(wrapper.policy_gem_id_or_create("general_capability_guardrail")) is None
    assert asyncio.run(wrapper.system_prompt_gem_id_or_create("You are helpful.")) is None
    with pytest.raises(RuntimeError, match="authenticated"):
        asyncio.run(wrapper.refresh_gems())


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
        enabled=True,
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
        assert wrapper.managed_gem_metrics()["managed_gems_rate_limit_skips"] == 1

        wrapper._prune_deleted_system_gem_ids(["g1"])
        assert wrapper._system_prompt_gem_ids == {}
        await wrapper.close()

    asyncio.run(_run())


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
            enabled=True,
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
