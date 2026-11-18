"""Conversation-store behaviour: lookup, retention, and index versioning.

The store tests open their own store under `tmp_path` via `open_isolated`. The last two cover
what the same cleanup pass does outside the store: expired media files, and the pool's answer for
a chat whose client is no longer configured.
"""

from datetime import datetime, timedelta

import lmdb
import orjson
import pytest
from conftest import CLIENT_ID

from app.models.core import AppContentItem, AppMessage
from app.services.lmdb import LMDBConversationStore
from app.utils.config import ChatMode

MODEL = "gemini-3-pro"


@pytest.fixture
def store(tmp_path):
    opened = LMDBConversationStore.open_isolated(db_path=str(tmp_path / "lmdb"))
    try:
        yield opened
    finally:
        opened.close()


def _exchange(prompt: str = "hello") -> list[AppMessage]:
    return [
        AppMessage(role="user", content=prompt),
        AppMessage(role="assistant", content="hi there"),
    ]


def _raw_keys(opened: LMDBConversationStore) -> list[str]:
    with opened._get_transaction() as txn:
        return [bytes(key).decode("utf-8") for key, _ in txn.cursor()]


def test_a_stored_conversation_is_found_again(store):
    messages = _exchange()
    store.store(CLIENT_ID, MODEL, messages, metadata=["c", "r", "rc"])

    found = store.find(MODEL, messages)
    assert found is not None
    assert found.client_id == CLIENT_ID
    assert found.metadata == ["c", "r", "rc"]


def test_a_different_model_does_not_match(store):
    messages = _exchange()
    store.store(CLIENT_ID, MODEL, messages, metadata=["c", "r", "rc"])
    assert store.find("gemini-3-flash", messages) is None


def test_inline_media_with_the_same_bytes_matches_and_different_bytes_does_not(store):
    """The content digest is what keeps two distinct images from colliding."""

    def with_image(payload: str) -> list[AppMessage]:
        return [
            AppMessage(
                role="user",
                content=[
                    AppContentItem(type="text", text="describe"),
                    AppContentItem(type="image_url", url=f"data:image/png;base64,{payload}"),
                ],
            ),
            AppMessage(role="assistant", content="a picture"),
        ]

    stored = with_image("aGVsbG8=")
    store.store(CLIENT_ID, MODEL, stored, metadata=["c", "r", "rc"])

    assert store.find(MODEL, with_image("aGVsbG8=")) is not None
    assert store.find(MODEL, with_image("d29ybGQ=")) is None


def test_raw_data_still_discriminates_and_ignores_key_order(store):
    """It is hashed inline rather than digested, so the outer sort has to canonicalize it."""

    def with_raw(raw: dict) -> list[AppMessage]:
        return [
            AppMessage(role="user", content=[AppContentItem(type="x", raw_data=raw)]),
            AppMessage(role="assistant", content="ok"),
        ]

    store.store(CLIENT_ID, MODEL, with_raw({"b": 1, "a": 2}), metadata=["c", "r", "rc"])

    assert store.find(MODEL, with_raw({"a": 2, "b": 1})) is not None
    assert store.find(MODEL, with_raw({"a": 2, "b": 99})) is None


def test_keys_reports_conversations_without_index_entries(store):
    store.store(CLIENT_ID, MODEL, _exchange(), metadata=["c", "r", "rc"])

    assert len(store.keys()) == 1
    # The indexes exist, they are just not conversations.
    assert len(_raw_keys(store)) > 1


def test_eviction_removes_the_record_and_its_indexes(store):
    messages = _exchange()
    store.store(CLIENT_ID, MODEL, messages, metadata=["c", "r", "rc"])
    conv = store.find(MODEL, messages)
    assert conv is not None

    assert store.evict(conv) is True
    assert store.find(MODEL, messages) is None
    assert _raw_keys(store) == []


def test_clear_empties_the_store(store):
    store.store(CLIENT_ID, MODEL, _exchange("one"), metadata=["c", "r", "rc"])
    store.store(CLIENT_ID, MODEL, _exchange("two"), metadata=["c", "r", "rc"])

    assert store.clear() == 2
    assert store.keys() == []
    assert _raw_keys(store) == []


def test_retention_keeps_a_conversation_that_is_still_in_use(store):
    """Retention follows last use; a long-running conversation is not old just because it began early."""
    messages = _exchange()
    store.store(CLIENT_ID, MODEL, messages, metadata=["c", "r", "rc"])

    conv = store.find(MODEL, messages)
    assert conv is not None
    key = store.keys()[0]
    conv.created_at = datetime.now() - timedelta(days=90)
    conv.updated_at = datetime.now()
    with store._get_transaction(write=True) as txn:
        txn.put(key.encode("utf-8"), orjson.dumps(conv.model_dump(mode="json")), overwrite=True)

    assert store.cleanup_before(datetime.now() - timedelta(days=14)) == 0
    assert store.find(MODEL, messages) is not None


def test_retention_removes_a_conversation_last_touched_before_the_cutoff(store):
    messages = _exchange()
    store.store(CLIENT_ID, MODEL, messages, metadata=["c", "r", "rc"])

    assert store.cleanup_before(datetime.now() + timedelta(seconds=1)) == 1
    assert store.find(MODEL, messages) is None
    assert _raw_keys(store) == []


def test_indexes_from_a_superseded_version_are_pruned(tmp_path):
    """A hash-shape change strands old entries that eviction can no longer find by hash."""
    db_path = tmp_path / "lmdb"
    env = lmdb.open(str(db_path), map_size=10_000_000, max_dbs=3, writemap=True)
    with env.begin(write=True) as txn:
        txn.put(b"hash:v1:deadbeef", b'["conv1"]')
        txn.put(b"fuzzy:v1:deadbeef", b'["conv1"]')
        txn.put(b"hash:legacy-unversioned", b'["conv1"]')
        txn.put(b"conv1", b'{"client_id":"a","model":"m","messages":[],"metadata":[]}')
    env.close()

    opened = LMDBConversationStore.open_isolated(db_path=str(db_path))
    try:
        assert opened.prune_stale_indexes() == 3
        # The conversation itself survives; only the unreachable lookup rows go.
        assert _raw_keys(opened) == ["conv1", opened._INDEX_VERSION_KEY]
        # The marker makes the next startup skip the scan entirely.
        assert opened.prune_stale_indexes() == 0
    finally:
        opened.close()


def test_pruning_leaves_current_version_entries_alone(store):
    messages = _exchange()
    store.store(CLIENT_ID, MODEL, messages, metadata=["c", "r", "rc"])

    assert store.prune_stale_indexes() == 0
    assert store.find(MODEL, messages) is not None


def test_normal_conversation_is_not_collected_for_remote_deletion(store):
    messages = _exchange()
    store.store(
        CLIENT_ID,
        MODEL,
        messages,
        metadata=["c_normal123", "r", "rc"],
        chat_mode=ChatMode.NORMAL,
    )

    key = store.keys()[0]
    conv = store.find(MODEL, messages)
    assert conv is not None
    conv.created_at = datetime.now() - timedelta(days=20)
    conv.updated_at = datetime.now() - timedelta(days=20)
    with store._get_transaction(write=True) as txn:
        txn.put(key.encode("utf-8"), orjson.dumps(conv.model_dump(mode="json")), overwrite=True)

    removed, remote_deletions = store.cleanup_expired(retention_days=14)
    assert removed == 1
    assert remote_deletions == []
    assert store.find(MODEL, messages) is None


def test_multi_turn_auto_delete_does_not_remotely_delete_cid_if_newer_turn_is_active(store):
    from app.services.lmdb import _hash_conversation

    turn1 = _exchange("turn 1")
    turn2 = [
        *turn1,
        AppMessage(role="user", content="turn 2"),
        AppMessage(role="assistant", content="reply 2"),
    ]

    cid = "c_shared_123"
    store.store(CLIENT_ID, MODEL, turn1, metadata=[cid, "r", "rc"], chat_mode=ChatMode.AUTO_DELETE)
    store.store(CLIENT_ID, MODEL, turn2, metadata=[cid, "r", "rc"], chat_mode=ChatMode.AUTO_DELETE)

    # Age turn 1 to be expired (> 24h), keep turn 2 fresh
    conv1 = store.find(MODEL, turn1)
    assert conv1 is not None
    conv1.updated_at = datetime.now() - timedelta(hours=25)
    key1 = _hash_conversation(CLIENT_ID, MODEL, turn1)
    with store._get_transaction(write=True) as txn:
        txn.put(key1.encode("utf-8"), orjson.dumps(conv1.model_dump(mode="json")), overwrite=True)

    # First cleanup: turn 1 expires, but turn 2 is active with the same cid
    removed, remote_deletions = store.cleanup_expired(auto_delete_ttl=86400)
    assert removed == 1
    assert remote_deletions == []  # cid must NOT be deleted yet because turn 2 is active!
    assert store.find(MODEL, turn1) is None
    assert store.find(MODEL, turn2) is not None

    # Now age turn 2 as well
    conv2 = store.find(MODEL, turn2)
    assert conv2 is not None
    conv2.updated_at = datetime.now() - timedelta(hours=25)
    key2 = _hash_conversation(CLIENT_ID, MODEL, turn2)
    with store._get_transaction(write=True) as txn:
        txn.put(key2.encode("utf-8"), orjson.dumps(conv2.model_dump(mode="json")), overwrite=True)

    # Second cleanup: turn 2 expires, no active conversations remain for cid
    removed2, remote_deletions2 = store.cleanup_expired(auto_delete_ttl=86400)
    assert removed2 == 1
    assert remote_deletions2 == [(CLIENT_ID, cid)]
    assert store.find(MODEL, turn2) is None


def test_pending_deletions_persisted_and_acknowledged(store):
    messages = _exchange()
    cid = "c_pending_test"
    store.store(
        CLIENT_ID,
        MODEL,
        messages,
        metadata=[cid, "r", "rc"],
        chat_mode=ChatMode.AUTO_DELETE,
    )

    key = store.keys()[0]
    conv = store.find(MODEL, messages)
    assert conv is not None
    conv.updated_at = datetime.now() - timedelta(hours=25)
    with store._get_transaction(write=True) as txn:
        txn.put(key.encode("utf-8"), orjson.dumps(conv.model_dump(mode="json")), overwrite=True)

    # First cleanup removes it locally and stages its remote deletion
    removed, remote_deletions = store.cleanup_expired(auto_delete_ttl=86400)
    assert removed == 1
    assert store.find(MODEL, messages) is None
    assert remote_deletions == [(CLIENT_ID, cid)]
    assert store.get_pending_deletions() == [(CLIENT_ID, cid)]

    # Subsequent cleanup without acknowledgment retries the pending deletion
    removed_retry, remote_deletions_retry = store.cleanup_expired(auto_delete_ttl=86400)
    assert removed_retry == 0
    assert remote_deletions_retry == [(CLIENT_ID, cid)]

    # Acknowledging deletion clears it from pending
    assert store.remove_pending_deletion(cid) is True
    assert store.get_pending_deletions() == []

    # Following cleanup has nothing to delete
    removed_after, remote_deletions_after = store.cleanup_expired(auto_delete_ttl=86400)
    assert removed_after == 0
    assert remote_deletions_after == []


def test_legacy_conversation_without_chat_mode_is_not_remotely_deleted(store):
    from app.utils import g_config

    orig_mode = g_config.gemini.chat_mode
    try:
        g_config.gemini.chat_mode = ChatMode.AUTO_DELETE

        messages = _exchange()
        cid = "c_legacy_123"
        store.store(
            CLIENT_ID,
            MODEL,
            messages,
            metadata=[cid, "r", "rc"],
            chat_mode=None,
        )

        key = store.keys()[0]
        conv = store.find(MODEL, messages)
        assert conv is not None
        assert conv.chat_mode is None

        # Age it beyond auto_delete_ttl (25h) but within retention_days (14d)
        conv.updated_at = datetime.now() - timedelta(hours=25)
        with store._get_transaction(write=True) as txn:
            txn.put(key.encode("utf-8"), orjson.dumps(conv.model_dump(mode="json")), overwrite=True)

        removed, remote_deletions = store.cleanup_expired(retention_days=14, auto_delete_ttl=86400)
        assert removed == 0
        assert remote_deletions == []
        assert store.find(MODEL, messages) is not None
    finally:
        g_config.gemini.chat_mode = orig_mode


def test_ephemeral_guest_conversation_is_not_staged_for_remote_deletion(store):
    messages = _exchange()
    cid = "c_guest_123"
    store.store(
        CLIENT_ID,
        MODEL,
        messages,
        metadata=[cid, "r", "rc"],
        chat_scope="guest:session_abc",
        chat_mode=ChatMode.AUTO_DELETE,
    )

    key = store.keys()[0]
    conv = store.find(MODEL, messages)
    assert conv is not None
    assert conv.chat_mode == ChatMode.AUTO_DELETE
    assert conv.chat_scope == "guest:session_abc"

    # Age it beyond auto_delete_ttl (25h)
    conv.updated_at = datetime.now() - timedelta(hours=25)
    with store._get_transaction(write=True) as txn:
        txn.put(key.encode("utf-8"), orjson.dumps(conv.model_dump(mode="json")), overwrite=True)

    removed, remote_deletions = store.cleanup_expired(auto_delete_ttl=86400)
    # The record should be removed from LMDB, but NOT staged for Google remote deletion
    assert removed == 1
    assert remote_deletions == []
    assert store.get_pending_deletions() == []
    assert store.find(MODEL, messages) is None


def test_pending_deletion_failure_retries_and_eviction(store):
    cid = "c_retry_fail"
    with store._get_transaction(write=True) as txn:
        txn.put(f"{store.PENDING_DELETE_PREFIX}{cid}".encode(), CLIENT_ID.encode())

    assert store.get_pending_deletions() == [(CLIENT_ID, cid)]

    # Attempt 1: retry increments, not evicted
    evicted = store.record_pending_deletion_failure(cid, max_retries=3)
    assert evicted is False
    assert store.get_pending_deletions() == [(CLIENT_ID, cid)]

    # Attempt 2: retry increments, not evicted
    evicted = store.record_pending_deletion_failure(cid, max_retries=3)
    assert evicted is False
    assert store.get_pending_deletions() == [(CLIENT_ID, cid)]

    # Attempt 3: reached max_retries=3, evicted!
    evicted = store.record_pending_deletion_failure(cid, max_retries=3)
    assert evicted is True
    assert store.get_pending_deletions() == []


def test_only_an_auto_delete_conversation_makes_the_store_report_auto_delete_work(store):
    assert store.has_auto_delete_records() is False

    messages = _exchange()
    store.store(
        CLIENT_ID,
        MODEL,
        messages,
        metadata=["c_norm", "r", "rc"],
        chat_mode=ChatMode.NORMAL,
    )
    assert store.has_auto_delete_records() is False

    store.store(
        CLIENT_ID,
        MODEL,
        messages,
        metadata=["c_auto", "r", "rc"],
        chat_mode=ChatMode.AUTO_DELETE,
    )
    assert store.has_auto_delete_records() is True


def test_cleanup_expired_media_with_ttl_seconds(tmp_path, monkeypatch):
    import os
    import time

    from app.server import middleware

    monkeypatch.setattr(middleware, "MEDIA_STORE_DIR", tmp_path)
    old_file = tmp_path / "old.png"
    old_file.write_text("old")
    new_file = tmp_path / "new.png"
    new_file.write_text("new")

    past = time.time() - 7200
    os.utime(old_file, (past, past))

    removed = middleware.cleanup_expired_media(ttl_seconds=3600)
    assert removed == 1
    assert not old_file.exists()
    assert new_file.exists()


@pytest.mark.anyio
async def test_pool_delete_chat_missing_client_acknowledges():
    from app.services.pool import GeminiClientPool

    pool = GeminiClientPool()
    res = await pool.delete_chat("non_existent_client_id_xyz", "cid_123")
    assert res is True
