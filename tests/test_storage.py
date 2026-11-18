"""Conversation-store behaviour: lookup, retention, and index versioning.

Every test opens its own store under `tmp_path` via `open_isolated`, so nothing touches the
singleton or the configured data directory.
"""

from datetime import datetime, timedelta

import lmdb
import orjson
import pytest

from app.models.core import AppContentItem, AppMessage
from app.services.lmdb import LMDBConversationStore

# `find` only searches the configured clients, so a stored conversation has to claim one.
CLIENT_ID = "client-id-1"
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


def test_echoed_reasoning_still_matches_the_stored_turn(store):
    """Reuse must survive a client replaying the reasoning this server emitted.

    `_persist_conversation` stores every assistant turn it produces with `reasoning_content=None`,
    but both request converters populate it from whatever the client sends back - and replaying
    the previous output is the normal pattern on the Responses API. If the hash counted reasoning,
    the newest stored prefix could never match and session reuse would collapse.
    """
    stored = [
        AppMessage(role="user", content="hello"),
        AppMessage(role="assistant", content="hi there", reasoning_content=None),
    ]
    store.store(CLIENT_ID, MODEL, stored, metadata=["c", "r", "rc"])

    echoed = [
        AppMessage(role="user", content="hello"),
        AppMessage(role="assistant", content="hi there", reasoning_content="let me think..."),
    ]
    found = store.find(MODEL, echoed)
    assert found is not None
    assert found.metadata == ["c", "r", "rc"]


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


def test_lookup_entries_are_written_under_the_current_index_version(store):
    store.store(CLIENT_ID, MODEL, _exchange(), metadata=["c", "r", "rc"])

    index_keys = [key for key in _raw_keys(store) if store._is_index_key(key)]
    assert index_keys
    assert all(
        key.startswith((store.HASH_LOOKUP_PREFIX, store.FUZZY_LOOKUP_PREFIX)) for key in index_keys
    )
    assert LMDBConversationStore.INDEX_VERSION in store.HASH_LOOKUP_PREFIX


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
