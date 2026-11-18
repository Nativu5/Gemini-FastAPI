import hashlib
import string
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Self, cast

import lmdb
import orjson
from lmdb import Environment, Error, Transaction
from loguru import logger

from app.models import (
    AppMessage,
    ConversationInStore,
)
from app.utils import g_config
from app.utils.config import ChatMode
from app.utils.helper import (
    normalize_llm_text,
    strip_system_hints,
    unescape_text,
)
from app.utils.singleton import Singleton

_VOLATILE_TRANS_TABLE = str.maketrans("", "", string.whitespace + string.punctuation)


def cid_of(metadata: list[str | None] | None) -> str | None:
    """Chat id from a metadata list, which stores it at index 0."""
    return metadata[0] if metadata else None


def _parse_pending_val(val_bytes: bytes) -> tuple[str, int]:
    """Parse pending deletion value into (client_id, attempts)."""
    val_str = val_bytes.decode("utf-8")
    if val_str.startswith("{"):
        with suppress(Exception):
            data = orjson.loads(val_bytes)
            return data.get("client_id", ""), data.get("attempts", 0)
    return val_str, 0


def _fuzzy_normalize(text: str | None) -> str | None:
    """
    Lowercase and remove all whitespace and punctuation.
    Used as a fallback for complex/malformed contents matching.
    """
    return None if text is None else text.lower().translate(_VOLATILE_TRANS_TABLE)


def _normalize_text(text: str | None, fuzzy: bool = False) -> str | None:
    """Perform safe semantic normalization for hashing using helper utilities."""
    if text is None:
        return None

    text = normalize_llm_text(text)
    text = unescape_text(text)
    text = strip_system_hints(text)

    return _fuzzy_normalize(text) if fuzzy else text.strip() or None


def _hash_message(message: AppMessage, fuzzy: bool = False) -> str:
    """Generate a stable, canonical hash for a single message."""
    core_data: dict[str, Any] = {
        "role": message.role,
        "name": message.name,
        "tool_call_id": message.tool_call_id,
    }

    content = message.content
    if isinstance(content, list) and all(item.type == "text" for item in content):
        # Text parts reach Gemini joined by newlines, the same prompt one string would give, so
        # they hash like that string. A client that replays an assistant turn as parts, as the
        # Responses API has it do, then still matches the turn stored as text.
        content = "\n".join(item.text or "" for item in content)
    if content is None:
        core_data["content"] = None
    elif isinstance(content, str):
        core_data["content"] = _normalize_text(content, fuzzy=fuzzy)
    elif isinstance(content, list):
        content_items: list[dict[str, Any]] = []
        for item in content:
            item_data: dict[str, Any] = {
                "type": item.type,
                "filename": item.filename,
                "url": item.url,
                "content_digest": item.content_digest,
            }
            if item.text is not None:
                item_data["text"] = _normalize_text(item.text, fuzzy=fuzzy)
            if item.raw_data is not None:
                # Included directly: the outer dump sorts keys recursively, so this is already
                # canonical, and digesting it would only serialize the same data a second time.
                item_data["raw_data"] = item.raw_data
            content_items.append(item_data)

        core_data["content"] = content_items or None

    if message.tool_calls:
        calls_data = []
        for tc in message.tool_calls:
            args = tc.function.arguments
            name = tc.function.name
            try:
                parsed = orjson.loads(args)
                canon_args = orjson.dumps(parsed, option=orjson.OPT_SORT_KEYS).decode("utf-8")
            except orjson.JSONDecodeError:
                canon_args = args

            calls_data.append(
                {
                    "name": name,
                    "arguments": canon_args,
                }
            )
        calls_data.sort(key=lambda x: (x["name"], x["arguments"]))
        core_data["tool_calls"] = calls_data
    else:
        core_data["tool_calls"] = None

    message_bytes = orjson.dumps(core_data, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(message_bytes).hexdigest()


def _hash_conversation(
    client_id: str, model: str, messages: list[AppMessage], fuzzy: bool = False
) -> str:
    """Generate a hash for a list of messages and model name, tied to a specific client_id."""
    combined_hash = hashlib.sha256()
    combined_hash.update((client_id or "").encode("utf-8"))
    combined_hash.update((model or "").encode("utf-8"))
    for message in messages:
        message_hash = _hash_message(message, fuzzy=fuzzy)
        combined_hash.update(message_hash.encode("utf-8"))
    return combined_hash.hexdigest()


class LMDBConversationStore(metaclass=Singleton):
    """LMDB-based storage for Message lists with hash-based key-value operations."""

    # Bump when _hash_message changes shape. Entries under an older version can never match
    # again, and their conversations would otherwise keep index rows no eviction can find,
    # so startup sweeps them instead of leaving them to accumulate.
    #
    # The conversation records themselves are keyed by the hash that produced them, so records
    # written under an older version stay unreachable after the sweep: a repeat of the same
    # conversation is replayed in full and stored again under a current key, and the superseded
    # record is left to expire on the normal retention schedule.
    INDEX_VERSION = "v3"
    FUZZY_LOOKUP_PREFIX = f"fuzzy:{INDEX_VERSION}:"
    PENDING_DELETE_PREFIX = "pending_del:"
    MAX_REMOTE_DELETION_ATTEMPTS = 5
    _INDEX_NAMESPACES = ("hash:", "fuzzy:")
    _INTERNAL_NAMESPACES = ("hash:", "fuzzy:", "meta:", "pending_del:")
    _INDEX_VERSION_KEY = "meta:index_version"

    @classmethod
    def open_isolated(
        cls,
        db_path: str,
        max_db_size: int | None = None,
        retention_days: int | None = None,
    ) -> Self:
        """Open a store outside the singleton, for maintenance commands and isolated tests.

        LMDB does not support two environments on one path in a single process, so `db_path`
        must not be the path the singleton already holds open.
        """
        return cast(
            Self,
            type.__call__(
                cls,
                db_path=db_path,
                max_db_size=max_db_size,
                retention_days=retention_days,
            ),
        )

    def __init__(
        self,
        db_path: str | None = None,
        max_db_size: int | None = None,
        retention_days: int | None = None,
    ):
        """Open the store; an argument left as None takes its value from `g_config.storage`."""
        if db_path is None:
            db_path = g_config.storage.path
        if max_db_size is None:
            max_db_size = g_config.storage.max_size
        if retention_days is None:
            retention_days = g_config.storage.retention_days

        self.db_path: Path = Path(db_path)
        self.max_db_size: int = max_db_size
        self.retention_days: int = max(0, int(retention_days))
        self._env: Environment | None = None

        self._ensure_db_path()
        self._init_environment()

    def _ensure_db_path(self) -> None:
        """Create the parent of the database directory; LMDB creates the directory itself."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_environment(self) -> None:
        try:
            self._env = lmdb.open(
                str(self.db_path),
                map_size=self.max_db_size,
                max_dbs=3,
                writemap=True,
                readahead=False,
                meminit=False,
            )
            logger.info(f"LMDB environment initialized at {self.db_path}")
        except Error as e:
            logger.error(f"Failed to initialize LMDB environment: {e}")
            raise

    def _get_transaction(self, write: bool = False) -> Transaction:
        """A transaction to use in a `with`: it commits on success and aborts on an error."""
        assert self._env is not None, "the store has been closed"
        return self._env.begin(write=write)

    def _update_index(self, txn: Transaction, fuzzy_hash: str, storage_key: str, add: bool) -> None:
        """Add a conversation to, or remove it from, those listed under its fuzzy hash."""
        idx_key = f"{self.FUZZY_LOOKUP_PREFIX}{fuzzy_hash}".encode()
        existing = txn.get(idx_key)
        keys = [key for key in orjson.loads(existing) if key != storage_key] if existing else []
        if add:
            keys.append(storage_key)
        if keys:
            txn.put(idx_key, orjson.dumps(keys))
        elif existing:
            txn.delete(idx_key)

    def store(
        self,
        client_id: str,
        model: str,
        messages: list[AppMessage],
        metadata: list[str | None],
        chat_scope: str | None = None,
        chat_mode: str | None = None,
    ) -> None:
        """Store a conversation, replacing one with the same client, model and messages."""
        if not messages:
            raise ValueError("Messages list cannot be empty")

        now = datetime.now()
        conv = ConversationInStore(
            model=model,
            client_id=client_id,
            metadata=metadata,
            messages=messages,
            chat_scope=chat_scope,
            chat_mode=chat_mode,
            created_at=now,
            updated_at=now,
        )
        # A conversation is stored under its exact hash, and listed under its fuzzy one.
        storage_key = _hash_conversation(conv.client_id, conv.model, conv.messages)
        fuzzy_hash = _hash_conversation(conv.client_id, conv.model, conv.messages, fuzzy=True)
        value = orjson.dumps(conv.model_dump(mode="json"))
        with self._get_transaction(write=True) as txn:
            txn.put(storage_key.encode("utf-8"), value, overwrite=True)
            self._update_index(txn, fuzzy_hash, storage_key, add=True)
        logger.debug(f"Stored {len(conv.messages)} messages with key: {storage_key[:12]}")

    def get(self, key: str) -> ConversationInStore | None:
        """The conversation stored under this key, or None if there is none or it is unreadable."""
        try:
            with self._get_transaction() as txn:
                data = txn.get(key.encode("utf-8"))
                return ConversationInStore.model_validate(orjson.loads(data)) if data else None
        except Exception as e:
            logger.error(f"Failed to retrieve/parse messages with key {key[:12]}: {e}")
            return None

    def find(self, model: str, messages: list[AppMessage]) -> ConversationInStore | None:
        """The stored conversation with exactly these messages, on any configured client.

        Matched by exact hash first, then by the fuzzy one, which ignores case, whitespace and
        punctuation for a client that reformats what it replays.
        """
        if not messages:
            return None
        clients = g_config.gemini.clients
        for client in clients:
            if conv := self.get(_hash_conversation(client.id, model, messages)):
                return conv
        for client in clients:
            fuzzy_hash = _hash_conversation(client.id, model, messages, fuzzy=True)
            with self._get_transaction() as txn:
                listed = txn.get(f"{self.FUZZY_LOOKUP_PREFIX}{fuzzy_hash}".encode())
            for key in reversed(orjson.loads(listed) if listed else []):  # newest first
                if conv := self.get(key):
                    logger.debug(f"Session found for '{model}' by fuzzy match.")
                    return conv
        return None

    def evict(self, conv: ConversationInStore) -> bool:
        """Delete a stored conversation given the record itself.

        Used to drop metadata that Google has already invalidated, so the next request
        does not rediscover the same dead session and fail again.
        """
        key = _hash_conversation(conv.client_id, conv.model, conv.messages)
        try:
            with self._get_transaction(write=True) as txn:
                return self._delete_conversation(txn, key, conv)
        except Error as e:
            logger.error(f"Failed to delete messages with key {key[:12]}: {e}")
            return False

    def _delete_conversation(self, txn: Transaction, key: str, conv: ConversationInStore) -> bool:
        """Delete a conversation and its index entry; False if it was not stored."""
        if not txn.delete(key.encode("utf-8")):
            return False
        fuzzy_hash = _hash_conversation(conv.client_id, conv.model, conv.messages, fuzzy=True)
        self._update_index(txn, fuzzy_hash, key, add=False)
        return True

    def _is_index_key(self, key: str) -> bool:
        """Whether a raw key is a lookup entry rather than a stored conversation."""
        return key.startswith(self._INDEX_NAMESPACES)

    def _is_internal_key(self, key: str) -> bool:
        """Whether a raw key is bookkeeping rather than a stored conversation."""
        return key.startswith(self._INTERNAL_NAMESPACES)

    def prune_stale_indexes(self) -> int:
        """Drop lookup entries written under a superseded INDEX_VERSION.

        Only the lookup entries go: the conversation records they pointed at are keyed by the
        old hash and cannot be re-indexed under the new one, so they are left to expire under
        the normal retention window.

        A marker records that the sweep ran for this version, so later startups skip the scan.
        """
        version_key = self._INDEX_VERSION_KEY.encode("utf-8")
        try:
            with self._get_transaction(write=True) as txn:
                if txn.get(version_key) == self.INDEX_VERSION.encode("utf-8"):
                    return 0

                stale = [
                    bytes(key)
                    for key, _ in txn.cursor()
                    if (decoded := bytes(key).decode("utf-8", "replace"))
                    and self._is_index_key(decoded)
                    and not decoded.startswith(self.FUZZY_LOOKUP_PREFIX)
                ]
                for key in stale:
                    txn.delete(key)
                txn.put(version_key, self.INDEX_VERSION.encode("utf-8"), overwrite=True)
        except Error as exc:
            logger.error(f"Failed to prune stale LMDB indexes: {exc}")
            return 0

        if stale:
            logger.info(
                f"Pruned {len(stale)} LMDB lookup entries from a superseded index version; "
                "the conversations behind them are replayed in full once and stored again "
                "under a current key, and the superseded records expire under retention."
            )
        return len(stale)

    def keys(self) -> list[str]:
        """Keys of the stored conversations; lookup and bookkeeping entries are left out."""
        try:
            with self._get_transaction() as txn:
                return [
                    key
                    for raw_key, _ in txn.cursor()
                    if not self._is_internal_key(key := bytes(raw_key).decode("utf-8"))
                ]
        except Error as e:
            logger.error(f"Failed to list keys: {e}")
            return []

    def get_pending_deletions(self) -> list[tuple[str, str]]:
        """Retrieve all conversations pending remote deletion as (client_id, cid) pairs."""
        deletions: list[tuple[str, str]] = []
        prefix_bytes = self.PENDING_DELETE_PREFIX.encode()
        try:
            with self._get_transaction() as txn:
                cursor = txn.cursor()
                if cursor.set_range(prefix_bytes):
                    for key_bytes, val_bytes in cursor:
                        k_bytes = bytes(key_bytes)
                        if not k_bytes.startswith(prefix_bytes):
                            break
                        cid = k_bytes[len(prefix_bytes) :].decode("utf-8")
                        client_id, _ = _parse_pending_val(bytes(val_bytes))
                        deletions.append((client_id, cid))
        except Error as e:
            logger.error(f"Failed to scan pending remote deletions: {e}")
        return deletions

    def remove_pending_deletion(self, cid: str) -> bool:
        """Acknowledge a completed remote deletion and remove its pending marker."""
        key = f"{self.PENDING_DELETE_PREFIX}{cid}".encode()
        try:
            with self._get_transaction(write=True) as txn:
                return txn.delete(key)
        except Error as e:
            logger.error(f"Failed to remove pending deletion marker for {cid}: {e}")
            return False

    def record_pending_deletion_failure(self, cid: str) -> None:
        """Count a failed remote deletion, and give up on it at MAX_REMOTE_DELETION_ATTEMPTS."""
        key = f"{self.PENDING_DELETE_PREFIX}{cid}".encode()
        try:
            with self._get_transaction(write=True) as txn:
                if not (val := txn.get(key)):
                    return
                client_id, attempts = _parse_pending_val(bytes(val))
                attempts += 1
                if attempts < self.MAX_REMOTE_DELETION_ATTEMPTS:
                    txn.put(key, orjson.dumps({"client_id": client_id, "attempts": attempts}))
                    return
                txn.delete(key)
                logger.error(
                    f"Remote deletion for chat {cid} (client {client_id}) failed "
                    f"{attempts} times; dropped from pending queue."
                )
        except Error as e:
            logger.error(f"Failed to update pending deletion for {cid}: {e}")

    def has_auto_delete_records(self) -> bool:
        try:
            with self._get_transaction(write=False) as txn:
                cursor = txn.cursor()
                for key_bytes, value_bytes in cursor:
                    key_str = bytes(key_bytes).decode("utf-8")
                    if self._is_internal_key(key_str):
                        continue
                    try:
                        storage_data = orjson.loads(value_bytes)
                        if storage_data.get("chat_mode") == ChatMode.AUTO_DELETE:
                            return True
                    except Exception:
                        continue
        except Error as e:
            logger.error(f"Failed to check for auto_delete records: {e}")
        return False

    def cleanup_expired(self) -> tuple[int, list[tuple[str, str]]]:
        """Delete the conversations the configuration says are due.

        A conversation goes `storage.retention_days` after it was last used (0 keeps it for
        good). One opened in `auto_delete` mode goes after `gemini.auto_delete_ttl` seconds
        instead, and is then also queued for deletion from the Google account.

        Returns how many were removed, and every (client_id, cid) waiting for remote deletion.
        """
        now = datetime.now()
        return self.cleanup_with_remote(
            normal_cutoff=now - timedelta(days=self.retention_days)
            if self.retention_days
            else None,
            auto_delete_cutoff=now - timedelta(seconds=g_config.gemini.auto_delete_ttl),
        )

    def cleanup_before(self, cutoff: datetime) -> int:
        """Delete conversations last used before `cutoff`, in any chat mode; returns how many."""
        removed, _ = self.cleanup_with_remote(normal_cutoff=cutoff, auto_delete_cutoff=cutoff)
        return removed

    def cleanup_with_remote(
        self,
        normal_cutoff: datetime | None = None,
        auto_delete_cutoff: datetime | None = None,
    ) -> tuple[int, list[tuple[str, str]]]:
        """Delete conversations last used before the cutoff for their kind; None keeps that kind.

        Returns how many were removed, and every (client_id, cid) waiting for remote deletion,
        those queued by earlier passes included.
        """
        expired_entries: list[tuple[str, ConversationInStore, bool]] = []
        active_cids: set[str] = set()

        try:
            with self._get_transaction(write=False) as txn:
                cursor = txn.cursor()
                for key_bytes, value_bytes in cursor:
                    key_str = bytes(key_bytes).decode("utf-8")
                    if self._is_internal_key(key_str):
                        continue

                    try:
                        storage_data = orjson.loads(value_bytes)
                        ts_raw = storage_data.get("updated_at") or storage_data.get("created_at")
                        if not ts_raw:
                            continue  # nothing to age it by
                        timestamp = datetime.fromisoformat(ts_raw)

                        is_auto_delete = storage_data.get("chat_mode") == ChatMode.AUTO_DELETE
                        cutoff = auto_delete_cutoff if is_auto_delete else normal_cutoff

                        raw_meta = storage_data.get("metadata")
                        cid = raw_meta[0] if raw_meta and isinstance(raw_meta, list) else None

                        if cutoff is not None and timestamp < cutoff:
                            conv = ConversationInStore.model_validate(storage_data)
                            expired_entries.append((key_str, conv, is_auto_delete))
                        else:
                            if cid:
                                active_cids.add(cid)
                    except (orjson.JSONDecodeError, Exception) as exc:
                        logger.warning(f"Failed to decode record for key {key_str}: {exc}")
                        continue
        except Error as exc:
            logger.error(f"Failed to scan LMDB for retention cleanup: {exc}")
            raise

        if not expired_entries:
            return 0, self.get_pending_deletions()

        remote_to_stage: list[tuple[str, str]] = []
        seen_remote_cids: set[str] = set()
        for _, conv, is_auto_delete in expired_entries:
            if is_auto_delete and conv.chat_scope is None:
                cid = cid_of(conv.metadata)
                if cid and cid not in active_cids and cid not in seen_remote_cids:
                    seen_remote_cids.add(cid)
                    remote_to_stage.append((conv.client_id, cid))

        removed = 0
        try:
            with self._get_transaction(write=True) as txn:
                for key_str, conv, _ in expired_entries:
                    removed += self._delete_conversation(txn, key_str, conv)

                for client_id, cid in remote_to_stage:
                    txn.put(
                        f"{self.PENDING_DELETE_PREFIX}{cid}".encode(),
                        client_id.encode(),
                    )
        except Error as exc:
            logger.error(f"Failed to delete expired conversations: {exc}")
            raise

        remote_deletions = self.get_pending_deletions()
        if removed or remote_deletions:
            logger.info(
                f"LMDB retention cleanup removed {removed} conversation(s) "
                f"({len(remote_deletions)} scheduled for Google deletion)."
            )

        return removed, remote_deletions

    def clear(self) -> int:
        """Delete every conversation and index entry from the store."""
        removed = len(self.keys())
        try:
            with self._get_transaction(write=True) as txn:
                keys = [bytes(key) for key, _ in txn.cursor()]
                for key in keys:
                    txn.delete(key)
        except Error as exc:
            logger.error(f"Failed to clear LMDB: {exc}")
            raise
        return removed

    def stats(self) -> Mapping[str, Any]:
        """LMDB's statistics for the environment; {} when they cannot be read."""
        if not self._env:
            logger.error("LMDB environment not initialized")
            return {}
        try:
            return self._env.stat()
        except Error as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

    def close(self) -> None:
        if self._env:
            self._env.close()
            self._env = None
            logger.info("LMDB environment closed")

    def __del__(self):
        self.close()
