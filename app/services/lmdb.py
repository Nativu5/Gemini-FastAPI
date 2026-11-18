import hashlib
import string
from collections.abc import Generator, Mapping
from contextlib import contextmanager, suppress
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
from app.utils.helper import (
    normalize_llm_text,
    remove_tool_call_blocks,
    unescape_text,
)
from app.utils.singleton import Singleton

_VOLATILE_TRANS_TABLE = str.maketrans("", "", string.whitespace + string.punctuation)


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
    text = remove_tool_call_blocks(text)

    return _fuzzy_normalize(text) if fuzzy else text.strip() or None


def _hash_message(message: AppMessage, fuzzy: bool = False) -> str:
    """
    Generate a stable, canonical hash for a single message.
    """
    core_data: dict[str, Any] = {
        "role": message.role,
        "name": message.name,
        "tool_call_id": message.tool_call_id,
    }

    content = message.content
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

    # `reasoning_content` is deliberately NOT hashed. `_persist_conversation` stores every
    # assistant turn it produces with `reasoning_content=None`, while both request converters
    # populate it from whatever the client echoes back - and this server does emit reasoning on
    # both surfaces. Hashing it would make the stored turn and the replayed turn disagree by
    # construction, so the newest prefix could never match and reuse would collapse.
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
    INDEX_VERSION = "v2"
    HASH_LOOKUP_PREFIX = f"hash:{INDEX_VERSION}:"
    FUZZY_LOOKUP_PREFIX = f"fuzzy:{INDEX_VERSION}:"
    _INDEX_NAMESPACES = ("hash:", "fuzzy:")
    _INTERNAL_NAMESPACES = ("hash:", "fuzzy:", "meta:")
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
        """
        Initialize LMDB store.

        Args:
            db_path: Path to LMDB database directory
            max_db_size: Maximum database size in bytes (default: 256 MB)
            retention_days: Number of days to retain conversations (default: 14, 0 disables cleanup)
        """
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
        """Create database directory if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_environment(self) -> None:
        """Initialize LMDB environment."""
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

    @contextmanager
    def _get_transaction(self, write: bool = False) -> Generator[Transaction]:
        """
        Context manager for LMDB transactions.

        Args:
            write: Whether the transaction should be writable.
        """
        if not self._env:
            raise RuntimeError("LMDB environment not initialized")

        txn: Transaction = self._env.begin(write=write)
        try:
            yield txn
            if write:
                txn.commit()
        except Error:
            if write:
                txn.abort()
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LMDB transaction: {e}")
            if write:
                txn.abort()
            raise

    @staticmethod
    def _decode_index_value(data: bytes | memoryview) -> list[str]:
        """Decode index value, handling both legacy single-string and new list-of-strings formats."""
        if not data:
            return []
        data = bytes(data)
        if data.startswith(b"["):
            with suppress(orjson.JSONDecodeError):
                val = orjson.loads(data)
                if isinstance(val, list):
                    return [str(v) for v in val]
        try:
            return [data.decode("utf-8")]
        except UnicodeDecodeError:
            return []

    def _update_index(self, txn: Transaction, prefix: str, hash_val: str, storage_key: str):
        """Add a storage key to the index for a given hash, avoiding duplicates."""
        idx_key = f"{prefix}{hash_val}".encode()
        existing = txn.get(idx_key)
        keys = self._decode_index_value(existing) if existing else []
        if storage_key not in keys:
            keys.append(storage_key)
            txn.put(idx_key, orjson.dumps(keys))

    def _remove_from_index(self, txn: Transaction, prefix: str, hash_val: str, storage_key: str):
        """Remove a specific storage key from the index for a given hash."""
        idx_key = f"{prefix}{hash_val}".encode()
        existing = txn.get(idx_key)
        if not existing:
            return
        keys = self._decode_index_value(existing)
        if storage_key in keys:
            keys.remove(storage_key)
            if keys:
                txn.put(idx_key, orjson.dumps(keys))
            else:
                txn.delete(idx_key)

    def store(
        self,
        client_id: str,
        model: str,
        messages: list[AppMessage],
        metadata: list[str | None],
        chat_scope: str | None = None,
    ) -> None:
        """
        Store a conversation model in LMDB.

        Args:
            client_id: The client identifier
            model: The model name
            messages: Unsanitized API messages
            metadata: Session metadata
            chat_scope: Identity of the ephemeral window owning the chat, None if it is a normal
                chat kept in the account's history
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        now = datetime.now()
        conv = ConversationInStore(
            model=model,
            client_id=client_id,
            metadata=metadata,
            messages=messages,
            chat_scope=chat_scope,
            created_at=now,
            updated_at=now,
        )
        message_hash = _hash_conversation(conv.client_id, conv.model, conv.messages)
        fuzzy_hash = _hash_conversation(conv.client_id, conv.model, conv.messages, fuzzy=True)
        storage_key = message_hash

        now = datetime.now()
        if conv.created_at is None:
            conv.created_at = now
        conv.updated_at = now

        value = orjson.dumps(conv.model_dump(mode="json"))

        try:
            with self._get_transaction(write=True) as txn:
                txn.put(storage_key.encode("utf-8"), value, overwrite=True)

                self._update_index(txn, self.HASH_LOOKUP_PREFIX, message_hash, storage_key)
                self._update_index(txn, self.FUZZY_LOOKUP_PREFIX, fuzzy_hash, storage_key)

                logger.debug(f"Stored {len(conv.messages)} messages with key: {storage_key[:12]}")

        except Error as e:
            logger.error(f"LMDB error while storing messages with key {storage_key[:12]}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error while storing messages with key {storage_key[:12]}: {e}"
            )
            raise

    def get(self, key: str) -> ConversationInStore | None:
        """
        Retrieve conversation data by key.

        Args:
            key: Storage key (hash or custom key)

        Returns:
            Conversation or None if not found
        """
        try:
            with self._get_transaction(write=False) as txn:
                return self._get_messages_from_database(txn, key)
        except (Error, orjson.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve/parse messages with key {key[:12]}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving messages with key {key[:12]}: {e}")
            return None

    @staticmethod
    def _get_messages_from_database(txn, key):
        data = txn.get(key.encode("utf-8"), default=None)
        if not data:
            return None

        storage_data = orjson.loads(data)
        conv = ConversationInStore.model_validate(storage_data)

        logger.debug(f"Retrieved {len(conv.messages)} messages with key: {key[:12]}")
        return conv

    def find(self, model: str, messages: list[AppMessage]) -> ConversationInStore | None:
        """
        Search conversation data by message list.
        Tries sanitized matching, and finally fuzzy matching.

        Args:
            model: Model name
            messages: List of messages to match

        Returns:
            ConversationInStore or None if not found
        """
        if not messages:
            return None

        if conv := self._find_by_message_list(model, messages):
            logger.debug(f"Session found for '{model}' with {len(messages)} cleaned messages.")
            return conv

        if conv := self._find_by_message_list(model, messages, fuzzy=True):
            logger.debug(
                f"Session found for '{model}' with {len(messages)} fuzzy matching messages."
            )
            return conv

        logger.debug(f"No session found for '{model}' with {len(messages)} messages.")
        return None

    def _find_by_message_list(
        self,
        model: str,
        messages: list[AppMessage],
        fuzzy: bool = False,
    ) -> ConversationInStore | None:
        """
        Internal find implementation based on a message list.

        Args:
            model: Model name
            messages: Message list to hash
            fuzzy: Whether to use fuzzy hashing

        Returns:
            ConversationInStore or None if not found
        """
        prefix = self.FUZZY_LOOKUP_PREFIX if fuzzy else self.HASH_LOOKUP_PREFIX
        target_len = len(messages)

        target_hashes = [_hash_message(m, fuzzy=fuzzy) for m in messages]

        for c in g_config.gemini.clients:
            message_hash = _hash_conversation(c.id, model, messages, fuzzy=fuzzy)
            key = f"{prefix}{message_hash}"
            try:
                with self._get_transaction(write=False) as txn:
                    if mapped := txn.get(key.encode("utf-8")):
                        candidate_keys = self._decode_index_value(mapped)
                        for ck in reversed(candidate_keys):
                            if conv := self.get(ck):
                                if len(conv.messages) != target_len:
                                    continue

                                match_found = all(
                                    _hash_message(conv.messages[i], fuzzy=fuzzy) == target_hashes[i]
                                    for i in range(target_len)
                                )
                                if match_found:
                                    return conv
            except Error as e:
                logger.error(
                    f"LMDB error while searching for hash {message_hash} and client {c.id}: {e}"
                )
                continue

            if conv := self.get(message_hash):
                return conv
        return None

    def evict(self, conv: ConversationInStore) -> bool:
        """Delete a stored conversation given the record itself.

        Used to drop metadata that Google has already invalidated, so the next request
        does not rediscover the same dead session and fail again.
        """
        key = _hash_conversation(conv.client_id, conv.model, conv.messages)
        return self.delete(key) is not None

    def exists(self, key: str) -> bool:
        """Check if a key exists in the store."""
        try:
            with self._get_transaction(write=False) as txn:
                return txn.get(key.encode("utf-8")) is not None
        except Error as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            return False

    def delete(self, key: str) -> ConversationInStore | None:
        """Delete conversation model by key."""
        try:
            with self._get_transaction(write=True) as txn:
                return self._delete_messages_from_database(txn, key)
        except (Error, orjson.JSONDecodeError) as e:
            logger.error(f"Failed to delete messages with key {key[:12]}: {e}")
            return None

    def _delete_messages_from_database(self, txn, key):
        data = txn.get(key.encode("utf-8"))
        if not data:
            return None

        storage_data = orjson.loads(data)
        conv = ConversationInStore.model_validate(storage_data)
        message_hash = _hash_conversation(conv.client_id, conv.model, conv.messages)
        fuzzy_hash = _hash_conversation(conv.client_id, conv.model, conv.messages, fuzzy=True)

        txn.delete(key.encode("utf-8"))

        self._remove_from_index(txn, self.HASH_LOOKUP_PREFIX, message_hash, key)
        self._remove_from_index(txn, self.FUZZY_LOOKUP_PREFIX, fuzzy_hash, key)

        logger.debug(f"Deleted messages with key: {key[:12]}")
        return conv

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
                    and not decoded.startswith((self.HASH_LOOKUP_PREFIX, self.FUZZY_LOOKUP_PREFIX))
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

    def keys(self, prefix: str = "", limit: int | None = None) -> list[str]:
        """List all keys in the store, optionally filtered by prefix."""
        keys = []
        try:
            with self._get_transaction(write=False) as txn:
                cursor = txn.cursor()
                cursor.first()

                count = 0
                for key, _ in cursor:
                    key_str = bytes(key).decode("utf-8")
                    if self._is_internal_key(key_str):
                        continue

                    if not prefix or key_str.startswith(prefix):
                        keys.append(key_str)
                        count += 1
                        if limit and count >= limit:
                            break
        except Error as e:
            logger.error(f"Failed to list keys: {e}")
        return keys

    def cleanup_expired(self, retention_days: int | None = None) -> int:
        """Delete conversations older than the given retention period."""
        retention_value = (
            self.retention_days if retention_days is None else max(0, int(retention_days))
        )
        if retention_value <= 0:
            logger.debug("Retention cleanup skipped because retention is disabled.")
            return 0

        cutoff = datetime.now() - timedelta(days=retention_value)
        return self.cleanup_before(cutoff)

    def cleanup_before(self, cutoff: datetime) -> int:
        """Delete conversations older than an explicit timestamp and repair both indexes."""
        expired_entries: list[tuple[str, ConversationInStore]] = []

        try:
            with self._get_transaction(write=False) as txn:
                cursor = txn.cursor()
                for key_bytes, value_bytes in cursor:
                    key_str = bytes(key_bytes).decode("utf-8")
                    if self._is_internal_key(key_str):
                        continue

                    try:
                        storage_data = orjson.loads(value_bytes)
                        conv = ConversationInStore.model_validate(storage_data)
                    except (orjson.JSONDecodeError, Exception) as exc:
                        logger.warning(f"Failed to decode record for key {key_str}: {exc}")
                        continue

                    # Last touched, not first created: a conversation still in active use has
                    # not expired no matter how long ago it started.
                    timestamp = conv.updated_at or conv.created_at
                    if not timestamp:
                        continue

                    if timestamp < cutoff:
                        expired_entries.append((key_str, conv))
        except Error as exc:
            logger.error(f"Failed to scan LMDB for retention cleanup: {exc}")
            raise

        if not expired_entries:
            return 0

        removed = 0
        try:
            with self._get_transaction(write=True) as txn:
                for key_str, conv in expired_entries:
                    key_bytes = key_str.encode("utf-8")
                    if not txn.delete(key_bytes):
                        continue

                    if message_hash := _hash_conversation(
                        conv.client_id, conv.model, conv.messages
                    ):
                        self._remove_from_index(txn, self.HASH_LOOKUP_PREFIX, message_hash, key_str)
                        fuzzy_hash = _hash_conversation(
                            conv.client_id, conv.model, conv.messages, fuzzy=True
                        )
                        self._remove_from_index(txn, self.FUZZY_LOOKUP_PREFIX, fuzzy_hash, key_str)
                    removed += 1
        except Error as exc:
            logger.error(f"Failed to delete expired conversations: {exc}")
            raise

        if removed:
            logger.info(
                f"LMDB retention cleanup removed {removed} conversation(s) older than {cutoff.isoformat()}."
            )

        return removed

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
        """Get database statistics."""
        if not self._env:
            logger.error("LMDB environment not initialized")
            return {}
        try:
            return self._env.stat()
        except Error as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

    def close(self) -> None:
        """Close the LMDB environment."""
        if self._env:
            self._env.close()
            self._env = None
            logger.info("LMDB environment closed")

    def __del__(self):
        """Cleanup on destruction."""
        self.close()
