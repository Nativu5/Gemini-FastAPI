import hashlib
import re
import string
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import lmdb
import orjson
from loguru import logger

from ..models import ContentItem, ConversationInStore, Message
from ..utils import g_config
from ..utils.helper import (
    extract_tool_calls,
    normalize_llm_text,
    remove_tool_call_blocks,
    strip_system_hints,
    unescape_text,
)
from ..utils.singleton import Singleton

_VOLATILE_TRANS_TABLE = str.maketrans("", "", string.whitespace + string.punctuation)


def _fuzzy_normalize(text: str | None) -> str | None:
    """
    Lowercase and remove all whitespace and punctuation.
    Used as a fallback for complex/malformed contents matching.
    """
    if text is None:
        return None
    return text.lower().translate(_VOLATILE_TRANS_TABLE)


def _normalize_text(text: str | None, fuzzy: bool = False) -> str | None:
    """Perform safe semantic normalization for hashing using helper utilities."""
    if text is None:
        return None

    text = normalize_llm_text(text)
    text = unescape_text(text)

    text = LMDBConversationStore.remove_think_tags(text)
    text = remove_tool_call_blocks(text)

    if fuzzy:
        return _fuzzy_normalize(text)

    # Always strip to ensure trailing newlines/spaces don't break exact matches
    return text.strip() if text.strip() else None


def _hash_message(message: Message, fuzzy: bool = False) -> str:
    """
    Generate a stable, canonical hash for a single message.
    """
    core_data: dict[str, Any] = {
        "role": message.role,
        "name": message.name or None,
        "tool_call_id": message.tool_call_id or None,
    }

    content = message.content
    if content is None:
        core_data["content"] = None
    elif isinstance(content, str):
        core_data["content"] = _normalize_text(content, fuzzy=fuzzy)
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            text_val = ""
            if isinstance(item, ContentItem) and item.type == "text":
                text_val = item.text
            elif isinstance(item, dict) and item.get("type") == "text":
                text_val = item.get("text")

            if text_val:
                normalized_part = _normalize_text(text_val, fuzzy=fuzzy)
                if normalized_part:
                    text_parts.append(normalized_part)
            elif isinstance(item, (ContentItem, dict)):
                item_type = item.type if isinstance(item, ContentItem) else item.get("type")
                if item_type == "image_url":
                    url = (
                        item.image_url.get("url")
                        if isinstance(item, ContentItem) and item.image_url
                        else item.get("image_url", {}).get("url")
                    )
                    text_parts.append(f"[image_url:{url}]")
                elif item_type == "file":
                    url = (
                        item.file.get("url") or item.file.get("filename")
                        if isinstance(item, ContentItem) and item.file
                        else item.get("file", {}).get("url") or item.get("file", {}).get("filename")
                    )
                    text_parts.append(f"[file:{url}]")

        core_data["content"] = "\n".join(text_parts) if text_parts else None

    if message.tool_calls:
        calls_data = []
        for tc in message.tool_calls:
            args = tc.function.arguments or "{}"
            try:
                parsed = orjson.loads(args)
                canon_args = orjson.dumps(parsed, option=orjson.OPT_SORT_KEYS).decode("utf-8")
            except orjson.JSONDecodeError:
                canon_args = args

            calls_data.append(
                {
                    "name": tc.function.name,
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
    client_id: str, model: str, messages: List[Message], fuzzy: bool = False
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

    HASH_LOOKUP_PREFIX = "hash:"
    FUZZY_LOOKUP_PREFIX = "fuzzy:"

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_db_size: Optional[int] = None,
        retention_days: Optional[int] = None,
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
        self._env: lmdb.Environment | None = None

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
        except lmdb.Error as e:
            logger.error(f"Failed to initialize LMDB environment: {e}")
            raise

    @contextmanager
    def _get_transaction(self, write: bool = False):
        """
        Context manager for LMDB transactions.

        Args:
            write: Whether the transaction should be writable.
        """
        if not self._env:
            raise RuntimeError("LMDB environment not initialized")

        txn: lmdb.Transaction = self._env.begin(write=write)
        try:
            yield txn
            if write:
                txn.commit()
        except lmdb.Error:
            if write:
                txn.abort()
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LMDB transaction: {e}")
            if write:
                txn.abort()
            raise

    @staticmethod
    def _decode_index_value(data: bytes) -> List[str]:
        """Decode index value, handling both legacy single-string and new list-of-strings formats."""
        if not data:
            return []
        if data.startswith(b"["):
            try:
                val = orjson.loads(data)
                if isinstance(val, list):
                    return [str(v) for v in val]
            except orjson.JSONDecodeError:
                pass
        try:
            return [data.decode("utf-8")]
        except UnicodeDecodeError:
            return []

    @staticmethod
    def _update_index(txn: lmdb.Transaction, prefix: str, hash_val: str, storage_key: str):
        """Add a storage key to the index for a given hash, avoiding duplicates."""
        idx_key = f"{prefix}{hash_val}".encode("utf-8")
        existing = txn.get(idx_key)
        keys = LMDBConversationStore._decode_index_value(existing) if existing else []
        if storage_key not in keys:
            keys.append(storage_key)
            txn.put(idx_key, orjson.dumps(keys))

    @staticmethod
    def _remove_from_index(txn: lmdb.Transaction, prefix: str, hash_val: str, storage_key: str):
        """Remove a specific storage key from the index for a given hash."""
        idx_key = f"{prefix}{hash_val}".encode("utf-8")
        existing = txn.get(idx_key)
        if not existing:
            return
        keys = LMDBConversationStore._decode_index_value(existing)
        if storage_key in keys:
            keys.remove(storage_key)
            if keys:
                txn.put(idx_key, orjson.dumps(keys))
            else:
                txn.delete(idx_key)

    def store(
        self,
        conv: ConversationInStore,
        custom_key: Optional[str] = None,
    ) -> str:
        """
        Store a conversation model in LMDB.

        Args:
            conv: Conversation model to store
            custom_key: Optional custom key, if not provided, hash will be used

        Returns:
            str: The key used to store the messages (hash or custom key)
        """
        if not conv:
            raise ValueError("Messages list cannot be empty")

        # Ensure consistent sanitization before hashing and storage
        sanitized_messages = self.sanitize_assistant_messages(conv.messages)
        conv.messages = sanitized_messages

        message_hash = _hash_conversation(conv.client_id, conv.model, conv.messages)
        fuzzy_hash = _hash_conversation(conv.client_id, conv.model, conv.messages, fuzzy=True)
        storage_key = custom_key or message_hash

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
                return storage_key

        except lmdb.Error as e:
            logger.error(f"LMDB error while storing messages with key {storage_key[:12]}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error while storing messages with key {storage_key[:12]}: {e}"
            )
            raise

    def get(self, key: str) -> Optional[ConversationInStore]:
        """
        Retrieve conversation data by key.

        Args:
            key: Storage key (hash or custom key)

        Returns:
            Conversation or None if not found
        """
        try:
            with self._get_transaction(write=False) as txn:
                data = txn.get(key.encode("utf-8"), default=None)
                if not data:
                    return None

                storage_data = orjson.loads(data)
                conv = ConversationInStore.model_validate(storage_data)

                logger.debug(f"Retrieved {len(conv.messages)} messages with key: {key[:12]}")
                return conv
        except (lmdb.Error, orjson.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve/parse messages with key {key[:12]}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving messages with key {key[:12]}: {e}")
            return None

    def find(self, model: str, messages: List[Message]) -> Optional[ConversationInStore]:
        """
        Search conversation data by message list.
        Tries raw matching, then sanitized matching, and finally fuzzy matching.

        Args:
            model: Model name
            messages: List of messages to match

        Returns:
            ConversationInStore or None if not found
        """
        if not messages:
            return None

        if conv := self._find_by_message_list(model, messages):
            logger.debug(f"Session found for '{model}' with {len(messages)} raw messages.")
            return conv

        cleaned_messages = self.sanitize_assistant_messages(messages)
        if cleaned_messages != messages:
            if conv := self._find_by_message_list(model, cleaned_messages):
                logger.debug(
                    f"Session found for '{model}' with {len(cleaned_messages)} cleaned messages."
                )
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
        messages: List[Message],
        fuzzy: bool = False,
    ) -> Optional[ConversationInStore]:
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

                                match_found = True
                                for i in range(target_len):
                                    if (
                                        _hash_message(conv.messages[i], fuzzy=fuzzy)
                                        != target_hashes[i]
                                    ):
                                        match_found = False
                                        break

                                if match_found:
                                    return conv
            except lmdb.Error as e:
                logger.error(
                    f"LMDB error while searching for hash {message_hash} and client {c.id}: {e}"
                )
                continue

            if conv := self.get(message_hash):
                return conv
        return None

    def exists(self, key: str) -> bool:
        """Check if a key exists in the store."""
        try:
            with self._get_transaction(write=False) as txn:
                return txn.get(key.encode("utf-8")) is not None
        except lmdb.Error as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            return False

    def delete(self, key: str) -> Optional[ConversationInStore]:
        """Delete conversation model by key."""
        try:
            with self._get_transaction(write=True) as txn:
                data = txn.get(key.encode("utf-8"))
                if not data:
                    return None

                storage_data = orjson.loads(data)
                conv = ConversationInStore.model_validate(storage_data)
                message_hash = _hash_conversation(conv.client_id, conv.model, conv.messages)
                fuzzy_hash = _hash_conversation(
                    conv.client_id, conv.model, conv.messages, fuzzy=True
                )

                txn.delete(key.encode("utf-8"))

                self._remove_from_index(txn, self.HASH_LOOKUP_PREFIX, message_hash, key)
                self._remove_from_index(txn, self.FUZZY_LOOKUP_PREFIX, fuzzy_hash, key)

                logger.debug(f"Deleted messages with key: {key[:12]}")
                return conv
        except (lmdb.Error, orjson.JSONDecodeError) as e:
            logger.error(f"Failed to delete messages with key {key[:12]}: {e}")
            return None

    def keys(self, prefix: str = "", limit: Optional[int] = None) -> List[str]:
        """List all keys in the store, optionally filtered by prefix."""
        keys = []
        try:
            with self._get_transaction(write=False) as txn:
                cursor = txn.cursor()
                cursor.first()

                count = 0
                for key, _ in cursor:
                    key_str = key.decode("utf-8")
                    # Skip internal index mappings
                    if key_str.startswith(self.HASH_LOOKUP_PREFIX) or key_str.startswith(
                        self.FUZZY_LOOKUP_PREFIX
                    ):
                        continue

                    if not prefix or key_str.startswith(prefix):
                        keys.append(key_str)
                        count += 1
                        if limit and count >= limit:
                            break
        except lmdb.Error as e:
            logger.error(f"Failed to list keys: {e}")
        return keys

    def cleanup_expired(self, retention_days: Optional[int] = None) -> int:
        """Delete conversations older than the given retention period."""
        retention_value = (
            self.retention_days if retention_days is None else max(0, int(retention_days))
        )
        if retention_value <= 0:
            logger.debug("Retention cleanup skipped because retention is disabled.")
            return 0

        cutoff = datetime.now() - timedelta(days=retention_value)
        expired_entries: list[tuple[str, ConversationInStore]] = []

        try:
            with self._get_transaction(write=False) as txn:
                cursor = txn.cursor()
                for key_bytes, value_bytes in cursor:
                    key_str = key_bytes.decode("utf-8")
                    if key_str.startswith(self.HASH_LOOKUP_PREFIX) or key_str.startswith(
                        self.FUZZY_LOOKUP_PREFIX
                    ):
                        continue

                    try:
                        storage_data = orjson.loads(value_bytes)
                        conv = ConversationInStore.model_validate(storage_data)
                    except (orjson.JSONDecodeError, Exception) as exc:
                        logger.warning(f"Failed to decode record for key {key_str}: {exc}")
                        continue

                    timestamp = conv.created_at or conv.updated_at
                    if not timestamp:
                        continue

                    if timestamp < cutoff:
                        expired_entries.append((key_str, conv))
        except lmdb.Error as exc:
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

                    message_hash = _hash_conversation(conv.client_id, conv.model, conv.messages)
                    if message_hash:
                        self._remove_from_index(txn, self.HASH_LOOKUP_PREFIX, message_hash, key_str)
                        fuzzy_hash = _hash_conversation(
                            conv.client_id, conv.model, conv.messages, fuzzy=True
                        )
                        self._remove_from_index(txn, self.FUZZY_LOOKUP_PREFIX, fuzzy_hash, key_str)
                    removed += 1
        except lmdb.Error as exc:
            logger.error(f"Failed to delete expired conversations: {exc}")
            raise

        if removed:
            logger.info(
                f"LMDB retention cleanup removed {removed} conversation(s) older than {cutoff.isoformat()}."
            )

        return removed

    def stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self._env:
            logger.error("LMDB environment not initialized")
            return {}
        try:
            return self._env.stat()
        except lmdb.Error as e:
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

    @staticmethod
    def remove_think_tags(text: str) -> str:
        """Remove all <think>...</think> tags and strip whitespace."""
        if not text:
            return text
        cleaned_content = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned_content.strip()

    @staticmethod
    def sanitize_messages(messages: list[Message]) -> list[Message]:
        """Clean all messages of internal markers, hints and normalize tool calls."""
        cleaned_messages = []
        for msg in messages:
            if isinstance(msg.content, str):
                text = LMDBConversationStore.remove_think_tags(msg.content)
                tool_calls = msg.tool_calls

                if msg.role == "assistant" and not tool_calls:
                    text, tool_calls = extract_tool_calls(text)
                else:
                    text = strip_system_hints(text)

                normalized_content = text.strip() or None

                if normalized_content != msg.content or tool_calls != msg.tool_calls:
                    cleaned_msg = msg.model_copy(
                        update={
                            "content": normalized_content,
                            "tool_calls": tool_calls or None,
                        }
                    )
                    cleaned_messages.append(cleaned_msg)
                else:
                    cleaned_messages.append(msg)
            elif isinstance(msg.content, list):
                new_content = []
                all_extracted_calls = list(msg.tool_calls or [])
                changed = False

                for item in msg.content:
                    if isinstance(item, ContentItem) and item.type == "text" and item.text:
                        text = LMDBConversationStore.remove_think_tags(item.text)
                        if msg.role == "assistant" and not msg.tool_calls:
                            text, extracted = extract_tool_calls(text)
                            if extracted:
                                all_extracted_calls.extend(extracted)
                                changed = True
                        else:
                            text = strip_system_hints(text)

                        if text != item.text:
                            changed = True
                            item = item.model_copy(update={"text": text.strip() or None})
                    new_content.append(item)

                if changed:
                    cleaned_messages.append(
                        msg.model_copy(
                            update={
                                "content": new_content,
                                "tool_calls": all_extracted_calls or None,
                            }
                        )
                    )
                else:
                    cleaned_messages.append(msg)
            else:
                cleaned_messages.append(msg)
        return cleaned_messages
