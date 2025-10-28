import hashlib
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import lmdb
import orjson
from loguru import logger

from ..models import ConversationInStore, Message
from ..utils import g_config
from ..utils.singleton import Singleton


def _hash_message(message: Message) -> str:
    """Generate a hash for a single message."""
    # Convert message to dict and sort keys for consistent hashing
    message_dict = message.model_dump(mode="json")
    message_bytes = orjson.dumps(message_dict, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(message_bytes).hexdigest()


def _hash_conversation(client_id: str, model: str, messages: List[Message]) -> str:
    """Generate a hash for a list of messages and client id."""
    # Create a combined hash from all individual message hashes
    combined_hash = hashlib.sha256()
    combined_hash.update(client_id.encode("utf-8"))
    combined_hash.update(model.encode("utf-8"))
    for message in messages:
        message_hash = _hash_message(message)
        combined_hash.update(message_hash.encode("utf-8"))
    return combined_hash.hexdigest()


class LMDBConversationStore(metaclass=Singleton):
    """LMDB-based storage for Message lists with hash-based key-value operations."""

    HASH_LOOKUP_PREFIX = "hash:"

    def __init__(self, db_path: Optional[str] = None, max_db_size: Optional[int] = None):
        """
        Initialize LMDB store.

        Args:
            db_path: Path to LMDB database directory
            max_db_size: Maximum database size in bytes (default: 128MB)
        """

        if db_path is None:
            db_path = g_config.storage.path
        if max_db_size is None:
            max_db_size = g_config.storage.max_size

        self.db_path: Path = Path(db_path)
        self.max_db_size: int = max_db_size
        self._env: lmdb.Environment | None = None

        self._ensure_db_path()
        self._init_environment()

    def _ensure_db_path(self) -> None:
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_environment(self) -> None:
        """Initialize LMDB environment."""
        try:
            self._env = lmdb.open(
                str(self.db_path),
                map_size=self.max_db_size,
                max_dbs=3,  # main, metadata, and index databases
                writemap=True,
                readahead=False,
                meminit=False,
            )
            logger.info(f"LMDB environment initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize LMDB environment: {e}")
            raise

    @contextmanager
    def _get_transaction(self, write: bool = False):
        """Get LMDB transaction context manager."""
        if not self._env:
            raise RuntimeError("LMDB environment not initialized")

        txn: lmdb.Transaction = self._env.begin(write=write)
        try:
            yield txn
            if write:
                txn.commit()
        except Exception:
            if write:
                txn.abort()
            raise
        finally:
            pass  # Transaction is automatically cleaned up

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

        # Generate hash for the message list
        message_hash = _hash_conversation(conv.client_id, conv.model, conv.messages)
        storage_key = custom_key or message_hash

        # Prepare data for storage
        now = datetime.now()
        if conv.created_at is None:
            conv.created_at = now
        conv.updated_at = now

        value = orjson.dumps(conv.model_dump(mode="json"))

        try:
            with self._get_transaction(write=True) as txn:
                # Store main data
                txn.put(storage_key.encode("utf-8"), value, overwrite=True)

                # Store hash -> key mapping for reverse lookup
                txn.put(
                    f"{self.HASH_LOOKUP_PREFIX}{message_hash}".encode("utf-8"),
                    storage_key.encode("utf-8"),
                )

                logger.debug(f"Stored {len(conv.messages)} messages with key: {storage_key}")
                return storage_key

        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
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

                storage_data = orjson.loads(data)  # type: ignore
                conv = ConversationInStore.model_validate(storage_data)

                logger.debug(f"Retrieved {len(conv.messages)} messages for key: {key}")
                return conv

        except Exception as e:
            logger.error(f"Failed to retrieve messages for key {key}: {e}")
            return None

    def find(self, model: str, messages: List[Message]) -> Optional[ConversationInStore]:
        """
        Search conversation data by message list.

        Args:
            model: Model name of the conversations
            messages: List of messages to search for

        Returns:
            Conversation or None if not found
        """
        if not messages:
            return None

        # --- Find with raw messages ---
        if conv := self._find_by_message_list(model, messages):
            logger.debug("Found conversation with raw message history.")
            return conv

        # --- Find with cleaned messages ---
        cleaned_messages = self.sanitize_assistant_messages(messages)
        if conv := self._find_by_message_list(model, cleaned_messages):
            logger.debug("Found conversation with cleaned message history.")
            return conv

        logger.debug("No conversation found for either raw or cleaned history.")
        return None

    def _find_by_message_list(
        self, model: str, messages: List[Message]
    ) -> Optional[ConversationInStore]:
        """Internal find implementation based on a message list."""
        for c in g_config.gemini.clients:
            message_hash = _hash_conversation(c.id, model, messages)

            key = f"{self.HASH_LOOKUP_PREFIX}{message_hash}"
            try:
                with self._get_transaction(write=False) as txn:
                    if mapped := txn.get(key.encode("utf-8")):  # type: ignore
                        return self.get(mapped.decode("utf-8"))  # type: ignore
            except Exception as e:
                logger.error(
                    f"Failed to retrieve messages by message list for hash {message_hash} and client {c.id}: {e}"
                )
                continue

            if conv := self.get(message_hash):
                return conv
        return None

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the store.

        Args:
            key: Storage key to check

        Returns:
            bool: True if key exists, False otherwise
        """
        try:
            with self._get_transaction(write=False) as txn:
                return txn.get(key.encode("utf-8")) is not None
        except Exception as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            return False

    def delete(self, key: str) -> Optional[ConversationInStore]:
        """
        Delete conversation model by key.

        Args:
            key: Storage key to delete

        Returns:
            ConversationInStore: The deleted conversation data, or None if not found
        """
        try:
            with self._get_transaction(write=True) as txn:
                # Get data first to clean up hash mapping
                data = txn.get(key.encode("utf-8"))
                if not data:
                    return None

                storage_data = orjson.loads(data)  # type: ignore
                conv = ConversationInStore.model_validate(storage_data)
                message_hash = _hash_conversation(conv.client_id, conv.model, conv.messages)

                # Delete main data
                txn.delete(key.encode("utf-8"))

                # Clean up hash mapping if it exists
                if message_hash and key != message_hash:
                    txn.delete(f"{self.HASH_LOOKUP_PREFIX}{message_hash}".encode("utf-8"))

                logger.debug(f"Deleted messages with key: {key}")
                return conv

        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return None

    def keys(self, prefix: str = "", limit: Optional[int] = None) -> List[str]:
        """
        List all keys in the store, optionally filtered by prefix.

        Args:
            prefix: Optional prefix to filter keys
            limit: Optional limit on number of keys returned

        Returns:
            List of keys
        """
        keys = []
        try:
            with self._get_transaction(write=False) as txn:
                cursor = txn.cursor()
                cursor.first()

                count = 0
                for key, _ in cursor:
                    key_str = key.decode("utf-8")
                    # Skip internal hash mappings
                    if key_str.startswith(self.HASH_LOOKUP_PREFIX):
                        continue

                    if not prefix or key_str.startswith(prefix):
                        keys.append(key_str)
                        count += 1

                        if limit and count >= limit:
                            break

        except Exception as e:
            logger.error(f"Failed to list keys: {e}")

        return keys

    def stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict with database statistics
        """
        if not self._env:
            logger.error("LMDB environment not initialized")
            return {}

        try:
            return self._env.stat()
        except Exception as e:
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
        """
        Remove <think>...</think> tags at the start of text and strip whitespace.
        """
        cleaned_content = re.sub(r"^(\s*<think>.*?</think>\n?)", "", text, flags=re.DOTALL)
        return cleaned_content.strip()

    @staticmethod
    def sanitize_assistant_messages(messages: list[Message]) -> list[Message]:
        """
        Create a new list of messages with assistant content cleaned of <think> tags.
        This is useful for store the chat history.
        """
        cleaned_messages = []
        for msg in messages:
            if msg.role == "assistant" and isinstance(msg.content, str):
                normalized_content = LMDBConversationStore.remove_think_tags(msg.content)
                # Only create a new object if content actually changed
                if normalized_content != msg.content:
                    cleaned_msg = Message(role=msg.role, content=normalized_content, name=msg.name)
                    cleaned_messages.append(cleaned_msg)
                else:
                    cleaned_messages.append(msg)
            else:
                cleaned_messages.append(msg)

        return cleaned_messages
