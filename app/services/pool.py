import asyncio
import random
from collections import deque

from loguru import logger

from app.utils import g_config
from app.utils.singleton import Singleton

from .client import GeminiClientWrapper


class GeminiClientPool(metaclass=Singleton):
    """Pool of GeminiClient instances identified by unique ids."""

    def __init__(self) -> None:
        if not g_config.gemini.clients:
            raise ValueError("No Gemini clients configured")

        self.clients = [
            GeminiClientWrapper(client_id=c.id, **c.model_dump(exclude={"id"}))
            for c in g_config.gemini.clients
        ]
        self._id_map = {client.id: client for client in self.clients}
        self._round_robin = deque(self.clients)

    @staticmethod
    async def _start(client: GeminiClientWrapper) -> bool:
        """Initialize a client; False if that failed, which `client.init` has logged."""
        try:
            await client.init()
            return True
        except Exception:
            return False

    async def init(self) -> None:
        """Initialize all clients in the pool with staggered start times."""
        clients_to_init = [c for c in self.clients if not c.running()]
        for i, client in enumerate(clients_to_init):
            await self._start(client)

            if i < len(clients_to_init) - 1:
                delay = random.uniform(5, 30)
                logger.info(f"Staggering next initialization by {delay:.2f}s")
                await asyncio.sleep(delay)

        if not any(client.running() for client in self.clients):
            raise RuntimeError("Failed to initialize any Gemini clients")

    async def acquire(
        self, client_id: str | None = None, require_account: bool = False
    ) -> GeminiClientWrapper:
        """Return a running client: the one with `client_id`, or the next in turn.

        `require_account` keeps guest sessions out of the rotation, for requests they cannot
        serve at all - file uploads. Otherwise a guest is used only once no authenticated client
        is left. A client asked for by id is returned whatever it is; the caller checks.
        """
        if client_id:
            client = self._id_map.get(client_id)
            if not client:
                raise ValueError(f"Client id {client_id} not found")
            if await self._ensure_client_ready(client):
                return client
            raise RuntimeError(
                f"Gemini client {client_id} is not running and could not be restarted"
            )

        # Authenticated clients first. A client whose cookies expired keeps answering text
        # prompts as a guest, so it stays usable and must not take the pool down, but it has no
        # history, no uploads and no model choice - traffic belongs elsewhere while it can.
        for account_only in (True,) if require_account else (True, False):
            for _ in range(len(self._round_robin)):
                client = self._round_robin[0]
                self._round_robin.rotate(-1)
                # Rechecked after readiness: a restart can itself land in a guest session.
                if account_only and client.is_guest():
                    continue
                if await self._ensure_client_ready(client) and not (
                    account_only and client.is_guest()
                ):
                    return client

            if account_only and not require_account and any(c.is_guest() for c in self.clients):
                logger.warning(
                    "No authenticated Gemini client is available; falling back to a guest "
                    "session until cookies are refreshed."
                )

        if require_account:
            raise RuntimeError(
                "No authenticated Gemini client is available. This request needs a file upload, "
                "which a guest session cannot do - refresh the client cookies."
            )
        raise RuntimeError("No Gemini clients are currently available")

    async def _ensure_client_ready(self, client: GeminiClientWrapper) -> bool:
        """Make sure the client is running, attempting a restart if needed."""
        if client.running():
            return True
        # The library's `init` holds a lock, and returns early when the client is already running.
        if await self._start(client):
            logger.info(f"Restarted Gemini client {client.id} after it stopped.")
            return True
        return False

    async def close(self) -> None:
        logger.info(f"Closing {len(self.clients)} Gemini clients...")
        await asyncio.gather(
            *(client.close() for client in self.clients if client.running()),
            return_exceptions=True,
        )
        logger.info("All Gemini clients closed.")

    def status(self) -> dict[str, bool]:
        return {client.id: client.is_healthy() for client in self.clients}

    async def delete_unused_gems(self) -> None:
        """Run the managed-gem cleanup on every account; one failing does not stop the rest."""
        for client in self.clients:
            try:
                if await self._ensure_client_ready(client):
                    await client.delete_unused_gems()
            except Exception as e:
                logger.warning(f"Managed gem cleanup failed for client {client.id}: {e}")

    async def delete_chat(self, client_id: str, cid: str) -> bool:
        """Delete a chat from its client's Google account. False means try again later; True
        that it is settled: deleted, or its client is no longer configured."""
        client = self._id_map.get(client_id)
        if not client:
            logger.warning(
                f"Cannot delete chat {cid}: client {client_id} not found in pool. "
                "Dropping pending deletion."
            )
            return True
        if client.is_guest():
            logger.warning(
                f"Cannot delete remote chat {cid}: client {client_id} is currently unauthenticated (guest session)"
            )
            return False
        try:
            if not await self._ensure_client_ready(client):
                logger.warning(
                    f"Cannot delete chat {cid}: client {client_id} is stopped and could not be restarted"
                )
                return False
            await client.delete_chat(cid)
            if client.latest_chat_cid == cid:
                client.latest_chat_cid = None
            logger.info(f"Deleted remote chat {cid} on Google for client {client_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete remote chat {cid} for client {client_id}: {e}")
            return False
