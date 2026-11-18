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
        self._clients: list[GeminiClientWrapper] = []
        self._id_map: dict[str, GeminiClientWrapper] = {}
        self._round_robin: deque[GeminiClientWrapper] = deque()
        self._restart_locks: dict[str, asyncio.Lock] = {}

        if len(g_config.gemini.clients) == 0:
            raise ValueError("No Gemini clients configured")

        for c in g_config.gemini.clients:
            client = GeminiClientWrapper(
                client_id=c.id,
                **c.model_dump(exclude={"id"}),
            )
            self._clients.append(client)
            self._id_map[c.id] = client
            self._round_robin.append(client)
            self._restart_locks[c.id] = asyncio.Lock()

    async def _init_one(self, client: GeminiClientWrapper) -> bool:
        """Initialize a single client; returns True on success."""
        return await self._init_attempt(client)

    async def _init_attempt(self, client: GeminiClientWrapper) -> bool:
        """Run library init; returns True on success."""
        try:
            await client.init()
            return True
        except Exception:
            return False

    async def init(self) -> None:
        """Initialize all clients in the pool with staggered start times."""
        clients_to_init = [c for c in self._clients if not c.running()]
        for i, client in enumerate(clients_to_init):
            await self._init_one(client)

            if i < len(clients_to_init) - 1:
                delay = random.uniform(5, 30)
                logger.info(f"Staggering next initialization by {delay:.2f}s")
                await asyncio.sleep(delay)

        success_count = sum(bool(client.running()) for client in self._clients)
        if success_count == 0:
            raise RuntimeError("Failed to initialize any Gemini clients")

    async def acquire(
        self, client_id: str | None = None, require_account: bool = False
    ) -> GeminiClientWrapper:
        """Return a healthy client by id or using round-robin.

        `require_account` excludes guest sessions, for requests they cannot serve at all - file
        uploads. Otherwise a guest is used only once no authenticated client is left.
        """
        if not self._round_robin:
            raise RuntimeError("No Gemini clients configured")

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

            if account_only and not require_account and any(c.is_guest() for c in self._clients):
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

        lock = self._restart_locks.get(client.id)
        if lock is None:
            return False

        async with lock:
            if client.running():
                return True

            if await self._init_attempt(client):
                logger.info(f"Restarted Gemini client {client.id} after it stopped.")
                return True
            return False

    @property
    def clients(self) -> list[GeminiClientWrapper]:
        """Return managed clients."""
        return self._clients

    async def close(self) -> None:
        """Close all clients in the pool."""
        if not self._clients:
            return

        logger.info(f"Closing {len(self._clients)} Gemini clients...")
        await asyncio.gather(
            *(client.close() for client in self._clients if client.running()),
            return_exceptions=True,
        )
        logger.info("All Gemini clients closed.")

    def status(self) -> dict[str, bool]:
        """Return healthy status for each client."""
        return {client.id: client.is_healthy() for client in self._clients}
