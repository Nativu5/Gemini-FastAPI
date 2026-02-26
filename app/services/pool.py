import asyncio
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
                secure_1psid=c.secure_1psid,
                secure_1psidts=c.secure_1psidts,
                proxy=c.proxy,
            )
            self._clients.append(client)
            self._id_map[c.id] = client
            self._round_robin.append(client)
            self._restart_locks[c.id] = asyncio.Lock()

    async def init(self) -> None:
        """Initialize all clients in the pool."""
        success_count = 0
        for client in self._clients:
            if not client.running():
                try:
                    await client.init(
                        timeout=g_config.gemini.timeout,
                        watchdog_timeout=g_config.gemini.watchdog_timeout,
                        auto_refresh=g_config.gemini.auto_refresh,
                        verbose=g_config.gemini.verbose,
                        refresh_interval=g_config.gemini.refresh_interval,
                    )
                except Exception:
                    logger.exception(f"Failed to initialize client {client.id}")

            if client.running():
                success_count += 1

        if success_count == 0:
            raise RuntimeError("Failed to initialize any Gemini clients")

    async def acquire(self, client_id: str | None = None) -> GeminiClientWrapper:
        """Return a healthy client by id or using round-robin."""
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

        for _ in range(len(self._round_robin)):
            client = self._round_robin[0]
            self._round_robin.rotate(-1)
            if await self._ensure_client_ready(client):
                return client

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

            try:
                await client.init(
                    timeout=g_config.gemini.timeout,
                    watchdog_timeout=g_config.gemini.watchdog_timeout,
                    auto_refresh=g_config.gemini.auto_refresh,
                    verbose=g_config.gemini.verbose,
                    refresh_interval=g_config.gemini.refresh_interval,
                )
                logger.info(f"Restarted Gemini client {client.id} after it stopped.")
                return True
            except Exception:
                logger.exception(f"Failed to restart Gemini client {client.id}")
                return False

    @property
    def clients(self) -> list[GeminiClientWrapper]:
        """Return managed clients."""
        return self._clients

    def status(self) -> dict[str, bool]:
        """Return running status for each client."""
        return {client.id: client.running() for client in self._clients}
