from collections import deque
from typing import Dict, List, Optional

from ..utils import g_config
from ..utils.singleton import Singleton
from .client import GeminiClientWrapper


class GeminiClientPool(metaclass=Singleton):
    """Pool of GeminiClient instances identified by unique ids."""

    def __init__(self) -> None:
        self._clients: List[GeminiClientWrapper] = []
        self._id_map: Dict[str, GeminiClientWrapper] = {}
        self._round_robin: deque[GeminiClientWrapper] = deque()

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

    async def init(self) -> None:
        """Initialize all clients in the pool."""
        for client in self._clients:
            if not client.running:
                await client.init(
                    timeout=g_config.gemini.timeout,
                    auto_refresh=g_config.gemini.auto_refresh,
                    verbose=g_config.gemini.verbose,
                    refresh_interval=g_config.gemini.refresh_interval,
                )

    def acquire(self, client_id: Optional[str] = None) -> GeminiClientWrapper:
        """Return a client by id or using round-robin."""
        if client_id:
            client = self._id_map.get(client_id)
            if not client:
                raise ValueError(f"Client id {client_id} not found")
            return client

        client = self._round_robin[0]
        self._round_robin.rotate(-1)
        return client

    @property
    def clients(self) -> List[GeminiClientWrapper]:
        """Return managed clients."""
        return self._clients

    def status(self) -> Dict[str, bool]:
        """Return running status for each client."""
        return {client.id: client.running for client in self._clients}
