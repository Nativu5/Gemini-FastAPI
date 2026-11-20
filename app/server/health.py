from fastapi import APIRouter, Response, status
from loguru import logger

from app.models import HealthCheckResponse
from app.services import GeminiClientPool, LMDBConversationStore
from app.utils import g_config
from app.utils.config import GuestMode

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(response: Response):
    pool = GeminiClientPool()
    db = LMDBConversationStore()
    client_status = pool.status()
    stat = db.stats()

    if not all(client_status.values()):
        down_clients = [client_id for client_id, status in client_status.items() if not status]
        logger.warning(f"One or more Gemini clients are unhealthy: {', '.join(down_clients)}")

    if not stat:
        logger.error("Failed to retrieve LMDB conversation store stats")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthCheckResponse(
            ok=False, error="LMDB conversation store unavailable", clients=client_status
        )

    guest_mode = g_config.gemini.guest_mode
    if guest_mode == GuestMode.STRICT:
        clients_unavailable = not all(client_status.values())
        client_error = "One or more Gemini clients are unhealthy"
    else:
        clients_unavailable = guest_mode == GuestMode.ADAPTIVE and not any(client_status.values())
        client_error = "No usable Gemini client is available"

    if clients_unavailable:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthCheckResponse(
            ok=False,
            error=client_error,
            storage=stat,
            clients=client_status,
        )

    return HealthCheckResponse(ok=True, storage=stat, clients=client_status)
