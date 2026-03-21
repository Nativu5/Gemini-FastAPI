from fastapi import APIRouter
from loguru import logger

from app.models import HealthCheckResponse
from app.services import GeminiClientPool, LMDBConversationStore

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    pool = GeminiClientPool()
    db = LMDBConversationStore()
    client_status = pool.status()
    stat = db.stats()

    if not all(client_status.values()):
        down_clients = [client_id for client_id, status in client_status.items() if not status]
        logger.warning(f"One or more Gemini clients not running: {', '.join(down_clients)}")

    if not stat:
        logger.error("Failed to retrieve LMDB conversation store stats")
        return HealthCheckResponse(
            ok=False, error="LMDB conversation store unavailable", clients=client_status
        )

    return HealthCheckResponse(ok=all(client_status.values()), storage=stat, clients=client_status)
