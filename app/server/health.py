from fastapi import APIRouter
from loguru import logger

from ..models import HealthCheckResponse
from ..services import GeminiClientPool, LMDBConversationStore

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    pool = GeminiClientPool()
    db = LMDBConversationStore()

    try:
        await pool.init()
    except Exception as e:
        logger.error(f"Failed to initialize Gemini clients: {e}")
        return HealthCheckResponse(ok=False, error=str(e))

    client_status = pool.status()

    if not all(client_status.values()):
        logger.warning("One or more Gemini clients not running")

    stat = db.stats()
    if not stat:
        logger.error("Failed to retrieve LMDB conversation store stats")
        return HealthCheckResponse(
            ok=False, error="LMDB conversation store unavailable", clients=client_status
        )

    return HealthCheckResponse(ok=all(client_status.values()), storage=stat, clients=client_status)
