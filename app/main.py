import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from loguru import logger

from .server.chat import router as chat_router
from .server.health import router as health_router
from .server.images import router as images_router
from .server.middleware import (
    add_cors_middleware,
    add_exception_handler,
    cleanup_expired_images,
)
from .services import GeminiClientPool, LMDBConversationStore

RETENTION_CLEANUP_INTERVAL_SECONDS = 6 * 60 * 60  # 6 hours


async def _run_retention_cleanup(stop_event: asyncio.Event) -> None:
    """
    Periodically enforce LMDB retention policy until the stop_event is set.
    """
    store = LMDBConversationStore()
    if store.retention_days <= 0:
        logger.info("LMDB retention cleanup disabled; skipping scheduler.")
        return

    logger.info(
        f"Starting LMDB retention cleanup task (retention={store.retention_days} day(s), interval={RETENTION_CLEANUP_INTERVAL_SECONDS} seconds)."
    )

    while not stop_event.is_set():
        try:
            store.cleanup_expired()
            cleanup_expired_images(store.retention_days)
        except Exception:
            logger.exception("LMDB retention cleanup task failed.")

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=RETENTION_CLEANUP_INTERVAL_SECONDS,
            )
        except asyncio.TimeoutError:
            continue

    logger.info("LMDB retention cleanup task stopped.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_stop_event = asyncio.Event()

    pool = GeminiClientPool()
    try:
        await pool.init()
    except Exception as e:
        logger.exception(f"Failed to initialize Gemini clients: {e}")
        raise

    cleanup_task = asyncio.create_task(_run_retention_cleanup(cleanup_stop_event))
    # Give the cleanup task a chance to start and surface immediate failures.
    await asyncio.sleep(0)
    if cleanup_task.done():
        try:
            cleanup_task.result()
        except Exception:
            logger.exception("LMDB retention cleanup task failed to start.")
            raise

    logger.info(f"Gemini clients initialized: {[c.id for c in pool.clients]}.")
    logger.info("Gemini API Server ready to serve requests.")

    try:
        yield
    finally:
        cleanup_stop_event.set()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            logger.debug("LMDB retention cleanup task cancelled during shutdown.")
        except Exception:
            logger.exception(
                "LMDB retention cleanup task terminated with an unexpected error during shutdown."
            )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gemini API Server",
        description="OpenAI-compatible API for Gemini Web",
        version="1.0.0",
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
    )

    add_cors_middleware(app)
    add_exception_handler(app)

    app.include_router(health_router, tags=["Health"])
    app.include_router(chat_router, tags=["Chat"])
    app.include_router(images_router, tags=["Images"])

    return app
