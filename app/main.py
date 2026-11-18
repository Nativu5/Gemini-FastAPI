import asyncio
import contextlib
import mimetypes
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .server.chat import available_models
from .server.chat import router as chat_router
from .server.gemini import add_exception_handlers
from .server.gemini import router as gemini_router
from .server.health import router as health_router
from .server.media import router as media_router
from .server.middleware import RequestBodyLimitMiddleware, cleanup_expired_media
from .services import GeminiClientPool, LMDBConversationStore
from .utils import g_config
from .utils.config import ChatMode, GeminiGemPolicyMode

# Canonical audio MIME types: Python's platform mapping yields legacy "x-"
# types (audio/x-wav, audio/x-aac, audio/x-flac) that Google's upload
# endpoint does not classify as audio, so the attached file never reaches
# the model as an audible attachment. Registered at import time, before any upload.
for _ext, _mime in (
    (".wav", "audio/wav"),
    (".aac", "audio/aac"),
    (".flac", "audio/flac"),
    (".m4a", "audio/mp4"),
):
    with contextlib.suppress(ValueError):
        mimetypes.add_type(_mime, _ext)

RETENTION_CLEANUP_INTERVAL_SECONDS = 6 * 60 * 60


def _get_cleanup_interval() -> int:
    """Seconds between cleanup passes: 6 hours, or half of `auto_delete_ttl` (at least a minute)
    when chats are deleted after that long."""
    interval = RETENTION_CLEANUP_INTERVAL_SECONDS
    if g_config.gemini.chat_mode == ChatMode.AUTO_DELETE:
        interval = min(interval, max(60, g_config.gemini.auto_delete_ttl // 2))
    return interval


async def _run_retention_cleanup() -> None:
    """Enforce local retention, remote chat deletion, media expiry and managed-gem cleanup, at
    startup and then periodically, until cancelled."""
    store = LMDBConversationStore()
    pool = GeminiClientPool()
    gems = g_config.gemini.gems
    gem_cleanup = gems.policy != GeminiGemPolicyMode.OFF and gems.cleanup.enabled
    next_gem_cleanup = 0.0

    if (
        store.retention_days <= 0
        and g_config.gemini.chat_mode != ChatMode.AUTO_DELETE
        and not store.get_pending_deletions()
        and not store.has_auto_delete_records()
        and not gem_cleanup
    ):
        logger.info("LMDB retention cleanup disabled; skipping scheduler.")
        return

    interval = _get_cleanup_interval()
    logger.info(
        f"Starting LMDB retention cleanup task (retention={store.retention_days} day(s), "
        f"auto_delete_ttl={g_config.gemini.auto_delete_ttl}s, interval={interval}s)."
    )

    while True:
        try:
            _, to_delete = store.cleanup_expired()
            for client_id, cid in to_delete:
                if await pool.delete_chat(client_id, cid):
                    store.remove_pending_deletion(cid)
                else:
                    store.record_pending_deletion_failure(cid)
                await asyncio.sleep(0.1)

            if g_config.gemini.chat_mode == ChatMode.AUTO_DELETE:
                cleanup_expired_media(g_config.gemini.auto_delete_ttl)
            else:
                cleanup_expired_media(store.retention_days * 24 * 60 * 60)
        except Exception:
            logger.exception("LMDB retention cleanup task failed.")

        if gem_cleanup and time.monotonic() >= next_gem_cleanup:
            # On its own clock: in auto_delete mode this loop can come round every minute.
            next_gem_cleanup = time.monotonic() + RETENTION_CLEANUP_INTERVAL_SECONDS
            await pool.delete_unused_gems()

        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = GeminiClientPool()
    try:
        await pool.init()
        available_models(pool)  # listed now, so `created` is the startup time
    except Exception as e:
        logger.exception(f"Failed to initialize Gemini clients: {e}")
        raise

    try:
        LMDBConversationStore().prune_stale_indexes()
    except Exception:
        logger.exception("Failed to prune stale LMDB indexes; continuing with startup.")

    cleanup_task = asyncio.create_task(_run_retention_cleanup())
    await asyncio.sleep(0)  # let it take its first step, so that a failure there stops the startup
    if cleanup_task.done():
        cleanup_task.result()

    if (gem_policy := g_config.gemini.gems.policy) != GeminiGemPolicyMode.OFF:
        logger.warning(
            f"Gems are enabled (gemini.gems.policy={gem_policy.value}), but Google is retiring "
            "them: personal accounts from 2026-11-17 (removed 2026-12-17), Workspace accounts in "
            "March 2027, education accounts in June 2027. "
            'See https://support.google.com/gemini/answer/18560919. Set the policy to "off" '
            "once the account has lost them: a gem request that Google rejects is retried for up "
            "to about 15 seconds, restarting the client each time, before the system prompt is "
            "sent inline. That happens on every request."
        )

    logger.info(f"Gemini clients initialized: {[c.id for c in pool.clients]}.")
    logger.info("Gemini API Server ready to serve requests.")

    try:
        yield
    finally:
        cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task

        try:
            await pool.close()
        except Exception:
            logger.exception("Failed to close Gemini client pool gracefully.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gemini API Server",
        description="OpenAI-compatible API for Gemini Web",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Order matters: the last middleware added is the outermost, so the body limit has to be
    # registered first for its 413 to still pass back out through CORS.
    app.add_middleware(
        RequestBodyLimitMiddleware, max_body_bytes=g_config.server.max_request_body_bytes
    )
    if (cors := g_config.cors).enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors.allow_origins,
            allow_credentials=cors.allow_credentials,
            allow_methods=cors.allow_methods,
            allow_headers=cors.allow_headers,
        )
    add_exception_handlers(app)

    app.include_router(health_router, tags=["Health"])
    app.include_router(chat_router, tags=["Chat"])
    app.include_router(media_router, tags=["Media"])
    app.include_router(gemini_router, tags=["Gemini"])

    return app
