"""Worker process entrypoint: consumer loop + janitor with graceful shutdown."""

from __future__ import annotations

import asyncio
import signal
import sys

import structlog

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.client import close_mongo, get_db
from app.db.redis import close_redis
from app.services.embedder import get_embedder
from app.services.vector_store import get_vector_store
from app.worker.consumer import run_consumer_loop
from app.worker.janitor import run_janitor_loop

log = structlog.get_logger(__name__)


async def main() -> None:
    """Run worker consumer loop and janitor task with graceful shutdown handling."""
    configure_logging()
    settings = get_settings()

    log.info(
        "Starting DocuGround Ingestion Worker",
        env=settings.app_env,
        consumer_group=settings.consumer_group,
        consumer_name=settings.consumer_name,
    )

    db = get_db()
    vector_store = get_vector_store()
    embedder = get_embedder()

    stop_event = asyncio.Event()

    def _handle_shutdown_signal(*_: object) -> None:
        log.info("Shutdown signal received — initiating graceful worker shutdown")
        stop_event.set()

    # Cross-platform signal registration
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _handle_shutdown_signal)
    else:
        # Windows fallback
        signal.signal(signal.SIGINT, _handle_shutdown_signal)
        signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    # Run consumer loop and janitor concurrently
    consumer_task = asyncio.create_task(
        run_consumer_loop(db, vector_store, embedder, stop_event=stop_event)
    )
    janitor_task = asyncio.create_task(
        run_janitor_loop(db, vector_store, stop_event=stop_event)
    )

    try:
        await asyncio.gather(consumer_task, janitor_task)
    except asyncio.CancelledError:
        log.info("Worker tasks cancelled")
    finally:
        log.info("Closing database and redis connections")
        await close_mongo()
        await close_redis()
        log.info("Worker shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
