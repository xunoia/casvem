"""
CaSVeM entry point.

Start everything:
  1. Weaviate store (graph + vector DB)
  2. Ollama providers (LLM + embedder)
  3. APScheduler (promotion/demotion engine)
  4. FastAPI (HTTP API)
"""

import logging
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(name)-25s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("casvem.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    log.info("Starting CaSVeM...")

    from database.weaviate_store import init_store
    from providers.router import init_providers
    import scheduler as sched

    try:
        init_store()
        log.info("Weaviate connected.")
    except Exception as e:
        log.error("Weaviate connection failed: %s", e)
        log.error("Make sure Weaviate is running:  docker compose up -d")
        raise

    try:
        init_providers()
        log.info("Ollama providers initialised.")
    except Exception as e:
        log.error("Ollama init failed: %s", e)
        log.error("Make sure Ollama is running:  ollama serve")
        raise

    sched.start()

    log.info("CaSVeM ready.  API: http://localhost:8000")
    log.info("Docs:          http://localhost:8000/docs")

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    log.info("Shutting down CaSVeM...")
    sched.stop()

    from providers.router import close_providers
    from database.weaviate_store import close_store
    await close_providers()
    close_store()
    log.info("CaSVeM stopped.")


# Wire the lifespan into the existing FastAPI app
from api import app
app.router.lifespan_context = lifespan


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = False,
        log_level = "info",
    )
