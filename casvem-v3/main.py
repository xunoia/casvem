from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.routes import router
from config import cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up encoder and storage on startup (avoids first-request latency spike)
    from core.encoder import get_encoder
    from core.storage import get_storage
    from core.memory.writer import get_bitmap
    from core.llm import get_llm_provider

    get_encoder()
    get_storage()
    get_bitmap()
    get_llm_provider()

    if cfg.cost_dashboard:
        from dashboard.live import start_dashboard_thread
        start_dashboard_thread()

    yield

    # Persist HNSW index on shutdown
    get_storage().close()


app = FastAPI(
    title="CaSVeM v3",
    description="Cached Smart Vector Memory — AI memory that gets cheaper as it scales.",
    version="3.0.0",
    lifespan=lifespan,
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
