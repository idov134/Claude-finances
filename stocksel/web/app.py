"""FastAPI transport for a cached, independently scheduled market scanner."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def create_app(cfg=None, feed_name=None, service=None):
    from ..config import load_config
    from .service import MarketService

    service = service or MarketService(cfg or load_config(), feed_name=feed_name)

    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="stocksel · Research desk", lifespan=lifespan)
    app.state.service = service
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(static / "index.html")

    @app.get("/api/health")
    async def health():
        snap = service.snapshot()
        return {"status": snap["status"], "ready": snap["status"] == "ready"}

    @app.get("/api/snapshot")
    async def snapshot():
        return service.snapshot()

    @app.get("/api/stocks/{ticker}")
    async def stock(ticker: str):
        snap = service.snapshot()
        for row in snap.get("recommendations", []):
            if row["ticker"] == ticker.upper():
                return row
        raise HTTPException(status_code=404, detail="Ticker is not in the scored eligible universe")

    return app
