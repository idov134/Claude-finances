"""FastAPI transport for a cached, independently scheduled market scanner."""
from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware


class PreviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    ticker: str = Field(pattern=r"^[A-Z0-9.\-]{1,32}$")
    position_pct: float = Field(ge=0, le=100)
    thesis: str = Field(default="", max_length=2000)
    counter_thesis: str = Field(default="", max_length=2000)


class SubmitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preview_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    idempotency_key: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_\-]+$")


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
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
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
                return await asyncio.to_thread(service.stock_detail, row) if hasattr(service, "stock_detail") else row
        raise HTTPException(status_code=404, detail="Ticker is not in the scored eligible universe")

    def mutation_guard(request):
        if request.headers.get("X-Stocksel-Request") != "paper":
            raise HTTPException(403, "Paper actions require the local application request header")
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "Cross-origin paper actions are refused")

    async def paper_action(request, fn, *args):
        mutation_guard(request)
        if not hasattr(service, "paper_lock"):
            raise HTTPException(409, "Paper workflow unavailable")
        async with service.paper_lock:
            try:
                result = await asyncio.to_thread(fn, *args)
                service._portfolio = await asyncio.to_thread(service._read_portfolio)
                return result
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.post("/api/paper/preview")
    async def preview(body: PreviewInput, request: Request):
        from .paper import PaperWorkflow
        return await paper_action(request, PaperWorkflow(service).preview, body.ticker, body.position_pct,
                                  body.thesis, body.counter_thesis)

    @app.post("/api/paper/submit")
    async def submit(body: SubmitInput, request: Request):
        from .paper import PaperWorkflow
        return await paper_action(request, PaperWorkflow(service).submit, body.preview_id, body.idempotency_key)

    @app.post("/api/paper/advance")
    async def advance(request: Request):
        mutation_guard(request)
        async with service.paper_lock:
            try:
                return await service.advance_paper()
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.get("/api/paper/export")
    async def export():
        from ..live.paper import PaperLedger
        return (await asyncio.to_thread(PaperLedger, service.cfg)).state

    @app.get("/api/research/comparison")
    async def comparison():
        import json
        path = Path(__file__).resolve().parents[2] / "docs" / "research_comparison" / "comparison.json"
        if not path.is_file():
            raise HTTPException(503, "Comparison evidence is not installed; see the documented reproduction command")
        return json.loads(path.read_text())

    return app
