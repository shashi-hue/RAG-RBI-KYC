import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.middleware import RequestLoggingMiddleware
from src.api.routes import query, health
from src.api.dependencies import get_chain
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models on startup — not on first request."""
    log.info("Warming up KYCChain...")
    get_chain()          # loads bge-large + bge-reranker + connects qdrant
    log.info("Startup complete.")
    yield
    log.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title       = "RBI KYC Master Direction RAG API",
        description = "Regulatory Q&A over RBI Master Direction KYC 2016 (updated 2025)",
        version     = "1.0.0",
        lifespan    = lifespan,
    )

    # Middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],   # tighten in prod
        allow_methods  = ["GET", "POST"],
        allow_headers  = ["*"],
    )

    TEMPLATES_DIR = Path(__file__).parent / "templates"

    @app.get("/")
    async def frontend():
        return FileResponse(TEMPLATES_DIR / "frontend.html")

    # Routers
    app.include_router(health.router)
    app.include_router(query.router)

    return app


app = create_app()
