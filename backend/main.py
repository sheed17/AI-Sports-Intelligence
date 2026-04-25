from pathlib import Path
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from backend.config import get_settings
from backend.db.database import init_db
from backend.api import videos, events, clips, ask, jobs, evals

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    await init_db()
    logger.info("ai_film_room.startup", environment=settings.environment)
    yield
    logger.info("ai_film_room.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI Film Room",
        description="Soccer match footage analysis with YOLO, PyTorch, and LangGraph",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.environment == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(videos.router, prefix="/videos", tags=["videos"])
    app.include_router(events.router, prefix="/videos", tags=["events"])
    app.include_router(clips.router, tags=["clips"])
    app.include_router(ask.router, prefix="/videos", tags=["ask"])
    app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
    app.include_router(evals.router, prefix="/evals", tags=["evals"])

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "data": None,
                "error": {
                    "status_code": exc.status_code,
                    "message": exc.detail,
                },
                "meta": {"path": request.url.path},
            },
            headers=exc.headers,
        )

    @app.get("/health", tags=["health"])
    async def health():
        return {"data": {"status": "ok", "version": "0.1.0"}, "error": None, "meta": {}}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard():
        html_path = Path(__file__).parent / "ui" / "index.html"
        return HTMLResponse(html_path.read_text())

    return app


app = create_app()
