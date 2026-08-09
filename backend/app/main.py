"""Feedback Lens FastAPI application."""

import os
import logging
import time

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse
from starlette.responses import Response

from backend.app.api import router
from backend.app.db import engine
from backend.app.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)
app = FastAPI(title="Feedback Lens API", version="0.1.0")
MAX_REQUEST_BODY_BYTES = 64 * 1024
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]


@app.middleware("http")
async def limit_request_size(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    """Reject unexpectedly large requests before parsing their bodies."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            body_size = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid Content-Length header"},
            )
        if body_size > MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={"detail": "Request body must be 64 KiB or smaller"},
            )
    return await call_next(request)


@app.middleware("http")
async def log_request(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Log method, route, status, and latency without logging request contents."""
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("api_error method=%s path=%s", request.method, request.url.path)
        raise
    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "api_request method=%s path=%s status=%s latency_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe: confirms the FastAPI process is running."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness probe: confirms the API can reach PostgreSQL."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as error:
        logger.warning("readiness_failed database_unavailable=true")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable",
        ) from error
    return {"status": "ready"}
