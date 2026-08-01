"""Feedback Lens FastAPI application."""

from fastapi import FastAPI

from backend.app.api import router


app = FastAPI(title="Feedback Lens API", version="0.1.0")
app.include_router(router)
