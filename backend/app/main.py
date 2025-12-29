"""FastAPI application entry point.

Configures CORS, includes routers, and creates the main app instance.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes.algorithms import router as algorithms_router
from app.api.v1.routes.environments import router as envs_router
from app.api.v1.routes.stream import router as stream_router
from app.api.v1.routes.training import router as training_router
from app.core.config import settings


app = FastAPI(
    title="RL Education Tool",
    description="Interactive web-based RL learning tool",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(algorithms_router, prefix=settings.api_v1_prefix)
app.include_router(envs_router, prefix=settings.api_v1_prefix)
app.include_router(stream_router, prefix=settings.api_v1_prefix)
app.include_router(training_router, prefix=settings.api_v1_prefix)


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}

