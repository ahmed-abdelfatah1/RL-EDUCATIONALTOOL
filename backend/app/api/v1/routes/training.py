"""Training API routes.

Exposes HTTP endpoints for running training sessions.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.training_schemas import TrainingRequest, TrainingRunResponse
from app.services.training_service import run_training


router = APIRouter(prefix="/train", tags=["training"])


@router.post("/run", response_model=TrainingRunResponse)
def run_training_endpoint(request: TrainingRequest) -> TrainingRunResponse:
    """Run a synchronous training session.

    Args:
        request: Training configuration.

    Returns:
        TrainingRunResponse: Results with episode metrics.
    """
    try:
        return run_training(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Not found: {exc}")

