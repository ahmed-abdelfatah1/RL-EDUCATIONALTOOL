"""Algorithm API routes.

Exposes read-only metadata about available algorithms and hyperparameters.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.algorithm_schemas import AlgorithmInfo, AlgorithmListResponse
from app.services.algorithm_registry import get_algorithm, list_algorithms


router = APIRouter(prefix="/algorithms", tags=["algorithms"])


@router.get("/", response_model=AlgorithmListResponse)
def get_algorithms() -> AlgorithmListResponse:
    """List all available algorithms.

    Returns:
        AlgorithmListResponse: List of algorithm metadata.
    """
    algorithms = list_algorithms()
    return AlgorithmListResponse(algorithms=algorithms)


@router.get("/{name}", response_model=AlgorithmInfo)
def get_algorithm_by_name(name: str) -> AlgorithmInfo:
    """Get metadata for a specific algorithm.

    Args:
        name: Algorithm name.

    Returns:
        AlgorithmInfo: Algorithm metadata.
    """
    try:
        return get_algorithm(name)
    except KeyError:
        raise HTTPException(status_code=404, detail="Algorithm not found")

