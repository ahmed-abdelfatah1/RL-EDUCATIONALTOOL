"""Algorithm schemas - Pydantic models for algorithm metadata.

Describes available algorithms and their hyperparameters for the frontend.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class HyperparameterInfo(BaseModel):
    """Metadata for a single hyperparameter."""

    name: str
    display_name: str
    type: str
    default: float | int
    min: float | int
    max: float | int
    step: float | int


class AlgorithmInfo(BaseModel):
    """Metadata for an available algorithm."""

    name: str
    display_name: str
    description: str
    supports_envs: List[str]
    hyperparameters: List[HyperparameterInfo]


class AlgorithmListResponse(BaseModel):
    """Response containing list of available algorithms."""

    algorithms: List[AlgorithmInfo]

