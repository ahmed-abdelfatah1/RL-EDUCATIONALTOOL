"""Environment schemas - Pydantic models for environment data.

Defines models for environment metadata and state snapshots.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class EnvironmentInfo(BaseModel):
    """Metadata about an available environment."""

    name: str
    display_name: str
    supports_discrete: bool
    supports_continuous: bool
    description: str


class StateSnapshot(BaseModel):
    """Snapshot of environment state for frontend rendering."""

    env_name: str
    state: Dict[str, Any]


class EnvironmentListResponse(BaseModel):
    """Response containing list of available environments."""

    environments: List[EnvironmentInfo]


class EnvironmentResetRequest(BaseModel):
    """Request to reset an environment."""

    env_name: str


class EnvironmentResetResponse(BaseModel):
    """Response after resetting an environment."""

    env_name: str
    initial_state: Dict[str, Any]
    render_state: Dict[str, Any]


class EnvironmentStepRequest(BaseModel):
    """Request to take a step in an environment."""

    env_name: str
    action: int


class EnvironmentStepResponse(BaseModel):
    """Response after taking a step in an environment."""

    env_name: str
    state: Dict[str, Any]
    reward: float
    done: bool
    render_state: Dict[str, Any]

