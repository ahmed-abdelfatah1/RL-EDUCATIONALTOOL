"""Environment API routes.

Exposes HTTP endpoints for listing, resetting, and stepping environments.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.domain.environments import EnvName
from app.schemas.environment_schemas import EnvironmentListResponse, StateSnapshot
from app.services.environment_service import create_environment, list_environments


router = APIRouter(prefix="/envs", tags=["environments"])


@router.get("/", response_model=EnvironmentListResponse)
def get_environments() -> EnvironmentListResponse:
    """List all available environments.

    Returns:
        EnvironmentListResponse: List of environment metadata.
    """
    environments = list_environments()
    return EnvironmentListResponse(environments=environments)


@router.post("/{env_name}/reset", response_model=StateSnapshot)
def reset_environment(env_name: EnvName) -> StateSnapshot:
    """Reset an environment and return initial state.

    Args:
        env_name: Name of the environment to reset.

    Returns:
        StateSnapshot: Initial state snapshot for rendering.
    """
    try:
        env = create_environment(env_name)
        env.reset()
        state = env.render_state()
        return StateSnapshot(env_name=env_name, state=state)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Environment '{env_name}' not found")


@router.post("/{env_name}/step/{action}", response_model=StateSnapshot)
def step_environment(env_name: EnvName, action: int) -> StateSnapshot:
    """Take a single step in an environment.

    For debugging/manual stepping. Creates a fresh env each call.

    Args:
        env_name: Name of the environment.
        action: Action to take.

    Returns:
        StateSnapshot: State snapshot after step.
    """
    try:
        env = create_environment(env_name)
        env.reset()
        env.step(action)
        state = env.render_state()
        return StateSnapshot(env_name=env_name, state=state)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Environment '{env_name}' not found")

