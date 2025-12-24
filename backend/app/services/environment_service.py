"""Environment service - exposes available environments and creates instances.

Provides service layer between API routes and domain environments.
"""

from __future__ import annotations

from typing import Dict

from app.domain.environments import EnvName, create_env
from app.schemas.environment_schemas import EnvironmentInfo


ENV_METADATA: Dict[EnvName, EnvironmentInfo] = {
    "gridworld": EnvironmentInfo(
        name="gridworld",
        display_name="GridWorld",
        supports_discrete=True,
        supports_continuous=False,
        description="Simple 2D grid for tabular RL.",
    ),
    "cartpole": EnvironmentInfo(
        name="cartpole",
        display_name="CartPole",
        supports_discrete=True,
        supports_continuous=False,
        description="Classic control problem balancing a pole.",
    ),
    "mountaincar": EnvironmentInfo(
        name="mountaincar",
        display_name="MountainCar",
        supports_discrete=True,
        supports_continuous=False,
        description="Underpowered car must build momentum to climb a hill.",
    ),
    "frozenlake": EnvironmentInfo(
        name="frozenlake",
        display_name="FrozenLake",
        supports_discrete=True,
        supports_continuous=False,
        description="Gridworld with slippery ice, holes, and a goal.",
    ),
    "gym4real": EnvironmentInfo(
        name="gym4real",
        display_name="Gym4Real",
        supports_discrete=True,
        supports_continuous=False,
        description="Abstract 2D point robot environment.",
    ),
}



def list_environments() -> list[EnvironmentInfo]:
    """Get list of all available environments.

    Returns:
        list[EnvironmentInfo]: Metadata for all environments.
    """
    return list(ENV_METADATA.values())


def get_environment_info(env_name: EnvName) -> EnvironmentInfo | None:
    """Get metadata for a specific environment.

    Args:
        env_name: Name of the environment.

    Returns:
        EnvironmentInfo | None: Metadata if found, None otherwise.
    """
    return ENV_METADATA.get(env_name)


def create_environment(env_name: EnvName):
    """Create a new environment instance.

    Args:
        env_name: Name of the environment to create.

    Returns:
        BaseEnv: New environment instance.
    """
    return create_env(env_name)

