"""Environment registry and factory for RL environments.

Provides centralized access to all environment implementations.
"""

from typing import Literal

from .base_env import BaseEnv
from .cartpole_env import CartPoleEnv
from .frozenlake_env import FrozenLakeEnv
from .gridworld_env import GridWorldEnv
from .gym4real_env import Gym4RealEnv
from .mountaincar_env import MountainCarEnv


EnvName = Literal[
    "gridworld",
    "cartpole",
    "mountaincar",
    "frozenlake",
    "gym4real",
]

ENV_REGISTRY: dict[EnvName, type[BaseEnv]] = {
    "gridworld": GridWorldEnv,
    "cartpole": CartPoleEnv,
    "mountaincar": MountainCarEnv,
    "frozenlake": FrozenLakeEnv,
    "gym4real": Gym4RealEnv,
}


def create_env(env_name: EnvName) -> BaseEnv:
    """Factory to create a new environment instance by name.

    Args:
        env_name: Name of the environment to create.

    Returns:
        BaseEnv: New instance of the requested environment.

    Raises:
        KeyError: If env_name is not in the registry.
    """
    return ENV_REGISTRY[env_name]()


def list_environments() -> list[str]:
    """Return list of available environment names."""
    return list(ENV_REGISTRY.keys())


__all__ = [
    "BaseEnv",
    "GridWorldEnv",
    "CartPoleEnv",
    "MountainCarEnv",
    "FrozenLakeEnv",
    "Gym4RealEnv",
    "EnvName",
    "ENV_REGISTRY",
    "create_env",
    "list_environments",
]

