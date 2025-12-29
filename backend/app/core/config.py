"""Application configuration.

Centralizes basic configuration for the RL Educational Tool backend.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    api_v1_prefix: str = "/api/v1"
    backend_cors_origins: list[str] = ["http://localhost:3000"]
    default_env_name: str = "gridworld"
    default_algorithm_name: str = "q_learning"
    max_steps_per_episode: int = 200



settings = Settings()

