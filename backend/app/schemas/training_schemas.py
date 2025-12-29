"""Training schemas - Pydantic models for training requests and responses.

Defines models for starting training and returning training snapshots.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class TrainingRequest(BaseModel):
    """Request to start a training session."""

    env_name: str
    algorithm_name: str
    num_episodes: int
    max_steps_per_episode: int
    discount_factor: float
    learning_rate: float | None = None
    epsilon: float | None = None
    n_step: int | None = None


class EpisodeMetrics(BaseModel):
    """Metrics for a single training episode."""

    episode: int
    total_reward: float
    length: int


class TrainingRunResponse(BaseModel):
    """Response containing training results."""

    env_name: str
    algorithm_name: str
    episodes: List[EpisodeMetrics]
    value_function: Dict[str, float] | None = None
    policy: Dict[str, int] | None = None


class TrainingSnapshot(BaseModel):
    """Current training state snapshot for frontend."""

    episode: int
    history: List[EpisodeMetrics]
    env_state: Dict[str, Any] | None = None

