"""Training service - connects environment, agent, and trainer.

Provides synchronous training execution for the API layer.
"""

from __future__ import annotations

from typing import List

from app.domain.algorithms import (
    MonteCarloAgent,
    NStepTDAgent,
    PolicyIteration,
    QLearningAgent,
    SarsaAgent,
    TDPrediction,
    ValueIteration,
)
from app.domain.environments import create_env
from app.domain.environments.base_env import BaseEnv
from app.domain.training.trainer import Trainer
from app.schemas.training_schemas import (
    EpisodeMetrics,
    TrainingRequest,
    TrainingRunResponse,
)
from app.services.stream_publisher import stream_publisher


# Step delay for live visualization (seconds)
LIVE_VIZ_STEP_DELAY = 0.03


def run_training(request: TrainingRequest) -> TrainingRunResponse:
    """Run a synchronous training session.

    Args:
        request: Training configuration request.

    Returns:
        TrainingRunResponse: Results of training run.
    """
    env = create_env(request.env_name)  # type: ignore[arg-type]

    agent = _create_agent(request, env)

    # Check if there are any subscribers for live visualization
    has_subscribers = stream_publisher.has_subscribers(request.env_name)

    trainer = Trainer(
        env=env,
        agent=agent,
        max_steps_per_episode=request.max_steps_per_episode,
        env_name=request.env_name,
        publish_state=has_subscribers,
        step_delay=LIVE_VIZ_STEP_DELAY if has_subscribers else 0.0,
    )

    history = trainer.run_episodes(request.num_episodes)

    episodes = _convert_history(history)

    return TrainingRunResponse(
        env_name=request.env_name,
        algorithm_name=request.algorithm_name,
        episodes=episodes,
    )


def _create_agent(request: TrainingRequest, env: BaseEnv):
    """Create agent based on algorithm name and environment.

    Args:
        request: Training request with algorithm config.
        env: Environment to get action space size from.

    Returns:
        Agent instance.

    Raises:
        ValueError: If algorithm name is not recognized.
    """
    learning_rate = request.learning_rate or 0.1
    epsilon = request.epsilon or 0.1
    n_step = request.n_step or 3
    num_actions = env.action_space_size

    if request.algorithm_name == "q_learning":
        return QLearningAgent(
            num_actions=num_actions,
            learning_rate=learning_rate,
            discount_factor=request.discount_factor,
            epsilon=epsilon,
        )

    if request.algorithm_name == "sarsa":
        return SarsaAgent(
            num_actions=num_actions,
            learning_rate=learning_rate,
            discount_factor=request.discount_factor,
            epsilon=epsilon,
        )

    if request.algorithm_name == "monte_carlo":
        return MonteCarloAgent(
            num_actions=num_actions,
            discount_factor=request.discount_factor,
            epsilon=epsilon,
        )

    if request.algorithm_name == "n_step_td":
        return NStepTDAgent(
            num_actions=num_actions,
            n_step=n_step,
            learning_rate=learning_rate,
            discount_factor=request.discount_factor,
            epsilon=epsilon,
        )

    if request.algorithm_name == "td_prediction":
        return TDPrediction(
            learning_rate=learning_rate,
            discount_factor=request.discount_factor,
        )

    if request.algorithm_name == "policy_iteration":
        return PolicyIteration(
            discount_factor=request.discount_factor,
            num_actions=num_actions,
        )

    if request.algorithm_name == "value_iteration":
        return ValueIteration(
            discount_factor=request.discount_factor,
            num_actions=num_actions,
        )

    raise ValueError(f"Unknown algorithm: {request.algorithm_name}")


def _convert_history(history: List[dict]) -> List[EpisodeMetrics]:
    """Convert trainer history to EpisodeMetrics models.

    Args:
        history: List of episode metric dicts from trainer.

    Returns:
        List[EpisodeMetrics]: Pydantic models for API response.
    """
    return [
        EpisodeMetrics(
            episode=entry["episode"],
            total_reward=entry["total_reward"],
            length=entry["length"],
        )
        for entry in history
    ]
