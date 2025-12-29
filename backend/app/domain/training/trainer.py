"""Trainer - manages multiple episodes and aggregates metrics.

Provides training orchestration for the API/WebSocket layer.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from app.domain.algorithms.base_agent import BaseAgent
from app.domain.environments.base_env import BaseEnv

from .episode_runner import run_episode


EpisodeMetrics = Dict[str, Any]
TrainingSnapshot = Dict[str, Any]


class Trainer:
    """Manages training sessions across multiple episodes.

    Aggregates metrics and prepares data for frontend visualization.
    """

    def __init__(
        self,
        env: BaseEnv,
        agent: BaseAgent,
        max_steps_per_episode: int,
        env_name: Optional[str] = None,
        publish_state: bool = False,
        step_delay: float = 0.0,
    ) -> None:
        """Initialize trainer.

        Args:
            env: Environment to train on.
            agent: Agent to train.
            max_steps_per_episode: Maximum steps per episode.
            env_name: Environment name for state publishing.
            publish_state: Whether to publish state for live visualization.
            step_delay: Delay between steps in seconds (for visualization).
        """
        self.env = env
        self.agent = agent
        self.max_steps_per_episode = max_steps_per_episode
        self.env_name = env_name
        self.publish_state = publish_state
        self.step_delay = step_delay

        self.episode: int = 0
        self.history: List[EpisodeMetrics] = []

    def run_episodes(self, num_episodes: int) -> List[EpisodeMetrics]:
        """Run multiple training episodes.

        Args:
            num_episodes: Number of episodes to run.

        Returns:
            List[EpisodeMetrics]: Updated history with new episode metrics.
        """
        for _ in range(num_episodes):
            metrics = self._run_single_episode()
            self.history.append(metrics)

        return self.history

    def run_single_episode(self) -> EpisodeMetrics:
        """Run a single training episode and return its metrics.

        Returns:
            EpisodeMetrics: Metrics for the completed episode.
        """
        metrics = self._run_single_episode()
        self.history.append(metrics)
        return metrics

    def latest_snapshot(self) -> TrainingSnapshot:
        """Get current training state for frontend.

        Returns:
            TrainingSnapshot: Dictionary containing:
                - "episode": Current episode count
                - "history": All episode metrics
        """
        return {
            "episode": self.episode,
            "history": self.history,
        }

    def get_recent_metrics(self, num_episodes: int = 10) -> List[EpisodeMetrics]:
        """Get metrics for recent episodes.

        Args:
            num_episodes: Number of recent episodes to return.

        Returns:
            List[EpisodeMetrics]: Recent episode metrics.
        """
        return self.history[-num_episodes:]

    def get_average_reward(self, num_episodes: int = 10) -> float:
        """Get average reward over recent episodes.

        Args:
            num_episodes: Number of episodes to average over.

        Returns:
            float: Average total reward.
        """
        recent = self.get_recent_metrics(num_episodes)
        if not recent:
            return 0.0
        return sum(m["total_reward"] for m in recent) / len(recent)

    def reset(self) -> None:
        """Reset training state."""
        self.episode = 0
        self.history = []

    def _run_single_episode(self) -> EpisodeMetrics:
        """Internal method to run one episode and compute metrics."""
        total_reward = 0.0
        length = 0
        episode_ended_naturally = False

        for step in run_episode(
            self.env,
            self.agent,
            self.max_steps_per_episode,
            env_name=self.env_name,
            publish_state=self.publish_state,
        ):
            total_reward += step["reward"]
            length += 1
            episode_ended_naturally = step.get("done", False)

            # Add delay between steps for visualization
            if self.step_delay > 0:
                time.sleep(self.step_delay)

        # For Monte Carlo and n-step TD, ensure episode is finished even if it hit max_steps
        # (episode runner breaks on done=True, but these algorithms need to process even if max_steps reached)
        if not episode_ended_naturally:
            # Monte Carlo: finish episode if it has steps recorded
            if hasattr(self.agent, 'finish_episode') and hasattr(self.agent, 'episode'):
                if len(self.agent.episode) > 0:
                    self.agent.finish_episode()
                    if hasattr(self.agent, 'start_episode'):
                        self.agent.start_episode()
            
            # N-step TD: finish remaining updates if episode hit max_steps
            if hasattr(self.agent, '_finish_remaining_updates') and hasattr(self.agent, 'states'):
                if len(self.agent.states) > 0 and not hasattr(self.agent, 'finish_episode'):
                    # Set T to episode length so _finish_remaining_updates works correctly
                    if self.agent.T is None:
                        self.agent.T = len(self.agent.rewards)
                    # Get the last state from the episode
                    last_state = self.agent.states[-1] if self.agent.states else {}
                    self.agent._finish_remaining_updates(last_state)
                    if hasattr(self.agent, 'start_episode'):
                        self.agent.start_episode()

        self.episode += 1

        return {
            "episode": self.episode,
            "total_reward": total_reward,
            "length": length,
        }
