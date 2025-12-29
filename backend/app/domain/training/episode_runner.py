"""Episode runner - reusable helper for training step/episode loops.

Connects an environment and agent to produce step data for logging.
Optionally publishes render state for live visualization.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

from app.domain.algorithms.base_agent import BaseAgent
from app.domain.environments.base_env import BaseEnv
from app.services.stream_publisher import publish_env_state


StateDict = Dict[str, Any]
StepRecord = Dict[str, Any]


def run_episode(
    env: BaseEnv,
    agent: BaseAgent,
    max_steps: int,
    env_name: Optional[str] = None,
    publish_state: bool = False,
) -> Iterator[StepRecord]:
    """Run a single episode and yield step records.

    This is the ONLY place that directly loops over the
    (state, action, reward, next_state, done) cycle.

    Args:
        env: Environment implementing BaseEnv protocol.
        agent: Agent implementing BaseAgent protocol.
        max_steps: Maximum steps before forced termination.
        env_name: Optional environment name for state publishing.
        publish_state: Whether to publish render state for live viz.

    Yields:
        StepRecord: Dictionary containing step information:
            - "t": Step index within episode
            - "state": State before action
            - "action": Action taken
            - "reward": Reward received
            - "next_state": State after action
            - "done": Whether episode terminated
    """
    state = env.reset()

    # Publish initial state
    if publish_state and env_name:
        render_state = env.render_state()
        publish_env_state(env_name, render_state)

    for t in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)

        # Publish state after each step for live visualization
        if publish_state and env_name:
            render_state = env.render_state()
            publish_env_state(env_name, render_state)

        yield {
            "t": t,
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }

        state = next_state

        if done:
            break


def run_episode_no_update(
    env: BaseEnv,
    agent: BaseAgent,
    max_steps: int,
    env_name: Optional[str] = None,
    publish_state: bool = False,
) -> Iterator[StepRecord]:
    """Run episode without updating agent (for evaluation).

    Args:
        env: Environment implementing BaseEnv protocol.
        agent: Agent implementing BaseAgent protocol.
        max_steps: Maximum steps before forced termination.
        env_name: Optional environment name for state publishing.
        publish_state: Whether to publish render state for live viz.

    Yields:
        StepRecord: Dictionary containing step information.
    """
    state = env.reset()

    # Publish initial state
    if publish_state and env_name:
        render_state = env.render_state()
        publish_env_state(env_name, render_state)

    for t in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        # Publish state after each step for live visualization
        if publish_state and env_name:
            render_state = env.render_state()
            publish_env_state(env_name, render_state)

        yield {
            "t": t,
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }

        state = next_state

        if done:
            break
