"""Monte Carlo control agent - episodic, on-policy, first-visit.

Learns from complete episodes using sample returns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MonteCarloAgent:
    """First-visit Monte Carlo control agent implementing BaseAgent protocol.

    Suitable for episodic tasks with discrete state/action spaces.
    Updates Q-values after each complete episode using sample returns.
    """

    def __init__(
        self,
        num_actions: int,
        discount_factor: float,
        epsilon: float,
    ) -> None:
        """Initialize Monte Carlo agent.

        Args:
            num_actions: Number of possible actions.
            discount_factor: Gamma for return calculation.
            epsilon: Exploration probability for epsilon-greedy.
        """
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.q_values: Dict[Tuple, np.ndarray] = {}
        self.returns_sum: Dict[Tuple, np.ndarray] = {}
        self.returns_count: Dict[Tuple, np.ndarray] = {}
        self.episode: List[Tuple[Dict[str, Any], int, float]] = []

    def start_episode(self) -> None:
        """Reset episode history for a new episode."""
        self.episode = []
        logger.debug("Monte Carlo: Starting new episode")

    def select_action(self, state: Dict[str, Any]) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state as dictionary.

        Returns:
            int: Selected action index.
        """
        key = self._state_to_key(state)
        self._ensure_state(key)

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
            logger.debug(f"Monte Carlo: EXPLORE action={action} (ε={self.epsilon:.3f}) state={key}")
            return action

        # Greedy action selection with tie-breaking
        q_vals = self.q_values[key].copy()
        max_q = np.max(q_vals)
        
        # If all Q-values are equal (e.g., all 0), break ties randomly
        # This prevents getting stuck always choosing action 0
        if np.allclose(q_vals, max_q):
            best_actions = np.where(np.isclose(q_vals, max_q))[0]
            action = int(np.random.choice(best_actions))
            logger.debug(f"Monte Carlo: EXPLOIT (tie-break) action={action} state={key} Q-values={q_vals}")
        else:
            action = int(np.argmax(q_vals))
            logger.debug(f"Monte Carlo: EXPLOIT action={action} state={key} Q-values={q_vals}")
        
        return action

    def record_step(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
    ) -> None:
        """Record a step in the current episode.

        Args:
            state: State at this step.
            action: Action taken.
            reward: Reward received.
        """
        self.episode.append((state, action, reward))

    def finish_episode(self) -> None:
        """Process completed episode and update Q-values.

        Uses first-visit Monte Carlo: only the first occurrence
        of each state-action pair in the episode is used.
        """
        if len(self.episode) == 0:
            logger.warning("Monte Carlo: finish_episode called with empty episode")
            return

        visited_state_actions: set[Tuple[Tuple, int]] = set()
        returns_from_step = self._compute_returns()

        logger.debug(
            f"Monte Carlo: Finishing episode with {len(self.episode)} steps, "
            f"total_return={returns_from_step[0] if returns_from_step else 0:.3f}"
        )

        updates_count = 0
        for step_idx, (state, action, _) in enumerate(self.episode):
            key = self._state_to_key(state)
            state_action = (key, action)

            if state_action in visited_state_actions:
                logger.debug(f"Monte Carlo: Skipping duplicate (s,a)={state_action} at step {step_idx}")
                continue

            visited_state_actions.add(state_action)
            self._ensure_state(key)

            g_return = returns_from_step[step_idx]
            old_q = self.q_values[key][action]
            self.returns_sum[key][action] += g_return
            self.returns_count[key][action] += 1
            new_q = self.returns_sum[key][action] / self.returns_count[key][action]
            self.q_values[key][action] = new_q
            updates_count += 1

            logger.debug(
                f"Monte Carlo UPDATE: state={key} action={action} "
                f"Q(s,a)={old_q:.4f}→{new_q:.4f} "
                f"return={g_return:.4f} count={self.returns_count[key][action]}"
            )

        logger.debug(f"Monte Carlo: Updated {updates_count} unique state-action pairs")

    def update(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """Update agent with a transition (BaseAgent protocol).

        Records the step and processes episode when done.

        Args:
            state: State before action.
            action: Action taken.
            reward: Reward received.
            next_state: State after action (unused in MC).
            done: Whether episode terminated.
        """
        # Record the step BEFORE checking done, so the terminal step is included
        self.record_step(state, action, reward)
        state_key = self._state_to_key(state)
        logger.debug(
            f"Monte Carlo: Recorded step state={state_key} action={action} "
            f"reward={reward:.3f} done={done} episode_length={len(self.episode)}"
        )

        if done:
            # Process the completed episode
            if len(self.episode) > 0:
                self.finish_episode()
            # Reset for next episode
            self.start_episode()

    def get_q_values(self, state: Dict[str, Any]) -> np.ndarray:
        """Get Q-values for a given state.

        Args:
            state: State as dictionary.

        Returns:
            np.ndarray: Array of Q-values for each action.
        """
        key = self._state_to_key(state)
        self._ensure_state(key)
        return self.q_values[key].copy()

    def get_best_action(self, state: Dict[str, Any]) -> int:
        """Get best action (greedy) for a given state.

        Args:
            state: State as dictionary.

        Returns:
            int: Best action index.
        """
        key = self._state_to_key(state)
        self._ensure_state(key)
        return int(np.argmax(self.q_values[key]))

    def set_epsilon(self, epsilon: float) -> None:
        """Update exploration rate."""
        self.epsilon = epsilon

    def _compute_returns(self) -> List[float]:
        """Compute discounted returns for each step in the episode.

        Returns:
            List[float]: Return G_t for each step t.
        """
        returns: List[float] = []
        g_return = 0.0

        for _, _, reward in reversed(self.episode):
            g_return = reward + self.discount_factor * g_return
            returns.insert(0, g_return)

        return returns

    def _state_to_key(self, state: Dict[str, Any]) -> Tuple:
        """Convert state dictionary to hashable tuple key.

        For GridWorld and FrozenLake, uses the observation/state_index directly
        to ensure consistent state keys. For other environments, uses the full
        sorted state dict.

        Args:
            state: State as dictionary.

        Returns:
            Tuple: Hashable representation of state.
        """
        # For GridWorld and FrozenLake, use observation/state_index directly
        # This ensures consistent state keys matching the environment's state space
        # and makes extraction in training_service.py work correctly
        if "observation" in state:
            # GridWorld: observation is the state ID (0-24 for 5x5 grid)
            obs = state["observation"]
            if isinstance(obs, (int, float)):
                return (int(obs),)
        elif "state_index" in state:
            # FrozenLake: state_index is the state ID
            state_idx = state["state_index"]
            if isinstance(state_idx, (int, float)):
                return (int(state_idx),)
        
        # For other environments (CartPole, MountainCar, etc.), use full sorted state dict
        sorted_items = sorted(state.items())
        return tuple(self._make_hashable(v) for _, v in sorted_items)

    def _make_hashable(self, value: Any) -> Any:
        """Convert value to hashable type."""
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, dict):
            return tuple(sorted(value.items()))
        if isinstance(value, np.ndarray):
            return tuple(value.flatten().tolist())
        return value

    def _ensure_state(self, key: Tuple) -> None:
        """Ensure state exists in all tables with zero-initialized values.

        Args:
            key: Hashable state key.
        """
        if key not in self.q_values:
            self.q_values[key] = np.zeros(self.num_actions, dtype=np.float64)
            self.returns_sum[key] = np.zeros(self.num_actions, dtype=np.float64)
            self.returns_count[key] = np.zeros(self.num_actions, dtype=np.float64)

