"""Monte Carlo control agent - episodic, on-policy, first-visit.

Learns from complete episodes using sample returns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


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
            return np.random.randint(self.num_actions)

        return int(np.argmax(self.q_values[key]))

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
        visited_state_actions: set[Tuple[Tuple, int]] = set()
        returns_from_step = self._compute_returns()

        for step_idx, (state, action, _) in enumerate(self.episode):
            key = self._state_to_key(state)
            state_action = (key, action)

            if state_action in visited_state_actions:
                continue

            visited_state_actions.add(state_action)
            self._ensure_state(key)

            g_return = returns_from_step[step_idx]
            self.returns_sum[key][action] += g_return
            self.returns_count[key][action] += 1
            self.q_values[key][action] = (
                self.returns_sum[key][action] / self.returns_count[key][action]
            )

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
        self.record_step(state, action, reward)

        if done:
            self.finish_episode()
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

        Args:
            state: State as dictionary.

        Returns:
            Tuple: Hashable representation of state.
        """
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

