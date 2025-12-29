"""Q-Learning agent - tabular implementation for discrete state/action spaces.

Classic off-policy TD control algorithm using epsilon-greedy exploration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QLearningAgent:
    """Tabular Q-learning agent implementing BaseAgent protocol.

    Suitable for discrete environments like GridWorld and FrozenLake.
    Uses epsilon-greedy exploration and maintains a Q-table as a dictionary.
    """

    def __init__(
        self,
        num_actions: int,
        learning_rate: float,
        discount_factor: float,
        epsilon: float,
    ) -> None:
        """Initialize Q-learning agent.

        Args:
            num_actions: Number of possible actions.
            learning_rate: Step size alpha for updates.
            discount_factor: Gamma for future reward weighting.
            epsilon: Exploration probability for epsilon-greedy.
        """
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.q_values: Dict[Tuple, np.ndarray] = {}

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
            logger.debug(f"Q-Learning: EXPLORE action={action} (ε={self.epsilon:.3f}) state={key}")
            return action

        # Greedy action selection with tie-breaking
        q_vals = self.q_values[key].copy()
        max_q = np.max(q_vals)
        
        # If all Q-values are equal (e.g., all 0), break ties randomly
        # This prevents getting stuck always choosing action 0
        if np.allclose(q_vals, max_q):
            best_actions = np.where(np.isclose(q_vals, max_q))[0]
            action = int(np.random.choice(best_actions))
            logger.debug(f"Q-Learning: EXPLOIT (tie-break) action={action} state={key} Q-values={q_vals}")
        else:
            action = int(np.argmax(q_vals))
            logger.debug(f"Q-Learning: EXPLOIT action={action} state={key} Q-values={q_vals}")
        
        return action

    def update(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """Update Q-value using Q-learning update rule.

        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]

        Args:
            state: State before action.
            action: Action taken.
            reward: Reward received.
            next_state: State after action.
            done: Whether episode terminated.
        """
        key = self._state_to_key(state)
        next_key = self._state_to_key(next_state)

        self._ensure_state(key)
        self._ensure_state(next_key)

        current_q = self.q_values[key][action]
        next_max_q = 0.0 if done else float(np.max(self.q_values[next_key]))

        target = reward + self.discount_factor * next_max_q
        td_error = target - current_q
        new_q = current_q + self.learning_rate * td_error

        logger.debug(
            f"Q-Learning UPDATE: state={key} action={action} "
            f"Q(s,a)={current_q:.4f}→{new_q:.4f} "
            f"r={reward:.3f} Q(s',a')={next_max_q:.4f} "
            f"target={target:.4f} error={td_error:.4f} α={self.learning_rate}"
        )

        self.q_values[key][action] = new_q

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
        """Ensure state exists in Q-table with zero-initialized values.

        Args:
            key: Hashable state key.
        """
        if key not in self.q_values:
            self.q_values[key] = np.zeros(self.num_actions, dtype=np.float64)


