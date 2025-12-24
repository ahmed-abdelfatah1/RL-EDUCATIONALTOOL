"""TD(0) prediction - one-step temporal difference for state-value estimation.

Estimates V(s) under a given policy using bootstrapping.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class TDPrediction:
    """TD(0) state-value prediction.

    Estimates V(s) using one-step temporal difference updates.
    This is policy evaluation only, not control.
    """

    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
    ) -> None:
        """Initialize TD prediction.

        Args:
            learning_rate: Step size alpha for updates.
            discount_factor: Gamma for bootstrapping.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.value_function: Dict[Tuple, float] = {}

    def select_action(self, state: Dict[str, Any]) -> int:
        """Select random action for episode runner compatibility.

        TD Prediction is policy evaluation only - it doesn't learn a policy.
        Returns random action to allow episode to proceed.

        Args:
            state: State as dictionary.

        Returns:
            int: Random action.
        """
        return int(np.random.randint(0, 4))  # Default 4 actions

    def update(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """Update value estimate using TD(0) rule.

        V(s) ← V(s) + α [ r + γ V(s') − V(s) ]

        Args:
            state: Current state.
            action: Action taken (unused for value prediction).
            reward: Reward received.
            next_state: Next state.
            done: Whether episode terminated.
        """
        key = self._state_to_key(state)
        next_key = self._state_to_key(next_state)

        self._ensure_state(key)
        self._ensure_state(next_key)

        current_value = self.value_function[key]
        next_value = 0.0 if done else self.value_function[next_key]

        td_target = reward + self.discount_factor * next_value
        td_error = td_target - current_value

        self.value_function[key] += self.learning_rate * td_error

    def get_value(self, state: Dict[str, Any]) -> float:
        """Get value estimate for a state.

        Args:
            state: State as dictionary.

        Returns:
            float: Estimated value V(s).
        """
        key = self._state_to_key(state)
        return self.value_function.get(key, 0.0)

    def get_td_error(
        self,
        state: Dict[str, Any],
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> float:
        """Compute TD error without updating.

        Args:
            state: Current state.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode terminated.

        Returns:
            float: TD error δ = r + γV(s') - V(s).
        """
        key = self._state_to_key(state)
        next_key = self._state_to_key(next_state)

        current_value = self.value_function.get(key, 0.0)
        next_value = 0.0 if done else self.value_function.get(next_key, 0.0)

        td_target = reward + self.discount_factor * next_value
        return td_target - current_value

    def reset(self) -> None:
        """Reset value function to empty."""
        self.value_function = {}

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
        """Ensure state exists in value function.

        Args:
            key: Hashable state key.
        """
        if key not in self.value_function:
            self.value_function[key] = 0.0

