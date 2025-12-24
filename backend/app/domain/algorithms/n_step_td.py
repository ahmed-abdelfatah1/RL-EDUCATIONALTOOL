"""N-step TD control agent - on-policy with n-step returns.

Uses a sliding window of transitions for multi-step bootstrapping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


class NStepTDAgent:
    """N-step TD control agent implementing BaseAgent protocol.

    Uses n-step returns for more efficient credit assignment
    while maintaining online updates.
    """

    def __init__(
        self,
        num_actions: int,
        n_step: int,
        learning_rate: float,
        discount_factor: float,
        epsilon: float,
    ) -> None:
        """Initialize n-step TD agent.

        Args:
            num_actions: Number of possible actions.
            n_step: Number of steps for return calculation.
            learning_rate: Step size alpha for updates.
            discount_factor: Gamma for discounting.
            epsilon: Exploration probability for epsilon-greedy.
        """
        self.num_actions = num_actions
        self.n_step = n_step
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        print(f"NStepTDAgent initialized with n_step={n_step}")

        self.q_values: Dict[Tuple, np.ndarray] = {}

        self.states: List[Dict[str, Any]] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.t: int = 0
        self.T: int | None = None

    def start_episode(self) -> None:
        """Reset episode buffers for a new episode."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.t = 0
        self.T = None

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

    def update(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """Update Q-values using n-step TD.

        Args:
            state: State before action.
            action: Action taken.
            reward: Reward received.
            next_state: State after action.
            done: Whether episode terminated.
        """
        self._store_transition(state, action, reward)

        if done:
            self.T = len(self.rewards)

        tau = self.t - self.n_step + 1

        if tau >= 0:
            self._update_q_value(tau, next_state)

        self.t += 1

        if self._is_episode_complete():
            self._finish_remaining_updates(next_state)
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

    def _store_transition(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
    ) -> None:
        """Store transition in episode buffers."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def _update_q_value(self, tau: int, next_state: Dict[str, Any]) -> None:
        """Update Q-value for state at time tau."""
        g_return = self._compute_n_step_return(tau, next_state)

        state_tau = self.states[tau]
        action_tau = self.actions[tau]
        key = self._state_to_key(state_tau)
        self._ensure_state(key)

        current_q = self.q_values[key][action_tau]
        td_error = g_return - current_q
        self.q_values[key][action_tau] += self.learning_rate * td_error

    def _compute_n_step_return(
        self,
        tau: int,
        next_state: Dict[str, Any],
    ) -> float:
        """Compute n-step return G from time tau.

        G = r_{tau+1} + γ r_{tau+2} + ... + γ^{n-1} r_{tau+n} + γ^n Q(s_{tau+n}, a_{tau+n})
        """
        end_idx = min(tau + self.n_step, len(self.rewards))
        g_return = 0.0
        gamma_power = 1.0

        for idx in range(tau, end_idx):
            g_return += gamma_power * self.rewards[idx]
            gamma_power *= self.discount_factor

        if self.T is None or tau + self.n_step < self.T:
            bootstrap_state = self._get_bootstrap_state(tau, next_state)
            bootstrap_key = self._state_to_key(bootstrap_state)
            self._ensure_state(bootstrap_key)
            bootstrap_action = int(np.argmax(self.q_values[bootstrap_key]))
            g_return += gamma_power * self.q_values[bootstrap_key][bootstrap_action]

        return g_return

    def _get_bootstrap_state(
        self,
        tau: int,
        next_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get state to bootstrap from."""
        bootstrap_idx = tau + self.n_step
        if bootstrap_idx < len(self.states):
            return self.states[bootstrap_idx]
        return next_state

    def _is_episode_complete(self) -> bool:
        """Check if all updates for this episode are done."""
        if self.T is None:
            return False
        return self.t >= self.T + self.n_step - 1

    def _finish_remaining_updates(self, next_state: Dict[str, Any]) -> None:
        """Complete remaining updates at episode end."""
        if self.T is None:
            return

        start_tau = max(0, self.t - self.n_step + 1)
        for tau in range(start_tau, self.T):
            if tau >= 0:
                self._update_q_value(tau, next_state)

    def _state_to_key(self, state: Dict[str, Any]) -> Tuple:
        """Convert state dictionary to hashable tuple key."""
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
        """Ensure state exists in Q-table with zero-initialized values."""
        if key not in self.q_values:
            self.q_values[key] = np.zeros(self.num_actions, dtype=np.float64)

