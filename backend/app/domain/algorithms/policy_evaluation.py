"""Policy Iteration algorithm for discrete MDPs.

Implements iterative policy evaluation and policy improvement
for tabular reinforcement learning problems.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np


class PolicyIteration:
    """Tabular Policy Iteration for finite MDPs.

    Alternates between policy evaluation and policy improvement
    until the policy converges to optimal.
    """

    def __init__(
        self,
        discount_factor: float,
        num_actions: int,
        theta: float = 1e-6,
    ) -> None:
        """Initialize Policy Iteration.

        Args:
            discount_factor: Gamma for future reward weighting.
            num_actions: Number of possible actions.
            theta: Convergence threshold for policy evaluation.
        """
        self.discount_factor = discount_factor
        self.num_actions = num_actions
        self.theta = theta

        self.value_function: Dict[Tuple, float] = {}
        self.policy: Dict[Tuple, int] = {}

    def evaluate_policy(
        self,
        states: Tuple[Tuple, ...],
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float]]],
    ) -> None:
        """Iterative policy evaluation.

        Computes V^π for the current policy until convergence.

        Args:
            states: Tuple of all state keys.
            transition_model: Function (s, a) -> [(next_state, prob, reward), ...].
        """
        self._ensure_states_initialized(states)

        while True:
            max_delta = 0.0

            for state in states:
                old_value = self.value_function[state]
                action = self.policy.get(state, 0)
                new_value = self._compute_state_value(state, action, transition_model)
                self.value_function[state] = new_value
                max_delta = max(max_delta, abs(old_value - new_value))

            if max_delta < self.theta:
                break

    def improve_policy(
        self,
        states: Tuple[Tuple, ...],
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float]]],
    ) -> bool:
        """Policy improvement step.

        Updates policy to be greedy with respect to current value function.

        Args:
            states: Tuple of all state keys.
            transition_model: Function (s, a) -> [(next_state, prob, reward), ...].

        Returns:
            bool: True if policy is stable (no changes), False otherwise.
        """
        policy_stable = True

        for state in states:
            old_action = self.policy.get(state, 0)
            best_action = self._get_best_action(state, transition_model)
            self.policy[state] = best_action

            if old_action != best_action:
                policy_stable = False

        return policy_stable

    def run_iteration(
        self,
        states: Tuple[Tuple, ...],
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float]]],
    ) -> None:
        """Run full policy iteration until convergence.

        Args:
            states: Tuple of all state keys.
            transition_model: Function (s, a) -> [(next_state, prob, reward), ...].
        """
        self._ensure_states_initialized(states)
        self._initialize_policy(states)

        while True:
            self.evaluate_policy(states, transition_model)
            stable = self.improve_policy(states, transition_model)

            if stable:
                break

    def get_action(self, state: Tuple) -> int:
        """Get action from current policy.

        Args:
            state: State key.

        Returns:
            int: Action from policy.
        """
        return self.policy.get(state, 0)

    def select_action(self, state: Dict[str, Any]) -> int:
        """Select action for episode runner compatibility.

        Converts state dict to tuple key and looks up policy.

        Args:
            state: State as dictionary.

        Returns:
            int: Action from policy (or random if not learned yet).
        """
        key = self._state_to_key(state)
        if key in self.policy:
            return self.policy[key]
        # Random action if policy not yet computed
        return int(np.random.randint(0, self.num_actions))

    def update(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """No-op update for episode runner compatibility.

        Policy Iteration is a planning algorithm that requires a model.
        Online updates are not applicable.
        """
        pass

    def _state_to_key(self, state: Dict[str, Any]) -> Tuple:
        """Convert state dictionary to hashable tuple key."""
        sorted_items = sorted(state.items())
        return tuple(self._make_hashable(v) for _, v in sorted_items)

    def _make_hashable(self, value) -> Any:
        """Convert value to hashable type."""
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, dict):
            return tuple(sorted(value.items()))
        if isinstance(value, np.ndarray):
            return tuple(value.flatten().tolist())
        return value

    def get_value(self, state: Tuple) -> float:
        """Get value of a state.

        Args:
            state: State key.

        Returns:
            float: Value of state.
        """
        return self.value_function.get(state, 0.0)

    def _compute_state_value(
        self,
        state: Tuple,
        action: int,
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float]]],
    ) -> float:
        """Compute expected value for state-action pair.

        V(s) = sum_{s'} p(s'|s,a)[r + γ V(s')]
        """
        transitions = transition_model(state, action)
        value = 0.0

        for next_state, prob, reward in transitions:
            next_value = self.value_function.get(next_state, 0.0)
            value += prob * (reward + self.discount_factor * next_value)

        return value

    def _compute_action_value(
        self,
        state: Tuple,
        action: int,
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float]]],
    ) -> float:
        """Compute Q(s, a) for given state-action pair."""
        return self._compute_state_value(state, action, transition_model)

    def _get_best_action(
        self,
        state: Tuple,
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float]]],
    ) -> int:
        """Find action with highest Q-value for given state."""
        action_values = np.zeros(self.num_actions)

        for action in range(self.num_actions):
            action_values[action] = self._compute_action_value(
                state, action, transition_model
            )

        return int(np.argmax(action_values))

    def _ensure_states_initialized(self, states: Tuple[Tuple, ...]) -> None:
        """Initialize value function for all states."""
        for state in states:
            if state not in self.value_function:
                self.value_function[state] = 0.0

    def _initialize_policy(self, states: Tuple[Tuple, ...]) -> None:
        """Initialize policy with action 0 for all states."""
        for state in states:
            if state not in self.policy:
                self.policy[state] = 0

