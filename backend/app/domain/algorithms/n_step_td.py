"""N-step TD control agent - on-policy with n-step returns.

Uses a sliding window of transitions for multi-step bootstrapping.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
        logger.debug(f"N-step TD initialized: n_step={n_step} α={learning_rate} γ={discount_factor} ε={epsilon}")

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
        logger.debug("N-step TD: Starting new episode")

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
            logger.debug(f"N-step TD: EXPLORE action={action} (ε={self.epsilon:.3f}) state={key}")
            return action

        # Greedy action selection with tie-breaking
        q_vals = self.q_values[key].copy()
        max_q = np.max(q_vals)
        
        # If all Q-values are equal (e.g., all 0), break ties randomly
        # This prevents getting stuck always choosing action 0
        if np.allclose(q_vals, max_q):
            best_actions = np.where(np.isclose(q_vals, max_q))[0]
            action = int(np.random.choice(best_actions))
            logger.debug(f"N-step TD: EXPLOIT (tie-break) action={action} state={key} Q-values={q_vals}")
        else:
            action = int(np.argmax(q_vals))
            logger.debug(f"N-step TD: EXPLOIT action={action} state={key} Q-values={q_vals}")
        
        return action

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
            logger.debug(f"N-step TD: Episode done at t={self.t}, T={self.T}")

        tau = self.t - self.n_step + 1

        if tau >= 0:
            self._update_q_value(tau, next_state)

        self.t += 1

        if self._is_episode_complete():
            logger.debug(f"N-step TD: Finishing remaining updates (t={self.t}, T={self.T})")
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
        new_q = current_q + self.learning_rate * td_error
        self.q_values[key][action_tau] = new_q

        logger.debug(
            f"N-step TD UPDATE: τ={tau} state={key} action={action_tau} "
            f"Q(s,a)={current_q:.4f}→{new_q:.4f} "
            f"n-step_return={g_return:.4f} error={td_error:.4f} α={self.learning_rate}"
        )

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
            
            # Greedy action selection with tie-breaking for bootstrap
            bootstrap_q_vals = self.q_values[bootstrap_key].copy()
            bootstrap_max_q = np.max(bootstrap_q_vals)
            if np.allclose(bootstrap_q_vals, bootstrap_max_q):
                best_actions = np.where(np.isclose(bootstrap_q_vals, bootstrap_max_q))[0]
                bootstrap_action = int(np.random.choice(best_actions))
            else:
                bootstrap_action = int(np.argmax(bootstrap_q_vals))
            
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
        """Check if all updates for this episode are done.
        
        Returns True when:
        - Episode terminated (T is set) AND all n-step updates are complete
        - OR episode hit max_steps (T is None but we've processed all transitions)
        """
        if self.T is not None:
            # Episode terminated naturally - check if all updates are done
            return self.t >= self.T + self.n_step - 1
        # If T is None and we have transitions, episode might have hit max_steps
        # This will be handled by trainer.py calling finish_episode if needed
        return False

    def _finish_remaining_updates(self, next_state: Dict[str, Any]) -> None:
        """Complete remaining updates at episode end.
        
        Handles both natural termination (T is set) and max_steps termination (T is None).
        """
        # Determine episode end: T if set, otherwise use current t (max_steps reached)
        episode_end = self.T if self.T is not None else len(self.rewards)
        
        if episode_end == 0:
            return

        start_tau = max(0, self.t - self.n_step + 1)
        for tau in range(start_tau, episode_end):
            if tau >= 0:
                self._update_q_value(tau, next_state)
        
        logger.debug(f"N-step TD: Finished remaining updates from τ={start_tau} to {episode_end}")

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
        """Ensure state exists in Q-table with zero-initialized values."""
        if key not in self.q_values:
            self.q_values[key] = np.zeros(self.num_actions, dtype=np.float64)

