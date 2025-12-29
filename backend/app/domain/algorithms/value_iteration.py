"""Value Iteration algorithm for discrete MDPs.

Implements standard value iteration for tabular reinforcement learning.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ValueIteration:
    """Tabular Value Iteration for finite MDPs.

    Iteratively updates value function until convergence,
    then extracts optimal policy.
    """

    def __init__(
        self,
        discount_factor: float,
        num_actions: int,
        theta: float = 1e-6,
    ) -> None:
        """Initialize Value Iteration.

        Args:
            discount_factor: Gamma for future reward weighting.
            num_actions: Number of possible actions.
            theta: Convergence threshold.
        """
        self.discount_factor = discount_factor
        self.num_actions = num_actions
        self.theta = theta

        self.value_function: Dict[Tuple, float] = {}
        self.policy: Dict[Tuple, int] = {}
        self.terminal_states: set[Tuple] = set()
        
        logger.debug(f"Value Iteration initialized: γ={discount_factor} num_actions={num_actions} θ={theta}")

    def run(
        self,
        states: Tuple[Tuple, ...],
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float, bool]]],
    ) -> None:
        """Run value iteration until convergence.

        Args:
            states: Tuple of all state keys.
            transition_model: Function (s, a) -> [(next_state, prob, reward, done), ...].
        """
        self._ensure_states_initialized(states)
        self._identify_terminal_states(states, transition_model)
        self._iterate_values(states, transition_model)
        self._extract_policy(states, transition_model)

    def _identify_terminal_states(
        self,
        states: Tuple[Tuple, ...],
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float, bool]]],
    ) -> None:
        """Identify and fix values for terminal states.
        
        Terminal states are states where ALL transitions from ALL actions have done=True.
        These are states where the episode has definitively ended (e.g., absorbing states).
        
        Note: In some environments (like GridWorld), the goal state can transition to
        non-goal states, so it's not a pure terminal state. We only mark states as terminal
        if they truly cannot leave (all actions lead to terminal transitions).
        """
        self.terminal_states: set[Tuple] = set()
        
        for state in states:
            # Check if all transitions from this state are terminal
            is_terminal = True
            max_terminal_reward = float('-inf')
            
            for action in range(self.num_actions):
                transitions = transition_model(state, action)
                if len(transitions) == 0:
                    is_terminal = False
                    break
                
                # Check if all transitions for this action are terminal
                all_terminal = True
                for trans in transitions:
                    if len(trans) >= 4:
                        next_state, prob, reward, done = trans[:4]
                        if done:
                            # This is a terminal transition - track max reward
                            max_terminal_reward = max(max_terminal_reward, reward)
                        else:
                            all_terminal = False
                            break
                    else:
                        # Old format without done flag - assume non-terminal
                        all_terminal = False
                        break
                
                if not all_terminal:
                    is_terminal = False
                    break
            
            if is_terminal:
                self.terminal_states.add(state)
                # For true terminal states, value is the reward for reaching them
                # Since episode ends, V(terminal) = max reward when reaching terminal
                terminal_value = max_terminal_reward if max_terminal_reward != float('-inf') else 0.0
                self.value_function[state] = terminal_value
                logger.debug(f"Value Iteration: Terminal state {state} fixed at V={terminal_value:.4f}")

    def _iterate_values(
        self,
        states: Tuple[Tuple, ...],
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float, bool]]],
    ) -> None:
        """Main value iteration loop.

        V(s) = max_a sum_{s'} p(s'|s,a)[r + γ V(s')]
        
        Terminal states are skipped (their values are fixed).
        """
        iteration = 0
        logger.debug(f"Value Iteration: Starting value iteration with {len(states)} states ({len(self.terminal_states)} terminal)")
        
        while True:
            delta = 0.0

            for state in states:
                # Skip terminal states - their values are fixed
                if state in self.terminal_states:
                    continue
                    
                old_value = self.value_function[state]
                new_value = self._compute_max_value(state, transition_model)
                self.value_function[state] = new_value
                delta = max(delta, abs(old_value - new_value))

            iteration += 1
            if iteration % 10 == 0 or delta < self.theta:
                logger.debug(f"Value Iteration: Iteration {iteration}, max_delta={delta:.6f}")

            if delta < self.theta:
                logger.debug(f"Value Iteration: Converged after {iteration} iterations")
                break

    def _extract_policy(
        self,
        states: Tuple[Tuple, ...],
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float, bool]]],
    ) -> None:
        """Extract greedy policy from converged value function."""
        logger.debug(f"Value Iteration: Extracting policy from {len(states)} states")
        for state in states:
            best_action = self._get_best_action(state, transition_model)
            self.policy[state] = best_action
        logger.debug("Value Iteration: Policy extraction complete")

    def _compute_max_value(
        self,
        state: Tuple,
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float, bool]]],
    ) -> float:
        """Compute maximum value over all actions.

        max_a sum_{s'} p(s'|s,a)[r + γ V(s')]
        """
        action_values = np.zeros(self.num_actions)

        for action in range(self.num_actions):
            action_values[action] = self._compute_action_value(
                state, action, transition_model
            )

        return float(np.max(action_values))

    def _compute_action_value(
        self,
        state: Tuple,
        action: int,
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float, bool]]],
    ) -> float:
        """Compute Q(s, a) for given state-action pair.

        Q(s, a) = sum_{s'} p(s'|s,a)[r + γ V(s')]
        
        For terminal transitions (done=True), the episode ends immediately after receiving the reward.
        So we use: Q(s, a) = r (no future value since episode ends).
        """
        transitions = transition_model(state, action)
        value = 0.0

        for trans in transitions:
            if len(trans) >= 4:
                next_state, prob, reward, done = trans[:4]
                if done:
                    # Terminal transition: episode ends, so Q(s,a) = reward (no future value)
                    # The reward is what you get for reaching the terminal state
                    value += prob * reward
                else:
                    # Non-terminal: standard Bellman equation
                    next_value = self.value_function.get(next_state, 0.0)
                    value += prob * (reward + self.discount_factor * next_value)
            else:
                # Backward compatibility: old format (next_state, prob, reward)
                next_state, prob, reward = trans[:3]
                next_value = self.value_function.get(next_state, 0.0)
                value += prob * (reward + self.discount_factor * next_value)

        return value

    def _get_best_action(
        self,
        state: Tuple,
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float, bool]]],
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
        # Extract state_id from state dict (handles both GridWorld and FrozenLake)
        state_id = state.get("observation", state.get("state_index", 0))
        key = (state_id,)
        if key in self.policy:
            action = self.policy[key]
            value = self.value_function.get(key, 0.0)
            logger.debug(f"Value Iteration: Using learned policy state={key} action={action} V(s)={value:.4f}")
            return action
        # Random action if policy not yet computed
        action = int(np.random.randint(0, self.num_actions))
        logger.debug(f"Value Iteration: Policy not learned yet, random action={action} state={key}")
        return action


    def update(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """No-op update for episode runner compatibility.

        Value Iteration is a planning algorithm that requires a model.
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

    def get_all_action_values(
        self,
        state: Tuple,
        transition_model: Callable[[Tuple, int], list[tuple[Tuple, float, float]]],
    ) -> np.ndarray:
        """Get Q-values for all actions in a state.

        Args:
            state: State key.
            transition_model: Transition function.

        Returns:
            np.ndarray: Array of Q-values for each action.
        """
        action_values = np.zeros(self.num_actions)

        for action in range(self.num_actions):
            action_values[action] = self._compute_action_value(
                state, action, transition_model
            )

        return action_values

