"""Base agent protocol for all RL algorithms.

Defines the interface that all agents must implement.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol


class BaseAgent(Protocol):
    """Protocol defining the required interface for all RL agents.

    All agents must implement select_action and update methods.
    Uses structural subtyping - no explicit inheritance required.
    """

    def select_action(self, state: Dict[str, Any]) -> int:
        """Select an action given the current state.

        Args:
            state: Current state as a dictionary.

        Returns:
            int: Selected action index.
        """
        ...

    def update(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """Update the agent's knowledge based on a transition.

        Args:
            state: State before action.
            action: Action taken.
            reward: Reward received.
            next_state: State after action.
            done: Whether episode terminated.
        """
        ...

