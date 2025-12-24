"""Base environment protocol for all RL environments.

Defines the interface that all environments must implement.
Uses typing.Protocol for structural subtyping (duck typing with type safety).
"""

from typing import Protocol


class BaseEnv(Protocol):
    """Protocol defining the required interface for all RL environments.

    All environments must implement these methods to be compatible
    with the training system. Uses structural subtyping - no explicit
    inheritance required.

    Attributes:
        action_space_size: Number of discrete actions available.
        state_space_size: Size of the state space (for discrete envs).
    """

    action_space_size: int
    state_space_size: int

    def reset(self) -> dict:
        """Reset the environment to initial state.

        Returns:
            dict: Initial state representation containing at minimum:
                - 'observation': The initial observation
                - 'info': Additional environment-specific information
        """
        ...

    def step(self, action: int) -> tuple[dict, float, bool]:
        """Execute one step in the environment.

        Args:
            action: Integer representing the action to take.

        Returns:
            tuple containing:
                - state (dict): New state after action
                - reward (float): Reward received from this step
                - done (bool): Whether episode has terminated
        """
        ...

    def render_state(self) -> dict:
        """Generate JSON-serializable state for frontend visualization.

        This method provides all data needed by the frontend to render
        the current environment state, including positions, values,
        and any visual elements.

        Returns:
            dict: JSON-serializable dictionary containing:
                - 'type': Environment type identifier
                - 'state': Current state data for rendering
                - 'metadata': Additional rendering information
        """
        ...

    def get_valid_actions(self, state: dict) -> list[int]:
        """Return list of valid actions for the given state.

        Args:
            state: Current state dictionary.

        Returns:
            list[int]: List of valid action indices.
        """
        ...

    def get_state_id(self, state: dict) -> int:
        """Convert state dict to unique integer identifier.

        Useful for tabular methods that need state indexing.

        Args:
            state: State dictionary to convert.

        Returns:
            int: Unique integer identifier for this state.
        """
        ...

