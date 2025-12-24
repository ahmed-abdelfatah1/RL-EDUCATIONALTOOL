"""Value snapshot - converts tabular value functions to 2D grids.

Provides visualization helpers for environments like GridWorld.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


StateKey = Tuple[Any, ...]
ValueFunction = Dict[StateKey, float]
QValueFunction = Dict[StateKey, np.ndarray]


def value_function_to_grid(
    value_function: ValueFunction,
    rows: int,
    cols: int,
) -> List[List[float]]:
    """Convert state-value function to 2D grid.

    Args:
        value_function: Dict mapping (row, col) to value.
        rows: Number of rows in grid.
        cols: Number of columns in grid.

    Returns:
        List[List[float]]: 2D grid of values for JSON serialization.
    """
    grid = np.zeros((rows, cols), dtype=np.float64)

    for state_key, value in value_function.items():
        if len(state_key) >= 2:
            row, col = state_key[0], state_key[1]
            if 0 <= row < rows and 0 <= col < cols:
                grid[row, col] = value

    return grid.tolist()


def q_values_to_grid(
    q_values: QValueFunction,
    rows: int,
    cols: int,
    num_actions: int,
) -> List[List[List[float]]]:
    """Convert Q-value function to 3D grid (row, col, action).

    Args:
        q_values: Dict mapping (row, col) to array of action values.
        rows: Number of rows in grid.
        cols: Number of columns in grid.
        num_actions: Number of actions per state.

    Returns:
        List[List[List[float]]]: 3D grid for JSON serialization.
    """
    grid = np.zeros((rows, cols, num_actions), dtype=np.float64)

    for state_key, action_values in q_values.items():
        if len(state_key) >= 2:
            row, col = state_key[0], state_key[1]
            if 0 <= row < rows and 0 <= col < cols:
                grid[row, col, :] = action_values[:num_actions]

    return grid.tolist()


def policy_to_grid(
    q_values: QValueFunction,
    rows: int,
    cols: int,
) -> List[List[int]]:
    """Extract greedy policy from Q-values as 2D grid of actions.

    Args:
        q_values: Dict mapping (row, col) to array of action values.
        rows: Number of rows in grid.
        cols: Number of columns in grid.

    Returns:
        List[List[int]]: 2D grid of best action indices.
    """
    grid = np.zeros((rows, cols), dtype=np.int32)

    for state_key, action_values in q_values.items():
        if len(state_key) >= 2:
            row, col = state_key[0], state_key[1]
            if 0 <= row < rows and 0 <= col < cols:
                grid[row, col] = int(np.argmax(action_values))

    return grid.tolist()

