"""FrozenLake environment - wrapper for Gymnasium's FrozenLake-v1.

Navigate a frozen lake from start (S) to goal (G) without falling in holes (H).
The ice is slippery, so movement is stochastic.
"""

import gymnasium as gym
import numpy as np


class FrozenLakeEnv:
    """FrozenLake-v1 wrapper implementing BaseEnv protocol.

    Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
    Tiles: S=start, F=frozen, H=hole, G=goal
    
    Rewards: Shaped reward system to make learning easier for tabular methods:
    - Distance reward: Closer to goal gets higher reward (based on Manhattan distance)
    - Step penalty: Small constant penalty per step (-0.01)
    - Hole penalty: Small penalty for being near holes (-0.05)
    - Goal bonus: Large bonus (10.0) when reaching goal
    """

    ACTION_LEFT: int = 0
    ACTION_DOWN: int = 1
    ACTION_RIGHT: int = 2
    ACTION_UP: int = 3

    # Reward shaping parameters
    DISTANCE_REWARD_SCALE: float = 0.5  # Reward for being closer to goal
    STEP_PENALTY: float = -0.01  # Small penalty per step
    HOLE_PENALTY: float = -0.05  # Penalty for being adjacent to holes
    GOAL_BONUS: float = 10.0  # Large bonus for reaching goal

    def __init__(self) -> None:
        """Initialize FrozenLake environment."""
        self.env = gym.make(
            "FrozenLake-v1",
            is_slippery=True,
            map_name="4x4",
            render_mode=None,
        )
        self.size: int = 4
        self.action_space_size: int = 4
        self.state_space_size: int = self.size * self.size

        self.last_observation: int | None = None
        self.last_reward: float = 0.0
        self.last_done: bool = False
        self.movement_trail: list[list[int]] = []
        self.last_action: int | None = None
        self.max_trail_length: int = 10

        self._tiles: list[list[str]] = self._extract_map()

    def reset(self) -> dict:
        """Reset environment and return initial state."""
        observation, info = self.env.reset()
        self.last_observation = int(observation)
        self.last_reward = 0.0
        self.last_done = False
        self.movement_trail = [[0, 0]]  # Start position
        self.last_action = None
        return self._build_state_dict(self.last_observation)

    def step(self, action: int) -> tuple[dict, float, bool]:
        """Execute action and return new state, reward, done."""
        if self.last_done:
            return self._build_state_dict(self.last_observation or 0), 0.0, True

        # Track action
        self.last_action = action
        
        # Get old position before step
        old_pos = self._index_to_pos(self.last_observation) if self.last_observation is not None else [0, 0]

        # Use Gymnasium's step for deterministic execution (model-free algorithms)
        # Note: The transition model (used by Value/Policy Iteration) has stochastic probabilities
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.last_observation = int(observation)
        self.last_done = terminated or truncated
        
        # Apply reward shaping to make learning easier for tabular methods
        self.last_reward = self._compute_shaped_reward(
            self.last_observation,
            self.last_done
        )
        
        # Get new position after step
        new_pos = self._index_to_pos(self.last_observation)
        
        # Add to trail if position changed
        if new_pos != old_pos:
            self.movement_trail.append(new_pos)
            # Keep trail length limited
            if len(self.movement_trail) > self.max_trail_length:
                self.movement_trail.pop(0)

        state_dict = self._build_state_dict(self.last_observation)
        return state_dict, self.last_reward, self.last_done

    def render_state(self) -> dict:
        """Return JSON-serializable state for frontend visualization."""
        if self.last_observation is None:
            agent_pos = [0, 0]
        else:
            agent_pos = self._index_to_pos(self.last_observation)

        return {
            "type": "frozenlake",
            "size": self.size,
            "agent_pos": agent_pos,
            "tiles": self._tiles,
            "movement_trail": self.movement_trail,
            "last_action": self.last_action,
            "reward": self.last_reward,
            "done": self.last_done,
        }

    def get_valid_actions(self, state: dict) -> list[int]:
        """Return valid actions (all four directions always valid)."""
        return [
            self.ACTION_LEFT,
            self.ACTION_DOWN,
            self.ACTION_RIGHT,
            self.ACTION_UP,
        ]

    def get_state_id(self, state: dict) -> int:
        """Return state index from state dictionary."""
        return state.get("state_index", 0)

    def _build_state_dict(self, state_index: int) -> dict:
        """Build state dictionary from state index."""
        agent_pos = self._index_to_pos(state_index)
        return {
            "state_index": state_index,
            "agent_pos": agent_pos,
            "size": self.size,
            "observation": state_index,
            "info": {"is_slippery": True},
        }

    def _index_to_pos(self, index: int) -> list[int]:
        """Convert linear index to [row, col] position."""
        row = index // self.size
        col = index % self.size
        return [row, col]

    def _pos_to_index(self, row: int, col: int) -> int:
        """Convert [row, col] position to linear index."""
        return row * self.size + col

    def _extract_map(self) -> list[list[str]]:
        """Extract tile map from gymnasium environment."""
        desc = self.env.unwrapped.desc
        tiles: list[list[str]] = []

        for row in desc:
            tile_row: list[str] = []
            for cell in row:
                if isinstance(cell, bytes):
                    tile_row.append(cell.decode("utf-8"))
                else:
                    tile_row.append(str(cell))
            tiles.append(tile_row)

        return tiles

    def get_tile_at(self, row: int, col: int) -> str:
        """Get tile type at given position."""
        if 0 <= row < self.size and 0 <= col < self.size:
            return self._tiles[row][col]
        return "F"

    def is_hole(self, row: int, col: int) -> bool:
        """Check if position is a hole."""
        return self.get_tile_at(row, col) == "H"

    def is_goal(self, row: int, col: int) -> bool:
        """Check if position is the goal."""
        return self.get_tile_at(row, col) == "G"

    def get_transition_model(self) -> dict:
        """Return transition dynamics for model-based algorithms.

        FrozenLake is slippery: intended action has 1/3 probability,
        and each perpendicular direction also has 1/3 probability.

        Returns:
            dict: Transition model with structure:
                  {state_id: {action: [(prob, next_state_id, reward, done)]}}
        """
        model: dict = {}

        # Direction vectors for each action: LEFT, DOWN, RIGHT, UP
        deltas = {
            0: (0, -1),   # LEFT
            1: (1, 0),    # DOWN
            2: (0, 1),    # RIGHT
            3: (-1, 0),   # UP
        }

        # Perpendicular actions for slippery transitions
        perpendicular = {
            0: [1, 3],  # LEFT -> can slip to DOWN or UP
            1: [0, 2],  # DOWN -> can slip to LEFT or RIGHT
            2: [1, 3],  # RIGHT -> can slip to DOWN or UP
            3: [0, 2],  # UP -> can slip to LEFT or RIGHT
        }

        for state_id in range(self.state_space_size):
            row = state_id // self.size
            col = state_id % self.size
            tile = self.get_tile_at(row, col)

            model[state_id] = {}

            # Terminal states (hole or goal)
            if tile == "H" or tile == "G":
                for action in range(4):
                    model[state_id][action] = [(1.0, state_id, 0.0, True)]
                continue

            for action in range(4):
                # Stochastic transitions: all 4 directions have equal probability
                transitions: list = []
                prob = 1.0 / 4.0  # Equal probability for each direction

                # Consider all 4 possible directions (LEFT, DOWN, RIGHT, UP)
                for direction in range(4):
                    dr, dc = deltas[direction]
                    new_row = row + dr
                    new_col = col + dc

                    # Clamp to grid bounds
                    if not (0 <= new_row < self.size and 0 <= new_col < self.size):
                        new_row, new_col = row, col

                    next_state_id = new_row * self.size + new_col
                    next_tile = self.get_tile_at(new_row, new_col)

                    is_goal = next_tile == "G"
                    is_hole = next_tile == "H"
                    reward = 1.0 if is_goal else 0.0
                    done = is_goal or is_hole

                    transitions.append((prob, next_state_id, reward, done))

                model[state_id][action] = transitions

        return model

    def _compute_shaped_reward(self, state_index: int, done: bool) -> float:
        """Compute shaped reward to guide learning toward the goal.
        
        Reward shaping components:
        1. Distance reward: Closer to goal = higher reward (based on Manhattan distance)
        2. Step penalty: Small constant penalty per step
        3. Hole penalty: Small penalty for being adjacent to holes (discourages risky paths)
        4. Goal bonus: Large bonus when reaching the goal
        
        This makes the sparse reward problem much easier for tabular methods
        while maintaining the original objective.
        
        Args:
            state_index: Current state index (0-15 for 4x4 grid).
            done: Whether episode terminated (goal reached or hole).
            
        Returns:
            float: Shaped reward value.
        """
        # Get current position
        row, col = self._index_to_pos(state_index)
        tile = self.get_tile_at(row, col)
        
        # Large bonus for reaching the goal
        if done and tile == "G":
            return self.GOAL_BONUS
        
        # Penalty for falling in a hole
        if done and tile == "H":
            return -1.0
        
        # Find goal position (typically bottom-right in 4x4)
        goal_row, goal_col = self._find_goal_position()
        
        # Distance-based reward: Manhattan distance to goal
        # Normalize by maximum possible distance (diagonal of grid)
        max_distance = (self.size - 1) * 2  # For 4x4: (4-1)*2 = 6
        manhattan_distance = abs(row - goal_row) + abs(col - goal_col)
        distance_normalized = 1.0 - (manhattan_distance / max_distance)  # Closer = higher value
        distance_reward = self.DISTANCE_REWARD_SCALE * distance_normalized
        
        # Hole proximity penalty: check if adjacent to any hole
        hole_penalty = 0.0
        adjacent_positions = [
            (row - 1, col),  # UP
            (row + 1, col),  # DOWN
            (row, col - 1),  # LEFT
            (row, col + 1),  # RIGHT
        ]
        for adj_row, adj_col in adjacent_positions:
            if self.is_hole(adj_row, adj_col):
                hole_penalty += self.HOLE_PENALTY
                break  # Only count once
        
        # Combine all components
        shaped_reward = (
            distance_reward
            + hole_penalty
            + self.STEP_PENALTY
        )
        
        return float(shaped_reward)
    
    def _find_goal_position(self) -> tuple[int, int]:
        """Find the goal position in the grid.
        
        Returns:
            tuple[int, int]: (row, col) of goal position.
        """
        for row in range(self.size):
            for col in range(self.size):
                if self.is_goal(row, col):
                    return (row, col)
        # Default to bottom-right if not found
        return (self.size - 1, self.size - 1)

    def close(self) -> None:
        """Clean up gymnasium environment."""
        self.env.close()

