"""Gym4Real environment - custom 2D point robot navigation.

A simple abstract environment for future physical/robotics-like setups.
Agent moves a 2D point towards a goal position.
State is discretized for compatibility with tabular RL methods.
"""

import numpy as np


ACTIONS: dict[int, str] = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
}

ACTION_DELTAS: dict[int, tuple[float, float]] = {
    0: (0.0, 1.0),   # UP: +y
    1: (1.0, 0.0),   # RIGHT: +x
    2: (0.0, -1.0),  # DOWN: -y
    3: (-1.0, 0.0),  # LEFT: -x
}


class Gym4RealEnv:
    """2D point robot environment implementing BaseEnv protocol.

    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    State: 2D position [x, y] on a plane (discretized for tabular methods)
    Goal: Reach target position [5.0, 5.0]
    Rewards: +1.0 at goal, -0.01 per step otherwise.
    """

    GOAL_THRESHOLD: float = 0.5
    STEP_PENALTY: float = -0.01
    GOAL_REWARD: float = 1.0

    # Grid bounds for discretization
    GRID_MIN: float = -1.0
    GRID_MAX: float = 7.0
    NUM_BINS: int = 8

    def __init__(self) -> None:
        """Initialize Gym4Real environment."""
        self.position: list[float] = [0.0, 0.0]
        self.goal: list[float] = [5.0, 5.0]
        self.step_size: float = 1.0
        self.step_count: int = 0
        self.max_steps: int = 50

        self.action_space_size: int = 4
        self.state_space_size: int = self.NUM_BINS * self.NUM_BINS

        self.last_reward: float = 0.0
        self.last_done: bool = False
        self.movement_trail: list[list[float]] = [[0.0, 0.0]]
        self.last_action: int | None = None
        self.max_trail_length: int = 20

    def reset(self) -> dict:
        """Reset environment to initial state."""
        self.position = [0.0, 0.0]
        self.step_count = 0
        self.last_reward = 0.0
        self.last_done = False
        self.movement_trail = [[0.0, 0.0]]
        self.last_action = None
        return self._build_state_dict()

    def step(self, action: int) -> tuple[dict, float, bool]:
        """Execute action and return new state, reward, done."""
        if self.last_done:
            return self._build_state_dict(), 0.0, True

        # Track action for direction indicator
        self.last_action = action

        self._update_position(action)
        self.step_count += 1

        # Update movement trail
        self.movement_trail.append(self.position.copy())
        if len(self.movement_trail) > self.max_trail_length:
            self.movement_trail.pop(0)

        reward = self._compute_reward()
        done = self._check_done()

        self.last_reward = reward
        self.last_done = done

        return self._build_state_dict(), reward, done

    def render_state(self) -> dict:
        """Return JSON-serializable state for frontend visualization."""
        return {
            "type": "gym4real",
            "position": self.position.copy(),
            "goal": self.goal.copy(),
            "movement_trail": [pos.copy() for pos in self.movement_trail],
            "last_action": self.last_action,
            "step": self.step_count,
            "reward": self.last_reward,
            "done": self.last_done,
            "max_steps": self.max_steps,
        }

    def get_valid_actions(self, state: dict) -> list[int]:
        """Return valid actions (all four directions always valid)."""
        return [0, 1, 2, 3]

    def get_state_id(self, state: dict) -> int:
        """Get unique state ID from discretized state."""
        x_bin = state.get("x_bin", 0)
        y_bin = state.get("y_bin", 0)
        return x_bin * self.NUM_BINS + y_bin

    def get_transition_model(self) -> dict:
        """Return transition dynamics for model-based algorithms.

        Returns:
            dict: Transition model with structure:
                  {state_id: {action: [(prob, next_state_id, reward, done)]}}
        """
        model: dict = {}
        goal_x_bin = self._discretize(
            self.goal[0], self.GRID_MIN, self.GRID_MAX, self.NUM_BINS
        )
        goal_y_bin = self._discretize(
            self.goal[1], self.GRID_MIN, self.GRID_MAX, self.NUM_BINS
        )
        goal_state_id = goal_x_bin * self.NUM_BINS + goal_y_bin

        for x_bin in range(self.NUM_BINS):
            for y_bin in range(self.NUM_BINS):
                state_id = x_bin * self.NUM_BINS + y_bin
                model[state_id] = {}

                for action in range(4):
                    dx, dy = ACTION_DELTAS[action]
                    next_x = x_bin + int(dx)
                    next_y = y_bin + int(dy)

                    next_x = max(0, min(self.NUM_BINS - 1, next_x))
                    next_y = max(0, min(self.NUM_BINS - 1, next_y))

                    next_state_id = next_x * self.NUM_BINS + next_y
                    is_goal = next_state_id == goal_state_id
                    reward = self.GOAL_REWARD if is_goal else self.STEP_PENALTY
                    done = is_goal

                    model[state_id][action] = [(1.0, next_state_id, reward, done)]

        return model

    def _build_state_dict(self) -> dict:
        """Build current state dictionary with discretized position."""
        x_bin = self._discretize(
            self.position[0], self.GRID_MIN, self.GRID_MAX, self.NUM_BINS
        )
        y_bin = self._discretize(
            self.position[1], self.GRID_MIN, self.GRID_MAX, self.NUM_BINS
        )
        return {
            "x_bin": x_bin,
            "y_bin": y_bin,
        }

    def _update_position(self, action: int) -> None:
        """Update position based on action."""
        delta = ACTION_DELTAS.get(action, (0.0, 0.0))
        self.position[0] += delta[0] * self.step_size
        self.position[1] += delta[1] * self.step_size

    def _compute_reward(self) -> float:
        """Compute reward based on distance to goal."""
        distance = self._distance_to_goal()
        if distance <= self.GOAL_THRESHOLD:
            return self.GOAL_REWARD
        return self.STEP_PENALTY

    def _check_done(self) -> bool:
        """Check if episode is done."""
        reached_goal = self._distance_to_goal() <= self.GOAL_THRESHOLD
        out_of_steps = self.step_count >= self.max_steps
        return reached_goal or out_of_steps

    def _distance_to_goal(self) -> float:
        """Calculate Euclidean distance to goal."""
        dx = self.position[0] - self.goal[0]
        dy = self.position[1] - self.goal[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def _discretize(
        self, value: float, min_val: float, max_val: float, num_bins: int
    ) -> int:
        """Discretize continuous value into bin index."""
        clipped = np.clip(value, min_val, max_val)
        normalized = (clipped - min_val) / (max_val - min_val)
        bin_index = int(normalized * (num_bins - 1))
        return bin_index
