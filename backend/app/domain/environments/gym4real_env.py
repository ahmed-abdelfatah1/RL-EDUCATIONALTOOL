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
    Rewards: Shaped reward system that encourages moving toward the goal.
    """

    GOAL_THRESHOLD: float = 0.5
    
    # Reward shaping parameters - tuned to prevent getting stuck
    DISTANCE_REWARD_SCALE: float = 15.0  # Strong reward for being closer to goal
    PROGRESS_BONUS_SCALE: float = 10.0  # Strong bonus for making progress
    DIRECTION_BONUS_SCALE: float = 3.0  # Bonus for moving in correct direction
    STEP_PENALTY: float = -0.01  # Very small penalty per step (reduced to encourage exploration)
    GOAL_BONUS: float = 100.0  # Very large bonus for reaching goal
    
    # Maximum possible distance (from corner to goal for normalization)
    MAX_DISTANCE: float = np.sqrt(5.0**2 + 5.0**2)  # ~7.07

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
        self.previous_distance: float | None = None  # Track previous distance for progress reward
        self.previous_position: list[float] | None = None  # Track previous position for directional reward

    def reset(self) -> dict:
        """Reset environment to initial state."""
        self.position = [0.0, 0.0]
        self.step_count = 0
        self.last_reward = 0.0
        self.last_done = False
        self.movement_trail = [[0.0, 0.0]]
        self.last_action = None
        self.previous_distance = self._distance_to_goal()
        self.previous_position = [0.0, 0.0]
        return self._build_state_dict()

    def step(self, action: int) -> tuple[dict, float, bool]:
        """Execute action and return new state, reward, done."""
        if self.last_done:
            return self._build_state_dict(), 0.0, True

        # Track action for direction indicator
        self.last_action = action

        hit_boundary = self._update_position(action)
        self.step_count += 1

        # Update movement trail
        self.movement_trail.append(self.position.copy())
        if len(self.movement_trail) > self.max_trail_length:
            self.movement_trail.pop(0)

        reward = self._compute_shaped_reward()
        
        # Add penalty for hitting boundary (discourages getting stuck at edges)
        if hit_boundary:
            reward -= 2.0
        
        done = self._check_done()

        self.last_reward = reward
        self.last_done = done
        self.previous_distance = self._distance_to_goal()
        self.previous_position = self.position.copy()

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
                    # Stochastic transitions: all 4 directions have equal probability
                    transitions: list = []
                    prob = 1.0 / 4.0  # Equal probability for each direction
                    
                    # Consider all 4 possible directions (UP, RIGHT, DOWN, LEFT)
                    for direction in range(4):
                        dx, dy = ACTION_DELTAS[direction]
                        next_x = x_bin + int(dx)
                        next_y = y_bin + int(dy)
                        
                        # Check if movement would go out of bounds
                        hit_boundary_x = next_x < 0 or next_x >= self.NUM_BINS
                        hit_boundary_y = next_y < 0 or next_y >= self.NUM_BINS
                        hit_boundary = hit_boundary_x or hit_boundary_y
                        
                        # Clamp to valid bounds
                        next_x = max(0, min(self.NUM_BINS - 1, next_x))
                        next_y = max(0, min(self.NUM_BINS - 1, next_y))

                        next_state_id = next_x * self.NUM_BINS + next_y
                        is_goal = next_state_id == goal_state_id
                        
                        # Compute shaped reward for transition model
                        if is_goal:
                            reward = self.GOAL_BONUS
                        elif hit_boundary:
                            # Penalty for hitting boundary (trying to move out of bounds)
                            # This discourages getting stuck at boundaries
                            reward = -2.0 + self.STEP_PENALTY
                        else:
                            # Calculate distance-based reward for next state
                            # Convert bin indices back to approximate continuous positions
                            next_x_pos = self.GRID_MIN + (next_x / (self.NUM_BINS - 1)) * (self.GRID_MAX - self.GRID_MIN)
                            next_y_pos = self.GRID_MIN + (next_y / (self.NUM_BINS - 1)) * (self.GRID_MAX - self.GRID_MIN)
                            next_distance = np.sqrt((next_x_pos - self.goal[0])**2 + (next_y_pos - self.goal[1])**2)
                            
                            # Strong distance-based reward with squared term for steeper gradient
                            normalized_distance = next_distance / self.MAX_DISTANCE
                            distance_reward = self.DISTANCE_REWARD_SCALE * ((1.0 - normalized_distance) ** 2)
                            
                            # Calculate current state position for progress calculation
                            current_x_pos = self.GRID_MIN + (x_bin / (self.NUM_BINS - 1)) * (self.GRID_MAX - self.GRID_MIN)
                            current_y_pos = self.GRID_MIN + (y_bin / (self.NUM_BINS - 1)) * (self.GRID_MAX - self.GRID_MIN)
                            current_distance = np.sqrt((current_x_pos - self.goal[0])**2 + (current_y_pos - self.goal[1])**2)
                            
                            # Progress bonus for moving closer
                            distance_improvement = current_distance - next_distance
                            progress_bonus = self.PROGRESS_BONUS_SCALE * max(0.0, distance_improvement)
                            
                            # Directional bonus: reward for moving toward goal
                            dx_movement = next_x_pos - current_x_pos
                            dy_movement = next_y_pos - current_y_pos
                            dx_to_goal = self.goal[0] - current_x_pos
                            dy_to_goal = self.goal[1] - current_y_pos
                            
                            movement_mag = np.sqrt(dx_movement**2 + dy_movement**2)
                            goal_mag = np.sqrt(dx_to_goal**2 + dy_to_goal**2)
                            
                            direction_bonus = 0.0
                            if movement_mag > 0.01 and goal_mag > 0.01:
                                dot_product = (dx_movement * dx_to_goal + dy_movement * dy_to_goal) / (movement_mag * goal_mag)
                                direction_bonus = self.DIRECTION_BONUS_SCALE * max(0.0, dot_product)
                            
                            reward = distance_reward + progress_bonus + direction_bonus + self.STEP_PENALTY
                        
                        done = is_goal
                        transitions.append((prob, next_state_id, reward, done))
                    
                    model[state_id][action] = transitions

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

    def _update_position(self, action: int) -> bool:
        """Update position based on action, clamping to grid bounds.
        
        Returns:
            bool: True if position was clamped (hit boundary), False otherwise.
        """
        delta = ACTION_DELTAS.get(action, (0.0, 0.0))
        new_x = self.position[0] + delta[0] * self.step_size
        new_y = self.position[1] + delta[1] * self.step_size
        
        # Check if movement would go out of bounds
        hit_boundary_x = new_x < self.GRID_MIN or new_x > self.GRID_MAX
        hit_boundary_y = new_y < self.GRID_MIN or new_y > self.GRID_MAX
        hit_boundary = hit_boundary_x or hit_boundary_y
        
        # Clamp position to grid bounds to prevent positions outside valid range
        self.position[0] = np.clip(new_x, self.GRID_MIN, self.GRID_MAX)
        self.position[1] = np.clip(new_y, self.GRID_MIN, self.GRID_MAX)
        
        return hit_boundary

    def _compute_shaped_reward(self) -> float:
        """Compute shaped reward that strongly encourages moving toward the goal.
        
        Reward components:
        - Very large bonus when reaching the goal
        - Strong distance-based reward (closer to goal = much higher reward)
        - Strong progress bonus for moving closer to goal
        - Directional bonus for moving in the correct direction (toward goal)
        - Very small step penalty (to encourage exploration)
        """
        distance = self._distance_to_goal()
        
        # Very large bonus for reaching the goal
        if distance <= self.GOAL_THRESHOLD:
            return self.GOAL_BONUS
        
        # Strong distance-based reward (normalized, closer = much higher)
        # Use squared distance to make reward gradient steeper near goal
        normalized_distance = distance / self.MAX_DISTANCE
        # Square the (1 - normalized_distance) to make reward increase faster as we get closer
        distance_reward = self.DISTANCE_REWARD_SCALE * ((1.0 - normalized_distance) ** 2)
        
        # Strong progress bonus: reward for moving closer to goal
        progress_bonus = 0.0
        if self.previous_distance is not None:
            distance_improvement = self.previous_distance - distance
            if distance_improvement > 0:
                # Strong bonus for making progress (scaled by improvement)
                progress_bonus = self.PROGRESS_BONUS_SCALE * distance_improvement
            elif distance_improvement < 0:
                # Small penalty for moving away (but not too harsh to allow exploration)
                progress_bonus = 2.0 * distance_improvement  # Negative value
        
        # Directional bonus: reward for moving in the direction toward goal
        direction_bonus = 0.0
        if self.previous_position is not None:
            # Calculate direction vector from previous to current position
            dx_movement = self.position[0] - self.previous_position[0]
            dy_movement = self.position[1] - self.previous_position[1]
            
            # Calculate direction vector from current position to goal
            dx_to_goal = self.goal[0] - self.position[0]
            dy_to_goal = self.goal[1] - self.position[1]
            
            # Normalize vectors
            movement_mag = np.sqrt(dx_movement**2 + dy_movement**2)
            goal_mag = np.sqrt(dx_to_goal**2 + dy_to_goal**2)
            
            if movement_mag > 0.01 and goal_mag > 0.01:  # Avoid division by zero
                # Dot product to measure alignment (cosine of angle)
                dot_product = (dx_movement * dx_to_goal + dy_movement * dy_to_goal) / (movement_mag * goal_mag)
                # Reward for moving in the direction of goal (0 to 1, where 1 is perfect alignment)
                direction_bonus = self.DIRECTION_BONUS_SCALE * max(0.0, dot_product)
        
        # Total reward: distance reward + progress bonus + direction bonus + step penalty
        total_reward = distance_reward + progress_bonus + direction_bonus + self.STEP_PENALTY
        
        return total_reward

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
