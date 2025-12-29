"""MountainCar environment - wrapper for Gymnasium's MountainCar-v0.

Classic control task: drive an underpowered car up a steep mountain.
The car must build momentum by swinging back and forth.

State is discretized for compatibility with tabular RL methods.
"""

import gymnasium as gym
import numpy as np


class MountainCarEnv:
    """MountainCar-v0 wrapper implementing BaseEnv protocol.

    Actions: 0=LEFT (accelerate left), 1=NONE, 2=RIGHT (accelerate right)
    Observations: car position (-1.2 to 0.6), car velocity (-0.07 to 0.07)
    
    Rewards: Shaped reward system to make learning easier for tabular methods:
    - Position reward: Higher positions (closer to goal) get higher rewards
    - Velocity reward: Higher velocity gets reward (especially rightward)
    - Step penalty: Small constant penalty per step (-0.1)
    - Goal bonus: Large bonus (100.0) when reaching goal at position >= 0.5
    
    State is discretized into bins for tabular method compatibility.
    """

    ACTION_LEFT: int = 0
    ACTION_NONE: int = 1
    ACTION_RIGHT: int = 2

    POSITION_MIN: float = -1.2
    POSITION_MAX: float = 0.6
    VELOCITY_MIN: float = -0.07
    VELOCITY_MAX: float = 0.07
    GOAL_POSITION: float = 0.5

    # Discretization bins
    NUM_POS_BINS: int = 20
    NUM_VEL_BINS: int = 20

    # Reward shaping parameters
    POSITION_REWARD_SCALE: float = 2.0  # Reward for being at higher positions
    VELOCITY_REWARD_SCALE: float = 1.0  # Reward for having higher velocity
    STEP_PENALTY: float = -0.1  # Small penalty per step (reduced from -1)
    GOAL_BONUS: float = 100.0  # Large bonus for reaching goal

    def __init__(self) -> None:
        """Initialize MountainCar environment."""
        self.env = gym.make("MountainCar-v0", render_mode=None)
        self.action_space_size: int = 3
        self.state_space_size: int = self.NUM_POS_BINS * self.NUM_VEL_BINS

        self.last_observation: np.ndarray | None = None
        self.last_reward: float = 0.0
        self.last_done: bool = False
        self.last_action: int | None = None

    def reset(self) -> dict:
        """Reset environment and return initial state."""
        observation, info = self.env.reset()
        self.last_observation = np.array(observation, dtype=np.float32)
        self.last_reward = 0.0
        self.last_done = False
        self.last_action = None
        return self._observation_to_dict(self.last_observation)

    def step(self, action: int) -> tuple[dict, float, bool]:
        """Execute action and return new state, reward, done."""
        if self.last_done:
            return self._get_current_state_dict(), 0.0, True

        observation, reward, terminated, truncated, info = self.env.step(action)
        self.last_observation = np.array(observation, dtype=np.float32)
        self.last_done = terminated or truncated
        self.last_action = action

        # Apply reward shaping to make learning easier for tabular methods
        self.last_reward = self._compute_shaped_reward(
            self.last_observation[0],  # position
            self.last_observation[1],  # velocity
            self.last_done
        )

        state_dict = self._observation_to_dict(self.last_observation)
        return state_dict, self.last_reward, self.last_done

    def render_state(self) -> dict:
        """Return JSON-serializable state for frontend visualization."""
        if self.last_observation is None:
            return self._empty_render_state()

        return {
            "type": "mountaincar",
            "position": float(self.last_observation[0]),
            "velocity": float(self.last_observation[1]),
            "action": self.last_action,
            "reward": self.last_reward,
            "done": self.last_done,
            "goal_position": self.GOAL_POSITION,
            "position_bounds": [self.POSITION_MIN, self.POSITION_MAX],
        }

    def get_valid_actions(self, state: dict) -> list[int]:
        """Return valid actions (all three always valid)."""
        return [self.ACTION_LEFT, self.ACTION_NONE, self.ACTION_RIGHT]

    def get_state_id(self, state: dict) -> int:
        """Get unique state ID from discretized state."""
        pos_bin = state.get("pos_bin", 0)
        vel_bin = state.get("vel_bin", 0)
        return pos_bin * self.NUM_VEL_BINS + vel_bin

    def _observation_to_dict(self, observation: np.ndarray) -> dict:
        """Convert numpy observation to discretized state dictionary.

        Returns discretized bins for tabular methods to work properly.
        """
        position = float(observation[0])
        velocity = float(observation[1])

        # Discretize for tabular methods
        pos_bin = self._discretize(
            position, self.POSITION_MIN, self.POSITION_MAX, self.NUM_POS_BINS
        )
        vel_bin = self._discretize(
            velocity, self.VELOCITY_MIN, self.VELOCITY_MAX, self.NUM_VEL_BINS
        )

        return {
            "pos_bin": pos_bin,
            "vel_bin": vel_bin,
        }

    def _get_current_state_dict(self) -> dict:
        """Get current state as dictionary."""
        if self.last_observation is None:
            return self._empty_state_dict()
        return self._observation_to_dict(self.last_observation)

    def _empty_state_dict(self) -> dict:
        """Return empty state dictionary before reset."""
        return {"pos_bin": 10, "vel_bin": 10}

    def _empty_render_state(self) -> dict:
        """Return empty render state before reset."""
        return {
            "type": "mountaincar",
            "position": -0.5,
            "velocity": 0.0,
            "action": None,
            "reward": 0.0,
            "done": False,
            "goal_position": self.GOAL_POSITION,
            "position_bounds": [self.POSITION_MIN, self.POSITION_MAX],
        }

    def _discretize(
        self, value: float, min_val: float, max_val: float, num_bins: int
    ) -> int:
        """Discretize continuous value into bin index."""
        clipped = np.clip(value, min_val, max_val)
        normalized = (clipped - min_val) / (max_val - min_val)
        bin_index = int(normalized * (num_bins - 1))
        return bin_index

    def _compute_shaped_reward(
        self, position: float, velocity: float, done: bool
    ) -> float:
        """Compute shaped reward to guide learning toward the goal.
        
        Reward shaping components:
        1. Position reward: Higher position = higher reward (normalized 0-1)
        2. Velocity reward: Higher absolute velocity = higher reward (especially rightward)
        3. Step penalty: Small constant penalty per step
        4. Goal bonus: Large bonus when reaching the goal
        
        This makes the sparse reward problem much easier for tabular methods
        while maintaining the original objective.
        
        Args:
            position: Current car position.
            velocity: Current car velocity.
            done: Whether episode terminated (goal reached).
            
        Returns:
            float: Shaped reward value.
        """
        # Large bonus for reaching the goal
        if done:
            return self.GOAL_BONUS
        
        # Position-based reward: normalize position to [0, 1] range
        # Higher positions (closer to goal) get higher rewards
        position_normalized = (position - self.POSITION_MIN) / (
            self.POSITION_MAX - self.POSITION_MIN
        )
        position_reward = self.POSITION_REWARD_SCALE * position_normalized
        
        # Velocity-based reward: reward higher absolute velocity
        # Slightly favor rightward velocity (positive) for building momentum
        velocity_normalized = abs(velocity) / abs(self.VELOCITY_MAX)
        velocity_bonus = 0.0
        if velocity > 0:  # Moving right (toward goal)
            velocity_bonus = self.VELOCITY_REWARD_SCALE * velocity_normalized * 1.2
        else:  # Moving left (building momentum)
            velocity_bonus = self.VELOCITY_REWARD_SCALE * velocity_normalized * 0.8
        
        # Combine all components
        shaped_reward = (
            position_reward
            + velocity_bonus
            + self.STEP_PENALTY
        )
        
        return float(shaped_reward)

    def close(self) -> None:
        """Clean up gymnasium environment."""
        self.env.close()
