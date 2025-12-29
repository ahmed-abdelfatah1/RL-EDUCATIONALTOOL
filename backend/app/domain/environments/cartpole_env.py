"""CartPole environment - wrapper for Gymnasium's CartPole-v1.

Classic control task: balance a pole on a cart by moving left or right.
State is discretized for compatibility with tabular RL methods.
"""

import gymnasium as gym
import numpy as np


class CartPoleEnv:
    """CartPole-v1 wrapper implementing BaseEnv protocol.

    Actions: 0=LEFT, 1=RIGHT
    Observations: cart position, cart velocity, pole angle, pole angular velocity
    Rewards: +1 for each step the pole remains upright.

    State is discretized into bins for tabular method compatibility.
    """

    ACTION_LEFT: int = 0
    ACTION_RIGHT: int = 1

    # Discretization parameters
    CART_POS_BINS: int = 10
    CART_VEL_BINS: int = 10
    POLE_ANGLE_BINS: int = 10
    POLE_VEL_BINS: int = 10

    # State bounds
    CART_POS_MIN: float = -2.4
    CART_POS_MAX: float = 2.4
    CART_VEL_MIN: float = -3.0
    CART_VEL_MAX: float = 3.0
    POLE_ANGLE_MIN: float = -0.21
    POLE_ANGLE_MAX: float = 0.21
    POLE_VEL_MIN: float = -3.0
    POLE_VEL_MAX: float = 3.0

    def __init__(self) -> None:
        """Initialize CartPole environment."""
        self.env = gym.make("CartPole-v1", render_mode=None)
        self.action_space_size: int = 2
        self.state_space_size: int = (
            self.CART_POS_BINS
            * self.CART_VEL_BINS
            * self.POLE_ANGLE_BINS
            * self.POLE_VEL_BINS
        )

        self.last_observation: np.ndarray | None = None
        self.last_reward: float = 0.0
        self.done: bool = False
        self.action_taken: int | None = None

    def reset(self) -> dict:
        """Reset environment and return initial state."""
        observation, info = self.env.reset()
        self.last_observation = np.array(observation, dtype=np.float32)
        self.last_reward = 0.0
        self.done = False
        self.action_taken = None
        return self._observation_to_dict(self.last_observation)

    def step(self, action: int) -> tuple[dict, float, bool]:
        """Execute action and return new state, reward, done."""
        if self.done:
            return self._get_current_state_dict(), 0.0, True

        # Track action for visualization
        self.action_taken = action

        observation, reward, terminated, truncated, info = self.env.step(action)
        self.last_observation = np.array(observation, dtype=np.float32)
        self.last_reward = float(reward)
        self.done = terminated or truncated

        state_dict = self._observation_to_dict(self.last_observation)
        return state_dict, self.last_reward, self.done

    def render_state(self) -> dict:
        """Return JSON-serializable state for frontend visualization."""
        if self.last_observation is None:
            return self._empty_render_state()

        return {
            "type": "cartpole",
            "cart_position": float(self.last_observation[0]),
            "cart_velocity": float(self.last_observation[1]),
            "pole_angle": float(self.last_observation[2]),
            "pole_angular_velocity": float(self.last_observation[3]),
            "action_taken": self.action_taken,
            "done": self.done,
            "reward": self.last_reward,
        }

    def get_valid_actions(self, state: dict) -> list[int]:
        """Return valid actions (both directions always valid)."""
        return [self.ACTION_LEFT, self.ACTION_RIGHT]

    def get_state_id(self, state: dict) -> int:
        """Get unique state ID from discretized state."""
        cp = state.get("cart_pos_bin", 0)
        cv = state.get("cart_vel_bin", 0)
        pa = state.get("pole_angle_bin", 0)
        pv = state.get("pole_vel_bin", 0)
        return (
            cp * self.CART_VEL_BINS * self.POLE_ANGLE_BINS * self.POLE_VEL_BINS
            + cv * self.POLE_ANGLE_BINS * self.POLE_VEL_BINS
            + pa * self.POLE_VEL_BINS
            + pv
        )

    def _observation_to_dict(self, observation: np.ndarray) -> dict:
        """Convert numpy observation to discretized state dictionary."""
        cart_pos_bin = self._discretize(
            float(observation[0]),
            self.CART_POS_MIN,
            self.CART_POS_MAX,
            self.CART_POS_BINS,
        )
        cart_vel_bin = self._discretize(
            float(observation[1]),
            self.CART_VEL_MIN,
            self.CART_VEL_MAX,
            self.CART_VEL_BINS,
        )
        pole_angle_bin = self._discretize(
            float(observation[2]),
            self.POLE_ANGLE_MIN,
            self.POLE_ANGLE_MAX,
            self.POLE_ANGLE_BINS,
        )
        pole_vel_bin = self._discretize(
            float(observation[3]),
            self.POLE_VEL_MIN,
            self.POLE_VEL_MAX,
            self.POLE_VEL_BINS,
        )

        return {
            "cart_pos_bin": cart_pos_bin,
            "cart_vel_bin": cart_vel_bin,
            "pole_angle_bin": pole_angle_bin,
            "pole_vel_bin": pole_vel_bin,
        }

    def _get_current_state_dict(self) -> dict:
        """Get current state as dictionary."""
        if self.last_observation is None:
            return self._empty_state_dict()
        return self._observation_to_dict(self.last_observation)

    def _empty_state_dict(self) -> dict:
        """Return empty state dictionary before reset."""
        return {
            "cart_pos_bin": 5,
            "cart_vel_bin": 5,
            "pole_angle_bin": 5,
            "pole_vel_bin": 5,
        }

    def _empty_render_state(self) -> dict:
        """Return empty render state before reset."""
        return {
            "type": "cartpole",
            "cart_position": 0.0,
            "cart_velocity": 0.0,
            "pole_angle": 0.0,
            "pole_angular_velocity": 0.0,
            "action_taken": None,
            "done": False,
            "reward": 0.0,
        }

    def _discretize(
        self, value: float, min_val: float, max_val: float, num_bins: int
    ) -> int:
        """Discretize continuous value into bin index."""
        clipped = np.clip(value, min_val, max_val)
        normalized = (clipped - min_val) / (max_val - min_val)
        bin_index = int(normalized * (num_bins - 1))
        return bin_index

    def close(self) -> None:
        """Clean up gymnasium environment."""
        self.env.close()
