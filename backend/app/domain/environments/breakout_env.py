"""Breakout environment - simplified simulation for educational purposes.

A simple brick-breaking game simulation without requiring Atari ROMs.
"""

import numpy as np


class BreakoutEnv:
    """Simplified Breakout environment implementing BaseEnv protocol.

    Actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
    Simulates basic paddle/ball/brick mechanics.
    """

    ACTION_NOOP: int = 0
    ACTION_FIRE: int = 1
    ACTION_RIGHT: int = 2
    ACTION_LEFT: int = 3

    DEFAULT_LIVES: int = 5
    MAX_STEPS: int = 1000

    def __init__(self) -> None:
        """Initialize Breakout environment."""
        self.action_space_size: int = 4
        self.state_space_size: int = -1

        self.lives: int = self.DEFAULT_LIVES
        self.score: float = 0.0
        self.step_count: int = 0
        self.last_reward: float = 0.0
        self.done: bool = False

        self.paddle_pos: float = 0.5
        self.ball_active: bool = False

    def reset(self) -> dict:
        """Reset environment and return initial state."""
        self.lives = self.DEFAULT_LIVES
        self.score = 0.0
        self.step_count = 0
        self.last_reward = 0.0
        self.done = False
        self.paddle_pos = 0.5
        self.ball_active = False
        return self._build_state_dict()

    def step(self, action: int) -> tuple[dict, float, bool]:
        """Execute action and return new state, reward, done."""
        if self.done:
            return self._build_state_dict(), 0.0, True

        self._apply_action(action)
        reward = self._simulate_step()

        self.last_reward = reward
        self.score += max(0, reward)
        self.step_count += 1

        self._check_done()

        return self._build_state_dict(), reward, self.done

    def render_state(self) -> dict:
        """Return JSON-serializable state for frontend visualization."""
        return {
            "type": "breakout",
            "score": self.score,
            "lives": self.lives,
            "step": self.step_count,
            "done": self.done,
            "last_reward": self.last_reward,
        }

    def get_valid_actions(self, state: dict) -> list[int]:
        """Return valid actions."""
        return [
            self.ACTION_NOOP,
            self.ACTION_FIRE,
            self.ACTION_RIGHT,
            self.ACTION_LEFT,
        ]

    def get_state_id(self, state: dict) -> int:
        """Return step count as state identifier."""
        return state.get("step", self.step_count)

    def _apply_action(self, action: int) -> None:
        """Apply action to move paddle."""
        if action == self.ACTION_LEFT:
            self.paddle_pos = max(0.0, self.paddle_pos - 0.1)
        elif action == self.ACTION_RIGHT:
            self.paddle_pos = min(1.0, self.paddle_pos + 0.1)
        elif action == self.ACTION_FIRE:
            self.ball_active = True

    def _simulate_step(self) -> float:
        """Simulate game step and return reward."""
        if not self.ball_active:
            return 0.0

        if np.random.random() < 0.1:
            return float(np.random.randint(1, 5))

        if np.random.random() < 0.02:
            self.lives -= 1
            self.ball_active = False
            return -1.0

        return 0.0

    def _check_done(self) -> None:
        """Check if game is over."""
        if self.lives <= 0 or self.step_count >= self.MAX_STEPS:
            self.done = True

    def _build_state_dict(self) -> dict:
        """Build current state dictionary."""
        return {
            "score": self.score,
            "lives": self.lives,
            "step": self.step_count,
            "done": self.done,
            "observation": self.step_count,
            "info": {"game": "breakout", "max_lives": self.DEFAULT_LIVES},
        }

    def close(self) -> None:
        """Clean up (no-op for simulated env)."""
        pass
