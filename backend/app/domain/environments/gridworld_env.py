"""GridWorld environment - classic 5x5 grid navigation task.

Agent navigates from start [0,0] to goal [4,4] while avoiding walls.
"""

import numpy as np


class GridWorldEnv:
    """5x5 GridWorld environment implementing BaseEnv protocol.

    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    Rewards: Distance-based rewards that encourage moving closer to the goal.
    - Reaching goal: +10
    - Hitting walls: -1
    - Moving closer to goal: +0.5
    - Moving farther from goal: -0.5
    - Staying same distance: -0.1
    """

    GRID_SIZE: int = 5
    START_POS: tuple[int, int] = (0, 0)
    GOAL_POS: tuple[int, int] = (4, 4)

    ACTION_UP: int = 0
    ACTION_RIGHT: int = 1
    ACTION_DOWN: int = 2
    ACTION_LEFT: int = 3

    ACTIONS: dict[int, tuple[int, int]] = {
        0: (-1, 0),  # UP
        1: (0, 1),   # RIGHT
        2: (1, 0),   # DOWN
        3: (0, -1),  # LEFT
    }

    def __init__(self) -> None:
        """Initialize GridWorld environment."""
        self.action_space_size: int = 4
        self.state_space_size: int = self.GRID_SIZE * self.GRID_SIZE
        self.agent_pos: tuple[int, int] = self.START_POS
        self.done: bool = False
        self.agent_direction: int = 2  # Start facing DOWN
        self.movement_trail: list[tuple[int, int]] = [self.START_POS]
        self.last_reward: float = 0.0
        self.last_action: int | None = None
        self.max_trail_length: int = 15

    def reset(self) -> dict:
        """Reset environment to initial state."""
        self.agent_pos = self.START_POS
        self.done = False
        self.agent_direction = 2  # Reset facing DOWN
        self.movement_trail = [self.START_POS]
        self.last_reward = 0.0
        self.last_action = None
        return self._get_state_dict()

    def step(self, action: int) -> tuple[dict, float, bool]:
        """Execute action and return new state, reward, done."""
        if self.done:
            return self._get_state_dict(), 0.0, True

        # Track action and update direction
        self.last_action = action
        self.agent_direction = action

        # Calculate distance before move
        old_distance = self._manhattan_distance(self.agent_pos, self.GOAL_POS)

        delta = self.ACTIONS.get(action, (0, 0))
        new_row = self.agent_pos[0] + delta[0]
        new_col = self.agent_pos[1] + delta[1]

        old_pos = self.agent_pos
        self._update_position(new_row, new_col)
        
        # Calculate distance after move
        new_distance = self._manhattan_distance(self.agent_pos, self.GOAL_POS)
        
        # Compute reward based on distance change
        reward = self._compute_reward(old_pos, self.agent_pos, old_distance, new_distance)
        self.last_reward = reward
        
        # Update trail if position changed
        if self.agent_pos != old_pos:
            self.movement_trail.append(self.agent_pos)
            # Keep trail length limited
            if len(self.movement_trail) > self.max_trail_length:
                self.movement_trail.pop(0)
        
        self._check_terminal()

        return self._get_state_dict(), reward, self.done

    def render_state(self) -> dict:
        """Return JSON-serializable state for frontend visualization."""
        grid = self._build_grid_values()
        return {
            "type": "gridworld",
            "grid": grid.tolist(),
            "agent_pos": list(self.agent_pos),
            "goal_pos": list(self.GOAL_POS),
            "agent_direction": self.agent_direction,
            "movement_trail": [list(pos) for pos in self.movement_trail],
            "last_reward": self.last_reward,
            "last_action": self.last_action,
            "grid_size": self.GRID_SIZE,
            "done": self.done,
        }

    def get_valid_actions(self, state: dict) -> list[int]:
        """Return all four actions as valid (agent can try any direction)."""
        return [0, 1, 2, 3]

    def get_state_id(self, state: dict) -> int:
        """Convert state dict to unique integer identifier."""
        pos = state.get("position", self.agent_pos)
        return pos[0] * self.GRID_SIZE + pos[1]

    def _get_state_dict(self) -> dict:
        """Build current state dictionary."""
        return {
            "observation": self.get_state_id({"position": self.agent_pos}),
            "position": self.agent_pos,
            "info": {"grid_size": self.GRID_SIZE, "goal": self.GOAL_POS},
        }

    def get_transition_model(self) -> dict:
        """Return transition dynamics for model-based algorithms.

        Returns:
            dict: Transition model with structure:
                  {state_id: {action: [(prob, next_state_id, reward, done)]}}
        """
        model: dict = {}
        goal_state_id = self.GOAL_POS[0] * self.GRID_SIZE + self.GOAL_POS[1]

        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                state_id = row * self.GRID_SIZE + col
                model[state_id] = {}
                
                # Check if current state is the goal state
                is_at_goal = state_id == goal_state_id

                for action in range(4):
                    # Stochastic transitions: intended action has 50% probability,
                    # other 3 directions each have ~16.67% probability
                    transitions: list = []
                    
                    # Consider all 4 possible directions (UP, RIGHT, DOWN, LEFT)
                    for direction in range(4):
                        if direction == action:
                            # Intended action has higher probability
                            prob = 0.5
                        else:
                            # Other directions share the remaining probability
                            prob = 0.5 / 3.0  # ~0.1667 each
                        
                        delta = self.ACTIONS[direction]
                        new_row = row + delta[0]
                        new_col = col + delta[1]

                        # Check bounds
                        if not self._is_valid_position(new_row, new_col):
                            new_row, new_col = row, col  # Stay in place

                        next_state_id = new_row * self.GRID_SIZE + new_col
                        is_goal = next_state_id == goal_state_id

                        # Compute reward
                        if is_goal:
                            reward = 10.0
                        elif next_state_id == state_id:
                            reward = -1.0  # Hit wall
                        else:
                            # Distance-based reward
                            old_dist = abs(row - self.GOAL_POS[0]) + abs(col - self.GOAL_POS[1])
                            new_dist = abs(new_row - self.GOAL_POS[0]) + abs(new_col - self.GOAL_POS[1])
                            if new_dist < old_dist:
                                reward = 0.5
                            elif new_dist > old_dist:
                                reward = -0.5
                            else:
                                reward = -0.1

                        # If we're at the goal state, all transitions are terminal
                        # If we're transitioning to the goal state, that's also terminal
                        done = is_at_goal or is_goal

                        transitions.append((prob, next_state_id, reward, done))
                    
                    model[state_id][action] = transitions

        return model

    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= row < self.GRID_SIZE and 0 <= col < self.GRID_SIZE

    def _manhattan_distance(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _compute_reward(
        self, 
        old_pos: tuple[int, int], 
        new_pos: tuple[int, int],
        old_distance: int,
        new_distance: int
    ) -> float:
        """Compute distance-based reward for move.
        
        Returns:
            +10.0 if reached goal
            -1.0 if hit wall (stayed in same position)
            +0.5 if moved closer to goal
            -0.5 if moved farther from goal
            -0.1 if stayed same distance
        """
        # Check if reached goal
        if new_pos == self.GOAL_POS:
            return 10.0
        
        # Check if hit wall (didn't move)
        if old_pos == new_pos:
            return -1.0
        
        # Distance-based reward
        if new_distance < old_distance:
            return 0.5  # Moved closer
        elif new_distance > old_distance:
            return -0.5  # Moved farther
        else:
            return -0.1  # Same distance (discourage lateral movement)

    def _update_position(self, new_row: int, new_col: int) -> None:
        """Update agent position if move is valid."""
        if self._is_valid_position(new_row, new_col):
            self.agent_pos = (new_row, new_col)

    def _check_terminal(self) -> None:
        """Check if episode has terminated."""
        if self.agent_pos == self.GOAL_POS:
            self.done = True

    def _build_grid_values(self) -> np.ndarray:
        """Build grid with cell values for visualization."""
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        grid[self.GOAL_POS] = 1.0
        grid[self.agent_pos] = 0.5
        return grid

