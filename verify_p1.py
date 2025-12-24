import sys
import os
import numpy as np

# Add backend directory to sys.path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from app.domain.environments.gridworld_env import GridWorldEnv
from app.domain.algorithms.policy_evaluation import PolicyIteration

def debug_policy_iteration():
    print("Debugging Policy Iteration on GridWorld...")
    
    env = GridWorldEnv()
    model = env.get_transition_model()
    
    # Build transition function
    def transition_fn(state: tuple, action: int):
        state_id = state[0]
        transitions = model.get(state_id, {}).get(action, [])
        # Return ((next_s,), prob, reward)
        return [((t[1],), t[0], t[2]) for t in transitions]

    num_states = env.state_space_size
    states = tuple((s,) for s in range(num_states))
    
    agent = PolicyIteration(
        discount_factor=0.99,
        num_actions=4,
        theta=0.0001
    )
    
    print("Running Policy Iteration...")
    agent.run_iteration(states, transition_fn)
    
    print("\nValue Function (Grid 5x5):")
    grid_values = np.zeros((5, 5))
    for s in range(num_states):
        grid_values[s // 5, s % 5] = agent.get_value((s,))
    print(grid_values)
    
    print("\nPolicy (Grid 5x5):")
    # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    policy_grid = np.zeros((5, 5), dtype=int)
    for s in range(num_states):
        policy_grid[s // 5, s % 5] = agent.get_action((s,))
    print(policy_grid)
    
    # Check goal neighbors
    # Goal is (4,4) -> state 24.
    # (4,3) -> state 23. Should move RIGHT (1).
    # (3,4) -> state 19. Should move DOWN (2).
    
    s23_action = agent.get_action((23,))
    s19_action = agent.get_action((19,))
    s23_value = agent.get_value((23,))
    s19_value = agent.get_value((19,))
    
    print(f"\nState 23 (4,3) Action: {s23_action} (Expected 1), Value: {s23_value}")
    print(f"State 19 (3,4) Action: {s19_action} (Expected 2), Value: {s19_value}")
    
    if s23_action == 1 and s19_action == 2:
        print("SUCCESS: Policy looks reasonable near goal.")
    else:
        print("FAILURE: Policy is incorrect near goal.")

if __name__ == "__main__":
    debug_policy_iteration()
