"""Training service - connects environment, agent, and trainer.

Provides synchronous training execution for the API layer.
"""

from __future__ import annotations

from typing import List

from app.domain.algorithms import (
    MonteCarloAgent,
    NStepTDAgent,
    PolicyIteration,
    QLearningAgent,
    SarsaAgent,
    TDPrediction,
    ValueIteration,
)
from app.domain.environments import create_env
from app.domain.environments.base_env import BaseEnv
from app.domain.training.trainer import Trainer
from app.schemas.training_schemas import (
    EpisodeMetrics,
    TrainingRequest,
    TrainingRunResponse,
)
from app.services.stream_publisher import stream_publisher


# Step delay for live visualization (seconds)
LIVE_VIZ_STEP_DELAY = 0.03


def run_training(request: TrainingRequest) -> TrainingRunResponse:
    """Run a synchronous training session.

    Args:
        request: Training configuration request.

    Returns:
        TrainingRunResponse: Results of training run.
    """
    env = create_env(request.env_name)  # type: ignore[arg-type]

    # Check if this is a model-based algorithm
    if request.algorithm_name in ["policy_iteration", "value_iteration"]:
        return _run_model_based_training(request, env)

    agent = _create_agent(request, env)

    # Add episode lifecycle hooks for MC and n-step TD
    if hasattr(agent, 'start_episode'):
        agent.start_episode()

    # Check if there are any subscribers for live visualization
    has_subscribers = stream_publisher.has_subscribers(request.env_name)

    trainer = Trainer(
        env=env,
        agent=agent,
        max_steps_per_episode=request.max_steps_per_episode,
        env_name=request.env_name,
        publish_state=has_subscribers,
        step_delay=LIVE_VIZ_STEP_DELAY if has_subscribers else 0.0,
    )

    history = trainer.run_episodes(request.num_episodes)

    episodes = _convert_history(history)
    
    # Extract learned knowledge
    value_function, policy = _extract_agent_knowledge(agent, env)

    return TrainingRunResponse(
        env_name=request.env_name,
        algorithm_name=request.algorithm_name,
        episodes=episodes,
        value_function=value_function,
        policy=policy,
    )


def _run_model_based_training(
    request: TrainingRequest, env
) -> TrainingRunResponse:
    """Run training for model-based algorithms (Policy/Value Iteration).

    These algorithms require a transition model and compute the optimal
    policy before running any episodes.
    """
    from app.domain.training.episode_runner import run_episode_no_update

    # Get transition model from environment
    if not hasattr(env, 'get_transition_model'):
        raise ValueError(
            f"Environment {request.env_name} does not support model-based algorithms"
        )

    model = env.get_transition_model()
    num_actions = env.action_space_size
    num_states = env.state_space_size

    # Build transition function for algorithm
    def transition_fn(state: tuple, action: int):
        """Convert model dict to list of (next_state, prob, reward, done) tuples."""
        state_id = state[0]
        # Note: PolicyIteration/ValueIteration use tuple keys for states
        # Model format: (prob, next_state_id, reward, done)
        transitions = model.get(state_id, {}).get(action, [])
        return [((t[1],), t[0], t[2], t[3]) for t in transitions]  # ((next_s,), prob, reward, done)

    # Create all state tuples for the algorithm
    states = tuple((s,) for s in range(num_states))

    # Create and run the algorithm
    if request.algorithm_name == "policy_iteration":
        agent = PolicyIteration(
            discount_factor=request.discount_factor,
            num_actions=num_actions,
            theta=0.0001,
        )
        agent.run_iteration(states, transition_fn)
    else:  # value_iteration
        agent = ValueIteration(
            discount_factor=request.discount_factor,
            num_actions=num_actions,
            theta=0.0001,
        )
        agent.run(states, transition_fn)

    # Run evaluation episodes with the learned policy
    has_subscribers = stream_publisher.has_subscribers(request.env_name)
    history: List[dict] = []

    for ep in range(request.num_episodes):
        total_reward = 0.0
        length = 0

        for step in run_episode_no_update(
            env,
            agent,
            request.max_steps_per_episode,
            env_name=request.env_name,
            publish_state=has_subscribers,
        ):
            total_reward += step["reward"]
            length += 1
            if has_subscribers:
                import time
                time.sleep(LIVE_VIZ_STEP_DELAY)

        history.append({
            "episode": ep + 1,
            "total_reward": total_reward,
            "length": length,
        })
        
    # Extract learned knowledge
    value_function, policy = _extract_agent_knowledge(agent, env)

    return TrainingRunResponse(
        env_name=request.env_name,
        algorithm_name=request.algorithm_name,
        episodes=_convert_history(history),
        value_function=value_function,
        policy=policy,
    )


def _extract_agent_knowledge(agent, env) -> tuple[Dict[str, float] | None, Dict[str, int] | None]:
    """Extract value function and policy from agent."""
    import numpy as np
    
    value_function = {}
    policy = {}
    
    # Handle DP algorithms (Policy/Value Iteration)
    if hasattr(agent, "value_function") and isinstance(agent.value_function, dict):
        # Keys are tuples like (state_id,)
        for key, value in agent.value_function.items():
            # Convert tuple key to string representation for JSON
            # For GridWorld/FrozenLake, key is (state_id,)
            if isinstance(key, tuple) and len(key) == 1:
                str_key = str(key[0])
            elif isinstance(key, tuple) and len(key) > 1:
                # Handle multi-element tuples - extract first element (state_id)
                str_key = str(key[0])
            else:
                str_key = str(key)
            value_function[str_key] = float(value)
            
        # Ensure all states are included (fill missing states with 0.0)
        # This is important for visualization - frontend expects all state IDs
        if hasattr(env, 'state_space_size'):
            num_states = env.state_space_size
            for state_id in range(num_states):
                str_key = str(state_id)
                if str_key not in value_function:
                    value_function[str_key] = 0.0
            
        if hasattr(agent, "policy") and isinstance(agent.policy, dict):
            for key, action in agent.policy.items():
                if isinstance(key, tuple) and len(key) == 1:
                    str_key = str(key[0])
                elif isinstance(key, tuple) and len(key) > 1:
                    str_key = str(key[0])
                else:
                    str_key = str(key)
                policy[str_key] = int(action)
            
            # Ensure all states have a policy (fill missing with action 0)
            if hasattr(env, 'state_space_size'):
                num_states = env.state_space_size
                for state_id in range(num_states):
                    str_key = str(state_id)
                    if str_key not in policy:
                        policy[str_key] = 0
                
        return value_function, policy

    # Handle Q-Learning / SARSA / n-step TD / Monte Carlo
    if hasattr(agent, "q_values") and isinstance(agent.q_values, dict):
        for key, q_vals in agent.q_values.items():
            # Extract state ID from key
            str_key = None
            
            # Handle simple tuple keys (e.g. from model-based algos or simple envs)
            if len(key) == 1:
                str_key = str(key[0])
            
            # Handle complex sorted tuple keys from Monte Carlo/Q-Learning/SARSA
            # These algorithms sort state dict items, creating tuples like:
            # GridWorld: (('info', {...}), ('observation', 5), ('position', [2, 3]))
            # FrozenLake: (('agent_pos', [1,2]), ('info', {...}), ('observation', 5), ...)
            elif isinstance(key, tuple) and len(key) > 1:
                # Search for 'observation' key in the tuple
                # Each element is a tuple like ('observation', value) or ('state_index', value)
                for item in key:
                    if isinstance(item, tuple) and len(item) == 2:
                        item_key, item_value = item
                        if item_key == "observation" or item_key == "state_index":
                            str_key = str(item_value)
                            break
                
                # Fallback: if no observation found, try to extract from first numeric value
                if str_key is None:
                    for item in key:
                        if isinstance(item, tuple) and len(item) == 2:
                            _, item_value = item
                            if isinstance(item_value, (int, float)):
                                str_key = str(int(item_value))
                                break
                
                # Last resort: use string representation of key
                if str_key is None:
                    str_key = str(key)
            
            else:
                str_key = str(key)
            
            # V(s) = max_a Q(s,a)
            value_function[str_key] = float(np.max(q_vals))
            
            # Policy(s) = argmax_a Q(s,a)
            policy[str_key] = int(np.argmax(q_vals))
            
        return value_function, policy
        
    return None, None


def _create_agent(request: TrainingRequest, env: BaseEnv):
    """Create agent based on algorithm name and environment.

    Args:
        request: Training request with algorithm config.
        env: Environment to get action space size from.

    Returns:
        Agent instance.

    Raises:
        ValueError: If algorithm name is not recognized.
    """
    learning_rate = request.learning_rate or 0.1
    epsilon = request.epsilon or 0.1
    n_step = request.n_step or 3
    num_actions = env.action_space_size

    if request.algorithm_name == "q_learning":
        return QLearningAgent(
            num_actions=num_actions,
            learning_rate=learning_rate,
            discount_factor=request.discount_factor,
            epsilon=epsilon,
        )

    if request.algorithm_name == "sarsa":
        return SarsaAgent(
            num_actions=num_actions,
            learning_rate=learning_rate,
            discount_factor=request.discount_factor,
            epsilon=epsilon,
        )

    if request.algorithm_name == "monte_carlo":
        return MonteCarloAgent(
            num_actions=num_actions,
            discount_factor=request.discount_factor,
            epsilon=epsilon,
        )

    if request.algorithm_name == "n_step_td":
        return NStepTDAgent(
            num_actions=num_actions,
            n_step=n_step,
            learning_rate=learning_rate,
            discount_factor=request.discount_factor,
            epsilon=epsilon,
        )

    if request.algorithm_name == "td_prediction":
        return TDPrediction(
            learning_rate=learning_rate,
            discount_factor=request.discount_factor,
            num_actions=num_actions,
        )

    if request.algorithm_name == "policy_iteration":
        return PolicyIteration(
            discount_factor=request.discount_factor,
            num_actions=num_actions,
        )

    if request.algorithm_name == "value_iteration":
        return ValueIteration(
            discount_factor=request.discount_factor,
            num_actions=num_actions,
        )

    raise ValueError(f"Unknown algorithm: {request.algorithm_name}")


def _convert_history(history: List[dict]) -> List[EpisodeMetrics]:
    """Convert trainer history to EpisodeMetrics models.

    Args:
        history: List of episode metric dicts from trainer.

    Returns:
        List[EpisodeMetrics]: Pydantic models for API response.
    """
    return [
        EpisodeMetrics(
            episode=entry["episode"],
            total_reward=entry["total_reward"],
            length=entry["length"],
        )
        for entry in history
    ]
