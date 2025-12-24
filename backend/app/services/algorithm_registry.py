"""Algorithm registry - metadata for available RL algorithms.

Provides hard-coded registry of algorithms and hyperparameters for UI.
"""

from __future__ import annotations

from typing import Dict, List

from app.schemas.algorithm_schemas import AlgorithmInfo, HyperparameterInfo


LEARNING_RATE = HyperparameterInfo(
    name="learning_rate",
    display_name="Learning Rate (α)",
    type="float",
    default=0.1,
    min=0.0,
    max=1.0,
    step=0.01,
)

DISCOUNT_FACTOR = HyperparameterInfo(
    name="discount_factor",
    display_name="Discount Factor (γ)",
    type="float",
    default=0.99,
    min=0.0,
    max=1.0,
    step=0.01,
)

EPSILON = HyperparameterInfo(
    name="epsilon",
    display_name="Exploration (ε)",
    type="float",
    default=0.1,
    min=0.0,
    max=1.0,
    step=0.01,
)

N_STEP = HyperparameterInfo(
    name="n_step",
    display_name="n-step",
    type="int",
    default=3,
    min=1,
    max=10,
    step=1,
)

# Environments with known transition dynamics (model-based DP works)
# GridWorld, FrozenLake, and Gym4Real all have deterministic/known transitions
MODEL_BASED_ENVS = ["gridworld", "frozenlake", "gym4real"]

# Environments where tabular methods work (discrete or discretized state space)
# CartPole and MountainCar use discretization, Breakout requires DQN (excluded)
TABULAR_ENVS = ["gridworld", "frozenlake", "cartpole", "mountaincar", "gym4real"]


ALGORITHM_METADATA: Dict[str, AlgorithmInfo] = {
    "q_learning": AlgorithmInfo(
        name="q_learning",
        display_name="Q-Learning",
        description="Off-policy TD control using greedy action selection from Q-values.",
        supports_envs=TABULAR_ENVS,
        hyperparameters=[LEARNING_RATE, DISCOUNT_FACTOR, EPSILON],
    ),
    "sarsa": AlgorithmInfo(
        name="sarsa",
        display_name="SARSA",
        description="On-policy TD control using the action actually taken in next state.",
        supports_envs=TABULAR_ENVS,
        hyperparameters=[LEARNING_RATE, DISCOUNT_FACTOR, EPSILON],
    ),
    "n_step_td": AlgorithmInfo(
        name="n_step_td",
        display_name="n-step TD",
        description="On-policy TD control using n-step returns for bootstrapping.",
        supports_envs=TABULAR_ENVS,
        hyperparameters=[LEARNING_RATE, DISCOUNT_FACTOR, EPSILON, N_STEP],
    ),
    "monte_carlo": AlgorithmInfo(
        name="monte_carlo",
        display_name="Monte Carlo",
        description="First-visit MC control learning from complete episodes.",
        supports_envs=TABULAR_ENVS,
        hyperparameters=[DISCOUNT_FACTOR, EPSILON],
    ),
    "td_prediction": AlgorithmInfo(
        name="td_prediction",
        display_name="TD Prediction",
        description="TD(0) state-value prediction - evaluates a random policy to demonstrate value learning.",
        supports_envs=["gridworld", "frozenlake"],  # Only small grids for educational value
        hyperparameters=[LEARNING_RATE, DISCOUNT_FACTOR],
    ),

    "policy_iteration": AlgorithmInfo(
        name="policy_iteration",
        display_name="Policy Iteration",
        description="Dynamic programming method alternating policy evaluation and improvement.",
        supports_envs=MODEL_BASED_ENVS,
        hyperparameters=[
            DISCOUNT_FACTOR,
            HyperparameterInfo(
                name="theta",
                display_name="Convergence Threshold (θ)",
                type="float",
                default=0.0001,
                min=0.000001,
                max=0.01,
                step=0.0001,
            ),
        ],
    ),
    "value_iteration": AlgorithmInfo(
        name="value_iteration",
        display_name="Value Iteration",
        description="Dynamic programming method computing optimal value function directly.",
        supports_envs=MODEL_BASED_ENVS,
        hyperparameters=[
            DISCOUNT_FACTOR,
            HyperparameterInfo(
                name="theta",
                display_name="Convergence Threshold (θ)",
                type="float",
                default=0.0001,
                min=0.000001,
                max=0.01,
                step=0.0001,
            ),
        ],
    ),
}


def list_algorithms() -> List[AlgorithmInfo]:
    """Get list of all available algorithms.

    Returns:
        List[AlgorithmInfo]: Metadata for all algorithms.
    """
    return list(ALGORITHM_METADATA.values())


def get_algorithm(name: str) -> AlgorithmInfo:
    """Get metadata for a specific algorithm.

    Args:
        name: Algorithm name.

    Returns:
        AlgorithmInfo: Algorithm metadata.

    Raises:
        KeyError: If algorithm not found.
    """
    return ALGORITHM_METADATA[name]


def get_algorithms_for_env(env_name: str) -> List[AlgorithmInfo]:
    """Get algorithms that support a specific environment.

    Args:
        env_name: Environment name.

    Returns:
        List[AlgorithmInfo]: Compatible algorithms.
    """
    return [
        algo for algo in ALGORITHM_METADATA.values()
        if env_name in algo.supports_envs
    ]

