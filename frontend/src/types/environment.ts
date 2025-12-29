/**
 * Environment render state types - matches backend render_state() outputs exactly.
 */

export interface GridWorldRenderState {
  type: "gridworld";
  grid: number[][];
  agent_pos: [number, number];
  goal_pos: [number, number];
  agent_direction: number;           // 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
  movement_trail: [number, number][]; // Recent positions
  last_reward: number;                // For visual feedback
  last_action: number | null;         // Action taken
  grid_size: number;
  done: boolean;
}

export interface CartPoleRenderState {
  type: "cartpole";
  cart_position: number;
  cart_velocity: number;
  pole_angle: number;
  pole_angular_velocity: number;
  action_taken: number | null;  // 0=LEFT, 1=RIGHT
  done: boolean;
  reward: number;
}

export interface MountainCarRenderState {
  type: "mountaincar";
  position: number;
  velocity: number;
  action: number | null;
  reward: number;
  done: boolean;
  goal_position: number;
  position_bounds: [number, number];
}

export interface FrozenLakeRenderState {
  type: "frozenlake";
  size: number;
  agent_pos: [number, number];
  tiles: string[][];
  movement_trail: [number, number][];  // Recent positions
  last_action: number | null;          // 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
  reward: number;
  done: boolean;
}

export interface BreakoutRenderState {
  type: "breakout";
  score: number;
  lives: number;
  step: number;
  done: boolean;
  last_reward: number;
}

export interface Gym4RealRenderState {
  type: "gym4real";
  position: [number, number];
  goal: [number, number];
  movement_trail: [number, number][];  // Recent positions
  last_action: number | null;          // 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
  step: number;
  reward: number;
  done: boolean;
  max_steps: number;
}

export type AnyEnvRenderState =
  | GridWorldRenderState
  | CartPoleRenderState
  | MountainCarRenderState
  | FrozenLakeRenderState
  | BreakoutRenderState
  | Gym4RealRenderState;
