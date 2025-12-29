/**
 * Training API - typed wrapper around /train endpoints.
 */

import { post } from "./client";

export interface TrainingRequest {
  env_name: string;
  algorithm_name: string;
  num_episodes: number;
  max_steps_per_episode: number;
  discount_factor: number;
  learning_rate?: number;
  epsilon?: number;
  n_step?: number;
}

export interface EpisodeMetrics {
  episode: number;
  total_reward: number;
  length: number;
}

export interface TrainingRunResponse {
  env_name: string;
  algorithm_name: string;
  episodes: EpisodeMetrics[];
  value_function?: Record<string, number>;
  policy?: Record<string, number>;
}

export async function runTraining(
  body: TrainingRequest
): Promise<TrainingRunResponse> {
  return post<TrainingRunResponse, TrainingRequest>("/train/run", body);
}
