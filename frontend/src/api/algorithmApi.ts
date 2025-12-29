/**
 * Algorithm API - typed wrapper around /algorithms endpoints.
 */

import { get } from "./client";

export interface HyperparameterInfo {
  name: string;
  display_name: string;
  type: "float" | "int";
  default: number;
  min: number;
  max: number;
  step: number;
}

export interface AlgorithmInfo {
  name: string;
  display_name: string;
  description: string;
  supports_envs: string[];
  hyperparameters: HyperparameterInfo[];
}

export interface AlgorithmListResponse {
  algorithms: AlgorithmInfo[];
}

export async function fetchAlgorithms(): Promise<AlgorithmInfo[]> {
  const data = await get<AlgorithmListResponse>("/algorithms");
  return data.algorithms;
}

