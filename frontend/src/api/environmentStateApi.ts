/**
 * Environment state API - calls backend for env reset and step.
 */

import { post } from "./client";
import type { AnyEnvRenderState } from "../types/environment";

export interface StateSnapshot {
  env_name: string;
  state: AnyEnvRenderState;
}

export async function resetEnvironment(envName: string): Promise<StateSnapshot> {
  return post<StateSnapshot, unknown>(`/envs/${envName}/reset`, {});
}

export async function stepEnvironment(
  envName: string,
  action: number
): Promise<StateSnapshot> {
  return post<StateSnapshot, unknown>(`/envs/${envName}/step/${action}`, {});
}

