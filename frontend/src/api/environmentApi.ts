/**
 * Environment API - typed wrapper around /envs endpoints.
 */

import { get } from "./client";

export interface EnvironmentInfo {
  name: string;
  display_name: string;
  supports_discrete: boolean;
  supports_continuous: boolean;
  description: string;
}

export interface EnvironmentListResponse {
  environments: EnvironmentInfo[];
}

export async function fetchEnvironments(): Promise<EnvironmentInfo[]> {
  const data = await get<EnvironmentListResponse>("/envs");
  return data.environments;
}

