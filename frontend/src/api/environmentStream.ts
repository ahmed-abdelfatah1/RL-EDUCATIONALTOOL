/**
 * SSE client for live environment state streaming.
 */

import type { AnyEnvRenderState } from "../types/environment";
import { API_BASE_URL } from "./client";

export interface StateSnapshot {
  env_name: string;
  state: AnyEnvRenderState;
}

export type EnvStreamCallback = (snapshot: StateSnapshot) => void;

/**
 * Subscribe to live environment state updates via SSE.
 *
 * @param envName - Name of the environment to subscribe to.
 * @param onMessage - Callback invoked with each state update.
 * @returns Unsubscribe function to close the connection.
 */
export function subscribeToEnvStream(
  envName: string,
  onMessage: EnvStreamCallback
): () => void {
  const url = `${API_BASE_URL}/envs/${envName}/stream`;
  const eventSource = new EventSource(url);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as StateSnapshot;
      onMessage(data);
    } catch (err) {
      console.error("Failed to parse env stream message:", err);
    }
  };

  eventSource.onerror = (err) => {
    console.error("Env stream error:", err);
    // Don't close on error - SSE will automatically reconnect
  };

  eventSource.onopen = () => {
    console.log(`Connected to env stream: ${envName}`);
  };

  // Return unsubscribe function
  return () => {
    console.log(`Disconnecting from env stream: ${envName}`);
    eventSource.close();
  };
}

