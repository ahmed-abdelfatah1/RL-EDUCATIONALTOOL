/**
 * React hook for subscribing to live environment state updates.
 */

import { useEffect, useCallback, useRef } from "react";
import type { AnyEnvRenderState } from "../types/environment";
import { subscribeToEnvStream } from "../api/environmentStream";

interface UseEnvironmentStreamOptions {
  /** Environment name to subscribe to */
  envName: string | null;
  /** Whether streaming is enabled */
  enabled: boolean;
  /** Callback when state is received */
  onState: (state: AnyEnvRenderState) => void;
}

/**
 * Hook to subscribe to live environment state updates via SSE.
 *
 * @param options - Configuration options for the stream.
 */
export function useEnvironmentStream({
  envName,
  enabled,
  onState,
}: UseEnvironmentStreamOptions): void {
  // Use ref to avoid recreating subscription on every onState change
  const onStateRef = useRef(onState);
  onStateRef.current = onState;

  const stableOnState = useCallback((state: AnyEnvRenderState) => {
    onStateRef.current(state);
  }, []);

  useEffect(() => {
    if (!envName || !enabled) {
      return;
    }

    const unsubscribe = subscribeToEnvStream(envName, (snapshot) => {
      stableOnState(snapshot.state);
    });

    return () => {
      unsubscribe();
    };
  }, [envName, enabled, stableOnState]);
}

