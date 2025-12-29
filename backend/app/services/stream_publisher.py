"""Stream publisher for real-time environment state updates.

Manages SSE subscribers and publishes environment state during training.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Any


class StreamPublisher:
    """Manages SSE subscribers for environment state streaming."""

    def __init__(self) -> None:
        """Initialize the publisher with empty subscriber map."""
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, env_name: str) -> asyncio.Queue:
        """Subscribe to state updates for an environment.

        Args:
            env_name: Name of the environment to subscribe to.

        Returns:
            asyncio.Queue: Queue that will receive state updates.
        """
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            if env_name not in self._subscribers:
                self._subscribers[env_name] = []
            self._subscribers[env_name].append(queue)
        return queue

    async def unsubscribe(self, env_name: str, queue: asyncio.Queue) -> None:
        """Unsubscribe from state updates.

        Args:
            env_name: Name of the environment.
            queue: The queue to remove.
        """
        async with self._lock:
            if env_name in self._subscribers:
                try:
                    self._subscribers[env_name].remove(queue)
                except ValueError:
                    pass
                if not self._subscribers[env_name]:
                    del self._subscribers[env_name]

    def publish(self, env_name: str, state: Dict[str, Any]) -> None:
        """Publish state to all subscribers of an environment.

        This is synchronous so it can be called from the training loop.

        Args:
            env_name: Name of the environment.
            state: The render state to publish.
        """
        if env_name not in self._subscribers:
            return

        for queue in self._subscribers[env_name]:
            try:
                queue.put_nowait(state)
            except asyncio.QueueFull:
                # Drop message if queue is full (subscriber is slow)
                pass

    def has_subscribers(self, env_name: str) -> bool:
        """Check if there are any subscribers for an environment.

        Args:
            env_name: Name of the environment.

        Returns:
            bool: True if there are subscribers.
        """
        return env_name in self._subscribers and len(self._subscribers[env_name]) > 0


# Global publisher instance
stream_publisher = StreamPublisher()


def publish_env_state(env_name: str, state: Dict[str, Any]) -> None:
    """Convenience function to publish environment state.

    Args:
        env_name: Name of the environment.
        state: The render state dict from env.render_state().
    """
    stream_publisher.publish(env_name, state)

