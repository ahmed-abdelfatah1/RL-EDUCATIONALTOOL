"""SSE streaming endpoint for live environment state updates."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.services.stream_publisher import stream_publisher

router = APIRouter(prefix="/envs", tags=["streaming"])


@router.get("/{env_name}/stream")
async def env_stream(env_name: str) -> StreamingResponse:
    """Stream environment state updates via Server-Sent Events.

    Args:
        env_name: Name of the environment to stream.

    Returns:
        StreamingResponse: SSE stream of state updates.
    """

    async def event_generator():
        """Generate SSE events from the subscriber queue."""
        queue = await stream_publisher.subscribe(env_name)

        try:
            while True:
                try:
                    # Wait for state update with timeout to allow cleanup
                    state = await asyncio.wait_for(queue.get(), timeout=30.0)

                    payload = json.dumps({
                        "env_name": env_name,
                        "state": state,
                    })

                    # SSE format: "data: <json>\n\n"
                    yield f"data: {payload}\n\n"

                except asyncio.TimeoutError:
                    # Send keepalive comment to prevent connection timeout
                    yield ": keepalive\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            await stream_publisher.unsubscribe(env_name, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )

