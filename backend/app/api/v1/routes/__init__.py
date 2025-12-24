"""API v1 routes package.

Re-exports router modules for clean imports.
"""

from . import algorithms, environments, stream, training

__all__ = ["algorithms", "environments", "stream", "training"]

