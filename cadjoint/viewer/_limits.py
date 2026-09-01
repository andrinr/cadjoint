"""The size budget every layer of the playground API agrees on.

The HTTP layer caps request bodies against ``MAX_SOURCE_BYTES``; the patch,
scene, and worker endpoints each refuse an oversized program with the same
``OVERSIZED_SOURCE_ERROR`` before doing any work with it.  One home for the
number keeps those four refusals identical.
"""

from __future__ import annotations

MAX_SOURCE_BYTES = 100_000
OVERSIZED_SOURCE_ERROR = f"Source is larger than the {MAX_SOURCE_BYTES:,}-byte limit."


def exceeds_source_limit(source: str) -> bool:
    """True when a program is too large to accept, measured as UTF-8 bytes."""
    return len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES
