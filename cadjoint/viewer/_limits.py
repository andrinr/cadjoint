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


#: The sampling lattice ``/api/export`` extracts on, in cells along the
#: object's longest axis.  The floor is where dual contouring stops finding a
#: surface at all; the ceiling is a memory budget — a 256-cell lattice is
#: 16.8 M samples and the largest grid a worker can extract inside its
#: timeout on a laptop.  The default matches the viewer's mesh overlay.
EXPORT_MIN_RESOLUTION = 8
EXPORT_MAX_RESOLUTION = 256
EXPORT_DEFAULT_RESOLUTION = 64
