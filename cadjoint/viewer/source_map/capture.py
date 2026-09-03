"""Record which source line built each construction object.

Half of the source map is recovered by parsing (see the ``locate_*`` modules);
this is the other half, collected while the user's program actually runs.  The
construction classes' initialisers are wrapped for the duration of the block
so every object — including ones passed straight into ``extrude()`` or
``Union()`` and never bound to a variable — is paired with the playground line
that created it.

Only runtime instrumentation belongs here.  Anything that reads the program
text goes in a locator module; anything that turns captured objects into
viewer JSON goes in :mod:`cadjoint.viewer.source_map.payload`.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

PLAYGROUND_FILENAME = "<cadjoint-playground>"


def _caller_line(filename: str) -> int | None:
    """Line number of the nearest frame executing *filename*."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename == filename:
            return frame.f_lineno
        frame = frame.f_back
    return None


@contextmanager
def capture_profiles(filename: str = PLAYGROUND_FILENAME):
    """Record every construction object built inside the block, with its line.

    Wraps the construction classes' initialisers for the duration, so objects
    are captured wherever they are built — including ones passed straight into
    ``extrude()`` or ``Union()`` that never get bound to a variable.

    Yields:
        A list of ``(object, line)`` pairs in construction order. ``line`` is
        None when construction did not originate from *filename*.
    """
    from cadjoint.construction.sketch import PolygonProfile
    from cadjoint.construction.solid import ConstructionPrimitive

    captured: list[tuple[object, int | None]] = []
    originals = {cls: cls.__init__ for cls in (PolygonProfile, ConstructionPrimitive)}

    def wrap(cls, original):
        def patched_init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            captured.append((self, _caller_line(filename)))

        cls.__init__ = patched_init

    for cls, original in originals.items():
        wrap(cls, original)
    try:
        yield captured
    finally:
        for cls, original in originals.items():
            cls.__init__ = original
