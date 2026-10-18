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


@contextmanager
def capture_scene_graph(filename: str = PLAYGROUND_FILENAME):
    """Record what the scene tree needs to name the SDF graph the program builds.

    Two things the SDF objects do not carry themselves: the playground line
    each boolean was constructed on (so it can be paired with the boolean
    *element* the source map found there), and which placed SDF each
    ``Solid.box``/``sphere``/``cylinder`` returned (so the graph node can be
    paired with its construction node).  A sketch needs neither: its features
    already record the solid they generated.

    Yields:
        A dict with ``booleans``, a list of ``(boolean, line)`` pairs in
        construction order (``line`` None unless the program called the
        boolean's class itself), and ``primitives``, a list of
        ``(primitive, placed_sdf)`` pairs.  Both hold the objects themselves,
        so their ``id()`` stays valid while the dict is alive.
    """
    from cadjoint.construction.solid import ConstructionPrimitive
    from cadjoint.sdf.boolean import Difference, Intersection, Union, Xor

    graph: dict[str, list] = {"booleans": [], "primitives": []}
    initialisers = {cls: cls.__init__ for cls in (Union, Difference, Intersection, Xor)}
    original_sdf = ConstructionPrimitive.sdf

    def wrap(cls, original):
        def patched_init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            # Only a boolean the program *called* has a call in the source to
            # pair with: one built by ``a | b`` or inside the library is
            # constructed from a cadjoint frame, and gets no line.
            caller = sys._getframe(1)
            line = caller.f_lineno if caller.f_code.co_filename == filename else None
            graph["booleans"].append((self, line))

        cls.__init__ = patched_init

    def patched_sdf(self):
        placed = original_sdf(self)
        graph["primitives"].append((self, placed))
        return placed

    for cls, original in initialisers.items():
        wrap(cls, original)
    ConstructionPrimitive.sdf = patched_sdf
    try:
        yield graph
    finally:
        for cls, original in initialisers.items():
            cls.__init__ = original
        ConstructionPrimitive.sdf = original_sdf
