"""Record which source line built each construction object.

Half of the source map is recovered by parsing (see the ``locate_*`` modules);
this is the other half, collected while the user's program actually runs.  The
construction classes report their own construction through core's capture
hook (:mod:`cadjoint.provenance`), so every object — including ones passed
straight into ``extrude()`` or ``Union()`` and never bound to a variable — is
paired with the playground line that created it.

Only runtime instrumentation belongs here.  Anything that reads the program
text goes in a locator module; anything that turns captured objects into
viewer JSON goes in :mod:`cadjoint.viewer.source_map.payload`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from cadjoint.provenance import Site, capture_constructions

PLAYGROUND_FILENAME = "<cadjoint-playground>"


@contextmanager
def capture_profiles(
    filename: str = PLAYGROUND_FILENAME,
) -> Iterator[list[tuple[object, int | None]]]:
    """Record every construction object built inside the block, with its line.

    Objects are captured wherever they are built — including ones passed
    straight into ``extrude()`` or ``Union()`` that never get bound to a
    variable.

    Yields:
        A list of ``(object, line)`` pairs in construction order. ``line`` is
        None when construction did not originate from *filename*.
    """
    from cadjoint.construction.sketch import PolygonProfile
    from cadjoint.construction.solid import ConstructionPrimitive

    captured: list[tuple[object, int | None]] = []

    def constructed(obj: Any, site: Site) -> None:
        if isinstance(obj, (PolygonProfile, ConstructionPrimitive)):
            captured.append((obj, site.line))

    with capture_constructions(filename, constructed=constructed):
        yield captured


@contextmanager
def capture_scene_graph(filename: str = PLAYGROUND_FILENAME) -> Iterator[dict[str, list]]:
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
    from cadjoint.sdf.boolean import Difference, Intersection, Union, Xor

    graph: dict[str, list] = {"booleans": [], "primitives": []}

    def constructed(obj: Any, site: Site) -> None:
        if isinstance(obj, (Union, Difference, Intersection, Xor)):
            # Only a boolean the program *called* has a call in the source to
            # pair with: one built by ``a | b`` or inside the library is
            # constructed from a cadjoint frame, and gets no line.
            graph["booleans"].append((obj, site.line if site.direct else None))

    def placed(primitive: Any, sdf: Any) -> None:
        graph["primitives"].append((primitive, sdf))

    with capture_constructions(filename, constructed=constructed, placed=placed):
        yield graph
