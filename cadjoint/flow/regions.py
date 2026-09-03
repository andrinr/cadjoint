"""Resolving a :class:`~cadjoint.fem.selection.NodeSelection` on a lattice.

The flow grid has no mesh, so it has no nodes and no boundary surface —
but a study still has to say *where* the heat goes in and *where* a
temperature is held.  Rather than invent a second selection language, this
module reuses the one scenes already speak (``Nodes.box``, ``Nodes.sphere``,
``Nodes.halfspace``, ``Nodes.cylinder`` and their ``&`` / ``|`` / ``~``
combinations) by interpreting the JSON payload
:meth:`~cadjoint.fem.selection.NodeSelection.describe` emits.  That payload
is the selection language's public, documented, round-trippable form, so
this reading cannot drift away from the mesh one without the description
itself changing.

**One semantic difference, and it is deliberate.**  On a mesh a selection is
always cut down to *boundary* nodes: a boundary condition acts on a surface.
On the lattice a selection is *volumetric* — every cell whose centre
satisfies the geometric criterion, interior cells included.  That is the
only reading that makes sense for the two things a flow study selects with
one: a heated region inside the solid, and a block of cells held at a
temperature.  ``~selection`` therefore means "every other cell", not "the
rest of the surface".

Two selection kinds are refused rather than approximated.  ``Nodes.side``
names the extreme plane *of a mesh's boundary*, which a lattice filled by an
SDF does not have (the extremes of the lattice are the duct, not the part).
``Nodes.predicate`` is not serializable, so its callable never reaches the
description this module reads.  Both raise with the alternative named.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["REGION_KINDS", "region_mask"]

#: The selection kinds a lattice region understands.
REGION_KINDS = ("box", "sphere", "halfspace", "cylinder", "and", "or", "not")

_REFUSED = {
    "side": (
        "Nodes.side names the extreme boundary plane of a mesh, which a flow "
        "lattice does not have — its extremes are the duct walls, not the part. "
        "Use Nodes.halfspace or Nodes.box against world coordinates instead."
    ),
    "predicate": (
        "Nodes.predicate is not serializable, so a flow study cannot read it. "
        "Use Nodes.box/sphere/halfspace/cylinder, combined with & | ~."
    ),
}


def region_mask(selection: Any, centers: np.ndarray) -> np.ndarray:
    """Boolean mask over cell centres for a node selection, volumetrically.

    Args:
        selection: A :class:`~cadjoint.fem.selection.NodeSelection`, or
            ``None`` for "every cell".
        centers: ``(..., 3)`` cell-centre world coordinates, as
            :meth:`~cadjoint.flow.FlowGrid.centers` returns.

    Returns:
        A boolean array shaped like ``centers`` without its last axis.

    Raises:
        ValueError: If the selection uses a kind a lattice cannot resolve
            (``Nodes.side``, ``Nodes.predicate``).
    """
    points = np.asarray(centers, dtype=np.float64)
    if selection is None:
        return np.ones(points.shape[:-1], dtype=bool)
    return _evaluate(selection.describe(), points)


def _evaluate(description: dict[str, Any], points: np.ndarray) -> np.ndarray:
    """Recursively evaluate a ``describe()`` payload on ``(..., 3)`` points."""
    kind = description["kind"]
    if kind in _REFUSED:
        raise ValueError(_REFUSED[kind])
    if kind == "not":
        return ~_evaluate(description["operand"], points)
    if kind in ("and", "or"):
        left, right = (_evaluate(operand, points) for operand in description["operands"])
        return left & right if kind == "and" else left | right
    if kind == "box":
        low = np.asarray(description["min_corner"])
        high = np.asarray(description["max_corner"])
        return np.all((points >= low) & (points <= high), axis=-1)
    if kind == "sphere":
        offsets = points - np.asarray(description["center"])
        return np.sum(offsets * offsets, axis=-1) <= float(description["radius"]) ** 2
    if kind == "halfspace":
        offsets = points - np.asarray(description["point"])
        return offsets @ np.asarray(description["normal"]) >= 0.0
    if kind == "cylinder":
        axis = np.asarray(description["axis"], dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        offsets = points - np.asarray(description["center"])
        axial = offsets @ axis
        radial_sq = np.sum(offsets * offsets, axis=-1) - axial**2
        inside = (radial_sq <= float(description["radius"]) ** 2) & (
            radial_sq >= float(description["inner"]) ** 2
        )
        half_length = description["half_length"]
        if half_length is not None:
            inside &= np.abs(axial) <= float(half_length)
        return inside
    raise ValueError(
        f"Selection kind {kind!r} is not one a flow lattice resolves ({REGION_KINDS})."
    )
