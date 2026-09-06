"""Resolving a :class:`~cadjoint.studies.selection.NodeSelection` on a lattice.

The flow grid has no mesh, so it has no nodes and no boundary surface — but
a study still has to say *where* the heat goes in and *where* a temperature
is held.  Rather than invent a second selection language, this module uses
the one scenes already speak (``Nodes.box``, ``Nodes.sphere``,
``Nodes.halfspace``, ``Nodes.cylinder`` and their ``&`` / ``|`` / ``~``
combinations), evaluated by the selection's own
:meth:`~cadjoint.studies.selection.NodeSelection.contains` — the same code
the mesh path runs, one boundary restriction short.

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
``Nodes.predicate`` would evaluate on cell centres perfectly well, but it is
not serializable, so a study carrying one cannot round-trip through the
viewer payload or a saved scene; refusing it here keeps the lattice studies
uniformly declarative.  Both raise with the alternative named.
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
        selection: A :class:`~cadjoint.studies.selection.NodeSelection`, or
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
    _refuse_unresolvable(selection.describe())
    return selection.contains(points)


def _refuse_unresolvable(description: dict[str, Any]) -> None:
    """Raise if any kind in the selection tree is one a lattice cannot read."""
    kind = description["kind"]
    if kind in _REFUSED:
        raise ValueError(_REFUSED[kind])
    if kind == "not":
        _refuse_unresolvable(description["operand"])
    elif kind in ("and", "or"):
        for operand in description["operands"]:
            _refuse_unresolvable(operand)
    elif kind not in REGION_KINDS:
        raise ValueError(
            f"Selection kind {kind!r} is not one a flow lattice resolves ({REGION_KINDS})."
        )
