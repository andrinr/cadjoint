"""Fixed-size ladders that keep the tet solves' array shapes stable.

What belongs here: the rung ladder itself and the pure-array padding that
puts a tet mesh on it — :func:`rung`, :func:`pad_tets`, :func:`pad_indices`.

Why it exists: every design edit moves the dual-contoured surface, so TetGen
returns a different node and element count, so every array downstream of the
mesh gets a shape it has never had.  JAX keys both its in-process caches and
its persistent compilation cache on the lowered program, and a new shape
lowers differently, so a novel design recompiles the whole assembly from
scratch and can never reuse the result (``research/performance.md`` §4.1,
§16.6).  Padding the mesh up to the next rung of a geometric ladder makes a
*run* of similar designs share one set of shapes, so the second design and
every one after it reads the first one's programs back.

**How the padding stays exact.**  The padded region is a *ghost body*: a
handful of extra nodes carrying extra copies of element 0 — the same
positions, in the same order, so every ghost element is as well-formed as
the real one it copies — plus enough repeats of a ghost element to reach the
cell rung.  Ghost cells name ghost nodes only, so the ghost body shares no
degree of freedom with the real mesh, and the caller pins every ghost node
with a Dirichlet condition.  A pinned, disconnected block contributes an
identity row and nothing else: the real solution is the one the unpadded
mesh gives, up to the rounding the larger factorization introduces
(measured at 4e-15 to 2e-14 on a unit-scale temperature field, i.e. a few
units in the last place of float64: ``research/performance.md`` §18.5).  Nothing is ever padded with zeros — a zero node
position is a point outside the mesh and a zero index is node 0 — which is
the same rule ``diff_brep.project.run_chunked`` follows on its batches.

What does *not* belong here: anything that knows what a jax-fem ``Problem``
is.  The surgery that pads a problem's *face selection* needs its internals
and lives beside the pruning that already does that
(:func:`cadjoint.fem.jaxfem._pad_surface_faces`); this module is plain NumPy
and imports no solver.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "PaddedTets",
    "enabled",
    "pad_cell_field",
    "pad_indices",
    "pad_tets",
    "rung",
]

#: The smallest rung.  Below it a solve is cheap enough that the padding
#: would cost more arithmetic than the compile it saves; the starter's own
#: boundary-condition node sets are a few hundred entries, so this is also
#: the scale at which the index ladder starts doing work.
_BASE = 64

#: The ladder's ratio, and the one knob here.  It reads like a trade of
#: padded arithmetic against compiles, and the measurement says it is not:
#: what it really sets is **how often a run of designs crosses a rung**,
#: and a crossing is a partial recompile.  A finer ladder wastes less
#: arithmetic and splits a design session across more rungs; a coarser one
#: does the reverse.
#:
#: Measured on the two bar cases of ``research/performance.md`` §18.5 —
#: one design per process, own empty cache per arm, arms interleaved.
#: *Misses* is the XLA compiles a design after the first still pays:
#:
#: ======== ================= ================== ==================
#: growth   later designs     misses (per arm)   padding (cells)
#: ======== ================= ================== ==================
#: 1.25      1.29-2.30 s       20, 39, 73         13-20 %
#: 1.50      1.19-1.33 s       20, 20, 20         21-27 %
#: 2.00      1.19-2.92 s       20, 105, 20        5-110 %
#: ======== ================= ================== ==================
#:
#: 1.5 is the only one of the three that put every design of a run on one
#: rung on *both* mesh sizes.  1.25 splits them — a 600-node design and a
#: 616-node design land on different rungs — and 2.0 both splits them and
#: pays 100 % padding for it.  The padding itself costs nothing worth
#: measuring: a design whose shapes are already cached solves in
#: 0.92-0.94 s at all three ratios.
_GROWTH = 1.5

#: Set to ``0``/``off``/``false`` to solve on the raw mesh.  The escape
#: hatch exists because the padding is exact only to rounding, so a test
#: that pins a solve bit for bit against a stored reference needs a way to
#: turn it off.
_ENV = "CADJOINT_FEM_RUNGS"


def enabled() -> bool:
    """Whether the tet solves pad their meshes onto the ladder.

    Returns:
        ``True`` unless ``CADJOINT_FEM_RUNGS`` names a falsehood.
    """
    return os.environ.get(_ENV, "1").strip().lower() not in ("0", "off", "false", "no")


def rung(count: int, *, base: int = _BASE, growth: float = _GROWTH) -> int:
    """The smallest ladder rung that holds ``count`` rows.

    The ladder is ``base * growth**k`` rounded up, so it is unbounded: a
    mesh of any size lands on a rung, and the padding is never more than
    ``growth - 1`` of the true count.

    Args:
        count: The true number of rows.
        base: The smallest rung.
        growth: The ratio between consecutive rungs.

    Returns:
        The rung, or ``0`` for an empty input (an empty array is already a
        stable shape and padding it would invent rows).

    Contract:
        Preconditions: ``count >= 0``; ``growth > 1``; ``base >= 1``.
        Postconditions: ``0 <= count <= rung(count) < max(count * growth,
            base + 1)``, and the result is one of ``base * growth**k``
            rounded up, so equal counts give equal rungs
            (``tests/fem/test_rungs.py::test_the_ladder_covers_and_bounds``).
        Frozen: everything — pure arithmetic on Python ints.
        Differentiable: none.
        Refuses: a negative ``count`` with ``ValueError``.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative; got {count}.")
    if count == 0:
        return 0
    # Climbed rather than solved for: ``ceil(log(count / base) / log(growth))``
    # reads one step too many at a count that *is* a rung (605 on this
    # ladder), and the overshoot is a shape nobody needed.  The loop runs
    # a few dozen times at the largest mesh anyone meshes.
    size = base
    while size < count:
        size = max(size + 1, math.ceil(size * growth))
    return size


@dataclass(frozen=True)
class PaddedTets:
    """A tet mesh grown to the ladder, with the ghost body's handles.

    Attributes:
        points: Padded node positions, ``(node_rung, 3)`` — traced when the
            caller's ``points`` were.
        base_points: The same, concrete, for problem construction.
        cells: Padded connectivity, ``(cell_rung, K)``.
        node_count: The true node count, i.e. where the ghost body starts
            and where the caller cuts the solution back.
        cell_count: The true cell count.
        ghost_nodes: Every padded node index, ``(node_rung - node_count,)``.
            The caller must pin all of them with a Dirichlet condition;
            the mesh is singular otherwise.
        ghost_cell: A padded cell index, for padding a face selection with
            faces that reach no real degree of freedom.
    """

    points: Any
    base_points: np.ndarray
    cells: np.ndarray
    node_count: int
    cell_count: int
    ghost_nodes: np.ndarray
    ghost_cell: int


def _ghost_layout(node_count: int, arity: int, ghosts: int) -> tuple[np.ndarray, np.ndarray]:
    """Ghost cells covering every ghost node, each a positional copy of element 0.

    An isolated node is not merely useless, it is fatal: it owns a row of
    the tangent that no element writes to, so the row has no diagonal and
    PETSc refuses to eliminate it ("Matrix is missing diagonal entry in the
    zeroed row").  Every ghost node therefore has to appear in some ghost
    cell.  Whole groups of ``arity`` take element 0's nodes in order; the
    ``ghosts % arity`` left over each get a cell of their own that replaces
    corner 0 of the first ghost cell, and since that leftover node carries
    corner 0's position too, the cell is element 0 again — same positions,
    same order, so the same volume and the same orientation.

    Args:
        node_count: The true node count; ghost indices start here.
        arity: Nodes per cell — 4 for TET4, 10 for TET10.
        ghosts: How many ghost nodes to cover; at least ``arity``.

    Returns:
        ``(rows, slots)`` — the ghost connectivity ``(g, arity)``, and for
            each ghost node the local slot of element 0 whose position it
            takes, ``(ghosts,)``.
    """
    groups, spare = divmod(ghosts, arity)
    rows = [np.arange(node_count + g * arity, node_count + (g + 1) * arity) for g in range(groups)]
    tail = np.arange(node_count + 1, node_count + arity)
    rows += [np.concatenate([[node_count + groups * arity + j], tail]) for j in range(spare)]
    slots = np.concatenate([np.tile(np.arange(arity), groups), np.zeros(spare, dtype=int)])
    return np.asarray(rows, dtype=np.int64), slots


def pad_tets(points: Any, base_points: np.ndarray, cells: np.ndarray) -> PaddedTets | None:
    """Grow a tet mesh to the next rung with a pinned ghost body.

    Args:
        points: Node positions, ``(N, 3)`` — a NumPy array or a JAX tracer.
        base_points: The same positions, concrete, ``(N, 3)``.
        cells: Connectivity, ``(C, K)`` with ``K`` 4 or 10.

    Returns:
        The padded mesh, or ``None`` when the ladder is switched off, in
        which case the caller solves the mesh it was given.  There is no
        "already on a rung" case: the ghost body needs a whole element's
        worth of nodes, so the node rung is always the one *above* the
        true count.

    Contract:
        Preconditions: ``base_points`` shaped ``(N, 3)`` with ``N >= K``;
            ``cells`` shaped ``(C, K)``, ``C >= 1``, every entry a valid
            index into the points.
        Postconditions: the first ``N`` rows of ``points`` and the first
            ``C`` rows of ``cells`` are the caller's, unchanged; every
            index in the padded rows is ``>= N``; every ghost node appears
            in some padded cell; every padded cell has ``K`` distinct
            indices and the positions of element 0 in element 0's order,
            so its volume equals element 0's exactly
            (``tests/fem/test_rungs.py``).
        Frozen: everything but ``points``, which is gathered from and so
            stays traced.
        Differentiable: through ``points`` — the ghost rows are a gather of
            real rows, so the padding adds no non-differentiable step.
        Refuses: a mesh with no cells, or fewer nodes than one cell names,
            with ``ValueError``.
    """
    if not enabled():
        return None
    base_points = np.asarray(base_points, dtype=np.float64)
    cells = np.asarray(cells)
    node_count, cell_count = int(base_points.shape[0]), int(cells.shape[0])
    if cell_count < 1:
        raise ValueError("Padding needs at least one cell to copy.")
    arity = int(cells.shape[1])
    if node_count < arity:
        raise ValueError(f"A {arity}-node element needs at least {arity} nodes; got {node_count}.")

    node_rung = rung(node_count + arity)
    ghosts = node_rung - node_count
    rows, slots = _ghost_layout(node_count, arity, ghosts)
    cell_rung = rung(cell_count + len(rows))

    template = cells[0]
    ghost_points = base_points[template][slots]
    padded_base = np.concatenate([base_points, ghost_points])
    if isinstance(points, np.ndarray):
        padded_points: Any = np.concatenate([points.astype(np.float64), ghost_points])
    else:
        # Traced, or a device array: gather the ghost rows from the caller's
        # own points so the padding is one more differentiable read of them.
        import jax.numpy as jnp

        moved = jnp.asarray(points)
        padded_points = jnp.concatenate([moved, moved[template][slots]])

    fill = np.repeat(rows[:1], cell_rung - cell_count - len(rows), axis=0)
    padded_cells = np.concatenate([cells, rows.astype(cells.dtype), fill.astype(cells.dtype)])
    return PaddedTets(
        points=padded_points,
        base_points=padded_base,
        cells=padded_cells,
        node_count=node_count,
        cell_count=cell_count,
        ghost_nodes=np.arange(node_count, node_rung, dtype=np.int32),
        ghost_cell=cell_rung - 1,
    )


def pad_cell_field(value: Any, padded: PaddedTets, rank: int) -> Any:
    """Grow a per-element field to the cell rung by repeating element 0's row.

    A heterogeneous solve carries its material as one value per element —
    ``sample_cell_property`` over the scene's material tree — and jax-fem
    reads it as an ``internal_vars`` entry shaped by the cell count, so a
    padded mesh with an unpadded field is a shape error, not a slow solve.
    The ghost rows take element 0's value because it is a real one: their
    contribution is eliminated either way, and a made-up zero would be a
    zero modulus or a zero conductivity, which are outside the domain the
    tensor map is written for.

    Args:
        value: The field, a scalar, or ``None``; anything whose leading
            axis is not the true cell count is returned unchanged, which
            is how scalars and a single ``(3,)`` body force pass through.
        padded: The mesh this field goes with.
        rank: The dimensionality a *per-element* field has here — 1 for a
            scalar per element, 2 for a vector per element.  It is what
            separates a ``(3,)`` body force from a ``(C, 3)`` one.

    Returns:
        The field with ``len(padded.cells)`` rows, or the input unchanged.

    Contract:
        Preconditions: ``padded`` from :func:`pad_tets`.
        Postconditions: the first ``padded.cell_count`` rows are the
            caller's; the rest equal row 0
            (``tests/fem/test_rungs.py::TestPerElementFields``).
        Frozen: everything but a traced ``value``, which stays traced.
        Differentiable: through ``value``.
        Refuses: nothing — an unrecognised shape is passed through for the
            solver's own validation to reject.
    """
    shape = getattr(value, "shape", None)
    if value is None or shape is None or len(shape) != rank or shape[0] != padded.cell_count:
        return value
    fill = len(padded.cells) - padded.cell_count
    if fill <= 0:
        return value
    if isinstance(value, np.ndarray):
        return np.concatenate([value, np.repeat(value[:1], fill, axis=0)])
    import jax.numpy as jnp

    return jnp.concatenate([value, jnp.repeat(value[:1], fill, axis=0)])


def pad_indices(indices: np.ndarray, filler: int | None = None) -> np.ndarray:
    """Grow an index set to the next rung by repeating one of its own entries.

    Repetition is the right filler because every consumer of these sets in
    the tet path is idempotent under it: membership (``jnp.isin``, which
    decides which faces a patch carries) does not count, and a Dirichlet
    ``.set`` writes the same value twice.  The one operation that *does*
    accumulate — jax-fem's ``res.at[nodes].add(-values)`` — is why
    :func:`pad_tets`' ghost nodes are the filler of choice for a Dirichlet
    patch: a doubled ghost row moves a degree of freedom that is pinned,
    disconnected and cut off the answer, whatever value the patch
    prescribes.

    Args:
        indices: The set to pad, ``(M,)``.
        filler: The index to repeat; the set's first entry by default.

    Returns:
        The padded set as ``int32``, ``(rung(M),)``.

    Contract:
        Preconditions: ``indices`` one-dimensional; ``filler`` present in
            the padded array's intended domain (an index the consumer may
            legally see twice).
        Postconditions: the first ``M`` entries are the caller's; the rest
            are ``filler``; the set of distinct values is unchanged when
            ``filler`` comes from ``indices``
            (``tests/fem/test_rungs.py::test_index_padding_keeps_the_set``).
        Frozen: everything.
        Differentiable: none — indices are integers.
        Refuses: an empty ``indices`` with no ``filler``, with ``ValueError``.
    """
    array = np.asarray(indices, dtype=np.int32).reshape(-1)
    target = rung(int(array.shape[0]))
    if target <= array.shape[0]:
        return array
    if filler is None:
        if array.shape[0] == 0:
            raise ValueError("An empty index set needs an explicit filler.")
        filler = int(array[0])
    return np.concatenate([array, np.full(target - array.shape[0], filler, dtype=np.int32)])
