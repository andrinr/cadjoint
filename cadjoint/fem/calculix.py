"""CalculiX (ccx) structural backend: deck writer, parsers, adjoint sensitivities.

CalculiX is an Abaqus-like Fortran FEA code with a native discrete adjoint:
a ``*SENSITIVITY`` step after ``*STATIC`` computes objective gradients
w.r.t. ``*DESIGN VARIABLES, TYPE=COORDINATE`` (a surface node set), one
adjoint-cost pass for all design nodes, projected on the outward surface
normal.  This module drives a ``ccx`` binary over a subprocess boundary:
HEX8 meshes from :func:`cadjoint.fem.sdf_to_hex_mesh` serialize 1:1 to C3D8
``.inp`` decks (VTK and Abaqus brick corner order coincide), tractions
enter as consistent nodal loads (2x2 Gauss on the bilinear boundary
quads, matching jax-fem's surface integration), and displacements /
stresses / sensitivities are parsed back from ``.dat`` and ``.frd``.

Sensitivity semantics (validated against central finite differences and
the ccx 2.23 sources): the raw ``DFDN`` values ccx writes for the
STRAINENERGY design response omit the Jacobian-variation term of the
frozen-displacement partial — ``objective_shapeener_dx.f`` integrates
``sigma . d(eps)`` over the perturbed volume but never adds
``w * d(detJ)``.  The true fixed-load strain-energy shape derivative is

    dE/ds_i = DFDN_i + sum_{e ∋ i} sum_q w_q detJ_q (grad N_i(q) . n_i)

where ``w`` is the strain-energy density of the unperturbed solution and
``n_i`` the outward node normal ccx itself reports in the ``NORM`` frd
block.  :func:`energy_volume_gradient` computes the correction field;
with it applied, sensitivities match finite differences to the precision
of ccx's text output (~1e-5 relative; see ``tests/fem/test_calculix.py``).

The ccx binary is located via the ``CADJOINT_CCX`` or ``CCX`` environment
variables or ``PATH``.  A conda-forge build works on macOS arm64::

    micromamba create -p ./ccx-env -c conda-forge calculix

Heterogeneous materials: a ccx deck names materials — ``*MATERIAL`` /
``*ELSET`` / ``*SOLID SECTION`` — and has no way to carry a per-element
property array, so a *continuously blended* interface (what the smooth
CSG booleans of :mod:`cadjoint.fem.properties` sample) **cannot be
represented exactly**.  The deck writer therefore discretizes the field
and says so; see :func:`write_elastic_deck` for the exact rule, and
:class:`MaterialQuantization` for what a caller gets told about it.

GPL note: CalculiX is GPL-2; it stays behind the subprocess boundary
(decks in, result files out — no linking).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cadjoint.fem.backends import _TESSERACT_EXTRA_MESSAGE, ElasticBCs, TesseractBackend, _x64_scope
from cadjoint.fem.boundary import _boundary_face_rows
from cadjoint.fem.elements import HEX_CORNER_SIGNS
from cadjoint.fem.properties import quantize_to_materials

__all__ = [
    "MATERIAL_GROUP_TOLERANCE",
    "MAX_MATERIAL_GROUPS",
    "CalculixBackend",
    "CalculixQuantizationWarning",
    "CcxElasticSolution",
    "ElasticDeck",
    "MaterialQuantization",
    "consistent_nodal_forces",
    "elastic_ccx_solve",
    "energy_volume_gradient",
    "find_ccx",
    "parse_dat_displacements",
    "parse_dat_stresses",
    "parse_frd_fields",
    "require_ccx",
    "run_ccx",
    "strain_energy_solve",
    "von_mises",
    "write_elastic_deck",
]

_CCX_INSTALL_MESSAGE = (
    "CalculiX binary (ccx) not found. Point the CADJOINT_CCX or CCX environment "
    "variable at a ccx executable, or put one on PATH — e.g. from conda-forge: "
    "micromamba create -p ./ccx-env -c conda-forge calculix"
)

# 2x2x2 Gauss abscissae (weights are all 1).
_GAUSS_1D = float(1.0 / np.sqrt(3.0))


def find_ccx(explicit: str | os.PathLike | None = None) -> str | None:
    """Locate the ccx executable.

    Resolution order: ``explicit`` argument, ``CADJOINT_CCX`` env var,
    ``CCX`` env var, ``ccx`` on PATH.

    Args:
        explicit: Optional explicit path to a ccx binary.

    Returns:
        The path as a string, or ``None`` when no binary is found.
    """
    for candidate in (explicit, os.environ.get("CADJOINT_CCX"), os.environ.get("CCX")):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return shutil.which("ccx")


def require_ccx(explicit: str | os.PathLike | None = None) -> str:
    """Like :func:`find_ccx` but raising with install instructions when missing."""
    ccx = find_ccx(explicit)
    if ccx is None:
        raise RuntimeError(_CCX_INSTALL_MESSAGE)
    return ccx


def _hex_gauss_gradients() -> np.ndarray:
    """Reference shape-function gradients ``dN/dxi`` at the 8 Gauss points.

    Returns:
        Array shaped ``(Q=8, nodes=8, dim=3)``.
    """
    g = _GAUSS_1D
    points = np.array(
        [(a, b, c) for a in (-g, g) for b in (-g, g) for c in (-g, g)], dtype=np.float64
    )
    s = HEX_CORNER_SIGNS  # (8, 3)
    out = np.zeros((8, 8, 3))
    for q, xi in enumerate(points):
        out[q, :, 0] = 0.125 * s[:, 0] * (1 + s[:, 1] * xi[1]) * (1 + s[:, 2] * xi[2])
        out[q, :, 1] = 0.125 * s[:, 1] * (1 + s[:, 0] * xi[0]) * (1 + s[:, 2] * xi[2])
        out[q, :, 2] = 0.125 * s[:, 2] * (1 + s[:, 0] * xi[0]) * (1 + s[:, 1] * xi[1])
    return out


_HEX_GAUSS_GRADS = _hex_gauss_gradients()


def consistent_nodal_forces(
    points: np.ndarray, faces: np.ndarray, traction: np.ndarray
) -> np.ndarray:
    """Consistent nodal forces of a constant traction on bilinear quads.

    Integrates ``t * N_i`` over each face with 2x2 Gauss quadrature — the
    same integration jax-fem uses for its HEX8 surface terms, so loads
    match across backends.

    Args:
        points: Mesh vertex positions, ``(N, 3)``.
        faces: Quad connectivity, ``(M, 4)`` (VTK corner order).
        traction: Constant traction vector (force per area), ``(3,)``.

    Returns:
        Per-node force array shaped ``(N, 3)``.
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 4)
    traction = np.asarray(traction, dtype=np.float64).reshape(3)
    forces = np.zeros_like(points)
    if faces.size == 0:
        return forces
    g = _GAUSS_1D
    corners = points[faces]  # (M, 4, 3)
    signs = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float64)
    for xi, eta in ((a, b) for a in (-g, g) for b in (-g, g)):
        shape = 0.25 * (1 + signs[:, 0] * xi) * (1 + signs[:, 1] * eta)  # (4,)
        d_xi = 0.25 * signs[:, 0] * (1 + signs[:, 1] * eta)
        d_eta = 0.25 * signs[:, 1] * (1 + signs[:, 0] * xi)
        tangent_xi = np.einsum("i,mid->md", d_xi, corners)
        tangent_eta = np.einsum("i,mid->md", d_eta, corners)
        area = np.linalg.norm(np.cross(tangent_xi, tangent_eta), axis=-1)  # (M,)
        weights = shape[None, :] * area[:, None]  # (M, 4)
        np.add.at(forces, faces.reshape(-1), (weights[..., None] * traction).reshape(-1, 3))
    return forces


#: Relative tolerance below which two elements share one ``*MATERIAL`` block.
#: Tight on purpose: it exists to collapse the *identical* properties of a
#: sharp region into one group, not to merge genuinely different materials.
MATERIAL_GROUP_TOLERANCE = 1e-9

#: Default cap on the number of ``*MATERIAL`` blocks a single deck may carry.
#: A blended interface produces one group per distinct blend fraction, i.e.
#: potentially one per element; past this many groups the field is snapped
#: onto reference materials instead (see :func:`write_elastic_deck`).
MAX_MATERIAL_GROUPS = 32

#: The property keys a CalculiX ``*ELASTIC`` card carries, in card order.
_ELASTIC_KEYS = ("youngs_modulus", "poisson_ratio")


class CalculixQuantizationWarning(UserWarning):
    """A blended material field was approximated by named ccx materials.

    Raised (as a warning) by :func:`write_elastic_deck` whenever the deck it
    produced cannot reproduce the requested per-element properties exactly.
    Catch it with ``warnings.catch_warnings`` to turn it into an error, or
    read the same numbers off :attr:`CcxElasticSolution.quantization`.
    """


@dataclass(frozen=True)
class MaterialQuantization:
    """What discretizing a material field onto named ccx materials cost.

    A CalculiX deck can only name materials, so a per-element property field
    becomes a finite set of ``*MATERIAL`` blocks.  This records how faithful
    that representation is.

    Attributes:
        num_groups: Number of ``*MATERIAL`` blocks the deck carries.
        moved: Elements whose properties the deck does *not* reproduce (their
            relative property error exceeds the grouping tolerance).  Zero
            for a sharp scene, however many materials it mixes.
        max_relative_error: Largest relative property error introduced over
            all elements (0.0 when nothing moved).
        quantized: True when the group cap was hit and elements were snapped
            onto reference materials; False when every group is a group the
            field itself contains.
        reference_names: Names of the materials the deck ended up using, in
            deck order (``MAT0``, ``MAT1``, ... map onto these).
    """

    num_groups: int
    moved: int
    max_relative_error: float
    quantized: bool
    reference_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _DeckMaterial:
    """A reference material read off the deck itself (duck-types Material.get)."""

    name: str
    youngs: float
    poisson: float

    def get(self, key: str) -> float | None:
        """Property lookup mirroring :meth:`cadjoint.render.material.Material.get`."""
        return {"youngs_modulus": self.youngs, "poisson_ratio": self.poisson}.get(key)


def _relative_bins(values: np.ndarray, tolerance: float) -> np.ndarray:
    """Bin values onto a relative grid: equal bins means "agrees to ``tolerance``".

    Log-space binning makes the tolerance relative (a modulus in Pa and the
    same modulus in MPa group identically) and exact for equal inputs — the
    property of the rule that matters, since sharp regions carry *bit-identical*
    properties and must collapse to a single group.

    Args:
        values: Per-element property values, ``(C,)``.
        tolerance: Relative grid spacing.

    Returns:
        An integer-valued ``(C, 2)`` array of (sign, log-magnitude bin).
    """
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.log(np.maximum(np.abs(values), 1e-300)) / np.log1p(tolerance)
    return np.stack([np.sign(values), np.rint(magnitude)], axis=1)


def _distinct_groups(
    table: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse per-element property tuples into groups, in order of appearance.

    Args:
        table: Per-element property tuples, ``(C, K)``.
        tolerance: Relative tolerance within which two elements agree.

    Returns:
        ``(assignment, values, counts)`` — the ``(C,)`` group index of each
        element, the ``(G, K)`` representative property tuple of each group
        (the first member's own values, never an invented average), and the
        ``(G,)`` element count per group.
    """
    keys = np.concatenate(
        [_relative_bins(table[:, k], tolerance) for k in range(table.shape[1])], 1
    )
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    order = np.argsort(first)  # groups numbered by first appearance in the mesh
    remap = np.empty(order.shape[0], dtype=np.int64)
    remap[order] = np.arange(order.shape[0])
    assignment = remap[inverse]
    counts = np.bincount(assignment, minlength=order.shape[0])
    return assignment, table[first[order]], counts


def _dominant_groups(values: np.ndarray, counts: np.ndarray, max_groups: int) -> np.ndarray:
    """Choose ``max_groups`` representative groups out of too many.

    Greedy weighted farthest-point selection: seed with the group holding the
    most elements, then repeatedly take the group maximizing ``count x
    (log-property distance to the already-chosen set)``.  The count factor
    keeps the scene's bulk materials (a sharp region is one huge group, a
    blend band a swarm of tiny ones); the distance factor stops the pick from
    piling up on near-identical groups, which is what makes the worst-case
    error small.  Ties resolve to the lowest group index, so the choice is
    deterministic.

    Args:
        values: Per-group property tuples, ``(G, K)``.
        counts: Elements per group, ``(G,)``.
        max_groups: How many groups to keep.

    Returns:
        Indices of the chosen groups, ``(<= max_groups,)``.
    """
    log_values = np.log(np.maximum(np.abs(values), 1e-30))
    weight = counts.astype(np.float64)
    chosen = [int(np.argmax(weight))]
    distance = np.linalg.norm(log_values - log_values[chosen[0]], axis=1)
    while len(chosen) < min(max_groups, values.shape[0]):
        pick = int(np.argmax(weight * distance))
        if distance[pick] <= 0.0:  # every remaining group duplicates a chosen one
            break
        chosen.append(pick)
        distance = np.minimum(distance, np.linalg.norm(log_values - log_values[pick], axis=1))
    return np.array(chosen, dtype=np.int64)


def _plan_materials(
    cell_youngs: np.ndarray,
    cell_poisson: np.ndarray,
    *,
    materials: Sequence[Any] | None,
    max_groups: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, MaterialQuantization]:
    """Decide the deck's ``*MATERIAL`` blocks for a per-element property field.

    See :func:`write_elastic_deck` for the rule this implements.

    Args:
        cell_youngs: Per-element Young's modulus, ``(C,)``.
        cell_poisson: Per-element Poisson ratio, ``(C,)``.
        materials: Reference materials to snap onto — always, when given
            (anything with ``get("youngs_modulus")`` / ``get("poisson_ratio")``,
            e.g. :func:`cadjoint.materials.catalogue`).  ``None`` snaps only
            past the cap, onto the deck's own dominant groups.
        max_groups: Cap on the number of ``*MATERIAL`` blocks.
        tolerance: Relative grouping tolerance.

    Returns:
        ``(assignment, values, quantization)`` — the ``(C,)`` group index per
        element, the ``(G, 2)`` ``(youngs, poisson)`` of each group, and the
        :class:`MaterialQuantization` report.

    Raises:
        ValueError: If ``max_groups`` is below 1, or an explicit reference set
            is larger than the cap.
    """
    if max_groups < 1:
        raise ValueError(f"max_material_groups must be at least 1; got {max_groups}.")
    if materials is not None and len(materials) > max_groups:
        raise ValueError(
            f"{len(materials)} reference materials exceed max_material_groups="
            f"{max_groups}; raise the cap or pass fewer materials."
        )
    sampled = np.stack([cell_youngs, cell_poisson], axis=1)
    assignment, values, counts = _distinct_groups(sampled, tolerance)
    if materials is None and values.shape[0] <= max_groups:
        return (
            assignment,
            values,
            MaterialQuantization(
                num_groups=int(values.shape[0]),
                moved=0,
                max_relative_error=0.0,
                quantized=False,
                reference_names=tuple(f"MAT{i}" for i in range(values.shape[0])),
            ),
        )
    references: list[Any] = (
        list(materials)
        if materials is not None
        else [
            _DeckMaterial(f"MAT{rank}", float(values[group, 0]), float(values[group, 1]))
            for rank, group in enumerate(_dominant_groups(values, counts, max_groups))
        ]
    )
    snapped, error = quantize_to_materials(
        {"youngs_modulus": cell_youngs, "poisson_ratio": cell_poisson},
        references,
        keys=_ELASTIC_KEYS,
    )
    used = np.unique(snapped)
    compress = np.full(len(references), -1, dtype=np.int64)
    compress[used] = np.arange(used.shape[0])
    group_values = np.array(
        [
            [float(references[i].get("youngs_modulus")), float(references[i].get("poisson_ratio"))]
            for i in used
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    names = tuple(str(getattr(references[i], "name", None) or f"MAT{i}") for i in used)
    return (
        compress[snapped],
        group_values,
        MaterialQuantization(
            num_groups=int(used.shape[0]),
            moved=int(np.count_nonzero(error > tolerance)),
            max_relative_error=float(np.max(error)) if error.size else 0.0,
            quantized=True,
            reference_names=names,
        ),
    )


def _warn_quantization(
    quantization: MaterialQuantization | None, num_cells: int, *, stacklevel: int
) -> None:
    """Warn when a deck cannot reproduce the requested per-element properties.

    Args:
        quantization: The grouping report, or ``None`` (single material).
        num_cells: Element count, for context in the message.
        stacklevel: Passed to :func:`warnings.warn` so the message points at
            the caller that asked for the solve.
    """
    if quantization is None or not quantization.moved:
        return
    warnings.warn(
        f"CalculiX cannot carry a blended material field: {quantization.moved} of "
        f"{num_cells} elements were snapped onto {quantization.num_groups} named "
        f"materials, introducing up to {quantization.max_relative_error:.3%} relative "
        "property error. Pass explicit reference materials (materials=...) or raise "
        "max_material_groups, or run the jax-fem backend for an exact per-element field.",
        CalculixQuantizationWarning,
        stacklevel=stacklevel,
    )


def _plan_deck_materials(
    youngs: Any,
    poisson: Any,
    *,
    num_cells: int,
    materials: Sequence[Any] | None,
    max_groups: int,
    tolerance: float,
) -> tuple[np.ndarray | None, np.ndarray, MaterialQuantization | None]:
    """Normalize scalar-or-array moduli into deck material groups.

    Args:
        youngs: Young's modulus — scalar or ``(C,)``.
        poisson: Poisson ratio — scalar or ``(C,)``.
        num_cells: Element count ``C``.
        materials: Reference materials for quantization, or ``None``.
        max_groups: Cap on the number of ``*MATERIAL`` blocks.
        tolerance: Relative grouping tolerance.

    Returns:
        ``(assignment, values, quantization)``; ``assignment`` and
        ``quantization`` are ``None`` for the scalar (single-material) path,
        which is byte-for-byte the deck this writer has always produced.

    Raises:
        ValueError: If a per-element array is not shaped ``(C,)``.
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in (youngs, poisson)]
    if all(array.ndim == 0 for array in arrays):
        return None, np.array([[float(arrays[0]), float(arrays[1])]]), None
    for name, array in zip(("youngs", "poisson"), arrays):
        if array.ndim > 1 or (array.ndim == 1 and array.shape[0] not in (1, num_cells)):
            raise ValueError(
                f"{name} must be a scalar or shaped ({num_cells},); got shape {array.shape}."
            )
    cell_youngs, cell_poisson = (np.broadcast_to(array, (num_cells,)) for array in arrays)
    return _plan_materials(
        np.ascontiguousarray(cell_youngs),
        np.ascontiguousarray(cell_poisson),
        materials=materials,
        max_groups=max_groups,
        tolerance=tolerance,
    )


@dataclass(frozen=True)
class ElasticDeck:
    """A rendered ccx input deck plus the metadata needed to interpret results.

    Attributes:
        text: Complete ``.inp`` file contents.
        nodal_forces: Consistent nodal load vector applied via *CLOAD,
            shaped ``(N, 3)`` — also the ``f`` in ``E = f . u / 2``.
        design_nodes: Node indices (0-based) declared as coordinate design
            variables, or ``None`` for a forward-only deck.
        num_nodes: Number of mesh vertices.
        num_cells: Number of hexahedra.
        cell_youngs: The Young's modulus the deck actually gives each element,
            ``(C,)`` — post-grouping, so it differs from the requested field
            exactly where :attr:`quantization` says it does.  ``None`` for a
            single-material deck.
        cell_poisson: Likewise for the Poisson ratio, ``(C,)`` or ``None``.
        quantization: How faithfully the deck represents the requested
            per-element field, or ``None`` for a single-material deck.
    """

    text: str
    nodal_forces: np.ndarray
    design_nodes: np.ndarray | None
    num_nodes: int
    num_cells: int
    cell_youngs: np.ndarray | None = None
    cell_poisson: np.ndarray | None = None
    quantization: MaterialQuantization | None = None


def _num(value: float) -> str:
    """Render a float within ccx's 20-character free-format field limit.

    ``.17g`` overflows the limit for e.g. 17-digit negatives in (-0.1, -0.01)
    (21 chars) or tiny snapped coordinates like ``9.9e-18`` (22 chars); ccx
    silently truncates such fields. Degrade precision only when a field would
    overflow — well below mesh accuracy either way (ccx prints 6 digits back).
    """
    for precision in (17, 15, 13):
        rendered = f"{value:.{precision}g}"
        if len(rendered) <= 20:
            return rendered
    return f"{value:.12g}"


def _id_lines(header: str, indices: np.ndarray) -> list[str]:
    """A set block with 1-based ids, 16 per line (ccx's free-format limit)."""
    ids = (np.asarray(indices, dtype=np.int64).reshape(-1) + 1).tolist()
    lines = [header]
    for start in range(0, len(ids), 16):
        lines.append(", ".join(str(i) for i in ids[start : start + 16]))
    return lines


def _nset_lines(name: str, nodes: np.ndarray) -> list[str]:
    """*NSET block with 1-based ids, 16 per line."""
    return _id_lines(f"*NSET, NSET={name}", nodes)


def _material_lines(
    assignment: np.ndarray | None,
    values: np.ndarray,
    quantization: MaterialQuantization | None,
) -> list[str]:
    """The ``*MATERIAL`` / ``*SOLID SECTION`` section of a deck.

    One group keeps the historical single-material form (``EALL``, no
    ``*ELSET``); several groups emit an ``*ELSET`` + ``*MATERIAL`` +
    ``*SOLID SECTION`` triple each, in group order.

    Args:
        assignment: Per-element group index ``(C,)``, or ``None`` for the
            single-material deck.
        values: Per-group ``(youngs, poisson)``, ``(G, 2)``.
        quantization: The grouping report, used only for the ``**`` comment
            naming each group's source material.

    Returns:
        The deck lines.
    """
    names = quantization.reference_names if quantization is not None else ()
    if values.shape[0] == 1:
        return [
            "*MATERIAL, NAME=MAT0",
            "*ELASTIC",
            f"{_num(values[0, 0])}, {_num(values[0, 1])}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT0",
        ]
    lines: list[str] = []
    assert assignment is not None  # several groups only arise from a per-element field
    for group, (youngs, poisson) in enumerate(values):
        elements = np.flatnonzero(assignment == group)
        lines += _id_lines(f"*ELSET, ELSET=EMAT{group}", elements)
        source = names[group] if group < len(names) else ""
        if source and source != f"MAT{group}":
            lines.append(f"** MAT{group}: {source}")
        lines += [
            f"*MATERIAL, NAME=MAT{group}",
            "*ELASTIC",
            f"{_num(youngs)}, {_num(poisson)}",
            f"*SOLID SECTION, ELSET=EMAT{group}, MATERIAL=MAT{group}",
        ]
    return lines


def write_elastic_deck(
    points: np.ndarray,
    cells: np.ndarray,
    bcs: ElasticBCs,
    *,
    youngs: float | np.ndarray,
    poisson: float | np.ndarray,
    design_nodes: np.ndarray | None = None,
    materials: Sequence[Any] | None = None,
    max_material_groups: int = MAX_MATERIAL_GROUPS,
    group_tolerance: float = MATERIAL_GROUP_TOLERANCE,
) -> ElasticDeck:
    """Render a linear-elastic ccx deck for a HEX8 mesh.

    C3D8 corner order equals the VTK/meshio hexahedron order, so cells
    serialize 1:1 (ids shifted to 1-based).  Clamps become ``*BOUNDARY``
    on an ``*NSET``; tractions become ``*CLOAD`` consistent nodal forces
    on the boundary faces spanned by each patch (all four corners in the
    patch set — the same rule as the jax-fem backend).  With
    ``design_nodes`` a second ``*SENSITIVITY`` step with the STRAINENERGY
    design response is appended and ``SEN`` output is requested.

    **Heterogeneous materials.**  ``youngs`` and ``poisson`` may be
    per-element ``(C,)`` arrays (as sampled from the scene's material field
    by :mod:`cadjoint.fem.properties`).  A ccx deck *names* materials, so it
    cannot carry a per-element array and cannot represent a continuously
    blended interface exactly; this is the rule that discretizes the field,
    in three steps:

    1. **Group.**  Elements whose ``(youngs, poisson)`` agree to
       ``group_tolerance`` relative (default ``1e-9``, log-space bins) become
       one group, which gets one ``*ELSET`` + ``*MATERIAL`` + ``*SOLID
       SECTION`` triple.  Groups are numbered ``MAT0``, ``MAT1``, ... by
       first appearance in the mesh, and each carries the properties of its
       first member (never an average).  A *sharp* multi-material scene has
       bit-identical properties per region, so it collapses to exactly one
       group per region and the deck is **exact**.
    2. **Cap.**  A blended interface produces a distinct blend fraction — so
       a distinct group — per element, which would emit thousands of
       ``*MATERIAL`` blocks.  Past ``max_material_groups`` (default 32) every
       element is instead snapped to its nearest reference material in
       log-property space (:func:`cadjoint.fem.properties.quantize_to_materials`;
       ties go to the lowest reference index).  Passing ``materials``
       explicitly (e.g. :func:`cadjoint.materials.catalogue`) always snaps
       onto exactly those, cap or no cap — that is what asking for named
       materials means.  With ``materials=None`` the reference set is drawn
       from the field itself: the group holding the most elements, then
       greedily whichever group maximizes ``elements x log-property distance
       to the set already chosen``.  For the scene this is meant for — sharp
       bulk regions joined by a thin blend band — that is precisely the list
       of bulk materials, and it stays well spread when the band is not thin.
    3. **Report.**  The deck reports what that cost:
       :attr:`ElasticDeck.quantization` carries the number of elements moved
       and the maximum relative property error, the same numbers reach a
       solve's caller on :attr:`CcxElasticSolution.quantization`, and a
       :class:`CalculixQuantizationWarning` is raised whenever any element
       moved, so the approximation is never silent.

    A single-material deck (scalar arguments, or a uniform per-element field)
    emits exactly the historical ``EALL`` section — no ``*ELSET``.

    Args:
        points: Vertex positions, ``(N, 3)``.
        cells: HEX8 connectivity, ``(C, 8)`` (VTK corner order).
        bcs: Array-level boundary conditions.
        youngs: Young's modulus — scalar, or per-element ``(C,)``.
        poisson: Poisson ratio — scalar, or per-element ``(C,)``.
        design_nodes: Optional 0-based node indices for coordinate design
            variables (must lie on the boundary surface).
        materials: Optional reference materials to quantize onto (anything
            answering ``get("youngs_modulus")`` / ``get("poisson_ratio")``,
            such as :class:`cadjoint.render.material.Material`).
        max_material_groups: Cap on the number of ``*MATERIAL`` blocks.
        group_tolerance: Relative tolerance for collapsing elements into one
            group.

    Returns:
        The rendered :class:`ElasticDeck`.

    Raises:
        ValueError: If a per-element property array is not shaped ``(C,)``,
            the cap is below 1, or an explicit reference set exceeds the cap.

    Warns:
        CalculixQuantizationWarning: When the deck cannot reproduce the
            requested per-element properties exactly.
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    assignment, group_values, quantization = _plan_deck_materials(
        youngs,
        poisson,
        num_cells=int(cells.shape[0]),
        materials=materials,
        max_groups=max_material_groups,
        tolerance=group_tolerance,
    )
    lines = ["*NODE, NSET=NALL"]
    for index, (x, y, z) in enumerate(points, start=1):
        lines.append(f"{index}, {_num(x)}, {_num(y)}, {_num(z)}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=EALL")
    for index, cell in enumerate(cells + 1, start=1):
        lines.append(f"{index}, " + ", ".join(str(n) for n in cell))

    if bcs.fixed_nodes:
        fixed = np.unique(np.concatenate([np.asarray(n).reshape(-1) for n in bcs.fixed_nodes]))
        lines += _nset_lines("FIXED", fixed)

    if design_nodes is not None:
        design_nodes = np.unique(np.asarray(design_nodes, dtype=np.int64).reshape(-1))
        lines += _nset_lines("DESIGN", design_nodes)
        lines += ["*DESIGN VARIABLES, TYPE=COORDINATE", "DESIGN"]

    lines += _material_lines(assignment, group_values, quantization)
    lines += ["*STEP", "*STATIC"]
    if bcs.fixed_nodes:
        lines += ["*BOUNDARY", "FIXED, 1, 3, 0.0"]

    boundary = _boundary_face_rows(cells)
    forces = np.zeros_like(points)
    for patch, vector in zip(bcs.traction_nodes, bcs.traction_vectors):
        indices = np.asarray(patch).reshape(-1)
        spanned = boundary[np.isin(boundary, indices).all(axis=1)]
        forces += consistent_nodal_forces(points, spanned, np.asarray(vector, dtype=np.float64))
    loaded = np.argwhere(np.abs(forces) > 0.0)
    if loaded.size:
        lines.append("*CLOAD")
        for node, dof in loaded:
            lines.append(f"{node + 1}, {dof + 1}, {_num(forces[node, dof])}")

    lines += ["*NODE PRINT, NSET=NALL", "U", "*EL PRINT, ELSET=EALL", "S"]
    lines += ["*NODE FILE", "U", "*END STEP"]
    if design_nodes is not None:
        lines += [
            "*STEP",
            "*SENSITIVITY",
            "*DESIGN RESPONSE, NAME=DOBJ",
            "STRAINENERGY",
            "*NODE FILE",
            "SEN",
            "*END STEP",
        ]
    lines.append("")
    _warn_quantization(quantization, int(cells.shape[0]), stacklevel=3)
    effective = None if assignment is None else group_values[assignment]
    return ElasticDeck(
        text="\n".join(lines),
        nodal_forces=forces,
        design_nodes=design_nodes,
        num_nodes=int(points.shape[0]),
        num_cells=int(cells.shape[0]),
        cell_youngs=None if effective is None else effective[:, 0],
        cell_poisson=None if effective is None else effective[:, 1],
        quantization=quantization,
    )


def run_ccx(
    deck_text: str,
    directory: str | os.PathLike,
    *,
    jobname: str = "job",
    ccx: str | os.PathLike | None = None,
) -> Path:
    """Write ``jobname.inp`` into ``directory`` and run ccx on it.

    Args:
        deck_text: Complete input deck contents.
        directory: Working directory (created if missing); result files
            land next to the deck.
        jobname: ccx job name (file stem).
        ccx: Optional explicit binary path (default: :func:`find_ccx`).

    Returns:
        The working directory as a :class:`~pathlib.Path`.

    Raises:
        RuntimeError: When no binary is found or the run fails (the ccx
            output tail is included in the message).
    """
    binary = require_ccx(ccx)
    workdir = Path(directory)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / f"{jobname}.inp").write_text(deck_text)
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "1")
    result = subprocess.run(
        [binary, jobname],
        cwd=workdir,
        capture_output=True,
        text=True,
        env=environment,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 or "*ERROR" in output or "Job finished" not in output:
        raise RuntimeError(f"ccx failed in {workdir} (exit {result.returncode}):\n{output[-3000:]}")
    return workdir


_DAT_ROW = re.compile(
    r"^\s*(\d+)((?:\s+[-+]?\d*\.\d+E[-+]\d+)+)\s*$",
)


def parse_dat_displacements(text: str, num_nodes: int) -> np.ndarray:
    """Parse the ``*NODE PRINT ... U`` block of a ccx ``.dat`` file.

    Args:
        text: Contents of the ``.dat`` file.
        num_nodes: Total mesh node count (rows index into this).

    Returns:
        Displacements shaped ``(num_nodes, 3)`` (unlisted nodes zero).

    Raises:
        ValueError: If no displacement block is present.
    """
    displacement = np.zeros((num_nodes, 3))
    in_block = False
    found = False
    for line in text.splitlines():
        if "displacements (vx,vy,vz)" in line:
            in_block = True
            found = True
            continue
        if in_block:
            match = _DAT_ROW.match(line)
            if match:
                values = [float(v) for v in match.group(2).split()]
                if len(values) == 3:
                    displacement[int(match.group(1)) - 1] = values
            elif line.strip() and not line.startswith(" "):
                in_block = False
    if not found:
        raise ValueError("No displacement block found in ccx .dat output.")
    return displacement


def parse_dat_stresses(text: str, num_cells: int) -> np.ndarray:
    """Parse ``*EL PRINT ... S`` integration-point stresses from ``.dat``.

    Args:
        text: Contents of the ``.dat`` file.
        num_cells: Total element count.

    Returns:
        Per-element mean stress ``(num_cells, 6)`` in ccx component order
        ``(sxx, syy, szz, sxy, sxz, syz)``.

    Raises:
        ValueError: If no stress block is present.
    """
    sums = np.zeros((num_cells, 6))
    counts = np.zeros(num_cells, dtype=np.int64)
    in_block = False
    found = False
    row = re.compile(r"^\s*(\d+)\s+(\d+)((?:\s+[-+]?\d*\.\d+E[-+]\d+){6})\s*$")
    for line in text.splitlines():
        if "stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz)" in line:
            in_block = True
            found = True
            continue
        if in_block:
            match = row.match(line)
            if match:
                element = int(match.group(1)) - 1
                sums[element] += [float(v) for v in match.group(3).split()]
                counts[element] += 1
            elif "for set" in line or "S T E P" in line:
                in_block = False
    if not found:
        raise ValueError("No stress block found in ccx .dat output.")
    counts = np.maximum(counts, 1)
    return sums / counts[:, None]


def von_mises(stresses: np.ndarray) -> np.ndarray:
    """Von Mises stress from ``(..., 6)`` components ``(sxx, syy, szz, sxy, sxz, syz)``."""
    s = np.asarray(stresses, dtype=np.float64)
    sxx, syy, szz, sxy, sxz, syz = (s[..., i] for i in range(6))
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + sxz**2 + syz**2)
    )


def parse_frd_fields(text: str, num_nodes: int) -> dict[str, np.ndarray]:
    """Parse the nodal result blocks of a ccx ``.frd`` file.

    Handles the ASCII format: each block opens with a ``-4  NAME`` line
    followed by ``-5`` component headers and ``-1`` node rows in fixed
    12-character float columns.  Typical block names: ``DISP`` (3
    components), ``NORM`` (design-node normals, 3), ``SENENER``
    (STRAINENERGY sensitivities: raw ``DFDN`` and filtered ``DFDNFIL``).

    Args:
        text: Contents of the ``.frd`` file.
        num_nodes: Total mesh node count.

    Returns:
        Mapping of block name to ``(num_nodes, components)`` array
        (nodes absent from a block stay zero).
    """
    fields: dict[str, np.ndarray] = {}
    name: str | None = None
    components = 0
    for line in text.splitlines():
        if line.startswith(" -4"):
            parts = line.split()
            name = parts[1]
            components = 0
        elif line.startswith(" -5") and name is not None:
            if not line.split()[1].startswith("ALL"):
                components += 1
                fields.setdefault(name, np.zeros((num_nodes, 0)))
                if fields[name].shape[1] < components:
                    grown = np.zeros((num_nodes, components))
                    grown[:, : fields[name].shape[1]] = fields[name]
                    fields[name] = grown
        elif line.startswith(" -1") and name in fields:
            node = int(line[3:13])
            values = [float(line[13 + 12 * q : 25 + 12 * q]) for q in range(fields[name].shape[1])]
            fields[name][node - 1] = values
        elif line.startswith(" -3"):
            name = None
    return fields


def energy_volume_gradient(
    points: np.ndarray,
    cells: np.ndarray,
    displacement: np.ndarray,
    *,
    youngs: float,
    poisson: float,
) -> np.ndarray:
    """Gradient of the strain-energy volume term w.r.t. node positions.

    Computes ``g[i, a] = d/dx_{ia} ( sum_q w_q detJ_q )`` at frozen
    strains: the strain-energy density ``w`` of the given solution times
    the derivative of the volume element, using the closed form
    ``d(detJ)/dx_{ia} = detJ * dN_i/dx_a``.  This is exactly the term ccx
    2.23 omits from its STRAINENERGY ``DFDN`` output (see the module
    docstring); adding ``g[i] . n_i`` recovers the true fixed-load shape
    derivative.

    Per-element moduli are supported and *required* for a multi-material
    deck: the omitted term is ``w_e d(detJ)`` with ``w_e`` the element's own
    strain-energy density, so each element must contribute with the material
    ccx actually solved it with (i.e. the deck's post-grouping properties,
    :attr:`ElasticDeck.cell_youngs`, not the raw blended field).

    Args:
        points: Vertex positions, ``(N, 3)``.
        cells: HEX8 connectivity, ``(C, 8)``.
        displacement: Nodal displacements from the forward solve, ``(N, 3)``.
        youngs: Young's modulus — scalar or per-element ``(C,)``.
        poisson: Poisson ratio — scalar or per-element ``(C,)``.

    Returns:
        Per-node gradient field shaped ``(N, 3)``.
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    displacement = np.asarray(displacement, dtype=np.float64)
    # Per-element moduli get a trailing quadrature axis so (C,) broadcasts
    # against the (C, Q) energy density below; scalars stay scalars.
    youngs = np.asarray(youngs, dtype=np.float64)
    poisson = np.asarray(poisson, dtype=np.float64)
    if youngs.ndim or poisson.ndim:
        youngs, poisson = (np.reshape(v, (-1, 1)) for v in np.broadcast_arrays(youngs, poisson))
    lame_lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    lame_mu = youngs / (2.0 * (1.0 + poisson))

    corners = points[cells]  # (C, 8, 3)
    corner_disp = displacement[cells]  # (C, 8, 3)
    d = _HEX_GAUSS_GRADS  # (Q, 8, 3)
    jacobian = np.einsum("cna,qnb->cqab", corners, d)  # (C, Q, 3, 3)
    det = np.linalg.det(jacobian)  # (C, Q)
    inverse = np.linalg.inv(jacobian)  # (C, Q, 3, 3)
    grad_phys = np.einsum("qnb,cqba->cqna", d, inverse)  # (C, Q, 8, 3)
    u_grad = np.einsum("cna,cqnb->cqab", corner_disp, grad_phys)  # (C, Q, 3, 3)
    strain = 0.5 * (u_grad + np.swapaxes(u_grad, -1, -2))
    trace = np.trace(strain, axis1=-2, axis2=-1)
    density = 0.5 * lame_lambda * trace**2 + lame_mu * np.einsum("cqab,cqab->cq", strain, strain)

    contribution = np.einsum("cq,cqna->cna", density * det, grad_phys)  # (C, 8, 3)
    gradient = np.zeros_like(points)
    np.add.at(gradient, cells.reshape(-1), contribution.reshape(-1, 3))
    return gradient


@dataclass(frozen=True)
class CcxElasticSolution:
    """Results of a ccx elastic solve (forward, optionally with adjoint).

    Attributes:
        displacement: Nodal displacements, ``(N, 3)``.
        strain_energy: Total strain energy ``f . u / 2`` under the
            consistent nodal loads.
        cell_stress: Per-element mean stress, ``(C, 6)``.
        strain_energy_gradient: ``d(strain energy)/d(points)`` shaped
            ``(N, 3)`` — nonzero only at design nodes and only along
            their outward normals — or ``None`` for forward-only solves.
        normals: ccx's outward design-node normals, ``(N, 3)`` (zero at
            non-design nodes), or ``None``.
        quantization: How faithfully the deck represented a per-element
            material field (see :func:`write_elastic_deck`); ``None`` for a
            scalar single-material solve, ``moved == 0`` when the field was
            sharp and the deck exact.
    """

    displacement: np.ndarray
    strain_energy: float
    cell_stress: np.ndarray
    strain_energy_gradient: np.ndarray | None = None
    normals: np.ndarray | None = None
    quantization: MaterialQuantization | None = None


def elastic_ccx_solve(
    points: np.ndarray,
    cells: np.ndarray,
    bcs: ElasticBCs,
    *,
    youngs: float | np.ndarray,
    poisson: float | np.ndarray,
    sensitivities: bool = False,
    design_nodes: np.ndarray | None = None,
    ccx: str | os.PathLike | None = None,
    workdir: str | os.PathLike | None = None,
    materials: Sequence[Any] | None = None,
    max_material_groups: int = MAX_MATERIAL_GROUPS,
    group_tolerance: float = MATERIAL_GROUP_TOLERANCE,
) -> CcxElasticSolution:
    """Run a ccx linear-elastic solve (and optionally the adjoint).

    ``youngs``/``poisson`` may be per-element ``(C,)`` arrays; the deck
    writer discretizes them onto named ccx materials (see
    :func:`write_elastic_deck` for the rule) and the result's
    :attr:`~CcxElasticSolution.quantization` reports what that cost.

    Args:
        points: Vertex positions, ``(N, 3)``.
        cells: HEX8 connectivity, ``(C, 8)``.
        bcs: Array-level boundary conditions.
        youngs: Young's modulus — scalar or per-element ``(C,)``.
        poisson: Poisson ratio — scalar or per-element ``(C,)``.
        sensitivities: Also run the ``*SENSITIVITY`` step and return the
            corrected strain-energy gradient (see module docstring).
        design_nodes: Coordinate design variables for the sensitivity
            step; defaults to every boundary node.
        ccx: Optional explicit binary path.
        workdir: Optional directory to keep the deck and result files in
            (default: a temporary directory, removed afterwards).
        materials: Optional reference materials to quantize onto (e.g.
            :func:`cadjoint.materials.catalogue`); ``None`` snaps only past
            the cap, onto the field's own dominant groups.
        max_material_groups: Cap on the number of ``*MATERIAL`` blocks.
        group_tolerance: Relative tolerance for collapsing elements into one
            material group.

    Returns:
        A :class:`CcxElasticSolution`.
    """
    cells = np.asarray(cells, dtype=np.int64)
    if sensitivities and design_nodes is None:
        design_nodes = np.unique(_boundary_face_rows(cells))
    deck = write_elastic_deck(
        points,
        cells,
        bcs,
        youngs=youngs,
        poisson=poisson,
        design_nodes=design_nodes if sensitivities else None,
        materials=materials,
        max_material_groups=max_material_groups,
        group_tolerance=group_tolerance,
    )
    # The correction below must use the properties ccx actually solved with,
    # which are the deck's post-grouping ones, not the requested field.
    solved_youngs = youngs if deck.cell_youngs is None else deck.cell_youngs
    solved_poisson = poisson if deck.cell_poisson is None else deck.cell_poisson

    def solve(directory: str | os.PathLike) -> CcxElasticSolution:
        run_ccx(deck.text, directory, ccx=ccx)
        dat = (Path(directory) / "job.dat").read_text()
        displacement = parse_dat_displacements(dat, deck.num_nodes)
        stress = parse_dat_stresses(dat, deck.num_cells)
        energy = 0.5 * float(np.sum(deck.nodal_forces * displacement))
        gradient = None
        normals = None
        if sensitivities:
            frd = (Path(directory) / "job.frd").read_text()
            fields = parse_frd_fields(frd, deck.num_nodes)
            normals = fields["NORM"]
            raw = fields["SENENER"][:, 0]  # DFDN (unfiltered)
            correction = energy_volume_gradient(
                points, cells, displacement, youngs=solved_youngs, poisson=solved_poisson
            )
            along_normal = raw + np.einsum("nd,nd->n", correction, normals)
            gradient = np.zeros_like(normals)
            design = deck.design_nodes
            gradient[design] = along_normal[design, None] * normals[design]
        return CcxElasticSolution(
            displacement=displacement,
            strain_energy=energy,
            cell_stress=stress,
            strain_energy_gradient=gradient,
            normals=normals,
            quantization=deck.quantization,
        )

    if workdir is not None:
        return solve(workdir)
    with tempfile.TemporaryDirectory(prefix="cadjoint-ccx-") as temporary:
        return solve(temporary)


def _unpack_elastic_bcs(
    fixed_nodes: np.ndarray,
    traction_nodes: np.ndarray,
    traction_offsets: np.ndarray,
    traction_vectors: np.ndarray,
) -> ElasticBCs:
    """Rebuild :class:`ElasticBCs` from the flat tesseract encoding."""
    offsets = np.asarray(traction_offsets, dtype=np.int64)
    nodes = np.asarray(traction_nodes, dtype=np.int32)
    vectors = np.asarray(traction_vectors, dtype=np.float64)
    return ElasticBCs(
        fixed_nodes=[np.asarray(fixed_nodes, dtype=np.int32)],
        traction_nodes=[nodes[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])],
        traction_vectors=list(vectors),
    )


class CalculixBackend(TesseractBackend):
    """Solver backend running CalculiX behind the packaged tesseract.

    Forward elastic solves route through
    ``cadjoint/fem/tesseracts/elastic_calculix`` (subprocess ccx, local
    ``Tesseract.from_tesseract_api``).  Differentiability is
    objective-valued: the tesseract's ``strain_energy`` output carries an
    adjoint VJP w.r.t. ``points`` (ccx ``*SENSITIVITY`` + the volume-term
    correction); cotangents on the raw displacement field raise
    ``NotImplementedError`` because ccx has no general displacement
    adjoint.  Thermal solves are not supported.
    """

    name = "calculix"

    def __init__(self, ccx: str | os.PathLike | None = None):
        try:
            from tesseract_core import Tesseract  # noqa: F401
        except ImportError as error:
            raise ImportError(_TESSERACT_EXTRA_MESSAGE) from error
        require_ccx(ccx)  # fail fast with install instructions
        if ccx is not None:
            os.environ["CADJOINT_CCX"] = str(ccx)  # the tesseract subprocess resolves via env
        api = Path(__file__).parent / "tesseracts" / "elastic_calculix" / "tesseract_api.py"
        self._api_paths = {"elastic": api}
        self._tesseracts: dict[str, Any] = {}

    def thermal(self, points, cells, bcs, *, conductivity, source, base_points=None):
        """Unsupported: the CalculiX integration covers elastic solves only."""
        raise NotImplementedError(
            "The CalculiX backend supports elastic solves only; "
            "use the 'jaxfem' or 'tesseract' backend for thermal studies."
        )

    def elastic(self, points, cells, bcs, *, youngs, poisson, base_points=None, body_force=None):
        """See :meth:`cadjoint.fem.backends.SolverBackend.elastic`.

        ``youngs``/``poisson`` may be per-element ``(C,)`` arrays; they cross
        the tesseract boundary as the optional ``cell_youngs``/``cell_poisson``
        arrays and become named ccx materials (see
        :func:`write_elastic_deck`), with a
        :class:`CalculixQuantizationWarning` if the deck cannot represent the
        field exactly.

        Args:
            points: Vertex positions, ``(N, 3)`` (may be traced).
            cells: HEX8 connectivity, ``(C, 8)``.
            bcs: Array-level boundary conditions.
            youngs: Young's modulus — scalar or per-element ``(C,)``.
            poisson: Poisson ratio — scalar or per-element ``(C,)``.
            base_points: Unused (the tesseract runtime hands its endpoints
                concrete arrays).
            body_force: Must be ``None``.

        Returns:
            Nodal displacements ``(N, 3)`` as a JAX array.

        Raises:
            NotImplementedError: If ``body_force`` is given — a body force
                needs ``*DLOAD``/``GRAV`` cards this deck writer does not
                emit, and silently dropping self-weight would be worse than
                saying so.
        """
        del base_points
        from tesseract_jax import apply_tesseract

        if body_force is not None:
            raise NotImplementedError(
                "The CalculiX backend does not apply body forces: self-weight would "
                "need *DLOAD/GRAV cards the ccx deck writer does not emit. Drop "
                "body_force, or use the 'jaxfem' backend for self-weight studies."
            )
        with _x64_scope():
            outputs = apply_tesseract(
                self._tesseract_for("elastic"),
                self._elastic_inputs(points, cells, bcs, youngs, poisson),
            )
            return outputs["displacement"]

    def elastic_strain_energy(self, points, cells, bcs, *, youngs, poisson, base_points=None):
        """Differentiable total strain energy of the elastic solve.

        Returns a JAX scalar whose VJP w.r.t. ``points`` is served by the
        ccx ``*SENSITIVITY`` adjoint (normal-projected design-node
        sensitivities plus the volume-term correction).  For a linear
        problem under fixed loads, compliance is twice this value.
        ``youngs``/``poisson`` accept per-element ``(C,)`` arrays exactly as
        :meth:`elastic` does.
        """
        del base_points
        from tesseract_jax import apply_tesseract

        with _x64_scope():
            outputs = apply_tesseract(
                self._tesseract_for("elastic"),
                self._elastic_inputs(points, cells, bcs, youngs, poisson),
            )
            return outputs["strain_energy"]

    def _elastic_inputs(self, points, cells, bcs, youngs, poisson) -> dict[str, Any]:
        """Pack BCs and materials into the flat tesseract schema.

        A scalar modulus leaves ``cell_youngs``/``cell_poisson`` empty, so the
        wire payload of a single-material solve is exactly what it always was.
        """
        import jax.numpy as jnp

        from cadjoint.fem.backends import _as_cell_array

        if bcs.fixed_nodes:
            fixed = np.unique(
                np.concatenate([np.asarray(n, dtype=np.int32) for n in bcs.fixed_nodes])
            ).astype(np.int32)
        else:
            fixed = np.zeros(0, dtype=np.int32)
        if bcs.traction_nodes:
            traction_nodes = np.concatenate(
                [np.asarray(n, dtype=np.int32) for n in bcs.traction_nodes]
            )
            traction_vectors = np.asarray(bcs.traction_vectors, dtype=np.float64)
        else:
            traction_nodes = np.zeros(0, dtype=np.int32)
            traction_vectors = np.zeros((0, 3), dtype=np.float64)
        offsets = np.concatenate(
            [[0], np.cumsum([len(n) for n in bcs.traction_nodes], dtype=np.int64)]
        ).astype(np.int32)
        num_cells = int(np.asarray(cells).shape[0])
        cell_youngs = _as_cell_array(youngs, num_cells)
        cell_poisson = _as_cell_array(poisson, num_cells)
        if cell_youngs.size or cell_poisson.size:
            # Warn on this side of the boundary: the tesseract runtime
            # redirects its endpoint's stderr into a log file, so the deck
            # writer's own warning would never reach the person solving.
            _warn_quantization(
                _plan_deck_materials(
                    youngs,
                    poisson,
                    num_cells=num_cells,
                    materials=None,
                    max_groups=MAX_MATERIAL_GROUPS,
                    tolerance=MATERIAL_GROUP_TOLERANCE,
                )[2],
                num_cells,
                stacklevel=4,
            )
        return {
            "points": jnp.asarray(points, dtype=jnp.float64),
            "cells": np.asarray(cells, dtype=np.int32),
            "fixed_nodes": fixed,
            "traction_nodes": traction_nodes,
            "traction_offsets": offsets,
            "traction_vectors": traction_vectors,
            # Exact-face targeting is a tet feature; this path is HEX8-only
            # and selects faces by node membership (empty offsets = disabled).
            "traction_faces": np.zeros((0, 3), dtype=np.int32),
            "traction_face_offsets": np.zeros(0, dtype=np.int32),
            "youngs": np.asarray(0.0 if cell_youngs.size else youngs, dtype=np.float64),
            "poisson": np.asarray(0.0 if cell_poisson.size else poisson, dtype=np.float64),
            "cell_youngs": cell_youngs,
            "cell_poisson": cell_poisson,
        }


def _scalar_or_cells(value: Any) -> Any:
    """A material property as a plain float, or passed through if per-element."""
    return value if np.ndim(value) else float(value)


def strain_energy_solve(
    mesh,
    *,
    youngs: float,
    poisson: float,
    dirichlet,
    tractions,
    points: Any = None,
    ccx: str | os.PathLike | None = None,
    backend: CalculixBackend | None = None,
):
    """Differentiable strain energy of an elastic study via CalculiX.

    The patch-resolution twin of :func:`cadjoint.fem.simulate.elastic_solve`
    for the objective-valued ccx gradient path: patches resolve exactly
    like there, the solve runs through :class:`CalculixBackend`, and the
    returned JAX scalar is differentiable w.r.t. ``points`` (compliance
    is twice the strain energy for fixed loads).

    Args:
        mesh: Hex mesh from :func:`cadjoint.fem.sdf_to_hex_mesh`.
        youngs: Young's modulus — scalar, or per-element ``(C,)`` (which the
            deck writer discretizes onto named materials; see
            :func:`write_elastic_deck`).
        poisson: Poisson ratio — scalar or per-element ``(C,)``.
        dirichlet: Fully-clamped patches (selections or predicates).
        tractions: ``(patch, vector)`` traction pairs.
        points: Optional traced override of ``mesh.points``.
        ccx: Optional explicit ccx binary path.
        backend: Optional pre-built :class:`CalculixBackend` to reuse
            (keeps the loaded tesseract warm across repeated solves).

    Returns:
        The total strain energy as a JAX scalar.
    """
    from cadjoint.fem.simulate import _face_patch, _node_patch

    bcs = ElasticBCs(
        fixed_nodes=[_node_patch(mesh, patch) for patch in dirichlet],
        traction_nodes=[_face_patch(mesh, patch) for patch, _ in tractions],
        traction_vectors=[np.asarray(vector, dtype=np.float64) for _, vector in tractions],
    )
    solve_points = mesh.points if points is None else points
    if backend is None:
        backend = CalculixBackend(ccx=ccx)
    return backend.elastic_strain_energy(
        solve_points,
        mesh.cells,
        bcs,
        youngs=_scalar_or_cells(youngs),
        poisson=_scalar_or_cells(poisson),
    )
