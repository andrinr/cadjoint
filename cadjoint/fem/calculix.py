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

GPL note: CalculiX is GPL-2; it stays behind the subprocess boundary
(decks in, result files out — no linking).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cadjoint.fem.backends import _TESSERACT_EXTRA_MESSAGE, ElasticBCs, TesseractBackend, _x64_scope
from cadjoint.fem.hexmesh import _boundary_face_rows

__all__ = [
    "CalculixBackend",
    "CcxElasticSolution",
    "ElasticDeck",
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

# Trilinear HEX8 corner signs in reference coordinates [-1, 1]^3 (VTK order).
_CORNER_SIGNS = np.array(
    [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ],
    dtype=np.float64,
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
    s = _CORNER_SIGNS  # (8, 3)
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
    """

    text: str
    nodal_forces: np.ndarray
    design_nodes: np.ndarray | None
    num_nodes: int
    num_cells: int


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


def _nset_lines(name: str, nodes: np.ndarray) -> list[str]:
    """*NSET block with 1-based ids, 16 per line."""
    ids = (np.asarray(nodes, dtype=np.int64).reshape(-1) + 1).tolist()
    lines = [f"*NSET, NSET={name}"]
    for start in range(0, len(ids), 16):
        lines.append(", ".join(str(i) for i in ids[start : start + 16]))
    return lines


def write_elastic_deck(
    points: np.ndarray,
    cells: np.ndarray,
    bcs: ElasticBCs,
    *,
    youngs: float,
    poisson: float,
    design_nodes: np.ndarray | None = None,
) -> ElasticDeck:
    """Render a linear-elastic ccx deck for a HEX8 mesh.

    C3D8 corner order equals the VTK/meshio hexahedron order, so cells
    serialize 1:1 (ids shifted to 1-based).  Clamps become ``*BOUNDARY``
    on an ``*NSET``; tractions become ``*CLOAD`` consistent nodal forces
    on the boundary faces spanned by each patch (all four corners in the
    patch set — the same rule as the jax-fem backend).  With
    ``design_nodes`` a second ``*SENSITIVITY`` step with the STRAINENERGY
    design response is appended and ``SEN`` output is requested.

    Args:
        points: Vertex positions, ``(N, 3)``.
        cells: HEX8 connectivity, ``(C, 8)`` (VTK corner order).
        bcs: Array-level boundary conditions.
        youngs: Young's modulus.
        poisson: Poisson ratio.
        design_nodes: Optional 0-based node indices for coordinate design
            variables (must lie on the boundary surface).

    Returns:
        The rendered :class:`ElasticDeck`.
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
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

    lines += [
        "*MATERIAL, NAME=MAT0",
        "*ELASTIC",
        f"{_num(youngs)}, {_num(poisson)}",
        "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT0",
        "*STEP",
        "*STATIC",
    ]
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
    return ElasticDeck(
        text="\n".join(lines),
        nodal_forces=forces,
        design_nodes=design_nodes,
        num_nodes=int(points.shape[0]),
        num_cells=int(cells.shape[0]),
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

    Args:
        points: Vertex positions, ``(N, 3)``.
        cells: HEX8 connectivity, ``(C, 8)``.
        displacement: Nodal displacements from the forward solve, ``(N, 3)``.
        youngs: Young's modulus.
        poisson: Poisson ratio.

    Returns:
        Per-node gradient field shaped ``(N, 3)``.
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    displacement = np.asarray(displacement, dtype=np.float64)
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
    """

    displacement: np.ndarray
    strain_energy: float
    cell_stress: np.ndarray
    strain_energy_gradient: np.ndarray | None = None
    normals: np.ndarray | None = None


def elastic_ccx_solve(
    points: np.ndarray,
    cells: np.ndarray,
    bcs: ElasticBCs,
    *,
    youngs: float,
    poisson: float,
    sensitivities: bool = False,
    design_nodes: np.ndarray | None = None,
    ccx: str | os.PathLike | None = None,
    workdir: str | os.PathLike | None = None,
) -> CcxElasticSolution:
    """Run a ccx linear-elastic solve (and optionally the adjoint).

    Args:
        points: Vertex positions, ``(N, 3)``.
        cells: HEX8 connectivity, ``(C, 8)``.
        bcs: Array-level boundary conditions.
        youngs: Young's modulus.
        poisson: Poisson ratio.
        sensitivities: Also run the ``*SENSITIVITY`` step and return the
            corrected strain-energy gradient (see module docstring).
        design_nodes: Coordinate design variables for the sensitivity
            step; defaults to every boundary node.
        ccx: Optional explicit binary path.
        workdir: Optional directory to keep the deck and result files in
            (default: a temporary directory, removed afterwards).

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
    )

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
                points, cells, displacement, youngs=youngs, poisson=poisson
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

    def elastic_strain_energy(self, points, cells, bcs, *, youngs, poisson, base_points=None):
        """Differentiable total strain energy of the elastic solve.

        Returns a JAX scalar whose VJP w.r.t. ``points`` is served by the
        ccx ``*SENSITIVITY`` adjoint (normal-projected design-node
        sensitivities plus the volume-term correction).  For a linear
        problem under fixed loads, compliance is twice this value.
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
        """Pack BCs into the flat tesseract schema (mirrors the base class)."""
        import jax.numpy as jnp

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
        return {
            "points": jnp.asarray(points, dtype=jnp.float64),
            "cells": np.asarray(cells, dtype=np.int32),
            "fixed_nodes": fixed,
            "traction_nodes": traction_nodes,
            "traction_offsets": offsets,
            "traction_vectors": traction_vectors,
            "youngs": np.asarray(youngs, dtype=np.float64),
            "poisson": np.asarray(poisson, dtype=np.float64),
        }


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
        youngs: Young's modulus.
        poisson: Poisson ratio.
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
        solve_points, mesh.cells, bcs, youngs=float(youngs), poisson=float(poisson)
    )
