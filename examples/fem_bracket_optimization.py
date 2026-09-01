"""End-to-end shape optimization of the L-bracket from ``scenes/bracket.py``.

Chain: design parameters -> bracket SDF -> HEX8 mesh (frozen topology) ->
linear elastic solve (jax-fem adjoint) -> compliance + mass objective ->
gradient descent on web thickness and rib height.

The bolt-hole regions of the base plate are clamped, a prying traction pulls
the tip of the vertical web, and the objective trades stiffness (compliance)
against material use (a smoothed volume integral of the SDF).  Topology is
extracted once at the nominal design; per candidate only the node positions
are recomputed differentiably (:func:`cadjoint.fem.hexmesh.recompute_points`),
which is what lets ``jax.grad`` flow from the objective back to the design
parameters through the solver's adjoint.

Run directly (requires the ``fem`` extra)::

    python examples/fem_bracket_optimization.py

With ``--backend calculix`` the compliance term instead runs through the
CalculiX tesseract (requires the ``tesseract`` extra and a ``ccx`` binary
— see :mod:`cadjoint.fem.calculix`): the objective becomes classical
compliance (``f . u``, twice the strain energy) plus the same mass term,
and its gradient flows through ccx's native ``*SENSITIVITY`` adjoint
instead of jax-fem's — a 1990s Fortran Abaqus clone one ``jax.grad``
away from the design parameters.

Writes ``fem_bracket_before.vtu`` / ``fem_bracket_after.vtu`` for ParaView.
"""
# Guarded jax-fem import must precede the cadjoint.fem imports.
# ruff: noqa: E402

from __future__ import annotations

try:  # importorskip-style guard: this demo needs the fem extra.
    import jax_fem  # noqa: F401
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise SystemExit(
        "This example requires jax-fem (install with: pip install cadjoint[fem])."
    ) from error

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from cadjoint.fem.hexmesh import GridSpec, HexMesh, recompute_points, sdf_to_hex_mesh
from cadjoint.fem.selection import Nodes
from cadjoint.fem.simulate import elastic_solve
from cadjoint.sdf.boolean.smooth import smooth_min
from cadjoint.sdf.primitives.box import Box
from cadjoint.sdf.primitives.cylinder import Cylinder
from cadjoint.sdf.primitives.polygon import ExtrudedPolygon

jax.config.update("jax_enable_x64", True)

# Nominal design (matches scenes/bracket.py): the two optimized parameters
# are the web thickness and the rib height; the plate thickness stays fixed.
NOMINAL_WEB_THICKNESS = 0.16
NOMINAL_RIB_HEIGHT = 0.88
PLATE_THICKNESS = 0.2

# Fixed geometry shared with the scene.
_PLATE_HALF = (1.2, 0.8)
_WEB_Y = -0.7  # web mid-plane
_RIB_THICKNESS = 0.12
_BOLT_XY = ((-0.7, 0.35), (0.7, 0.35))
_BOLT_RADIUS = 0.16
_FILLET = 0.05

# Load and objective weights.  The traction acts on the outer (-y) side face
# of the web tip only, so the applied force does not scale with the web
# thickness being optimized.  The mass penalty makes "grow everything"
# non-optimal.
_TRACTION = (0.0, -2.0, 0.0)  # prying load on the web tip
_MASS_WEIGHT = 1.0
_YOUNGS = 1000.0
_POISSON = 0.3

# Per-parameter step sizes for plain gradient descent: the compliance is far
# more sensitive to the web thickness than to the rib height, so the rib gets
# a proportionally larger step (a fixed diagonal preconditioner).
_LEARNING_RATES = (1.5e-3, 4e-2)


def bracket_sdf(p, web_thickness, rib_height, plate_thickness=PLATE_THICKNESS):
    """Signed distance of the bracket, differentiable in the design parameters.

    Mirrors ``scenes/bracket.py`` with pure primitive fields: base plate box,
    trapezoidal web extruded through ``web_thickness``, triangular rib whose
    apex height is ``rib_height``, filleted union, two sharp bolt holes.

    Args:
        p: Query point(s), shape ``(..., 3)``.
        web_thickness: Extrusion depth of the vertical web.
        rib_height: Height of the gusset rib's tip above the plate.
        plate_thickness: Total thickness of the base plate.

    Returns:
        Signed distance, shape ``(...)``.
    """
    p = jnp.asarray(p)
    half_plate = plate_thickness / 2.0
    plate = Box.sdf(
        p - jnp.array([0.0, 0.0, 1.0]) * half_plate,
        jnp.stack([jnp.asarray(_PLATE_HALF[0]), jnp.asarray(_PLATE_HALF[1]), half_plate]),
    )

    # Web: profile in the local XY plane (world x, world z), extruded along
    # world y about the mid-plane y = _WEB_Y.
    q_web = jnp.stack([p[..., 0], p[..., 2], p[..., 1] - _WEB_Y], axis=-1)
    web = ExtrudedPolygon.sdf(
        q_web,
        depth=web_thickness,
        v0=jnp.array([-1.1, 0.0]),
        v1=jnp.array([1.1, 0.0]),
        v2=jnp.array([0.85, 1.2]),
        v3=jnp.array([-0.85, 1.2]),
    )

    # Rib: triangular profile in (world y, world z), extruded along world x.
    q_rib = jnp.stack([p[..., 1], p[..., 2], p[..., 0]], axis=-1)
    rib = ExtrudedPolygon.sdf(
        q_rib,
        depth=_RIB_THICKNESS,
        v0=jnp.array([0.55, 0.02]),
        v1=jnp.array([-0.62, 0.02]),
        v2=jnp.stack([jnp.asarray(-0.62), jnp.asarray(rib_height)]),
    )

    body = smooth_min(smooth_min(plate, web, _FILLET), rib, _FILLET)
    for bolt_x, bolt_y in _BOLT_XY:
        hole = Cylinder.sdf(
            p - jnp.array([bolt_x, bolt_y, half_plate]), _BOLT_RADIUS, plate_thickness
        )
        body = jnp.maximum(body, -hole)
    return body


def build_grid() -> GridSpec:
    """Sampling lattice enclosing the bracket with a small margin."""
    return GridSpec.from_bounds((-1.3, -0.95, -0.06), (2.6, 1.9, 1.42), (24, 17, 13))


def build_mesh() -> HexMesh:
    """Extract the frozen-topology HEX8 mesh at the nominal design."""

    def nominal(p):
        return bracket_sdf(p, NOMINAL_WEB_THICKNESS, NOMINAL_RIB_HEIGHT)

    return sdf_to_hex_mesh(nominal, build_grid())


# Clamped nodes: a ball around each bolt hole (through the plate thickness).
BOLT_CLAMP = Nodes.sphere([-0.7, 0.35, 0.1], 0.33) | Nodes.sphere([0.7, 0.35, 0.1], 0.33)

# Loaded nodes: the outer (-y) wall of the web above z = 1.0.  Restricting
# the patch to the outward-facing side wall keeps the loaded area — and with
# it the total applied force — independent of the web thickness the
# optimizer is changing (the traction acts on faces whose four corners are
# all selected, so inner-wall and top faces never qualify).
WEB_TIP_LOAD = Nodes.halfspace([0.0, -0.7, 0.0], [0.0, -1.0, 0.0]) & Nodes.halfspace(
    [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]
)


def make_objective(mesh: HexMesh, backend: str | None = None):
    """Objective ``theta = (web_thickness, rib_height) -> compliance + mass``.

    With the default backend, compliance is the total squared displacement
    under the prying load (jax-fem adjoint); with ``backend="calculix"``
    it is the classical compliance ``f . u`` (twice the strain energy,
    ccx ``*SENSITIVITY`` adjoint).  Mass is a smoothed volume integral of
    the inside indicator on the (fixed) lattice of cell centers, so both
    terms are differentiable in ``theta``.
    """
    grid = build_grid()
    nx, ny, nz = grid.cells
    spacing = np.asarray(grid.spacing)
    index = np.stack(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    centers = jnp.asarray(np.asarray(grid.origin) + (index + 0.5) * spacing)
    cell_volume = float(np.prod(spacing))
    sharpness = 0.5 * float(np.min(spacing))

    ccx_backend = None
    if backend == "calculix":
        from cadjoint.fem.calculix import CalculixBackend

        ccx_backend = CalculixBackend()

    def objective(theta):
        web_thickness, rib_height = theta[0], theta[1]

        def sdf(p):
            return bracket_sdf(p, web_thickness, rib_height)

        points = recompute_points(sdf, mesh)
        if ccx_backend is not None:
            from cadjoint.fem.calculix import strain_energy_solve

            compliance = 2.0 * strain_energy_solve(
                mesh,
                youngs=_YOUNGS,
                poisson=_POISSON,
                dirichlet=[BOLT_CLAMP],
                tractions=[(WEB_TIP_LOAD, list(_TRACTION))],
                points=points,
                backend=ccx_backend,
            )
        else:
            result = elastic_solve(
                mesh,
                youngs=_YOUNGS,
                poisson=_POISSON,
                dirichlet=[BOLT_CLAMP],
                tractions=[(WEB_TIP_LOAD, list(_TRACTION))],
                points=points,
            )
            compliance = jnp.sum(result.displacement**2)
        mass = cell_volume * jnp.sum(jax.nn.sigmoid(-sdf(centers) / sharpness))
        return compliance + _MASS_WEIGHT * mass

    return objective


def run_optimization(
    steps: int = 4,
    learning_rates=_LEARNING_RATES,
    export_vtk: bool = False,
    backend: str | None = None,
):
    """Gradient descent on (web thickness, rib height) with frozen topology.

    Args:
        steps: Gradient-descent iterations.
        learning_rates: Per-parameter step sizes; small so node motion stays
            well inside the frozen mesh's snap clamp.
        export_vtk: Write before/after VTK files for ParaView.
        backend: ``None`` (jax-fem, squared-displacement compliance) or
            ``"calculix"`` (ccx adjoint, classical ``f . u`` compliance).

    Returns:
        ``(history, theta)`` — a list of ``(objective, web_thickness,
        rib_height)`` per evaluated design, and the final parameters.
    """
    mesh = build_mesh()
    print(f"mesh: {mesh.num_cells} hexes, {mesh.num_points} nodes")
    objective = make_objective(mesh, backend=backend)
    value_and_grad = jax.value_and_grad(objective)
    rates = jnp.asarray(learning_rates, dtype=jnp.float64)

    theta = jnp.array([NOMINAL_WEB_THICKNESS, NOMINAL_RIB_HEIGHT], dtype=jnp.float64)
    history: list[tuple[float, float, float]] = []
    for step in range(steps):
        value, gradient = value_and_grad(theta)
        history.append((float(value), float(theta[0]), float(theta[1])))
        print(
            f"step {step}: objective={float(value):.6f} "
            f"web_thickness={float(theta[0]):.4f} rib_height={float(theta[1]):.4f} "
            f"grad=({float(gradient[0]):+.4f}, {float(gradient[1]):+.4f})"
        )
        theta = theta - rates * gradient
    final = float(objective(theta))
    history.append((final, float(theta[0]), float(theta[1])))
    print(
        f"final: objective={final:.6f} "
        f"web_thickness={float(theta[0]):.4f} rib_height={float(theta[1]):.4f}"
    )

    if export_vtk:
        for name, candidate in (("before", history[0]), ("after", history[-1])):

            def sdf(p, candidate=candidate):
                return bracket_sdf(p, candidate[1], candidate[2])

            # Bake the candidate's recomputed node positions into a concrete
            # mesh so the exported file shows the actual candidate geometry.
            points = np.asarray(recompute_points(sdf, mesh))
            candidate_mesh = dataclasses.replace(mesh, points=points)
            result = elastic_solve(
                candidate_mesh,
                youngs=_YOUNGS,
                poisson=_POISSON,
                dirichlet=[BOLT_CLAMP],
                tractions=[(WEB_TIP_LOAD, list(_TRACTION))],
            )
            path = f"fem_bracket_{name}.vtu"
            result.vtk_export(path)
            print(f"wrote {path}")

    return history, theta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        choices=("jaxfem", "calculix"),
        default="jaxfem",
        help="solver/adjoint for the compliance term (calculix needs a ccx binary)",
    )
    parser.add_argument("--steps", type=int, default=4, help="gradient-descent iterations")
    arguments = parser.parse_args()
    run_optimization(
        steps=arguments.steps,
        export_vtk=True,
        backend=None if arguments.backend == "jaxfem" else arguments.backend,
    )
