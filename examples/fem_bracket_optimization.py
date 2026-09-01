"""End-to-end shape optimization of the L-bracket from ``scenes/bracket.py``.

The flagship demonstration of the whole differentiable chain::

    named CAD parameters -> bracket SDF -> HEX8 mesh (frozen topology,
    periodically re-extracted) -> linear elastic solve (jax-fem adjoint,
    or CalculiX's native *SENSITIVITY) -> compliance + mass objective ->
    optax Adam with box projection

Three named design parameters are optimized — ``web_thickness``,
``rib_height`` and ``plate_thickness``, the same named Scalars the scene
program declares — under box bounds that keep the geometry meshable.  The
bolt-hole regions of the base plate are clamped, a prying traction pulls the
tip of the vertical web, and the objective trades stiffness (compliance)
against material use (a smoothed volume integral of the SDF).

The discrete/continuous split drives the re-extraction schedule: which cells
are inside and how they connect is a *discrete* decision that cannot be
differentiated, so topology is frozen while the optimizer runs and only the
node positions are recomputed differentiably per candidate
(:func:`cadjoint.fem.hexmesh.recompute_points` — motion is clamped to half a
cell diagonal).  Every ``--remesh-every`` steps the topology is re-extracted
at the current design so large shape changes stay well represented; the
objective may jump slightly at those steps because the discretization
changes.

Artifacts land in ``examples/output/``: convergence history as CSV, a
convergence figure (PNG), and before/after VTK files for ParaView.

Run directly (requires the ``fem`` extra plus ``optax``)::

    python examples/fem_bracket_optimization.py
    python examples/fem_bracket_optimization.py --smoke   # 2 cheap steps
    python examples/fem_bracket_optimization.py --backend calculix

With ``--backend calculix`` the compliance term instead runs through the
CalculiX tesseract (requires the ``tesseract`` extra and a ``ccx`` binary —
see :mod:`cadjoint.fem.calculix`): the objective becomes classical
compliance (``f . u``, twice the strain energy) and its gradient flows
through ccx's native ``*SENSITIVITY`` adjoint instead of jax-fem's — a
1990s Fortran Abaqus clone one ``jax.grad`` away from the design
parameters.
"""
# Guarded third-party imports must precede the cadjoint.fem imports.
# ruff: noqa: E402

from __future__ import annotations

try:  # importorskip-style guard: this demo needs the fem extra.
    import jax_fem  # noqa: F401
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise SystemExit(
        "This example requires jax-fem (install with: pip install cadjoint[fem])."
    ) from error

try:
    import optax
except ImportError as error:  # pragma: no cover - exercised without optax
    raise SystemExit("This example requires optax (install with: pip install optax).") from error

import csv
import time
from pathlib import Path

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

# The three optimized parameters (order everywhere: theta[0], theta[1],
# theta[2]) and their nominal values — matching the named Scalars in
# ``scenes/bracket.py``.
PARAMETER_NAMES = ("web_thickness", "rib_height", "plate_thickness")
NOMINAL = (0.16, 0.88, 0.20)

# Box bounds project every Adam update back into meshable geometry: the
# lower thickness bounds stay above one grid cell so the thin walls never
# drop out of the extracted topology, and the upper bounds keep the part
# inside the sampling lattice.
LOWER_BOUNDS = (0.12, 0.35, 0.14)
UPPER_BOUNDS = (0.26, 1.15, 0.30)

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

DEFAULT_RESOLUTION = (30, 21, 16)
SMOKE_RESOLUTION = (14, 10, 8)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def bracket_sdf(p, web_thickness, rib_height, plate_thickness):
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


def theta_sdf(theta):
    """Close ``bracket_sdf`` over a parameter vector ``theta``."""

    def sdf(p):
        return bracket_sdf(p, theta[0], theta[1], theta[2])

    return sdf


def build_grid(resolution=DEFAULT_RESOLUTION) -> GridSpec:
    """Sampling lattice enclosing the bracket with a small margin."""
    return GridSpec.from_bounds((-1.3, -0.95, -0.06), (2.6, 1.9, 1.42), resolution)


def extract_topology(theta, grid: GridSpec) -> HexMesh:
    """Extract a frozen-topology HEX8 mesh at the design ``theta``."""
    return sdf_to_hex_mesh(theta_sdf(np.asarray(theta, dtype=np.float64)), grid)


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


def make_objective(mesh: HexMesh, grid: GridSpec, backend: str | None = None):
    """Objective ``theta -> (compliance + mass, aux)`` on a frozen mesh.

    With the default backend, compliance is the total squared displacement
    under the prying load (jax-fem adjoint); with ``backend="calculix"``
    it is the classical compliance ``f . u`` (twice the strain energy,
    ccx ``*SENSITIVITY`` adjoint).  Mass is a smoothed volume integral of
    the inside indicator on the (fixed) lattice of cell centers, so both
    terms are differentiable in ``theta``.  Returns ``has_aux``-style
    ``(total, {"compliance": ..., "mass": ...})``.
    """
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
        sdf = theta_sdf(theta)
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
        return compliance + _MASS_WEIGHT * mass, {"compliance": compliance, "mass": mass}

    return objective


def check_gradient(objective, theta, eps: float = 1e-5, parameters=None):
    """Compare the adjoint gradient against central finite differences.

    Both sides differentiate the same frozen-topology objective, so the
    discrepancy measures adjoint correctness (plus FD truncation/solver
    noise), not remeshing effects.

    Args:
        objective: ``theta -> (value, aux)`` objective.
        theta: Parameter vector to check at.
        eps: Central-difference half step.
        parameters: Indices to check (default: all).

    Returns:
        List of ``(name, adjoint, fd, rel_delta)`` tuples.
    """
    _, gradient = jax.value_and_grad(objective, has_aux=True)(theta)
    rows = []
    for i in parameters if parameters is not None else range(len(PARAMETER_NAMES)):
        unit = jnp.zeros_like(theta).at[i].set(1.0)
        plus, _ = objective(theta + eps * unit)
        minus, _ = objective(theta - eps * unit)
        fd = float((plus - minus) / (2.0 * eps))
        adjoint = float(gradient[i])
        rel = abs(adjoint - fd) / max(abs(fd), 1e-12)
        rows.append((PARAMETER_NAMES[i], adjoint, fd, rel))
    return rows


def export_vtk(theta, grid: GridSpec, path: Path) -> None:
    """Solve at ``theta`` on a freshly extracted mesh and write a VTU file."""
    mesh = extract_topology(theta, grid)
    result = elastic_solve(
        mesh,
        youngs=_YOUNGS,
        poisson=_POISSON,
        dirichlet=[BOLT_CLAMP],
        tractions=[(WEB_TIP_LOAD, list(_TRACTION))],
    )
    result.vtk_export(str(path))
    print(f"wrote {path}")


def write_history_csv(history, path: Path) -> None:
    """Save the convergence history as one CSV row per evaluated design."""
    fields = [
        "step",
        "objective",
        "compliance",
        "mass",
        "grad_norm",
        "proj_grad_norm",
        *PARAMETER_NAMES,
        *[f"grad_{name}" for name in PARAMETER_NAMES],
        "remeshed",
        "eval_seconds",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    print(f"wrote {path}")


def plot_convergence(history, path: Path) -> None:
    """Render the convergence figure: objective, gradient norm, parameters."""
    try:
        import matplotlib
    except ImportError:  # pragma: no cover - plotting is optional
        print("matplotlib not installed; skipping the convergence figure")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, text, muted, grid_color = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e4e0"
    series = {"web_thickness": "#2a78d6", "rib_height": "#eb6834", "plate_thickness": "#1baf7a"}
    steps = [row["step"] for row in history]
    remesh_steps = [row["step"] for row in history if row["remeshed"] and row["step"] > 0]

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(7.2, 8.0), dpi=150)
    figure.patch.set_facecolor(surface)
    titles = ("Objective (compliance + mass)", "Projected gradient norm", "Parameters")
    for axis, title in zip(axes, titles):
        axis.set_facecolor(surface)
        axis.set_title(title, loc="left", fontsize=11, color=text)
        axis.grid(True, color=grid_color, linewidth=0.8)
        axis.tick_params(colors=muted, labelsize=9)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for step in remesh_steps:
            axis.axvline(step, color="#c6c6c0", linewidth=1.0, linestyle=(0, (3, 3)), zorder=0)

    axes[0].plot(steps, [row["objective"] for row in history], color=series["web_thickness"], lw=2)
    axes[1].plot(
        steps, [row["proj_grad_norm"] for row in history], color=series["web_thickness"], lw=2
    )
    axes[1].set_yscale("log")
    for name, color in series.items():
        values = [row[name] for row in history]
        axes[2].plot(steps, values, color=color, lw=2, label=name)
        axes[2].annotate(
            name,
            (steps[-1], values[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize=8,
            color=muted,
        )
    axes[2].legend(loc="best", frameon=False, fontsize=8, labelcolor=muted)
    axes[2].set_xlabel("optimizer step (vertical lines: topology re-extraction)", color=muted)
    axes[2].set_xmargin(0.14)  # room for the direct labels
    figure.tight_layout()
    figure.savefig(path, facecolor=surface, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def print_summary(history) -> None:
    """Print a before/after summary table of the optimization."""
    first, last = history[0], history[-1]
    rows = [
        ("objective", first["objective"], last["objective"]),
        ("compliance", first["compliance"], last["compliance"]),
        ("mass", first["mass"], last["mass"]),
        *[(name, first[name], last[name]) for name in PARAMETER_NAMES],
    ]
    print(f"\n{'quantity':<16}{'initial':>12}{'final':>12}{'change':>10}")
    print("-" * 50)
    for name, before, after in rows:
        change = 100.0 * (after - before) / abs(before)
        print(f"{name:<16}{before:>12.4f}{after:>12.4f}{change:>+9.1f}%")


def run_optimization(
    steps: int = 30,
    learning_rate: float = 0.015,
    remesh_every: int = 6,
    resolution=DEFAULT_RESOLUTION,
    backend: str | None = None,
    fd_check: bool = True,
    export: bool = True,
    output_dir: Path = OUTPUT_DIR,
):
    """Projected Adam on (web thickness, rib height, plate thickness).

    Args:
        steps: Optimizer iterations.
        learning_rate: Adam learning rate (parameters are all O(0.1-1)).
        remesh_every: Re-extract the mesh topology every this many steps;
            in between, topology is frozen and only node positions move.
        resolution: Meshing lattice resolution (cells per axis).
        backend: ``None`` (jax-fem, squared-displacement compliance) or
            ``"calculix"`` (ccx adjoint, classical ``f . u`` compliance).
        fd_check: Validate the adjoint gradient against central finite
            differences at the initial design.
        export: Write CSV / PNG / before-after VTU artifacts.
        output_dir: Directory for the artifacts.

    Returns:
        ``(history, theta)`` — one dict per evaluated design, and the final
        parameter vector.
    """
    grid = build_grid(resolution)
    lower = jnp.asarray(LOWER_BOUNDS, dtype=jnp.float64)
    upper = jnp.asarray(UPPER_BOUNDS, dtype=jnp.float64)
    theta = jnp.asarray(NOMINAL, dtype=jnp.float64)

    mesh = extract_topology(theta, grid)
    print(f"mesh: {mesh.num_cells} hexes, {mesh.num_points} nodes")
    objective = make_objective(mesh, grid, backend=backend)
    value_and_grad = jax.value_and_grad(objective, has_aux=True)

    if fd_check:
        print("adjoint vs central finite differences at the initial design:")
        for name, adjoint, fd, rel in check_gradient(objective, theta):
            print(f"  d/d {name:<16} adjoint {adjoint:+.6f}  fd {fd:+.6f}  rel {rel:.2e}")

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(theta)
    history: list[dict] = []

    def record(step, value, aux, gradient, remeshed, seconds):
        # Projected gradient: at an active box bound the component pushing
        # further out is not a usable descent direction, so it is zeroed —
        # this is the norm that should shrink toward stationarity.
        blocked = ((theta <= lower) & (gradient > 0)) | ((theta >= upper) & (gradient < 0))
        projected = jnp.where(blocked, 0.0, gradient)
        entry = {
            "step": step,
            "objective": float(value),
            "compliance": float(aux["compliance"]),
            "mass": float(aux["mass"]),
            "grad_norm": float(jnp.linalg.norm(gradient)),
            "proj_grad_norm": float(jnp.linalg.norm(projected)),
            "remeshed": int(remeshed),
            "eval_seconds": round(seconds, 3),
        }
        entry.update({name: float(theta[i]) for i, name in enumerate(PARAMETER_NAMES)})
        entry.update({f"grad_{name}": float(gradient[i]) for i, name in enumerate(PARAMETER_NAMES)})
        history.append(entry)
        print(
            f"step {step:>3}: objective={entry['objective']:.6f} "
            f"(compliance={entry['compliance']:.4f} mass={entry['mass']:.4f}) "
            f"|proj grad|={entry['proj_grad_norm']:.4f} theta=("
            + ", ".join(f"{float(theta[i]):.4f}" for i in range(3))
            + (")  [remeshed]" if remeshed and step > 0 else ")")
            + f"  {seconds:.1f}s"
        )

    for step in range(steps):
        remeshed = step > 0 and remesh_every > 0 and step % remesh_every == 0
        if remeshed:
            # Discrete refresh: re-extract cells + connectivity at the current
            # design; the continuous chain restarts from the new topology.
            mesh = extract_topology(theta, grid)
            objective = make_objective(mesh, grid, backend=backend)
            value_and_grad = jax.value_and_grad(objective, has_aux=True)
        started = time.perf_counter()
        (value, aux), gradient = value_and_grad(theta)
        record(step, value, aux, gradient, remeshed, time.perf_counter() - started)
        updates, opt_state = optimizer.update(gradient, opt_state, theta)
        theta = jnp.clip(optax.apply_updates(theta, updates), lower, upper)

    # Final evaluation on a freshly extracted mesh: the number reported for
    # the optimized design does not depend on the last frozen topology.
    mesh = extract_topology(theta, grid)
    objective = make_objective(mesh, grid, backend=backend)
    started = time.perf_counter()
    (value, aux), gradient = jax.value_and_grad(objective, has_aux=True)(theta)
    record(steps, value, aux, gradient, True, time.perf_counter() - started)
    print_summary(history)

    if export:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_history_csv(history, output_dir / "fem_bracket_convergence.csv")
        plot_convergence(history, output_dir / "fem_bracket_convergence.png")
        export_vtk(
            jnp.asarray(NOMINAL, dtype=jnp.float64), grid, output_dir / "fem_bracket_before.vtu"
        )
        export_vtk(theta, grid, output_dir / "fem_bracket_after.vtu")

    return history, theta


def run_smoke() -> None:
    """Two cheap optimizer steps at low resolution; asserts descent.

    Used by ``--smoke`` and by ``examples/test_fem_bracket_optimization.py``
    to keep the flagship example from rotting: the adjoint must agree with
    finite differences and two Adam steps must not increase the objective.
    """
    history, _ = run_optimization(
        steps=2,
        remesh_every=0,
        resolution=SMOKE_RESOLUTION,
        fd_check=False,
        export=False,
    )
    grid = build_grid(SMOKE_RESOLUTION)
    theta = jnp.asarray(NOMINAL, dtype=jnp.float64)
    objective = make_objective(extract_topology(theta, grid), grid)
    (name, adjoint, fd, rel) = check_gradient(objective, theta, parameters=[0])[0]
    print(f"smoke gradient check d/d {name}: adjoint {adjoint:+.6f} fd {fd:+.6f} rel {rel:.2e}")
    assert rel < 5e-2, f"adjoint disagrees with finite differences (rel {rel:.2e})"
    descent = history[-1]["objective"] - history[0]["objective"]
    assert descent < 1e-9, f"objective did not descend over the smoke steps ({descent:+.6f})"
    print(f"smoke ok: descent {history[0]['objective']:.6f} -> {history[-1]['objective']:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        choices=("jaxfem", "calculix"),
        default="jaxfem",
        help="solver/adjoint for the compliance term (calculix needs a ccx binary)",
    )
    parser.add_argument("--steps", type=int, default=30, help="optimizer iterations")
    parser.add_argument("--lr", type=float, default=0.015, help="Adam learning rate")
    parser.add_argument(
        "--remesh-every", type=int, default=6, help="re-extract topology every N steps (0: never)"
    )
    parser.add_argument(
        "--skip-fd-check", action="store_true", help="skip the finite-difference gradient check"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="2 cheap steps at low resolution, assert descent"
    )
    arguments = parser.parse_args()
    if arguments.smoke:
        run_smoke()
    else:
        run_optimization(
            steps=arguments.steps,
            learning_rate=arguments.lr,
            remesh_every=arguments.remesh_every,
            backend=None if arguments.backend == "jaxfem" else arguments.backend,
            fd_check=not arguments.skip_fd_check,
        )
