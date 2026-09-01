"""Local server for the CADJOINT browser playground.

Serves the built frontend (``cadjoint/viewer/static``) and a small JSON API:

- ``GET  /api/session``  session token and the starter program
- ``POST /compile``      run the user's Python in a disposable child process
                         and return WGSL shaders plus the construction tree
- ``POST /patch``        rewrite sketch vertex literals in the user's source
- ``POST /api/mesh``     run the source again and return only the dual-contour
                         mesh edges (requested lazily by the viewer)
- ``POST /api/simulate`` run a study the program declares (``cadjoint.fem``)
                         picked by ``name`` (``kind="study"``); the response
                         carries the solved surface, the result summary, and
                         the built mesh's inspection report
- ``POST /api/mesh_inspect`` build a declared ``SimMesh`` (or a study's
                         implicit mesh) and return its inspection report plus
                         a renderable surface with a scaled-jacobian quality
                         heatmap — look at a mesh before solving on it
- ``POST /api/optimize`` run an ``Optimization`` the program declares
                         (``cadjoint.optimize``), picked by ``name``. The
                         response STREAMS as chunked NDJSON: one
                         ``{"event": "progress", ...}`` line per optimizer
                         step, then ``{"event": "done", ...}`` carrying the
                         full result — the optimized free-parameter values
                         patched back into ``source`` exactly like
                         ``/patch``, plus a ``simulate`` block (the solved
                         optimized design) for study-backed runs
- ``GET  /api/scenes``   list saved scene files in ``./scenes``
- ``POST /api/scenes/load``  read one saved scene file
- ``POST /api/scenes/save``  write one scene file into ``./scenes``

Everything is loopback-only and token-gated. ``/compile`` and ``/api/mesh``
execute the editor's Python on this machine — only run code you trust.
``/patch`` is pure text surgery and never executes anything. Scene files are
confined to the ``scenes`` directory under the server's working directory.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import subprocess
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cadjoint.viewer._patch import OPERATIONS, PatchError, apply_operation

DEFAULT_PORT = 8765
MAX_SOURCE_BYTES = 100_000
COMPILE_TIMEOUT_SECONDS = 20
STATIC_ROOT = Path(__file__).resolve().parent / "static"

CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
}

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; "
    "img-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)

EXAMPLE_SOURCE = '''"""Parametric heat sink: finned extrusion, copper slug, press-fit bushings.

A compact power-module heat sink and the default tour of the toolchain: the
fin comb is one parameter-backed sketch profile extruded through a named
depth, the copper heat slug under the die is a revolved section, and two
steel bushings carry the mounting screws. A named SimMesh discretizes the
sink, the declared thermal study conducts the die's heat flux up into the
fins on it, and the single declared optimization at the bottom descends that
SAME simulation — peak temperature against a material-volume penalty —
differentiably, straight through the geometry the viewport renders, with
the mesher and the solver each crossing a Tesseract boundary.

Named design parameters:
  - ``fin_depth``: extrusion depth of the fin comb (along y)
  - ``base_width``: driving dimension across the base deck
  - ``bushing_spacing``: distance between the two mounting bushings

The comb sketch keeps twelve meaningful design freedoms under its
constraints — fin depth, and per fin its tip height, root width, and
tip width (taper and lean emerge; side walls carry no verticality),
plus symmetric outer-fin spacing and deck thickness; everything else is
a relation (horizontal/mirror) or pinned (base span, the slug's die
interface).
"""

import jax
import jax.numpy as jnp

from cadjoint import extract_parameters, functionalize
from cadjoint.constraints import (
    DistanceConstraint,
    EqualLengthConstraint,
    FixedConstraint,
    HorizontalConstraint,
    VerticalConstraint,
    satisfy_constraints,
)
from cadjoint.construction import PolygonProfile, SketchPlane, Solid, extrude, revolve
from cadjoint.fem import Dirichlet, HeatFlux, Nodes, SimMesh, ThermalStudy
from cadjoint.geometry import Scalar, Vector, Vector2
from cadjoint.optimize import Optimization
from cadjoint.render import Material
from cadjoint.sdf.boolean import Union

# ── design parameters ────────────────────────────────────────────────────────
# fin_depth is genuinely free (the optimizer moves it); the named scalars
# are driving dimensions: constraints hold the sketch to them, so editing a
# value here re-dimensions the part without freeing it to drift.
fin_depth = Scalar(1.2, free=True, name="fin_depth")
base_width = Scalar(1.8, name="base_width")
bushing_spacing = Scalar(1.56, name="bushing_spacing")

aluminum = Material(name="aluminum", color=[0.8, 0.82, 0.85], roughness=0.3, metallic=0.9)
copper = Material(name="copper", color=[0.9, 0.45, 0.22], roughness=0.18, metallic=0.95)
steel = Material(name="steel", color=[0.55, 0.57, 0.6], roughness=0.4, metallic=0.85)

# ── fin comb: base deck + three fins as one sketch profile ───────────────────
# Sketch plane normal +Y gives in-plane axes u = -X, v = +Z: profile y is
# world height. Extrusion spans ±fin_depth/2 around y = 0. Every vertex is a
# live sketch point — drag a fin tip in the viewport and rerun.
# The initial comb is DELIBERATELY overbuilt — brick-thick fins wasting
# material — so running cool-sink visibly slims and tapers it into a
# well-proportioned sink instead of nudging an already-decent one.
base_l = Vector2(value=[-0.9, 0.0], free=True, name="base_l")
base_r = Vector2(value=[0.9, 0.0], free=True, name="base_r")
deck_r = Vector2(value=[0.9, 0.18], free=True, name="deck_r")
fin1_root_r = Vector2(value=[0.75, 0.18], free=True, name="fin1_root_r")
fin1_tip_r = Vector2(value=[0.75, 0.85], free=True, name="fin1_tip_r")
fin1_tip_l = Vector2(value=[0.45, 0.85], free=True, name="fin1_tip_l")
fin1_root_l = Vector2(value=[0.45, 0.18], free=True, name="fin1_root_l")
fin2_root_r = Vector2(value=[0.15, 0.18], free=True, name="fin2_root_r")
fin2_tip_r = Vector2(value=[0.15, 0.85], free=True, name="fin2_tip_r")
fin2_tip_l = Vector2(value=[-0.15, 0.85], free=True, name="fin2_tip_l")
fin2_root_l = Vector2(value=[-0.15, 0.18], free=True, name="fin2_root_l")
fin3_root_r = Vector2(value=[-0.45, 0.18], free=True, name="fin3_root_r")
fin3_tip_r = Vector2(value=[-0.45, 0.85], free=True, name="fin3_tip_r")
fin3_tip_l = Vector2(value=[-0.75, 0.85], free=True, name="fin3_tip_l")
fin3_root_l = Vector2(value=[-0.75, 0.18], free=True, name="fin3_root_l")
deck_l = Vector2(value=[-0.9, 0.18], free=True, name="deck_l")
comb_profile = PolygonProfile(
    [
        base_l,
        base_r,
        deck_r,
        fin1_root_r,
        fin1_tip_r,
        fin1_tip_l,
        fin1_root_l,
        fin2_root_r,
        fin2_tip_r,
        fin2_tip_l,
        fin2_root_l,
        fin3_root_r,
        fin3_tip_r,
        fin3_tip_l,
        fin3_root_l,
        deck_l,
    ],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="fin comb",
)
sink = extrude(comb_profile, depth=fin_depth, material=aluminum)

# The comb is a production-style constrained sketch that still keeps real
# design freedom. Relations — horizontal/vertical squaring, symmetric
# margins about the base — shape HOW the comb may move; the pinned base
# span anchors it. Each fin keeps its OWN tip height and root width (the
# optimizer sizes all three individually), and fin side walls carry no
# verticality constraint — root and tip widths move independently, so
# TAPER and lean are free to emerge per fin, alongside deck thickness,
# symmetric outer-fin spacing, and fin_depth.
# Every descent step is projected back onto this system (see
# cadjoint.optimize).
FixedConstraint(base_l, [-0.9, 0.0])
DistanceConstraint(base_l, base_r, base_width)
HorizontalConstraint(base_l, base_r)
VerticalConstraint(base_r, deck_r)
VerticalConstraint(base_l, deck_l)
HorizontalConstraint(deck_l, deck_r)
HorizontalConstraint(fin1_root_r, deck_r)
HorizontalConstraint(fin1_root_l, deck_r)
HorizontalConstraint(fin2_root_r, deck_r)
HorizontalConstraint(fin2_root_l, deck_r)
HorizontalConstraint(fin3_root_r, deck_r)
HorizontalConstraint(fin3_root_l, deck_r)
HorizontalConstraint(fin1_tip_r, fin1_tip_l)
HorizontalConstraint(fin2_tip_r, fin2_tip_l)
HorizontalConstraint(fin3_tip_r, fin3_tip_l)
EqualLengthConstraint(base_l, fin3_root_l, base_r, fin1_root_r)
EqualLengthConstraint(base_l, fin2_root_l, base_r, fin2_root_r)

# ── copper heat slug: revolved section under the die, screw bore on axis ─────
# Revolve spins the profile around the plane's local Y axis (world z here):
# profile x is radius, profile y runs along the axis. The slug presses into
# the deck from below; the die contacts its bottom face.
#
# Design rule: boundary-condition regions sit on pinned geometry. The slug
# bottom carries the die's heat-flux BC, so its outline is NOT free — the
# optimizer shapes fins and depth, never the chip interface. (The fin-top
# Dirichlet region below is anchored generously instead: its threshold sits
# far under the tips, so the height freedom cannot move the fins out of it.)
slug_bore_low = Vector2(value=[0.05, -0.18], name="slug_bore_low")
slug_rim_low = Vector2(value=[0.26, -0.18], name="slug_rim_low")
slug_rim_high = Vector2(value=[0.26, 0.04], name="slug_rim_high")
slug_bore_high = Vector2(value=[0.05, 0.04], name="slug_bore_high")
slug_profile = PolygonProfile(
    [slug_bore_low, slug_rim_low, slug_rim_high, slug_bore_high],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="slug section",
)
slug = revolve(slug_profile, material=copper)

# ── mounting bushings: fixed pattern, spacing tied by a constraint ───────────
# Standard press-fit parts: their radius/height are pinned Scalars (not
# free), so optimization never resizes catalog hardware.
bushing_a = Vector([0.78, 0.0, 0.1], free=True, name="bushing_a")
bushing_b = Vector([-0.78, 0.0, 0.1], free=True, name="bushing_b")
FixedConstraint(bushing_a, [0.78, 0.0, 0.1])
DistanceConstraint(bushing_a, bushing_b, bushing_spacing)
bush_a = Solid.cylinder(
    radius=Scalar(0.07), height=Scalar(0.12), position=bushing_a, material=steel, name="bush_a"
)
bush_b = Solid.cylinder(
    radius=Scalar(0.07), height=Scalar(0.12), position=bushing_b, material=steel, name="bush_b"
)

scene = Union(sink, slug, bush_a, bush_b, smoothness=0.03)
satisfy_constraints(scene, steps=2)

# ── simulation mesh: the sink volume on a named grid ─────────────────────────
# First-class meshing intent: the study below solves on it, the viewer
# inspects it (counts, bounds, element quality), and the optimization
# refreezes it as the design moves. method="tet10" is the quality path —
# boundary-conforming quadratic tets from the dual-contoured surface
# (method="hex" is the fast voxelize+snap alternative).
sink_mesh = SimMesh(
    name="sink-mesh",
    resolution=(18, 13, 11),
    bounds=(-1.05, -0.8, -0.3),
    size=(2.1, 1.6, 1.4),
    method="tet10",
)

# ── thermal study: die flux on the slug bottom, ambient at the fin field ─────
# Node selections are programmatic: the flux enters through the boundary
# faces of the slug's bottom disc; the upper fin field is held at ambient
# (an idealized convection sink).
heat_study = ThermalStudy(
    name="sink-conduction",
    conductivity=2.0,
    bcs=[
        HeatFlux(
            Nodes.halfspace([0.0, 0.0, -0.12], [0.0, 0.0, -1.0])
            & Nodes.sphere([0.0, 0.0, -0.18], 0.4),
            6.0,
        ),
        Dirichlet(Nodes.halfspace([0.0, 0.0, 0.45], [0.0, 0.0, 1.0]), 0.0),
    ],
    mesh=sink_mesh,
)

# The regularizer is a real reverse-mode derivative path through sketch
# points -> extrusion -> final SDF evaluation: the (smoothed) aluminum
# volume of the fin comb as a function of the free parameters above.
sink_parameters, sink_fixed, _ = extract_parameters(sink)
sink_sdf = functionalize(sink)

axes = [jnp.linspace(-1.0, 1.0, 15), jnp.linspace(-0.7, 0.7, 15), jnp.linspace(-0.05, 0.95, 15)]
cells = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
cell_volume = float((2.0 / 14) * (1.4 / 14) * (1.0 / 14))


def material_volume(parameters):
    sdf = sink_sdf(parameters, sink_fixed)
    return cell_volume * jnp.sum(jax.nn.sigmoid(-sdf(cells) / 0.03))


# The declared optimization closes the loop end to end: the thermal study
# above becomes the objective. Per step, the frozen simulation mesh follows
# the design differentiably (node positions re-projected through the traced
# SDF), the study solves on it, and the PEAK temperature descends against
# the material-volume regularizer while every update projects back onto the
# sketch constraints — run it from the viewer and the optimized part
# arrives with its temperature field attached, values written back here.
# (Peak, not mean: the die is the hot spot, and a mean-temperature objective
# degenerately rewards deleting hot material instead of cooling the chip.)
cool_sink = Optimization(
    name="cool-sink",
    study="sink-conduction",
    metric="max",
    regularizer=material_volume,
    regularizer_weight=0.4,
    steps=12,
    learning_rate=0.004,
    # Run the loop through two Tesseracts: the tetfill mesher (TetGen behind
    # an exact pass-through VJP) feeding the jax-fem solver tesseract.  The
    # dual contouring upstream stays differentiable in JAX against the true
    # SDF, which is what keeps the fin creases sharp.  "direct" runs the
    # same objective fully in-process.
    gradient_path="tesseract-dc",
)
'''

MISSING_BUILD_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>CADJOINT Playground</title></head>
<body style="font: 15px/1.6 system-ui; max-width: 46rem; margin: 4rem auto; padding: 0 1.5rem">
<h1>Frontend build missing</h1>
<p>The playground UI has not been built into
<code>cadjoint/viewer/static</code>. Build it with:</p>
<pre style="background:#f4f4f2;padding:1rem;border-radius:8px">cd frontend
npm install
npm run build</pre>
<p>Or run the Vite dev server (<code>npm run dev</code>) alongside this server.</p>
</body></html>
"""


def _run_worker(
    source: str, mode: str, timeout: float, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run one compile-worker request in a disposable child process."""
    if not isinstance(source, str):
        return {"ok": False, "error": "Source must be a string."}
    if len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        return {
            "ok": False,
            "error": f"Source is larger than the {MAX_SOURCE_BYTES:,}-byte limit.",
        }

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "cadjoint.viewer._compile_worker"],
            input=json.dumps({**(extra or {}), "source": source, "mode": mode}),
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"Compilation exceeded the {timeout:g}-second timeout.",
        }

    if completed.returncode != 0:
        detail = completed.stderr.strip() or "The compiler process exited unexpectedly."
        return {"ok": False, "error": detail[-8_000:]}
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return {"ok": False, "error": f"Invalid compiler response:\n{detail[-8_000:]}"}
    if not isinstance(result, dict):
        return {"ok": False, "error": "Invalid compiler response."}
    return result


def compile_source(source: str, timeout: float = COMPILE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Compile playground source in a disposable child process."""
    return _run_worker(source, "compile", timeout)


def mesh_source(source: str, timeout: float = COMPILE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Extract only the dual-contour mesh edges, in a disposable child process.

    Mesh extraction dominates a full compile, so the viewer requests it lazily
    through this path only while a mesh overlay is actually turned on.
    """
    return _run_worker(source, "mesh", timeout)


# FEM solves cover meshing plus the assembled solve, which can far exceed the
# ordinary compile budget.
SIMULATE_TIMEOUT_SECONDS = 180
SIMULATE_KINDS = ("study",)
# Mesh inspection only builds the hex mesh (no solve), but big grids still
# outgrow the compile budget.
MESH_INSPECT_TIMEOUT_SECONDS = 60


def simulate_source(
    request: dict[str, Any], timeout: float = SIMULATE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Validate a simulation request and run it in a disposable child process.

    ``kind="study"`` (the only kind) runs a study the scene program itself
    declares (a first-class :mod:`cadjoint.fem.study` object), picked by
    ``name`` — mesh, material, and boundary conditions all come from the
    declaration.  With ``cached=True`` the worker serves the study's
    ``last_result`` without re-solving when one exists; every request runs a
    fresh worker process, so that cache only ever holds a result the scene
    program computed itself (a module-level ``solve()``) — it is per worker
    process, never shared across requests.  The worker reports a missing
    jax-fem extra as ``error_kind="fem_unavailable"``, which the HTTP layer
    maps to 501.
    """
    kind = request.get("kind", "study")
    if kind not in SIMULATE_KINDS:
        return {
            "ok": False,
            "error": (
                "Simulation `kind` must be `study`: declare a ThermalStudy/ElasticStudy "
                "in the program and run it by name."
            ),
        }
    name = request.get("name")
    if not isinstance(name, str) or not name.strip():
        return {
            "ok": False,
            "error": "A study simulation needs `name`: the declared study to run.",
        }
    cached = request.get("cached", False)
    if not isinstance(cached, bool):
        return {"ok": False, "error": "Simulation `cached` must be a boolean."}
    return _run_worker(
        request.get("source"),
        "simulate",
        timeout,
        extra={"kind": kind, "name": name, "cached": cached},
    )


def mesh_inspect_source(
    request: dict[str, Any], timeout: float = MESH_INSPECT_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Build one declared SimMesh in a disposable child process and report it.

    ``name`` picks a declared :class:`cadjoint.fem.SimMesh` by name — or a
    declared study, whose mesh (explicit or implicit) is built instead.  With
    no ``name``, a single declared mesh (or, failing that, a single study) is
    used.  The response carries the JSON inspection summary (``info``), a
    renderable boundary surface (``mesh``), and the per-vertex
    scaled-jacobian quality field (``quality_scalars``) so the viewer can
    show a mesh-quality heatmap before anything is solved.
    """
    name = request.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        return {"ok": False, "error": "`name` must be a non-empty string when given."}
    extra = {"name": name} if name is not None else {}
    return _run_worker(request.get("source"), "mesh_inspect", timeout, extra=extra)


# Optimizations run the differentiable objective once per step, which can far
# exceed the compile budget; the step cap keeps one request's work bounded.
# Study-backed optimizations additionally run a full FEM solve (plus its
# adjoint) per step, so the timeout covers a panel-sized study run and the
# worker enforces a tighter measured per-run step cap for them
# (``STUDY_OPTIMIZE_STEP_LIMIT`` in the compile worker).
OPTIMIZE_TIMEOUT_SECONDS = 300
OPTIMIZE_MAX_STEPS = 200


def _validate_optimize_request(request: dict[str, Any]) -> tuple[dict | None, dict[str, Any]]:
    """``(error, extra)`` — request-shape validation shared by both entry points."""
    name = request.get("name")
    if not isinstance(name, str) or not name.strip():
        return {
            "ok": False,
            "error": "An optimization run needs `name`: the declared Optimization to run.",
        }, {}
    extra: dict[str, Any] = {"name": name}
    steps = request.get("steps")
    if steps is not None:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            return {"ok": False, "error": "Optimization `steps` must be a positive integer."}, {}
        if steps > OPTIMIZE_MAX_STEPS:
            return {
                "ok": False,
                "error": f"Optimization `steps` is capped at {OPTIMIZE_MAX_STEPS} per request.",
            }, {}
        extra["steps"] = steps
    source = request.get("source")
    if not isinstance(source, str):
        return {"ok": False, "error": "Source must be a string."}, {}
    if len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        return {
            "ok": False,
            "error": f"Source is larger than the {MAX_SOURCE_BYTES:,}-byte limit.",
        }, {}
    return None, extra


def _stream_optimize_worker(source: str, extra: dict[str, Any], timeout: float):
    """Run the optimize worker and yield its NDJSON events as they arrive.

    The worker prints one flushed ``{"event": "progress", ...}`` line per
    optimizer step and its ordinary result object as the final line; this
    generator tails the pipe, relays the progress events immediately, and
    closes with one ``{"event": "done", ...result}`` event.  A watchdog
    kills the worker at *timeout* and reports it as a clear error event.
    """
    import threading

    process = subprocess.Popen(
        [sys.executable, "-m", "cadjoint.viewer._compile_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr_text: list[str] = []
    drain = threading.Thread(target=lambda: stderr_text.append(process.stderr.read()), daemon=True)
    drain.start()
    timed_out = threading.Event()

    def _kill() -> None:
        timed_out.set()
        process.kill()

    watchdog = threading.Timer(timeout, _kill)
    watchdog.start()
    final: dict[str, Any] | None = None
    try:
        process.stdin.write(json.dumps({**extra, "source": source, "mode": "optimize"}))
        process.stdin.close()
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # stray non-JSON output must not corrupt the stream
            if not isinstance(event, dict):
                continue
            if event.get("event") == "progress":
                yield event
            else:
                final = event
        process.wait()
    finally:
        watchdog.cancel()
        drain.join(timeout=2)
        if process.poll() is None:  # client went away mid-run: stop the work
            process.kill()
    if timed_out.is_set():
        yield {
            "event": "done",
            "ok": False,
            "error": f"Optimization exceeded the {timeout:g}-second timeout.",
        }
        return
    if final is None or process.returncode != 0:
        detail = (stderr_text[0].strip() if stderr_text and stderr_text[0] else "") or (
            "The optimizer process exited unexpectedly."
        )
        yield {"event": "done", "ok": False, "error": detail[-8_000:]}
        return
    yield {"event": "done", **final}


def optimize_source_events(request: dict[str, Any], timeout: float = OPTIMIZE_TIMEOUT_SECONDS):
    """Yield the NDJSON event stream for one optimize request.

    ``/api/optimize`` responses are chunked NDJSON: one
    ``{"event": "progress", "step", "steps", "objective", "grad_norm",
    "elapsed"}`` line per optimizer step as it completes (``step`` counts
    finished evaluations, 1-based), then exactly one final
    ``{"event": "done", ...}`` line carrying the entire ordinary response
    object — ``ok``/``source``/``history``/``trajectory``/``parameters``
    (plus ``simulate`` for study-backed runs), or ``ok: false`` with
    ``error`` (and ``error_kind`` such as ``fem_unavailable``).  Request
    validation failures arrive the same way, as an immediate ``done``.
    """
    error, extra = _validate_optimize_request(request)
    if error is not None:
        yield {"event": "done", **error}
        return
    yield from _stream_optimize_worker(request["source"], extra, timeout)


def optimize_source(
    request: dict[str, Any], timeout: float = OPTIMIZE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Validate an optimize request and run it in a disposable child process.

    Runs an :class:`cadjoint.optimize.Optimization` the scene program itself
    declares, picked by ``name`` — objective, target object, and optimizer
    settings all come from the declaration; the optional ``steps`` overrides
    the declared step count (both are capped at ``OPTIMIZE_MAX_STEPS``;
    study-backed runs are further capped by the worker's measured
    ``STUDY_OPTIMIZE_STEP_LIMIT``).  The worker writes the optimized
    free-parameter values back into the program text through the patch
    machinery, so the response's ``source`` is the patched program — the
    client adopts it and recompiles, exactly like a ``/patch`` response.
    A study-backed optimization's response additionally carries a
    ``simulate`` block (``field``/``mesh``/``result``/``mesh_info``): the
    optimized design solved on a fresh mesh, in the exact shapes
    ``/api/simulate`` responses use.  A missing jax-fem extra surfaces as
    ``error_kind="fem_unavailable"``.

    This is the buffered form of :func:`optimize_source_events` (the
    ``/api/optimize`` endpoint streams the events instead): progress events
    are consumed and the final ``done`` event is returned as a plain
    response dict.
    """
    final: dict[str, Any] = {}
    for event in optimize_source_events(request, timeout):
        if event.get("event") == "done":
            final = {key: value for key, value in event.items() if key != "event"}
    return final


# Saved scenes live in one directory under the server's working directory.
# Requests supply bare ``*.py`` names only; anything resembling a path is
# rejected before it touches the filesystem.
SCENES_DIRNAME = "scenes"
MAX_SCENE_NAME_LENGTH = 128
_SCENE_STEM = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._ -]*")


def scenes_root() -> Path:
    """Directory holding saved scene files (created lazily on first save)."""
    return Path.cwd() / SCENES_DIRNAME


def sanitize_scene_name(name: Any) -> str | None:
    """Validate a scene file name, or return None if it is unacceptable.

    Accepts bare file names such as ``bracket.py``. Path separators, traversal
    (``../evil.py``), hidden files, and non-``.py`` suffixes are all refused.
    """
    if not isinstance(name, str) or not name.endswith(".py"):
        return None
    if len(name) > MAX_SCENE_NAME_LENGTH:
        return None
    if "/" in name or "\\" in name or "\x00" in name:
        return None
    stem = name[: -len(".py")]
    if not _SCENE_STEM.fullmatch(stem):
        return None
    return name


def _scene_path(name: str) -> Path | None:
    """Resolve a sanitized name inside the scenes directory, or None."""
    candidate = (scenes_root() / name).resolve()
    try:
        candidate.relative_to(scenes_root().resolve())
    except ValueError:
        return None
    return candidate


def list_scenes() -> dict[str, Any]:
    """List saved scene files as bare names, newest directory state wins."""
    root = scenes_root()
    if not root.is_dir():
        return {"ok": True, "files": []}
    names = sorted(
        path.name for path in root.glob("*.py") if path.is_file() and sanitize_scene_name(path.name)
    )
    return {"ok": True, "files": names}


def load_scene(request: dict[str, Any]) -> dict[str, Any]:
    """Read one saved scene file: ``{"name"}`` → ``{"source"}``."""
    name = sanitize_scene_name(request.get("name"))
    if name is None:
        return {"ok": False, "error": "Scene `name` must be a bare `*.py` file name."}
    path = _scene_path(name)
    if path is None or not path.is_file():
        return {"ok": False, "error": f"No saved scene named {name!r}."}
    if path.stat().st_size > MAX_SOURCE_BYTES:
        return {"ok": False, "error": f"{name!r} is larger than the source limit."}
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {"ok": False, "error": f"Could not read {name!r}."}
    return {"ok": True, "name": name, "source": source}


def save_scene(request: dict[str, Any]) -> dict[str, Any]:
    """Write one scene file: ``{"name", "source"}`` → ``{"name"}``."""
    name = sanitize_scene_name(request.get("name"))
    if name is None:
        return {"ok": False, "error": "Scene `name` must be a bare `*.py` file name."}
    source = request.get("source")
    if not isinstance(source, str):
        return {"ok": False, "error": "The save request must contain a string `source` field."}
    if len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        return {"ok": False, "error": f"Source is larger than the {MAX_SOURCE_BYTES:,}-byte limit."}
    path = _scene_path(name)
    if path is None:
        return {"ok": False, "error": "Scene `name` must stay inside the scenes directory."}
    try:
        scenes_root().mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    except OSError as error:
        return {"ok": False, "error": f"Could not save {name!r}: {error}."}
    return {"ok": True, "name": name}


def patch_source(request: dict[str, Any]) -> dict[str, Any]:
    """Apply one viewer edit to the user's program text.

    Args:
        request: ``{"source", "op", "line", "index"}`` plus ``"xy"`` for
            operations that place a vertex.

    Returns:
        ``{"ok": True, "source": ...}`` or ``{"ok": False, "error": ...}``.
    """
    source = request.get("source")
    operation = request.get("op")
    if not isinstance(source, str):
        return {"ok": False, "error": "The patch request must contain a string `source` field."}
    if len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        return {"ok": False, "error": f"Source is larger than the {MAX_SOURCE_BYTES:,}-byte limit."}
    if not isinstance(operation, str):
        return {"ok": False, "error": "The patch request must contain a string `op` field."}
    if operation not in OPERATIONS:
        # Reject up front: otherwise an operation this server does not know
        # falls through to the vertex-edit checks and complains about a missing
        # `line`, which points nowhere near the real problem — usually a browser
        # running newer assets than the server process.
        return {
            "ok": False,
            "error": (
                f"This server does not support the patch operation {operation!r}. "
                "If you updated cadjoint, restart the playground server."
            ),
        }

    def numbers(value, count: int | None = None) -> list[float] | None:
        """Validate a list of plain numbers, optionally of a fixed length."""
        if not isinstance(value, (list, tuple)):
            return None
        if count is not None and len(value) != count:
            return None
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return None
        return [float(item) for item in value]

    arguments: dict[str, Any] = {}

    if operation == "add_sketch":
        origin = numbers(request.get("origin"), 3)
        if origin is None:
            return {"ok": False, "error": "The patch request needs `origin` as three numbers."}
        try:
            return {"ok": True, "source": apply_operation(source, operation, origin=origin)}
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation == "add_primitive":
        kind = request.get("kind")
        if not isinstance(kind, str):
            return {"ok": False, "error": "The patch request needs a string `kind`."}
        position = numbers(request.get("position"), 3)
        if position is None:
            return {"ok": False, "error": "The patch request needs `position` as three numbers."}
        raw = request.get("dimensions")
        if not isinstance(raw, dict):
            return {"ok": False, "error": "The patch request needs a `dimensions` object."}
        dimensions: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                dimensions[key] = float(value)
                continue
            vector = numbers(value)
            if vector is None:
                return {"ok": False, "error": f"Dimension `{key}` must be a number or numbers."}
            dimensions[key] = vector
        arguments = {"kind": kind, "position": position, "dimensions": dimensions}
        try:
            return {"ok": True, "source": apply_operation(source, operation, **arguments)}
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation == "add_material":
        color = numbers(request.get("color"), 3)
        if color is None or any(value < 0.0 or value > 1.0 for value in color):
            return {
                "ok": False,
                "error": "The patch request needs `color` as three numbers from 0 to 1.",
            }
        properties: dict[str, float] = {}
        ranges = {
            "roughness": (0.0, 1.0, 0.4),
            "metallic": (0.0, 1.0, 0.0),
            "opacity": (0.0, 1.0, 1.0),
            "ior": (1.0, 3.0, 1.45),
            "reflectivity": (0.0, 1.0, 0.0),
        }
        for key, (low, high, default) in ranges.items():
            raw = request.get(key, default)
            if (
                not isinstance(raw, (int, float))
                or isinstance(raw, bool)
                or not low <= float(raw) <= high
            ):
                return {
                    "ok": False,
                    "error": f"The patch request needs `{key}` from {low:g} to {high:g}.",
                }
            properties[key] = float(raw)
        try:
            return {
                "ok": True,
                "source": apply_operation(
                    source,
                    operation,
                    color=color,
                    **properties,
                ),
            }
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation == "assign_material":
        line = request.get("line")
        material = request.get("material")
        if not isinstance(line, int) or isinstance(line, bool):
            return {"ok": False, "error": "The patch request needs an integer `line`."}
        if not isinstance(material, str) or not material.isidentifier():
            return {
                "ok": False,
                "error": "The patch request needs `material` as a Python identifier.",
            }
        try:
            return {
                "ok": True,
                "source": apply_operation(
                    source,
                    operation,
                    line=line,
                    material=material,
                ),
            }
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation == "add_extrusion":
        line = request.get("line")
        depth = request.get("depth", 0.5)
        if not isinstance(line, int) or isinstance(line, bool):
            return {"ok": False, "error": "The patch request needs an integer `line`."}
        if not isinstance(depth, (int, float)) or isinstance(depth, bool):
            return {"ok": False, "error": "The patch request needs a numeric `depth`."}
        try:
            return {
                "ok": True,
                "source": apply_operation(source, operation, line=line, depth=float(depth)),
            }
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation == "add_constraint":
        line = request.get("line")
        kind = request.get("kind")
        indices = request.get("indices")
        valued_kinds = {"fixed": 1, "distance": 2}
        relational_kinds = {
            "horizontal": 2,
            "vertical": 2,
            "coincident": 2,
            "parallel": 4,
            "perpendicular": 4,
        }
        if not isinstance(line, int) or isinstance(line, bool):
            return {"ok": False, "error": "The patch request needs an integer `line`."}
        if kind not in valued_kinds and kind not in relational_kinds:
            allowed = ", ".join(sorted({**valued_kinds, **relational_kinds}))
            return {"ok": False, "error": f"Constraint `kind` must be one of: {allowed}."}
        arity = valued_kinds.get(kind) or relational_kinds[kind]
        if not (
            isinstance(indices, list)
            and len(indices) == arity
            and all(isinstance(index, int) and not isinstance(index, bool) for index in indices)
        ):
            return {"ok": False, "error": f"`{kind}` takes exactly {arity} integer `indices`."}
        value = None
        if kind in valued_kinds:
            raw_value = request.get("value")
            scalar = (
                float(raw_value)
                if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool)
                else None
            )
            vector = numbers(raw_value)
            if scalar is None and vector is None:
                return {"ok": False, "error": "The constraint needs a numeric `value`."}
            value = scalar if scalar is not None else vector
        try:
            return {
                "ok": True,
                "source": apply_operation(
                    source,
                    operation,
                    line=line,
                    kind=kind,
                    indices=indices,
                    value=value,
                ),
            }
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation in {"delete_constraint", "set_constraint_value"}:
        line = request.get("line")
        index = request.get("index")
        if not isinstance(line, int) or isinstance(line, bool):
            return {"ok": False, "error": "The patch request needs an integer `line`."}
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            return {"ok": False, "error": "The patch request needs a non-negative `index`."}
        arguments = {"line": line, "index": index}
        if operation == "set_constraint_value":
            raw_value = request.get("value")
            scalar = (
                float(raw_value)
                if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool)
                else None
            )
            vector = numbers(raw_value)
            if scalar is None and vector is None:
                return {"ok": False, "error": "The constraint needs a numeric `value`."}
            arguments["value"] = scalar if scalar is not None else vector
        try:
            return {"ok": True, "source": apply_operation(source, operation, **arguments)}
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation == "add_revolution":
        line = request.get("line")
        offset = request.get("offset", 0.0)
        if not isinstance(line, int) or isinstance(line, bool):
            return {"ok": False, "error": "The patch request needs an integer `line`."}
        if not isinstance(offset, (int, float)) or isinstance(offset, bool):
            return {"ok": False, "error": "The patch request needs a numeric `offset`."}
        try:
            return {
                "ok": True,
                "source": apply_operation(source, operation, line=line, offset=float(offset)),
            }
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation == "add_loft":
        line_a = request.get("line_a")
        line_b = request.get("line_b")
        height = request.get("height", 1.0)
        if not all(
            isinstance(value, int) and not isinstance(value, bool) for value in (line_a, line_b)
        ):
            return {"ok": False, "error": "The patch request needs integer `line_a` and `line_b`."}
        if not isinstance(height, (int, float)) or isinstance(height, bool):
            return {"ok": False, "error": "The patch request needs a numeric `height`."}
        try:
            return {
                "ok": True,
                "source": apply_operation(
                    source, operation, line_a=line_a, line_b=line_b, height=float(height)
                ),
            }
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation == "solve_sketch":
        line = request.get("line")
        if not isinstance(line, int) or isinstance(line, bool):
            return {"ok": False, "error": "The patch request needs an integer `line`."}
        method = request.get("method", "newton")
        iterations = request.get("iterations", 8)
        if method not in {"newton", "adam", "sgd"}:
            return {
                "ok": False,
                "error": "Solver `method` must be `newton`, `adam`, or `sgd`.",
            }
        if (
            not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or not 1 <= iterations <= 512
        ):
            return {
                "ok": False,
                "error": "Solver `iterations` must be an integer from 1 to 512.",
            }
        try:
            return {
                "ok": True,
                "source": apply_operation(
                    source,
                    operation,
                    line=line,
                    method=method,
                    iterations=iterations,
                ),
            }
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation == "set_value":
        for key in ("name", "argument"):
            value = request.get(key)
            if not isinstance(value, str):
                return {"ok": False, "error": f"The patch request needs a string `{key}`."}
            arguments[key] = value
        line = request.get("line")
        if not isinstance(line, int) or isinstance(line, bool):
            return {"ok": False, "error": "The patch request needs an integer `line`."}
        arguments["line"] = line
        raw_value = request.get("value")
        scalar = (
            float(raw_value)
            if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool)
            else None
        )
        vector = numbers(raw_value)
        if scalar is None and vector is None:
            return {"ok": False, "error": "The patch request needs `value` as a number or numbers."}
        if arguments["argument"] in {"planeOrigin", "planeNormal"}:
            if vector is None or len(vector) != 3:
                return {
                    "ok": False,
                    "error": "A sketch-plane edit needs `value` as three numbers.",
                }
            if arguments["argument"] == "planeNormal" and not any(
                abs(component) > 1e-9 for component in vector
            ):
                return {"ok": False, "error": "A sketch-plane normal must not be zero."}
        arguments["value"] = scalar if scalar is not None else vector
        try:
            return {"ok": True, "source": apply_operation(source, operation, **arguments)}
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation == "delete_object":
        line = request.get("line")
        if not isinstance(line, int) or isinstance(line, bool):
            return {"ok": False, "error": "The patch request needs an integer `line`."}
        try:
            return {"ok": True, "source": apply_operation(source, operation, line=line)}
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation == "add_study":
        kind = request.get("kind")
        if kind not in {"thermal", "elastic"}:
            return {"ok": False, "error": "Study `kind` must be `thermal` or `elastic`."}
        name = request.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            return {"ok": False, "error": "Study `name` must be a non-empty string."}
        try:
            return {"ok": True, "source": apply_operation(source, operation, kind=kind, name=name)}
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation in {"delete_study", "add_study_bc", "delete_study_bc", "set_study_value"}:
        study = request.get("study")
        if not (
            (isinstance(study, int) and not isinstance(study, bool) and study >= 0)
            or (isinstance(study, str) and study.strip())
        ):
            return {
                "ok": False,
                "error": "The patch request needs `study` as a name or a non-negative index.",
            }
        arguments = {"study": study}

        if operation == "add_study_bc":
            bc_type = request.get("bc_type")
            if bc_type not in {"dirichlet", "heat_flux", "fixed", "traction"}:
                return {
                    "ok": False,
                    "error": ("`bc_type` must be one of: dirichlet, heat_flux, fixed, traction."),
                }
            selection = request.get("selection")
            if not isinstance(selection, dict):
                return {
                    "ok": False,
                    "error": "The patch request needs `selection` as a description object.",
                }
            arguments.update(bc_type=bc_type, selection=selection)
            raw_value = request.get("value")
            if bc_type == "fixed":
                if raw_value is not None:
                    return {"ok": False, "error": "A `fixed` boundary condition takes no value."}
            elif bc_type == "traction":
                vector = numbers(raw_value, 3)
                if vector is None:
                    return {
                        "ok": False,
                        "error": "A `traction` boundary condition needs `value` as three numbers.",
                    }
                arguments["value"] = vector
            else:
                if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
                    return {
                        "ok": False,
                        "error": f"A `{bc_type}` boundary condition needs a numeric `value`.",
                    }
                arguments["value"] = float(raw_value)

        if operation == "delete_study_bc":
            bc = request.get("bc")
            if not isinstance(bc, int) or isinstance(bc, bool) or bc < 0:
                return {"ok": False, "error": "The patch request needs a non-negative `bc` index."}
            arguments["bc"] = bc

        if operation == "set_study_value":
            bc = request.get("bc")
            argument = request.get("argument")
            if (bc is None) == (argument is None):
                return {
                    "ok": False,
                    "error": "The patch request needs exactly one of `bc` or `argument`.",
                }
            if bc is not None:
                if not isinstance(bc, int) or isinstance(bc, bool) or bc < 0:
                    return {
                        "ok": False,
                        "error": "The patch request needs a non-negative `bc` index.",
                    }
                arguments["bc"] = bc
            else:
                if not isinstance(argument, str):
                    return {"ok": False, "error": "The patch request needs a string `argument`."}
                arguments["argument"] = argument
            raw_value = request.get("value")
            if argument in {"mesh", "domain"}:
                # These reference declared objects by name, not numbers.
                if not isinstance(raw_value, str) or not raw_value.strip():
                    return {
                        "ok": False,
                        "error": f"The patch request needs `value` as a `{argument}` name.",
                    }
                arguments["value"] = raw_value
            else:
                scalar = (
                    float(raw_value)
                    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool)
                    else None
                )
                vector = numbers(raw_value)
                if scalar is None and vector is None:
                    return {
                        "ok": False,
                        "error": "The patch request needs `value` as a number or numbers.",
                    }
                arguments["value"] = scalar if scalar is not None else vector

        try:
            return {"ok": True, "source": apply_operation(source, operation, **arguments)}
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation == "add_mesh":
        name = request.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            return {"ok": False, "error": "Mesh `name` must be a non-empty string."}
        try:
            return {"ok": True, "source": apply_operation(source, operation, name=name)}
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation in {"delete_mesh", "set_mesh_value"}:
        mesh = request.get("mesh")
        if not (
            (isinstance(mesh, int) and not isinstance(mesh, bool) and mesh >= 0)
            or (isinstance(mesh, str) and mesh.strip())
        ):
            return {
                "ok": False,
                "error": "The patch request needs `mesh` as a name or a non-negative index.",
            }
        arguments = {"mesh": mesh}

        if operation == "set_mesh_value":
            argument = request.get("argument")
            if not isinstance(argument, str):
                return {"ok": False, "error": "The patch request needs a string `argument`."}
            arguments["argument"] = argument
            raw_value = request.get("value")
            if argument == "domain":
                if not isinstance(raw_value, str) or not raw_value.strip():
                    return {
                        "ok": False,
                        "error": "The patch request needs `value` as a `domain` name.",
                    }
                arguments["value"] = raw_value
            elif argument == "method":
                if raw_value not in {"hex", "tet4", "tet10"}:
                    return {
                        "ok": False,
                        "error": "Mesh `method` must be one of: hex, tet4, tet10.",
                    }
                arguments["value"] = raw_value
            else:
                scalar = (
                    float(raw_value)
                    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool)
                    else None
                )
                vector = numbers(raw_value)
                if scalar is None and vector is None:
                    return {
                        "ok": False,
                        "error": "The patch request needs `value` as a number or numbers.",
                    }
                arguments["value"] = scalar if scalar is not None else vector

        try:
            return {"ok": True, "source": apply_operation(source, operation, **arguments)}
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    if operation in {"delete_optimization", "set_optimization_value"}:
        optimization = request.get("optimization")
        valid_index = (
            isinstance(optimization, int)
            and not isinstance(optimization, bool)
            and optimization >= 0
        )
        if not (valid_index or (isinstance(optimization, str) and optimization.strip())):
            return {
                "ok": False,
                "error": (
                    "The patch request needs `optimization` as a name or a non-negative index."
                ),
            }
        arguments = {"optimization": optimization}

        if operation == "set_optimization_value":
            argument = request.get("argument")
            if argument not in {"steps", "learning_rate"}:
                return {
                    "ok": False,
                    "error": "Optimization `argument` must be `steps` or `learning_rate`.",
                }
            raw_value = request.get("value")
            if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
                return {"ok": False, "error": "The patch request needs a numeric `value`."}
            arguments.update(argument=argument, value=raw_value)

        try:
            return {"ok": True, "source": apply_operation(source, operation, **arguments)}
        except PatchError as error:
            return {"ok": False, "error": str(error)}

    for key in ("line", "index"):
        value = request.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            return {"ok": False, "error": f"The patch request needs an integer `{key}`."}
        arguments[key] = value
    if operation in {"set_vertex", "insert_vertex"}:
        xy = numbers(request.get("xy"), 2)
        if xy is None:
            return {"ok": False, "error": "The patch request needs `xy` as two numbers."}
        arguments["xy"] = (xy[0], xy[1])

    try:
        return {"ok": True, "source": apply_operation(source, operation, **arguments)}
    except PatchError as error:
        return {"ok": False, "error": str(error)}


def _is_loopback_host(host_header: str | None) -> bool:
    if not host_header:
        return False
    hostname = urlsplit(f"//{host_header}").hostname
    return hostname in {"127.0.0.1", "localhost", "::1"}


def resolve_static(url_path: str) -> Path | None:
    """Map a URL path to a file inside the static root, or None if unavailable.

    Refuses anything that escapes the static directory, so ``..`` traversal and
    absolute paths cannot reach the rest of the filesystem.
    """
    relative = url_path.lstrip("/") or "index.html"
    candidate = (STATIC_ROOT / relative).resolve()
    try:
        candidate.relative_to(STATIC_ROOT)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def make_handler(token: str):
    """Build a request handler bound to one session token."""

    class PlaygroundHandler(BaseHTTPRequestHandler):
        server_version = "cadjoint-playground"
        protocol_version = "HTTP/1.1"

        def _send(self, status: HTTPStatus, body: bytes, content_type: str, **headers: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            self._send(
                status,
                json.dumps(payload).encode("utf-8"),
                "application/json; charset=utf-8",
                Cache_Control="no-store",
            )

        def _stream_ndjson(self, events) -> None:
            """Relay an event stream as chunked NDJSON, flushed per line.

            HTTP/1.1 chunked framing written by hand: each event becomes one
            ``<hex size>\\r\\n<json line>\\r\\n`` chunk pushed to the socket
            immediately, so the client sees progress while the worker still
            runs.  The stream always ends with the zero-length chunk; a
            client that disconnects mid-run just terminates the generator
            (its ``finally`` stops the worker).
            """
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for event in events:
                    chunk = (json.dumps(event) + "\n").encode("utf-8")
                    self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                    self.wfile.write(chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def _reject(self, status: HTTPStatus, message: str) -> None:
            """Refuse a request without reading its body.

            Closes the connection so any unread body cannot be mistaken for the
            next request on a keep-alive connection.
            """
            self.close_connection = True
            self._send_json(status, {"ok": False, "error": message})

        def _host_allowed(self) -> bool:
            if _is_loopback_host(self.headers.get("Host")):
                return True
            self._reject(HTTPStatus.FORBIDDEN, "Invalid Host header.")
            return False

        def _token_valid(self) -> bool:
            supplied = self.headers.get("X-Cadjoint-Token", "")
            if secrets.compare_digest(supplied, token):
                return True
            self._reject(HTTPStatus.FORBIDDEN, "Invalid session token.")
            return False

        def _read_json(self) -> dict[str, Any] | None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_SOURCE_BYTES * 2:
                self._reject(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size.")
                return None
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON request."}
                )
                return None
            if not isinstance(payload, dict):
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON request."}
                )
                return None
            return payload

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._host_allowed():
                return
            path = urlsplit(self.path).path

            if path == "/api/session":
                # Same-origin read only: a cross-origin page cannot see this
                # response, so the token still gates POSTs against CSRF.
                self._send_json(
                    HTTPStatus.OK, {"ok": True, "token": token, "example": EXAMPLE_SOURCE}
                )
                return

            if path == "/api/scenes":
                self._send_json(HTTPStatus.OK, list_scenes())
                return

            target = resolve_static(path)
            if target is None:
                if path == "/":
                    self._send(
                        HTTPStatus.OK,
                        MISSING_BUILD_PAGE.encode("utf-8"),
                        "text/html; charset=utf-8",
                        Cache_Control="no-store",
                    )
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            body = target.read_bytes()
            content_type = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
            is_entry = target.name == "index.html"
            self._send(
                HTTPStatus.OK,
                body,
                content_type,
                Cache_Control="no-store" if is_entry else "public, max-age=31536000, immutable",
                Content_Security_Policy=CONTENT_SECURITY_POLICY,
            )

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._host_allowed():
                return
            path = urlsplit(self.path).path

            if path == "/api/optimize":
                # Streamed: progress events per optimizer step, then `done`
                # with the full response — see optimize_source_events.
                if not self._token_valid():
                    return
                payload = self._read_json()
                if payload is None:
                    return
                self._stream_ndjson(optimize_source_events(payload))
                return

            handlers = {
                "/compile": lambda payload: compile_source(payload.get("source")),
                "/api/mesh": lambda payload: mesh_source(payload.get("source")),
                "/api/simulate": simulate_source,
                "/api/mesh_inspect": mesh_inspect_source,
                "/patch": patch_source,
                "/api/scenes/load": load_scene,
                "/api/scenes/save": save_scene,
            }
            handler = handlers.get(path)
            if handler is None:
                self._reject(HTTPStatus.NOT_FOUND, "Not found.")
                return
            if not self._token_valid():
                return
            payload = self._read_json()
            if payload is None:
                return

            result = handler(payload)
            if result.get("ok"):
                status = HTTPStatus.OK
            elif result.get("error_kind") == "fem_unavailable":
                # A missing optional solver extra is "not implemented here",
                # not a bad request; the UI shows it as an install hint.
                status = HTTPStatus.NOT_IMPLEMENTED
            else:
                status = HTTPStatus.UNPROCESSABLE_ENTITY
            self._send_json(status, result)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}")

    return PlaygroundHandler


def create_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Create a loopback-only playground server."""
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(token))
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="local port (default: 8765)")
    parser.add_argument("--open", action="store_true", help="open the playground in your browser")
    args = parser.parse_args()

    server = create_server(args.port)
    address, port = server.server_address
    url = f"http://{address}:{port}/"
    print(f"CADJOINT playground: {url}")
    if not (STATIC_ROOT / "index.html").is_file():
        print("Frontend build missing — run `npm install && npm run build` in frontend/.")
    print("Local source is executed as Python in a timed child process. Only run code you trust.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping playground.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
