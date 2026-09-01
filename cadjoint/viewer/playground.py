"""Local server for the CADJOINT browser playground.

Serves the built frontend (``cadjoint/viewer/static``) and a small JSON API:

- ``GET  /api/session``  session token and the starter program
- ``POST /compile``      run the user's Python in a disposable child process
                         and return WGSL shaders plus the construction tree
- ``POST /patch``        rewrite sketch vertex literals in the user's source
- ``POST /api/mesh``     run the source again and return only the dual-contour
                         mesh edges (requested lazily by the viewer)
- ``POST /api/simulate`` mesh the scene into hexahedra and run a thermal or
                         elastic FEM solve (or just probe the face groups);
                         ``kind="study"`` runs a study the program declares
                         (``cadjoint.fem.study``) picked by ``name``
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

EXAMPLE_SOURCE = """import jax
import jax.numpy as jnp

from cadjoint import extract_parameters, functionalize
from cadjoint.construction import PolygonProfile, SketchPlane, Solid, extrude, revolve
from cadjoint.constraints import DistanceConstraint, FixedConstraint, satisfy_constraints
from cadjoint.geometry import Scalar, Vector, Vector2
from cadjoint.render import Material
from cadjoint.sdf.boolean import Union
from cadjoint.sdf.primitives import Plane

# Named dimensions and points remain editable from either the code or viewport.
wall_width = Scalar(2.2, name="wall_width")
wall_height = Scalar(1.0, name="wall_height")
base_left = Vector2(value=[-1.1, -0.7], free=True, name="base_left")
base_right = Vector2(value=[1.1, -0.7], free=True, name="base_right")
eave_right = Vector2(value=[1.1, 0.3], free=True, name="eave_right")
roof_peak = Vector2(value=[0.0, 1.0], free=True, name="roof_peak")
eave_left = Vector2(value=[-1.1, 0.3], free=True, name="eave_left")

profile = PolygonProfile(
    [base_left, base_right, eave_right, roof_peak, eave_left],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0]),
    name="house",
)

# These constraints own part of the sketch shape. Move a point, then use
# Satisfy in the sketch panel to project it back onto this system.
FixedConstraint(base_left, [-1.1, -0.7])
DistanceConstraint(base_left, base_right, wall_width)
DistanceConstraint(base_right, eave_right, wall_height)

# Named materials appear in the material browser and can be dragged onto solids.
clay = Material(name="clay", color=[0.85, 0.45, 0.12], roughness=0.35)
glass_material = Material(
    name="glass_material",
    color=[0.72, 0.86, 1.0],
    roughness=0.04,
    opacity=0.18,
    ior=1.45,
)
brass = Material(
    name="brass",
    color=[0.95, 0.78, 0.35],
    roughness=0.12,
    metallic=1.0,
    reflectivity=0.55,
)
polished_copper = Material(
    name="polished_copper",
    color=[0.9, 0.38, 0.16],
    roughness=0.18,
    metallic=0.92,
    reflectivity=0.48,
)
ground = Material(name="ground", color=[0.12, 0.14, 0.18], roughness=0.8)

# Object positions can also be parameters. The initial glass position is a
# seed; the distance constraint below drives it to `fixture_spacing`.
glass_position = Vector([2.2, -0.3, 0.35], free=True, name="glass_position")
metal_position = Vector([-1.9, -0.65, 0.0], free=True, name="metal_position")
fixture_spacing = Scalar(3.8, name="fixture_spacing")
body_depth = Scalar(0.9, free=True, name="body_depth")

body = extrude(profile, depth=body_depth, material=clay)
glass = Solid.sphere(
    radius=0.5,
    position=glass_position,
    material=glass_material,
    name="glass",
)
metal = Solid.cylinder(
    radius=0.36,
    height=0.55,
    position=metal_position,
    rotation=[1.5708, 0.0, 0.0],
    material=brass,
    name="metal",
)

FixedConstraint(metal_position, [-1.9, -0.65, 0.0])
DistanceConstraint(metal_position, glass_position, fixture_spacing)

# A second parameter-backed sketch demonstrates the revolve operator. Its
# X coordinate is radius from the local Y axis; revolving the small section
# produces the copper ring while preserving editable source points.
ring_inner_low = Vector2(value=[0.62, -0.16], free=True, name="ring_inner_low")
ring_outer_low = Vector2(value=[0.9, -0.16], free=True, name="ring_outer_low")
ring_outer_high = Vector2(value=[0.9, 0.16], free=True, name="ring_outer_high")
ring_inner_high = Vector2(value=[0.62, 0.16], free=True, name="ring_inner_high")
ring_profile = PolygonProfile(
    [ring_inner_low, ring_outer_low, ring_outer_high, ring_inner_high],
    plane=SketchPlane(origin=[0.0, 1.65, 0.15], normal=[0.0, 0.0, 1.0]),
    name="revolve section",
)
ring = revolve(ring_profile, material=polished_copper)

scene = Union(
    body,
    glass,
    metal,
    ring,
    Plane(-1.25, material=ground),
    smoothness=0.0,
)
satisfy_constraints(scene, steps=2)

# This is a real reverse-mode derivative through sketch points -> extrusion ->
# final SDF evaluation. Change the profile or depth and rerun: both
# sensitivities update in the compact AD panel above the code.
body_parameters, body_fixed, _ = extract_parameters(body)
body_sdf = functionalize(body)

def clearance_metric(parameters):
    sdf = body_sdf(parameters, body_fixed)
    side_clearance = sdf(jnp.array([1.6, 0.0, 0.0]))
    top_clearance = sdf(jnp.array([0.0, 0.0, 0.8]))
    return side_clearance + top_clearance

clearance, clearance_gradient = jax.value_and_grad(clearance_metric)(body_parameters)
differentiability_demo = {
    "pipeline": "Profile -> Extrude -> SDF",
    "metric": "two-probe clearance",
    "value": float(clearance),
    "parameter_count": len(body_parameters),
    "sensitivities": [
        {"parameter": "eave_right.x", "value": float(clearance_gradient["eave_right"][0])},
        {"parameter": "body_depth", "value": float(clearance_gradient["body_depth"])},
    ],
}
"""

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


# FEM solves cover meshing plus a PETSc-assembled solve, which can far exceed
# the ordinary compile budget.
SIMULATE_TIMEOUT_SECONDS = 180
SIMULATE_KINDS = ("probe", "thermal", "elastic", "study")
SIMULATE_BC_TYPES = ("dirichlet", "traction")
SIMULATE_MATERIAL_KEYS = ("conductivity", "source", "youngs", "poisson")
MIN_SIMULATE_RESOLUTION = 4
MAX_SIMULATE_RESOLUTION = 64


def _validate_simulate_bc(bc: Any) -> str | None:
    """Return an error message for one boundary-condition entry, or None."""
    if not isinstance(bc, dict):
        return "Each boundary condition must be an object."
    spec = bc.get("group")
    if isinstance(spec, dict):
        if spec.get("axis") not in {"x", "y", "z"} or spec.get("side") not in {"+", "-"}:
            return "A predicate spec needs `axis` of x/y/z and `side` of +/-."
    elif not (isinstance(spec, str) and len(spec) == 2 and spec[0] in "+-" and spec[1] in "xyz"):
        return "Each boundary condition needs `group` as an id like `+x` or a predicate spec."
    if bc.get("type") not in SIMULATE_BC_TYPES:
        return "Boundary condition `type` must be `dirichlet` or `traction`."
    value = bc.get("value", 0.0)
    if bc["type"] == "traction":
        if not (
            isinstance(value, (list, tuple))
            and len(value) == 3
            and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
        ):
            return "A traction boundary condition needs `value` as three numbers."
    elif not isinstance(value, (int, float)) or isinstance(value, bool):
        return "A dirichlet boundary condition needs a numeric `value`."
    return None


def simulate_source(
    request: dict[str, Any], timeout: float = SIMULATE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Validate a simulation request and run it in a disposable child process.

    ``kind="probe"`` only meshes the scene, returning the boundary surface and
    its face-group catalog for the BC UI; ``"thermal"``/``"elastic"`` also
    solve, as an ad-hoc preview from request-supplied BCs.  ``kind="study"``
    instead runs a study the scene program itself declares (a first-class
    :mod:`cadjoint.fem.study` object), picked by ``name`` — resolution,
    material, and boundary conditions all come from the declaration.  The
    worker reports a missing jax-fem extra as
    ``error_kind="fem_unavailable"``, which the HTTP layer maps to 501.
    """
    kind = request.get("kind", "probe")
    if kind not in SIMULATE_KINDS:
        allowed = ", ".join(SIMULATE_KINDS)
        return {"ok": False, "error": f"Simulation `kind` must be one of: {allowed}."}
    if kind == "study":
        name = request.get("name")
        if not isinstance(name, str) or not name.strip():
            return {
                "ok": False,
                "error": "A study simulation needs `name`: the declared study to run.",
            }
        return _run_worker(
            request.get("source"),
            "simulate",
            timeout,
            extra={"kind": kind, "name": name},
        )
    resolution = request.get("resolution", 20)
    if (
        not isinstance(resolution, int)
        or isinstance(resolution, bool)
        or not MIN_SIMULATE_RESOLUTION <= resolution <= MAX_SIMULATE_RESOLUTION
    ):
        return {
            "ok": False,
            "error": (
                f"Simulation `resolution` must be an integer from "
                f"{MIN_SIMULATE_RESOLUTION} to {MAX_SIMULATE_RESOLUTION}."
            ),
        }
    bcs = request.get("bcs", [])
    if not isinstance(bcs, list):
        return {"ok": False, "error": "Simulation `bcs` must be a list."}
    for bc in bcs:
        error = _validate_simulate_bc(bc)
        if error is not None:
            return {"ok": False, "error": error}
    material = request.get("material", {})
    if not isinstance(material, dict):
        return {"ok": False, "error": "Simulation `material` must be an object."}
    for key, value in material.items():
        if key not in SIMULATE_MATERIAL_KEYS:
            allowed = ", ".join(SIMULATE_MATERIAL_KEYS)
            return {"ok": False, "error": f"Material key `{key}` is not one of: {allowed}."}
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return {"ok": False, "error": f"Material `{key}` must be a number."}
    return _run_worker(
        request.get("source"),
        "simulate",
        timeout,
        extra={"kind": kind, "resolution": resolution, "bcs": bcs, "material": material},
    )


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
            handlers = {
                "/compile": lambda payload: compile_source(payload.get("source")),
                "/api/mesh": lambda payload: mesh_source(payload.get("source")),
                "/api/simulate": simulate_source,
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
