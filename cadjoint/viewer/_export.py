"""``POST /api/export``: one object of the program, as a file.

The playground already extracts every scene it shows — dual contouring for
the mesh overlay, the derived B-rep for its sharp edges, a hex or tet mesh
for a study — and :mod:`cadjoint.meshing.export`, :mod:`cadjoint.brep.step`
and :meth:`cadjoint.fem.result.SimulationResult.to_vtk` can each already
write what those passes produce.  This module is the seam between them: a
request names an object and a :class:`~cadjoint.enums.ExportFormat`, the
worker extracts and writes it, and the response is the file.

The two halves live together because they share the vocabulary — which
format takes which options, how the file is named, what content type it
travels under:

- **In the server** (:func:`export_source`): validate the request against
  :class:`~cadjoint.viewer.schema.requests.ExportRequest`, run the worker
  with a temporary path to write to, and hand the bytes back with their
  filename and content type.  The route in :mod:`cadjoint.viewer.playground`
  registers the run as an ``export`` job, so the Processes window sees it
  and can cancel it, and sends the bytes as an attachment.
- **In the worker** (:func:`export_scene`): execute the program, resolve
  the named object, extract on a lattice sized to the object, and write.

**Which STEP writer.**  Two exist.  :func:`cadjoint.meshing.export.save_step`
facets a dual-contour mesh — it recovers flat regions by merging coplanar
quads under an angle threshold and writes a ``PLANE`` per region, with
every curved surface left as triangles.  :func:`cadjoint.brep.step.save_brep_step`
starts from the ownership graph instead: a face there *is* one patch's
zero set, so a plane is exact with its boundary loops (holes included)
collapsed onto the real intersection curves, a full cylindrical band is a
``CYLINDRICAL_SURFACE`` with ``CIRCLE`` edges, and only what the graph
cannot certify (blend faces) is faceted.  The graph writer's output reads
back into OCCT as one valid closed solid with the analytic volume
(``tests/brep/test_step_kernel.py``); the mesh writer's is approximate by
construction.  The export takes the graph path, on the same
:func:`~cadjoint.brep.extract_brep` pass the viewer's edge overlay runs on
every scene, and falls back to the faceted writer when the graph cannot be
derived for an object — so a STEP file is always produced, and the report
says which path produced it.  ``analytic=False`` asks for the faceted
writer outright, which is the comparison baseline the tests use.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cadjoint.enums import ExportFormat, listed
from cadjoint.viewer._limits import OVERSIZED_SOURCE_ERROR, exceeds_source_limit
from cadjoint.viewer.schema.requests import ExportRequest

__all__ = [
    "EXPORT_CONTENT_TYPES",
    "EXPORT_EXTENSIONS",
    "EXPORT_TIMEOUT_SECONDS",
    "export_filename",
    "export_scene",
    "export_source",
    "validate_export_request",
]

# A geometry export is one lattice sweep plus, for STEP, the graph
# extraction the mesh overlay pays for every scene (~17 s warm on the
# starter at the overlay's resolution, and the request may ask for finer).
# A VTK export is a full solve, so it takes the simulate budget instead
# (see :func:`export_source`).  Either is visible and cancellable as a job.
EXPORT_TIMEOUT_SECONDS = 300

#: The file extension each format is written under.
EXPORT_EXTENSIONS: dict[ExportFormat, str] = {
    ExportFormat.OBJ: "obj",
    ExportFormat.STL: "stl",
    ExportFormat.STEP: "step",
    ExportFormat.VTK: "vtk",
}

#: What the response is served as.  OBJ and STL have IANA ``model/`` types;
#: STEP's registered type is ``model/step``, which browsers do not know, so
#: the download is what matters and ``application/step`` is what CAD tools
#: send.  Legacy VTK has no registered type at all.
EXPORT_CONTENT_TYPES: dict[ExportFormat, str] = {
    ExportFormat.OBJ: "model/obj",
    ExportFormat.STL: "model/stl",
    ExportFormat.STEP: "application/step",
    ExportFormat.VTK: "application/octet-stream",
}


def export_filename(name: str, fmt: ExportFormat) -> str:
    """``scene.stl`` — the object's name under the format's extension.

    The name is a Python identifier (or a study name), so the only thing to
    guard is a name that is not a safe file name at all.
    """
    stem = "".join(char if (char.isalnum() or char in "-_.") else "_" for char in name.strip())
    return f"{stem.strip('.') or 'export'}.{EXPORT_EXTENSIONS[fmt]}"


def _validation_message(error: ValidationError) -> str:
    """The first field failure, in the dialog's own words."""
    first = error.errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "request"
    if location == "format":
        return f"Export `format` must be one of: {listed(ExportFormat)}."
    return f"Export `{location}`: {first.get('msg', 'is invalid')}."


def validate_export_request(
    request: dict[str, Any],
) -> tuple[dict[str, Any] | None, ExportRequest | None]:
    """``(error, parsed)`` — the request against its model, before any work.

    Args:
        request: The raw request object.

    Returns:
        ``(None, model)`` for a valid request, or ``({"ok": False, ...},
        None)`` with the message the dialog shows.
    """
    try:
        parsed = ExportRequest.model_validate(request)
    except ValidationError as error:
        return {"ok": False, "error": _validation_message(error)}, None
    if exceeds_source_limit(parsed.source):
        return {"ok": False, "error": OVERSIZED_SOURCE_ERROR}, None
    return None, parsed


# ── the server half ─────────────────────────────────────────────────────────


def export_source(request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
    """Validate an export request and write the file in a disposable child process.

    The worker writes to a temporary path this process owns; the bytes are
    read back and the directory removed before returning, so nothing is
    left on disk and nothing large crosses the worker's JSON pipe.

    Args:
        request: The raw request body.
        timeout: The worker budget; by default :data:`EXPORT_TIMEOUT_SECONDS`,
            or the simulate budget for ``vtk`` (which solves the study).

    Returns:
        On success ``{"ok": True, "format", "name", "filename",
        "content_type", "size", "data": bytes, "report", "output"}``; on
        failure the worker's ``{"ok": False, "error"}`` (with
        ``error_kind="fem_unavailable"`` when a VTK export has no solver).
    """
    from cadjoint.viewer._worker_client import SIMULATE_TIMEOUT_SECONDS, _run_worker

    error, parsed = validate_export_request(request)
    if error is not None or parsed is None:
        return error or {"ok": False, "error": "Invalid export request."}
    if timeout is None:
        timeout = (
            SIMULATE_TIMEOUT_SECONDS
            if parsed.format is ExportFormat.VTK
            else EXPORT_TIMEOUT_SECONDS
        )
    filename = export_filename(parsed.name, parsed.format)
    with tempfile.TemporaryDirectory(prefix="cadjoint-export-") as folder:
        path = Path(folder) / filename
        extra = parsed.model_dump(mode="json", exclude={"source"})
        extra["path"] = str(path)
        result = _run_worker(parsed.source, "export", timeout, extra=extra)
        if not result.get("ok"):
            return result
        if not path.is_file():
            return {"ok": False, "error": "The export worker finished without writing a file."}
        data = path.read_bytes()
    return {
        "ok": True,
        "format": parsed.format.value,
        "name": parsed.name,
        "filename": filename,
        "content_type": EXPORT_CONTENT_TYPES[parsed.format],
        "size": len(data),
        "data": data,
        "report": result.get("report") or {},
        "output": result.get("output", ""),
    }


# ── the worker half ─────────────────────────────────────────────────────────


def _exportable_names(namespace: dict[str, Any]) -> list[str]:
    """Every variable of the program bound to an SDF, ``scene`` first."""
    from cadjoint.sdf.base import SDF

    names = [
        name
        for name, value in namespace.items()
        if not name.startswith("_") and isinstance(value, SDF)
    ]
    return sorted(names, key=lambda name: (name != "scene", name))


def _named_object(namespace: dict[str, Any], name: str) -> Any:
    """The SDF the program bound to *name*, or raise listing the alternatives."""
    from cadjoint.sdf.base import SDF

    value = namespace.get(name)
    if isinstance(value, SDF) and not name.startswith("_"):
        return value
    exportable = ", ".join(repr(candidate) for candidate in _exportable_names(namespace))
    if value is None:
        raise ValueError(
            f"The program binds no SDF object named {name!r} (exportable: {exportable or 'none'})."
        )
    raise ValueError(
        f"{name!r} is a {type(value).__name__}, not an SDF object "
        f"(exportable: {exportable or 'none'})."
    )


def _object_grid(sdf: Any, resolution: int) -> Any:
    """A uniform lattice fitted to the object's inside region.

    The bounds come from the same coarse scan a ``SimMesh`` without explicit
    bounds uses, so an exported part and its simulation mesh agree on where
    the part is.  ``resolution`` counts cells along the longest axis and
    the other two axes take proportionally fewer, which keeps the spacing
    isotropic — a cube of cells over a flat plate would spend most of its
    budget on the thin axis.
    """
    import numpy as np

    from cadjoint.fem.simmesh import _scan_bounds
    from cadjoint.meshing import GridSpec

    bounds, size = _scan_bounds(sdf, padding=0.0)
    extent = np.asarray(size, dtype=np.float64)
    cells = tuple(int(max(1, round(resolution * axis / float(extent.max())))) for axis in extent)
    return GridSpec.from_bounds(bounds, size, cells)


def _write_geometry(obj: Any, request: ExportRequest, path: Path) -> dict[str, Any]:
    """Extract one SDF object and write it as OBJ, STL or STEP."""
    import warnings

    import jax.numpy as jnp

    from cadjoint.meshing.dual_contouring import extract_mesh
    from cadjoint.meshing.export import save_obj, save_step, save_stl

    sdf = lambda p: jnp.asarray(obj(p))  # noqa: E731
    grid = _object_grid(sdf, request.resolution)
    report: dict[str, Any] = {
        "resolution": list(grid.cells),
        "bounds": list(grid.origin),
        "size": [spacing * count for spacing, count in zip(grid.spacing, grid.cells)],
    }

    if request.format is ExportFormat.STEP and request.analytic:
        # The graph path: exact planes and cylinders, faceted where the graph
        # cannot certify a face.  See the module docstring for why this
        # writer and not the mesh one.
        from cadjoint.brep import extract_brep, save_brep_step

        try:
            brep = extract_brep(obj, grid)
            written = save_brep_step(brep, path)
        except Exception as error:  # noqa: BLE001 - the faceted writer is the fallback
            report["fallback"] = f"{type(error).__name__}: {error}"[:500]
        else:
            report["path"] = "brep"
            report["faces"] = written.get("faces", {})
            return report

    with warnings.catch_warnings():
        # The lattice is fitted to the object with a margin, so the open
        # boundary warning can only mean a scene that fills the default
        # scan volume (a ground plane); clipping it is the point.
        warnings.filterwarnings("ignore", message="The isosurface crosses the extraction boundary")
        mesh = extract_mesh(sdf, grid)
    report["vertices"] = int(mesh.vertices.shape[0])
    report["triangles"] = int(mesh.faces.shape[0])
    if request.format is ExportFormat.OBJ:
        save_obj(mesh, path, merge_planar=request.merge_planar)
    elif request.format is ExportFormat.STL:
        save_stl(mesh, path, binary=request.binary)
    else:
        save_step(mesh, path)
        report["path"] = "mesh"
    return report


def _write_result(
    scene: Any, studies: list[Any], request: ExportRequest, path: Path
) -> dict[str, Any]:
    """Solve (or reuse) the named study and write its fields as VTK."""
    import jax.numpy as jnp

    from cadjoint.viewer._worker_scene import _FEM_UNAVAILABLE_MESSAGE, _named_study

    study = _named_study(studies, request.name)
    try:
        import jax_fem  # noqa: F401
    except ImportError:
        return {"ok": False, "error_kind": "fem_unavailable", "error": _FEM_UNAVAILABLE_MESSAGE}
    sdf = lambda p: jnp.asarray(scene(p))  # noqa: E731
    # A result the program computed itself (a module-level solve) is the
    # result the user is looking at; only solve when there is none.
    cached = study.last_result is not None
    result = study.last_result if cached else study.solve(sdf)
    result.to_vtk(str(path))
    return {"ok": True, "report": {"study": study.name, "cached": cached}}


def export_scene(request: dict[str, Any]) -> dict[str, Any]:
    """The worker's ``mode="export"``: run the program, write one file.

    ``request["path"]`` is where the server wants the file; everything else
    is an :class:`ExportRequest` field.  Returns ``{"ok": True, "report",
    "output"}`` — the bytes stay on disk for the server to collect.
    """
    # The worker protocol adds `mode` and this module adds `path`; neither
    # is a field of the request the model describes.
    fields = {key: value for key, value in request.items() if key not in ("mode", "path")}
    error, parsed = validate_export_request(fields)
    if error is not None or parsed is None:
        return error or {"ok": False, "error": "Invalid export request."}
    path = Path(str(request.get("path") or ""))
    if not path.name:
        return {"ok": False, "error": "The export worker was given no path to write to."}

    from cadjoint.viewer._worker_scene import _execute_scene

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        namespace = _execute_scene(parsed.source)
        if parsed.format is ExportFormat.VTK:
            result = _write_result(namespace["scene"], namespace["__studies__"], parsed, path)
        else:
            report = _write_geometry(_named_object(namespace, parsed.name), parsed, path)
            result = {"ok": True, "report": report}
    result["output"] = captured.getvalue()[-8_000:]
    return result
