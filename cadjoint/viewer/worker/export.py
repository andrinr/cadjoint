"""``POST /api/export``: one object of the program, as a file.

The playground already extracts every scene it shows — dual contouring for
the mesh overlay, the derived B-rep for its sharp edges, a hex or tet mesh
for a study — and :mod:`cadjoint.meshing.export`, the ``step_export``
plugin kind and :meth:`cadjoint.fem.result.SimulationResult.to_vtk` can
each already write what those passes produce.  This module is the seam between them: a
request names an object and a :class:`~cadjoint.enums.ExportFormat`, the
worker extracts and writes it, and the response is the file.

This module is the worker's half, :func:`export_scene`: execute the
program, resolve the named object, extract on a lattice sized to the
object, and write.  It also holds the vocabulary both halves share — which
format takes which options, how the file is named, what content type it
travels under.  The server's half, which runs this in a child process and
hands the bytes back, is :mod:`cadjoint.viewer._export_client`.

**Which STEP writer.**  Two exist, and one of them is the private tier's.
:func:`cadjoint.meshing.export.save_step` — public, always available —
facets a dual-contour mesh: it recovers flat regions by merging coplanar
quads under an angle threshold and writes a ``PLANE`` per region, with
every curved surface left as triangles.  The ``step_export`` plugin kind
starts from the derived B-rep's ownership graph instead: a face there *is*
one patch's zero set, so a plane is exact with its boundary loops (holes
included) collapsed onto the real intersection curves, a full cylindrical
band is a ``CYLINDRICAL_SURFACE`` with ``CIRCLE`` edges, and only what the
graph cannot certify (blend faces) is faceted.  That writer's output reads
back into OCCT as one valid closed solid with the analytic volume; the
faceted writer's is approximate by construction.

The export asks for the ``step_export`` kind and takes the faceted writer
when the kind is unfilled (no ``diff-brep``) or when the graph cannot be
derived for an object — so a STEP file is **always** produced, and
``report["path"]`` says which writer produced it (``"brep"`` or
``"mesh"``), with ``report["tier"]`` carrying :func:`cadjoint.tier.message`
when the analytic writer was asked for and the private tier is absent.
``analytic=False`` asks for the faceted writer outright, which is the
comparison baseline the tests use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cadjoint.enums import ExportFormat, PluginKind, listed
from cadjoint.viewer._limits import OVERSIZED_SOURCE_ERROR, exceeds_source_limit
from cadjoint.viewer.schema.requests import ExportRequest

__all__ = [
    "EXPORT_CONTENT_TYPES",
    "EXPORT_EXTENSIONS",
    "export_filename",
    "export_scene",
    "validate_export_request",
]


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
    """A uniform lattice fitted to the object, at whatever scale it is modelled.

    The bounds come from :func:`cadjoint.meshing.bounds.fit_bounds`, which
    scans growing cubes until one holds the part, so a 100 mm part
    modelled in millimetres is not clipped to the ±3 default volume and a
    plate thinner than the scan spacing is still found (issue #49).
    ``resolution`` counts cells along the longest axis and the other two
    axes take proportionally fewer, which keeps the spacing isotropic — a
    cube of cells over a flat plate would spend most of its budget on the
    thin axis.
    """
    import numpy as np

    from cadjoint.meshing import GridSpec
    from cadjoint.meshing.bounds import fit_bounds

    bounds, size = fit_bounds(sdf)
    extent = np.asarray(size, dtype=np.float64)
    cells = tuple(int(max(1, round(resolution * axis / float(extent.max())))) for axis in extent)
    return GridSpec.from_bounds(bounds, size, cells)


def _write_geometry(obj: Any, request: ExportRequest, path: Path) -> dict[str, Any]:
    """Extract one SDF object and write it as OBJ, STL or STEP."""
    import warnings

    import jax.numpy as jnp

    from cadjoint.meshing.bounds import boundary_inside
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
        # The analytic path: exact planes and cylinders, faceted where the
        # graph cannot certify a face.  It is the private tier's, so it is
        # asked for by kind and the faceted writer below is the fallback —
        # see the module docstring.
        from cadjoint import tier

        exporter = tier.component(PluginKind.STEP_EXPORT)
        if exporter is None:
            report["tier"] = tier.message(PluginKind.STEP_EXPORT)
        else:
            try:
                written = exporter.step_export(obj, grid, path)
            except Exception as error:  # noqa: BLE001 - the faceted writer is the fallback
                report["fallback"] = f"{type(error).__name__}: {error}"[:500]
            else:
                report["path"] = "brep"
                report["faces"] = written.get("faces", {})
                return report

    with warnings.catch_warnings():
        # Clipping is measured below, on the field itself: the extractor's
        # warning fires only where the surface crosses the boundary, not
        # when the lattice lies wholly inside the part.
        warnings.filterwarnings("ignore", message="The isosurface crosses the extraction boundary")
        mesh = extract_mesh(sdf, grid)
    clipped = boundary_inside(sdf, grid)
    if mesh.faces.shape[0] == 0:
        raise ValueError(
            f"The export lattice holds no surface of {request.name!r}: it lies wholly "
            + ("inside" if clipped else "outside")
            + " the part."
        )
    report["clipped"] = clipped
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

    from cadjoint.capture import one_named
    from cadjoint.viewer.worker.scene import _FEM_UNAVAILABLE_MESSAGE

    study = one_named(studies, request.name, "study")
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

    from cadjoint.viewer.worker.scene import _execute_scene, captured_output

    with captured_output() as output:
        namespace = _execute_scene(parsed.source)
        if parsed.format is ExportFormat.VTK:
            result = _write_result(namespace["scene"], namespace["__studies__"], parsed, path)
        else:
            report = _write_geometry(_named_object(namespace, parsed.name), parsed, path)
            result = {"ok": True, "report": report}
    result["output"] = output()
    return result
