"""Child process used by the local browser playground compiler.

The worker's entry point and stdin/stdout protocol: it reads one JSON
request object from stdin, dispatches on its ``mode``, and writes the JSON
response to stdout.  ``mode="optimize"`` additionally streams NDJSON
progress lines to that same stdout before the final response object (see
:mod:`cadjoint.viewer.worker.optimize`).  Any exception becomes an
``{"ok": false, "error": <traceback>}`` response.

``mode="compile"`` — the worker's namesake, implemented here — runs the
program and builds the shaders, construction tree, and declaration entries
the viewer opens with, then checks the whole payload against
:mod:`cadjoint.viewer.schema.payloads` before sending it: the models are
what the frontend's generated types are emitted from, so a payload that
does not match them would be a type the browser was promised and did not
get.  The other modes live beside it:
:mod:`cadjoint.viewer._edge_overlay` (``mesh``), :mod:`.fem`
(``simulate``, ``mesh_inspect``), :mod:`.optimize` (``optimize``), and
:mod:`.export` (``export``).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import traceback
from typing import Any

from cadjoint.backends.wgsl import compile_scene_to_wgsl
from cadjoint.cache import enable_compilation_cache
from cadjoint.constraints.solve import capture_constraint_solves
from cadjoint.viewer._edge_overlay import (  # noqa: F401 - re-exported for callers
    _MESH_EDGE_RESOLUTION,
    _MESH_EDGE_SIZE,
    _mesh_edge_payload,
)
from cadjoint.viewer._pathtracer import (
    WGSL_PRESENT_TEMPLATE,
    build_path_tracer_shader,
)
from cadjoint.viewer._webgpu import build_viewer_shader
from cadjoint.viewer.schema.payloads import validate_compile_payload
from cadjoint.viewer.source_map import (
    PLAYGROUND_FILENAME,
    build_construction_payload,
    build_construction_relations,
    build_material_payload,
    capture_profiles,
    describe_identities,
)
from cadjoint.viewer.source_map.elements import build_element_payload
from cadjoint.viewer.worker.declarations import (
    _mesh_entries,
    _optimization_entries,
    _study_entries,
)
from cadjoint.viewer.worker.fem import (  # noqa: F401 - re-exported for callers
    _inspect_mesh,
    _mesh_inspect_source,
    _simulate_source,
    _simulate_study,
)
from cadjoint.viewer.worker.optimize import _optimize_source
from cadjoint.viewer.worker.protocol import RETIRE_FLAG
from cadjoint.viewer.worker.scene import _execute_scene


def _mesh_source(source: str) -> dict[str, Any]:
    """Extract only the dual-contour mesh edges for the viewer.

    The mesh view is optional and expensive (it re-runs dual contouring on a
    dense grid), so the playground requests it lazily through ``/api/mesh``
    instead of paying for it on every compile.
    """
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        namespace = _execute_scene(source)
        mesh_edges = _mesh_edge_payload(namespace["scene"])
    return {
        "ok": True,
        "mesh_edges": mesh_edges,
        "output": captured.getvalue()[-8_000:],
    }


def _shader_options() -> tuple[bool, bool, str]:
    """``(uniforms, culling, scope)`` — the switches, and where they come from.

    The defaults are what the viewer ships; the environment overrides exist
    so a benchmark can measure the forms against each other without a code
    edit, and so a scene can be put back on the traced path if the direct
    one ever draws something wrong.

    * ``CADJOINT_SHADER_FORM=traced`` uses the JAX/StableHLO backend, which
      was the default until the direct one was verified in a browser.  Same
      buffer contract, same three entry points, 3.5-6.7x slower to compile
      (``scenes/motor_shield.py``: 10.5 s against 1.6 s).
    * ``CADJOINT_SHADER_FORM=literal`` folds the parameters back into the
      source, which is the form the frontend used before it learned to read
      a buffer.
    * ``CADJOINT_SHADER_CULL=0`` emits the flat field that
      :mod:`cadjoint.backends.wgsl._culling` is tested against.
    * ``CADJOINT_SHADER_SCOPE=all`` puts every parameter in the buffer
      rather than only the free ones.  This is 31x slower per frame on
      ``scenes/end_cap.py`` and exists to be measured, not to be used —
      see :func:`cadjoint.backends.wgsl.compile_scene_with_uniforms`.
    ``direct`` is the default: :mod:`cadjoint.backends.wgsl.direct` walks
    the SDF graph rather than tracing it, which is most of the compile time,
    and emits a far smaller module because profile vertices are buffer slots
    and patterns loop instead of unrolling.  A scene using a node it has no
    kernel for falls back to the traced form and says so on stdout, so
    coverage gaps cost speed rather than correctness.
    """
    uniforms = os.environ.get("CADJOINT_SHADER_FORM", "direct").lower() != "literal"
    culling = os.environ.get("CADJOINT_SHADER_CULL", "1") != "0"
    scope = os.environ.get("CADJOINT_SHADER_SCOPE", "free").lower()
    return uniforms, culling, scope


def _scene_shader(scene) -> tuple[str, dict | None]:
    """The scene's WGSL, and the uniform contract that goes with it.

    Returns:
        ``(source, program)`` — ``program`` is ``None`` in the literal forms,
            where the parameters are baked into ``source`` and any edit is a
            different module.
    """
    form = os.environ.get("CADJOINT_SHADER_FORM", "direct").lower()
    if form == "direct":
        direct = _direct_shader(scene)
        if direct is not None:
            return direct
    uniforms, culling, scope = _shader_options()
    compiled = compile_scene_to_wgsl(scene, uniforms=uniforms, culling=culling, scope=scope)
    if uniforms:
        return compiled.wgsl, compiled.as_dict()
    return compiled, None


def _direct_shader(scene) -> tuple[str, dict | None] | None:
    """The direct backend's module, or None when it cannot emit this scene.

    The free parameters go in the same uniform buffer the traced form uses,
    so a handle drag stays a buffer write and a redraw rather than a round
    trip — the module is byte-identical for every value of every free
    parameter.

    The profile vertices are inlined as a constant array, because the
    viewer's shader contract carries that one uniform buffer and no vertex
    buffer.  That keeps the module a drop-in at the cost of the storage-buffer
    size win: the vertices are still data rather than code, just data spelled
    in the module.  See ``research/performance.md`` §16.
    """
    from cadjoint.backends.wgsl.codegen import (
        DEFAULT_PARAMETER_BINDING,
        DEFAULT_PARAMETER_GROUP,
        PARAMETER_SLOT_BYTES,
        RESERVED_PARAMETER_SLOTS,
        ShaderParameter,
        ShaderProgram,
    )
    from cadjoint.backends.wgsl.direct import UnsupportedNode, compile_sdf_direct

    try:
        program = compile_sdf_direct(scene, uniforms=True)
    except UnsupportedNode as reason:
        print(f"note: the direct shader backend cannot emit this scene ({reason}); tracing it.")
        return None
    source = program.wgsl
    if program.vertices.size:
        literals = ", ".join(f"vec2<f32>({x:.9g}, {y:.9g})" for x, y in program.vertices.tolist())
        source = source.replace(
            "@group(1) @binding(0) var<storage, read> profile_vertices: array<vec2<f32>>;",
            f"const profile_vertices = array<vec2<f32>, {len(program.vertices)}>({literals});",
        )
    slots = tuple(
        ShaderParameter(
            name=name,
            offset=index * PARAMETER_SLOT_BYTES,
            components=components,
            value=tuple(values),
            free=True,
        )
        for index, (name, components, values) in enumerate(program.parameters)
    )
    # The two reserved slots are declared and written but never read: this
    # form spells no NaN constant and does no bounds culling. Keeping them
    # means the frontend writes one buffer layout, not two.
    contract = ShaderProgram(
        wgsl=source,
        parameters=slots,
        buffer_bytes=(len(slots) + RESERVED_PARAMETER_SLOTS) * PARAMETER_SLOT_BYTES,
        group=DEFAULT_PARAMETER_GROUP,
        binding=DEFAULT_PARAMETER_BINDING,
        nan_offset=len(slots) * PARAMETER_SLOT_BYTES,
        cull_margin_offset=(len(slots) + 1) * PARAMETER_SLOT_BYTES,
    )
    return source, contract.as_dict()


def _compile_source(source: str) -> dict[str, Any]:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        namespace = _execute_scene(
            source,
            capture=(
                ("__solver_reports__", capture_constraint_solves),
                ("__profiles__", lambda: capture_profiles(PLAYGROUND_FILENAME)),
            ),
        )
        solver_reports = namespace["__solver_reports__"]
        profiles = namespace["__profiles__"]
        sim_meshes = namespace["__sim_meshes__"]
        studies = namespace["__studies__"]
        optimizations = namespace["__optimizations__"]
        scene_code, program = _scene_shader(namespace["scene"])
        preview_shader = build_viewer_shader(scene_code)
        path_shader = build_path_tracer_shader(scene_code)
        shader_hash = hashlib.sha256(
            preview_shader.encode() + b"\0" + path_shader.encode()
        ).hexdigest()
        construction = build_construction_payload(profiles, source)
        relations = build_construction_relations(profiles)
        materials = build_material_payload(namespace, source)
        # Declaration only: studies, meshes, and optimizations are serialized
        # from their describe() payloads — no meshing, solving, or descending
        # happens at compile time.
        studies_payload = _study_entries(studies, source)
        sim_meshes_payload = _mesh_entries(sim_meshes, source)
        optimizations_payload = _optimization_entries(optimizations, source, namespace["scene"])
        node_ids = {
            id(obj): (f"{obj.kind}_{index}" if hasattr(obj, "kind") else f"profile_{index}")
            for index, (obj, _) in enumerate(profiles)
        }
        solver_runs = [
            {
                "node": node_ids.get(report["target_id"]),
                "method": report["method"],
                "iterations": report["iterations"],
                "losses": report["losses"],
            }
            for report in solver_reports
        ]

    return validate_compile_payload(
        {
            "ok": True,
            "sdf": scene_code,
            "shader": preview_shader,
            "scene_wgsl": scene_code,
            "preview_shader": preview_shader,
            "path_shader": path_shader,
            "present_shader": WGSL_PRESENT_TEMPLATE,
            "program": program,
            "shader_hash": shader_hash,
            "construction": construction,
            # Every stable id the text declares, so the viewer can name anything
            # the payload reports only by line.
            "identities": describe_identities(source),
            "elements": build_element_payload(source),
            "relations": relations,
            "materials": materials,
            "studies": studies_payload,
            "sim_meshes": sim_meshes_payload,
            "optimizations": optimizations_payload,
            # The mesh-edge view is requested lazily via `mode: "mesh"` — computing
            # it here used to dominate the compile round-trip.
            "mesh_edges": None,
            # Which private kinds this worker process can reach, so the
            # title block can say EDGES LATTICE / GEOMETRY FROZEN rather
            # than the frontend guessing (:mod:`cadjoint.tier`).
            "tier": _tier_flags(),
            "solver_runs": solver_runs,
            "output": captured.getvalue()[-8_000:],
        }
    )


def _tier_flags() -> dict[str, bool] | None:
    """``{kind: available}`` for the private kinds, or ``None`` if unreadable.

    Never raises: a status report that fails must not fail a compile.
    """
    try:
        from cadjoint import tier

        return tier.status().flags()
    except Exception:  # noqa: BLE001 - the payload field is optional
        return None


#: Requests one served worker answers before the client retires it.
#:
#: A bound on drift rather than on any measured leak: a process that has run
#: a hundred scenes has a hundred scenes' worth of JAX caches in it, and
#: throwing it away is cheaper than reasoning about what it still holds.
SERVE_REQUEST_LIMIT = 64


def _answer(request: dict[str, Any]) -> dict[str, Any]:
    """Run one request, turning any exception into an error response."""
    try:
        source = request.get("source")
        if not isinstance(source, str):
            raise TypeError("The compile request must contain a string `source` field.")
        mode = request.get("mode", "compile")
        if mode == "mesh":
            return _mesh_source(source)
        if mode == "simulate":
            return _simulate_source(request)
        if mode == "mesh_inspect":
            return _mesh_inspect_source(request)
        if mode == "optimize":
            return _optimize_source(request)
        if mode == "export":
            from cadjoint.viewer.worker.export import export_scene

            return export_scene(request)
        if mode == "compile":
            return _compile_source(source)
        raise ValueError(f"Unknown compile worker mode: {mode!r}.")
    except Exception:
        return {"ok": False, "error": traceback.format_exc()}


def _serve() -> None:
    """Answer NDJSON requests on stdin until EOF, one response line each.

    The reusable form of :func:`main`, for a client that keeps the process
    between requests.  Everything expensive that a fresh process throws away
    survives here — the imports, JAX's tracing caches, the compilation cache
    reads — which on ``scenes/motor_shield.py`` is most of a compile.

    **A served worker retires itself when the process is no longer neutral.**
    ``jax_enable_x64`` is process-global, a scene is arbitrary Python, and a
    scene that flips it (``scenes/duct_sink.py``'s docstring warns against
    exactly this) would silently make every later request in this process
    float64.  A disposable worker contained that; a served one must notice
    it.  So the flag is read before and after each request and a process
    that changed it exits, which the client sees as a closed pipe and
    replaces.  Same for any other config the check grows to cover: the rule
    is that a tainted process is thrown away, never repaired.
    """
    import jax

    served = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        before = jax.config.jax_enable_x64
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            response: dict[str, Any] = {"ok": False, "error": "Malformed worker request."}
        else:
            response = _answer(request)
        served += 1
        # Told, not inferred. The client cannot poll for our exit without
        # racing it, and a request sent to a process that is on its way out
        # is lost, so the last response a worker sends carries the notice.
        retiring = jax.config.jax_enable_x64 != before or served >= SERVE_REQUEST_LIMIT
        if retiring:
            response = {**response, RETIRE_FLAG: True}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
        if retiring:
            return


def main() -> None:
    # Every request runs in a fresh process, so without this each edit
    # recompiles the same XLA programs from scratch (see cadjoint.cache).
    enable_compilation_cache()
    if "--serve" in sys.argv[1:]:
        _serve()
        return
    try:
        request = json.load(sys.stdin)
        result = _answer(request)
    except Exception:
        result = {"ok": False, "error": traceback.format_exc()}
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
