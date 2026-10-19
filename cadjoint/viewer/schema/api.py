"""Every HTTP route the playground serves: what it takes and what it answers.

:mod:`.payloads` and :mod:`.requests` model the compile payload and the
``/api/patch`` requests.  This module models everything else the server and
the browser exchange — simulate, mesh inspection, optimize and its event
stream, the mesh overlay, the editor intelligence, scenes, capabilities and
the job registry — and lists every route in :data:`ROUTES`, with the
request and response models that belong to it.

:data:`ROUTES` is the contract's table of contents.  A test pins it against
the server's routing, so a route added to the server without an entry here
fails rather than reaching the frontend untyped, and the emitter turns it
into the ``ApiRoutes`` map the frontend's client is typed by.

Two conventions hold for every route:

- **Every failure is a** :class:`~.payloads.WorkerFailure`-shaped
  :class:`Failure`: ``{"ok": false, "error"}``, with ``error_kind`` when the
  reason is one the client acts on.  Success models say ``ok: true``, so
  ``ok`` discriminates every response union.
- **A tracked route's response carries** ``job_id``.  The job registry
  (:mod:`cadjoint.viewer._jobs`) adds it to whatever the endpoint answered,
  success or failure, so it is modelled once, as :class:`JobStamped`, and
  the route says whether it applies rather than every model repeating it.

:data:`API_VERSION` is the number a client checks before it trusts any of
this.  It is served by ``/api/session`` and in the ``X-Cadjoint-Api`` header
of every response; bump it whenever a change here would make an older
frontend misread a newer server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from cadjoint.viewer.schema.payloads import (
    BRepInspectionPayload,
    CompilePayload,
    MeshEdgePayload,
    Open,
    Strict,
    StudyDescription,
    Vector3,
    WorkerFailure,
)
from cadjoint.viewer.schema.requests import (
    ExportFormat,
    ExportRequest,
    PatchRequest,
    PatchResponse,
)

API_VERSION = 2
"""The version of this contract.

1 was every server before the contract carried a number; a frontend that
finds no ``api_version`` in ``/api/session`` is talking to one of those.
"""

API_VERSION_HEADER = "X-Cadjoint-Api"
"""The response header that carries :data:`API_VERSION`."""

MAX_CLIENT_ID = 64
"""Longest client request label the server accepts (see ``client_id``).

It is an opaque token the client matches against itself, never anything the
server parses, so the only thing worth bounding is how much of one a request
can make the server remember.
"""


# ── Shared ──────────────────────────────────────────────────────────────────


class Failure(WorkerFailure):
    """What every route answers with when it refuses or fails."""

    job_id: str | None = None
    """Set on a tracked route, as on its success."""


class JobStamped(Strict):
    """What the job registry adds to a tracked route's response."""

    job_id: str
    """The registered job that ran this request; fetch its result again with
    ``GET /api/jobs/{job_id}/result``."""


ClientId = Annotated[str, Field(min_length=1, max_length=MAX_CLIENT_ID)]

ParameterValue = float | list[float]
"""A design parameter's value: a scalar, or its components flattened."""


class SourceRequest(Strict):
    """A request that carries nothing but the program."""

    source: str


class EmptyRequest(Strict):
    """A command that takes no arguments; the client sends ``{}``."""


# ── Session and capabilities ────────────────────────────────────────────────


class SessionResponse(Strict):
    """``GET /api/session``: the write token, the starter program, the contract version."""

    ok: Literal[True]
    token: str
    """Sent back as ``X-Cadjoint-Token`` on every POST and every job read."""
    example: str
    api_version: int
    """:data:`API_VERSION`.  A client built for another number must not proceed."""


class KindStatus(Strict):
    """One private plugin kind's standing in the server process (:mod:`cadjoint.tier`)."""

    kind: str
    available: bool
    plugin: str | None
    version: str | None
    contract_version: int | None
    compatible: bool
    reason: str | None


class CapabilitiesResponse(Strict):
    """``GET /api/capabilities``: which private plugin kinds this server can reach."""

    ok: Literal[True]
    installed: bool
    provider: str
    contract_version: int
    kinds: dict[str, KindStatus]


# ── Compile and the mesh-edge overlay ───────────────────────────────────────


class CompileRequest(SourceRequest):
    """``POST /api/compile`` and ``POST /api/mesh``."""

    client_id: ClientId | None = None
    """The client's own label, so it can cancel the request while it still runs
    (``POST /api/jobs/cancel_client``)."""


class MeshResult(Strict):
    """``POST /api/mesh``: the dual-contour mesh edges of the program."""

    ok: Literal[True]
    mesh_edges: MeshEdgePayload | None
    output: str


class BrepResult(Strict):
    """``POST /api/brep``: the derived B-rep, for the inspector.

    Without the private tier this is still ``ok``: ``brep.available`` is
    false and ``brep.reason`` says why.
    """

    ok: Literal[True]
    brep: BRepInspectionPayload
    output: str


class MeshRefreshRequest(Strict):
    """``POST /api/mesh_refresh``: move the last extracted overlay to new values.

    ``source`` is the program the overlay was extracted from — the base of a
    values-only edit, not the edit — and ``values`` maps free-parameter names
    to their new components, whole parameters at a time.
    """

    source: str
    values: dict[str, list[float]]
    certify: bool = True


class MeshRefreshResult(Strict):
    """The overlay at the new values, and how much of it could not be certified."""

    ok: Literal[True]
    mesh_edges: MeshEdgePayload
    stale: float
    """Fraction of points that left the boundary or changed surface; above a few
    per mille the topology changed and the client extracts again."""


# ── FEM: the render surface and its summaries ───────────────────────────────


class QualitySummary(Strict):
    """``{min, mean, max}`` of one per-element (or per-node) quantity."""

    min: float
    mean: float
    max: float


class SurfaceGroup(Strict):
    """One boundary face group of a simulation mesh — a BC target.

    ``axis``/``side`` are set for a hex mesh's gradient-axis groups (``+x``,
    ``-z``) and null for any other group.
    """

    id: str
    axis: Literal["x", "y", "z"] | None
    side: Literal["+", "-"] | None
    center: Vector3
    area: float
    faces: int
    start: int
    """This group's first triangle in the payload's index buffer."""
    count: int


class SurfacePayload(Strict):
    """A mesh's boundary surface, triangulated, with one scalar per vertex.

    Hex and tet meshes leave the worker in this one shape.  A solved study
    adds its whole per-vertex field catalog (``fields``, ``ranges``) and, for
    an elastic solve, the raw ``displacements`` for a warped view.
    """

    positions: list[float]
    """Flat xyz, three floats per vertex."""
    scalars: list[float]
    indices: list[int]
    """Flat triangle list into the vertex array."""
    edges: list[int] | None = None
    """Element boundary-face edges as flat index pairs (no triangulation diagonals)."""
    groups: list[SurfaceGroup]
    range: Annotated[list[float], Field(min_length=2, max_length=2)]
    vertex_count: int
    fields: dict[str, list[float]] | None = None
    ranges: dict[str, Annotated[list[float], Field(min_length=2, max_length=2)]] | None = None
    displacements: list[Vector3] | None = None


class MeshBounds(Strict):
    min: list[float]
    max: list[float]


class MeshGrid(Strict):
    origin: list[float]
    spacing: list[float]
    cells: list[int]


class MeshInspectInfo(Open):
    """``SimMesh.inspect()``: what was built and how good its elements are.

    Open because it is the library's own ``inspect()``; ``quality`` maps each
    metric the method has (``scaled_jacobian`` for hexes, ``radius_ratio``
    and ``aspect_ratio`` for tets) to its summary.
    """

    name: str
    method: str
    mesher: str | None = None
    frozen_geometry: bool | None = None
    nodes: int
    elements: int
    bounds: MeshBounds
    grid: MeshGrid | None
    quality: dict[str, QualitySummary]


class Refinement(Strict):
    """How far the tet mesher had to re-dice the declared grid."""

    declared: list[int]
    used: list[int]
    attempts: int


class ResultSummary(Open):
    """``SimulationResult.describe()``: the solved study, summarised."""

    name: str
    kind: Literal["thermal", "elastic", "flow"]
    field: str
    """The display field ``mesh.scalars`` carries."""
    mesh: str | None
    """The SimMesh the study solved on, or null for an implicit mesh."""
    nodes: int
    elements: int
    range: Annotated[list[float], Field(min_length=2, max_length=2)]
    fields: dict[str, QualitySummary]
    mass: float | None = None
    safety_factor: float | None = None
    refinement: Refinement | None = None


# ── Simulate and mesh inspection ────────────────────────────────────────────


class SimulateRequest(SourceRequest):
    """``POST /api/simulate``: solve one study the program declares, by name."""

    kind: Literal["study"] = "study"
    name: str
    cached: bool = False
    """Serve the study's ``last_result`` when the program already solved it."""


class SimulateResult(Strict):
    """A solved study: its declaration, its render surface, and its summaries."""

    ok: Literal[True]
    kind: Literal["study"]
    study: StudyDescription
    field: str
    mesh: SurfacePayload
    result: ResultSummary
    mesh_info: MeshInspectInfo | None
    cached: bool
    output: str


class MeshInspectRequest(SourceRequest):
    """``POST /api/mesh_inspect``: build a declared mesh (or a study's) without solving.

    With no ``name``, the program's one mesh — or, failing that, its one
    study — is used.
    """

    name: str | None = None


class MeshInspectResult(Strict):
    """A built mesh: its report and a quality heatmap of its surface."""

    ok: Literal[True]
    kind: Literal["mesh_inspect"]
    name: str
    field: str | None
    """The quality metric ``mesh.scalars`` carries, or null for a cut-cell mesh."""
    info: MeshInspectInfo
    mesh: SurfacePayload
    quality_scalars: list[float]
    output: str


# ── Optimize ────────────────────────────────────────────────────────────────


class OptimizeRequest(SourceRequest):
    """``POST /api/optimize``: run one declared ``Optimization`` by name."""

    name: str
    steps: int | None = Field(default=None, ge=1)
    """Overrides the declared step count; capped per request by the server."""


class OptimizeStep(Strict):
    """One recorded optimizer evaluation."""

    step: int
    objective: float
    grad_norm: float


class OptimizeFrame(Strict):
    """One replayable trajectory frame; step 0 is the initial design."""

    step: int
    objective: float
    parameters: dict[str, ParameterValue]


class OptimizeSimulateBlock(Strict):
    """A study-backed run's final design, solved — the shapes ``/api/simulate`` uses."""

    field: str
    mesh: SurfacePayload
    result: ResultSummary
    mesh_info: MeshInspectInfo | None


class OptimizeResult(Strict):
    """A finished run: the program with the optimized values written back."""

    ok: Literal[True]
    kind: Literal["optimize"]
    name: str
    method: str
    steps: int
    source: str
    """The patched program; the client adopts it exactly like a patch response."""
    history: list[OptimizeStep]
    trajectory: list[OptimizeFrame]
    """At most 100 frames, evenly thinned, first and last always kept."""
    parameters: dict[str, ParameterValue]
    initial: dict[str, ParameterValue]
    simulate: OptimizeSimulateBlock | None = None
    output: str


class OptimizeProgressEvent(Strict):
    """One ``/api/optimize`` stream line per finished optimizer step."""

    event: Literal["progress"]
    step: int
    """Finished evaluations, 1-based."""
    steps: int
    objective: float
    grad_norm: float
    elapsed: float
    job_id: str | None = None


class OptimizeDone(OptimizeResult):
    """The stream's last line when the run succeeded."""

    event: Literal["done"]
    job_id: str | None = None


class OptimizeFailed(Failure):
    """The stream's last line when the run (or its request) failed."""

    event: Literal["done"]


# ── Export ──────────────────────────────────────────────────────────────────


class ExportSummary(Strict):
    """``X-Cadjoint-Export``: what the server says about the file beside its bytes.

    ``report`` is the writer's own: ``resolution``/``bounds``/``size`` of the
    lattice for a geometry format, ``study``/``cached`` for ``vtk``.
    """

    format: ExportFormat
    name: str
    size: int
    report: dict[str, Any]


# ── Editor intelligence ─────────────────────────────────────────────────────
#
# Lines are 1-based and columns 0-based everywhere below — CodeMirror's
# convention, so a position is `doc.line(from_line).from + from_col`.


class LintFixEdit(Strict):
    from_line: int
    from_col: int
    to_line: int
    to_col: int
    content: str


class LintFix(Strict):
    """A ruff autofix: a label and the edits that apply it in one go."""

    message: str
    applicability: Literal["safe", "unsafe", "display"]
    edits: list[LintFixEdit]


class LintDiagnostic(Strict):
    """One diagnostic: ruff's, or the last failed compile's traceback line."""

    from_line: int
    from_col: int
    to_line: int
    to_col: int
    severity: Literal["error", "warning", "info"]
    message: str
    code: str
    source: Literal["ruff", "runtime"]
    url: str | None
    fix: LintFix | None


class LintResult(Strict):
    """``POST /api/lint``."""

    ok: Literal[True]
    diagnostics: list[LintDiagnostic]
    runtime: bool
    """Whether a remembered compile traceback contributed a diagnostic."""


class CursorRequest(SourceRequest):
    """``POST /api/complete`` and ``POST /api/signature``: a caret in the program."""

    line: int
    column: int


class CompletionItem(Strict):
    """One jedi completion, shaped like CodeMirror's ``Completion``."""

    label: str
    type: str
    detail: str
    info: str | None
    """Signature and docstring; only the head of the list pays for one."""
    apply: str


class CompleteResult(Strict):
    """``POST /api/complete``."""

    ok: Literal[True]
    from_line: int
    from_column: int
    """Where the already-typed prefix starts."""
    completions: list[CompletionItem]
    truncated: bool


class SignatureParameter(Strict):
    name: str
    label: str


class SignatureInfo(Strict):
    name: str
    label: str
    active_parameter: int | None
    parameters: list[SignatureParameter]
    documentation: str | None


class SignatureResult(Strict):
    """``POST /api/signature``."""

    ok: Literal[True]
    signatures: list[SignatureInfo]


# ── Scenes ──────────────────────────────────────────────────────────────────


class SceneCounts(Strict):
    """A scene's declarations, counted statically; ``parameters`` only the named ones."""

    parameters: int
    free: int
    studies: int
    meshes: int
    optimizations: int
    materials: int


class SceneEntry(Strict):
    """One scene file, described with ``ast`` and never by running it."""

    name: str
    path: str
    bytes: int
    modified: str | None
    """ISO-8601 UTC, or null when the file could not be stat'd."""
    source_hash: str | None
    summary: str
    counts: SceneCounts
    materials: list[str]
    error: str | None


class SceneList(Strict):
    """``GET /api/scenes``."""

    ok: Literal[True]
    files: list[str]
    scenes: list[SceneEntry]


class SceneLoadRequest(Strict):
    name: str


class SceneLoadResult(Strict):
    ok: Literal[True]
    name: str
    source: str


class SceneSaveRequest(Strict):
    name: str
    source: str


class SceneSaveResult(Strict):
    ok: Literal[True]
    name: str


# ── The job registry ────────────────────────────────────────────────────────

JobKind = Literal[
    "compile", "mesh", "brep", "mesh_inspect", "simulate", "optimize", "export", "lint", "warmup"
]
JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]


class JobProgress(Strict):
    """The per-step figures an optimize job mirrors out of its stream."""

    step: int | None = None
    steps: int | None = None
    objective: float | None = None
    grad_norm: float | None = None
    elapsed: float | None = None


class JobSample(Strict):
    """One resource sample of a running worker, relative to the job's start."""

    t: float
    cpu_percent: float
    rss_bytes: int


class JobSummary(Strict):
    """A job as ``GET /api/jobs`` lists it: everything but the result and samples."""

    job_id: str
    kind: JobKind
    status: JobStatus
    fields: dict[str, str | int | float | bool]
    source_hash: str | None
    source_bytes: int
    submitted_at: float
    started_at: float | None
    finished_at: float | None
    elapsed_s: float
    pid: int | None
    ok: bool | None
    error: str | None
    progress: JobProgress | None
    cpu_percent: float
    rss_bytes: int
    peak_cpu_percent: float
    peak_rss_bytes: int
    cpu_seconds: float
    sampling: Literal["psutil", "rusage", "none"]
    samples: int
    result_available: bool
    result_bytes: int


class JobDetail(JobSummary):
    sample_series: list[JobSample]
    progress_events: list[JobProgress]


class ServerLoad(Strict):
    cpu_percent: float
    rss_bytes: int
    pid: int


class HostCapacity(Strict):
    cpu_count: int | None
    mem_total: int | None
    mem_available: int | None


class JobTotals(Strict):
    running: int
    cpu_percent: float
    rss_bytes: int
    uptime_s: float
    server: ServerLoad
    host: HostCapacity
    sampling: Literal["psutil", "rusage"]


class JobStore(Strict):
    jobs: int
    max_jobs: int
    max_lint_jobs: int
    result_bytes: int
    max_result_bytes: int
    evicted_jobs: int
    evicted_results: int


class JobsSnapshot(Strict):
    """``GET /api/jobs``: every registered job, newest first, and the live totals."""

    ok: Literal[True]
    jobs: list[JobSummary]
    totals: JobTotals
    store: JobStore


class JobDetailResult(Strict):
    """``GET /api/jobs/{job_id}``."""

    ok: Literal[True]
    job: JobDetail


class JobFailure(Failure):
    """A job route's refusal: unknown id (404), still running (409), evicted (410)."""

    job_id: str
    status: JobStatus | None = None


class JobCancelled(Strict):
    """``POST /api/jobs/{job_id}/cancel``."""

    ok: Literal[True]
    job_id: str
    status: Literal["cancelled"]


class JobsCleared(Strict):
    """``POST /api/jobs/clear``: finished jobs dropped; running ones are kept."""

    ok: Literal[True]
    cleared: int
    remaining: int


class CancelClientRequest(Strict):
    """``POST /api/jobs/cancel_client``."""

    client_id: ClientId


class ClientCancelled(Strict):
    ok: Literal[True]
    client_id: str
    cancelled: list[str]
    """The job ids actually stopped; empty when the work had already finished."""


# ── The route table ─────────────────────────────────────────────────────────

Response = tuple[type[BaseModel], ...]


@dataclass(frozen=True)
class Route:
    """One route: its request, its response union, and how it is framed.

    Attributes:
        method: ``GET`` or ``POST``.
        path: The path, with ``{job_id}`` standing for a path segment.
        name: The generated TypeScript alias for the response union.
        request: The JSON body model, or None for a GET.
        response: The models a JSON response may match; ``ok`` tells them apart.
        tracked: Whether the job registry stamps ``job_id`` onto the response.
        stream: The response is NDJSON, one :data:`response` model per line.
        file: A success is a file attachment rather than JSON; only a failure
            is JSON (see ``X-Cadjoint-Export`` and :class:`ExportSummary`).
    """

    method: Literal["GET", "POST"]
    path: str
    name: str
    """The TypeScript name of the response union, e.g. ``CompileResponse``."""
    request: Any
    """A model, the :data:`~.requests.PatchRequest` union, or None."""
    response: Response
    tracked: bool = False
    stream: bool = False
    file: bool = False


ROUTES: tuple[Route, ...] = (
    Route("GET", "/api/session", "SessionResponse", None, (SessionResponse,)),
    Route("GET", "/api/capabilities", "CapabilitiesResponse", None, (CapabilitiesResponse,)),
    Route(
        "POST",
        "/api/compile",
        "CompileResponse",
        CompileRequest,
        (CompilePayload, Failure),
        tracked=True,
    ),
    Route("POST", "/api/patch", "PatchResponse", PatchRequest, (PatchResponse,)),
    Route("POST", "/api/mesh", "MeshResponse", CompileRequest, (MeshResult, Failure), tracked=True),
    Route("POST", "/api/brep", "BrepResponse", CompileRequest, (BrepResult, Failure), tracked=True),
    Route(
        "POST",
        "/api/mesh_refresh",
        "MeshRefreshResponse",
        MeshRefreshRequest,
        (MeshRefreshResult, Failure),
    ),
    Route(
        "POST",
        "/api/simulate",
        "SimulateResponse",
        SimulateRequest,
        (SimulateResult, Failure),
        tracked=True,
    ),
    Route(
        "POST",
        "/api/mesh_inspect",
        "MeshInspectResponse",
        MeshInspectRequest,
        (MeshInspectResult, Failure),
        tracked=True,
    ),
    Route(
        "POST",
        "/api/optimize",
        "OptimizeStreamLine",
        OptimizeRequest,
        (OptimizeProgressEvent, OptimizeDone, OptimizeFailed),
        tracked=True,
        stream=True,
    ),
    Route("POST", "/api/export", "Failure", ExportRequest, (Failure,), tracked=True, file=True),
    Route("POST", "/api/lint", "LintResponse", SourceRequest, (LintResult, Failure), tracked=True),
    Route("POST", "/api/complete", "CompleteResponse", CursorRequest, (CompleteResult, Failure)),
    Route("POST", "/api/signature", "SignatureResponse", CursorRequest, (SignatureResult, Failure)),
    Route("GET", "/api/scenes", "SceneListResponse", None, (SceneList, Failure)),
    Route(
        "POST",
        "/api/scenes/load",
        "SceneLoadResponse",
        SceneLoadRequest,
        (SceneLoadResult, Failure),
    ),
    Route(
        "POST",
        "/api/scenes/save",
        "SceneSaveResponse",
        SceneSaveRequest,
        (SceneSaveResult, Failure),
    ),
    Route("GET", "/api/jobs", "JobsSnapshot", None, (JobsSnapshot,)),
    Route("GET", "/api/jobs/{job_id}", "JobDetailResponse", None, (JobDetailResult, JobFailure)),
    # The stored result is byte-identical to what the job's own route
    # answered, so its type is that route's response: the client says which.
    Route("GET", "/api/jobs/{job_id}/result", "JobFailure", None, (JobFailure,)),
    Route(
        "POST",
        "/api/jobs/{job_id}/cancel",
        "JobCancelResponse",
        EmptyRequest,
        (JobCancelled, JobFailure),
    ),
    Route("POST", "/api/jobs/clear", "JobsCleared", EmptyRequest, (JobsCleared,)),
    Route(
        "POST",
        "/api/jobs/cancel_client",
        "CancelClientResponse",
        CancelClientRequest,
        (ClientCancelled, Failure),
    ),
)
"""Every route the server answers, in the order the module docstring lists them."""
