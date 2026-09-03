# Two tiers — the public `cadjoint` and the private `diff-brep`: the design

Status: design memo, 2026-09-03, on `refactor/code-quality` at `b9f6a98`.
Nothing was moved, no code was changed, and the private repository
(`https://github.com/andrinr/diff-brep`, private, empty) was not written to.
Every file, line and number below was read from this tree or from the
research notes it cites; the one figure that is not in a note is marked.

The user's direction, verbatim: *"ok we need to redesign the whole project
into a two tier project. There has to be a second private github repo that
has some of the advanced step meshing tools that are currently not publicly
available also that second repo should really have an extra layer of
cleanliness to it and we should eval writing more things in c / rust there
for performance"*; then, on the boundary: *"i think we can move gmsh tet10
to the public one as well, the only 'secret' there is the surface
interpolation"*; and on the seam: *"tesseracts do actually provide a good
contract for those components, but I wonder if it really makes sense to use
tesseracts everywhere?"*

Names: the repository is `diff-brep`, the distribution `diff-brep`, the
package `diff_brep` (`import diff_brep`). Decisions the memo asks for are
tagged **D1 … D12** so each can be answered on its own.

## 0. The one paragraph

The line falls between *reading a lattice* and *reading the fields*. Public
`cadjoint` keeps everything lattice-driven — dual contouring, Hermite data,
the leaf-level seam projection and lattice feature classifier, faceted
STEP/OBJ/STL, TetGen tet4/tet10, HEX8, every solver — **and now the Gmsh
tet10 route**: STEP or STL in, Gmsh HXT order 2, a static tet10 mesh out,
each node tagged with the patches that own it, packaged as the `tet_gmsh`
Tesseract exactly as today. Private `diff_brep` keeps everything that
solves against the fields with a derivative: the projection kernel with
its implicit-function adjoint, the ownership graph, the tracer, the
analytic STEP writer, the drag inverse problem, the PLC, and the map from
design parameters to node positions that makes a Gmsh mesh differentiable.
That map is the seam's centrepiece: its input is a public record (`seed,
patch ids, arity` per node) that public `assign_ownership` produces, its
output is positions with an IFT VJP, and it is consumed in-process inside
the jitted study objective — so it is a *plugin* in the registry's sense
but not a *Tesseract*, and answering the user's question is the design:
one contract, two transports, Tesseract where a component is coarse (one
apply and one VJP per optimizer step, ~10 ms of loopback cost) and an
in-process Python object where it is fine (the kernel inside `vmap`, a
tracer step, a per-compile overlay — where a 0.14 s round trip would be
paid hundreds of times for microseconds of work). Without `diff_brep`
installed, the app compiles, meshes with Gmsh, solves, exports faceted
STEP, draws lattice feature edges, and says in one place — `cadjoint.tier`
— that geometry gradients through a Gmsh mesh, analytic STEP and graph
edges are the private tier's. On C/Rust the honest answer stays *not yet*:
the private seconds are XLA compile and dispatch, the one loop XLA fits
badly has a cheaper JAX fix first, and Rust keeps differentiability only in
the shape "Rust decides topology, JAX re-solves positions".

---

## 1. The boundary

### 1.1 What stays public and what moves

| Module (lines) | Tier | Why | Public baseline |
|---|---|---|---|
| `cadjoint/sdf`, `constraints`, `construction`, `geometry`, `materials`, `extraction`, `functionalize`, `parametrization`, `cache`, `enums`, `fluent`, `optimize` | **public** | the language | — |
| `cadjoint/meshing/*` incl. `features.py`'s lattice classifier (`classify_feature_cells`, `feature_cell_links`, exported at `meshing/__init__.py:66-68`), `patch_fields.py`, `export.py` (faceted STEP/OBJ/STL) | **public** | the published DC core; the faceted writers are the baseline every consumer needs | itself |
| `cadjoint/sdf/**.patch_fields()` on twelve primitive/transform classes (`box.py:66`, `cylinder.py:64`, `polygon.py:349,427`, `sphere.py:81`, `torus.py:53`, `capsule.py:55`, `round_box.py:56`, `plane.py:53`, three affines) and `meshing/patch_fields.py` (`ScenePatchFields`, `scene_patch_fields`, `signature_function`, `world_frame_leaves`) | **public** (**D2**) | the decomposition lives on the primitives and the sharp-QEF DC consumes it; it cannot move without the primitives. What it leaks is a decomposition, not the graph, and the public Gmsh ownership tagging (1.2) needs it | itself |
| `cadjoint/fem/motion.py::project_points` (arity 1, the DC/TetGen path's node motion) | **public** | shipped, documented, the hackathon-era gradient contract of `recompute_tet_points` | itself |
| `cadjoint/viewer/_edge_overlay.py` lattice half — `_design_leaves`, `_project_to_seam` (2-field, forward-only, lines 253–320), `_project_seam_groups(_reference)`, `_seam_residual`, the wire layer of `_mesh_edge_payload` | **public** | the batched §6.2 path (`performance.md`: 5.59 s → 0.685 s) the public overlay falls back to | itself + `feature_cell_links` |
| **`cadjoint/brep/mesh_gmsh.py` — the Gmsh route** (`gmsh_available`, `gmsh_version`, `_gmsh_session`, `gmsh_topology`, `_check_import`, `_harvest`, `_tet_cells`, `_reorder`, `gmsh_tet_mesh`, `_plugin_topology`, `assign_ownership`, `_surface_owner`, `_owner_rows`, `GmshMesh`, `tet_mesh_from_gmsh`) — about 850 of its 1032 lines | **public**, moved to `cadjoint/fem/gmsh.py` | the user's call: the mesher and the ownership *tag* are not the secret; the tag is a residual test against public patch fields (1.2) | itself; `SimMesh(method="tet10", mesher="gmsh")` becomes a real public keyword |
| `cadjoint/fem/tesseracts/tet_gmsh/` (283) and the `gmsh` extra | **public**, unchanged | GPL-2.0-or-later behind a process boundary, as today (`tesseract_api.py:6-12`) | itself |
| **`cadjoint/brep/mesh_gmsh.py` — the node map** (`recompute_gmsh_points`, `_smoothed`, `_node_adjacency`, `parameterised_points`; ~180 lines) | **private** — `diff_brep.nodemap` | *the* secret: design parameters → node positions, per-arity Newton onto the owning patches with the IFT adjoint, midsides re-solved on their surfaces, interior nodes following by Laplacian relaxation, and the VJP of all of it | none: a Gmsh mesh is frozen geometry (2.5) |
| `cadjoint/brep/project.py` (622) — `project`, `project_batched`, `project_fields`, `batched_residuals`, `field_residuals`, `transversal`, `stacked_fields`, the `custom_vjp` (`project.py:162-203`), and `trace_curves` | **private** (**D3**, argued below) | the kernel *is* the surface interpolation the user named: "positioned by one Newton projection kernel with an implicit-function-theorem adjoint" is the sentence the private tier owns | `project_points` (arity 1, public) and the overlay's `_project_to_seam` (arity 2, no derivative) |
| `cadjoint/brep/graph.py` (1125) — patch table, two-stage ownership, blend test, face components, loops, edges, vertices, surface fits | **private** — `diff_brep.graph` | the moat; a prototype with three stable failure classes in the axiom battery; where the next year of work goes | none — the public tier has no B-rep object |
| `cadjoint/brep/step.py` (669) — loop simplification, the shared-decision rule, `PLANE`/`CYLINDRICAL_SURFACE`/`CIRCLE` emission | **private** — `diff_brep.step` | exact STEP (181 entities and 1e-7 volume vs ~10 000 and 1.6e-3) | `meshing/export.py::save_step`, faceted, always closes |
| `cadjoint/brep/drag.py` (451), `cadjoint/brep/plc.py` (331) | **private** — `diff_brep.drag`, `diff_brep.plc` | graph-dependent products; nothing public calls them today (no import edge from the viewer) | none; TetGen from DC quads |
| the planned `interval.py` (census, Krawczyk, jaxpr interpreter), LM `damped=True`, the tangency classifier, watch-field termination (`brep-edge-tracing.md` §10) | **private from the start** | graph work | — |
| `_edge_overlay.py` graph half — `_extract_graph`, `_design_patches`, `_corners_on_curve`, `_between_corners`, `_attach_corners`, `_edge_polyline`, `_traced_polylines`, `_sampled_polylines`, `_sharp_polylines`, `_sharp_chords`, `_prune_debris`, the tuning constants at lines 79–186 | **private** — `diff_brep.edges` | the tracer's consumer | the lattice half |
| `cadjoint/viewer/_export.py` | **public**, one branch cut (line 265) | — | faceted STEP with `tier` in the report |
| `cadjoint/plugins/*` | **public**, gains one transport (2.1) | the seam's registry | — |
| `tests/brep/*` except the static-mesh half of `test_mesh_gmsh.py`; `tests/viewer/test_edge_overlay_brep.py` | **private** | they test private code; the axiom battery is the private gate (3.4) | — |
| `tests/brep/test_mesh_gmsh.py` static half (counts, quality, `TestThePlugin`, the packaging assertions at lines 462–540) | **public** → `tests/fem/test_gmsh.py` | — | — |
| `tests/viewer/test_edge_artifacts.py` (771) | **both** | public runs it on the lattice layer with its own lower baselines; private keeps coverage 1.000 | — |
| `research/brep-architecture.md`, `brep-edge-tracing.md`, `brep-axioms.md`, `research/brep-axioms/` (31 MB gallery), `research/design/light-chrome/edges-*.png`, `docs/brep.qmd` | **private** | they document private code | one public `docs/tier.qmd` and a Gmsh section in `docs/simulation.qmd` |

**D1 — the boundary as tabled: yes/no.**

**D2 — `patch_fields()` and `scene_patch_fields` stay public.** Required
now by the public Gmsh ownership tag as well as by the sharp QEF. The
alternative — diff_brep monkey-patching `patch_fields` onto public classes
at import — is the kind of thing 3.2 forbids. Recommend yes.

**D3 — the projection kernel is private.** The earlier draft of this memo
had it public because the arity-1 form already is (`project_points`) and
the arity-2 form was the overlay's before the graph existed. The user's
"the only secret is the surface interpolation" settles it the other way:
the map from design to position *is* the kernel plus ownership, and the
public tier does not need the general kernel for anything — the DC path
has `project_points`, the lattice overlay has its own forward-only
`_project_to_seam`, and the public Gmsh route only *tags* ownership (a
residual test, no Newton). The cost is that two Gauss–Newton
implementations exist again across the boundary; the private parity test
keeps pinning `project ≡ project_points` at arity 1 to one ulp and
`project ≡ _project_to_seam` at arity 2, so the public ones stay honest
without knowing it. Recommend yes.

### 1.2 The Gmsh route without the graph

`gmsh_tet_mesh(brep, …)` today takes a `BRep`, writes its analytic STEP
(`mesh_gmsh.py:723-735`), and tags ownership by a nearest-quad *vote* on
the graph's `quad_face` table confirmed by `|f_patch| ≤ bar` on the
entity's own nodes (`_surface_owner`, `_face_of_position`). Two things the
public route has to supply without a graph.

**Face identity in the input geometry.** Gmsh needs a solid whose surfaces
are the part's faces, or it constrains the mesh to every facet edge. Three
ways to get one:

| option | what Gmsh sees | element size on planes | on curved faces | midsides | cost to build |
|---|---|---|---|---|---|
| (i) planar-merged faceted STEP (`meshing.export.save_step` extended to keep hole loops — today it "gives up" on the plate's cap and writes 3 079 faces) | one `PLANE` per merged region, one face per facet elsewhere | by the part | constrained to the 64³ facets | chord midpoints | ~1 day (inner loops in the merger) |
| (ii) a public subset of the graph (ownership → components → one facet group per face, no fits, no loops, no tracing) | one discrete surface per face | by the part | by the part | chord midpoints | it *is* `graph.py` steps 1–4 — the moat. **No.** |
| (iii) the DC triangles as STL, then Gmsh's own STL route: `classifySurfaces(angle, boundary=True, forReparametrization=True)` + `createGeometry()` (tutorial t13) | one reparametrised discrete surface per smooth region, split at the lattice's feature cells | by the part | by the part | on the facets | one afternoon to try, 1–2 days to land; the risk is classification on DC slivers |

**Recommend (iii), with (i) as the exporter's own improvement** (it is
worth having regardless — the public STEP export gets holes). The
`tet_gmsh` Tesseract input gains `geometry_format: "step" | "stl"` beside
the existing `step: str` (renamed `geometry`), and *with* `diff_brep`
installed the same mesher is fed the analytic STEP and the midsides land on
the true cylinder (`brep-architecture.md` §5.4: every bore midside at
r = 0.25 to 1e-6 against a chord midpoint up to 1.4e-3 inside). So the
public route is not a different mesher with a different contract; it is
the same mesher with a coarser input, and the private tier improves the
input. **D4 — (iii): yes/no.**

**Ownership tagging without the vote.** The confirmation half of
`_surface_owner` needs only the public patch table: per Gmsh surface
entity, evaluate `|f_p|` on its interior nodes for every patch `p` (one
batched program per patch, the same 0.20 s on the plate that §5.4 already
measures as "one batched JAX call per patch"), keep the patches whose
*maximum* residual is under the bar (the §5.4 fix, "confirm on the maximum
not the median"), and apply the same bar at every curve and vertex node
(the second §5.4 fix). No graph face is consulted; arity is the count of
confirming patches; a node no patch confirms is a blend node. The output is
the public record the private map consumes:

```
OwnedNodes                       # cadjoint.fem.gmsh, frozen dataclass of NumPy arrays
  seeds        (P, 3) float64    # node positions at the design they were meshed at
  patches      (P, 3) int32      # global patch indices (scene_patch_fields order), -1 padded
  arity        (P,)   int8       # count of non -1; 0 = blend / scene-owned
  entity_dim   (P,)   int8       # Gmsh's 0 vertex, 1 curve, 2 surface, 3 volume
  blend        (P,)   bool       # arity 0 and entity_dim < 3
  midside      (P,)   bool;  edge_parents (M, 2) int32   # the TetMesh layout's midside block
  bar          float             # the residual bar used
  design       dict[str, float]  # parameter values at meshing (for the postcondition in 2.4)
```

So the Gmsh route **can be public without the graph**, and the only thing
the graph adds to it is a better input file.

### 1.3 Every import edge that crosses the boundary today, re-checked

From `grep -rn "cadjoint.brep\|from cadjoint import brep\|mesh_gmsh\|tet_gmsh" cadjoint tests frontend docs scenes` on `b9f6a98`; docstring-only mentions omitted.

**Public → private (the edges to cut):**

| # | Public file : line | Imports | How it is cut |
|---|---|---|---|
| 1 | `cadjoint/viewer/_edge_overlay.py:44` | `BRep, BRepEdge` (TYPE_CHECKING) | goes to `diff_brep.edges` with the graph half |
| 2 | `_edge_overlay.py:472` | `extract_brep` in `_extract_graph` | `_mesh_edge_payload` resolves the `feature_edges` kind (2.3); absent → lattice layer |
| 3–5 | `_edge_overlay.py:656,926,993` | `batched_residuals`, `trace_curves`, `project_batched` | go to `diff_brep.edges` |
| 6 | `cadjoint/viewer/_export.py:265` | `extract_brep, save_brep_step` | `step_export` kind; absent → faceted with `report["tier"]` |
| 7 | `cadjoint/fem/tesseracts/tet_gmsh/tesseract_api.py:130` | `cadjoint.brep.mesh_gmsh.gmsh_topology` | becomes `cadjoint.fem.gmsh.gmsh_topology` — public → public |
| 8 | the new `cadjoint/fem/gmsh.py` (from `mesh_gmsh.py:70-72,967-968`) | `BRep`, `project_batched`, `project_fields`, `save_brep_step`, `drag.patch_field_fn` | `gmsh_tet_mesh(geometry: str | Path, scene, …)` takes geometry text and the scene (for the patch table) instead of a `BRep`; `assign_ownership(scene, topology)` tags by residual (1.2); `recompute_gmsh_points` / `parameterised_points` leave for `diff_brep.nodemap`; `tet_mesh_from_gmsh` becomes static (`points` are the seeds) and attaches `OwnedNodes` |
| 9 | `tests/brep/test_mesh_gmsh.py` | `extract_brep`, `parameterised_points`, `TestTheDerivative` (lines 38-39, 23) | the derivative class and the analytic-STEP-fed cases move to diff-brep; the rest becomes `tests/fem/test_gmsh.py` fed by (iii) |
| 10 | `tests/viewer/test_edge_overlay_brep.py:26,158,312,379,545` | tracer, residuals, `BRepEdge`, `brep.graph` | file moves |
| 11 | `tests/meshing/test_patch_fields.py:241` | docstring mention of `drag.patch_field_fn` | reword |
| 12 | `_quarto.yml:176-205` (`B-rep — *` sections), `docs/brep.qmd`, `docs/plugins.qmd:329-346` (the "cuts at the geometry" paragraph names `cadjoint.brep.step` and `assign_ownership`), `docs/simulation.qmd:66-70`, `docs/meshing.qmd:162-165,197`, `docs/viewer.qmd:301,318`, `docs/getting-started.qmd:144,265,342`, `README.md:45-53,74,80,258-264,436-447,840-842` | prose and API entries | §5 step 5 |

**Now staying public (were cuts in the first draft):** `plugins/registry.py:76,89`
(`tet_gmsh` builtin and the `tet_mesher` default), `enums.py:408`
(`PluginKind.TET_MESHER`), `tests/fem/test_tesseract_packaging.py:51`,
`pyproject.toml:50-64` (the `gmsh` extra — its comment loses the
`cadjoint.brep` path), `docs/plugins.qmd:68,248`, `README.md:145,662,710,736`,
`docs/getting-started.qmd:44`.

`frontend/` and `scenes/` have **no** edge: the wire payload is
`MeshEdgePayload{wire, sharp, resolution}` (`viewer/schema/payloads.py:342`)
and the renderer counts `sharp.length` (`renderer.ts:806`); a
lattice-sourced `sharp` needs no frontend change.

**Private → public (diff-brep's dependency on cadjoint):**

| Private module | Public thing it needs | Note |
|---|---|---|
| `diff_brep.graph` (from `graph.py:52-55`) | `meshing.dual_contouring.{Mesh, extract_mesh}`, `meshing.edge_detection.GridSpec`, `meshing.patch_fields.{ScenePatchFields, scene_patch_fields}` | stable public API |
| `diff_brep.step` (from `step.py:39`) | `meshing.export.{_STEP_HEADER, _STEP_BOILERPLATE, _step_real, _weld_degenerate_edges}` | **underscore names across a repo boundary** — promote to a public `cadjoint.meshing.step_scaffold` first (§5 step 1a) |
| `diff_brep.nodemap` | `cadjoint.fem.gmsh.{GmshMesh, OwnedNodes}`, `fem.tetmesh.TetMesh` | the seam's records |
| `diff_brep.plc` | `fem.tetmesh.TetMesh`, `tetgen` | public |
| `diff_brep.drag` | `constraints.solve.project_to_manifold`, `extraction.{extract_parameters, apply_parameters}`, the concrete shoelace sign in `sdf/primitives/polygon.py` | public; the §4.3 recommendation (orientation as an argument) is a two-line public change |
| `diff_brep.edges` | `meshing.patch_fields.world_frame_leaves`, `construction.faces.FaceSet` | public |
| `diff_brep.plugins` | `cadjoint.plugins.PluginSpec`, the contracts module (2.1) | public |

Nothing private is needed by `fem/tetmesh.py`, `motion.py`, `simmesh.py`,
`optimize.py` or `fem/tesseracts/chain.py` today (checked); the chain
gains exactly one optional call, the node map (2.5).

---

## 2. The seam

### 2.1 One contract, two transports — the answer to "Tesseracts everywhere?"

A Tesseract is the right contract for a **coarse-grained differentiable
component**: arrays in, arrays out, one `apply` and one
`vector_jacobian_product` per call, deployable `local`/`container`/`remote`
with one constructor each. The numbers say where that stops being true:

| call | cost | source |
|---|---|---|
| a served Tesseract over loopback, per optimizer step (mesher, 1285 nodes) | +10 ms on a 548 ms step, 1.3–1.8 %, identical to 13 digits | `docs/plugins.qmd:385-398` |
| an in-process Tesseract apply/VJP round trip | ~0.14 s | `README.md:721` |
| `_project_to_seam`, 2 fields, 300 points, jitted | 0.4 ms | `performance.md` §6.5 |
| `edge_hermite_data`, 1 492 edges, jitted | 3 ms | §6.5 |
| one tracer `advance` step (a jitted program over a batch of curves) | sub-millisecond; hundreds per overlay | `project.py:536`, §8.2 |
| the retired Rust QEF behind a `qef` Tesseract | "a plugin boundary around a 2 ms in-process kernel, the wrong shape" | `native-mesher.md` |
| zero-size arrays over HTTP | refused by tesseract-core 1.11; discovery mode must run in-process anyway | `docs/plugins.qmd:406-418` |

So a Tesseract round trip costs a hundred to a thousand times what the
private tier's fine-grained calls compute, and it cannot fuse with XLA,
cannot sit inside `vmap` of the kernel, and gives you a VJP but not the
`jvp`/batching semantics of your own `custom_vjp`. Everything called inside
a trace or per compile — the kernel under `vmap`/`jit`, a tracer step, the
viewer's overlay, interval evaluation, ownership tagging — is the wrong
grain for it. Everything called **once per optimizer step or once per job**
— solvers, meshers, the flow solver, and the node map when a whole chain
runs on a cluster — is exactly the right grain, and the measured 1.3–1.8 %
is the proof.

**The design: one contract, two transports.** A component declares one
typed in-process interface — a `Protocol` in a public
`cadjoint/plugins/contracts.py`, with frozen-dataclass / pytree payloads —
and the registry can bind that interface either to a Python object
imported in-process or to a Tesseract. The Tesseract package is then a
thin wrapper over the same object: `tesseract_api.py` re-exports the
component's pydantic `InputSchema`/`OutputSchema` and binds `apply` and
`vector_jacobian_product` to it — which is already how `tet_gmsh`'s
`tesseract_api.py` is written (its `_discover` is one call into
`gmsh_topology`). The `Plugin` protocol already says this is the intent:
"`Plugin` is the protocol and `TesseractPlugin` the one implementation,
because Tesseract is how plugins are implemented today, not what a plugin
is" (`plugins/plugin.py:18-20`).

**What the registry gains** (all public, ~150 lines):

- `PluginTransport.PYTHON` — `transport = "python"`, target field `object =
  "diff_brep.plugins:NODE_MAP"`; `PluginSpec.open()` does
  `importlib.import_module` + `getattr`, no tesseract-core needed
  (`spec.py:253-284` gains one branch; `_TARGET` at line 52 one row). The name
  `local` keeps meaning "a Tesseract in this process" so no existing
  `plugins.toml` changes meaning.
- `PythonPlugin(Plugin)` beside `TesseractPlugin`: `apply`/`vjp` call the
  object's methods; `as_jax()` returns the object's own JAX callable (its
  `custom_vjp`), not `tesseract_jax.apply_tesseract`; `inputs`/`outputs`/
  `capabilities` come from the contract's declared payload types
  (`Differentiable` marked by a `typing.Annotated` tag in
  `contracts.py`, so the in-process path needs neither pydantic nor
  tesseract-core); `probe()` returns the object's `version` and a hash of
  the contract's signature.
- entry-point discovery unchanged (`registry.py:242`): diff-brep's
  `pyproject.toml` registers its kinds in the existing `cadjoint.plugins`
  group.

**D5 — the in-process transport in the registry, and diff-brep's
capabilities as plugin kinds, rather than a separate `cadjoint.pro` entry
point.** For: one discovery mechanism, one `plugins.toml`, the same
`probe()`/staleness story, and the Tesseract form of any component is a
transport choice rather than a rewrite. Against: the registry was
"deliberately thin". The memo's view is that it is thinner *with* this
than with a second discovery path beside it. Recommend yes.

### 2.2 Which component gets which transport, and why

| kind | component | transport(s) | why |
|---|---|---|---|
| `mesher`, `tetfill`, `thermal_solver`, `elastic_solver`, `flow_solver` | the six public Tesseracts | Tesseract (local / container / remote), as today | coarse, once per step, licence boundaries (ccx), conda stacks (petsc) |
| `tet_mesher` | `tet_gmsh` — **public** | Tesseract, as today | GPL boundary; once per mesh |
| `node_map` | `diff_brep.nodemap` — **private** | **python** (primary): the map runs inside the jitted frozen objective (`optimize.py`) and `recompute_tet_points`, once per step but *traced*, so its `custom_vjp` must be a JAX primitive in the caller's program. **Tesseract** (generated, secondary): for a chain whose every stage is remote, `apply(scene_source, parameter_values, owned) → positions`, `vjp` w.r.t. `parameter_values`; it execs the scene source the way `_compile_worker` does, which is fine on the user's own cluster | one call per optimizer step is Tesseract grain (~10 ms) when the *rest* of the chain is already remote; in-process grain when it is not |
| `feature_edges` | `diff_brep.edges` — private | python only | per compile, display segments, no derivative; needs the live scene |
| `brep` | `diff_brep.graph.extract_brep` — private | python only | returns a live object with callables; consumed by `step_export`, `drag`, the private PLC |
| `step_export` | `diff_brep.step` — private | python only (a job) | a file; in-process is a function call, a Tesseract would ship a scene to get a path back |
| `drag` | `diff_brep.drag` — private | python only | interactive, sub-second, needs the scene and the constraints |
| (none) | kernel, tracer, census, ownership tag | not plugins at all | library code called inside the above |

### 2.3 Discovery: how `cadjoint` finds `diff_brep`

diff-brep's `pyproject.toml`:

```toml
[project]
name = "diff-brep"
dependencies = ["cadjoint>=0.2,<0.3", "numpy", "scipy"]

[project.entry-points."cadjoint.plugins"]
node_map      = "diff_brep.plugins:NODE_MAP"        # PluginSpec(kind="node_map", transport="python", object=...)
feature_edges = "diff_brep.plugins:FEATURE_EDGES"
brep          = "diff_brep.plugins:BREP"
step_export   = "diff_brep.plugins:STEP_EXPORT"
drag          = "diff_brep.plugins:DRAG"
```

`import diff_brep` is never written in the public tree. Public code asks
`plugin_for_kind("node_map")` and gets the private object or the
registry's own `KeyError("No plugin registered for kind 'node_map'")`.
Direct use from a script is ordinary: `from diff_brep import extract_brep,
save_brep_step, drag_handle, node_positions`.

**`cadjoint/tier.py` is the one place that spells "not installed".**

```python
KINDS = ("node_map", "feature_edges", "brep", "step_export", "drag")
def status() -> TierStatus       # per kind: filled by whom, version, compatible, reason
def require(kind) -> Plugin       # plugin_for_kind, or TierUnavailable(kind, reason)
class TierUnavailable(RuntimeError)   # message written for a public user
```

`require` also checks `plugin.probe().version` against
`cadjoint.tier.CONTRACT_VERSION` (bumped when `contracts.py` changes), so a
stale diff-brep reports "installed but built for cadjoint 0.2; found 0.3"
rather than failing inside a trace. The viewer surfaces `status()` in
three existing places: the compile payload gains an optional
`tier: {node_map: bool, feature_edges: bool, step_export: bool}` (a
pydantic field with a default, so the generated TS is a no-op for old
clients); the title block prints `EDGES LATTICE · DIFF-BREP NOT INSTALLED`
under the mesh line; `GET /api/capabilities` (three lines in `_http.py`)
returns `status()` for the Processes window.

### 2.4 The contracts

**`node_map`** — the centrepiece.

```python
class NodeMap(Protocol):                                    # cadjoint/plugins/contracts.py
    version: str
    def positions(self, scene_fn, params: Params, owned: OwnedNodes) -> Array:   # (P, 3), traced in params
        ...
```

Payload in: `scene_fn` (the functionalized scene, `functionalize_parametric`,
so parameters are arguments and the patch table can be rebuilt under the
trace), the parameter pytree, and the public `OwnedNodes` record of 1.2.
Payload out: `(P, 3)` positions, differentiable in `params`.

Contract (the docstring block 3.3 requires):

- *Preconditions*: `owned.seeds` were produced at `owned.design`; every
  `patches` row names patches of `scene_fn`'s table; midsides' parents
  precede them (the `TetMesh` layout, `tetmesh.py`).
- *Forward*: boundary nodes with arity `k ≥ 1` — one `k`-field Gauss–Newton
  from the seed onto their owning patches, batched by arity
  (`project_batched`); the seed chooses the branch, never the position on
  it. Blend nodes — one field, the scene itself. Order-2 midsides — solved
  on their own patch set (not the chord midpoint), which at the first call
  moves a midside meshed on a facet by at most the chord error (§5.4:
  1.4e-3 on the plate's bore, 2.9e-2 for blend nodes). Interior nodes —
  Laplacian follow of the boundary displacement, `k` passes over the
  node adjacency (`_smoothed`), linear in the displacement.
- *Frozen*: topology, ownership, arity, adjacency, the number of passes.
- *Differentiable*: `params` only, by the IFT — `dx* = P dx₀ −
  Jᵀ(JJᵀ)⁻¹ ∂f/∂θ dθ` per node with the tangential projector `P` dropping
  the seed's tangential motion; the interior follow contributes its
  transpose; a node whose Gram fails the transversality guard carries zero
  derivative and is counted in `stats`. The `custom_vjp` is
  `project.py:162-203`, moved private.
- *Postconditions*: at `owned.design`, `|positions − seeds| ≤ bar` for
  every patch-owned corner node (asserted, the §5.4 diagnostic that found
  both ownership bugs); every boundary node satisfies `|f| ≤ tol` on its
  patches.
- *Refuses*: an `OwnedNodes` whose `bar` differs from the map's, a patch
  index outside the table, a design hash mismatch without `resnap=True`.

**`feature_edges`** — in: scene root, `GridSpec`, design-leaf mask,
`blend_cells`; out: `EdgeSet{polylines: list[(k,3)], closed, patches (n,2),
kind, residual, vertices (n,2), stats}` (NumPy, no derivative). The
private side runs its own DC pass on the overlay grid; `stats["mesh"]`
lets the public wire layer reuse it so the pass is not doubled (the public
test keeps counting calls, as `test_edge_overlay_brep.py` does now).

**`brep`, `step_export`, `drag`** — `extract(scene, grid, **opts) -> BRep`
(opaque), `step_export(scene, grid, path) -> report`, `drag(scene, brep,
kind, index, target, **opts) -> DragResult`. The `BRep` never leaves the
process.

**`tet_mesher`** — public, its wire contract unchanged except for the
`geometry`/`geometry_format` input of 1.2 and `OwnedNodes` being computed on
the caller's side, as ownership already is (`tesseract_api.py:26-34`).

### 2.5 The public FEM path without diff-brep

`SimMesh(method="tet10", mesher="gmsh")` builds through `tet_mesher`
(discovery once, in-process or container), tags `OwnedNodes` by residual,
and returns a `TetMesh` with `owned` attached and `positions_fn = None`.
Then:

- `study.solve()` — works; nothing in a solve moves a node.
- `mesh.inspect()`, the Meshes window, VTK export — work; the payload
  carries `frozen_geometry: true` and the window shows `GEOMETRY FROZEN ·
  DIFF-BREP NOT INSTALLED` where it shows the refinement rung today.
- `Optimization` on a study over that mesh, or any `jax.grad` reaching
  `recompute_tet_points` — `TierUnavailable("node positions of a Gmsh mesh
  do not follow the design without diff-brep: the mesh is frozen geometry.
  Use mesher='tetgen' for a differentiable mesh, or install diff-brep.")`,
  raised before the trace, at `Optimization` validation; the Optimize
  window disables Run with that text as the reason.
- `mesher="tetgen"` (the default) — unchanged, differentiable through
  `project_points` as today.

**D6 — no lossy public gradient for a Gmsh mesh.** The public tier *could*
move every boundary node of a Gmsh mesh by the arity-1 `project_points`
onto the scene surface and call that a gradient; it would slide crease
nodes off their creases and put midsides back on chords, silently. A
refusal that names the tier is better than a derivative that is quietly
wrong. Recommend no lossy fallback.

**D7 — `SimMesh(mesher="gmsh")` becomes a public keyword.** Its
implementation is public now, so the earlier rule "no public API whose only
implementation is private" is satisfied; what is private is a derivative,
and 2.5 says exactly when it is missing. Recommend yes.

### 2.6 How the derivative crosses

| kind | transport | where the adjoint runs | what crosses |
|---|---|---|---|
| `node_map` | python | in the caller's trace: the private `custom_vjp` is a primitive of the public jitted objective | Python objects in, a traced array out |
| `node_map` | Tesseract (generated) | inside the served process, on `parameter_values`; `tesseract_jax` lifts it as today | scene source, flattened parameters, `OwnedNodes` arrays |
| `tet_mesher` | Tesseract | none in the mesher (topology is discrete); positions come from `node_map` on the caller's side; the frozen call's VJP w.r.t. `node_positions` is the identity's transpose (`tesseract_api.py:52-55`) | geometry text, integer topology |
| `feature_edges`, `step_export` | python | none (display segments; a file) | NumPy; a path |
| `brep`, `drag` | python | `drag.py`'s split of autodiff `∂f/∂x` from concrete `∂f/∂θ`, in-process | live objects |

### 2.7 Version pinning across the seam

- `cadjoint` gets a real version (`0.2.0` at the split; `pyproject.toml:6`
  says `0.1.0`), and `cadjoint.tier.CONTRACT_VERSION = 1` bumps only when
  `contracts.py` changes.
- diff-brep declares `cadjoint>=0.2,<0.3`; each of its plugin objects
  carries `contract_version = 1`; `require()` refuses a mismatch with a
  reason.
- diff-brep's CI installs cadjoint at the pinned tag **and** at `main`
  (allowed to fail), so a public change that breaks the seam is seen the
  day it lands. cadjoint's CI never installs diff-brep and never references
  the private URL; the developer's environment installs both editable
  (`uv pip install -e ~/code/diff-brep` into the cadjoint venv, or the
  reverse). A deployment installs `diff-brep @ git+ssh://git@github.com/
  andrinr/diff-brep@v0.2.0` with a deploy key, or from a private index
  later.

---

## 3. The cleanliness layer for `diff-brep`

Stricter than the public standard (ruff `E W F I B C4 UP ARG SIM NPY`,
line 100, no type checker, `>=3.9`). Every rule below is machine-checked
in CI or it is not a rule.

### 3.1 Layout

```
diff-brep/
  pyproject.toml            # name diff-brep; src layout; requires-python >=3.11;
                            # deps cadjoint>=0.2,<0.3, numpy, scipy; extras: stepcheck, dev
  uv.lock                   # committed; CI runs --frozen
  LICENSE                   # D8
  README.md
  .github/workflows/ci.yml
  .pre-commit-config.yaml   # ruff pinned to the same version the lock installs
  src/diff_brep/
    __init__.py             # __version__, the public names, nothing else
    py.typed
    plugins.py              # the five PluginSpecs (kind, transport="python", object=...)
    kernel/                 # project.py (Newton + custom_vjp), damped.py (LM), batched.py
    graph/                  # ownership.py, faces.py, edges.py, vertices.py, fits.py, brep.py
    tracing/                # trace.py, classify.py, census.py (interval), seeds.py (snap census)
    step/                   # loops.py, surfaces.py, writer.py
    drag/                   # inverse.py, events.py
    plc/                    # plc.py
    nodemap/                # nodemap.py (the map + VJP), follow.py (interior), tesseract_api.py (generated wrapper)
    edges/                  # feature_edges.py (the overlay's graph half)
  tests/
    axioms/                 # catalogue.py, measure.py, render.py, test_axioms.py — the gate
    property/               # hypothesis
    parity/                 # public-kernel equivalences, FD adjoint tables
    artifacts/              # the 40-scene overlay battery at 1.000
    nodemap/                # the §5.4 derivative table, the postconditions
  research/                 # every numbered claim carries a [T: …] tag (3.4)
  docs/                     # quarto; rendered privately (or a PDF per release)
```

`graph.py` (1125 lines) becomes five files because the private standard
caps a module at ~400 lines and one responsibility. `diff_brep` never
imports `gmsh` (3.5) and never imports the viewer.

### 3.2 Typing and lint

- **pyright `strict`** on `src/` (`reportMissingTypeStubs = false` for
  `tetgen`), `basic` on `tests/`. `jax.Array` / `np.ndarray` everywhere;
  `Any` needs a `# why:` on the same line (a ruff `ANN401` allowlist
  checked by a 20-line script).
- ruff with the public set **plus** `D` (Google convention), `ANN`, `PT`,
  `RUF`, `PL` (`PLR2004` on: every tolerance is a named constant with a
  docstring, as `_TANGENT_FLOOR` and `_EDGE_RESIDUAL_FRACTION` already
  are), `TRY`, `ERA`, `T20`, `TID251` (banned imports: `gmsh`,
  `cadjoint.viewer`). Line 100, the same formatter.
- `from __future__ import annotations`; frozen dataclasses for every
  record; no mutable module state except caches behind a lock (the
  `registry.py` pattern).
- pre-commit and the lock pin the *same* ruff (the public repo's 0.8.0 vs
  0.14.13 skew is a recurring cost).

### 3.3 Docstrings are contracts

After `Args/Returns/Raises`, every public function has a **Contract**
block in fixed order — *Preconditions*, *Postconditions* (with the
tolerance named), *Frozen* (which decisions are discrete and taken here),
*Differentiable* (which arguments, through what: "IFT via `project`",
"none: LM placement", "pass-through"), *Refuses*. It is the rule
`research/editing-operations.md` already imposes on the patch operations
(memory: editing-operations-rigor), applied to geometry; 2.4 shows the
form. An AST check in CI fails a public function without the block.

### 3.4 The gate

- **The axiom battery is the merge gate.** `tests/axioms/test_axioms.py`
  on every PR (3 min 47 s today), `xfail(strict=True)` throughout. A PR
  that flips an xfail updates the research table and re-renders the
  gallery in the same PR; a PR that adds one names its taxonomy item
  (`brep-axioms.md` §3). Nineteen cases and three offsets are the floor; a
  new primitive or boolean adds a case before it merges.
- **Property tests** (hypothesis) for invariants the notes state as
  sentences: the tangential projector (a tangential seed move does not
  change `project`'s answer), loop closure of every face, `simplify_loop`
  symmetric under reversal (the §3.3 shell bug), ownership stable under
  lattice offset, Euler characteristic per case, the node map's
  postcondition at the meshing design.
- **Parity tests, reference-first** (`native-mesher.md`'s lesson): `project`
  ≡ `project_points` (arity 1) to one ulp and ≡ `_project_to_seam` (arity
  2) to `rel 1e-6`; the FD tables of `brep-architecture.md` §1.1 and §5.4
  as asserted numbers.
- **No research prose without a matching test.** A numbered claim in
  `research/*.md` carries `[T: tests/…::name]`; a CI script greps the tags
  and fails on a missing test. Timings are tagged `[wall]` and exempt.
- **Render and look.** CI regenerates the gallery as an artifact on every
  PR touching `graph/`, `tracing/` or `edges/`; the reviewer opens it
  (the standing visual-QA rule, made mechanical).

### 3.5 Licensing — precisely what may link to what

- `cadjoint`: Apache-2.0, unchanged. It now hosts the Gmsh route, and the
  existing rule holds: nothing under `cadjoint/` imports `gmsh` at module
  scope (`mesh_gmsh.py:180,195,211` are function-local imports and stay so
  in `fem/gmsh.py`); the supported production route is the
  `cadjoint_tet_gmsh` image where the licence boundary is a process
  boundary; `pip install 'cadjoint[gmsh]'` links it into your own process
  "knowingly" (`pyproject.toml:53-58`). Calling a GPL library from
  Apache-2.0 code in the user's own process is the user's choice, as the
  extra's comment already says; distributing the *image* makes the image a
  GPL work, and the image contains only Apache-2.0 code plus Gmsh, which
  is compatible with GPL-2.0-or-later via v3.
- `diff-brep`: **D8 — decided: proprietary, all rights reserved.** No
  open-source or source-available licence for now; the LICENSE file grants
  no rights, distribution is by private git access only, nothing goes to a
  public index, and — since Python wheels ship source — no wheel leaves the
  private repo's release page either.
- **`diff_brep` never imports `gmsh`** (`TID251`), and the `tet_gmsh`
  image never contains `diff_brep`. The generated `node_map` Tesseract
  image contains `diff_brep` and *no* Gmsh, so no private code ever shares
  a process with GPL code. After the split this is true by construction
  and checkable by `grep`; today `tesseract_api.py:130` imports a module
  (`mesh_gmsh.py`) that also holds the node map, which is the one thing to
  fix before anything is distributed.
- OCCT (`cadquery-ocp`, LGPL with exception): dev-only `stepcheck` extra in
  both repos, used to validate files, never on a shipped path.

### 3.6 CI matrix

| axis | values |
|---|---|
| Python | 3.11, 3.12, 3.13 (`tomllib`, `StrEnum`, pyright strict all want 3.11) |
| OS | `ubuntu-latest` and `macos-latest` — TetGen topology is platform-dependent (`plugins.qmd:420`), so both are required |
| `JAX_ENABLE_X64` | off (product) and on (parity / FD jobs) |
| cadjoint reference | pinned tag (required); `main` (allowed to fail) |
| extras | `stepcheck` where wheels exist; the `gmsh` extra is cadjoint's, installed for the nodemap tests on ubuntu only |
| jobs | lint + pyright (2 min), unit + property (~8 min), axioms (~4 min), artifacts (~10 min), nodemap FD table (~3 min), build the `node_map` Tesseract image (ubuntu), docs render |
| runners | GitHub-hosted are fine for a private repo; images to GHCR under the org, private |

### 3.7 Reporting a private-tier bug without leaking code

- Public issues are filed on `cadjoint` with the scene source (source is
  the truth in this app, so it *is* the repro), the public version, and
  `cadjoint.tier.report()` — a JSON of versions, which kind failed, the
  `TierError` code, graph *counts* (`BRep.stats`) and timings. No private
  frames: `_compile_worker`'s error path collapses frames under
  `diff_brep/` to one line `diff_brep: <Code>: <message>`, and messages
  are written to be shown to a public user (they name a taxonomy item, not
  a function).
- Private reproduction: the scene file plus the report JSON; a new failure
  becomes an axiom case.

---

## 4. C / Rust — evaluated honestly

### 4.1 What the Rust core taught, in one line

It saved 5 ms of a 5 650 ms request (`native-mesher.md`, `performance.md`
§6.4), cost a toolchain, a 1.39 GB image, a parity suite and a plugin
boundary around a 2 ms kernel, and was retired in `1390239`. "Port only
what the profile says dominates a *request*, not a micro-benchmark."

### 4.2 The candidate hot loops, with what is actually known

Numbers from `brep-architecture.md` (§1.2, §5.4, §8.1), `brep-axioms.md`
§2.1, `performance.md` §1.2/§6.4, all on the same Apple-Silicon machine.
"Caller" marks the one figure handed to this memo that is not in a note.

| loop | tier | measured share | what the time *is* | would Rust/C help? |
|---|---|---|---|---|
| dual-contouring pass (discovery) | public | 4.2 s of 17.6 s (thermal body, 40³); 1.0–1.7 s of 5.1 s per axiom case; overlay `edge_hermite_data` 0.92 s warm / 20.8 s cold, `sample_grid` 0.09 s warm | JAX bisection + Newton (the exactness contract) and XLA compile; the discrete stages are 7.8 ms in NumPy | **no** — settled last week |
| batched projection kernel (`project_batched` by arity, `_own_patch`, face fits) | private | the rest of the 17.6 s: "entirely dispatch" (83.5 s unbatched → 17.6 s batched, same answer); after `1390239` the kernel compiles its loop and the end-cap overlay fell 215 s → 32.5 s; the starter *warm* row regressed 4.3 → 5.2–6.0 s because "the compiled programs are built per call and a second call recompiles them" (§8.1) | XLA compile per call + Python tracing; the arithmetic is nanoseconds | **no** — JAX already compiles it; what is left is a compile cache (§9.6) |
| edge tracing predictor–corrector (`trace_curves`) | private | not isolated in any note; inside the starter's 7.7–8.6 s cold / 5.2–6.0 s warm and the end-cap's 27–32.5 s | one jitted `advance` program dispatched once per step for a batch of curves — sequential, data-dependent, tens to hundreds of small dispatches; the Python bookkeeping around it is milliseconds (the overlay's four Python loops total ~10 ms, §1.2) | **maybe, later** — the one loop XLA fits badly, but `lax.while_loop` over the whole trace is the first fix and needs no new language (4.4) |
| interval / affine census (planned, `brep-edge-tracing.md` §5) | private | estimated there: one batched program at ~4× `sample_grid` per patch (`sample_grid` is 0.7 % of a request) × 28–50 patches; bisection "on the few cells that need it" | as the §5.3 jaxpr interpreter, JAX expresses it fine; what JAX cannot express is the *recursive* bisection and Keeter's tape shortening | **no for the census, open for the bisection** — a Fidget-style evaluator is the second evaluator §12.10 declined; count the ambiguous cells first (4.4) |
| Gmsh ownership tagging (`assign_ownership`, now public) | public | 0.20 s on the plate, one batched JAX call per patch (§5.4) | dispatch | **no** |
| node map (`recompute_gmsh_points`): per-arity projections + Laplacian follow | private | inside the §5.4 numbers; the follow is a sparse mat-vec per pass | the same batched programs as row 2 | **no** |
| STEP writing (`save_brep_step`) | private | 4 ms on the plate; ~21 000 entities on the thermal body well under a second | string formatting | **no** |
| Gmsh itself | GPL process | 50 ms on the plate | already C++ | — |
| `extract_brep` inside a Gmsh run | private | plate 8.7 s, thermal body 18 s, end-cap 53.5 s (§5.4); *caller: 15.6 s of a 22 s Gmsh meshing run* | the DC pass + batched projections | **no** — same as row 2; and with the public STL route (1.2) the public tier does not pay it at all |
| Python graph bookkeeping (`_components`, `_region_loops`, `_quad_adjacency`, `_chain_segments`, `_link_edge_vertices`) | private | **never measured on its own**; O(quads) dict/set loops, 6 522 quads on the thermal body at 40³ | Python | **not until measured** — by the overlay's precedent it is milliseconds; profile the end-cap at 64³ (4.4) |

What is left after JAX is three things: (a) compile cost paid per call for
programs that could be cached — the biggest, a caching problem; (b)
sequential control flow in the tracer — a `while_loop` problem first; (c)
data-dependent recursion in a future census — NumPy on few cells unless
the few turn out to be many. None of them is arithmetic.

### 4.3 If any of it were Rust: the binding, the cost, the derivative

| binding | fits | cost in a private repo | differentiability |
|---|---|---|---|
| **PyO3 + maturin** | a library kernel called thousands of times per request | Rust toolchain in CI, a wheel matrix (linux x86-64, macOS arm64), private wheel index, a pure-Python reference kept as the oracle with reference-first parity tests — everything `native-mesher.md` listed minus the plugin boundary | none of its own |
| ctypes cdylib | the retired shape | no wheel story (the `.so` was a git-ignored artefact copied into worktrees by hand) | none |
| Tesseract | a service: a licence or language boundary with a data contract | the runtime, ~0.14 s per apply/VJP, JSON+base64 | a hand-written `vector_jacobian_product` |
| C++ via pybind11 | only if OCCT's C++ API is ever the point (B-spline fitting, sewing) — `cadquery-ocp` already exposes it | C++ toolchain, OCCT build | none |

The derivative is the decisive constraint and has one clean answer. A Rust
kernel that *places* points needs a hand-written VJP, as the retired
Tikhonov QEF had — and it needed a Tesseract to reach `jax.grad`. But
`brep-edge-tracing.md` §6 already separates the roles: **topology and
sampling are discrete and frozen; positions are re-solved by `project`
under the trace**. A Rust tracer whose output is `(samples, pairs, vertex
ids, classifier state)` costs no differentiability at all — JAX
re-projects the frozen samples with the IFT adjoint, exactly as the node
map re-solves Gmsh's frozen nodes. The only Rust that would ever be written
here is *discrete-only*, and it needs a forward-only evaluator of `f_a,
∇f_a` at every step without a callback per step (which gives the dispatch
cost back):

1. **A quadric patch table.** The patch fields are analytic — planes,
   cylinders, spheres, cones, the half-planes of an extruded polygon, all
   under affines — so most of the table is a world-frame quadric
   `xᵀAx + bᵀx + c` with closed-form gradients (a torus is quartic;
   `round_box`/`capsule` are offsets/unions of quadrics; blends are not
   patches; the loft's ruled surfaces of §9.2 are hyperbolic paraboloids,
   also quadrics). Export coefficients, trace in Rust. Exact for the
   mechanical-part cases the battery covers; fails for user SDFs.
2. **A tape evaluator** (Fidget/libfive): compile a patch's jaxpr to
   bytecode and interpret or JIT it in Rust. General, and it is the second
   evaluator `performance.md` §12.10 declined to build.

Route 1 is the only one worth an experiment.

### 4.4 Ranked recommendation, and the experiment that settles each

1. **Cache the compiled projection programs** per `(patch count, arity,
   point-batch bucket)` in the process, pad point batches to buckets so
   shapes repeat (`brep-architecture.md` §8, §9.6). Expected: the starter's
   warm overlay from 5.2–6.0 s back toward 4.3 s; axiom cases from 5.1 s
   toward the 1–1.7 s DC floor. *Experiment:* `extract_brep` twice in one
   process on the thermal body at 40³ and each axiom case; report
   second-call wall time and the `pjit` dispatch count. If the second call
   is not within 1.5× of the DC pass, the cache is missing and nothing
   below matters yet. **Not Rust.** Half a day.
2. **Move the tracer's loop into `lax.while_loop`**, one edge batch per
   dispatch, guards as loop state. *Experiment:* dispatch count and wall of
   `_traced_polylines` on starter / bracket / end-cap before and after;
   parity of samples to 1e-9 after re-projection. **Not Rust.** One to two
   days.
3. **Only if (2) leaves tracing above ~30 % of an overlay: a Rust
   discrete tracer over a quadric patch table** (PyO3, maturin, private
   wheels), output `(samples, pairs, vertex ids)`, positions re-solved in
   JAX. *Experiment:* a 300-line prototype on the plate and the fin comb
   from an exported quadric table; wall against (2); every sample
   re-projects by `project` to < 1e-9; every battery count unchanged. Two
   to three days; go/no-go on the number.
4. **Interval census: JAX first** (the §5.3 jaxpr interpreter, one
   program). *Experiment:* implement the census only; count ambiguous cells
   per axiom case at 32 and 64. Under 1 % of crossed cells: bisection in
   NumPy is free and Rust is off the table. Thousands on some case: revisit
   a Fidget-style evaluator — as an *evaluator* decision, which is far
   bigger than a tracer.
5. **Never:** the DC stages (measured), STEP writing (4 ms), ownership
   tagging (0.2 s of dispatch), the node map (batched programs), Gmsh
   (already native), and any kernel that computes a differentiable
   position.

**Rust over C** wherever a native piece is ever justified: memory safety in
a geometry kernel, `rayon`, maturin's wheel story, and the retired core in
git (`native/src/core.rs` at `a4eb963`) as a known-good start. C/C++ only
where an existing C++ API is the point (OCCT), which `cadquery-ocp` already
supplies from Python.

**D9 — accept this ranking (1 and 2 in JAX before any Rust; 3 gated on a
measured share; 4 gated on a count): yes/no.** The user asked to "eval
writing more things in C/Rust"; the eval says *measure first*, and given
what was measured a week ago the memo would be dishonest to say otherwise.

---

## 5. Migration plan — against the empty `andrinr/diff-brep`

Ordered so both trees are green at every step and the public tree never
holds a half-cut import. Effort is focused days for one person who knows
the code; steps 1a–1e can be parallelised by agents with disjoint file
ownership (repo-operations memory). Other agents are live in this tree on
`README.md` and the banner assets; step 5's README edits wait for that
work to land.

| # | step | verified before moving on | effort |
|---|---|---|---|
| 0 | Answer D1–D12; pick the licence (D8) and the public version (`0.2.0`). | — | 0.5 d |
| 1a | **Public pre-split refactor, no deletions.** Promote `meshing/export.py`'s `_STEP_HEADER`, `_STEP_BOILERPLATE`, `_step_real`, `_weld_degenerate_edges` to `cadjoint.meshing.step_scaffold`; add hole loops to the planar merger (1.2 option i). | `pytest tests -q --ignore=tests/fem` green (~7 min); both STEP kernel tests; the plate's merged STEP has 7 faces in OCCT. | 1 d |
| 1b | **Make the Gmsh route public.** Move the mesher, `assign_ownership` (re-based on residual-only tagging, 1.2), `GmshMesh`, `OwnedNodes`, `tet_mesh_from_gmsh` (static) to `cadjoint/fem/gmsh.py`; leave `recompute_gmsh_points`, `_smoothed`, `_node_adjacency`, `parameterised_points` in `cadjoint/brep/` for now; `tet_gmsh/tesseract_api.py:130` → `cadjoint.fem.gmsh`; the `geometry`/`geometry_format` input; the STL route (1.2 option iii) as `geometry_format="stl"`; `SimMesh(mesher="gmsh")` (D7) with `frozen_geometry` in the inspect payload. | `tests/fem/test_gmsh.py` (the static half of `test_mesh_gmsh.py`) green with the `gmsh` extra; `test_tesseract_packaging.py` green; the plate meshes from STL with worst radius ratio reported and compared to the analytic-STEP number (0.3191) — record it; ownership counts on the plate from residual-only tagging equal the vote's (`{0: 1098, 1: 1424, 2: 196, 3: 8}` at `target_size=0.16`, `docs/brep.qmd`). | 2 d |
| 1c | **The registry's in-process transport** (2.1): `PluginTransport.PYTHON`, `PythonPlugin`, `contracts.py` with the five Protocols and `OwnedNodes`; `cadjoint/tier.py`. | `tests/plugins/test_registry.py` + new `test_python_transport.py` (a stub object registered by kind, `probe`, `as_jax` returns the object's callable). | 1 d |
| 1d | **Wire the seams** with the private code still in-tree, registered *from inside cadjoint* as a temporary `PluginSpec` (`object="cadjoint.brep.plugins:…"`): `_mesh_edge_payload` → `feature_edges` else the lattice layer restored from `a4eb963` (batched form); `_write_geometry` → `step_export` else faceted with `tier`; `recompute_tet_points` → `node_map` for Gmsh meshes else `TierUnavailable` at `Optimization` validation; the `tier` payload field; `/api/capabilities`; the title-block line. | The suite **twice**: as is, and with the temporary specs unregistered (a `conftest` flag). Both green; `test_edge_artifacts.py` gains its lattice-baseline rows (recorded, not asserted at 1.000); the starter's Optimize window refuses a Gmsh-meshed study with the message; screenshots of the starter's feature edges in both states, looked at; `npm test` and e2e unchanged. | 2 d |
| 1e | Split `test_edge_overlay_brep.py` into the public fallback tests and the private graph tests; commit 1a–1e as one public PR ("Put the derived B-rep behind a seam"); tag `v0.2.0-seam`. | CI green on the PR. | 0.5 d |
| 2 | **Fill `diff-brep` with history.** From a fresh clone (filter-repo refuses a non-fresh one): `git clone --no-local ~/code/jaxcad <scratch>/tier-filter && cd <scratch>/tier-filter && git filter-repo --path cadjoint/brep --path tests/brep --path tests/viewer/test_edge_overlay_brep.py --path research/brep-architecture.md --path research/brep-edge-tracing.md --path research/brep-axioms.md --path research/brep-axioms --path docs/brep.qmd --path-rename cadjoint/brep/:src/diff_brep/ --path-rename tests/brep/:tests/ --path-rename docs/brep.qmd:docs/index.qmd`, then `git remote add origin git@github.com:andrinr/diff-brep.git && git push origin main`. The paths carry 1–5 commits each (`cadjoint/brep` 4, `tests/brep` 3, the notes 1–4), so history is thin — filter-repo still buys the gallery's and the measurements' commit trail for an hour's work, where `subtree split` would drag the 33.7 MiB pack. **First commit on top** ("Bootstrap diff-brep"): `pyproject.toml` (3.1), `uv.lock`, `LICENSE` (D8), `README.md` (what it is, how it is found, how to install it from git), `.gitignore`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` (3.6), `src/diff_brep/__init__.py`, `py.typed`, `src/diff_brep/plugins.py` (the five specs), `tests/conftest.py`, `research/README.md`; a mechanical `sed` of `cadjoint.brep` → `diff_brep` and the deletion of the now-public half of `mesh_gmsh.py`. Branch `main`, protected; work on `refactor/*` branches by PR, as in the public repo. | `uv sync --frozen` in the new repo against `cadjoint @ git+https://github.com/andrinr/cadjoint@v0.2.0-seam`; `pytest` green (brep, axioms 45 passed / 38 xfail, the overlay graph tests, the §5.4 derivative table). | 1 d |
| 3 | **Cleanliness pass in diff-brep** (§3): the layout split, pyright strict (the bulk: ~4 300 lines of NumPy/JAX with `Any` at the seams), the ruff set, Contract blocks, the claim-tag script over the three notes, hypothesis tests for the six invariants, the generated `node_map` Tesseract package, CI matrix, GHCR build. | pyright 0 errors; ruff clean; axioms unchanged; artifacts at 1.000; the `node_map` image builds and a transport test drives it against the in-process object with identical positions and VJP to 13 digits (the `flow_brinkman` precedent). | 4–5 d |
| 4 | **Delete from cadjoint.** Remove `cadjoint/brep/`, the temporary specs, the private overlay half, `tests/brep/`, the moved research notes and `docs/brep.qmd`. | cadjoint CI green **without diff-brep installed** — the only state it ever runs; diff-brep CI green against cadjoint `main`; `grep -rn "cadjoint.brep\|diff_brep" cadjoint tests` returns nothing. | 0.5 d |
| 5 | **Docs split.** Public: `docs/tier.qmd` (what diff-brep adds, `cadjoint.tier.status()`, discovery, each degraded behaviour); a "Gmsh tet10" section in `docs/simulation.qmd` (from `brep.qmd`'s "Meshing from the graph", minus the derivative) replacing lines 66–70; `docs/plugins.qmd:329-346` reworded around `OwnedNodes` and the `node_map` kind; `meshing.qmd:162-165,197`, `viewer.qmd:301,318`, `getting-started.qmd:144,265,342`; `_quarto.yml` loses the `B-rep — *` API sections and the three research entries, gains `fem.gmsh.*`. README: 45–53 (the B-rep paragraph becomes two sentences ending "in diff-brep"), 74 and 80, 258–264 (keep the figure, caption it as the private tier's edges), 436–447 (analytic STEP is diff-brep's; `cadjoint.fem.gmsh` is public), 840–842. Private: `docs/index.qmd` is the pro guide; rendered to a private site or a PDF per release. | `quarto render` clean in both; a link checker in the docs job; the README banner work by other agents untouched. | 1–1.5 d |
| 6 | **Release mechanics.** cadjoint `v0.2.0`; diff-brep `v0.2.0` tagged, installable by `uv pip install "diff-brep @ git+ssh://git@github.com/andrinr/diff-brep@v0.2.0"`; a `plugins.toml` example pointing `node_map` at the private image for a cluster. | Fresh venv: cadjoint alone → `tier.status()` reports every kind absent and the app runs degraded; add diff-brep → every kind filled, the starter's edges come from the graph, `cool-sink` runs on a Gmsh-meshed study. | 1 d |
| 7 | **The C/Rust experiments** (4.4 items 1–4) as diff-brep milestones in that order, each number recorded in diff-brep's `research/performance.md` before the next starts. | per item | 0.5 + 1.5 + (2–3) + 1 d |

Steps 0–6: about **14–16 focused days**; step 7 is gated and separate.

**D10 — seam first in cadjoint (steps 1a–1e), then fill diff-brep.** The
alternative, copy first and cut later, leaves two live copies of the graph
for a week with every fix landing twice. Recommend the order above.

**D11 — diff-brep starts from the filtered history, not from an empty
tree.** An hour, and the axiom gallery and every measurement keep their
commit trail. Recommend yes.

**D12 — the temporary in-tree `PluginSpec`s in step 1d.** They let the seam
be verified with the private code still present, at the cost of one
throwaway `cadjoint/brep/plugins.py` that step 4 deletes. Recommend yes.

---

## 6. Decisions asked for, in one list

| | decision | recommendation |
|---|---|---|
| D1 | the boundary of 1.1: Gmsh route public, node map / kernel / graph / tracer / STEP / drag / PLC private | yes |
| D2 | `patch_fields()` and `scene_patch_fields` stay public | yes |
| D3 | the projection kernel (`project.py`, incl. the `custom_vjp` and `trace_curves`) is private | yes |
| D4 | the public Gmsh input is the DC surface as STL through Gmsh's classify/createGeometry route, with the planar-merged STEP as the exporter's own improvement | yes (one-afternoon experiment first) |
| D5 | an in-process `python` transport in the plugin registry; diff-brep's capabilities are kinds in the existing `cadjoint.plugins` entry-point group; `cadjoint.tier` is the one status/refusal module | yes |
| D6 | no lossy public gradient for a Gmsh mesh — refusal with the reason instead | yes |
| D7 | `SimMesh(mesher="gmsh")` becomes a public keyword | yes |
| D8 | diff-brep's licence | **decided 2026-09-03: proprietary, all rights reserved — no open-source or source-available licence for now.** The LICENSE file states that no rights are granted; installs are by private git access only; no public index. |
| D9 | C/Rust ranking: cache, then `while_loop`, then a gated discrete Rust tracer, then a gated census evaluator | yes |
| D10 | seam first in cadjoint, then fill diff-brep | yes |
| D11 | diff-brep's `main` starts from `git filter-repo` history | yes |
| D12 | temporary in-tree specs while the seam is verified | yes |

---

## 7. Status

**2026-09-03 — migration step 1 (the seam) is built inside `cadjoint`.**
Nothing has moved yet: `cadjoint/brep/` is still here, `diff-brep` is still
empty, and no commit has been pushed to it. What landed is everything the
seam needs so that the move is a deletion rather than a redesign.

| memo | landed as |
|---|---|
| 2.1 D5 — one contract, two transports | `PluginTransport.PYTHON`, `PythonPlugin` (`plugins/plugin.py`), `PluginSpec.object` + `_import_object` (`plugins/spec.py`), `BUILTIN_PYTHON` and `PluginRegistry.without/unregister` (`plugins/registry.py`) |
| 2.4 — the contracts | `cadjoint/plugins/contracts.py`: `NodeMap`, `FeatureEdges`, `BRepExtractor`, `StepExporter`, `Drag`, the `OwnedNodes`, `EdgeSet` and `DragOutcome` payloads, `CONTRACT_VERSION = 1`, `Differentiable[...]` and `contract_signature` |
| 2.3 — one refusal | `cadjoint/tier.py`: `KINDS`, `status`, `available`, `require`, `component`, `message`, `report`, `absent` |
| 1.1, 1.2, D1, D4, D7 | `cadjoint/fem/gmsh.py` (public: the mesher, `dc_surface_stl`, residual `assign_ownership`, `owned_nodes`, `GmshMesh`, static `tet_mesh_from_gmsh`, `sdf_gmsh_tet_mesh`); `cadjoint/brep/mesh_gmsh.py` keeps only the node map (`node_positions`, `recompute_gmsh_points`, `parameterised_points`) and the analytic-STEP-fed `gmsh_tet_mesh`; `SimMesh(mesher="gmsh")` |
| D12 — temporary in-tree specs | `cadjoint/brep/plugins.py` (five objects), registered as `python` specs while `cadjoint.brep.plugins` is importable |
| 1.3 — the twelve edges | overlay ×5 → `cadjoint/brep/edges.py` behind the `feature_edges` kind; `_export.py` → the `step_export` kind; `tesseract_api.py` → `cadjoint.fem.gmsh`; `docs/plugins.qmd`, `docs/meshing.qmd`, `docs/getting-started.qmd` reworded |
| 2.5, D6 — graceful degradation | `SimMesh.frozen_geometry`, `Optimization._refuse_frozen_geometry`, the `tier` compile field, `GET /api/capabilities`, `MeshEdgePayload.edges` |

**Numbers measured on the way (D4, the STL route).** Plate at
`target_size=0.16`, `PLATE_GRID` 20³, snap on: 2 801 nodes / 1 435 cells,
12 CAD surfaces from 3 079 facets, worst radius ratio **0.293** (median
0.742), 264 of 1 700 boundary nodes snapped, **0 blend nodes and 0 blend
surfaces** (265 and 2 without the snap), arity `{0: 1105, 1: 1494, 2: 194,
3: 8}`. The analytic STEP of the same part: 2 707 nodes, worst ratio
**0.308**, arity `{0: 1075, 1: 1428, 2: 196, 3: 8}`. Bore-owned nodes land
within **2.41e-3** of `r = 0.25` against a bar of 2.66e-3, median 1e-8.
Starter thermal body: 4 820 nodes, worst ratio 0.171 snapped / 0.175
unsnapped / 0.215 from the analytic STEP; the snap moves 963 nodes and
changes no tag (713 blend nodes either way).

Two corrections to §1.2's expectations: the ownership counts from
residual-only tagging are *near* the vote's but not equal to it
(`docs/brep.qmd` records `{0: 1098, 1: 1424, 2: 196, 3: 8}` over 2 726
nodes; the same route now gives 2 707 nodes and `{0: 1075, 1: 1428,
2: 196, 3: 8}`), so the step-1b acceptance test is "the same shape, no
blends" rather than an equality; and the STL route needs the **snap** of
1.2's own residual bar to be usable at all, which the memo did not
anticipate — without it the plate's bore reads as a blend.

**What is not done and belongs to later steps.** `docs/brep.qmd`, the
`B-rep — *` sections of `_quarto.yml`, `docs/simulation.qmd:66-70`,
`docs/viewer.qmd`, the README, and `tests/meshing/test_patch_fields.py`'s
docstring mention still name `cadjoint.brep` (memo §5 step 5, and other
agents own some of those files today). `tests/viewer/test_edge_overlay_brep.py`
still imports `cadjoint.brep.edges` directly — it *is* the private tier's
test file and moves with the code in step 2. `cadjoint.plugins.contracts`
is not yet promoted alongside `cadjoint.meshing.step_scaffold` (step 1a's
other half, the underscore names in `meshing/export.py`).
