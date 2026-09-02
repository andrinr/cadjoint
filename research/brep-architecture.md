# A derived B-rep for cadjoint — architecture, prototype, measurements

Status: working prototype in `cadjoint/brep/`, tested in `tests/brep/`
(55 tests). Everything numbered below was measured on this machine
(Apple Silicon arm64, macOS 25.4, CPython 3.14.5, jax 0.8.2 CPU backend,
`cadquery-ocp` for the STEP round-trip) on 2026-09-02; §11 says how to
reproduce each one. Nothing outside `cadjoint/brep/**` and `tests/brep/**`
was changed.

---

## 0. The one sentence

**cadjoint does not need to store a B-rep, because it can derive one — and
the derivation is differentiable.**

Every hard primitive here is a `min`/`max` over smooth *patch fields* with
exact surface ownership (`cadjoint/meshing/patch_fields.py`). So a face is
the part of one patch's zero set that survived the booleans, an edge is where
two patch zero sets meet, a vertex is where three do. Each of those is the
solution of a small system in the design parameters, and each differentiates
by the implicit function theorem. Which patches meet where is discrete and
frozen per extraction; where they meet is continuous and exact.

That single observation is what lets the three products the user asked for
share almost all of their code:

| product | what it needs | what it uses |
|---|---|---|
| clean extraction + STEP / polygon export | faces with surfaces, edges with curves, loops | `graph.py` → `step.py` |
| a draggable B-rep | the same graph, plus ∂(vertex)/∂(design) | `graph.py` → `drag.py` |
| a differentiable simulation mesh | the same graph, plus node ownership | `graph.py` → `plc.py` |

and all three go through **one** kernel, `project.py`, which is the only
place a position is computed.

---

## 1. The kernel: one Newton, three arities, one adjoint

`cadjoint/brep/project.py`.

A face point solves `f_a(x) = 0`. An edge point solves `f_a(x) = f_b(x) = 0`.
A vertex solves three. All three are the same minimum-norm Gauss-Newton step

```
x ← x − Jᵀ(JJᵀ)⁻¹ f(x)
```

which at `m = 1` is literally `f ∇f / |∇f|²` — the step
`cadjoint.fem.motion.project_points` takes — and at `m = 2` is literally the
step `cadjoint/viewer/_edge_overlay.py:_project_to_seam` takes. Those were
two implementations of one thing; there is now one, and the tests pin both
equivalences (`tests/brep/test_project.py`).

**The adjoint is the IFT, not the unrolled loop.** Write the projection as
"displace the seed inside the normal space until every field vanishes",
`x* = x₀ + Jᵀλ` with `f(x*) = 0`, and differentiate the whole system:

```
dx* = P dx₀ − Jᵀ(JJᵀ)⁻¹ (∂f/∂θ) dθ,     P = I − Jᵀ(JJᵀ)⁻¹J
```

`P` is the tangential projector of the intersection. The second term is the
parameter pull every arity shares. The first term is the interesting one: it
says a seed's *tangential* placement does not affect the answer, which is
exactly the frozen-topology contract — **the seed chooses the branch, never
the position on it**. The forward iteration runs on `stop_gradient` and a
`jax.custom_vjp` attaches this.

The guard is the repo's double-`where`: a transversality test on the Gram
matrix `G = JJᵀ`, combining the viewer's rank test
(`λ_min(G) > 10⁻² tr(G)/m`) with `project_points`'s dead-gradient floor
(`tr(G) > 10⁻⁸`, which at `m = 1` *is* `|∇f|² > 10⁻⁸`). Where it fails the
step is suppressed in both passes: the point stays, and carries no parameter
derivative.

### 1.1 Measured: the adjoint against central differences, per arity

Sphere of radius `r`, plane `z = h`, plane `x = o`; loss is a fixed linear
functional of the projected points; `h = 10⁻³` central differences.

| arity | ∂/∂radius (adjoint / FD) | ∂/∂height | ∂/∂offset | worst rel. error |
|---|---|---|---|---|
| 1 (sphere) | +0.14715713 / +0.14722347 | 0 / 0 | 0 / 0 | 4.5 × 10⁻⁴ |
| 2 (sphere ∩ plane) | −0.51912320 / −0.51915646 | +0.35973024 / +0.35977361 | 0 / 0 | 1.2 × 10⁻⁴ |
| 3 (sphere ∩ 2 planes) | −0.95850134 / −0.95850223 | +0.49492359 / +0.49489734 | +0.65253830 / +0.65249199 | 7.1 × 10⁻⁵ |

The residual disagreement is the float32 FD truncation, not the adjoint: at
arity 3, where the solution is a single well-conditioned point, the two agree
to 9 × 10⁻⁷. The zero entries are exact zeros in both, which is the guard
and the geometry agreeing (a sphere's projection cannot depend on a plane
that is not in its system).

Additionally, `project` at one field reproduces `project_points` to within
one float32 ulp, and its *gradient* to `rel=10⁻⁶`.

### 1.2 Batching, because a JAX call costs the same whatever it moves

`research/performance.md` §6.2 measured that in eager mode one projected
point costs as much as three hundred. A body has hundreds of distinct
owner-patch subsets, so "one `project` call per subset" is the whole cost.
`project_batched` evaluates every patch field at every point once per
iteration and lets each point gather its own rows — the trick
`_project_seam_groups` plays for seam groups, generalized.

**Measured on the starter's thermal body at 40³** (6522 quads, 28 patches):
one call per owner subset takes **83.5 s** for a full graph extraction;
batching by arity — three programs for the whole mesh, one for all the face
fits, one for all the edges — takes **17.6 s**, of which 4.2 s is the dual
contouring itself. Same answer, 4.7× less time, and the difference is
entirely dispatch.

---

## 2. The graph: what a face, an edge and a vertex are

`cadjoint/brep/graph.py`.

1. Dual-contour the scene. This is **discovery**, not geometry.
2. Project every quad centroid onto the *scene's* zero set, then ask which
   `(leaf, patch)` owns it, by the two-stage `argmin` of
   `patch_fields.signature_function` (nearest leaf, then nearest patch within
   it — a global `argmin` would let a distant solid's unbounded cap-plane
   field steal a point).
3. **The blend test is exact and thresholdless.** At a surface point, the
   owning patch field's value is zero if the patch really owns it, and of
   order the blend radius if the point is on a fillet. A smooth union creates
   surface that lies on *no* patch's zero set, and that is the definition
   used, not an angle.
4. Faces are connected components of same-owner quads; loops are the region
   boundary chains (all of them, so a plate keeps its bore as a second loop).
5. Edges are the chains of mesh edges separating two faces; vertices are the
   mesh vertices three or more faces meet at.
6. **Everything is then re-solved by the kernel.** Every mesh vertex is
   tagged with the patch set it belongs to (`BRep.owner_patches`,
   `BRep.owner_arity`) and placed by a 1-, 2- or 3-field projection.

Surface *type* is structural — read off the primitive's own documented patch
ordering (`Box` → planes, `Cylinder` patch 0 → cylinder and 1/2 → planes,
`RevolvedPolygon` edge `k` → cylinder / plane / cone by the profile edge's
direction) — and only the *placement* is fitted, from the face's own
re-projected samples. Every fit carries its measured residual, so a caller
can refuse a surface it cannot certify.

### 2.1 Measured: the plate (box − cylinder, hard Difference)

`Difference(Box(0.6, 0.6, 0.4), Cylinder(r = 0.25), smoothness = 0)`, 20³ grid.

| | count |
|---|---|
| patches in the scene | 9 |
| dual-contour quads | 1536 |
| **faces** | **7** — 6 `plane`, 1 `cylinder` |
| **edges** | **14** — 12 straight, 2 closed rim circles |
| **vertices** | **8** |
| blend faces / blend quads | 0 / 0 |
| ambiguous vertices | 0 |
| non-simple faces | 0 |

That is the textbook B-rep of the part, and it is exact: the eight corners
land on `(±0.6, ±0.6, ±0.4)` with residual `< 10⁻⁹`, the fitted bore radius
is `0.250000006` with fit residual `1.3 × 10⁻⁸`, and every plane's fit
residual is `< 10⁻⁹`. Extracting on a *shifted* lattice finds the same
7/14/8 — topology is discovered, not sampled.

**Dual contouring is not the geometry.** Extracted with the Tikhonov
(`sharp=False`) placement instead of the rank-revealing one, the plate's
corners carry the regularizer's mass-point bias — `4.4 × 10⁻⁵` off the true
corner. Re-solving them against their three patches puts them below `10⁻⁶`.
The graph does not inherit the extractor's placement.

### 2.2 Measured: the starter's thermal body (blends, revolve, bushings)

`Union(sink, slug, bush_a, bush_b, smoothness = 0.03)`.

| | 40³ grid | the scene's own SimMesh grid (18,13,11) |
|---|---|---|
| world-frame leaves / patches | 4 / 28 | 4 / 28 |
| dual-contour quads | 6522 | 858 |
| **faces** | **58** — 39 plane, 6 cylinder, 13 blend | **41** — 23 plane, 1 cylinder, 17 blend |
| **edges** | **127** (58 analytic, 69 blend-adjacent) | **98** (55 analytic) |
| **vertices** | **76** (32 clean triples, 44 ambiguous) | **60** (28 ambiguous) |
| blend quads | 433 of 6522 | 56 of 858 |
| worst analytic face fit residual | `< 10⁻⁵` | `< 10⁻⁵` |
| worst edge residual | 3.0 × 10⁻⁸ | — |
| worst vertex residual | 9.5 × 10⁻¹⁶ | — |
| owner arity histogram | 580 blend / 5120 face / 792 edge / 32 corner | — |

The fitted cylinders are the ones the scene declares: the slug rim at
`r = 0.26`, the slug bore at `r = 0.05` (correctly reported as a bore), the
two bushings at `r = 0.07`.

### 2.3 Where the graph is ambiguous — the honest list

- **Blends dominate near seams.** 13 of 58 faces on the thermal body are
  blend, and they are not a defect: they are real surface with no closed
  form. But a blend *also* removes the analytic trimming curve from every
  face it borders — 69 of 127 edges on the thermal body are blend-adjacent
  and have no exact curve. This is the single biggest limitation and §9.3
  discusses it.
- **Four or more faces at a point.** 44 of 76 vertices on the thermal body
  are `ambiguous`: more than three incident faces, or a blend among them.
  These are counted and reported (`stats["ambiguous_vertices"]`), never
  silently resolved by dropping a patch.
- **Tangent patches.** Two coincident or tangent zero sets have a
  rank-deficient Gram and no transversal intersection; the kernel refuses to
  move such points and `transversal()` reports it. Tested directly.
- **Sliver faces from ownership flicker.** At 40³ the first version produced
  20 one- and two-quad faces (0.3 % of area) whose plane fit had too few
  samples. Fixed by fitting on the region's *corners as well as its
  centroids*: 45 of 45 non-blend faces are now certified analytic. The
  flicker itself is still there; it is a lattice artefact and shows up as
  extra small faces, not as wrong geometry.

---

## 3. Export: analytic where the graph can certify it

`cadjoint/brep/step.py`, built on `cadjoint/meshing/export.py`'s STEP
scaffolding (`_STEP_HEADER`, `_STEP_BOILERPLATE`, `_step_real`,
`_weld_degenerate_edges`), which is imported, not modified.

Three things the merge-based mesh writer structurally cannot do:

- **Loops collapse.** A plate side face's boundary is 64 dual-contour
  vertices; after re-solving they are collinear to float noise, and the
  exported `FACE_OUTER_BOUND` has **4**. The collapse is measured, not
  assumed (`simplify_loop` keeps a deviation bound).
- **Holes survive.** A cap is one `PLANE` with a `FACE_OUTER_BOUND` and a
  `FACE_BOUND`, not a region the merger has to give up on.
- **Cylinders stay cylinders.** A full band bounded by two rim circles is a
  `CYLINDRICAL_SURFACE` with `CIRCLE` edges — and the *same* `EDGE_CURVE`
  entity is shared with the planar cap's hole loop, which is what makes the
  shell sew.

### 3.1 Measured: the plate

| | analytic | `analytic=False` (same graph, faceted) | `meshing.export.save_step` |
|---|---|---|---|
| `ADVANCED_FACE` | **7** | 3072 | 3079 |
| `PLANE` / `CYLINDRICAL_SURFACE` / `CIRCLE` | 6 / 1 / 2 | 3072 / 0 / 0 | — |
| `EDGE_CURVE` / `VERTEX_POINT` | 14 / 10 | — | — |
| total STEP entities | **181** | ~10 000 | ~10 000 |
| OCCT: solids / shells / faces | **1 / 1 / 7** | 1 / 1 / 3072 | 1 / 1 / … |
| OCCT: `BRepCheck` valid | yes | yes | yes |
| OCCT volume | **0.99492046** | 0.99649 | 0.99649 |
| exact volume `1.152 − π·0.25²·0.8` | 0.99492037 | — | — |
| relative volume error | **9.8 × 10⁻⁸** | 1.6 × 10⁻³ | 1.6 × 10⁻³ |

The analytic file's residual error is the fitted radius' last float32 digit
(`0.250000006` instead of `0.25`), not discretization: **there is no
discretization left**. The faceted file's 1.6 × 10⁻³ is the 28-gon standing
in for the bore.

### 3.2 Measured: the starter's thermal body at 40³

| | B-rep STEP | `meshing.export.save_step` on the same mesh |
|---|---|---|
| `ADVANCED_FACE` | **1181** (39 exact `PLANE`, 1142 facets) | 5521 |
| total entities | **21 318** | 98 258 |
| OCCT | 1 valid solid, 1 shell | 1 valid solid, 1 shell |
| OCCT volume | 1.158210 | 1.161262 |

4.6× fewer entities for the same solid. The 1142 facets are the blend faces
and the six cylinders whose rims a blend has eaten (§9.3).

The two solids differ in volume by 0.26 % (1.158210 against 1.161262),
because their surfaces are genuinely different points: the graph's are
re-solved onto their owner patches, the mesh writer's are the raw dual-contour
vertices. **Which is closer cannot be decided on this body**, and the
prototype does not claim it is: a sigmoid-free occupancy count of the SDF
gives 1.177 at 120³ and 1.158 at 180³, a spread far wider than the 0.003 in
question. The volume claim that *is* decidable is the plate's (§3.1), where
the answer is known in closed form and the analytic file is right to 10⁻⁷.

### 3.3 Two mistakes worth recording, because both cost a shell

Both were found by asking OCCT, not by reading the file.

1. **Order-dependent simplification splits a shared edge.** The first
   `simplify_loop` was a greedy iterative collapse. Two faces sharing one
   straight run walk it in opposite directions and from different starts, so
   they kept *different* subsets of it and then claimed different curves
   between the same two points: 8 free edges in a shell of 1927, and no
   solid. The fix is to judge every vertex against its **original**
   neighbours in a single pass, which is symmetric by construction.
2. **A circle on one side and a polygon on the other.** A planar face whose
   hole is a circle, meeting a *faceted* cylinder, is the same split in
   another disguise. Analytic treatment is therefore refused for any loop
   touching a vertex a faceted face still uses.

The general rule the prototype ended on: **an exactness decision is only
legal if every face sharing the boundary makes the same one.** It applies to
the STEP writer and, independently, to the mesher (§5).

---

## 4. Dragging: the inverse problem, solved on the design

`cadjoint/brep/drag.py`.

A stored B-rep lets you drag a vertex because the vertex *is* the geometry.
Here the vertex is the solution of `f_a = f_b = f_c = 0`, so dragging is:
find the parameter update that moves the solution there, without breaking the
sketch's constraints. One Gauss-Newton step is

```
[J_h ; J_c] Δθ = [target − h(θ) ; −c(θ)]
```

solved for the minimum-norm `Δθ` by least squares — the handle row asks for
the motion, the constraint rows keep the sketch legal, and least squares
picks the smallest edit that does both. That is
`constraints.solve._newton_projection`'s `Δ = Jᵀ(JJᵀ)⁻¹c` with the drag
stacked on top; a final `project_to_manifold` restores the constraints
exactly.

### 4.1 Measured: dragging a fin-comb corner

The starter's fin comb on its own SimMesh grid gives an all-analytic graph
(18 faces, 48 edges, 32 vertices, 0 blend). Vertex 7 lands on fin 1's outer
tip corner, world `(−0.75, 0.6, 0.85)`, to `2.3 × 10⁻⁷`. Drag it by
`+0.05` in world x:

| | |
|---|---|
| final error `|achieved − target|` | **1.19 × 10⁻⁸** |
| constraint residual after the solve | **0** |
| Gauss-Newton iterations | 4 |
| parameters that moved | **`fin1_tip_r`, by 0.050000** — and nothing else |

Sixteen sketch points, seventeen constraints, and the answer is the one
sketch point that owns the corner. That is the least-squares minimum-norm
step doing exactly what a direct-manipulation UI should do.

Restricting the edit works too: the same corner dragged `+0.05` in world y
(the extrusion direction) with `parameters=["fin_depth"]` moves only
`fin_depth` and still lands under `10⁻⁶`.

An **edge** handle moves only across its own curve. Dragging perpendicular to
the local tangent lands under `10⁻⁵`; dragging *along* it is a measured no-op
— the projection puts the handle straight back — which is right, because a
point on a curve has no identity along it.

### 4.2 Topology changes are detected, not solved

A vertex exists only while its three patches still bound the solid there.
The test needs no re-extraction and no heuristic: solve the drag, then ask
whether the moved handle still lies on the **scene's** zero set. Dragging the
same fin corner `−0.9` in z buries it inside the deck:

```
topology_changed = True, applied = False
reason: the handle left the solid's boundary (|sdf| = 0.6 > 0.011);
        the frozen graph cannot represent the new topology — re-extract instead
```

The parameters are returned so a caller can decide; they are not written back.

### 4.3 What fought me here, and why the answer is interesting

`∂(handle)/∂θ` was supposed to be `jax.jacrev` through the kernel. It is
not, and cannot be:
`ExtrudedPolygon.patch_fields` → `_edge_half_plane_fields` reads the
profile's **shoelace sign** with `float(...)` to orient all its walls, and
`float()` on a tracer raises `ConcretizationTypeError`.

That is not a bug to route around. The winding is a *discrete* fact — the
frozen topology — and the pipeline's whole doctrine is that discrete facts
are recomputed concretely and continuous ones are traced. So the drag splits
the IFT the same way:

- `J = ∂f/∂x` is autodiff, exact, with the **point** as the only traced
  argument — nothing discrete is in the way there;
- `∂f/∂θ` is central differences on the three field *values* at the fixed
  point, rebuilding the fields at each perturbed design, which re-derives the
  discrete decisions concretely.

The expensive part — the projection — is not repeated at all: 2·dof scalar
evaluations, not 2·dof Newton solves. The kernel's own `custom_vjp` is still
the exercised path wherever the fields close over traceable parameters, and
`patch_field_fn` builds it.

**Recommendation:** if full tracing is wanted, `_edge_half_plane_fields`
should take its orientation as an argument (defaulted to the concrete
shoelace test), so a caller with traced vertices can pass the frozen sign.
That is a two-line change in `cadjoint/sdf/primitives/polygon.py` and it
would let `drag.py` use `jacrev` end to end.

---

## 5. Meshing from the graph — a measured spike

`cadjoint/brep/plc.py`.

Today's simulation mesh hands TetGen the dual-contour quad soup: one triangle
pair per crossed cell, preserved exactly with `-Y`. The surface triangulation
is therefore *lattice-driven* — its node count and its edge lengths come from
where the grid cut the model.

The graph offers a different input. A planar face is a polygon; it needs its
boundary, not one triangle per cell. So the PLC is: analytic planar faces
re-triangulated from their own simplified loops (ear clipping in the face
plane), curved and blend faces kept as their dual-contour triangles, and
every node tagged with its owner patch set so `recompute_plc_points` moves it
by *its own* arity.

### 5.1 Measured: re-projection alone (`coarsen=False`)

Starter thermal body, the scene's own SimMesh grid, TetGen with the same
`-Y -q1.5/10` bounds as `sdf_to_tet_mesh`:

| | DC path (`sdf_to_tet_mesh`) | B-rep PLC, re-projected |
|---|---|---|
| tets / nodes | 2957 / 956 | 3003 / 963 |
| radius ratio, min | 0.0594 | **0.0607** |
| radius ratio, 1st pct | 0.2512 | **0.2658** |
| radius ratio, mean | 0.6701 | **0.6821** |
| aspect ratio, max | 3.766 | **3.651** |
| aspect ratio, mean | 2.017 | **1.991** |
| volume | 1.161975 | 1.162087 |

Small and consistent: every metric improves. Re-projecting the surface onto
its owner patches instead of onto the scene SDF does not cost mesh quality
and slightly helps it, mostly by putting creases exactly on their creases.

On the plate the same comparison gives min radius ratio 0.0425 → **0.0521**
and 1st percentile 0.228 → **0.261**, on 8456 tets against 8662.

### 5.2 Measured: coarsening (`coarsen=True`)

The starter's fin comb alone is 18 planar faces and no blends, so it
coarsens completely:

| | DC | coarsened PLC |
|---|---|---|
| surface triangles | 1636 | **60** |
| surface nodes | 820 | **32** |
| tets | 2499 | 54 |
| volume | 1.1123998 | 1.1123998 |
| radius ratio, min / mean | 0.163 / 0.675 | 0.0026 / 0.259 |

**27× fewer triangles for bit-identical volume — and much worse tets.** That
is the finding, and it is not a defeat:

> The B-rep PLC decouples geometric fidelity from element size. The lattice
> used to supply both. Once the surface is only as fine as the geometry
> requires, the element size has to be *asked for*.

And here the spike hit a wall worth naming precisely: **`-Y` (`nobisect`) is
what pins the frozen-topology contract** — it is why "the first `len(vertices)`
output nodes are the input vertices verbatim" holds, which is what
`recompute_tet_points` needs. It is also exactly what stops TetGen refining a
coarse boundary: passing `maxvolume` at the DC mesh's mean tet volume, and at
half of it, changed **nothing** (54 tets both times).

The way out is available and is the graph's own: a refinement point inserted
on a known face is a one-field projection, on a known edge a two-field one.
The frozen-topology contract can be restated as "every boundary node has an
owner" instead of "no boundary node was added", and then boundary refinement
is legal. See §9.5.

### 5.3 Coarsening is all-or-nothing across an edge

Same rule as §3.3: coarsening one side of a shared boundary and not the other
leaves a T-junction, and TetGen sees a crack. Pinning the shared vertices
instead only moves the problem into the triangulation — a facet with
collinear boundary vertices has no ear to clip. So a face coarsens only if
every face it borders does; blocked faces are counted.

Measured consequence, and it is the important one:

| scene | coarsenable | coarsened | blocked |
|---|---|---|---|
| fin comb (all planar) | 18 | **18** | 0 |
| plate + bore (curved bore, capped holes) | 4 | 0 | 4 |
| starter thermal body (blended) | 22 | **0** | 22 |

**A blend face's tessellation pins the boundary of every analytic face it
touches.** A blended body cannot be coarsened at all without a blend-face
remesher. This is the same conclusion §3.2 reached from the other direction
(1142 of 1181 STEP faces are facets), and it is the prototype's most
actionable result.

### 5.4 The other route: hand the exact STEP to a CAD mesher

`cadjoint/brep/mesh_gmsh.py`, `cadjoint/fem/tesseracts/tet_gmsh/`.

§5.1–5.3 keep TetGen and change what it is fed. The alternative is to stop
triangulating altogether: §3 already writes the part as *exact geometry*, and
a CAD mesher can size elements by the model instead of by the lattice. Gmsh
4.15.2 (HXT, `Mesh.ElementOrder = 2`) reads that file, and the whole question
is whether the ownership survives the round trip — OCC renumbers every
entity, and reports a cylinder's type as `Unknown`, so neither the tag nor
the type identifies a face. What identifies it is the patch field: a nearest-
quad vote proposes a face, and `|f_patch|` on the entity's own nodes confirms
or refuses it.

**The plate (box − cylinder), TET10, macOS/arm64:**

| | DC path (`sdf_to_tet_mesh` + `tet10_mesh`) | Gmsh from the exact STEP |
|---|---|---|
| wall time | 2.08 s | **0.26 s** |
| tets / nodes | 8529 / 14445 | 4735 / 8228 |
| radius ratio, min | 0.0425 | **0.3191** |
| radius ratio, mean | 0.7318 | **0.7613** |
| volume (exact 0.994920) | 0.996481 | 0.997987 |
| what set the element size | the (20,20,20) lattice | the part (`target_size = 0.10`) |

The worst element is **7.5x better** and the mesh arrives in an eighth of the
time. Inside that 0.26 s, Gmsh itself is 50 ms; the rest is writing the STEP
(4 ms) and assigning ownership (0.20 s, one batched JAX call per patch).
Neither column counts `extract_brep`, which only the Gmsh route needs — see
the last paragraph of this section, which is where the honest total lives.

**The bore is where the two orders differ.** `tet10_mesh` promotes a linear
mesh, so a midside on the bore lands at the chord's midpoint. Gmsh's
`setOrder(2)` puts it on the `CYLINDRICAL_SURFACE`, and re-solving it against
its own patch keeps it there when the radius moves: measured, every bore
midside sits at r = 0.25 to 1e-6, while its chord midpoint is up to 1.4e-3
inside. `tests/brep/test_mesh_gmsh.py` asserts both.

**Positions differentiate; topology does not.** Gmsh's decision — how many
nodes, which cells, which entity owns which node — is discovered once and
frozen. What moves under a design change is the positions, recomputed by
`cadjoint.brep.project` at the arity ownership gives, midsides included.
Central differences against `jax.grad`, plate at `target_size = 0.16`
(2706 nodes, 29 distinct owner sets), x64:

| objective | parameter | analytic | central FD (h = 1e-5) | rel. error |
|---|---|---|---|---|
| mesh volume | bore radius | −1.193067 | −1.193067 | 1.8e-12 |
| mesh volume | plate half-thickness | +2.505492 | +2.505492 | 3.7e-13 |
| mean bore-node radius | bore radius | +1.000000 | +1.000000 | 1.4e-11 |
| mean bore-node radius | plate half-thickness | 0 | 0 | exact |

The analytic derivatives are the discretised ones, and they should be: the
exact −2πrh = −1.2566 belongs to the cylinder, while an inscribed polygon
bore is what the mesh has. The claim being tested is that the adjoint matches
*this* mesh's own volume, and it does to 1e-12.

**Blends.** The starter's thermal body writes 23 planes and 148 facet faces,
and Gmsh meshes the facets as discrete surfaces. Nodes on a surface no patch
owns are solved against the *scene's* own zero set instead, and there are a
lot of them:

| target size | nodes | tets | blend nodes | blend surfaces | worst radius ratio, before → after re-solve |
|---|---|---|---|---|---|
| 0.16 | 4941 | 2406 | 410 (8.3%) | 22 | 0.2152 → **0.2171** |
| 0.1167 (the grid's own) | 8896 | 4563 | 608 (6.8%) | 64 | 0.1859 → **0.1895** |

A blend node moves up to 2.9e-2 in the re-solve, which is the STEP's chord
error being repaired rather than a drift; a patch-owned node moves at most
2.9e-3, which is the ownership bar. At the coarser size the 410 blend nodes
spread over nine faces, plus 184 on curves whose bounding facets are too small
for Gmsh to give them an interior node to vote with; those inherit nothing and
fall to the scene, which is the conservative answer and is reported under face
`-1` rather than folded into a face they only nearly belong to.

**Two ways the ownership went wrong, both found by measuring the re-solve.**
At the nominal design a patch-owned node should not move at all, so anything
that does is a misassignment, and it showed up as quality loss:

1. A surface entity was confirmed against the *median* residual over its
   nodes. An entity straddling a blend's edge has most of its nodes hugging
   the neighbouring plane and a few peeling away, so it passed as that plane
   and the projection dragged the outliers onto the plane's unbounded
   extension. Confirming on the **maximum** instead is the fix; worst radius
   ratio 0.1982 → 0.2171.
2. A curve node *inherited* the patches of the surfaces bounding it, with no
   check at the node itself. Two adjacent facet surfaces can both point at
   the same plane while the curve between them runs along the blend. Applying
   the same bar at the node dropped the worst owned displacement from 1.7e-2
   — nearly six times the 3.0e-3 bar — to 2.4e-3, and turned 44 more nodes
   into blend nodes, which is what they were.

**The end-cap, and what it is really evidence of.** `scenes/end_cap.py`'s
housing at its declared `(26, 26, 13)` does not mesh on either route, and for
one root cause. The DC path spends **520.4 s** walking its refinement ladder
— (26,26,13), (39,39,20), (59,59,30), each ~55 s of extraction plus ~10 s of
projection — and TetGen refuses all three as self-intersecting. The Gmsh
route refuses it in **0.87 s** after the same extraction, because the graph's
2247-face shell does not sew: OCCT reads it as one 12-face solid plus a free
shell, and Gmsh's highest-dimension-only import would otherwise have meshed a
0.1 m chip out of a 2 m casting, quickly and wrongly. Counting `ADVANCED_FACE`
in the file against the surfaces that arrive is what makes that an error
instead of a plausible mesh (`_check_import`). Gmsh does not rescue an
invalid B-rep — it fails faster, and says why.

**What this route does not remove.** `extract_brep` still runs, and it is the
expensive step: 8.7 s on the plate, 18 s on the starter's thermal body, and
53.5 s on the end-cap housing (that last figure is the DC ladder's own
measurement of the same call, since both routes make it). The Gmsh route
replaces the *meshing*, not the extraction, so on a cold graph it is not
eight times faster end to end — it is faster only where the graph is already
being built for the exporter and the drag handles, which is the whole premise
of §6. It is also why the plate table quotes the two meshers against each
other and says so.

---

## 6. What the three products actually share

```
                     cadjoint.brep.project        <- the only place a
                     (1|2|3 fields, IFT adjoint)      position is computed
                              |
                     cadjoint.brep.graph          <- one cached extraction:
                     faces / edges / vertices         faces, loops, owners
                     + owner_patches per node
                     /            |            \
          step.py           drag.py            plc.py
      analytic STEP     inverse problem     PLC + owned nodes
      OBJ / STL         on the design       for TetGen
```

Concretely shared: the patch table, the ownership signature, the loops, the
simplification (`brep_loops` is used by both the exporter and the mesher),
the projection, and the adjoint. The three consumers differ only in what they
do with a `BRep`.

---

## 7. Where the prototype is honest about being a prototype

- Analytic STEP covers `PLANE` and `CYLINDRICAL_SURFACE`. `SPHERICAL_SURFACE`
  and `SURFACE_OF_REVOLUTION` are *classified* (the graph knows a face is a
  sphere or a cone, with a fitted centre/apex and a residual) but not
  *emitted*, because emitting them needs their trimming curves, and a general
  trimming curve is not a line or a circle — see §9.1.
- Extraction is 17.6 s on the thermal body at 40³ and ~3 s on the SimMesh
  grid. That is eager-mode JAX dispatch, not algorithm: §9.6.
- The blend test needs the scene projected first (one extra 1-field
  projection of every quad centroid). That is a real cost and a real
  simplification.

---

## 8. Migration order for the three existing DC consumers

All three re-extract independently today. The graph is the natural cache
because all three want the same thing out of it.

1. **The viewer's feature-edge overlay** (`cadjoint/viewer/_edge_overlay.py`)
   — lowest risk, highest immediate payoff. It already does §2 by hand:
   world-frame leaves, seam groups, a 2-field projection, chain filtering.
   Replacing its bespoke path with `extract_brep(...).edges` gives it *typed*
   edges (which two patches, closed or not, and a residual) and deletes the
   duplicated `_project_to_seam`. The design-subtree rule (`_design_leaves`)
   becomes a filter on `BRepFace.leaf`. **Do this first**, and keep
   `_project_seam_groups` as the oracle its tests already treat it as.
2. **The mesher plugin / native path** (`cadjoint/meshing/native.py`) — no
   change needed; it produces the `Mesh` the graph consumes. Add the graph
   *above* it.
3. **The SimMesh tet path** (`cadjoint/fem/tetmesh.py`,
   `cadjoint/fem/motion.py`) — last, because it is the one with a
   gradient-correctness contract. The migration is: `sdf_to_tet_mesh` takes a
   `BRep` instead of a bare sdf; `TetMesh` gains `owner_patches` /
   `owner_arity`; `recompute_tet_points` calls `project_batched` per arity
   instead of `project_points` once. `project_points` stays — it is the
   arity-1 case and this prototype reproduces it exactly, so the change is
   provably a superset.

A shared `extract_brep` cache keyed by `(scene identity, grid, parameter
hash)` should sit next to `cadjoint/cache.py`. The extraction is
deterministic, so caching is safe.

### 8.1 Migration 1 of 3: the overlay — done, and what it cost

`_mesh_edge_payload` now runs `extract_brep` on the viewer's own 64³ grid
(one dual-contouring pass — `tests/viewer/test_edge_overlay_brep.py`
counts the calls) and reads two things off the graph:

- **wire** — the same dual-contour quad edges as before, drawn on
  `BRep.points`. The wire layer stays the quad edges rather than the PLC
  tessellation because the quads are already in hand: `brep.mesh.quads` is a
  by-product of the pass the sharp layer needs anyway, while `brep_plc`
  triangulates every face loop, roughly doubles the segment count for the
  same picture, and needs the loops to be simple — which the extraction
  reports as *not* holding on some scenes. Nothing to buy there.
- **sharp** — every `BRepEdge` the graph can certify, **traced** rather
  than sampled: see §8.2. The lattice says where an edge starts and which
  two patches it belongs to; nothing else about it comes from the grid.

Three things came out of the migration rather than going into it.

**`BRepEdge.vertices` is now populated** (`_link_edge_vertices`). It was
documented and always `(-1, -1)`, because edges are chained before the
triple points exist. This turned out to be the load-bearing addition: a
triple point is where a trace starts and where it stops (§8.2), and it is
what makes the edges meeting at a corner share one endpoint exactly instead
of three near-misses.

**A corner must not be re-projected onto a subset of its own patches.** It
is the 3-field solution; projecting it again onto the 2 patches of one
incident edge pulls it a few thousandths off the third face and the chains
stop touching. It is pinned in the sampled fallback and is an exact
endpoint in a trace — worth three debris fragments on the artifact
battery.

**The kernel was tracing, not computing.** Op-by-op, JAX re-traces every
`vmap(value_and_grad(f))` on every call, so a fifty-patch table over four
Newton steps is two hundred traces of which a hundred and fifty are
redundant. `project_batched`, `project_fields` and `batched_residuals` now
compile the whole unrolled iteration. This is where the overlay's cost went.

Measured, `_mesh_edge_payload` on `scenes/starter.py`, wall clock:

| `scenes/starter.py` | before (lattice links) | after (graph) |
|---|---|---|
| cold process — what the viewer pays, `mesh_source` forks per request | 16.9 s | **8.6 s** |
| warm, second call in one process | 4.3 s | 5.2–6.0 s |
| sharp segments | 453 | 1006 |

| `scenes/end_cap.py` | before | after |
|---|---|---|
| cold process | 215 s | **32.5 s** |
| warm | 219 s | **29.5 s** |
| sharp segments | 357 | 1383 |

Both columns are timed with `JAX_ENABLE_X64=1`, so the comparison is
like-for-like; the shipped float32 default is faster still — starter 7.7 s
cold, end-cap 27.1 s — and lands a handful of segments either way as the
residual gate falls differently (starter 996, end-cap 1422).

The cold number is the one the product pays, and the end-cap is where the
difference shows: its cost was never compilation (before and after, warm
equals cold on it) but the *count* of programs, and the graph asks for a
handful of big ones where the old path asked for a per-seam-group crowd.

The starter's warm row is the one honest regression: ~1.3× slower, because
the compiled programs are built per call and a second call in the same
process recompiles them. Worth fixing with a cache keyed the way §8's
`extract_brep` cache would be, and not worth fixing before that cache
exists — nothing in the product calls this twice in one process.

Quality, on the artifact battery (`tests/viewer/test_edge_artifacts.py`, all
40 unedited): crossings 0 and debris 0 on all eleven configurations, as
before, and **every** analytic curve's coverage — both by the link set and
by a single connected chain — is now exactly `1.000`. Before-and-after
renders of the starter are in `research/design/light-chrome/edges-before-after*.png`;
the visible differences are the press-fit bush rims (drawn as circles, and not drawn
at all before) and the curves that now run into their corners.

### 8.2 Tracing the edge instead of chaining the lattice

The first version of the sharp layer took each edge's seeds — the graph's
projected midpoints of the mesh boundary between two face regions — put them
in the order the boundary walk visited them, and drew that. It looked right
at low zoom and was wrong: at the starter's fin roots and bushings the
polylines drew spikes and flags, and small rims drew as octagons.

**The order was the bug, not the positions.** One bracket edge arrives as

```
x = -0.084, -0.113, -0.106, 0, 0.106, 0.113, 0.084     (y, z constant)
```

— seven points *all exactly on one straight line*, to 4·10⁻⁸, delivered as a
fold. The chain walk is not at fault either: its mesh path is a clean
unbranched path. The boundary it follows simply staircases across the curve,
so projecting its midpoints onto the curve is many-to-one in a way that
scrambles the parameter. No amount of re-ordering heuristics fixes this
reliably: the seeds' spacing ranges over an order of magnitude on one edge
(0.007 to 0.085), which is enough to defeat nearest-neighbour chaining, and
the seeds run *past* the triple points the edge is supposed to end at.

**So the curve is traced.** Where `f_a = f_b = 0` the tangent is
`∇f_a × ∇f_b` in closed form, so a point on the edge can be continued:
predictor along the tangent, corrector back onto both zero sets with the
same Gauss-Newton step the projection kernel already runs. That is
`cadjoint.brep.project.trace_curves`, and §9.1 is where it was already
recommended. The lattice is asked only for a seed and a patch pair.

Three properties fall out rather than being engineered:

- **Order is monotone by construction.** There is nothing left to sort.
- **Sampling is scale-free.** The step is driven towards a fixed turning
  budget per chord (15°) and any step that overshoots by more than twice it
  is re-taken, so a straight edge runs at half a cell and a rim of radius
  `r` settles at `r · 15°`. A screw-head rim of radius 0.07 — nine half-cell
  chords, an octagon — now gets 24. That is the whole of the small-loop fix.
- **Tangency answers itself.** `|∇f_a × ∇f_b|` *is* the sine of the angle
  between the two normals, so where the surfaces touch instead of crossing
  it vanishes, and the trace reports "no edge" rather than pushing a
  singular system. That is exactly the blend case.

Seeds come from the graph's triple points, so a traced edge starts and ends
on the corners it shares with its neighbours and the layer stays connected
chains. A corner has to earn that: it is solved against *its own* three
patches, and where those are not this edge's two it can sit a full cell off
the line (`_corners_on_curve` measures each against this edge's pair —
without it, one bracket edge kinks by 87° at each end). An edge with no
certified corner has no end to stop at, so it keeps the old sampled path as
a fallback; on the three scenes the tracer takes 64 of 101 edges on the
starter, 43 of 61 on the bracket and 234 of 316 on the end-cap, and the
fallback covers most of the rest.

Two gates remain, and the populations they separate are not close:

| | genuine edges | everything else |
|---|---|---|
| seed residual `|f|` | under 10⁻⁷ | 10⁻⁵ and up (nothing in between, on any scene) |
| sharpest joint drawn | 13–16° | 180° |

The residual gate sits in that empty band (`1e-5` of a cell). It is the only
thing that catches an ownership island *between two planes of different
solids*, which draws a smooth arc that no turning test can fault — two
planes meet in a line, so any arc is wrong, and its residual of 8.6·10⁻⁴ is
the only local evidence. The turning gate (45°, three times the sampling law)
is the last word on anything the fallback produced.

Measured after the change, worst case over every drawn edge:

| | starter | bracket | end-cap |
|---|---|---|---|
| sharpest joint | 26.6° | 16.3° | 32.3° |
| straight edge, length ÷ chord | 1.0000 | 1.0000 | 1.0000 |
| sharp segments | 964 | 1210 | 1165 |

Before/after at 2× in `research/design/light-chrome/edges-starter-detail.png`
(a fin root) and `edges-starter-rims.png` (a screw-head rim).

`research/brep-edge-tracing.md` maps the literature onto this tracer and
recommends four things beyond it. One is in:

- **Branch-jump guard** — *adopted*. The corrector now also re-takes a step
  at half length when it has to pull the predicted point back by more than
  half the step, which is the signature of landing on a different branch of
  the same pair rather than continuing this one. It fires on none of the
  three scenes, so it costs nothing and is there as a guard.
- **Watch-field termination** — *not adopted*, and it is the one worth doing
  next. Today a trace needs a *certified corner* to know where to stop, so
  an edge without one falls back to the sampled path (the tracer takes 64 of
  101 edges on the starter, 43 of 61 on the bracket, 234 of 316 on the
  end-cap). Watching the other patches of the two leaves for a sign change
  would end an edge and solve its vertex in one arity-3 step, keyed by patch
  triple, and would raise those fractions. It is a redesign of termination
  rather than a guard, and it changes which edges are traced, so it wants
  its own verification pass against the artifact battery.
- **Levenberg–Marquardt below sin θ ≈ 0.3** — *not adopted*. It buys
  robustness in the near-tangent band, which is precisely where this overlay
  deliberately answers "no edge" (`_TANGENT_FLOOR`, 0.1). Worth it for
  export, where a near-tangent seam still has to be written out; not for a
  layer whose policy there is to draw nothing.
- **Per-point ownership check** — *not adopted*. Testing `owner_pair(x) =
  {a, b}` at every corrected point means evaluating the whole patch table
  per step, which is the cost the batching exists to avoid. The residual
  gate and the turning gate already catch what it would catch on these
  scenes; if a case appears that they miss, this is the answer to it.

**Blends: a fillet finer than a cell is the edge it rounds.** Rendered three
ways on the starter (`research/design/light-chrome/edges-blends.png`): nothing,
the fillet's own boundary curves, and the virtual sharp edge the fillet
replaced. On a sub-cell fillet the boundary curves are out on their own
numbers — on the starter, 61 blend-adjacent edges totalling 4.1 of arc length
against 34.6 for the analytic edges, an average of 0.067, well under a cell
each — and they render as a scribble, not a curve; the battery's debris rule
would reject them on their own metric. That leaves the choice between nothing
and the virtual sharp edge, and it is not a matter of taste: it is a question
of *radius against cell*, and the graph already has the dial.

`smooth_min(a, b, k)` is `min(a, b) - h²/(16k)` with `h = max(4k - |a-b|, 0)`,
so it pulls the surface down from the sharp corner by exactly `k` where the two
operands meet and by nothing at the band's edges. The blend test asks the
owning patch for its value on the scene's own zero set, so `|f_patch|` across a
fillet runs from 0 to `k` — which makes `blend_tolerance` **directly** the
largest radius still counted as an edge, in the same units as the number the
user typed. No calibration factor, and
`tests/viewer/test_edge_overlay_brep.py::test_the_threshold_is_the_radius_the_user_typed`
pins the identity.

The overlay therefore sets `blend_tolerance` to **one cell**
(`_BLEND_AS_EDGE_CELLS`), where its own resolution runs out: a fillet finer
than a cell cannot be *shown* as curvature on a 64-cell grid — dual contouring
rounds it into one vertex — so its virtual sharp edge lands within a cell of
the surface and reads as the edge a CAD user is looking for. Above a cell the
fillet is curvature the viewport genuinely renders, and a line buried inside it
would be a line that is not on the model. The transition is sharp and measured,
on a plate with a rounded bore:

| fillet radius | blend faces | rim drawn | rim radius error |
|---|---|---|---|
| 0.02 (0.21 cell) | 0 | 86 points | 1.9 × 10⁻⁸ |
| 0.047 (0.50 cell) | 0 | 86 points | 2.1 × 10⁻⁸ |
| 0.188 (2.0 cells) | 2 | none | — |
| 0.281 (3.0 cells) | 4 | none | — |

The graph's own default — a thousandth of the grid diagonal, 0.0104 here, about
a *ninth* of a cell — is calibrated for export, where any rounding at all must
be honoured. Left in place it deleted most of `scenes/bracket.py`, which rounds
every junction (`Union(..., smoothness=0.05)`, `Difference(..., 0.02)`), and
the bore rims and the web-to-plate line were simply absent:

| `scenes/bracket.py` | export default (0.11 cell) | overlay (1 cell) |
|---|---|---|
| faces / of which blend | 55 / 27 | 28 / **0** |
| edges / drawn | 121 / 35 | 63 / **61** |
| sharp segments | 749 | **1246** |

The two undrawn edges at one cell are refused by the residual gate, which is
the safety net doing its job. The same shift on the starter is 78 faces / 26
blend and 81 of 163 edges drawn, against 57 / 0 and 103 of 114. Before and
after in `research/design/light-chrome/edges-bracket.png`.

This is a threshold, not a style choice, so it stays a named constant and gets
no payload flag.

---

## 9. The hard parts, with a recommendation each

### 9.1 Curved–curved edges need curve marching

Two cylinders meeting, or a cylinder meeting a sphere, give an intersection
curve that is neither a line nor a circle. The graph *has* it — the
2-field projection puts a polyline on it to `3 × 10⁻⁸` — but STEP wants a
curve, and a polyline of DC-density points is not one.

**Recommend:** march the curve properly instead of inheriting the lattice's
sampling. From one solved point, step along `t = ∇f_a × ∇f_b` normalized,
re-project with the same 2-field kernel, and adapt the step to a chord-height
tolerance. This is ~40 lines on top of the existing kernel and it replaces
"one point per grid cell" with "one point per unit of curvature". Then fit a
B-spline (`B_SPLINE_CURVE_WITH_KNOTS`) to the marched points, with the fit
residual reported the way the surface fits already are. The same marching
gives the mesher a size-independent edge discretization, which §5.2 needs.

### 9.2 Lofts and B-spline surfaces

`loft` currently has no patch decomposition, so it falls into the
single-opaque-patch path and the whole loft is one `freeform` face. The right
answer is not to fit a B-spline to the extraction; it is to give
`LoftedPolygon` a `patch_fields()` — a loft between two polygons of matching
vertex count sweeps one ruled surface per profile edge, and a ruled surface
between two lines is a plane or a hyperbolic paraboloid, both of which have
closed forms. **Recommend:** add `patch_fields()` to the loft primitive
first; the graph then classifies its faces for free, and only genuinely
freeform lofts need surface fitting.

### 9.3 Blend faces are the real limitation

Measured twice, from both ends: 13 of 58 faces on the thermal body are blend;
1142 of 1181 STEP faces are facets because of them; and 22 of 22 coarsenable
faces are blocked by them.

Three options, in order of how much they buy:

1. **Emit blends as fitted B-spline surfaces.** A smooth-union fillet between
   two known patches is a rolling-ball-like surface with a natural
   `(u, v)` parametrization from the two patches' own coordinates. Fit a
   bicubic to a *marched* grid on it (same machinery as §9.1, one dimension
   up) and emit `B_SPLINE_SURFACE_WITH_KNOTS`. This is the real fix and it
   removes both the STEP facet count and the coarsening blocker.
2. **Recognize the rolling-ball case exactly.** `smooth_min(a, b, k)` between
   two planes is a cylinder of radius related to `k`; between a plane and a
   cylinder it is a torus. A small pattern-matcher over
   `(patch kind, patch kind, smoothness)` would turn a large fraction of real
   blend faces into `CYLINDRICAL_SURFACE`/`TOROIDAL_SURFACE` with no fitting
   at all. Cheap, high hit rate on mechanical parts.
3. **Do nothing and facet them.** What the prototype does. Correct, valid,
   and 4.6× better than the mesh writer already.

**Recommend 2 then 1.** 2 is a day's work and covers the common cases; 1 is
the general answer and shares its marcher with §9.1.

### 9.4 Topology events under drag

The prototype detects them exactly (§4.2) and refuses. The next step is not
to solve them — no parameter update can, because the graph is frozen — but to
*bound* them: for a given drag direction, find the largest step before any
vertex leaves the boundary or any edge shrinks to zero, by watching
`scene_sdf(handle)` and each incident edge's arclength as functions of the
step. That gives a UI a safe drag range and a place to re-extract, which is
the interaction a CAD user actually expects (drag freely; the model rebuilds
at the event).

### 9.5 The `-Y` contract has to be restated

`nobisect` is currently doing two jobs: preserving the frozen node identity,
and forbidding boundary refinement. Only the first is wanted. **Recommend:**
let TetGen split boundary facets, and recover the contract by *owning* the
new nodes — a node TetGen inserts inside face `f` gets `owner = (f.patch,)`,
one inside edge `e` gets `owner = e.patches`. `recompute_plc_points` already
handles arbitrary per-node arity, so nothing else changes; the mapping from
TetGen's output faces back to B-rep faces is the only new bookkeeping. This
unlocks §5.2's coarse PLC with a proper size field.

### 9.6 Cost

17.6 s per extraction on the thermal body at 40³ is eager-mode dispatch, and
`research/performance.md` §6.5 already identifies `jax.jit` as the largest
available lever. The three loops that dominate (`_own_patch`,
`project_batched`, the per-face normal evaluations) are all fixed-shape and
jittable as written. **Recommend** jitting `project_batched` per
`(field count, arity)` and caching the compiled program on the patch table;
that is the same shape-sharing argument the performance study makes.

---

## 10. What this prototype claims, and what it only suggests

**Claimed and tested** (`tests/brep/`, 55 tests):

- one kernel reproduces `project_points` and the viewer's seam projection,
  and its adjoint matches central differences at every arity;
- a hard CSG solid derives its exact textbook B-rep, stable against the
  lattice;
- that B-rep exports as one OCCT-valid solid whose volume is the analytic
  one to 10⁻⁷;
- blends are separated from analytic faces exactly, and never mistaken for
  geometry;
- a corner drag becomes the single sketch edit that owns it, with the
  constraints held, and a topology-changing drag is refused;
- a PLC from the graph tet-meshes at least as well as the current path.

**Suggested, not proved**: that coarsening is the right long-term surface
input (it needs §9.5 first), that blend B-splines are worth the machinery
(§9.3), and the migration order in §8.

---

## 11. Reproducing every number

```bash
.venv/bin/pytest tests/brep -q            # 55 passed, ~3 min
.venv/bin/pytest tests/meshing -q         # 224 passed, unchanged baseline
.venv/bin/ruff check cadjoint/brep tests/brep
```

- §1.1 FD table — `tests/brep/test_project.py::test_the_adjoint_matches_central_differences`,
  printed by the scratch script in that test's shape.
- §2.1 / §2.2 counts — `BRep.report()`; the plate's fixture is
  `tests/brep/conftest.py`, the thermal body's grid is the scene's own
  `sink_mesh` declaration.
- §3.1 / §3.2 — `save_brep_step(...)` returns the entity counts; the OCCT
  numbers are `tests/brep/test_step_kernel.py` (needs the `stepcheck` extra).
- §4.1 / §4.2 — `tests/brep/test_drag.py`.
- §5.1 / §5.2 / §5.3 — `tests/brep/test_plc.py` plus
  `cadjoint.brep.plc_quality` against `cadjoint.fem.tetmesh.sdf_to_tet_mesh`
  on the same grid.
- §5.4 — `tests/brep/test_mesh_gmsh.py` (needs the `gmsh` extra; the module
  skips without it). Quality is `cadjoint.fem.quality.tet_radius_ratios` on
  both meshes; the FD table is the two parametrised cases of
  `TestTheDerivative`; the counts are `GmshMesh.stats` and
  `GmshMesh.blend_nodes_by_face()`. The end-cap comparison is
  `scenes/end_cap.py`'s `housing` on its own `cap_mesh` grid, run outside the
  suite because the dual-contour side takes 520 s to fail.
- §8.1 — `_mesh_edge_payload` timed directly on `scenes/starter.py` and
  `scenes/end_cap.py`; the quality row is the metric table
  `tests/viewer/test_edge_artifacts.py` prints under `-s`, and the blend
  counts come from the graph's own `BRepFace.kind`.
- Timings are wall clock in a warm process with a populated
  `CADJOINT_CACHE_DIR`, second run quoted; §8.1's cold row is a fresh
  process, which is what `mesh_source` forks for every request.
