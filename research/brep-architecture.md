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
- Timings are wall clock in a warm process with a populated
  `CADJOINT_CACHE_DIR`, second run quoted.
