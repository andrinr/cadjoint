# Tracing the edges of a derived B-rep — what the literature does, mapped to ours

Status: research memo, 2026-09-02. No code was changed. Companion to
`research/brep-architecture.md` (the derivation; §8.2 there documents the
current tracer) and the sibling battery `research/brep-axioms.md` /
`tests/brep/test_axioms.py` (the axioms), which this memo does not duplicate.
Read against `cadjoint/brep/project.py` (`trace_curves`),
`cadjoint/brep/graph.py`, and `cadjoint/viewer/_edge_overlay.py`
(`_traced_polylines` / `_sampled_polylines`).

## 0. The problem in our own terms

An edge here is not "the intersection curve of two surfaces". It is a
connected component of

    E(a,b) = { f_a = f_b = 0 }  ∩  { patches a and b own the scene surface here }

where the patch fields are *unbounded* (a `Box` face is an infinite plane, a
`Cylinder` side an infinite cylinder — `sdf/primitives/box.py:66`,
`cylinder.py:64`) and ownership is decided by the booleans. So the curve
`f_a = f_b = 0` is analytic (quadric ∩ quadric) and usually much longer than
the edge; the edge is that curve *trimmed* by every other patch's zero set.
That is classical SSI turned inside out: the CAD literature has parametric
patches with a known domain and must discover the curve; we have the curve
in closed form and must discover the trimming. Everything below is read
with that inversion in mind.

| sub-problem | today | witness |
|---|---|---|
| which pairs meet, where each component starts | DC ownership regions, chained mesh edges (`graph.py:_build_edges`) | lattice |
| where the points are, in order | `trace_curves` (predictor–corrector); else midpoints re-ordered by a planar fit (`_sampled_polylines`) | fields |
| where the edge ends | mesh vertices with ≥ 3 faces re-solved at arity 3 (`_build_vertices`), certified per edge by residual (`_corners_on_curve`) | lattice, then fields |

The recommendation (§8) makes the lattice a witness for *existence only*,
interval bounds (§5) the witness for *completeness*, and the fields the
witness for everything else, including where an edge ends. Architecture
§8.2 measures why the last matters: the tracer takes 64 of 101 edges on the
starter, 43 of 61 on the bracket, 234 of 316 on the end-cap; the rest fall
back to sampling *only because they lack two certified corners to run between*.

## 1. Surface–surface intersection by marching

### 1.1 Literature

**Predictor.** Everyone steps along `t = ∇f_a × ∇f_b / |·|` — Patrikalakis &
Maekawa eq. (5.101), integrated as an ODE with Runge–Kutta or adaptive
stepping [PM §5.8.2.3]; Hartmann's implicit-curve tracer is the Euler
predictor plus the foot-point corrector `x ← x − F∇F/|∇F|²` [Wiki-IC,
Hartmann98]. Bajaj, Hoffmann, Lynch & Hopcroft use a third-order Taylor
predictor with variable step, then Newton [BHLH88]; Barnhill & Kersey need
only positions and tangents and give a step-length estimate [BK90]. Second
order is cheap for us: Ye & Maekawa's curvature vector of an implicit–
implicit intersection [YM99] follows from differentiating `∇f·t = 0`,

    [ t ; ∇f_a ; ∇f_b ] k = [ 0 ; −tᵀH_a t ; −tᵀH_b t ],    x_pred = x + s t + ½ s² k

and `jax.hessian` of an analytic patch is a few flops. Use it.

**Corrector.** Gauss–Newton onto `f_a = f_b = 0`; ours (`project.py:_newton`,
min-norm `x ← x − Jᵀ(JJᵀ)⁻¹f`) is the textbook one. Continuation calls the
pair "Euler–Newton" [AG90].

**Step control.** Geometric (turning per chord, chord height `s²κ/8`) [BK90,
PM] and numerical: Allgower & Georg's Den Heijer–Rheinboldt rule measures the
contraction of the first two corrector iterates and grows/shrinks the step
to hold it near a target, rejecting when the corrector moves more than a
fraction of the predictor step [AG90]. `trace_curves` has the geometric
half (retry above `2·max_turn`, gain toward `max_turn`) and none of the
numerical half: a fixed 4-step corrector, residual inspected only afterwards.

**Start points and completeness.** Marching alone cannot do it: Hartmann
"traces only connected parts ... has to be started several times"
[Wiki-IC]; Bloomenthal's polygonizer follows one component from one seed
cube [Bloom94]. Certified starts on every component come from: border /
turning / singular points of the projected curve [PM §5.8.1.5];
collinear-normal points — a closed loop implies a line normal to both
surfaces, so disjoint normal cones mean no interior loops, otherwise split
at those points so every branch reaches a boundary [SM88, PM §5.8.2.3];
Krishnan & Manocha's matrix formulation, which also bounds the step to
prevent *component jumping* and finds all branches at a singularity [KM97];
Grandine & Klein's topology resolution *before* any curve is computed, the
curves then solved as an arclength-parametrised DAE boundary-value problem
[GK97], with [TOG23] the modern descendant; and Plantinga & Vegter's
interval-certified small-normal-variation octree, which guarantees isotopy
[PV04] — the entry point to §5.

**Termination, closure, branch jumping.** Parametric methods stop at the
domain border; loops close on re-entering the start's neighbourhood *with a
compatible tangent* [AG90, KM97]. Two nearby branches (a rim and a chamfer
a cell apart, the two lines of a thin rib) are the classic failure; guards
are corrector displacement ≤ ½ step [AG90], KM97's distance-to-other-
component bound, and tangent agreement. `trace_curves` stops on
`|x − target| ≤ step` or `|x − start| ≤ step` after six points — proximity
only, no tangent test — clamps corrector drift at 1× the step (too loose),
never checks that the corrected point is still owned by `(a,b)`, and picks
its direction once by heading toward the target, which is wrong for any arc
that first leaves it.

### 1.2 Mapped to us

- Tangent and GN corrector: ours. Add the parabola predictor and the
  contraction acceptance test.
- Start points: the lattice seeds one per face-pair boundary chain — one per
  component *with at least one DC cell*. Sederberg loop detection is
  unnecessary (a loop with a cell is seeded; a loop without one is
  sub-cell), but "sub-cell" is exactly the hole, and §5 closes it with
  interval bounds; §8(b) turns the closed-loop invariant of the face loops
  into the certificate — Grandine–Klein's "resolve topology, then check it",
  done on the ownership graph instead of parameter space.
- Termination is where we should differ from all of them: an edge ends where
  a **third patch field crosses zero**, and we know which fields to watch
  (the other patches of the two leaves, and every patch of a leaf whose
  bounds meet the pair's curve). Watching sign changes along the trace,
  bracketing, and solving arity 3 is both the exact end *and* the vertex
  (§3). No target is needed, so every edge is traceable and the sampled
  fallback goes away.

## 2. Tangential and near-tangential intersections

### 2.1 Literature

At a tangential point `∇f_b = λ∇f_a` the cross product vanishes and PM defer
to §6.4, which goes to second order: in the common tangent plane the
direction must satisfy a quadratic in its coefficients built from the two
second fundamental forms (eq. 6.64), and the discriminant classifies —
`Δ < 0` isolated contact, `Δ = 0` a tangential intersection curve, `Δ > 0`
a branch point where two curves cross, all coefficients zero a higher-order
contact [PM §6.4, §6.4.1; YM99]. In implicit form it is the restriction of
`M = H_b − λH_a` to the tangent plane: definite, rank 1, indefinite, zero.

Near tangency the Gram is merely ill-conditioned (`λ_min/tr ≈ sin²θ`) and
GN slows or wanders. Levenberg–Marquardt with `μ = ‖f‖²` converges
*quadratically* under a local error bound — no nonsingular Jacobian needed,
non-isolated solution sets allowed [YF01, LM-EB]. That is precisely a curve
of tangency: a curve of solutions with a rank-1 Jacobian, where LM lands on
the nearest solution and `(JJᵀ)⁻¹` blows up. Constraint-based CAD avoids the
numerics by *declaring* the tangency (a fillet is G1 by construction) and
storing a seam, not an edge; Barnhill & Kersey's "tangent tracks" are the
marching analogue [BK90].

### 2.2 Mapped to our blends — the designed near-tangency

Our only source of tangency between design surfaces is `smooth_min`
(`sdf/boolean/smooth.py`): `h = max(4k − |A−B|, 0)`, value
`min(A,B) − h²/(16k)`; band `|A−B| < 4k`, surface pulled `k` below the
sharp corner at the midline. Two consequences the code does not use
(checked numerically against the function on 2026-09-02):

1. **The band boundary is a transversal intersection.** At the band's edge
   `h = 0` and `min(A,B) = 0`, i.e. locally (`A = f_a`, `B = f_b`, a owning
   A's side) `f_a = 0` **and** `f_b = 4k`. The seam where the fillet meets
   face a is `{f_a = 0} ∩ {f_b − 4k = 0}` — patch a against the 4k-offset
   of patch b, crossing at the same angle the patches cross. It traces with
   the existing kernel by substituting `f_b − 4k`. The 69 "blend-adjacent
   edges with no exact curve" of architecture §2.3 do have one: two exact
   seams (G1, not drawn sharp) plus the virtual sharp edge `f_a = f_b = 0`,
   which sits `k` inside the material at the midline.
2. **The fillet is a parabola, not a circle.** Between planes `u, v` the
   band surface is `u − (4k − (v−u))²/(16k) = 0`; its quadratic part is
   `−(u−v)²/(16k)`, rank 1, so the conic is a parabola, C1 at the band
   edges. Architecture §9.3's "cylinder" pattern is wrong; match a
   parabolic cylinder or emit the B-spline of option 1.

So `blend` is not a degenerate `crossing`; it is its own classifier state
with three exact curves, and the overlay's "draw the virtual edge if
`k <` one cell" (`_BLEND_AS_EDGE_CELLS`) becomes a rendering choice over a
fully classified edge. Genuine tangency between analytic patches (cylinder
tangent to a plane, two equal cylinders, flush coplanar faces) is
non-generic but is exactly what the axiom battery builds. The classifier
must name it, never solve it: `tangent` seams are face boundaries for STEP
and the mesher, not sharp edges, placed by LM with *no* IFT derivative (the
double-`where` guard already refuses, correctly); `branch` points get four
half-edges along the two roots of the quadratic; `coincident` merges faces.

## 3. Corners

EMC and DC find a corner as the least-squares point of a cell's tangent
planes — EMC classifies by normal-cone opening angle (`θ_sharp = 0.9`,
`φ_corner = 0.7` on cosines) and solves `N p = [nᵢ·sᵢ]` by SVD pseudo-inverse,
zeroing the smallest singular value for an edge cell [KBSS01 §4]; DC
minimises the QEF with a truncated pseudo-inverse and classifies nothing
[JLSW02]. One point per cell, from sampled normals.

Ours is the arity-3 solve, exact to `1e-15` on the plate. What is not robust
is *seeding* and *sharing*: the seed is a mesh vertex that happens to touch
three regions, so ownership flicker gives 4-region "ambiguous" vertices (44
of 76 on the thermal body), and the three edges meeting at a corner each
end on their own chain endpoint and are reconciled afterwards
(`_link_edge_vertices`, `_corners_on_curve`).

Recommended: corners come from the tracer. A watched field `f_c` changing
sign between consecutive samples of `(a,b)` brackets a vertex; solve arity
3 from the interpolant; key it by the sorted triple and by position (merge
within `tol`); every edge arriving there — `(a,c)` and `(b,c)` will, from
their own traces — references the same id and is pinned to the same point.
Sharing holds by construction. Degenerate cases:

- *Four faces at a point* (pyramid apex, box corner on a plane): four
  fields, three unknowns; GN least squares, accept as one vertex if every
  `|f| ≤ tol`, record the full patch set; its incident edges are whichever
  pairs own surface around it, read off the arriving traces.
- *Coincident patches* (flush boxes, a rib on a plate): classifier says
  `coincident`; merge faces before tracing; a corner on the seam is arity 2.
- *Sub-tolerance slivers*: two vertices within `tol` merge; the edge between
  them is zero-length and dropped — reported, not hidden.

## 4. Ownership fields versus the Hermite-data literature

What EMC/DC get right: with exact normals at crossings, one least-squares
point per cell lands on a crease or corner at O(h²) instead of O(h), and DC
needs no classification [KBSS01, JLSW02]; `sharp_qef_vertices`
(`meshing/dual_contouring.py:167`) is that with a rank-revealing SVD.

What they cannot do, in their own words: Kobbelt assumes "only one sharp
feature within each cell ... for reasonable models and sufficient grid
refinement" and restricts to one feature sample per cell [KBSS01 §4]; DC is
one vertex per cell, so sub-cell features are unresolved and output can be
non-manifold [JLSW02]. The fixes are all *more lattice* — multiple vertices
per cell [SW04], subdivision until "each voxel has at most one sharp
feature" [VKKM03], interval-certified refinement [PV04] — and all recover
the feature from samples, so a chamfer thinner than the finest cell is gone.
The BlobTree is the nearest relative: "feature edges occur when the field
value is contributed by different nodes in the tree", found by polygonising
then subdividing along the seam [WGG99, WvO96] — leaf-level ownership,
what the retired `_project_seam_groups` did, which cannot separate a box's
twelve edges. Hence patches.

What exact patch fields buy that none of them can: (1) sub-cell features
are *unseeded, not lost* — the curve `f_a = f_b = 0` exists whether or not
a cell sees it, and the patch table says which pairs to look for, so a
missing component can be seeded from a bound (§5) or algebraically (§8b);
(2) placement is not O(h^anything); (3) classification is exact — "does the
owning field vanish here", and the band width `4k` is read off the design.

## 5. Interval and affine arithmetic over the SDF graph

### 5.1 Literature

An inclusion function `F([x]) ⊇ { f(x) : x ∈ [x] }` turns "the lattice
happened to see it" into "the box provably contains it". Snyder's SOLVE
and MINIMIZE do constraint solving and global minimisation by interval
subdivision, applied to ray tracing, interference and CSG on parametric
solids [Snyder92]; Duff renders CSG of implicit functions by recursive
subdivision with interval arithmetic and gets collision detection from the
same bounds [Duff92]; Plantinga & Vegter make the bound a *topological*
guarantee (small normal variation ⇒ isotopic mesh) [PV04]. Keeter evaluates
the expression in a shallow hierarchy of tiles, skips tiles whose interval
excludes zero, and — the important trick — *shortens the tape* wherever an
interval decides a `min`/`max`: "expression complexity decreases by two
orders of magnitude between the original and reduced expressions", needing
only C0 continuity, no Lipschitz bound [Keeter20]. Sharp & Jacobson apply
range analysis to neural implicits and report that plain interval
arithmetic was too loose in practice; affine-arithmetic variants gave
guaranteed ray casting, closest points, hierarchies and mesh extraction
[SJ22]. Fryazinov, Pasko & Comninos extend *revised* affine arithmetic
with rules for R-function set operations, blends and conditionals, and
find it the fastest interval technique for both ray–surface intersection
and cell enumeration in polygonisation [FPC10]. (Harnack tracing is
Gillespie, Yang, Botsch & Crane [GYBC24], a growth bound for *harmonic*
level sets, not range analysis; it does not apply to our fields.)

### 5.2 What it gives the edge finder

(a) **Census with a guarantee.** Per cell `[c]` and patch `a`: if
`0 ∉ F_a([c])` no face, edge or vertex of `a` touches the cell. Per pair: if
`0 ∈ F_a([c])`, `0 ∈ F_b([c])`, `0 ∈ F_scene([c])` and the interval cross
product `[∇F_a] × [∇F_b]` has a component bounded away from zero, then the
curve `f_a = f_b = 0` crosses the cell as a *single monotone branch* (a
graph over that axis) — Plantinga–Vegter's condition one arity up — so a
seed found by bisection along that axis is unique and cannot be missed. A
cell where the cone test fails is *ambiguous* (a small rim, a tangency, a
corner) and is bisected until it passes or reaches tolerance, never
silently skipped. For vertices the arity-3 system is square, and the
Krawczyk / interval-Newton test `K([x]) ⊂ int([x])` proves existence and
uniqueness of the corner in the box [Moore]. That is topological
completeness for edges and vertices, independent of whether the DC pass
ever placed a quad there.

(b) **Sub-cell features are detected by the bound, not by sampling.** A rim
finer than a cell has `0 ∈ F_side ∩ F_cap` in its cell; the cone test fails
(the normal turns through 360°), the cell subdivides, and the rim emerges
at the level where its curvature fits — exactly the feature EMC/DC lose (§4).

(c) **Culling work, not duplication.** `research/performance.md` §12.10 is
explicit: "none of it removes *work* ... a `min` over a bounding-box-rejected
branch is a `select`, not a skipped branch". Under XLA a `where` cannot skip,
so the only way intervals remove work is Keeter's: *specialise the program
per region*. In our dispatch model that is one program per distinct
**active leaf set** — the leaves whose interval contains zero on a cell —
with cells grouped by that set (the `project_batched` gather in reverse).
On the end-cap, 42 leaves per evaluation becomes a handful per cell; the
program count is the number of distinct active sets, which on a CSG part is
small. The same census gives every node the conservative bounding volume
§12.10 asks for, computed on the existing graph.

### 5.3 Implementing it on cadjoint's graph

Two routes. (i) An `interval` method on every node — every primitive,
boolean, transform and pattern gets an extension by hand (~40 classes, and
`patch_fields` doubles it). (ii) **Interpret the jaxpr.** `jax.make_jaxpr`
of a node's `__call__` (taken under `vectorized_lowering`, so the same
structure the compile path emits) is a list of primitives — `add, mul, neg,
abs, max, min, sqrt, sin, cos, atan2, dot_general, select_n, integer_pow,
reduce_min/max` — each of which has a textbook interval rule; a small
interpreter maps every intermediate to a `(lo, hi)` pytree, and the
interpretation is itself a JAX function of the box bounds, so it jits and
`vmap`s over all cells in one program. No primitive changes; a new
primitive is a new rule or an error, never a silent unsound bound. Route
(ii) is the one Keeter's tape and Sharp–Jacobson's rules are; recommend it.

Specifics: `smooth_min` is monotone non-decreasing in each argument
(`∂/∂a = 1 − h/(8k) ≥ ½` on one side of the band, `h/(8k) ≥ 0` on the
other), so its *tight* extension is `[smooth_min(a.lo, b.lo),
smooth_min(a.hi, b.hi)]` — one rule, exact, rather than the loose composite
of `abs`/`max`/`mul` the interpreter would otherwise produce. `Rotate`,
`Scale`, `Translate` are linear, and here the **dependency problem** bites:
the interval image of a box under a rotation is the box's rotated AABB,
inflated by up to `√3` per axis, compounding through nested transforms and
`PolarPattern`'s rotated copies (`operations.py:370`, a `min` over rotated
child evaluations). Two fixes: subdivision (intervals converge as O(w), so
halving the box halves the slack — Keeter's answer) or **affine arithmetic**
— represent each quantity as `x₀ + Σ xᵢ εᵢ`; linear maps are then exact and
only nonlinear ops (the `sqrt` in cylinder/sphere, `abs`/`min`/`max` in
booleans, `sin`/`cos` in `Twist`) add a noise symbol, with O(w²) convergence
[CS93, FPC10, SJ22]. In JAX an affine form is a pytree `(centre, coeffs[n],
slack)` with `n` fixed by the graph (3 coordinate symbols plus one per
nonlinear op), so shapes are static and the interpreter jits unchanged.

Cost: an interval op is two to four float ops, so a full-scene census on
the overlay's 64³ = 262k cells is one batched program at ~4× `sample_grid`,
which `performance.md` measures at 0.7 % of a request; the per-patch census
multiplies by the patch count (28–50) but stays one program. The
data-dependent part — bisecting the ambiguous cells — runs on the few
cells that need it, as fixed-depth masked levels or in NumPy.

## 6. Differentiability

- **Positions differentiate by the IFT.** A traced sample solves
  `f_a = f_b = 0`, a vertex three equations; re-running `project` on the
  frozen sample as its own seed gives `dx* = P dx₀ − Jᵀ(JJᵀ)⁻¹ ∂f/∂θ dθ`
  with the tangential projector `P` making the tracer's choice of arclength
  station irrelevant. Nothing changes here as long as the output is stored
  as (sample, pair, vertex ids) and every consumer calls `project`, never
  `trace_curves`, under a traced θ.
- **Topology is frozen and discrete**: how many edges, which pairs, samples
  per edge, which vertex ends which edge, each pair's classifier state, and
  the interval census — recomputed concretely per extraction, like the
  shoelace sign in `drag.py`.

Consequences: curvature-adaptive sampling is set at extraction (a rim that
grows keeps its chord count — fine within a frozen topology); a vertex is an
IFT point only while transversal, so the classifier state belongs in the
record where a consumer can see non-differentiability instead of a silent
zero; and the events that invalidate the record are exactly what
architecture §9.4 wants to bound — a watched field's sign change moving
past a sample, a vertex leaving the scene's zero set, `sin θ` crossing the
floor at any sample, a blend band swallowing a corner (`|f| ≤ 4k`), a
census cell flipping its active set. Each is a scalar in θ at the frozen
samples (the interval bounds are functions of θ too), so the safe drag
range is a 1-D root bracket per event. The viewer must then draw what the
record says: the graph stores the traced samples, overlay and drag path
both render `project_batched(samples, pairs)`, nobody re-traces between
events. LM is the one solver here with no IFT: `tangent` seam positions
carry no derivative, and the record should say so rather than floor one.

## 7. The solvers, placed

| solver | here | belongs in | change |
|---|---|---|---|
| Newton / Gauss–Newton | min-norm `Jᵀ(JJᵀ)⁻¹f`, fixed count, IFT adjoint (`project.py`) | every transversal projection incl. the tracer's corrector | keep; accept the tracer's corrector by contraction, not by count |
| Levenberg–Marquardt | `(JJᵀ + μI)⁻¹`, `μ = ‖f‖²` [YF01] | `project.py` `damped=True`: corrector when `sin θ < 0.3`, `tangent` seam placement, 4-field corner least squares | new, ~30 lines, forward only (no IFT at rank deficiency; keep the guard's zero derivative) |
| SQP | `min ‖Δθ‖² s.t. h(θ) = target, c(θ) = 0`, inequalities | `brep/drag.py` (with the §6 event bounds as trust region), `constraints/solve.py` beside `_newton_projection`, `optimize.py` for constrained design | today's drag is one stacked GN step + `project_to_manifold` — a single linearised SQP iteration, no merit function, no active set, no bounds. Use `scipy.optimize.minimize(method="SLSQP")` with IFT Jacobians for the *concrete* small-dof drag and constrained-optimisation paths (FEM evaluations are the cost, so few SQP iterations beat many optax steps); optax stays for unconstrained studies |
| interval Newton / Krawczyk | `K([x]) = m − Y f(m) + (I − Y J([x]))([x] − m)` | `brep/interval.py` (new): vertex existence/uniqueness per census cell | discrete certificate, no derivative; runs once per extraction |

## 8. Recommended algorithm

### (a) Trace one edge from a seed on `f_a = f_b = 0`

```
trace_edge(a, b, x0, watch, dir, tol, max_step, max_turn):
    # watch: fields whose zero crossing ends the edge — the other patches of
    #        leaf(a), leaf(b), and every patch active (§5) in cells the
    #        pair's curve crosses; for a smooth pair substitute f_b − 4k (§2.2).
    x = corrector(a, b, x0)                       # GN; LM if sinθ(x) < 0.3
    require classify(a, b, x) == crossing         # (c)
    t = dir·unit(∇f_a × ∇f_b)(x);  k = curvature_vector(a, b, x, t)    # §1.1
    s = min(max_step, max_turn/|k|);  samples = [x];  W = sign(watch(x))
    loop:
        x_pred = x + s t + ½ s² k
        x_new, iters, ratio = corrector(a, b, x_pred, contraction=True)
        t_new = unit(∇f_a × ∇f_b)(x_new) aligned with t
        reject if |x_new − x_pred| > ½ s              # branch-jump guard [AG90]
               or angle(t, t_new) > max_turn          # geometric guard
               or ratio > ½ or iters == max            # contraction guard
               or owner_pair(x_new) ≠ {a, b}           # left the edge's surface
               → s ← s/2, retry; below min_step report `broken`
        W_new = sign(watch(x_new))
        if W_new ≠ W:                                 # a third field crossed
            c = the flipped field; x_v = solve3(a, b, c, lerp(x, x_new))
            samples.append(x_v); return samples, vertex(a, b, c, x_v)
        if |x_new − x0| < s and t_new·t0 > cos(max_turn) and len(samples) ≥ 4:
            return samples, None                      # loop closed, same heading
        samples.append(x_new); x, t, W = x_new, t_new, W_new
        k = curvature_vector(...); s = clamp(s·max_turn/angle, min_step, max_step)
```

Trace outward from a known vertex; from an interior seed trace *both*
directions and concatenate — never choose a direction by heading to a target.

### (b) All edges and vertices of a scene from the lattice

```
extract_edges(scene, grid):
    census = interval_census(patches, scene, grid)        # §5.2(a): per cell the
                                                          # active patches, certified
                                                          # pairs, ambiguous cells
    brep0  = discover(scene, grid, active=census)         # graph.py steps 1–3 on
                                                          # per-cell active sets
    pairs  = {(a, b, one seed per chain component)}       # from _build_edges' chains
           ∪ {(a, b, bisection seed) for certified cells no chain covers}
    states = {(a, b): classify(a, b, seed)}               # (c)
    merge faces of every `coincident` pair; drop those pairs
    crossing: trace_edge both ways from each seed
    blend(k): trace the virtual edge f_a=f_b=0 and the seams
              f_a=0 ∧ f_b=4k, f_b=0 ∧ f_a=4k
    tangent:  seam polyline by LM from the chain seeds; no vertices, no derivative
    branch:   split at the branch point along the two roots; trace four half-edges
    vertices = dedupe(vertex(a,b,c) by sorted triple, |Δx| < tol); 4-field corners (§3);
               Krawczyk-certify each against its census cell
    certify:                                              # the completeness test
        every face loop is a closed cycle of traced edges through shared vertices
        every certified (a,b) cell holds a traced sample; every ambiguous cell was
        bisected to tolerance or claimed by a vertex / tangent / branch record
        every edge: residual ≤ tol, no watch field flips inside it
    return faces, edges(samples, pair, state, vertex ids), vertices, census
```

### (c) Tangency classifier at `x` with `f_a(x) = f_b(x) = 0`

```
classify(a, b, x, k_blend):
    g_a, g_b = ∇f_a(x), ∇f_b(x);  sinθ = |g_a × g_b| / (|g_a||g_b|)
    if sinθ > τ_cross (0.1 ≈ 6°, the overlay's _TANGENT_FLOOR):  return crossing
    if the pair's boolean is smooth with radius k_blend:           return blend(k)
    n = unit(g_a); λ = (g_b·n)/|g_a|;  M = (H_b − λH_a) restricted to n⊥   # 2×2
    if ‖M‖ ≤ ε(‖H_a‖ + ‖H_b‖):   return coincident      # same surface to 2nd order
    Δ = −det(M)                    # indefinite ⇔ two real tangent directions
    if Δ < 0:   return contact     # isolated touching point
    if |Δ| ≤ ε: return tangent     # curve of tangency: a seam
    return branch(t1, t2)          # roots of tᵀMt = 0
```

`τ_cross` is the one threshold with a geometric meaning (below it the Gram is
too ill-conditioned for GN); `ε` is float noise. Between `τ_cross` and
`sin θ ≈ 0.3` the pair is `crossing` but the corrector is LM.

## 9. Comparison

| method | at tangency | completeness | cost | differentiability |
|---|---|---|---|---|
| lattice chains + planar re-order (`_sampled_polylines`) | none — folds; planar fit fails off-plane | one seed per DC-visible component, order unreliable | one DC pass + 2 batched projections | positions yes (IFT); order/extent unstable |
| tangent marching, GN corrector (`trace_curves`) | refuses below `sinθ = 0.1`, wanders below ~0.3 | needs a target per open edge; proximity-only loop test | ~4 GN steps × samples, batched | positions yes via re-projection |
| §8(a): curvature predictor, contraction/ownership guards, watch-field termination | LM below 0.3; seams classified | every seeded component, exact ends; loop-closure certificate flags misses | ≈ same, fewer rejects | positions yes; topology + state frozen |
| interval / affine census (§5) + bisection of ambiguous cells | flags tangency (cone test fails), does not resolve it | certified: no branch or corner missed at tolerance, sub-cell included | 2–4× a field evaluation per cell per patch, one program; bisection on the few ambiguous cells | none needed (discrete); bounds are functions of θ, so census flips are bounded events |
| algebraic pairs (quadric∩quadric, trimmed by ownership) | exact classification | complete regardless of lattice, sub-cell included | per pair; trimming needs ownership evaluations | positions yes; curve coefficients differentiable |
| topology-first (Grandine–Klein, TOG23) | full by construction | certified | high; parametric-domain machinery we lack | topology is the output, not differentiable |
| Hermite features (EMC / DC / SW04 / VKKM03) | angle thresholds on samples | one feature per cell, sub-cell lost | one pass | QEF vertex positions only, O(h) at features |

## 10. What survives, what is replaced

**Survives unchanged**: `project.py` (`project`, `project_batched`, the
`custom_vjp`, `_usable`, `_solve_masked`); `graph.py` discovery (patch table,
two-stage ownership, blend test, face components, loops, quad adjacency);
the overlay's design-subtree rule, residual gate, debris pruning,
`_BLEND_AS_EDGE_CELLS`; `drag.py`'s split of autodiff `∂f/∂x` from concrete
`∂f/∂θ`; `_lowering.py` (the interval interpreter reads jaxprs taken under
it, it does not change it).

**Survives, extended**: `trace_curves` — keep the batched `advance` program
and the turn controller; add the parabola predictor, a contraction-checked
corrector with an LM branch, the ½-step branch-jump guard, the ownership
check, watch-field termination, both-direction tracing; drop `targets`.
`project.py` gains `damped=True`; `BRepEdge` gains a `state` and its seams;
`BRepVertex` its full patch set; `BRep` its census.

**New**: `brep/interval.py` — the jaxpr interval/affine interpreter, the
`smooth_min` monotone rule, the per-cell census and cone test, Krawczyk for
vertices; later the active-set grouping that lets the compile path cull.

**Replaced**: `_build_edges`' midpoint polylines (chains become seeds only);
`_build_vertices`' "mesh vertex with ≥ 3 faces" (vertices come from
watch-field crossings keyed by patch triple; the ambiguous count becomes
classifier output); `_link_edge_vertices`, `_corners_on_curve` (sharing is
by construction). In the overlay, `_sampled_polylines`, `_in_curve_order`,
`_planar_parameter`, `_between_corners`, `_attach_corners`, `_MAX_EDGE_TURN`
exist only because seeds were unordered — delete once every edge is traced;
`_traced_polylines` becomes a read of `BRepEdge.samples`.

**Not recommended**: parametric-domain topology resolution (no parameter
domain here), Lipschitz-based pruning (user fields break it; intervals need
only C0), angle-threshold feature detection (the ownership field is exact).

## 11. References (read at the URLs unless marked)

- [PM §5.8.1.5] Patrikalakis & Maekawa, *Shape Interrogation for CAD/CAM*, "Computing starting points for all branches" — https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node105.html; [PM §5.8.2.3] "Marching methods" — node109.html; [PM §6.4, §6.4.1] "Intersection curve at tangential intersection points", "Tangential direction" — node122.html, node123.html
- [BHLH88] Bajaj, Hoffmann, Lynch, Hopcroft, "Tracing surface intersections", CAGD 5(4) 1988 — https://www.sciencedirect.com/science/article/abs/pii/0167839688900106 (abstract)
- [BK90] Barnhill & Kersey, "A marching method for parametric surface/surface intersection", CAGD 7 1990 — https://www.sciencedirect.com/science/article/abs/pii/016783969090035P (abstract)
- [GK97] Grandine & Klein, "A new approach to the surface intersection problem", CAGD 14(2) 1997 — https://www.sciencedirect.com/science/article/abs/pii/S0167839696000246 (abstract)
- [SM88] Sederberg & Meyers, "Loop detection in surface patch intersections", CAGD 5 1988 — https://www.sciencedirect.com/science/article/abs/pii/0167839688900295 (abstract)
- [KM97] Krishnan & Manocha, "An efficient surface intersection algorithm based on lower-dimensional formulation", ACM TOG 16(1) 1997 — https://dl.acm.org/doi/10.1145/237748.237751 (abstract)
- [YM99] Ye & Maekawa, "Differential geometry of intersection curves of two surfaces", CAGD 16(8) 1999 — https://www.sciencedirect.com/science/article/abs/pii/S0167839699000187 (abstract)
- [TOG23] "Topology Guaranteed B-Spline Surface/Surface Intersection", ACM TOG 2023 — https://dl.acm.org/doi/10.1145/3618349 (title only)
- [Hartmann98] Hartmann, "A marching method for the triangulation of surfaces", Visual Computer 14(3) 1998 — https://www2.mathematik.tu-darmstadt.de/~ehartmann/pub/tri_abs/tri_abs.html; [Wiki-IC] "Implicit curve" — https://en.wikipedia.org/wiki/Implicit_curve
- [Bloom94] Bloomenthal, "An implicit surface polygonizer", Graphics Gems IV — https://people.eecs.berkeley.edu/~jrs/meshpapers/Bloomenthal.pdf (via survey)
- [WGG99] Wyvill, Guy, Galin, "Extending the CSG tree ...", CGF 18(2) 1999 — https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-8659.00365; [WvO96] Wyvill & van Overveld, polygonization with CSG (cited there)
- [KBSS01] Kobbelt, Botsch, Schwanecke, Seidel, "Feature sensitive surface extraction from volume data", SIGGRAPH 2001 — https://graphics.stanford.edu/courses/cs164-10-spring/Handouts/paper_p57-kobbelt.pdf (full text)
- [JLSW02] Ju, Losasso, Schaefer, Warren, "Dual contouring of hermite data", SIGGRAPH 2002 — https://www.cs.rice.edu/~jwarren/papers/dualcontour.pdf
- [SW04] Schaefer & Warren, "Dual contouring: the secret sauce", Rice TR 2004 — https://people.eecs.berkeley.edu/~jrs/meshpapers/SchaeferWarren2.pdf
- [VKKM03] Varadhan, Krishnan, Kim, Manocha, "Feature-sensitive subdivision and isosurface reconstruction", IEEE Vis 2003 — https://dl.acm.org/doi/abs/10.1109/VISUAL.2003.1250360
- [PV04] Plantinga & Vegter, "Isotopic approximation of implicit curves and surfaces", SGP 2004 — https://pure.rug.nl/ws/files/2952308/2004ProcGeomProcPlantinga.pdf
- [Snyder92] Snyder, "Interval analysis for computer graphics", SIGGRAPH 1992 — https://www.gg.caltech.edu/papers/intervalabstract.html (abstract); [Duff92] Duff, "Interval arithmetic and recursive subdivision for implicit functions and constructive solid geometry", SIGGRAPH 1992 — https://dl.acm.org/doi/10.1145/133994.134027 (abstract)
- [Keeter20] Keeter, "Massively parallel rendering of complex closed-form implicit surfaces", SIGGRAPH 2020 — https://www.mattkeeter.com/research/mpr/ ; code https://github.com/mkeeter/mpr (libfive)
- [SJ22] Sharp & Jacobson, "Spelunking the Deep: guaranteed queries on general neural implicit surfaces via range analysis", SIGGRAPH 2022 — https://arxiv.org/abs/2202.02444
- [FPC10] Fryazinov, Pasko, Comninos, "Fast reliable interrogation of procedurally defined implicit surfaces using extended revised affine arithmetic", Computers & Graphics 34(6) 2010 — https://www.sciencedirect.com/science/article/abs/pii/S009784931000107X (abstract)
- [GYBC24] Gillespie, Yang, Botsch, Crane, "Ray tracing harmonic functions", SIGGRAPH 2024 — https://dl.acm.org/doi/10.1145/3658201 (abstract; cited to correct the attribution)
- [YF01] Yamashita & Fukushima, "On the rate of convergence of the Levenberg–Marquardt method", Computing Suppl. 15, 2001; [LM-EB] https://arxiv.org/pdf/1703.07461 (abstracts)
- Not fetched, from the books: [AG90] Allgower & Georg, *Numerical Continuation Methods*, 1990; [Moore] Moore, Kearfott & Cloud, *Introduction to Interval Analysis*, 2009 (Krawczyk operator); [CS93] Comba & Stolfi, "Affine arithmetic and its applications to computer graphics", SIBGRAPI 1993; Nocedal & Wright, *Numerical Optimization*, ch. 10, 18; quadric∩quadric closed forms (Levin 1976; Dupont et al. 2008), named for §8(b)'s algebraic seeding only.
