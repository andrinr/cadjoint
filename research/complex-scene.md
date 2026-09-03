# Pushing the modelling language with a real part

*What broke when a genuinely complex mechanical part was modelled in cadjoint,
what was fixed, and what was deliberately left alone. The part is
`scenes/end_cap.py`; every number below was measured on this branch, on this
machine, warm.*

---

## 1. The part, and why this one

A cast aluminium **gearbox output end-cap**: the thing that closes a gearbox
and carries the output shaft's bearing. It was chosen because it has to do
four unrelated jobs at once, and each one lands on a different corner of the
modelling language rather than exercising the same corner four times.

| job | feature | what it tests |
| --- | --- | --- |
| carry the bearing | stepped seat + circlip groove | `revolve`, and the `Axis` it declares |
| | through bore | `Face.hole` sharing a live `Scalar` |
| bolt to the case | four corner holes | `PolarPattern` about a feature's own axis |
| stay stiff | eight gusset ribs | one constrained sketch + `PolarPattern` |
| locate on the case | dowel above and below the flange | `Mirror` across a derived `midplane` |
| feed oil | flared side port | `loft`, and a `PolarPattern` about a *horizontal* line |
| seal | lip seal | `Shell` |

The feature tree is three `SketchPlane.on` links deep — flange → bearing boss
→ seal land → retainer pad — with each plane an expression in the feature
below it. Four materials (aluminium, bronze, steel, nitrile), six cuts, both
pattern kinds, a `ThermalStudy` on a `SimMesh` scoped to the housing.

Measured on the built scene:

```
module exec (incl. constraint solve)     1.9 s
world-frame CSG leaves, housing           14
world-frame CSG leaves, whole scene       17
total profile vertices, whole scene      122
construction nodes in the viewer payload  12
hex mesh of the housing        3 198 nodes / 2 005 elements
thermal solve, in process                2.5 s
peak temperature                       1.3067
```

---

## 2. Limitations found

Ten. Five fixed here with tests and docs, five recorded and deferred with a
reason. They are ordered by how much they cost the person modelling.

### 2.1 FIXED — a face reference did not survive a boolean

**What was tried.** Build the housing, then sketch the next feature on it:

```python
housing = Difference(body, bore, bolt_holes)
pad = PolygonProfile(PAD, plane=SketchPlane.on(housing.cap("+")))
```

**What happened.** `AttributeError: 'Difference' object has no attribute 'cap'`.
`attach_faces` binds `faces`/`face`/`cap`/`side`/`axis` onto the instance the
*generator* returned, so the reference existed on `flange` and died the moment
the flange was combined with anything. In practice that means: the instant a
plate has a hole in it, it stops being something you can sketch on — which is
most of what feature-based CAD is.

The workaround is to keep a variable pointing at every pre-boolean feature and
sketch on *that*, which is exactly the "stored surface" bookkeeping the face
system exists to remove.

**The fix.** `SDF.__getattr__` forwards those five names to the node's *first*
child, gated on an opt-in `inherits_faces` class flag. `BooleanOp`,
`LinearPattern` and `PolarPattern` set it; nothing else does.

The gate is the whole design. A boolean does not move the base body's surface
— cutting a hole in a plate leaves the plate's top face in the same plane —
and copy 0 of a pattern *is* the original, so a forwarded plane is genuinely
on the result's own surface. Every affine transform, `Shell` and `Offset`
displaces that surface, so those keep raising: a face quietly in the wrong
place is far worse than a loud `AttributeError`. Only the *base* operand is
consulted, never a later one, so a `Difference` never hands back a face of the
thing being removed.

**What the fix does not give you.** The forwarded face keeps its plane exact
and its *boundary polygon* stale: a face a tool has since carved into reports
the outline it had before the cut, and `Face.contains` is correspondingly
optimistic there. That is the same contract a B-rep modeller offers when it
keeps a face's identity across features, and it is documented on
`SDF.__getattr__`.

Tests: `tests/sdf/test_face_inheritance.py` (25).

### 2.2 FIXED — a redundant constraint turned the whole model into NaN

Found by accident, and the most serious thing in this document.

**What was tried.** The gusset sketch was constrained the way a person draws
it: horizontal root, vertical inner edge, square corner, named length, sloping
face collinear.

**What happened.** Nothing, in float32. Then `cap_mesh.build()` raised
`No cell center lies inside the SDF`, and every free parameter in the program
— including `flange_thickness` and `bore_radius`, which the sketch does not
touch — read `nan`.

The cause: `VerticalConstraint` was implied by the horizontal plus the
perpendicular, so the constraint Jacobian was rank-deficient, `J Jᵀ` singular,
and `jnp.linalg.solve` answered with NaN. `satisfy_constraints` then wrote
that NaN back over *every* free parameter in the scene.

Two things made it nasty:

- **It was silent.** No error, no warning; just geometry that evaluates to NaN
  everywhere and a mesher that reports an empty domain.
- **It depended on the dtype.** In float32 roundoff masks the singularity and
  the solve succeeds. Only the FEM path's x64 makes it exact — so a scene
  renders perfectly and dies the moment it is meshed, with the symptom two
  subsystems away from the cause.

**The fix.** Solve the same Newton step as a least-squares problem
(`jnp.linalg.lstsq`) rather than an exact inverse. The two agree to rounding
whenever `J Jᵀ` is invertible, and where it is singular the least-squares step
is also the semantically right answer: take the minimum-norm correction over
the constraints that are actually independent. A redundant-but-consistent
constraint set is an ordinary thing to draw and every CAD sketcher tolerates
it.

Tests: `tests/constraints/test_redundant.py` (6), in float32 *and* float64,
asserting the constraints end up satisfied rather than merely finite.

Not done, and worth doing separately: **report** the redundancy. Tolerating it
is right, but a sketch with a rank-deficient Jacobian is still information the
modeller wants ("this relation is implied"), and nothing surfaces it.

### 2.3 FIXED — a pattern and a mirror could only address the world origin

**What was tried.** A bolt circle about the bearing's axis, and three screws
about the lubrication port's horizontal axis:

```python
bolt_holes = PolarPattern(bolt_hole, count=4, axis=bore_axis)
port_screws = PolarPattern(port_screw, count=3, axis=port_axis)
```

**What happened.** `PolarPattern only supports axis 'z'`, and `Mirror` took
only `'x'`/`'y'`/`'z'`. Both meant the *world* coordinate plane or axis
through the *world origin* — the one line a real part's features are almost
never on. The workaround is to wrap the child in `Translate`/`Rotate` to bring
the feature to the origin, pattern it, and move it back, which puts the
placement in two places at once and breaks as soon as the part moves.

**The fix.** `PolarPattern(..., axis=Axis(origin, direction))` and
`Mirror(..., Face | SketchPlane)`. `PolarPattern`'s *string* form still
accepts only `'z'`, deliberately: a letter can say which way an axis points
but not *where it is*, and where it is happens to be most of what a bolt
circle means, so a second spelling of the same underdetermined thing would be
a trap. `Mirror` keeps its three letters because a coordinate plane through
the origin is a real, common answer for a symmetric part.

Both default paths are **numerically identical** to the previous
implementations — bit-for-bit on sampled points — which the tests assert,
because the general Rodrigues form folds to the old expression when the axis
is `+z` through the origin and the angles are static.

The end-cap uses `bore_axis = seat_cut.axis`: the line the bearing seat was
revolved around is the line the bolt circle turns about, stated once.

### 2.4 FIXED — a profile could not draw a curve

**What was tried.** A round flange, a circular boss, a circular loft section.

**What happened.** `PolygonProfile` takes vertices and nothing else, so a
32-segment bore meant typing 32 coordinate pairs, and a rounded flange corner
meant computing an arc by hand. That is not a nuance; it makes any part with
a circular feature unwritable.

**The fix.** `PolygonProfile.circle`, `.regular` and `.rounded_rect` generate
the vertex list. `.rounded_rect` is the honest way to get a rounded corner *in
a profile*: the corner is traced as vertices before the solid exists, so the
round survives extrusion exactly and costs nothing to evaluate.

**The deliberate part.** Generated vertices are **pinned** (`free=False`),
unlike typed ones. An individual vertex of a circle is not a design freedom —
dragging one makes the circle not-a-circle — and freeing 40 of them would
swamp the design space with meaningless variables. `free=True` is available
for the case where you do want them loose.

A zero corner radius is special-cased to emit the plain rectangle: the arcs
would otherwise collapse to repeated points, and a zero-length edge is a
divide-by-zero in the polygon distance. That one is a latent trap in
`polygon_sdf_2d` for any caller, not just these.

### 2.5 FIXED — a face knew its plane but not how to be cut

**What was tried.** Put a bore in a face.

**What happened.** Nothing existed. You built a `Cylinder`, computed Euler
angles from the face normal by hand, translated it to the face origin less
half the depth, and hoped. Every hole in the part is that same six-line
incantation.

**The fix.** `Face.hole(radius, depth, at=..., through=...)` and
`Face.pocket(vertices, depth, ...)` return the **tool**, plus `Face.center`,
`Face.point` and `Face.plane` as the anchors they need.

Returning the tool rather than a cut solid is the important choice. In an
implicit modeller the cut *is* the boolean, so keeping them apart is what lets
one tool be patterned, mirrored, or subtracted from several bodies at once —
which is exactly how the end-cap's bolt circle is written: one `Face.hole`,
one `PolarPattern`, one `Difference`.

The bore is a real `Cylinder` on the face normal, not a polygonal
approximation, so it stays round at every render scale and — the reason it
matters — its radius stays a live `Parameter`. `bore_radius` is
finite-difference-checked through the whole housing because of this.

`Face.plane(offset=...)` is the other quietly important one; see 2.6.

---

### 2.6 DEFERRED — every extrusion straddles its sketch plane

`extrude(profile, depth)` spans `±depth/2` about the plane. There is no
one-sided extrude, so a boss that should *sit on* a face has to be sketched on
that face pushed up by half its own depth:

```python
boss_plane = flange.cap("+").plane(offset=boss_height / 2.0)
```

The offset must be kept in sync with the depth by hand, in two places. This is
the single most repeated wart in the scene — it appears at every level of the
face chain.

**Why deferred.** The fix (`extrude(..., symmetric=False)`, spanning
`[0, depth]` above the plane) also has to reach `extrusion_faces`, which reads
`profile.plane` directly to place the caps. Doing it properly means threading
a placement plane through the generator and its face builder together, which
changes what `profile.plane` *means* for every face consumer. That deserves
its own change with its own tests, not a rider on this one.

`Face.plane(offset=...)` was added as the ergonomic half of it, so the idiom
is at least one legible expression instead of a `SketchPlane.offset` call.

### 2.7 DEFERRED — a derived plane is not a differentiable link

`faces.py` and `sketch.py` both claimed that "a gradient taken through a child
solid reaches the parent's `depth`", and called it the thing a B-rep cannot
do. Measured:

```
base depth 0.4 -> boss cap+ at z = 0.35     the plane does follow
base depth 0.5 -> boss cap+ at z = 0.40

d(boss sdf)/d(base_depth), by rebuild + finite difference   -0.499994
d(boss sdf)/d(base_depth), reverse mode via functionalize   +0.000000
free parameters reaching the boss: ['boss_v0'...'boss_v3']   -- no base_depth
```

Half the claim is true and it is the useful half: the plane is *re-derived*
every time the program runs, so editing `flange_thickness` really does lift
the whole stack, which a stored B-rep surface cannot do. But the derived
origin is evaluated at construction time and stored as an ordinary fixed
`Parameter`, so `extract_parameters` on a child never sees the parent's depth
and the gradient is exactly zero.

**Why deferred.** Making it live needs *derived* parameters — a `Parameter`
that is an expression over other parameters rather than a leaf holding a
value. That is an architectural change to the parameter model, which
`extract_parameters`, `functionalize`, the constraint residual packer and the
WGSL path all depend on being flat. Well out of scope here.

**Done instead:** both docstrings now say what actually happens, and the
scene's own module docstring warns the reader, because the consequence is
practical — a driving parameter must be handed *directly* to the feature it
dimensions to survive into the gradient. That is why `bore_radius` is a
`Face.hole` radius and not a 32-gon drawn at that radius.

### 2.8 DEFERRED — `loft` pairs vertices by index, so two generated outlines do not line up

Lofting a circle to a rounded rectangle with the same vertex count produces a
visibly skewed transition: `circle(16)` puts vertex *i* at 22.5·*i* degrees,
while `rounded_rect(…, segments=3)` distributes its 16 points by corner arc,
so vertex 4 of one sits at 90° and vertex 4 of the other at about 127°. The
loft is doing exactly what it documents — vertex *i* of A to vertex *i* of B —
and the outlines simply disagree about what *i* means.

**Why deferred.** The fix is a resampling rule (match by normalized arc
length, or by angle about the centroid) and it is a real design decision:
arc-length matching is right for a smooth transition and wrong when the user
*wants* a specific corner to map to a specific corner. It needs an opt-in and
a way to state the correspondence, which is a feature, not a patch.

**Worked around** in the scene by lofting two profiles from the same generator
at the same segment count, where the correspondence is exact by construction.
Noted in the scene's comments so the next person does not rediscover it.

### 2.9 DEFERRED — a generated profile is invisible to the source map

The viewer's construction payload confirms it:

```
profile_0   flange         line=100 editable=False faces=22 ops=[]
profile_3   retainer pad   line=138 editable=True  faces=6  ops=['extrude']
```

A profile built by `PolygonProfile.rounded_rect(...)` renders and declares its
faces, but reports `editable: False` and no operators, so its vertices cannot
be dragged and its faces are marked unusable as sketch targets. The cause is
in `cadjoint/viewer/source_map/calls.py`: `locate_profile_call` matches a call
*named* `PolygonProfile` whose vertices are literal two-number lists, and a
classmethod call is named `rounded_rect`.

This is defensible as far as vertex dragging goes — a generated vertex should
not be draggable, which is the same argument that pins it (2.4) — but it is
wrong for `set_sketch_plane`: there is no reason a boss on a generated
circular flange should not be re-planted from the viewer.

**Why deferred.** It is viewer work, in files another agent is actively
changing on this branch, and the right fix (teach the locators the three
generator classmethods, and pair operators for them) is a self-contained
change that will conflict less on its own.

**Practical consequence, worked around:** the sketches the scene means you to
edit — the retainer pad and the gusset rib — are written as literal `Vector2`
points, and both round-trip through the viewer (§4).

### 2.10 DEFERRED — an SDF cannot fillet a selected edge, and should stop implying it

Worth stating plainly because it is the question every mechanical engineer
asks first.

`Union(..., smoothness=k)` rounds **where two operands meet**, and that is a
genuine, useful fillet — the end-cap uses `smoothness=0.04` on its union to
round the rib-to-flange and boss-to-flange roots, which is what a casting
does. `Difference(..., smoothness=0.012)` breaks the edges of the bores the
same way.

What it cannot do is round *this edge and not that one*. A smooth boolean is a
property of the pair of fields, not of a curve on the result, so:

- the blend applies to **every** place those two operands meet;
- an edge internal to one operand (a polygon profile's own corner) has no
  second field to blend against and cannot be filleted at all — the honest
  answer there is to round it *in the profile*, which is what
  `rounded_rect` now makes possible (2.4);
- the blend radius is a field parameter, not a measured radius; the resulting
  surface is not a circular arc of radius `k`.

`Offset(Offset(x, -r), +r)` gives a genuine constant-radius rounding of
concave features, but again globally, and it doubles the field evaluation
cost.

**No fix attempted.** Selective edge filleting needs edge identity on the
result, which is exactly what an implicit representation does not carry. The
existing `patch_fields` protocol is the nearest thing — it knows where a
node's own feature edges are — and a `Fillet(node, patch_pair, radius)` built
on it is conceivable, but it would only work within a single primitive, not
across a boolean, which is where users want it. Better to document the
boundary than to ship something that rounds the wrong edges.

---

## 3. Performance: the edge overlay is priced in profile vertices

The one stage that does not fit. Driving a live playground, warm cache:

```
compile          8.49 s   ok, 12 construction nodes
mesh overlay    90.06 s   FAILED: exceeded the 90-second timeout
mesh_inspect    11.07 s   ok, 3 198 nodes / 2 005 elements
simulate        22.50 s   ok, temperature range [0.0, 1.3067]
```

`/api/mesh` is the *edge* overlay — the crease and seam wireframe — and it is
the only thing here that fails. The simulation mesh (`mesh_inspect`), the
solve, the compile and every patch are comfortably inside their budgets.

**The mechanism is not the one `research/performance.md` describes**, because
that document's finding has since been fixed. It attributed the cost to a
fixed ~0.5 s per *seam group*; the base branch's `_project_seam_groups` now
projects every group in one program, and on this part that took the overlay
from 212 s to 118 s. Seam projection is no longer dominant. Profiling what is
left (17 leaves, 148 s under cProfile):

| stage | cumulative |
| --- | ---: |
| `edge_hermite_data` | 71.7 s |
| `_project_seam_groups` (already batched) | 21.9 s |
| `sample_grid` | 16.2 s |
| — underneath all of them, `_polygon_distance` | 38.7 s |

`_polygon_distance` unrolls **one op chain per profile vertex** — deliberately,
so the WGSL backend only ever sees `vec2` math — so every stage that evaluates
the scene costs linearly in the *total vertex count of every profile in it*.
That is the price of a polygon-only profile representation, and it is what the
new generated outlines (§2.4) buy circles with.

Rebuilding the same part at three segment budgets, leaf count held at 17:

| budget | total profile vertices | edge overlay (warm, one process) |
| --- | ---: | ---: |
| as first written | 168 | 112.2 s |
| as shipped | 116 | 59.7 s |
| coarse | 98 | 53.6 s |

1.45× the vertices cost 1.9× the time; the leaf count never moved. The scene
ships at the middle row — every circle is the coarsest count that still reads
as round — and the segment counts carry a comment saying they are a budget.

Even so it does not quite fit. In a **cold** process, which is what the
`/api/mesh` worker actually is, the shipped scene costs **77.1 s** of overlay
on top of 1.9 s of module exec, and the request as a whole crosses 90 s.

Three things worth saying plainly:

1. **The budget is set against the wrong quantity.** `MESH_TIMEOUT_SECONDS`
   caps wall clock on something that scales with how many vertices the part's
   profiles carry. A bearing boss is a circle; drawing it as a 24-gon rather
   than a 40-gon is a rendering decision that should not decide whether the
   viewer works.
2. **A polygon-only profile makes every curve expensive twice** — once in the
   vertex count a circle needs to look round, and again because that count is
   an unrolled op chain rather than a closed form. A native `Circle`/`Arc`
   profile primitive with an analytic distance would collapse both. That is a
   real feature request and the largest single lever here.
3. **This part is not exotic.** Eight ribs, four bolts, three screws, six cuts
   and five circular features is a normal casting. The edge overlay is the
   only part of the toolchain that could not take it.

## 4. Viewer round-trip

The scene loads from `scenes/end_cap.py`, compiles, lists its nodes, solves,
and accepts patches that go back through the source. Measured against a live
playground on port 4517:

**Construction payload** — 12 nodes: 11 profiles and one `ConstructionPrimitive`.
The studies (`cap-conduction`), meshes (`cap-mesh`) and all four materials are
listed. Faces are present on the generated profiles too (22 on the flange, 42
on the boss), because `register_feature` runs regardless of whether the source
map can find the call.

**Three patches, each recompiled from the returned source:**

| op | result |
| --- | --- |
| `set_vertex` on the retainer pad, index 1 | rewrote `pad_outer_low = Vector2(value=[0.52, -0.07], …)`; recompiled ok in 4.7 s |
| `set_sketch_plane` on the gusset rib, onto `boss.cap("+")` | inserted `plane=SketchPlane.on(boss.cap('+'))`; recompiled ok in 4.5 s |
| `set_value` on the bolt head's `radius` | ok |

The `set_sketch_plane` round-trip is the one worth reading twice. After the
patch the payload reports

```
plane reference: {'constructor': 'on', 'owner': 'boss', 'accessor': 'cap', 'argument': "'+'"}
plane origin:    [0.0, 0.0, 0.6]
```

— the plane was re-derived from the boss's cap at z = 0.60, not from a
number written into the file. A face reference survived source → payload →
patch → source → payload intact.

---

## 5. Differentiability

Not an optimisation; a proof that the part is still traceable after all of the
above. `housing_volume` is the smoothed aluminium volume of the housing,
sampled on a 17×17×13 lattice, as a function of the free parameters.

| parameter | reverse mode | central difference (h = 2e-3) | relative |
| --- | ---: | ---: | ---: |
| `flange_thickness` | +2.809313 | +2.809316 | 1.0e-6 |
| `bore_radius` | −0.927981 | −0.927925 | 6.1e-5 |

Signs are physical: a thicker flange adds metal, a wider bore removes it. Both
are asserted in `tests/scenes/test_end_cap.py`.

Both parameters are shared `Scalar`s handed directly to the feature they
dimension — `flange_thickness` is the flange's extrusion depth, `bore_radius`
is a `Face.hole` radius. That is not incidental; per 2.7 it is the *only* way
a parameter survives into the gradient, and a scene that drives geometry
through a derived plane or a generated vertex will silently get zeros.

---

## 6. Summary

| # | limitation | status |
| --- | --- | --- |
| 2.1 | face references died at the first boolean | **fixed** — `inherits_faces` forwarding |
| 2.2 | redundant constraint → NaN across the whole program, dtype-dependent | **fixed** — least-squares Newton step |
| 2.3 | `PolarPattern`/`Mirror` could only address the world origin | **fixed** — `Axis` / `Face` / `SketchPlane` |
| 2.4 | profiles could not draw curves | **fixed** — `circle` / `regular` / `rounded_rect` |
| 2.5 | no way to cut a face | **fixed** — `Face.hole` / `pocket` / `center` / `point` / `plane` |
| 2.6 | extrusion is always symmetric about its plane | deferred — needs a placement plane through the face builders |
| 2.7 | derived planes are rebuild links, not gradient links | deferred — needs derived parameters; docs corrected |
| 2.8 | `loft` pairs vertices by index | deferred — needs a correspondence rule and an opt-in |
| 2.9 | generated profiles invisible to the source map | deferred — viewer files under active change |
| 2.10 | no selective edge fillet | won't fix — documented boundary of the representation |
| 3 | edge overlay is priced in profile vertices, and a normal casting does not fit its 90 s budget | **reported** — cut 168 vertices to 116; the real lever is an analytic arc profile |
