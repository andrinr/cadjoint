# Pushing the modelling language until it gives out

*A second, harder part. `scenes/end_cap.py` found ten limitations
(`research/complex-scene.md`); this one was built to find the next ten, and to
find out where the **pipeline** — shader, mesher, B-rep, tets, FEM, optimiser,
viewer — stops keeping up rather than where the language does. The part is
`scenes/motor_shield.py`. Every number below was measured on this branch, on
this machine, with the cache state stated.*

---

## 1. The part

A **liquid-cooled motor end shield**: the drive-end shield of a
totally-enclosed electric motor, cast in aluminium, with a helical coolant
passage cast into the wall of its bearing tower. It was chosen over the
gearbox end-cap because every job it does lands on a *harder* corner than the
last part did, and because several of them land on the B-rep extraction's
known failure classes **on purpose**. That extraction is no longer part of
this repository; §6 records what it did while it was.

| job | feature | what it tests |
| --- | --- | --- |
| cool the bearing | helical channel | `extrude(twist=…)` — the language has no sweep-along-a-curve |
| | inlet/outlet bores | a bore *tangent* to and *coincident* with the helix's end discs |
| | two cast risers | cylinders **externally tangent** to the tower (`cyl_tangent`) |
| | two pipe flanges | a flange face **coplanar** with a boss cap (`boxes_coplanar`) |
| carry the bearing | stepped seat + circlip groove + seal counterbore | three `revolve`s on one `Axis` |
| | grease reservoir + nipple bore | a 0.85-cell wall, a half-cell bore |
| | through bore | `Face.hole` sharing a live `Scalar` |
| bolt to the stator | four corner lugs | extruded in the flange's own plane → **coplanar caps** |
| | counterbored bolt circle | a hard `Union` of two tools, then `PolarPattern` |
| | four tie bolts, a locating spigot | a schedule of angular stations; two coaxial cylinders 0.5 cell apart |
| carry the shroud | four pairs of tapped holes | `LinearPattern` **inside** a `PolarPattern` |
| move air | seven twisted blades + hub | `twist` as a named free-ish parameter, `PolarPattern` |
| | conical shroud, 8 louvres | `Shell` two viewport cells thick, cut by a patterned box |
| hold the fan | threaded shaft end | a 16-gon with one tooth, twisted 2.5 turns |
| | **diamond**-knurled locknut | `Intersection` of one profile twisted both ways |
| | splined drive end | a toothed profile |
| stiffen | six gussets on an **eight**-station ring | `PolarPattern(skip=(3, 5))` — new |
| | four tangential stiffeners | a plane meeting a cylinder along a line |
| cast it | drafted drain boss | `extrude(draft=…)` — and the faces it costs |
| | nameplate pad on the tower wall | `SketchPlane.tangent` on a surface with no face |
| instrument it | encoder pocket + cable gland | `Face.pocket` in the face's own frame; an inclined tool |

The feature tree is three `SketchPlane.on` links deep (flange → bearing tower
→ seal boss → retainer pad), plus one `SketchPlane.tangent` link off the
tower's curved wall. Eight materials with handbook properties, eighteen cuts,
both pattern kinds and a pattern of patterns, a `SimMesh` in hex and one in
tet10, a thermal *and* an elastic study on node selections that are the part's
own cylinders, and an optimisation with a mass budget and a minimum-wall
constraint as penalties.

Of those, the hex mesh, both studies and the optimisation all run (§7, §8);
**the tet10 mesh does not build at all, on either mesher** (§5.2), which is
the hardest single ceiling the part found and the reason it is still declared
in the file.

Sub-cell and super-cell fillets sit side by side on purpose: the tower root
gets a 0.12 blend (1.3 viewport cells, the size the axiom battery says
fragments into slivers), everything else 0.03 (0.3 cells, the size it says is
invisible to the graph), and the cuts 0.012.

---

## 2. Limitations found

Eight fixed here with a test each, twelve recorded and deferred with a proposed
design. The fixed ones come first (§2.1 – §2.8), then the deferred
(§2.9 – §2.20); within each block they are roughly in the order they were
found. The three most expensive are not first: **§2.6 stopped both studies
from running at all**, §2.8 stopped the optimisation, and §2.19 is the one
that most limits what a file in this language can be.

### 2.1 FIXED — a pattern could not leave one of its stations empty

**What happened.** The coolant gallery is fed from a cast riser and runs
radially inward to the helix. It therefore crosses the ring of gusset ribs.
At 227° and 133° — stations 5 and 3 of an eight-way ring — the feed bore goes
straight through a rib's root, and a rib with a 0.22 hole in its root is not a
rib. There is no phase of an eight-way ring that clears the schedule: the
riser bores, the tie bolts and the encoder pocket between them forbid every
residue mod 45°, and I checked all of them.

Every CAD system has the answer — *suppress instance 3 and instance 5* — and
this one did not. `PolarPattern` emitted `count` copies and that was the whole
of its vocabulary. The alternatives were to abandon the pattern and write six
ribs out by hand (six sketches, six sets of constraints, no shared
`rib_thickness`) or to let the bore drill the rib and say nothing.

**Fix.** `skip=` on both patterns:

```python
ribs = PolarPattern(rib, count=8, axis=bore_axis, skip=(3, 5))
holes = LinearPattern(hole, direction=[1, 0, 0], count=6, spacing=0.2, skip=(2,))
```

The kept copies keep the angles and offsets they would have had — suppressing
one leaves a gap rather than respacing the ring, which is the only behaviour
that lets a suppressed station be *used* by something else. Instance 0 is the
seed the child's analytic faces are declared against and cannot be suppressed;
the error says to rotate the seed instead.

`cadjoint/sdf/operations.py`, tests in `tests/sdf/test_operations.py`
(`TestSuppressedInstances`) and `tests/scenes/test_motor_shield.py`.

**The one ugly part**, recorded here because it is a design smell rather than
a bug: the suppressed set travels as a **bit mask in one scalar parameter**
(`skip_mask`), because the parameter plumbing carries numbers and a set of
indices is not one. `static_params` already had to exist so that `count` stays
a Python int under `jit`; `skip_mask` rides the same rail. The cost is a cap
at 24 instances when anything is suppressed (checked, with an error that says
so). The real fix is a *structural parameter* kind that may hold any hashable
Python value and is guaranteed never to be traced — see §2.19.

### 2.2 FIXED — `SketchPlane.tangent` read the normal off the wrong side of the surface

**What happened.** The nameplate pad is sketched on the tower's cylindrical
wall, which has no analytic face, so its plane comes from
`SketchPlane.tangent`. The plane came back with its normal pointing **into**
the solid, and with length 0.94 rather than 1.

`tangent` projected the point onto the zero set and then took
`jax.grad(field)` *there*. But the zero set is exactly where a CSG field
changes branch: an extruded polygon's field is `maximum(polygon_distance,
cap_distance)`, and on the wall the first of those is exactly zero. Autodiff
is entitled to return any subgradient of a `maximum`/`where` at the tie, and
for this field it returned one pointing the wrong way. The existing tests
never saw it because they all used a `Sphere`, whose `|p| - r` is smooth
through its own surface.

Silently flipping a sketch frame is about as bad as a modelling bug gets: the
pad still straddled the wall (an extrusion straddles its plane), so nothing
looked wrong, but its `u` axis was mirrored and any hole placed at `at=(x, y)`
would have gone to `(-x, y)`.

**Fix.** Read the normal by a **central difference across the surface**
instead. Both samples are off the surface, where the field is smooth and its
gradient is the real one; the result is still an expression in the shape's
parameters, which is the property the plane needs. A `normal_step` keyword
(default `1e-3`) sets the half-width.

`cadjoint/construction/sketch.py` (`_surface_normal`), tests in
`tests/construction/test_reference_planes.py`
(`test_the_normal_points_out_of_a_faceted_wall`,
`test_a_pad_on_a_faceted_wall_straddles_it`).

### 2.3 FIXED — no node selection could pick a bore, a seat wall or a jacket

**What happened.** Both studies need boundary conditions on *annular* regions
of the part's own axis: the bearing seat wall for the friction heat and the
belt pull, the coolant jacket for the cold sink. `Nodes` offered box, sphere,
halfspace, side and predicate. A box picks up the corners of the tower, a
sphere the wrong height, and `Nodes.predicate` is not serializable — so the
study could not be captured by the viewer at all.

**Fix.** `Nodes.cylinder(center, axis, radius, inner=…, half_length=…)`,
hollow and axially bounded, serializable both ways.
`cadjoint/fem/selection.py`, `cadjoint/fem/study.py`, `docs/simulation.qmd`,
tests in `tests/fem/test_selection.py`.

### 2.4 FIXED — an inclined tool's three rotation angles were always free

**What happened.** `ConstructionPrimitive` cast `rotation` through
`jnp.asarray`, so a rotation could only ever be three numbers, and
`_free_scalar` then made all three **free optimisation variables**. Nine
inclined tools in this part (the two feed bores, the gland boss and bore, the
gland nut, the two gallery bosses…) meant twenty-seven free angles the
optimiser was entitled to tilt. The first optimisation run did exactly that.

**Fix.** `rotation` accepts `Scalar`s and adopts them as-is, so a bore drilled
at a fixed inclination pins its angles the same way it pins its radius:
`rotation=[Scalar(math.pi / 2, free=False), …]`.
`cadjoint/construction/solid.py`, tests in `tests/construction/test_solid.py`.

### 2.5 FIXED — two unnamed primitives in one program collided

**What happened.** Every primitive names its parameters after itself
(`cylinder_position`, `cylinder_radius`), and `extract_parameters` refuses two
free parameters with one name. So the *second* unnamed `Solid.cylinder` in a
program was an error — and not at the call site: at the first constraint solve,
hundreds of lines later. Same for a second unnamed `PolygonProfile`.

**Fix.** Per-kind counters: `cylinder`, `cylinder_2`, …; `profile`,
`profile_2`, …. `cadjoint/construction/solid.py`,
`cadjoint/construction/sketch.py`, tests in `tests/construction/`.

### 2.6 FIXED — a hole erased the material of the metal around it

**What happened.** Both studies refused to run. Not "gave a strange answer" —
refused, on the whole domain:

```
ValueError: study 'bearing-heat': the scene's material field does not specify
'conductivity' for 3748 of 3748 elements. Give every material in the simulated
domain a 'conductivity' value (see cadjoint.materials for a catalogue of real
ones), or pass an explicit scalar to the study.
```

Every element. And the shield is a single casting of `aluminium_6061()`,
whose conductivity is 167 W/m·K and is *stated*, in the one place a material
is stated. The same message came back for `youngs_modulus` from the elastic
study.

The material field is sampled by walking the CSG tree, and at a boolean node
the two children's materials are lerped by a weight that comes from the
distance field, so that a smooth CSG interface is a smooth (and
differentiable) property interface. The lerp was `Material.blend`, which is

```python
b * (1.0 - t) + a * t
```

An unspecified property is `nan`. IEEE 754 says `nan * 0.0 == nan`. So a
child that specifies *nothing* poisons the result even where its weight is
**exactly zero** — at the centre of the flange, a metre of nothing away from
the hole in question.

And every cut tool is such a child. A `Face.hole` is geometry, not a
substance; it has no conductivity because the question is meaningless. Of the
eighteen tools in this part's top-level `Difference`, the ones that carry a
material carry it only for the render. The consequence is not subtle and not
local: **any solid with a single hole in it reported `nan` for every physical
property at every interior point**, and no `FROM_MATERIAL` study over it could
be assembled. This is not specific to the shield. `scenes/end_cap.py` has
holes too; it survives because it happens to hand its studies explicit
scalars.

It is worth being clear about how invisible this was. The *render* was
perfect: `color` is specified by every material, so the one property that
reaches a pixel was the one property that never went `nan`. The shield looked
right in the viewer for as long as it took to ask it a physical question.

**Fix.** `blend_materials` in `cadjoint/sdf/boolean/base.py`, shared by all
three boolean nodes, adds one rule to the lerp: a property is unspecified only
when *neither* side specifies it.

```python
lerp = b * (1.0 - weight) + a * weight
return jnp.where(jnp.isnan(a), b, jnp.where(jnp.isnan(b), a, lerp))
```

Two specified values still blend exactly as before — the smooth property
interface is untouched, and it is still differentiable, because `jnp.where` on
a `nan` *constant* is a select on a concrete predicate, not a branch on a
traced value. A property neither side specifies stays `nan`, which is what
lets the study's error message keep working when it is telling the truth.

`Xor.material_at` had the same lerp at a *fixed* weight of 0.5, where the
erasure is unconditional; it uses `blend_materials` now too.

**And the fix had to be paid for twice.** The single `jnp.where` above gives
the right value and a `nan` **gradient**. The VJP of `jnp.where` multiplies
the unselected branch's cotangent by zero, `lerp`'s derivative in the blend
weight is `a - b`, and `0.0 * nan` is `nan` — the double-`where` trap, which
is a trap precisely because the primal is correct and nothing looks wrong
until an optimiser stops five minutes in with

```
ValueError: Optimization 'stiff-shield' left the finite range at step 0
(objective=198.96150236622026, grad_norm=nan); lower the learning rate or
rescale the objective.
```

— advice that would not have helped, since the objective was finite and the
learning rate was irrelevant. That is a *worse* failure than the one being
fixed: a `nan` value is checkable at the point you compute it, and a `nan`
gradient surfaces somewhere else entirely, wearing the mask of a badly scaled
problem. Both operands are made finite *before* the lerp now, and the second
`where` puts the `nan` back over an expression that never carried one.

Tests in `tests/sdf/boolean/test_boolean.py::TestMaterialThroughBooleans`:
a drilled plate keeps its conductivity at a point clear of the hole; a union,
an intersection and an xor with a material-less child keep theirs; two
specified values still blend to something strictly between them; a property
neither side specifies is still reported as unspecified; the blend is
differentiable where one side is unspecified; and — the guard against
over-fixing — a genuine two-material blend's gradient still matches central
finite differences to 2 %.

### 2.7 FIXED — a `Difference` took its material from the first tool and ignored the other seventeen

**What happened.** Found while fixing §2.6, in the same three lines.
`Union.material_at` folds over every operand:

```python
for child in self.sdfs[1:]:
    ...
```

`Difference.material_at` and `Intersection.material_at` did not. They read
`self.sdfs[0]` and `self.sdfs[1]`, blended those two, and returned — even
though both nodes are variadic and this file's `Difference` has **a body and
eighteen tools**. So the material on seventeen of the eighteen cut walls was
whatever the *bore* said, and the bore is the first tool only because it is
written first. Reorder the argument list and the part's material field
changes.

The same held for `Intersection`, where the shroud is built from more than two
operands.

Alone, this bug is milder than §2.6 — with the `nan` rule in place the body's
value survives the blend anyway, so the field is right almost everywhere. It
bites exactly where the two rules interact: a tool that *does* carry a
material (the shield has several, for the render) tints only the wall it cut
if the fold reaches it, and before the fix, seventeen tools could not tint
anything.

**Fix.** Fold, the way `Union` does, carrying the running distance so that
each tool's blend weight is measured against the shape as cut *so far*:

```python
for tool in self.sdfs[1:]:
    d = tool(p)
    t = jnp.clip(0.5 + 0.5 * (result_d + d) / k, 0.0, 1.0)
    result_m = blend_materials(result_m, tool.material_at(p), t)
    result_d = jnp.maximum(result_d, -d)
```

Test: `test_every_tool_is_folded_in_not_just_the_first` — a steel plate cut by
two spheres, the second of which is copper, asserts the copper reaches the
wall it cut. Before the fix that point read 52 W/m·K (steel); after it reads
above 100.

**Why the scene tests did not catch either.** They asserted *geometry* —
where the metal is — which is what §4.1 says is the only reliable evidence on
a part this size. Neither reads the material field, because until the studies
were run there was no reason to think a boolean touched it. The lesson is in
the same direction as §4.1 and one step further: a scene test that means to
stand in for the simulation has to assert the *field the simulation reads*,
not only the surface.

### 2.8 FIXED — a study that reads the scene's materials could not be optimised

**What happened.** With §2.6 and §2.7 in, both studies solved. The
optimisation over the elastic one still did not, and it failed twice, in two
different places, for the same underlying reason: `Optimization` calls a study
along two paths and **neither of them hands it the scene**.

*During* the loop it builds the mesh once, to freeze the topology so the
gradient is a gradient of the physics and not of the mesher, and then calls

```python
result = study.solve(mesh=mesh, points=points)
```

`_solve_mesh` sees an already-extracted `HexMesh`, returns "no SimMesh
attached", and `_material_source` had exactly two rungs: the `sdf` argument
(absent) and the built SimMesh's domain (absent). So:

```
ValueError: Study 'belt-pull' derives 'youngs_modulus' from the scene's materials
but got no SDF to sample: pass the scene to solve(sdf), give the study's SimMesh a
domain=, or set an explicit youngs_modulus value on the study.
```

The scene had *taken that advice already* — `shield_mesh` is declared
`domain=shield`, and it is the same object `study.mesh` points at. The error
message was telling the author to do the thing they had done, because the
lookup never asked the study what mesh it declared, only what mesh it was
handed.

*After* the loop it writes the final parameters back with `apply_parameters`,
re-extracts, and re-solves with the functionalized field:

```python
result = study.solve(final_field)
```

`final_field` is a bare closure. It has no `material_at`, so the sampler
refused it — 300 seconds into a run, after every gradient step had already
succeeded.

**Fix.** `_material_source` becomes a ladder, and skips a candidate that
cannot answer `material_at`: the `solve` argument, then the built SimMesh's
domain, then **the study's declared SimMesh's domain**, then the study's own
`domain=`. If nothing carries materials, whatever was handed in is returned
unchanged, so the sampler still raises the error that names what it got.

The second rung of that ladder is not a mere tolerance. By the time
`solve(final_field)` runs, the optimiser has already written the final
parameters onto the scene object, so the declared domain *is* the geometry the
callable describes — and it also knows what it is made of, which the callable
never will.

`cadjoint/fem/study.py`, tests in
`tests/fem/test_material_properties.py::TestTheGeometryAStudyFallsBackOn`
(both optimiser call shapes, that an explicit SDF still wins over the
fallbacks, and that a study naming no geometry at all still raises), and
`tests/scenes/test_motor_shield.py::TestTheFieldTheSimulationReads`.

### 2.9 DEFERRED — there is no sweep along a curve, and `twist` is the nearest thing to one

**What the part needs.** A cast coolant jacket is a tube of circular section
following a helix around the bearing tower. That is a sweep: a profile carried
along a 3D path, kept normal to it.

**What the language has.** `extrude(profile, depth, twist=…)`, which rotates
the *query point* about the extrusion's local z by `twist · z / depth`. A
circle drawn 0.75 off the axis and extruded with a full turn of twist does
sweep a helix of radius 0.75, and that is what the part uses:

```python
channel_profile = PolygonProfile.circle(radius=0.11, center=(0.75, 0.0), segments=10, …)
channel = extrude(channel_profile, depth=0.44, twist=channel_turns.value * 360.0)
```

Four things are wrong with it and all four had to be designed around.

1. **The section is normal to z, not to the path.** The tube's cross-section
   in the plane perpendicular to the helix is an *ellipse* — stretched by
   `1/cos(λ)` where λ is the helix angle. Here λ = atan(0.44 / (2π·0.75)) =
   5.3°, so the tube is 0.4 % out of round and nobody would notice. Double the
   turns and it is 1.7 % out; make it a real screw and it stops being a tube.
2. **The pitch is `depth / turns` and cannot be named.** There is no `pitch`
   to hand a `Scalar`, so `channel_turns` drives the twist through
   `channel_turns.value * 360.0` — a Python float read out of the parameter,
   not a live link. Change the turns and the helix changes; *differentiate*
   with respect to the turns and nothing happens. The one parameter a coolant
   designer actually tunes is the one that does not reach the gradient.
3. **The field is not 1-Lipschitz.** Rotating the query point stretches
   distance by up to `1 + |twist| · r / depth`; at r = 0.86 and 360°/0.44 that
   is a factor of ~12 near the outer wall. Sphere-tracing under-steps, dual
   contouring's edge bisection needs more iterations, and every consumer that
   assumes a metric field is on notice. It is documented in `extrude`; it is
   still the reason the mesh times in §5 are what they are.
4. **A twisted extrusion declares no faces** (see §2.10).

**The nearest honest expression** — and what this file settles for — is a
twisted extrusion whose end discs are placed where the tangency argument
needs them: the twist is zero at mid-depth and ±180° at the caps, so *both*
end discs land at 180°, on the −x side, and the feed bores can be plain
cylinders along ±y that are tangent to the helix's circle at exactly those
points. That is a real cast jacket's geometry, arrived at backwards from the
one primitive available.

**Proposed design.** `sweep(profile, path, *, twist=0.0, scale=1.0)` where
`path` is a `Curve` node (initially `Helix(radius, pitch, turns)`, `Arc`,
`Polyline`) with a differentiable `point(t)`, `tangent(t)` and an
approximate inverse `nearest_t(p)`. The SDF evaluates
`profile_distance(to_frame(p, nearest_t(p)))` in the path's Frenet (or
rotation-minimising) frame. `nearest_t` is the hard part and is the same
Newton projection `SketchPlane.tangent` already runs; for a helix it has a
closed-form initial guess (`atan2(y, x)` unwrapped by z). The payoff is a
named `pitch`, a section normal to the path, and a field that is 1-Lipschitz
wherever the path's curvature radius exceeds the profile's.

### 2.10 DEFERRED — a drafted or twisted extrusion declares no faces, so it cannot be built on

**What happened.** A sand casting's walls have to leave the mould, so the
drain-plug boss is drafted: `extrude(drain_profile, depth=0.18,
draft=drain_draft)`. The natural next line is
`drain_boss.cap("+").hole(0.06, depth=…)` — and there is no `cap("+")`.
`extrusion_faces` returns an empty `FaceSet` for any non-zero draft or twist,
because the walls are no longer half-planes and the caps are no longer the
profile. That is *correct* for the walls, and needlessly strict for the caps:
a drafted extrusion's two caps are still exactly planar, still at
`±depth/2`, and still the polygon (bottom) or the polygon shrunk by
`tan(draft)·depth` (top). Only their *outline* changes.

The consequence in the file is that the drain bore is a free-standing
`Solid.cylinder` positioned by hand at `(_drain_center[0],
_drain_center[1], 0.19)`, with a magic `0.19` and a magic half-height, rather
than a `Face.hole` that would track the boss. The same happens on every
twisted extrusion — the fan blades, the thread, both halves of the knurl, the
helix — so nothing can be built on any of them.

**Proposed design.** Split `extrusion_faces` in two: emit the two cap faces
whenever the *caps* are planar (always, for draft and twist alike, with the
top cap's polygon scaled by the draft and rotated by half the twist), and emit
the side faces only when they are half-planes (undrafted, untwisted). The
`FaceSet` is already keyed, so `solid.cap("+")` would work and
`solid.side(i)` would keep raising. A drafted boss could then carry a hole,
which is the single most common thing a cast boss does.

### 2.11 DEFERRED — nothing checks whether two features collide

**What happened, three times.** The flange perimeter is a schedule of angular
stations, and every occupant is placed by a number typed in this file: eight
rib stations, four stiffeners, four tie bolts, four shroud legs, four
counterbored bolts, two risers with two flanges and four screws, an encoder
pocket, a cable gland, a drain boss. Nothing in the language relates them.
The first schedule drilled a tie bolt through a gusset. The second put the
encoder pocket under a stiffener. The third — the one that survived into the
draft — ran the coolant gallery through two ribs, and the *only* reason it was
caught is that a test asserted metal where the gallery is.

None of the three produced an error, a warning, or a visible artefact at
viewport resolution. A hole through a rib root looks exactly like a rib.

**Proposed design.** A cheap, opt-in `interference(a, b, grid)` returning the
overlap volume of two subtrees, sampled on the viewport grid and reported by
the compile worker as a warning list — not a hard error, because half the
overlaps in this file are *deliberate* (every union is one). The useful form
is narrower: **`Difference` should be able to say which of its tools cut which
of its body's named sub-solids**, since a tool that removes material from a
solid the author did not name is nearly always a mistake. That is one extra
pass over the tree with the tools already compiled, and it is the check that
would have caught all three.

### 2.12 DEFERRED — a cylinder's `height` is a half-height and an extrusion's `depth` is not

`Solid.cylinder(height=0.5)` is 1.0 tall. `Solid.box(size=[a, b, c])` is
`2a × 2b × 2c`. `extrude(profile, depth=0.5)` is 0.5 deep, straddling its
plane. Three placement conventions, three different meanings of the size
argument, and the SDF-primitive convention (half-extents) leaking into the
construction layer where the sketch convention (totals) already lives.

The first draft of this file had every bore twice as long as it meant. The
gland bore reached through the flange and into the bearing tower; the drain
bore came out of the underside of the flange; the riser bores went past the
gallery. Every one of them was found by looking at a screenshot, not by an
error.

**Proposed design.** Deprecate `height` and `size` on `Solid.cylinder`/`box`
in favour of `length`/`extent` meaning the total, keeping the old names
working for one release with a warning. The SDF primitives keep half-extents;
`ConstructionPrimitive` is the layer that should speak the author's units, and
it is already the layer that translates.

### 2.13 DEFERRED — `Difference` blends by default, at a radius that is part-scale

`Union`, `Difference` and `Intersection` all default to `smoothness=0.1`.
On a part whose viewport cell is 0.094 and whose thinnest wall is 0.12, a 0.1
blend on a *cut* is not a fillet, it is a third of the wall. The locknut's
bore had to be written `smoothness=0.0` explicitly, and so did the shroud's
window cuts and the counterbore's `Union`; every one of the eighteen cuts in
the shield is either explicit or deliberate.

A default that is wrong at part scale is worse than no default, because it is
invisible: the part just comes out slightly rounder and slightly lighter than
it was drawn. **Proposed:** default `smoothness=0.0` on `Difference` and
`Intersection` (a cut is sharp unless you ask), keep the blend default on
`Union` where it is at least a fillet, and make both scale-relative — a blend
given as a fraction of the child's bounding-box diagonal rather than in world
units — so that a default cannot be part-scale by accident.

### 2.14 DEFERRED — a circle is a polygon, and everything downstream believes it

`PolygonProfile.circle(radius=1.0, segments=28)` is a 28-gon inscribed in the
circle. Three things in this part depend on it and two of them are wrong by a
measurable amount:

* the bearing tower's wall is between 0.9937 and 1.0 from the axis — an
  **0.63 % apothem error**, five times the tolerance a bearing seat is
  machined to;
* `SketchPlane.tangent` on that wall lands on a *facet*, with the facet's
  normal, up to **6.4° off radial** (`360°/28/2`); the nameplate pad is
  therefore not quite tangent to the cylinder it looks tangent to, and the
  test had to be written to allow it;
* the risers' external tangency (§`cyl_tangent`) is tangency to a *true*
  cylinder — `Solid.cylinder` is analytic — against a 28-gon tower, so the
  "tangent line" is really a tangent line to a facet, offset inward by
  0.0063.

None of this is a bug; it is the language being honest about having one 2D
primitive. It is recorded because it sets a floor on every geometric claim
this part makes, and because the fix is cheap: **an `Arc`/`Circle` profile
segment** whose SDF is exact (`|xy - c| - r` clipped to the arc's wedge),
alongside the polygon segments, is the same change §2.9 needs for its paths.

### 2.15 DEFERRED — `SketchPlane.origin` is a parameter, `Face.origin` is an array

`float(face.origin[2])` works. `float(plane.origin[2])` raises
`TypeError: float() argument must be … not 'Vector'`, because a plane hands
back its live `Vector` parameter and a face hands back a plain array. Both are
"the origin of a plane". The test in `tests/scenes/test_motor_shield.py` says
`plane.origin.xyz` and carries a comment explaining why.

**Proposed:** give `Face` the same `.origin`/`.normal` parameter objects, or
give `Vector` an `__array__` and `__float__`-able indexing. The second is one
method and fixes every caller.

### 2.16 DEFERRED — `Solid.box`'s size is one parameter, so one edge cannot share a `Scalar`

The corner lugs must be exactly as thick as the flange, because their point is
that their caps are *coplanar* with the flange's (the `boxes_coplanar` axiom
case). `Solid.box(size=…)` takes one `Vector`, and one component of a `Vector`
cannot be bound to the shared `flange_thickness` `Scalar`. So a lug is a
four-vertex `PolygonProfile` with four constraints, extruded from the flange's
own sketch plane to `depth=flange_thickness` — which is better modelling
anyway, but it was forced, not chosen.

**Proposed:** let `Vector` accept a per-component `Scalar`
(`Vector([a, b, thickness])`), the way `rotation` now accepts per-angle
`Scalar`s after §2.4. The machinery is the same and §2.4 is the precedent.

### 2.17 DEFERRED — an optimisation has no constraints, only a regularizer

Both manufacturing limits in this part — the casting must not grow past a mass
budget, and the wall between the bearing seat and the coolant channel must not
fall below 0.12 — are *constraints*, and `Optimization` takes a
`regularizer`/`regularizer_weight` pair. So they are softplus hinges with
hand-tuned scales:

```python
over_budget = 20.0 * jax.nn.softplus((shield_volume(parameters) - volume_cap) / 0.02)
thin_wall = jax.nn.softplus((seat_radius - (0.64 - min_wall)) / 0.005)
```

The three magic numbers (20.0, 0.02, 0.005) exist only to make a penalty
behave like a constraint, they interact with `learning_rate`, and a violated
constraint is reported as a slightly higher objective rather than as a
violated constraint.

**Proposed:** `Optimization(constraints=[LessThan(shield_volume, 4.40),
GreaterThan(seat_wall_thickness, 0.12)])`, implemented as an augmented
Lagrangian over the same gradient — the multipliers replace the hand-tuned
scales, and the report can then say *which* constraint bound.

### 2.18 DEFERRED — `Shell` hollows every face, so an open end needs a cut

The shroud is a thin conical wall around the fan. `Shell(loft(...),
thickness=0.20)` produces `|f| - t/2`, which is a closed shell: it walls the
two ends as well as the cone. Opening it takes a tall cylinder differenced
away — a tool whose radius (1.0) and height (0.6) have to be kept larger than
the shroud and are unrelated to it, so a change to the shroud silently leaves
a lid on or eats the wall.

**Proposed:** `Shell(solid, thickness, open=[face, …])` taking the faces to
leave off. `loft` already declares its two caps as `solid.cap("±")`, so
`Shell(shroud_blank, 0.20, open=[shroud_blank.cap("+"), shroud_blank.cap("-")])`
would be exact and would track the loft.

### 2.19 DEFERRED — a parameter cannot appear in an expression, so every derived dimension is dead

**This is the largest gap the part found.** `Scalar` has no arithmetic:

```python
>>> Scalar(1.0, name="turns") * 360.0
TypeError: unsupported operand type(s) for *: 'Scalar' and 'float'
```

Any dimension that is a *function* of another dimension therefore has to be
computed in Python, off `.value`, which severs both the live link and the
gradient. Six places in this one file:

| written | what it means | what it costs |
| --- | --- | --- |
| `twist=channel_turns.value * 360.0` | one turn of helix | `channel_turns` cannot be optimised |
| `plane(offset=tower_height.value / 2.0)` | the tower's own mid-plane | the seal boss stops tracking the tower's height |
| `_riser_y = sqrt(1.16**2 - 0.75**2)` | riser axis at `tower_r + riser_r` | the external tangency is a coincidence of three literals |
| `SketchPlane(origin=[0, 1.06, 0])` | tower radius + half `rib_thickness` | the tangency holds only while `rib_thickness == 0.12` |
| `seat_radius = bore_radius + 0.15` | the seat is 0.15 wider than the bore | fine — because it is written *inside* the jitted penalty, over the parameter *dict* |
| `at=(1.12 cos θ, 1.12 sin θ)` | a station on a bolt circle | the bolt circle radius is not a parameter at all |

The last row of that table is the tell: the arithmetic *does* work, and stays
differentiable, as soon as you are inside a traced function over the free
parameter dictionary (`shield_volume`, `manufacturing_penalty`). It is only
the *construction* layer, where the author writes, that cannot do it. So the
part is fully differentiable in the eleven parameters that happen to be handed
directly to a feature, and structurally rigid in every relation between them —
which is the opposite of what a parametric CAD file is for.

The constraint system covers exactly one corner of this: relations *between
sketch vertices in one profile* (`DistanceConstraint`, `PointOnLine`, …). It
cannot relate a plane's origin to a cylinder's radius, and those are the
relations a machine part is made of.

**Proposed design.** Make `Scalar` (and `Vector`) build a small expression
graph under `+ - * / **`, `min`, `max` and the trig functions, producing a
`Derived` parameter that is (a) not free, (b) evaluated lazily from its
operands, and (c) traced through `functionalize` like any other node — the
machinery is `jax`, and the expression is already a jax expression; what is
missing is that `Parameter` is a `dataclass` holding an array rather than a
node. `extract_parameters` would walk a `Derived` to its free leaves and
report the leaves; `apply_parameters` would refuse to write to a `Derived`.
Two lines of the six above (`_riser_y`, the stiffener plane) would then state
the tangency they only currently imply, and `channel_turns` would reach the
optimiser.

### 2.20 DEFERRED — the lowered material field is a second, disagreeing implementation

**What happened.** §2.6 and §2.7 fixed `material_at` on the SDF *objects*.
There is a second copy of that fold, in `cadjoint/functionalize.py`
(`functionalize_scene`'s `mat_eval` for a `BooleanOp`), and it was not
touched — because it is the one the compiled render path uses, which another
line of work owns this week. Reading the two side by side, it disagrees with
the objects in three ways at once:

* it uses `Material.blend`, so it has §2.6's `nan` erasure and §2.6's `nan`
  gradient, latent;
* it applies **Union's** blend weight — `t = clip(0.5 + 0.5·(d − d₀)/k)` — to
  every boolean node, `Difference` and `Intersection` included;
* it advances its running distance with `smooth_min` for all of them, so
  after the first operand of a `Difference` the weights are measured against
  the wrong shape.

Only the last two are reachable today: the lowered `mat_eval` returns the six
*render* properties (`color`, `roughness`, `metallic`, `opacity`, `ior`,
`reflectivity`) and defaults each of them at the leaf, so no `nan` ever
enters it. The visible consequence is mild and one-directional — inside a
`Difference`, `d − d₀` is positive almost everywhere, so `t` saturates at 1
and the body's material simply wins: a tool that carries a material to tint
the wall it cut never tints it in the compiled render, while it does in the
interpreted one. Nothing in this part looks wrong because of it. It is
recorded because *two implementations of one rule* is the interesting fact,
not the size of today's discrepancy.

**Proposed design.** One fold, in `cadjoint/sdf/boolean/base.py`, over
already-evaluated `(distance, material)` pairs:

```python
def fold_materials(kind, first, rest, smoothness): ...
```

`Union.material_at`, `Difference.material_at`, `Intersection.material_at`,
`Xor.material_at` and `functionalize`'s `mat_eval` all call it, the last one
passing the node's own `kind`, which it already has in hand as `obj`. Then a
test can assert the two agree on the same tree — which is the test that does
not exist and is the reason this drifted.


---

## 3. Program size and compile time

All figures on this branch, `JAX_PLATFORMS` default, one core busy. "Cold"
means an empty `CADJOINT_CACHE_DIR`; "warm" means the same request run twice.
The end-cap column is from `research/complex-scene.md` where it was measured
the same way.

| | end-cap | shield |
| --- | ---: | ---: |
| module exec, incl. constraint solve | 1.9 s | **4.5 s** |
| free parameters (whole scene) | — | 41 |
| fixed parameters (whole scene) | — | 1 192 |
| world-frame CSG leaves | 17 | **52** |
| profile vertices, named profiles | 122 | **259** |
| construction nodes in the viewer payload | 12 | **40** |
| compile payload | — | **23.9 MB** |

The SDF program, lowered by `functionalize` and measured with
`jax.jit(...).lower(...)`:

| program | HLO | ops | lower | compile |
| --- | ---: | ---: | ---: | ---: |
| one point query | 622 KB | 8 223 | 0.44 s | 0.68 s |
| `vmap` over 4 096 points | 791 KB | 9 598 | 0.56 s | 0.96 s |
| `grad` over the free set | 1 313 KB | 15 756 | 1.20 s | 3.49 s |

Three things are worth saying about that table.

**The batched program is barely larger than the single-point one** (+27 % HLO,
+17 % ops). `vmap` costs almost nothing structurally, which is the whole
argument for the functional lowering — the pattern nodes trace their instances
*once* over a batch axis rather than unrolling, so a 7-blade fan and an
8-station rib ring are each one copy of their geometry plus a batched rest.

**The gradient is 2.1× the forward program and 5× its compile time.** Every
edit round-trip that touches the optimiser pays that 3.5 s.

**Nothing here is the bottleneck.** The whole SDF program compiles in about
5 s. What the viewer waits for is elsewhere: the shader and the mesher.

### 3.1 Where the time actually goes

Measured twice, six hours apart on the same scene, because the WGSL backend
was being changed underneath it by another line of work (§3.2). Both columns
are given; the second is the state this report was finished in.

| stage | cold (early) | cold (final) | warm (final) | budget (`_worker_client.py`) |
| --- | ---: | ---: | ---: | ---: |
| `compile` worker | 27.0 s | **23.3 s** | 16.2 s | 90 s ✓ |
| `mesh` worker (edge overlay, 64³) | 149.0 s | **129.2 s** ✗ | 49.5 s ✓ | 90 s |
| compile payload | 29.1 MB | **23.9 MB** | | |

**The cold edge overlay blows the mesh budget by 44 %.** A first `mesh`
request on this part, on a machine whose XLA cache does not already hold it,
is killed at 90 s and reported to the author as *"Meshing exceeded the
90-second timeout"* — with no partial result and no indication that a second
attempt would succeed.

The playground's `warm_start()` does not rescue it either, quite: it issues a
background `compile` and a background `mesh` for **every** scene under
`scenes/`, and it holds them to `MESH_TIMEOUT_SECONDS` too. So the warm-up's
own mesh of this part is killed at 90 s as well. What saves the interactive
path is that XLA's persistent cache is written *per executable as each one
finishes*, so the killed warm-up still leaves most of the cache behind and the
user's first real request lands on the 49.5 s warm path. The part is usable in
the app (§6) by that accident rather than by design, and on a machine that has
never opened it, a direct `mesh` request fails.

The overlay itself: **16 261 wire segments, 3 777 sharp** at resolution 64.
The end-cap's was 8 000-odd. Almost exactly a doubling of profile vertices
(122 → 259) for almost exactly a doubling of overlay work, which matches the
pricing model in `research/complex-scene.md` §3: the overlay is priced in
profile vertices, not in features.

### 3.2 The shader

`compile_scene_to_wgsl` emitted **5.78 MB of WGSL, 101 721 `let` bindings, 69
functions, in 42.2 s**.

That number is *not* attributable to this scene alone and should not be read
as one: the WGSL backend (`cadjoint/backends/wgsl/`, `cadjoint/sdf/_lowering.py`)
has substantial uncommitted work in this tree from another line of work,
including a new culling pass. The same script on an earlier draft of this
scene, earlier the same day and before those edits, reported 6.8 MB / 69 100
lets / 6.5 s. The scene grew by roughly a third between those drafts; the
shader emission time grew by 6.5×. Whoever owns that backend should measure
it against a fixed scene; from here all that can be said is that **shader
emission, not the SDF program, is the largest single item in a cold compile**,
and that at ~100 k `let`s the emitter is well into the range where a browser's
WGSL compiler becomes the next wall.

---

## 4. The limitation table

| # | limitation | status | where |
| --- | --- | --- | --- |
| 2.1 | a pattern could not leave a station empty | **fixed** — `skip=` | `cadjoint/sdf/operations.py` |
| 2.2 | `SketchPlane.tangent` read an inward, non-unit normal | **fixed** — central difference | `cadjoint/construction/sketch.py` |
| 2.3 | no node selection for a bore, seat wall or jacket | **fixed** — `Nodes.cylinder` | `cadjoint/fem/selection.py` |
| 2.4 | an inclined tool's three angles were always free | **fixed** — `Scalar` rotations | `cadjoint/construction/solid.py` |
| 2.5 | two unnamed primitives collided on parameter names | **fixed** — per-kind counters | `solid.py`, `sketch.py` |
| 2.6 | a hole erased the material of the metal around it — and masking the `nan` cost a `nan` gradient | **fixed** — `blend_materials`, as a double `where` | `cadjoint/sdf/boolean/base.py`, `xor.py` |
| 2.7 | `Difference`/`Intersection` blended only their first two operands | **fixed** — fold every operand | `sdf/boolean/{difference,intersection}.py` |
| 2.8 | a `FROM_MATERIAL` study could not be optimised | **fixed** — a material-source ladder | `cadjoint/fem/study.py` |
| 2.9 | no sweep along a curve; `twist` is the nearest thing | deferred — `sweep(profile, path)` | — |
| 2.10 | a drafted/twisted extrusion declares *no* faces | deferred — emit the caps | `construction/faces.py` |
| 2.11 | nothing checks whether two features collide | deferred — per-tool cut report | `sdf/boolean/difference.py` |
| 2.12 | `cylinder(height=)` is a half-height, `extrude(depth=)` is not | deferred — rename to totals | `construction/solid.py` |
| 2.13 | `Difference` blends by default at part scale | deferred — sharp by default, relative radii | `sdf/boolean/` |
| 2.14 | a circle is a polygon (0.63 % apothem, 6.4° facet normals) | deferred — `Arc` profile segments | `construction/sketch.py` |
| 2.15 | `SketchPlane.origin` is a `Vector`, `Face.origin` is an array | deferred — `Vector.__array__` | `geometry/parameters.py` |
| 2.16 | `box(size=)` is one `Vector`, so an edge cannot share a `Scalar` | deferred — per-component `Scalar` | `geometry/parameters.py` |
| 2.17 | an optimisation has no constraints, only a regularizer | deferred — augmented Lagrangian | `cadjoint/optimize.py` |
| 2.18 | `Shell` hollows every face; an open end needs a cut tool | deferred — `Shell(open=[face])` | `sdf/operations.py` |
| 2.19 | **a parameter cannot appear in an expression** | deferred — `Derived` parameters | `geometry/parameters.py` |
| 2.20 | the lowered material field is a second, disagreeing fold | deferred — one `fold_materials` | `cadjoint/functionalize.py` |

### 4.2 What the *pipeline* could not do

The table above is the modelling language. These are the things downstream of
it that stopped keeping up on this part; each is measured in the section
named, and none has a fix in this change.

| # | limitation | where | section |
| --- | --- | --- | --- |
| P1 | neither tet route can mesh the part — the refinement ladder walks all three rungs and the surface is still self-intersecting at 4.5× the cells | `cadjoint/fem/tetmesh.py` | §5.2 |
| P2 | the Gmsh route reports a raw Gmsh sentence and walks no ladder | `cadjoint/fem/gmsh.py` | §5.2 |
| P3 | the hex route silently includes geometry outside the declared box; the tet route warns | `cadjoint/fem/{hexmesh,tetmesh}.py` | §5.2 |
| P4 | no quality floor: a scaled Jacobian of 0.0079 is handed to both solvers | `cadjoint/fem/simmesh.py` | §5.1 |
| P5 | B-rep extraction takes 156 s, reports χ = 60 against a true −54, and gives one octagonal flange 142 faces — *measured before the module left this repo* | `cadjoint/brep/` (removed) | §6 |
| P6 | mass and safety factor are `None` on a single-alloy casting, vetoed by a structural check that predates §2.6 | `cadjoint/fem/properties.py` | §7.1 |
| P7 | the optimiser's re-meshed objective is 16 % worse than its last frozen-mesh one, and is not reported next to it | `cadjoint/optimize.py` | §8 |
| P8 | a parameter with an exactly-zero gradient is indistinguishable from one with a small one | `cadjoint/optimize.py` | §8 |
| P9 | the cold edge overlay is 129 s against a 90 s budget; the app is usable only because a killed warm-up still leaves its XLA cache behind | `cadjoint/viewer/_worker_client.py` | §3.1, §9 |
| P10 | the compile budget is wall clock, so a job using 55 CPU-seconds of 90 is killed as "too slow" | `cadjoint/viewer/_worker_client.py` | §9.1 |
| P11 | the construction overlay draws 259 vertices across 40 features at once and is unreadable | `frontend/src/components/viewer/` | §9.2 |
| P12 | `Nodes.cylinder` has no `describeSelection`, no `selectionEval` and no `types.ts` variant, so the BCs it selects render blank | `frontend/src/{studies,selectionEval,types}.ts` | §9.3 |

### 4.1 Which of these were *silent*

The task asked for three separate things: what the language could not express,
what it expressed awkwardly, and what it silently got wrong. The third list is
the one that matters, because the first two announce themselves.

| silently wrong | how it showed up | how it was caught |
| --- | --- | --- |
| `SketchPlane.tangent`'s inward normal (2.2) | the pad still straddled the wall, so it *looked* right; only its `u` axis was mirrored | a test that asserted the normal pointed out |
| a hole erasing the metal's material (2.6) | the render was **perfect** — `color` is the one property nothing leaves unspecified | asking the part a physical question: both studies refused, on 3 748 of 3 748 elements |
| seventeen of eighteen cut walls taking the first tool's material (2.7) | nothing at all, while §2.6 stood; a reorderable material field | reading `Difference.material_at` next to `Union.material_at` |
| the `nan`-masking fix's own `nan` *gradient* (2.6) | the field was right and `grad_norm` was `nan`, 300 s into an optimisation | asking for the gradient of a drilled plate's conductivity |
| a bore twice as long as written (2.12) | the gland bore reached into the bearing tower | looking at a screenshot |
| `Difference`'s 0.1 default blend (2.13) | the locknut came out rounder and lighter than drawn | reading the wall thickness off the sketch |
| the coolant gallery through two ribs (2.11) | a rib with a 0.22 hole in its root still looks like a rib at 64³ | a test that asserted metal at every rib station |
| `channel_turns.value * 360.0` (2.19) | the optimiser reports **gradient zero**, not an error | asking for the gradient and getting 0 |
| a 28-gon "cylinder" (2.14) | 0.63 % apothem error, facet normals 6.4° off | a test written against the true radius, which failed |

Four of the six were caught by a *test that asserted geometry*, not by the
viewer and not by an exception. That is the argument for
`tests/scenes/test_motor_shield.py` being as long as it is: on a part this
size, the render is not evidence.

---

## 5. Meshing

Both `SimMesh`es are declared over the shield alone, on the same box
(3.20 × 3.20 × 1.45 from (−1.60, −1.60, −0.15)) at the same declared
resolution (30, 30, 14) — so the lattice spacing is **0.1067 × 0.1067 ×
0.1036**, and the sampling cell is a little *coarser* than the 0.094 the
viewport overlay uses.

That number is the whole story of this section, because the part's thin
dimensions are stated in viewport cells and the lattice is coarser than they
are:

| feature | thickness | lattice cells |
| --- | ---: | ---: |
| gusset rib / tangential stiffener | 0.12 | 1.13 |
| coolant tube bore | 0.22 | 2.06 |
| wall outside the coolant tube | 0.14 | 1.31 |
| wall inside it, to the circlip groove | 0.12 | 1.13 |
| locating spigot wall | 0.10 | 0.94 |
| riser pipe flange | 0.06 | **0.56** |
| nameplate pad, retainer pad | 0.06 | **0.56** |
| circlip groove depth | 0.05 | **0.47** |

Three features are **below half a cell**. They are in the part on purpose: the
question was what the mesher does with them, and the answer is in the quality
figures.

### 5.1 The hex lattice (`shield-hex`)

`mesh_inspect` took **144.7 s** (budget 300 s, ✓).

```
nodes 5 482   elements 3 748
bounds  [-1.500, -1.500, -0.192] .. [1.500, 1.500, 1.196]
scaled Jacobian   min 0.0079   mean 0.894   max 1.000
aspect ratio      min 1.03     mean 1.54    max 18.74
```

The mean is healthy and the minimum is not: **0.0079 is a degenerate
element**, three orders of magnitude below the mean, and an aspect ratio of
18.7 is a needle. Both come from the sub-cell features in the table above — a
0.06 pad or a 0.05 groove crosses a 0.107 cell, and the dual vertex lands
almost on a cell face.

Nothing rejects it. The mesh is handed to both studies as it is, and a
scaled Jacobian of 0.008 in an elastic solve is a stiffness-matrix condition
number in the billions. A quality *floor* — "refuse, or refine, below 0.05"
— belongs in `SimMesh.build`; the ladder that already exists for tets (§5.2)
is the machinery for it.

### 5.2 The tet lattices — the part's hardest ceiling

**Neither tet route can mesh this part, at any rung of the ladder.** This is
the clearest ceiling the shield found, and both meshers hit it for the same
reason and report it very differently.

```
tet4  / tetgen   FAILED after 142.0 s
tet10 / tetgen   FAILED after 141.9 s
  RuntimeError: TetGen rejected the surface: The input surface mesh contain
  self-intersections. The surface stays self-intersecting up to (68, 68, 32)
  (declared (30, 30, 14), refined x1.5 and x2.25); the part likely has features
  thinner than two cells at the declared resolution — raise the declared
  resolution or use method='hex'.

tet10 / gmsh     FAILED after 19.9 s
  Exception: Wrong topology of triangulation for parametrization:
  one edge is incident to 4 triangles
```

**The refinement ladder fires, all the way, and it is not enough.** The
warning trail shows it walking three rungs — (30, 30, 14) → ×1.5 →
×2.25 = (68, 68, 32) — and trying each one twice, once with exact sharp
feature placement and once without. Six dual-contouring extractions of a
52-leaf part in 142 seconds, and the surface is still non-manifold at 4.5×
the declared cell count.

The cause is in §5's table and is not the mesher's fault: **three features
are below half a lattice cell** (the 0.06 riser flange, the 0.06 nameplate
and retainer pads, the 0.05 circlip groove), and a 0.06 wall crossed by a
0.107 cell yields dual vertices on both sides of the same cell, which is a
self-intersection by construction. The hex route tolerates it — it snaps
cell-centred nodes and produces the scaled Jacobian of 0.0079 in §5.1
instead — and the tet route, which needs a *watertight, manifold* surface
before TetGen will look at it, cannot.

Three separate observations follow, and they are the useful part.

**1. The TetGen error is a model of what an error should be.** It names the
mesher's own complaint, the ladder it walked with the actual resolutions, the
likely cause in the author's vocabulary ("features thinner than two cells"),
and two remedies. Nothing else in this report's failure list is written that
well.

**2. The Gmsh route says nothing usable.** `mesher="gmsh"` fails **seven
times faster** — 19.9 s against 142 s — which is a real advantage when the
answer is "no", but it fails with a raw third-party sentence. "one edge is
incident to 4 triangles" *is* the same fact as "the surface self-intersects",
stated in Gmsh's internal vocabulary, and an author who has never read Gmsh's
source has no way to get from it to "your circlip groove is half a cell
deep". The gmsh route also does **not** walk the refinement ladder: it takes
the declared resolution, hands it over, and reports what comes back. So the
comparison is: TetGen tries hard and explains itself; Gmsh fails fast and
does not.

The fix is small and worth doing: catch the Gmsh exception where TetGen's is
caught, wrap it in the same sentence, and give the Gmsh route the same
ladder — it is per-mesher today for no reason visible from here.

**3. The extraction box is a hair too small, and only one route says so.**

```
UserWarning: The isosurface crosses the extraction boundary on 144 grid edges;
the returned mesh is open.       (144 → 224 → 336 as the ladder refines)
```

The shield's locating spigot reaches z = −0.192; the declared box floor is
z = −0.15. The **hex** route absorbs this silently, because its nodes extend
half a cell past the declared bounds (−0.15 − 0.052 = −0.202) — which is why
§5.1's mesh reports a lower bound of −0.192 and nobody noticed. The **tet**
route warns, correctly, and the count *grows* with refinement because the
cells get smaller while the overhang stays the same size.

Two meshers, one declared box, one part: one reports an open surface and the
other quietly includes geometry outside the box the author drew. Whichever is
right, they should agree, and the author should be told once.

---

## 6. B-rep extraction

> **Read this section as a record, not as a reproducible measurement.** It was
> taken with `cadjoint.brep` in the tree. That module and
> `research/brep-axioms.md` were removed while this report was being written,
> and `from cadjoint.brep import extract_brep` now raises
> `ModuleNotFoundError`. The numbers below are what the extractor did on this
> part on this branch on the day it was still here, and every axiom name cited
> is from `research/brep-axioms.md` as it stood in `9734324`. Nothing in §6
> can be re-run from this tree.

`extract_brep(shield, grid)` on the viewport's own lattice — 64 cells over the
6-unit box, **cell 0.0938** — took **155.8 s** and returned:

```
faces 605   edges 1433   vertices 858   patches 168
face kinds:  plane 295   blend 210   cylinder 58   opaque 42
analytic faces 344   freeform faces 261
quads 4804   blend quads 1791   mesh vertices 4766
non-simple faces 8   ambiguous vertices 690
edge pairs 1365   tangent-or-blend edges 859
χ (open cells) = 60
```

It **completes**, on a part with fifty-two world-frame CSG leaves, and it
completes with a plausible face count. That is the headline and it is a good
one: nothing in `research/brep-axioms.md` promised that the extractor would
survive a part this size at all. What it does *not* do is produce a B-rep
anyone could hand to a CAM system, and the numbers say exactly why.

### 6.1 The Euler characteristic is 60

A solid of genus *g* has χ = 2 − 2g. The shield's real genus is
countable: the through bore, four counterbored bolt holes, four tie bolts,
eight shroud taps, two riser bores, four riser screws, the gland bore, the
drain bore, the grease port, the coolant helix and its two feed bores — call
it 28 handles, χ = −54. The extraction reports **+60**.

That is not a near miss in either direction, and its sign is the informative
part: χ too *high* means faces and vertices that the solid does not have —
the extractor is splitting single faces into several and inventing vertices
where two patches nearly agree, rather than merging things it should not.
`ambiguous_vertices 690` out of 858 says the same thing from the other side:
**80 % of the vertices could not be assigned to a definite set of faces.**

### 6.2 The residuals are infinite, and that is a category of failure, not a number

```
edge residual /cell:    median inf   p90 nan   max inf   (934 of 1433 above the 0.1-cell gate)
vertex residual /cell:  median inf   max inf
edge polyline points:   median 1     max 68    741 singletons
```

A *median* of `inf` means more than half of the edges have no finite
distance to the analytic curve they are supposed to lie on — because there is
no analytic curve: 859 of 1433 edges are classed `tangent_or_blend`, and 741
of them are **single points**, not polylines. An edge that is one point is a
placeholder. The `nan` at p90 is `inf − inf` inside `numpy.percentile`, which
is its own small bug in the reporting path but tells you the same thing.

This is the axiom battery's `fillet_*cell` row happening at scale
(`research/brep-axioms.md` §2.1): a blend produces `blend` faces whose edges
have no closed form, the "edge" degenerates to a point, and χ goes wrong.
The shield has 210 blend faces — **35 % of all its faces** — because
every one of its unions carries a smoothness.

### 6.3 Which axiom failure classes it lands on, and where

The part was built to land on four of them deliberately. All four are
identifiable in the output.

| class | where in the shield | what shows up |
| --- | --- | --- |
| `fillet_{1,2}cell` — a blend wider than a cell | the tower root, `smoothness=0.12` (1.3 cells) | most of the 210 blend faces and the 859 tangent-or-blend edges |
| `fillet_0.2cell` — a blend narrower than a cell | every other union, `smoothness=0.03` (0.32 cells) | invisible to the graph, as the battery predicts: those joins come out as plain plane/plane edges with the blend simply lost |
| `cyl_tangent` — antiparallel normals along a seam | the two cast risers against the tower; the four tangential stiffeners against it | part of the 690 ambiguous vertices; the seam is not recovered as an edge |
| `boxes_coplanar` — a shared plane that should be ONE face | the four corner lugs' caps with the flange's; the riser flange with the riser cap | the `faces per leaf` histogram is the evidence: leaf 1 (the flange) carries **142 faces**, leaf 4 carries 73, leaf 0 carries 66 |

That first column of the histogram is the clearest single result in this
section. The flange is one extruded octagon. It has, honestly counted, about
ten faces. The extractor gives it **142**, because every coplanar join with a
lug, a rib toe, a stiffener heel, a boss or a pad cuts the shared plane into
another piece instead of merging into one face — which is precisely what
`boxes_coplanar` predicts (`12/10` faces on two boxes; here it compounds
across eleven of them).

### 6.4 The face areas say the same thing a third way

```
face area: total 33.340   median 0.00907
           297 faces below 1 cell²   438 faces below 4 cell²
```

The median face is **0.0091 units², about one cell² (0.0088)**. Nearly half
the faces — 297 of 605 — are smaller than a single sampling cell. A B-rep
whose median face is one cell across is not a boundary representation of the
part; it is the isosurface mesh with face labels attached. The 344 analytic
faces (planes and cylinders that *were* matched to a leaf) are the part that
is real, and 261 freeform faces are the part that is not.

### 6.5 What this says about the priority

Nothing here is a surprise given `research/brep-axioms.md`, and that is worth
stating plainly: the axiom battery predicted every one of these failures on
two-primitive cases, and the shield reproduces all of them at once. The
battery is doing its job as the gate. What the shield adds is a **scale**
reading:

* extraction of a 52-leaf part takes **156 s** — 1.7× the whole compile
  budget, and there is no incremental path, so any edit re-derives everything;
* the failure is not graceful. There is no subset of the output that is
  trustworthy on its own, because `analytic` and `blend` faces share vertices
  with each other and 80 % of the vertices are ambiguous;
* the single highest-value fix is not a new algorithm, it is **coplanar face
  merging** — 142 faces on one octagonal flange is a bookkeeping failure, not
  a geometric one, and it is the difference between "605 faces" and something
  near the true count.

---

## 7. The two studies

Both studies solve on the hex lattice of §5.1, take every material property
from `aluminium_6061()` (`FROM_MATERIAL`, no scalar anywhere), and select
their boundary nodes with `Nodes.cylinder` on the part's own axis.

| | wall | result |
| --- | ---: | --- |
| hex mesh build (cached after the first) | 30.6 s | 5 482 nodes, 3 748 elements |
| `bearing-heat` (thermal), first solve | 12.6 s | ΔT 0 → **54.8 K**, mean 12.2 K |
| `bearing-heat`, warm | 9.2 – 10.8 s | identical to 1e-12 |
| `belt-pull` (elastic) | 4.6 – 5.1 s | displacement −7.7 µm … **+37.5 µm** |

**Are they sane?** Two closed-form checks, both to within a factor of ~2,
which is all a first-order check can promise on a shape like this:

*Thermal.* 2.0 × 10⁴ W/m² over the seat wall (an annular band of radius 0.55,
half-length 0.28 → ≈1.9 m²) is ≈3.9 × 10⁴ W, conducted ≈0.2 m through
≈2 m² of 167 W/m·K aluminium: ΔT ≈ QL/kA ≈ **23 K** against a solved
**54.8 K**. The solve is higher, which is the right direction — the real path
is not a straight prism, it is a tower wall interrupted by a helical void.

*Elastic.* 3.0 MPa of traction on the seat, aluminium at 68.9 GPa: strain
≈ σ/E = 4.4 × 10⁻⁵ over a ≈0.5 m lever gives ≈**22 µm** against a solved
**37.5 µm**. Again higher, again for the right reason: the load is a
cantilever moment on a tower, not pure tension.

Both fields are finite everywhere and the thermal minimum is exactly 0.0 on
the Dirichlet set, as it must be. **Neither of these numbers existed before
§2.6**: until the material blend stopped erasing the alloy, both studies
refused to assemble at all.

### 7.1 What the studies cannot report

`result.mass` is `None`, and so is the elastic safety factor, on a part made
of one alloy whose density is 2 700 kg/m³ and whose yield strength is stated.

The reason is a *structural* gate rather than a sampled one.
`maybe_sample_cell_property` — the permissive path used for reporting, as
against the strict `sample_cell_property` used for solving — asks
`specifies_everywhere(sdf, key)` first, which is true only when **every**
material in the subtree specifies the key. The shield's subtree holds 18
materials: `aluminium_6061` and **17 anonymous defaults belonging to the cut
tools**, none of which states a density, because a bore is not a substance.

So this is §2.6's mistake one layer up: the sampled field is now correct
everywhere, and a structural precondition that predates the fix still vetoes
the sample before it is taken. It costs the shield its mass and its safety
factor — and the mass is exactly the quantity §2.17's manufacturing
constraint is a proxy for, so the optimisation is regularising on a
hand-rolled sigmoid volume integral while the study that could have reported
the real mass declines to.

**Deferred** (`cadjoint/fem/properties.py` and `cadjoint/render/material.py`
are not this change's to touch). **Proposed:** `specifies_everywhere` should
ask whether the property is specified *where the material field is defined*,
not on every leaf — the cheap version being to keep the structural check as a
fast path but fall through to a sampled check when it fails, since
`maybe_sample_cell_property` already catches the sampler's `ValueError` and
returns `None`. That is a two-line change and it would restore both
reports.

---

## 8. The optimisation

`stiffen_shield` minimises the elastic study's compliance in the free
parameters of the shield, with a mass budget and a minimum wall as softplus
penalties (§2.17). Adam, learning rate 0.01, six steps, **264.9 s** — about
44 s a step, of which the frozen-topology mesh build is amortised and the
rest is the gradient of the FEM solve through the SDF.

| step | objective | ‖grad‖ |
| ---: | ---: | ---: |
| 0 | 198.96 | 3 698 |
| 1 | 142.62 | 3 280 |
| 2 | 110.10 | 875 |
| 3 | 101.12 | 108 |
| 4 | 100.61 | 34.3 |
| 5 | 100.57 | 40.9 |

**It converges**, and convincingly: the objective falls 49 % and the gradient
norm falls by two orders of magnitude in six steps, which is what a
well-scaled problem looks like. Three things about it are worth recording.

**The re-meshed objective is 16 % worse than the frozen-mesh one.**
`run.objective` — the final design re-extracted on a fresh mesh and re-solved
— is **116.69**, against the 100.57 the last frozen-topology step reported.
The gradient is a gradient of the physics on a fixed lattice, and the
optimiser has spent part of its 49 % moving the design into a place where
that lattice flatters it. On a part whose thinnest walls are one cell across
(§5) that is not surprising, and it is not a bug; it is the price of freezing
the topology, and it should be *reported* rather than left for the reader to
compute. **Proposed:** `OptimizationRun` already carries both numbers — the
run summary should print the re-meshed objective next to the last frozen one
whenever they differ by more than a few percent.

**The constraint that was written held; the one that was not written was
eaten.** `bore_radius` grew 0.320 → **0.3663**, which puts the seat radius at
0.516 against the minimum-wall hinge's limit of 0.64 − 0.12 = **0.52** — the
penalty bound, to three digits, exactly as intended. Meanwhile
`rib_thickness` fell 0.120 → **0.0748**, from 1.13 lattice cells to 0.70:
below what the mesh can resolve, below what the tet route already refuses
(§5.2), and below any sand-casting minimum section. Nothing objected, because
nobody wrote that hinge. A manufacturing constraint system (§2.17) with a
*global* minimum-section rule rather than one hand-written hinge per
dimension is what this wants.

**Eight of the twenty-three free parameters did not move at all.** The four
`lug_*` sketch vertices and the four `stiffener_a/b` x-coordinates came back
bit-identical. The lugs are at the flange corners, far from the loaded seat,
and their compliance sensitivity is genuinely near zero — but "genuinely near
zero" and "silently disconnected" look the same in the output, and after
§2.19 (a parameter cannot appear in an expression) the reader has real reason
to wonder which one they are seeing. **Proposed:** report the per-parameter
gradient at step 0 alongside the trajectory; a parameter whose gradient is
*exactly* 0.0 is disconnected, and one whose gradient is merely small is not.
---

## 9. In the app

Driven with Playwright against `python -m cadjoint.viewer.playground --port 5742`
on this branch's committed bundle. Screenshots:

* `research/design/scenes/motor-shield-model.png` — the whole casting, MODEL mode
* `research/design/scenes/motor-shield-tower.png` — the bearing tower, riser and ribs
* `research/design/scenes/motor-shield-underside.png` — the flange underside and drain boss
* `research/design/scenes/motor-shield-simulate.png` — SIMULATE mode: both studies, their meshes and their boundary conditions

**It compiles and it renders.** The object tree comes back with **40
construction objects** — flange, lug, spigot, bearing tower, seal boss,
retainer pad, bearing seat, seal counterbore and the rest — the WGSL path
traces the part in Ultra preview on Metal, and the sketch overlay, dimensions
and lock glyphs are all live on it. On the evidence of the screenshots the
part in the viewport is the part the file describes.

### 9.1 The 90-second compile budget is a coin toss on this part

Three consecutive attempts, same server, same scene:

| attempt | conditions | result |
| --- | --- | --- |
| 1 | immediately after the warm-up's four killed `mesh` jobs, browser rendering the default scene | **failed at exactly 90.0 s** — *"Compilation exceeded the 90-second timeout"* |
| 2 | `POST /compile` with `curl`, machine otherwise idle | ok, **30.8 s**, 23.9 MB |
| 3 | through the UI, cache now warm | ok, **35–37 s** |

The worker's own cold time is 23.3 s (§3.1). The failure is not the scene's
size, it is **contention**: the budget is a wall-clock timeout on a
subprocess that shares a laptop with a path tracer and with the playground's
own warm-up, and a job that takes a third of the budget on an idle machine
takes all of it on a busy one. When it trips, the author is told
*"Compilation exceeded the 90-second timeout"* and gets nothing else — no
partial result, no note that it nearly finished, no suggestion to retry.

**Proposed:** measure the budget in *worker CPU seconds* rather than wall
clock (the job registry already samples `cpu_seconds` per job — attempt 1's
record shows `cpu_seconds: 54.6` against `elapsed_s: 90.0`), and on a timeout
say which it was. A job that used 55 CPU-seconds of a 90-second wall budget
was starved, and telling the author "compilation is too slow" in that case is
simply false.

### 9.2 The construction overlay is unreadable at this scale

259 profile vertices across 40 construction nodes, all drawn at once, over a
part 3 units across. The screenshots show the result: several hundred grey
handle dots and red vertices layered over the casting, with the green
extrusion cages of every feature crossing each other. Every individual glyph
is correct and the composite conveys nothing — you cannot see the part
through its own sketches, and you cannot pick a vertex you meant.

This is not a defect in any one piece of it; it is the absence of a rule for
what to draw when there are forty features rather than four. The end-cap
(twelve nodes) is at the edge of readable. **Proposed:** draw the overlay for
the *selected* feature and its immediate parents at full strength and the
rest at a fraction, or gate the whole overlay on a feature being selected —
the object tree already knows the selection, and MODEL mode already dims
non-selected geometry.

### 9.3 `Nodes.cylinder` reaches the viewer and stops there

The SIMULATE panel renders both studies and every boundary condition, and one
BC row is legible:

```
Fixed support    halfspace at [0, 0, 0.01] · n [0, 0, -1]
```

The other three — the two `Nodes.cylinder` selections that §2.3 added, on the
bearing seat wall and the coolant jacket — render **no description at all**.
Their values (`Heat flux 20000`, `Fixed value 0`, `Traction …`) are right;
the region each applies to is blank.

The cause is exact and is the downstream half of §2.3's own change:
`frontend/src/studies.ts`'s `describeSelection` has cases for `box`, `sphere`,
`halfspace`, `side`, `predicate`, `and`, `or`, `not` — and no `cylinder`.
Neither does `frontend/src/selectionEval.ts`, which is what previews and
highlights the selected nodes in the viewport, nor the `StudySelection` union
in `frontend/src/types.ts`. So the new selection kind serializes, round-trips
and solves correctly, and is invisible in the one place a person would check
it.

A Python-side node selection is not finished when it is serializable. It is
finished when the three places that consume the serialization —
`describeSelection`, `selectionEval`, and the `BcBuilder`'s kind list —
know about it. That is the change §2.3 should have carried and did not;
`frontend/**` belongs to another line of work this week, so it is recorded
here rather than made.

### 9.4 Two things seen and not chased

* **A WebGPU validation error** appears in the viewport after the scene
  swap: *"[Buffer "SDF parameters"] used in submit while destroyed"*. It is
  non-fatal — the frame after it renders correctly, which is why the
  screenshots are clean — and it comes from the renderer's buffer lifecycle
  during a program change, in `frontend/src/viewer/` and
  `cadjoint/viewer/_webgpu.py`, both of which are being actively changed by
  another line of work. Recorded, not diagnosed.
* **The elastic study's traction reads `3000`** in the panel where the file
  says `3.0e6`. This is most likely the input box clipping a seven-digit
  number rather than a wrong value — the thermal study's `20000` is exact and
  the solved displacements match a 3 MPa hand check (§7) — but a numeric field
  that silently shows a *plausible wrong number* is worth a look by whoever
  owns it.

---

## 10. Where the system gives out

Ordered by how hard the wall is.

1. **Tet meshing, absolutely.** Neither TetGen nor Gmsh can mesh this part at
   any rung of the refinement ladder (§5.2). This is the only outright
   *cannot*, and its cause is a modelling decision the language happily
   allowed: sub-cell features. A mesh-resolution-aware warning at declaration
   time — "`shield-tet10` samples at 0.107 and your thinnest wall is 0.05" —
   would have said so in one second instead of 142.
2. **B-rep extraction, informatively.** It completed in 156 s and returned a
   structure that was wrong in the ways the axiom battery predicted, at a scale
   that made each one visible: χ = 60 against −54, 80 % ambiguous vertices,
   142 faces on one octagonal flange (§6 — measured before the module was
   removed from this repository).
3. **Derived dimensions, structurally.** `Scalar` has no arithmetic, so every
   relation between two dimensions is a dead Python float (§2.19). This is the
   largest *language* gap and the one that most changes what the file can be.
4. **The 90-second budgets, intermittently.** The cold edge overlay is 129 s
   against 90 (§3.1) and the compile fails under contention at a third of its
   own budget's CPU (§9.1).
5. **The overlay and the panels, at forty features.** Both are correct and
   both stop being usable somewhere between twelve features and forty
   (§9.2, §9.3).

And the four things that **did** hold up, which are worth saying because the
rest of this document is a list of what did not:

* the **SDF program** — 52 leaves, 41 free parameters, and the whole thing
  lowers, `vmap`s and differentiates in about 5 s, with the batched program
  only 27 % larger than the single-point one (§3);
* the **hex mesher and both solvers** — 3 748 elements, a thermal and an
  elastic solve in 10 s and 5 s, both within a factor of two of a hand
  calculation (§7);
* the **optimiser** — 49 % objective reduction in six steps with the gradient
  norm falling two orders of magnitude, on a 23-parameter design whose
  gradient runs through a FEM solve (§8);
* the **constraint solver** — eleven constrained sketches, three `SketchPlane`
  levels deep, solved in the 4.5 s the module takes to import at all.
