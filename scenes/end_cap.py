"""Gearbox output end-cap: flanged, ribbed, bored, and bolted.

A cast aluminium end-cap of the kind that closes a gearbox and carries the
output shaft's bearing — the part exists to do four things at once, and every
one of them is a different modelling idea:

  * **carry the bearing.** A stepped seat with a circlip groove, cut by a
    ``revolve``, and a through bore whose radius is a live design parameter.
  * **bolt to the case.** Four corner holes on a bolt circle, patterned about
    the bearing's *own* axis of revolution rather than about a letter.
  * **stay stiff and shed heat.** Eight gusset ribs in a polar pattern,
    driven by one constrained sketch.
  * **feed oil to the bearing.** A lofted port on the flank, with its own
    three-screw circle about a horizontal axis.

The feature tree is deliberately deep. The bearing boss is sketched on the
flange's top face, the seal land on the boss's top face, and the retainer pad
on the seal land's top face — three ``SketchPlane.on`` links, each one an
expression in the feature below it, so re-dimensioning the flange carries the
whole stack with it.

Named design parameters:
  - ``flange_thickness``: extrusion depth of the mounting flange
  - ``bore_radius``: the shaft bore, and the seat the bearing presses into
  - ``boss_height``: how far the bearing boss stands off the flange
  - ``rib_height``: distance constraint from a rib's heel to its crest
  - ``bolt_circle``: radius of the four-hole flange bolt circle
  - ``port_length``: length of the lofted lubrication port

Read the comments in order: this file is meant to teach the modelling
language, and each section names the one idea it is there to show.

Two conventions worth knowing before reading:

**Every extrusion straddles its sketch plane.** ``extrude`` spans ±depth/2, so
a boss that should *sit on* a face is sketched on that face pushed up by half
its own depth — ``face.plane(offset=depth / 2)``. There is no one-sided
extrude; see ``research/complex-scene.md``.

**A driving parameter must be shared, not derived.** A ``Scalar`` handed
straight to the feature it drives (``flange_thickness`` below) stays a live
edge in the gradient graph. A number *computed* from one — a generated vertex,
a derived plane origin — is snapshotted, and the gradient does not reach back
through it. That is why the bore is a ``Face.hole`` sharing ``bore_radius``
rather than a 32-gon drawn at that radius.
"""

import math

import jax
import jax.numpy as jnp

from cadjoint import extract_parameters, functionalize
from cadjoint.constraints import (
    DistanceConstraint,
    FixedConstraint,
    HorizontalConstraint,
    PerpendicularEdgesConstraint,
    PointOnLineConstraint,
    VerticalConstraint,
    satisfy_constraints,
)
from cadjoint.construction import Axis, PolygonProfile, SketchPlane, Solid, extrude, loft, revolve
from cadjoint.fem import Dirichlet, HeatFlux, Nodes, SimMesh, ThermalStudy
from cadjoint.geometry import Scalar, Vector2
from cadjoint.render import Material
from cadjoint.sdf.boolean import Difference, Union
from cadjoint.sdf.transforms.fields import Mirror, Shell
from cadjoint.sdf.transforms.patterns import LinearPattern, PolarPattern

# ── design parameters ────────────────────────────────────────────────────────
# The two marked free are the ones the differentiability test drives; both are
# handed directly to the feature they dimension, which is what keeps them in
# the gradient graph.
flange_thickness = Scalar(0.18, free=True, name="flange_thickness")
bore_radius = Scalar(0.30, free=True, name="bore_radius")
boss_height = Scalar(0.42, name="boss_height")
rib_height = Scalar(0.40, name="rib_height")
# The bolt circle sits out on the flange's diagonals, past the ribs. That is
# not decoration: the ribs sweep r = 0.60 to 0.95, so a bolt circle inside
# that band drills straight through them.
bolt_circle = Scalar(1.05, name="bolt_circle")
port_length = Scalar(0.46, name="port_length")

aluminum = Material(name="aluminum", color=[0.78, 0.80, 0.84], roughness=0.42, metallic=0.85)
bronze = Material(name="bronze", color=[0.72, 0.50, 0.26], roughness=0.30, metallic=0.90)
steel = Material(name="steel", color=[0.55, 0.57, 0.62], roughness=0.35, metallic=0.88)
nitrile = Material(name="nitrile", color=[0.12, 0.12, 0.14], roughness=0.85, metallic=0.02)

# ── 1. mounting flange: a generated outline ──────────────────────────────────
# A cast flange has rounded corners, and a profile here is a polygon and
# nothing else — so the rounding is traced as vertices before the solid
# exists. `rounded_rect` does that; writing its 16 vertices by hand is the
# workaround it removes. Generated vertices are PINNED: a corner of a fillet
# is a consequence of the radius, not a freedom of its own.
#
# Segment counts throughout this file are a BUDGET, not a preference. The
# polygon distance unrolls one op chain per vertex, so the viewer's edge
# overlay costs roughly linearly in the total vertex count over the whole
# scene: 168 vertices took 112 s and blew the 90 s budget, 116 take 60 s.
# Every circle here is the coarsest count that still reads as round.
# The plane sits at half the nominal thickness so the flange's mounting face
# lands on z = 0 and every height in this file reads as "above the mounting
# face" (the convention scenes/bracket.py uses for its base plate). An
# extrusion straddles its plane, so without this the whole stack would sit
# half a flange low.
flange_profile = PolygonProfile.rounded_rect(
    2.0, 2.0, 0.36, segments=3, plane=SketchPlane(origin=[0.0, 0.0, 0.09]), name="flange"
)
flange = extrude(flange_profile, depth=flange_thickness, material=aluminum)

# ── 2. bearing boss: sketched ON the flange's top face (chain depth 1) ───────
# `flange.cap("+")` is not a stored surface — it is recomputed from the
# flange's own depth every time this program runs, so editing
# flange_thickness above lifts the boss, the seal land and the retainer pad
# with it. `.plane(offset=...)` pushes the sketch up half the boss height so
# the boss sits ON the flange instead of straddling it.
boss_profile = PolygonProfile.circle(
    radius=0.62,
    segments=24,
    plane=flange.cap("+").plane(offset=boss_height.value / 2.0),
    name="bearing boss",
)
boss = extrude(boss_profile, depth=boss_height, material=aluminum)

# ── 3. seal land: sketched on the boss's top face (chain depth 2) ────────────
# The raised land the shaft seal presses into.
land_profile = PolygonProfile.circle(
    radius=0.50,
    segments=20,
    plane=boss.cap("+").plane(offset=0.06),
    name="seal land",
)
seal_land = extrude(land_profile, depth=0.12, material=aluminum)

# ── 4. retainer pad: sketched on the seal land's top face (chain depth 3) ────
# Three `SketchPlane.on` links deep. Unlike the two generated outlines above,
# this profile is written as literal Vector2 points, which is what makes it
# draggable in the viewer — the source-map can only rewrite a vertex it can
# see as a literal.
pad_inner_low = Vector2(value=[0.32, -0.07], free=True, name="pad_inner_low")
pad_outer_low = Vector2(value=[0.48, -0.07], free=True, name="pad_outer_low")
pad_outer_high = Vector2(value=[0.48, 0.07], free=True, name="pad_outer_high")
pad_inner_high = Vector2(value=[0.32, 0.07], free=True, name="pad_inner_high")
pad_profile = PolygonProfile(
    [pad_inner_low, pad_outer_low, pad_outer_high, pad_inner_high],
    plane=seal_land.cap("+").plane(offset=0.03),
    name="retainer pad",
)
retainer_pad = extrude(pad_profile, depth=0.06, material=aluminum)

# The pad is a rectangle and is held as one: two horizontals, two verticals.
HorizontalConstraint(pad_inner_low, pad_outer_low)
HorizontalConstraint(pad_inner_high, pad_outer_high)
VerticalConstraint(pad_inner_low, pad_inner_high)
VerticalConstraint(pad_outer_low, pad_outer_high)

# ── 5. bearing seat: a revolved cut with a circlip groove ────────────────────
# Sketch plane normal +Y gives in-plane axes u = -X, v = +Z, and `revolve`
# spins about the plane's local Y — so the axis is world Z and profile
# coordinates read as (radius, height). The step out at 0.45 and back is the
# snap-ring groove.
seat_profile = PolygonProfile(
    [
        [0.28, -0.06],
        [0.40, -0.06],
        [0.40, 0.20],
        [0.45, 0.20],
        [0.45, 0.26],
        [0.40, 0.26],
        [0.40, 0.32],
        [0.28, 0.32],
    ],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="bearing seat",
    free=False,
)
seat_cut = revolve(seat_profile)

# A revolve has no planar face to name, but it does know the line it was swept
# around — and that line, not a letter, is what the bolt circle needs.
bore_axis = seat_cut.axis

# ── 6. gusset ribs: one constrained sketch, patterned about the bearing axis ─
# The one sketch in this part with real design freedom, and the one carrying a
# driving dimension. In this plane profile x runs along -world-X and profile y
# is world height, so the rib lives on the +X side at negative profile x.
rib_heel = Vector2(value=[-0.95, 0.16], free=True, name="rib_heel")
rib_toe = Vector2(value=[-0.60, 0.16], free=True, name="rib_toe")
rib_crest = Vector2(value=[-0.60, 0.56], free=True, name="rib_crest")
rib_slope = Vector2(value=[-0.78, 0.36], free=True, name="rib_slope")
rib_profile = PolygonProfile(
    [rib_heel, rib_toe, rib_crest, rib_slope],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="gusset rib",
)
rib = extrude(rib_profile, depth=0.11, material=aluminum)

# The constraint set says what the rib IS, not where its points happen to be:
# the heel is pinned to the flange edge, the root lies flat on the flange, the
# toe rises squarely up the boss wall for a named distance, and the sloping
# face is straight — rib_slope is not a free corner, it is a point ON the line
# from heel to crest. Editing rib_height re-proportions the gusset; nothing
# else has to be touched.
# Note what is NOT here: a VerticalConstraint on the toe-to-crest edge. The
# horizontal root plus the perpendicular corner already force it, and a third
# statement of the same fact makes the constraint Jacobian rank-deficient.
# That is a legal thing to draw and the solver now tolerates it, but it is
# still one relation too many to write down.
FixedConstraint(rib_heel, [-0.95, 0.16])
HorizontalConstraint(rib_heel, rib_toe)
PerpendicularEdgesConstraint(rib_heel, rib_toe, rib_toe, rib_crest)
DistanceConstraint(rib_toe, rib_crest, rib_height)
PointOnLineConstraint(rib_slope, rib_heel, rib_crest)

ribs = PolarPattern(rib, count=8, axis=bore_axis)

# ── 7. lubrication port: a loft on the flank ─────────────────────────────────
# Two circular profiles of the SAME vertex count, flaring outward into a
# mounting pad. Loft pairs vertex i of one profile with vertex i of the other,
# so both come from the same generator at the same segment count — mixing
# generators here would twist the transition (research/complex-scene.md).
# Half a rib pitch round from the nearest gusset. That is a real constraint,
# not a preference: the ribs are 45 degrees apart and the port's bore is
# 0.10 across, so a port on a rib centreline drills straight down the rib.
port_angle = math.radians(22.5)
port_direction = [math.cos(port_angle), math.sin(port_angle), 0.0]
port_center = [0.62 * math.cos(port_angle), 0.62 * math.sin(port_angle), 0.40]
port_plane = SketchPlane(origin=port_center, normal=port_direction)
port_root = PolygonProfile.circle(radius=0.22, segments=12, plane=port_plane, name="port root")
port_face = PolygonProfile.circle(radius=0.34, segments=12, plane=port_plane, name="port face")
port = loft(port_root, port_face, height=port_length, material=aluminum)

# ── 8. locating dowels: one boss, mirrored across the flange's own midplane ──
# `midplane` builds the plane halfway between the flange's two faces, and
# `Mirror` reflects across it — putting an identical dowel on the underside.
# Neither the plane nor the mirror is a world coordinate plane, which is the
# ordinary case for a real part and the reason both take a reference.
dowel_profile = PolygonProfile.circle(
    radius=0.075,
    center=(0.0, 0.80),
    segments=14,
    plane=flange.cap("+").plane(offset=0.055),
    name="dowel",
)
dowel = extrude(dowel_profile, depth=0.11, material=steel)
flange_midplane = SketchPlane.midplane(flange.cap("-"), flange.cap("+"))
dowel_under = Mirror(dowel, flange_midplane)

# ── 9. the cuts ──────────────────────────────────────────────────────────────
# `Face.hole` turns a face into the tool you subtract from it: a true cylinder
# on the face's normal, at face-local coordinates, sharing the radius
# parameter. It returns the TOOL rather than a cut solid, which is what lets
# the bolt hole below be patterned before it is ever subtracted.
bore = seal_land.cap("+").hole(bore_radius, depth=0.85, through=0.05)
_corner = float(bolt_circle.value) * 0.5**0.5
bolt_hole = flange.cap("+").hole(0.075, depth=0.26, through=0.05, at=(_corner, _corner))
bolt_holes = PolarPattern(bolt_hole, count=4, axis=bore_axis)
port_bore = port.cap("+").hole(0.10, depth=0.60, through=0.03)

# The port's screw circle turns about a HORIZONTAL line — the port's own axis.
# This is the case a named axis cannot express: a letter says which way the
# line points, but not that it passes through the port's own centre.
port_axis = Axis(origin=port_center, direction=port_direction)
port_screw = port.cap("+").hole(0.045, depth=0.13, through=0.02, at=(0.26, 0.0))
port_screws = PolarPattern(port_screw, count=3, axis=port_axis)

# Two tapped holes in the retainer pad, spaced along the pad — a linear
# pattern is the right shape for a row, a polar one for a circle.
pad_tap = retainer_pad.cap("+").hole(0.022, depth=0.05, at=(0.36, 0.0))
pad_taps = LinearPattern(pad_tap, direction=[1.0, 0.0, 0.0], count=2, spacing=0.08)

# ── 10. the housing ──────────────────────────────────────────────────────────
# The smoothness on the union is a deliberate fillet: it rounds where the ribs
# and boss meet the flange, which is what a casting does and what a sharp
# `min` would not. It is 0.022 against a rib 0.11 thick — a fifth of the
# thickness, which reads as a cast radius; at 0.04 the blend ate most of the
# rib and the gussets rendered as soft swellings instead of ribs. The smaller
# one on the difference breaks the edges of the bores. This is what an implicit modeller offers instead of selecting an edge
# and asking for a radius — see research/complex-scene.md on what it can and
# cannot do.
housing_body = Union(
    flange, boss, seal_land, retainer_pad, ribs, port, dowel, dowel_under, smoothness=0.022
)
housing = Difference(
    housing_body, bore, seat_cut, bolt_holes, port_bore, port_screws, pad_taps, smoothness=0.012
)
housing.name = "housing"  # named: the simulation mesh selects it as its domain

# ── 11. the rest of the assembly: rendered, not simulated ────────────────────
# Context the study never sees. The SimMesh below meshes only `housing`.
race_profile = PolygonProfile(
    [[0.30, 0.0], [0.395, 0.0], [0.395, 0.30], [0.30, 0.30]],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="bearing race",
    free=False,
)
bearing = revolve(race_profile, material=bronze)

# A lip seal is a thin-walled part, which is exactly what `Shell` makes: a
# wall of a given thickness centred on the blank's surface.
seal_blank_profile = PolygonProfile(
    [[0.22, 0.52], [0.295, 0.52], [0.295, 0.64], [0.22, 0.64]],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="lip seal blank",
    free=False,
)
lip_seal = Shell(revolve(seal_blank_profile, material=nitrile), thickness=0.035)

bolt_head = Solid.cylinder(
    radius=Scalar(0.13),
    height=Scalar(0.035),
    position=[_corner, _corner, 0.215],
    material=steel,
    name="bolt_head",
)
bolt_heads = PolarPattern(bolt_head, count=4, axis=bore_axis)

scene = Union(housing, bearing, lip_seal, bolt_heads, smoothness=0.008)
satisfy_constraints(scene, steps=2)

# ── simulation mesh: the housing only, on a named hex grid ───────────────────
# Hexes, not tets: at matched accuracy a hex mesh costs about a quarter of a
# TET10 one, and nothing here is a thin inclined feature the lattice would
# staircase over badly. The box has to contain the whole housing, port
# included, with a margin.
cap_mesh = SimMesh(
    name="cap-mesh",
    resolution=(26, 26, 13),
    domain=housing,
    bounds=(-1.08, -1.08, -0.18),
    size=(2.16, 2.16, 1.05),
)

# ── thermal study: bearing friction in, mounting face out ────────────────────
# The bearing dissipates friction heat into the bore wall; the flange is
# bolted to a comparatively cold gearbox case, so its underside is held at
# ambient. Node selections are programmatic and anchored on geometry the
# design parameters do not move far: the flux box surrounds the bore, and the
# ambient halfspace catches the flange's bottom face.
cap_study = ThermalStudy(
    name="cap-conduction",
    conductivity=2.4,
    bcs=[
        HeatFlux(
            Nodes.box((-0.34, -0.34, 0.0), (0.34, 0.34, 0.62)),
            5.0,
        ),
        Dirichlet(Nodes.halfspace([0.0, 0.0, 0.005], [0.0, 0.0, -1.0]), 0.0),
    ],
    mesh=cap_mesh,
)

# ── traceability: the aluminium volume as a function of the parameters ───────
# Not an optimization — a proof that the part is still differentiable end to
# end after all of the above. `tests/scenes/test_end_cap.py` finite-difference
# checks d(volume)/d(flange_thickness) and d(volume)/d(bore_radius) against
# these exact reverse-mode gradients.
housing_parameters, housing_fixed, _ = extract_parameters(housing)
housing_sdf = functionalize(housing)

_axes = [
    jnp.linspace(-1.0, 1.0, 17),
    jnp.linspace(-1.0, 1.0, 17),
    jnp.linspace(-0.12, 0.82, 13),
]
volume_cells = jnp.stack(jnp.meshgrid(*_axes, indexing="ij"), axis=-1).reshape(-1, 3)
volume_cell = float((2.0 / 16) * (2.0 / 16) * (0.94 / 12))


def housing_volume(parameters):
    """Smoothed aluminium volume of the housing, differentiable in `parameters`."""
    sdf = housing_sdf(parameters, housing_fixed)
    return volume_cell * jnp.sum(jax.nn.sigmoid(-sdf(volume_cells) / 0.03))
