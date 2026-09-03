"""Liquid-cooled motor end shield: helical jacket, bearing tower, fan and shroud.

The drive-end shield of a totally-enclosed electric motor, cast in aluminium,
with a helical coolant passage cast into the wall of its bearing tower. It
was chosen over the gearbox end-cap (``scenes/end_cap.py``) because every job
it does lands on a *harder* corner of the modelling language than the last
part did, and several of them land on the B-rep extraction's known failure
classes on purpose:

  * **cool the bearing.** A helical channel — a ``twist``-ed extrusion of a
    small circle drawn off the tower axis — fed and drained by two bores that
    meet the helix's end discs *tangentially* and *coincidentally*, from two
    cast-in risers whose walls are *externally tangent* to the tower along a
    line (the ``cyl_tangent`` axiom case).
  * **carry the bearing.** Two revolved cuts about one axis: the stepped seat
    with its circlip groove, and the seal counterbore above it. The through
    bore is a ``Face.hole`` sharing the live ``bore_radius``.
  * **bolt to the stator.** Four corner lugs whose top and bottom faces are
    *coplanar* with the flange's (the ``boxes_coplanar`` case), each carrying
    a counterbored bolt hole — one hard ``Union`` of two coaxial tools,
    patterned four ways about the bearing axis. Four tie bolts and a
    revolved locating spigot finish the joint.
  * **carry the shroud.** Four pairs of tapped holes in the tower's top face:
    a ``LinearPattern`` inside a ``PolarPattern`` — a pattern of patterns.
  * **move air.** A seven-blade fan whose blades are twisted extrusions
    (``twist=blade_twist``, a named parameter) in a polar pattern, inside a
    thin conical shroud made by ``Shell`` — two viewport cells of wall.
  * **hold the fan.** A threaded shaft end (a twisted extrusion of a circle
    with one tooth — the only way this language can spell a thread), a
    diamond-knurled locknut (the ``Intersection`` of one toothed profile
    twisted both ways) and a splined drive end (a toothed profile).
  * **locate and seal.** A revolved spigot ring under the mounting face,
    a grease reservoir behind the bearing with a half-cell nipple bore, an
    encoder pocket cut from a constrained sketch in the flange face's own
    frame, with a cable-gland boss and bore on an inclined line.
  * **stiffen.** An eight-station gusset ring with two stations *suppressed*
    (``PolarPattern(..., skip=(3, 5))``) because the coolant gallery is cast
    where those ribs would have been, and four tangential stiffeners whose
    inner faces are the tower's tangent planes — ribs meeting a shell
    tangentially, on purpose — all sharing one free ``rib_thickness``.
  * **cast it.** A drafted drain-plug boss (``extrude(draft=...)``, whose
    price is that it declares no faces) and a nameplate pad sketched on the
    tower's curved wall through ``SketchPlane.tangent`` — a plane meeting a
    cylinder tangentially, the planar cousin of ``cyl_tangent`` again.

The shield's feature tree is three ``SketchPlane.on`` links deep: flange →
bearing tower → seal boss → retainer pad, with one ``SketchPlane.tangent``
link off the tower's curved wall. Sub-cell fillets (``smoothness``
below a viewport cell) sit next to a real one (above a cell) so both ends of
the blend-band failure class are in one part.

Named design parameters:
  - ``flange_thickness``: extrusion depth of the mounting flange (free)
  - ``bore_radius``: the through bore under the bearing seat (free)
  - ``tower_height``: how far the bearing tower stands off the flange
  - ``rib_height``: distance constraint from a gusset's toe to its crest
  - ``blade_twist``: total twist of one fan blade, hub to tip, in degrees
  - ``channel_turns``: how many turns the coolant helix makes
  - ``blade_chord``: distance constraint across one blade's chord
  - ``rib_thickness``: extrusion depth shared by the radial and tangential ribs (free)
  - ``lug_size``: distance constraints on the corner lug's two edges
  - ``pocket_width``: distance constraint across the encoder pocket
  - ``stiffener_length``: distance constraint along a tangential stiffener
  - ``drain_draft``: mould draft angle on the drain-plug boss, in degrees

Two conventions carry over from ``scenes/end_cap.py`` and matter here too:
every extrusion straddles its sketch plane (so a boss that sits ON a face is
sketched at ``face.plane(offset=depth / 2)``), and a driving parameter must
be handed *directly* to the feature it dimensions to survive into the
gradient — which is why ``bore_radius`` is a ``Face.hole`` radius and why the
seat's revolve does not touch it.

The viewer's edge-overlay grid is 64 cells over a 6-unit box, 0.094 per cell.
Every "thin" dimension in this file is stated in those cells.
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
from cadjoint.construction import PolygonProfile, SketchPlane, Solid, extrude, loft, revolve
from cadjoint.fem import (
    Dirichlet,
    ElasticStudy,
    Fixed,
    HeatFlux,
    Nodes,
    SimMesh,
    ThermalStudy,
    Traction,
)
from cadjoint.geometry import Scalar, Vector, Vector2
from cadjoint.materials import (
    aluminium_6061,
    copper_c11000,
    fr4,
    pla,
    steel_1018,
    thermal_pad,
    titanium_ti6al4v,
)
from cadjoint.optimize import Optimization
from cadjoint.render import Material
from cadjoint.sdf.boolean import Difference, Intersection, Union
from cadjoint.sdf.transforms.fields import Mirror, Shell
from cadjoint.sdf.transforms.patterns import LinearPattern, PolarPattern

# ── design parameters ────────────────────────────────────────────────────────
flange_thickness = Scalar(0.20, free=True, name="flange_thickness")
bore_radius = Scalar(0.32, free=True, name="bore_radius")
tower_height = Scalar(0.80, name="tower_height")
rib_height = Scalar(0.45, name="rib_height")
blade_twist = Scalar(40.0, name="blade_twist")
channel_turns = Scalar(1.0, name="channel_turns")
blade_chord = Scalar(0.30, name="blade_chord")
rib_thickness = Scalar(0.12, free=True, name="rib_thickness")
lug_size = Scalar(0.50, name="lug_size")
pocket_width = Scalar(0.34, name="pocket_width")
stiffener_length = Scalar(1.20, name="stiffener_length")
drain_draft = Scalar(8.0, name="drain_draft")

# ── materials: one catalogue entry per functional part ───────────────────────
# Real handbook properties (SI) — the studies below take their conductivity,
# modulus, Poisson ratio and density from these rather than stating numbers.
aluminium = aluminium_6061()  # the cast shield
steel = steel_1018()  # shaft, bearing, shroud, shroud legs
plastic = pla()  # the fan
titanium = titanium_ti6al4v()  # the locknut
brass = copper_c11000()  # the cable gland nut (copper stands in for brass)
board = fr4()  # the encoder board
gap_pad = thermal_pad()  # the pad under the encoder board
nitrile = Material(name="nitrile", color=[0.12, 0.12, 0.14], roughness=0.85, metallic=0.02)

# ── 1. mounting flange ───────────────────────────────────────────────────────
# Mounting face on z = 0; the sketch plane sits at half the depth so the
# extrusion's underside lands there. A generated outline: 16 pinned vertices.
flange_profile = PolygonProfile.rounded_rect(
    2.6, 2.6, 0.45, segments=3, plane=SketchPlane(origin=[0.0, 0.0, 0.10]), name="flange"
)
flange = extrude(flange_profile, depth=flange_thickness, material=aluminium)

# ── 2. corner lugs: COPLANAR with the flange, on purpose ─────────────────────
# A lug is a square extruded through the flange's own depth from the flange's
# own sketch plane, so its top and bottom faces lie IN the flange's top and
# bottom planes. That is the `boxes_coplanar` axiom case: two patches sharing
# a zero set, where the B-rep's residual gate is blind. One lug, patterned
# four ways. (It is not a `Solid.box`: a box's size is one Vector parameter,
# and one component of it cannot be tied to a shared Scalar — see the report.)
# The lug is a constrained sketch: its inner corner is pinned, its two
# edges carry the named `lug_size`, and the square is held square.
lug_a = Vector2(value=[1.0, 1.0], free=True, name="lug_a")
lug_b = Vector2(value=[1.5, 1.0], free=True, name="lug_b")
lug_c = Vector2(value=[1.5, 1.5], free=True, name="lug_c")
lug_d = Vector2(value=[1.0, 1.5], free=True, name="lug_d")
lug_profile = PolygonProfile([lug_a, lug_b, lug_c, lug_d], plane=flange_profile.plane, name="lug")
lug = extrude(lug_profile, depth=flange_thickness, material=aluminium)
FixedConstraint(lug_a, [1.0, 1.0])
HorizontalConstraint(lug_a, lug_b)
VerticalConstraint(lug_b, lug_c)
HorizontalConstraint(lug_c, lug_d)
VerticalConstraint(lug_d, lug_a)
DistanceConstraint(lug_a, lug_b, lug_size)
DistanceConstraint(lug_b, lug_c, lug_size)

# A locating spigot: a revolved ring under the mounting face that registers
# in the stator bore. Its outer wall (r = 1.05) is a cylinder coaxial with
# the tower's (r = 1.0) — two coaxial cylinders 0.05 apart, half a cell.
spigot_profile = PolygonProfile(
    [[0.95, -0.10], [1.05, -0.10], [1.05, 0.02], [0.95, 0.02]],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="spigot",
    free=False,
)
spigot = revolve(spigot_profile, material=aluminium)

# ── 3. bearing tower: sketched on the flange's top face (chain depth 1) ──────
tower_profile = PolygonProfile.circle(
    radius=1.0,
    segments=28,
    plane=flange.cap("+").plane(offset=tower_height.value / 2.0),
    name="bearing tower",
)
tower = extrude(tower_profile, depth=tower_height, material=aluminium)

# ── 4. seal boss on the tower's top face (chain depth 2) ─────────────────────
boss_profile = PolygonProfile.circle(
    radius=0.60, segments=20, plane=tower.cap("+").plane(offset=0.07), name="seal boss"
)
seal_boss = extrude(boss_profile, depth=0.14, material=aluminium)

# ── 5. retainer pad on the seal boss's top face (chain depth 3) ──────────────
# Literal Vector2 points so the viewer can drag them; held square by
# constraints. 0.18 by 0.16 by 0.06: the whole feature is under two cells.
pad_inner_low = Vector2(value=[0.34, -0.08], free=True, name="pad_inner_low")
pad_outer_low = Vector2(value=[0.52, -0.08], free=True, name="pad_outer_low")
pad_outer_high = Vector2(value=[0.52, 0.08], free=True, name="pad_outer_high")
pad_inner_high = Vector2(value=[0.34, 0.08], free=True, name="pad_inner_high")
pad_profile = PolygonProfile(
    [pad_inner_low, pad_outer_low, pad_outer_high, pad_inner_high],
    plane=seal_boss.cap("+").plane(offset=0.03),
    name="retainer pad",
)
retainer_pad = extrude(pad_profile, depth=0.06, material=aluminium)
HorizontalConstraint(pad_inner_low, pad_outer_low)
HorizontalConstraint(pad_inner_high, pad_outer_high)
VerticalConstraint(pad_inner_low, pad_inner_high)
VerticalConstraint(pad_outer_low, pad_outer_high)

# ── 6. bearing seat and seal counterbore: two revolves about one axis ────────
# Plane normal +Y: profile x is radius, profile y is world height. The seat's
# inner edge (0.30) is INSIDE the through bore (0.32) so the bore's own radius
# stays the live one everywhere except the seat band.
seat_profile = PolygonProfile(
    [
        [0.30, -0.06],
        [0.47, -0.06],
        [0.47, 0.42],
        [0.52, 0.42],
        [0.52, 0.48],
        [0.47, 0.48],
        [0.47, 0.55],
        [0.30, 0.55],
    ],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="bearing seat",
    free=False,
)
seat_cut = revolve(seat_profile)
bore_axis = seat_cut.axis

seal_profile = PolygonProfile(
    [[0.30, 0.96], [0.45, 0.96], [0.45, 1.20], [0.30, 1.20]],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="seal counterbore",
    free=False,
)
seal_cut = revolve(seal_profile)

# A grease reservoir behind the bearing: an annular cavity between the seat
# and the coolant channel. Its outer wall (r = 0.56) is 0.08 from the
# channel's inner edge (r = 0.64) — 0.85 of a cell, deliberately.
reservoir_profile = PolygonProfile(
    [[0.47, 0.56], [0.56, 0.56], [0.56, 0.78], [0.47, 0.78]],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="grease reservoir",
    free=False,
)
reservoir_cut = revolve(reservoir_profile)

# ── 7. the helical coolant channel: a twisted extrusion ──────────────────────
# A circle drawn 0.75 off the tower axis and extruded with a full turn of
# twist sweeps a helix of that radius. The twist is zero at mid-depth and
# ±180° at the caps, so BOTH end discs sit at angle 180°, on the -x side,
# 0.44 apart: that is where the inlet and outlet bores meet it. The tube is
# 0.22 across (2.3 cells); the wall outside it is 0.14 (1.5 cells), the wall
# inside it to the circlip groove is 0.12 (1.3 cells), the land between
# consecutive turns is 0.22. A twisted extrusion declares no faces and its
# field is not 1-Lipschitz — both are documented, both are measured in
# research/complex-scene-2.md.
channel_profile = PolygonProfile.circle(
    radius=0.11,
    center=(0.75, 0.0),
    segments=10,
    plane=flange.cap("+").plane(offset=0.32),
    name="coolant channel",
)
channel = extrude(channel_profile, depth=0.44, twist=channel_turns.value * 360.0)
_channel_low = 0.20 + 0.32 - 0.22  # z of the bottom end disc
_channel_high = 0.20 + 0.32 + 0.22  # z of the top end disc

# Inlet and outlet: horizontal bores along ±y whose axes are tangent to the
# helix's circle at its end points, and whose end discs COINCIDE with the
# helix's end discs. A tangent-and-coincident junction is the hardest thing
# in the axiom battery, and it is what a real cast jacket does.
# Every `Solid` dimension is a FREE parameter unless told otherwise, and that
# includes an inclined tool's three rotation angles — so the bores pin all of
# theirs, or the optimiser below would tilt and slide them. And a cylinder's
# `height` is its HALF height (an extrusion's `depth` and a box's `size` are
# totals): the first draft of this file had every bore twice as long as it
# meant, the gland bore reaching into the tower — see the report.
_pinned = {"free": False}
inlet_bore = Solid.cylinder(
    radius=Scalar(0.11, **_pinned),
    height=Scalar(0.50, **_pinned),
    position=Vector([-0.75, -0.50, _channel_low], **_pinned),
    rotation=[Scalar(math.pi / 2.0, **_pinned), Scalar(0.0, **_pinned), Scalar(0.0, **_pinned)],
    name="inlet_bore",
)
outlet_bore = Solid.cylinder(
    radius=Scalar(0.11, **_pinned),
    height=Scalar(0.50, **_pinned),
    position=Vector([-0.75, 0.50, _channel_high], **_pinned),
    rotation=[Scalar(math.pi / 2.0, **_pinned), Scalar(0.0, **_pinned), Scalar(0.0, **_pinned)],
    name="outlet_bore",
)

# ── 8. risers: EXTERNALLY TANGENT to the tower along a line ──────────────────
# Each riser's axis is exactly tower radius + riser radius from the tower
# axis, so the two cylinder walls touch along one vertical line with
# antiparallel normals: the `cyl_tangent` case. The second riser is the
# first mirrored across the world XZ plane — the letter form of Mirror is
# right here because that plane really is a symmetry plane of the part.
_riser_y = math.sqrt(1.16**2 - 0.75**2)
riser = Solid.cylinder(
    radius=Scalar(0.16, **_pinned),
    height=Scalar(0.425, **_pinned),
    position=Vector([-0.75, -_riser_y, 0.575], **_pinned),
    material=aluminium,
    name="riser",
)
riser_mirrored = Mirror(riser, "y")
riser_bore = riser.cap("+").hole(0.09, depth=0.70, through=0.05)
riser_bore_mirrored = Mirror(riser_bore, "y")

# A square pipe flange at the top of each riser. Its TOP face is coplanar
# with the riser's own cap AND with the tower's top face: a flange coplanar
# with a boss face, on purpose. Two screw holes through it, in a row.
riser_flange_profile = PolygonProfile(
    [[-0.97, -1.105], [-0.53, -1.105], [-0.53, -0.665], [-0.97, -0.665]],
    plane=SketchPlane(origin=[0.0, 0.0, 0.97]),
    name="riser flange",
    free=False,
)
riser_flange = extrude(riser_flange_profile, depth=0.06, material=aluminium)
riser_flange_mirrored = Mirror(riser_flange, "y")
riser_screw = riser_flange.cap("+").hole(0.04, depth=0.10, through=0.02, at=(-0.91, -_riser_y))
riser_screws = LinearPattern(riser_screw, direction=[1.0, 0.0, 0.0], count=2, spacing=0.32)
riser_screws_mirrored = Mirror(riser_screws, "y")

# ── 9. gusset ribs: one constrained sketch, eight copies ─────────────────────
# Plane normal +Y again: profile x runs along -world-X, so the rib lives on
# the +X side at negative profile x. 0.12 thick — 1.3 cells.
rib_heel = Vector2(value=[-1.22, 0.18], free=True, name="rib_heel")
rib_toe = Vector2(value=[-0.96, 0.18], free=True, name="rib_toe")
rib_crest = Vector2(value=[-0.96, 0.63], free=True, name="rib_crest")
rib_slope = Vector2(value=[-1.09, 0.405], free=True, name="rib_slope")
rib_profile = PolygonProfile(
    [rib_heel, rib_toe, rib_crest, rib_slope],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="gusset rib",
)
rib = extrude(rib_profile, depth=0.12, material=aluminium)
FixedConstraint(rib_heel, [-1.22, 0.18])
HorizontalConstraint(rib_heel, rib_toe)
PerpendicularEdgesConstraint(rib_heel, rib_toe, rib_toe, rib_crest)
DistanceConstraint(rib_toe, rib_crest, rib_height)
PointOnLineConstraint(rib_slope, rib_heel, rib_crest)
# Eight stations, six ribs. The inlet gallery crosses the rib ring at 227°
# and the outlet at 133°, which are stations 5 and 3 of an eight-way ring —
# and a rib with a 0.22 hole through its root is not a rib. A foundry answers
# this by casting the gallery where the rib would have been, so the ring is
# patterned eight ways and two instances are suppressed. The first draft of
# this file had no way to say that (`PolarPattern` emitted all eight and the
# bore drilled straight through one), which is what put `skip` into
# `cadjoint/sdf/transforms/{fields,patterns}.py` — see the report.
ribs = PolarPattern(rib, count=8, axis=bore_axis, skip=(3, 5))
lugs = PolarPattern(lug, count=4, axis=bore_axis)

# The cast gallery bosses that stand in for the two suppressed ribs: a barrel
# of metal around each feed bore, from the riser wall to the tower wall. They
# are unioned before the bores are cut, so the gallery is a passage in metal
# rather than a slot in air.
inlet_gallery = Solid.cylinder(
    radius=Scalar(0.22, **_pinned),
    height=Scalar(0.20, **_pinned),
    position=Vector([-0.75, -0.74, _channel_low], **_pinned),
    rotation=[Scalar(math.pi / 2.0, **_pinned), Scalar(0.0, **_pinned), Scalar(0.0, **_pinned)],
    material=aluminium,
    name="inlet_gallery",
)
outlet_gallery = Solid.cylinder(
    radius=Scalar(0.22, **_pinned),
    height=Scalar(0.20, **_pinned),
    position=Vector([-0.75, 0.74, _channel_high], **_pinned),
    rotation=[Scalar(math.pi / 2.0, **_pinned), Scalar(0.0, **_pinned), Scalar(0.0, **_pinned)],
    material=aluminium,
    name="outlet_gallery",
)

# Tangential stiffeners: a wall parallel to a tangent of the tower, placed
# so that its INNER face (y = 1.06 - 0.06) is the tangent plane of the
# tower cylinder at x = 0. A plane touching a cylinder along a line is the
# planar cousin of `cyl_tangent`, and it is how a cast rib meets a shell.
# The tangency holds only while `rib_thickness` is 0.12: the plane origin is
# a number, not an expression in the thickness — see the report.
stiffener_a = Vector2(value=[-0.60, 0.18], free=True, name="stiffener_a")
stiffener_b = Vector2(value=[0.60, 0.18], free=True, name="stiffener_b")
stiffener_c = Vector2(value=[0.60, 0.50], free=True, name="stiffener_c")
stiffener_d = Vector2(value=[-0.60, 0.50], free=True, name="stiffener_d")
stiffener_profile = PolygonProfile(
    [stiffener_a, stiffener_b, stiffener_c, stiffener_d],
    plane=SketchPlane(origin=[0.0, 1.06, 0.0], normal=[0.0, 1.0, 0.0]),
    name="tangential stiffener",
)
stiffener = extrude(stiffener_profile, depth=rib_thickness, material=aluminium)
FixedConstraint(stiffener_a, [-0.60, 0.18])
HorizontalConstraint(stiffener_a, stiffener_b)
VerticalConstraint(stiffener_b, stiffener_c)
HorizontalConstraint(stiffener_c, stiffener_d)
VerticalConstraint(stiffener_d, stiffener_a)
DistanceConstraint(stiffener_a, stiffener_b, stiffener_length)
stiffeners = PolarPattern(stiffener, count=4, axis=bore_axis)

# The encoder pocket and its cable gland, at 67.5° — between the 45° and
# 90° ribs, opposite no tie bolt — on
# the flange's top face. The pocket is a constrained sketch in the FACE's
# own coordinates, cut with `Face.pocket`; the gland is a boss on the
# flange edge with a bore along the same inclined line.
_pocket_angle = math.radians(67.5)
_pocket_direction = [math.cos(_pocket_angle), math.sin(_pocket_angle), 0.0]
_pocket_center = (1.12 * math.cos(_pocket_angle), 1.12 * math.sin(_pocket_angle))
pocket_a = Vector2(value=[-0.17, -0.12], free=True, name="pocket_a")
pocket_b = Vector2(value=[0.17, -0.12], free=True, name="pocket_b")
pocket_c = Vector2(value=[0.17, 0.12], free=True, name="pocket_c")
pocket_d = Vector2(value=[-0.17, 0.12], free=True, name="pocket_d")
encoder_pocket = flange.cap("+").pocket(
    [pocket_a, pocket_b, pocket_c, pocket_d], depth=0.12, through=0.05, at=_pocket_center
)
FixedConstraint(pocket_a, [-0.17, -0.12])
HorizontalConstraint(pocket_a, pocket_b)
VerticalConstraint(pocket_b, pocket_c)
HorizontalConstraint(pocket_c, pocket_d)
VerticalConstraint(pocket_d, pocket_a)
DistanceConstraint(pocket_a, pocket_b, pocket_width)

# A cylinder's local z is turned onto the pocket's radial line by two
# pinned rotations: -90° about x (z -> +y), then -22.5° about z.
_gland_rotation = [
    Scalar(-math.pi / 2.0, **_pinned),
    Scalar(0.0, **_pinned),
    Scalar(_pocket_angle - math.pi / 2.0, **_pinned),
]
gland_boss = Solid.cylinder(
    radius=Scalar(0.10, **_pinned),
    height=Scalar(0.15, **_pinned),
    position=Vector([1.36 * d for d in _pocket_direction[:2]] + [0.10], **_pinned),
    rotation=_gland_rotation,
    material=aluminium,
    name="gland_boss",
)
gland_bore = Solid.cylinder(
    radius=Scalar(0.06, **_pinned),
    height=Scalar(0.25, **_pinned),
    position=Vector([1.35 * d for d in _pocket_direction[:2]] + [0.10], **_pinned),
    rotation=_gland_rotation,
    name="gland_bore",
)

# A drain plug boss on the flange, at 157.5° — the one perimeter station the
# rib ring, the tie bolts, the legs and the risers all leave alone. It is the
# only DRAFTED feature in the part: a sand casting's walls have to leave the
# mould, and `extrude(draft=...)` tapers them. The price is stated in the
# extrude docs and paid here: a drafted extrusion declares no faces, so the
# boss has no `cap("+")` to hang a `Face.hole` on and the drain has to be a
# free-standing `Solid.cylinder` placed by hand — see the report.
_drain_angle = math.radians(157.5)
_drain_center = (1.12 * math.cos(_drain_angle), 1.12 * math.sin(_drain_angle))
drain_profile = PolygonProfile.circle(
    radius=0.16,
    center=_drain_center,
    segments=16,
    plane=flange.cap("+").plane(offset=0.09),
    name="drain boss",
)
drain_boss = extrude(drain_profile, depth=0.18, draft=drain_draft, material=aluminium)
drain_bore = Solid.cylinder(
    radius=Scalar(0.06, **_pinned),
    height=Scalar(0.24, **_pinned),
    position=Vector([_drain_center[0], _drain_center[1], 0.19], **_pinned),
    name="drain_bore",
)

# A nameplate pad on the tower's cylindrical WALL, at 337.5°. A cylinder wall
# has no analytic face — `revolve` and a circle's `extrude` both decline to
# name one — so the plane comes from `SketchPlane.tangent`, which Newton-
# projects a nearby point onto the zero set and takes the field's gradient
# there. That makes the pad's placement differentiable in the tower's own
# parameters, and it makes the pad a plane meeting a cylinder tangentially:
# a fourth deliberate visit to the axiom battery's failure classes.
_plate_angle = math.radians(337.5)
plate_plane = SketchPlane.tangent(
    tower, near=[1.0 * math.cos(_plate_angle), 1.0 * math.sin(_plate_angle), 0.62]
)
plate_profile = PolygonProfile.rounded_rect(
    0.30, 0.24, 0.05, segments=2, plane=plate_plane, name="nameplate"
)
nameplate = extrude(plate_profile, depth=0.06, material=aluminium)

# ── 10. the cuts: bore, counterbored bolt circle, pattern of patterns ────────
bore = seal_boss.cap("+").hole(bore_radius, depth=1.30, through=0.10)

# One counterbored hole is a hard Union of two coaxial tools whose ends are
# coplanar — then patterned about the bearing axis. The bolt circle sits at
# the corners, r = 1.77, between the ribs (which stop at r = 1.22).
_corner = 1.25
bolt_hole = flange.cap("+").hole(0.09, depth=0.30, through=0.05, at=(_corner, _corner))
counterbore = flange.cap("+").hole(0.16, depth=0.08, through=0.05, at=(_corner, _corner))
bolt_tool = Union(bolt_hole, counterbore, smoothness=0.0)
bolt_holes = PolarPattern(bolt_tool, count=4, axis=bore_axis)

# The stator tie bolts: four plain holes on a circle, clear of the ribs, the
# encoder pocket and the shroud legs — the flange perimeter is a schedule
# of angular slots, and a collision here is silent (see the report: the
# first two schedules drilled a rib and the encoder pocket respectively).
_tie_angle = math.radians(22.5)
tie_bolt = flange.cap("+").hole(
    0.07, depth=0.30, through=0.05, at=(1.16 * math.cos(_tie_angle), 1.16 * math.sin(_tie_angle))
)
tie_bolts = PolarPattern(tie_bolt, count=4, axis=bore_axis)

# The grease nipple: a half-cell bore from the seal boss down into the
# reservoir, at 90° so it misses the retainer pad.
grease_port = seal_boss.cap("+").hole(0.05, depth=0.40, through=0.05, at=(0.0, 0.515))

# Shroud mounting: two tapped holes in a row, the row repeated four ways.
tap = tower.cap("+").hole(0.06, depth=0.07, at=(0.80, -0.08))
tap_pair = LinearPattern(tap, direction=[0.0, 1.0, 0.0], count=2, spacing=0.16)
shroud_taps = PolarPattern(tap_pair, count=4, axis=bore_axis)

# ── 11. the shield ───────────────────────────────────────────────────────────
# Two fillet scales side by side. The tower root gets a REAL fillet, 0.12
# (1.3 viewport cells); everything else a sub-cell one, 0.03 (0.3 cells).
# The axiom battery says the first fragments into slivers and the second is
# invisible to the graph; here they are adjacent.
core = Union(flange, tower, spigot, smoothness=0.12)
shield_body = Union(
    core,
    lugs,
    seal_boss,
    retainer_pad,
    ribs,
    stiffeners,
    riser,
    riser_mirrored,
    riser_flange,
    riser_flange_mirrored,
    gland_boss,
    inlet_gallery,
    outlet_gallery,
    drain_boss,
    nameplate,
    smoothness=0.03,
)
shield = Difference(
    shield_body,
    bore,
    seat_cut,
    seal_cut,
    reservoir_cut,
    grease_port,
    channel,
    inlet_bore,
    outlet_bore,
    riser_bore,
    riser_bore_mirrored,
    riser_screws,
    riser_screws_mirrored,
    bolt_holes,
    tie_bolts,
    shroud_taps,
    encoder_pocket,
    gland_bore,
    drain_bore,
    smoothness=0.012,
)
shield.name = "shield"  # the simulation domain

# ── 12. shaft, bearing, seal: rendered context ───────────────────────────────
shaft = Solid.cylinder(
    radius=0.30, height=1.00, position=[0.0, 0.0, 1.00], material=steel, name="shaft"
)

# The drive end below the mounting face is splined: twelve teeth, 0.06 deep,
# as a toothed profile — a spline is a knurl with fewer, deeper teeth.
_spline_vertices = []
for _i in range(24):
    _angle = 2.0 * math.pi * _i / 24
    _radius = 0.30 if _i % 2 == 0 else 0.24
    _spline_vertices.append([_radius * math.cos(_angle), _radius * math.sin(_angle)])
spline_profile = PolygonProfile(
    _spline_vertices, plane=SketchPlane(origin=[0.0, 0.0, -0.18]), name="spline", free=False
)
spline = extrude(spline_profile, depth=0.36, material=steel)

# Encoder hardware in and beside the pocket: board, gap pad, gland nut.
_pocket_xyz = [_pocket_center[0], _pocket_center[1], 0.10]
encoder_board = Solid.box(
    size=Vector([0.26, 0.18, 0.02], **_pinned),
    position=Vector(_pocket_xyz, **_pinned),
    rotation=[Scalar(0.0, **_pinned), Scalar(0.0, **_pinned), Scalar(_pocket_angle, **_pinned)],
    material=board,
    name="encoder_board",
)
encoder_pad = Solid.box(
    size=Vector([0.26, 0.18, 0.015], **_pinned),
    position=Vector([_pocket_center[0], _pocket_center[1], 0.0875], **_pinned),
    rotation=[Scalar(0.0, **_pinned), Scalar(0.0, **_pinned), Scalar(_pocket_angle, **_pinned)],
    material=gap_pad,
    name="encoder_pad",
)
gland_nut = Solid.cylinder(
    radius=Scalar(0.13, **_pinned),
    height=Scalar(0.05, **_pinned),
    position=Vector([1.56 * d for d in _pocket_direction[:2]] + [0.10], **_pinned),
    rotation=_gland_rotation,
    material=brass,
    name="gland_nut",
)
bearing_profile = PolygonProfile(
    [[0.31, 0.02], [0.465, 0.02], [0.465, 0.40], [0.31, 0.40]],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="bearing",
    free=False,
)
bearing = revolve(bearing_profile, material=steel)
seal_ring_profile = PolygonProfile(
    [[0.31, 1.00], [0.44, 1.00], [0.44, 1.16], [0.31, 1.16]],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="shaft seal",
    free=False,
)
shaft_seal = revolve(seal_ring_profile, material=nitrile)

# ── 13. the thread: a twisted extrusion of a circle with one tooth ───────────
# The language has no helix primitive. What it has is `twist`, and a
# 16-gon with one tooth pushed out to r = 0.36, twisted 2.5 turns over its
# depth, sweeps a single-start thread of pitch 0.15 (1.6 cells) and depth
# 0.06 (0.6 cells). The 16-gon itself disappears inside the exact shaft.
_thread_vertices = []
for _i in range(16):
    _angle = 2.0 * math.pi * _i / 16
    if _i == 0:
        _thread_vertices += [[0.36, -0.03], [0.36, 0.03]]
    else:
        _thread_vertices.append([0.30 * math.cos(_angle), 0.30 * math.sin(_angle)])
thread_profile = PolygonProfile(
    _thread_vertices,
    plane=SketchPlane(origin=[0.0, 0.0, 1.81]),
    name="thread",
    free=False,
)
thread = extrude(thread_profile, depth=0.38, twist=2.5 * 360.0, material=steel)

# ── 14. the locknut: a DIAMOND knurl ─────────────────────────────────────────
# Sixteen teeth, 0.05 deep (half a cell). A straight knurl is a toothed
# profile and nothing more; a diamond knurl is the intersection of two of
# them twisted opposite ways, which is what a knurling wheel pair actually
# leaves behind. Both copies extrude the SAME profile object, so the diamond
# costs one sketch and two solids — and the intersection is hard (0.0), or a
# blend would round the crossings back into a straight knurl.
_knurl_vertices = []
for _i in range(32):
    _angle = 2.0 * math.pi * _i / 32
    _radius = 0.47 if _i % 2 == 0 else 0.42
    _knurl_vertices.append([_radius * math.cos(_angle), _radius * math.sin(_angle)])
knurl_profile = PolygonProfile(
    _knurl_vertices, plane=SketchPlane(origin=[0.0, 0.0, 1.80]), name="locknut", free=False
)
locknut_blank = Intersection(
    extrude(knurl_profile, depth=0.22, twist=180.0, material=titanium),
    extrude(knurl_profile, depth=0.22, twist=-180.0, material=titanium),
    smoothness=0.0,
)
# `Difference` blends at 0.1 by default — a third of this nut's wall — so the
# bore is cut hard. The default is a trap at part scale; see the report.
locknut = Difference(
    locknut_blank,
    Solid.cylinder(radius=0.30, height=0.25, position=[0, 0, 1.80]),
    smoothness=0.0,
)

# ── 15. the fan: a hub and seven twisted blades ──────────────────────────────
# The blade is sketched on a plane whose normal points radially (+X), so the
# extrusion runs hub to tip and the twist changes the blade's pitch angle
# along the radius — which is what a fan blade is. Plane normal +X gives
# u = -Z, v = +Y: the chord is drawn along u (axial), the thickness along v.
# 0.09 thick — one viewport cell.
hub = Solid.cylinder(
    radius=0.42, height=0.18, position=[0.0, 0.0, 1.38], material=plastic, name="hub"
)
blade_le_root = Vector2(value=[-0.15, -0.045], free=True, name="blade_le_root")
blade_te_root = Vector2(value=[0.15, -0.045], free=True, name="blade_te_root")
blade_te_tip = Vector2(value=[0.15, 0.045], free=True, name="blade_te_tip")
blade_le_tip = Vector2(value=[-0.15, 0.045], free=True, name="blade_le_tip")
blade_profile = PolygonProfile(
    [blade_le_root, blade_te_root, blade_te_tip, blade_le_tip],
    plane=SketchPlane(origin=[0.69, 0.0, 1.38], normal=[1.0, 0.0, 0.0]),
    name="fan blade",
)
blade = extrude(blade_profile, depth=0.62, twist=blade_twist, material=plastic)
HorizontalConstraint(blade_le_root, blade_te_root)
HorizontalConstraint(blade_le_tip, blade_te_tip)
VerticalConstraint(blade_le_root, blade_le_tip)
VerticalConstraint(blade_te_root, blade_te_tip)
DistanceConstraint(blade_le_root, blade_te_root, blade_chord)
blades = PolarPattern(blade, count=7, axis=bore_axis)
fan = Union(hub, blades, smoothness=0.02)

# ── 16. the shroud: a thin conical shell around the fan ──────────────────────
# `Shell` turns a lofted frustum into a wall of 0.20 total thickness — two
# viewport cells, three hex cells of the simulation lattice — and a tall
# cylinder cut opens both ends. Both loft sections come from the same
# generator at the same count, so their vertices pair up by index.
shroud_plane = SketchPlane(origin=[0.0, 0.0, 1.38])
shroud_low = PolygonProfile.circle(radius=1.35, segments=16, plane=shroud_plane, name="shroud low")
shroud_high = PolygonProfile.circle(
    radius=1.22, segments=16, plane=shroud_plane, name="shroud high"
)
shroud_blank = Shell(loft(shroud_low, shroud_high, height=0.44, material=steel), thickness=0.20)
# Eight louvre windows through the wall: a box tool in a polar pattern.
window = Solid.box(
    size=Vector([0.50, 0.22, 0.16], **_pinned),
    position=Vector([1.30, 0.0, 1.38], **_pinned),
    name="window",
)
windows = PolarPattern(window, count=8, axis=bore_axis)
shroud = Difference(
    shroud_blank,
    Solid.cylinder(radius=1.0, height=0.6, position=[0.0, 0.0, 1.38]),
    windows,
    smoothness=0.0,
)

# Four legs from the flange to the shroud's lower rim, on the diagonals
# between the rib heels (r = 1.22) and the lugs (r = 1.41).
_leg_angle = math.radians(45.0)
leg = Solid.box(
    size=[0.10, 0.10, 1.05],
    position=[1.30 * math.cos(_leg_angle), 1.30 * math.sin(_leg_angle), 0.70],
    material=steel,
    name="leg",
)
legs = PolarPattern(leg, count=4, axis=bore_axis)

scene = Union(
    shield,
    shaft,
    spline,
    thread,
    locknut,
    bearing,
    shaft_seal,
    fan,
    shroud,
    legs,
    encoder_board,
    encoder_pad,
    gland_nut,
    smoothness=0.008,
)
satisfy_constraints(scene, steps=2)

# ── simulation meshes: the shield only ───────────────────────────────────────
# The hex lattice is the fast path the studies solve on. The tet10 mesh is
# declared for the measurement in `research/complex-scene-2.md` §5.2, and the
# measurement is that it **cannot be built**: the refinement ladder walks all
# three rungs — (30, 30, 14) → x1.5 → x2.25 = (68, 68, 32) — and TetGen still
# refuses the surface as self-intersecting, after 142 s, because three
# features in this part are below half a lattice cell (the 0.06 riser flange
# and pads, the 0.05 circlip groove). `mesher="gmsh"` fails on the same
# surface in 20 s with a raw Gmsh message and no ladder. It is left in the
# file deliberately: it is the part's hardest ceiling and it should stay
# visible.
_mesh_bounds = (-1.60, -1.60, -0.15)
_mesh_size = (3.20, 3.20, 1.45)
shield_mesh = SimMesh(
    name="shield-hex",
    resolution=(30, 30, 14),
    domain=shield,
    bounds=_mesh_bounds,
    size=_mesh_size,
)
shield_mesh_tet10 = SimMesh(
    name="shield-tet10",
    resolution=(30, 30, 14),
    domain=shield,
    bounds=_mesh_bounds,
    size=_mesh_size,
    method="tet10",
)

# ── node selections on the part's own cylinders ──────────────────────────────
# The bearing seat wall and the coolant jacket are both annular; a box or a
# sphere cannot pick one without the other, so both use `Nodes.cylinder`.
seat_wall = Nodes.cylinder([0.0, 0.0, 0.25], [0.0, 0.0, 1.0], 0.55, inner=0.40, half_length=0.28)
jacket_wall = Nodes.cylinder([0.0, 0.0, 0.52], [0.0, 0.0, 1.0], 0.90, inner=0.60, half_length=0.36)
mounting_face = Nodes.halfspace([0.0, 0.0, 0.005], [0.0, 0.0, -1.0])

# ── thermal study: bearing friction in, coolant jacket held cold ─────────────
# Conductivity comes from the aluminium's own datasheet value (FROM_MATERIAL).
bearing_heat = ThermalStudy(
    name="bearing-heat",
    bcs=[
        HeatFlux(seat_wall, 2.0e4),
        Dirichlet(jacket_wall, 0.0),
    ],
    mesh=shield_mesh,
)

# ── elastic study: belt pull on the bearing, flange bolted down ──────────────
# Modulus, Poisson ratio and density from the material; self-weight on.
belt_pull = ElasticStudy(
    name="belt-pull",
    gravity=(0.0, 0.0, -9.81),
    bcs=[
        Fixed(mounting_face),
        Traction(seat_wall, (3.0e6, 0.0, 0.0)),
    ],
    mesh=shield_mesh,
)

# ── manufacturing constraints as penalties, and the optimisation ─────────────
# `Optimization` has no constraint argument, so both limits are soft: a cap
# on the cast volume (a mass budget) and a minimum wall between the bearing
# seat and the coolant channel, each a softplus hinge on the free parameters.
shield_parameters, shield_fixed, _ = extract_parameters(shield)
shield_sdf = functionalize(shield)

_axes = [
    jnp.linspace(-1.55, 1.55, 25),
    jnp.linspace(-1.55, 1.55, 25),
    jnp.linspace(-0.10, 1.25, 13),
]
volume_cells = jnp.stack(jnp.meshgrid(*_axes, indexing="ij"), axis=-1).reshape(-1, 3)
volume_cell = float((3.1 / 24) * (3.1 / 24) * (1.35 / 12))

volume_cap = 4.40  # the casting must not grow past this (it starts at 4.45)
min_wall = 0.12  # seat wall to channel: the channel's inner edge is at r = 0.64


def shield_volume(parameters):
    """Smoothed aluminium volume of the shield, differentiable in `parameters`."""
    sdf = shield_sdf(parameters, shield_fixed)
    return volume_cell * jnp.sum(jax.nn.sigmoid(-sdf(volume_cells) / 0.03))


def manufacturing_penalty(parameters):
    """Soft mass budget plus a soft minimum wall under the coolant channel."""
    over_budget = 20.0 * jax.nn.softplus((shield_volume(parameters) - volume_cap) / 0.02)
    seat_radius = parameters["bore_radius"] + 0.15  # the seat is 0.15 wider than the bore
    thin_wall = jax.nn.softplus((seat_radius - (0.64 - min_wall)) / 0.005)
    return over_budget + thin_wall


stiffen_shield = Optimization(
    name="stiff-shield",
    study=belt_pull,
    metric="compliance",
    regularizer=manufacturing_penalty,
    regularizer_weight=1.0,
    steps=6,
    learning_rate=0.01,
)
