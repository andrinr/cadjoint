"""Inline-four cylinder block: bores, water jacket, bulkheads, skirt.

A cast-iron engine block for a ~1.5 L inline four, reverse-engineered *in
kind* from the M15 block on GrabCAD.  It is one casting doing five jobs, and
each one is a different modelling idea:

  * **hold the bores.** Four barrels on a fixed pitch, cut as one
    ``Face.hole`` from the deck and patterned — not four hand-placed
    cylinders.
  * **carry coolant.** A water jacket that is a *cored void*: the pocket
    inside the outer wall, minus the barrel columns, minus the head-bolt
    bosses.  The bosses are not modelled as features at all; they are what
    the jacket cut leaves standing.
  * **carry the crank.** Five main-bearing bulkheads on the same pitch as
    the bores, their saddles opened by one crank tunnel bored the length of
    the block, with the parting face on the crank axis.
  * **close the crankcase.** A skirt that flares from the deck width down to
    the pan rail — a ``loft``, because the taper is what clears the
    counterweights — and a pan-rail flange sketched on the skirt's own
    underside.
  * **bolt the head down.** Ten head-bolt holes, two per bore boundary,
    drilled down the bosses the jacket left behind.

**What came from the reference and what did not.**  Nothing came from the
reference.  ``grabcad.com/library/m15-engine-cylindreblock-1`` answers a
plain fetch with a 403 and, with a browser user-agent, with an empty
single-page-app shell; its public JSON endpoints return 500 and 404.  The
model itself is behind a login, which was not attempted.  So **every
dimension below is assumed**, from published class proportions for a 1.5 L
four, and only one of them is corroborated by a public source at all: the
Suzuki M15A — the engine an "M15" of this displacement most likely is — is
published as 78.0 mm bore x 78.0 mm stroke, 1490 cc, and the bore diameter
here is that 78 mm.  Bore pitch (92 mm), deck height (212 mm above the crank
axis), block length (410 mm), deck width (152 mm), pan-rail width (208 mm),
wall thickness (13 mm), barrel outside diameter (88 mm), main saddle radius
(27 mm) and every bolt size are **class-typical numbers chosen here**, not
measurements.  A block drawn from these will look right and will not
interchange with anything.

Named design parameters:
  - ``bore_radius``: the bore.  Free, and the one parameter with a real
    engineering edge to it — the barrel outside diameter is fixed by the
    casting core, so opening the bore thins the barrel wall rather than
    growing the casting.  That is exactly what limits an overbore.
  - ``jacket_depth``: how far the water jacket reaches below the deck.  Free.
    The extrusion straddles its plane, so the floor drops by half of any
    change while the top rises further above a deck it already clears.
  - ``main_saddle_radius``: the bored radius of the main bearing tunnel —
    journal plus shell.  Free.
  - ``bulkhead_thickness``: how thick a main-bearing web is.  Free, and the
    weight-against-stiffness trade a block actually argues about.
  - ``block_height``: the casting's total height, pan rail to deck.  A
    driving dimension, **not** free, and the reason is worth stating.
    Re-run this program with a different value and the whole part follows:
    the skirt, the pan-rail flange, the bores, the jacket and the ten head
    bolts are all sketched on a *face* of this extrusion, and a face is
    re-derived from the depth every time the file executes.  The frozen
    functional form is another matter — it substitutes parameters into the
    nodes that hold them, and a plane origin derived from a face is a
    number by then.  Handing this to an optimizer would therefore raise the
    deck and leave the bores behind.  Every free parameter above is one no
    other feature is derived from, which is the property that makes it safe
    to optimize; see ``research/complex-scene.md``.
  - ``bore_pitch``: driving dimension.  It is the spacing of *three*
    patterns at once — the bores, the bulkheads and the head-bolt columns —
    which is the whole reason a block's bore pitch is the number an engine
    family is designed around.  Named rather than free: a pattern's spacing
    is live but its seed is a computed number, so freeing it would walk
    three bores away from a first one that stayed put.
  - ``saddle_pad_width``: the width of a bulkhead's main-cap parting face,
    held by a constraint on the one sketch in the file with real freedom.

**Open deck, deliberately.**  The jacket pocket's top sits 6 mm *above* the
deck face, so the jacket is open where the head gasket lands.  A cast-iron
block of this class is usually closed-deck with small transfer holes; open
deck was chosen because it is the honest way to make the jacket visible —
a closed deck renders as a flat slab and hides whether the jacket cut
worked at all.

**Scale.**  One scene unit is 200 mm, so the whole casting sits inside the
+-2.5 box that ``tests/zeroset`` and ``tests/backends`` sample.  Every
number below is written in millimetres and multiplied by ``MM``, so the
prose above and the source agree digit for digit.

**Segment counts are a budget.**  Two segments per rounded corner, twelve
vertices per rectangle: the viewer's edge overlay costs roughly linearly in
the scene's total vertex count, and eight rounded rectangles at three
segments each would have put this file over what the end-cap already
measured as the affordable ceiling.

**What this block does not have.**  Five features are modelled properly and
the rest are named here rather than half-drawn.  Missing: the cam drive — no
cam galleries, no chain case, no tunnel bosses (this is drawn as an OHC
block, so the cams live in the head, but the drive end wall is still bare);
the whole oil system — no main gallery down the cam side, no drillings from
the gallery to the saddles, no lifter feeds, no pump mounting; coolant
transfer holes through the deck into the head, which an open deck of this
shape would need at the bore boundaries; freeze plugs and their core holes;
the bellhousing bolt pattern and the starter aperture on the rear face; the
motor-mount and accessory bosses on the flanks — which is why the side
elevation renders as a plain slab, and it should; dipstick and breather
bosses; a knock-sensor pad; the small draft angle a real casting carries on
every wall; and thread reliefs and counterbores at every tapped hole.  The
main caps are drawn as plain blocks with no register step and no bolts.

**No simulation study.**  A block's real load case is head-bolt preload plus
gas load reacted through the bulkheads into the main caps, on a mesh fine
enough to resolve a 5 mm barrel wall.  Declaring a study that could not
carry that would be worse than declaring none, so this file declares none
and ends with a volume functional instead — the same end-to-end
differentiability proof ``scenes/end_cap.py`` ends with.
"""

import math

import jax
import jax.numpy as jnp

from cadjoint import extract_parameters, functionalize
from cadjoint.constraints import (
    DistanceConstraint,
    FixedConstraint,
    VerticalConstraint,
    satisfy_constraints,
)
from cadjoint.construction import PolygonProfile, SketchPlane, Solid, extrude, loft
from cadjoint.geometry import Scalar, Vector, Vector2
from cadjoint.render import Material
from cadjoint.sdf.boolean import Difference, Union
from cadjoint.sdf.transforms.fields import Mirror
from cadjoint.sdf.transforms.patterns import LinearPattern

# ── scale ────────────────────────────────────────────────────────────────────
# One unit is 200 mm. Everything below is millimetres times MM.
MM = 1.0 / 200.0

# ── the casting's fixed proportions ──────────────────────────────────────────
# Datum: z = 0 is the CRANK AXIS and the main-bearing parting plane, x runs
# along the crank (cylinder 1 at -x), y is across the block. Every height in
# this file reads as "above the crank axis", which is how a block is drawn.
BLOCK_LENGTH = 410.0  # over the end walls
DECK_WIDTH = 152.0  # across the deck face
PAN_RAIL_WIDTH = 208.0  # across the skirt at the pan rail
SKIRT_DEPTH = 78.0  # crank axis down to the pan rail
DECK_Z = 212.0  # crank axis up to the deck face
WALL = 13.0  # nominal outer wall
# The body extrusion straddles its plane, so the plane sits at the casting's
# own mid-height and `block_height` below is the full span.
BODY_PLANE_Z = (DECK_Z - SKIRT_DEPTH) / 2.0  # 67.0

BORE_COUNT = 4
MAIN_COUNT = 5  # an inline four has a main bearing between and outside every bore
BARREL_RADIUS = 44.0  # the casting core: bore + 5 mm of barrel wall at nominal
BARREL_BOTTOM_Z = 50.0  # how far the barrels hang into the crankcase
CRANKCASE_CEILING_Z = 70.0  # underside of the jacket floor web
JACKET_DEPTH = 128.0  # nominal; the free `jacket_depth` below carries it
HEAD_BOLT_Y = 58.0  # just outboard of the barrel, inboard of the deck edge
HEAD_BOLT_BOSS_RADIUS = 12.5
MAIN_BOLT_Y = 45.0  # clear of the tunnel, inside the parting pad
CAST_FILLET = 4.0  # smooth-union radius where webs meet walls
EDGE_BREAK = 1.2  # smooth-difference radius on the cuts

# ── design parameters ────────────────────────────────────────────────────────
# The four free ones are handed straight to the feature they dimension, which
# is what keeps them live in the gradient graph.
bore_radius = Scalar(39.0 * MM, free=True, name="bore_radius")
jacket_depth = Scalar(JACKET_DEPTH * MM, free=True, name="jacket_depth")
main_saddle_radius = Scalar(27.0 * MM, free=True, name="main_saddle_radius")
bulkhead_thickness = Scalar(20.0 * MM, free=True, name="bulkhead_thickness")
# Driving dimensions: live values, held rather than optimized. `block_height`
# is here rather than above because five features are sketched on this
# extrusion's faces — see the note in the module docstring.
block_height = Scalar((DECK_Z + SKIRT_DEPTH) * MM, name="block_height")
bore_pitch = Scalar(92.0 * MM, name="bore_pitch")
barrel_radius = Scalar(BARREL_RADIUS * MM, name="barrel_radius")
head_bolt_radius = Scalar(5.5 * MM, name="head_bolt_radius")
main_bolt_radius = Scalar(6.0 * MM, name="main_bolt_radius")
saddle_pad_width = Scalar(140.0 * MM, name="saddle_pad_width")

BORE_PITCH = float(bore_pitch.value) / MM  # 92.0, in mm, for the pattern seeds
# Copy 0 of a LinearPattern is the seed and the rest march in +x, so every
# pattern in this file starts at the -x end of its own row.
FIRST_BORE_X = -1.5 * BORE_PITCH  # -138
FIRST_MAIN_X = -2.0 * BORE_PITCH  # -184

cast_iron = Material(
    name="cast iron",
    color=[0.36, 0.36, 0.39],
    roughness=0.72,
    metallic=0.55,
    density=7200.0,
    conductivity=52.0,
    specific_heat=460.0,
    youngs_modulus=110e9,
    poisson_ratio=0.26,
    yield_strength=250e6,
    thermal_expansion=11.0e-6,
)
nodular_iron = Material(
    name="nodular iron",
    color=[0.24, 0.25, 0.28],
    roughness=0.55,
    metallic=0.7,
    density=7100.0,
    conductivity=33.0,
    specific_heat=460.0,
    youngs_modulus=169e9,
    poisson_ratio=0.28,
    yield_strength=420e6,
)

# ── 1. the blank: one extrusion from the pan rail to the deck ────────────────
# The deck outline is the block in plan; the skirt's flare is added below.
# `block_height` is the depth, so `body.cap("+")` IS the deck face and moves
# with it — which is why every feature further down is sketched on that face
# rather than at the number 212.
body_profile = PolygonProfile.rounded_rect(
    BLOCK_LENGTH * MM,
    DECK_WIDTH * MM,
    18.0 * MM,
    segments=2,
    plane=SketchPlane(origin=[0.0, 0.0, BODY_PLANE_Z * MM]),
    name="block outline",
)
body = extrude(body_profile, depth=block_height, material=cast_iron)
deck = body.cap("+")
pan_rail = body.cap("-")

# ── 2. crankcase skirt: a loft, because the taper is the point ───────────────
# The skirt flares from the deck width at the crank axis out to the pan rail,
# which is what gives the counterweights room to swing. A loft pairs vertex i
# of one profile with vertex i of the other, so both outlines come from the
# same generator at the same segment count. `pan_rail`'s normal points DOWN,
# so profile A (at -height/2 along that normal) is the upper, narrower one.
skirt_top_profile = PolygonProfile.rounded_rect(
    BLOCK_LENGTH * MM,
    DECK_WIDTH * MM,
    18.0 * MM,
    segments=2,
    plane=pan_rail.plane(offset=-SKIRT_DEPTH / 2.0 * MM),
    name="skirt top",
)
skirt_foot_profile = PolygonProfile.rounded_rect(
    BLOCK_LENGTH * MM, PAN_RAIL_WIDTH * MM, 24.0 * MM, segments=2, name="skirt foot"
)
skirt = loft(skirt_top_profile, skirt_foot_profile, height=SKIRT_DEPTH * MM, material=cast_iron)

# ── 3. pan rail flange: sketched on the blank's own underside ────────────────
# 5 mm proud of the skirt all round — the lip the sump bolts to.
flange_profile = PolygonProfile.rounded_rect(
    (BLOCK_LENGTH + 10.0) * MM,
    (PAN_RAIL_WIDTH + 10.0) * MM,
    24.0 * MM,
    segments=2,
    plane=pan_rail.plane(offset=-7.0 * MM),
    name="pan rail",
)
pan_flange = extrude(flange_profile, depth=14.0 * MM, material=cast_iron)

blank = Union(body, skirt, pan_flange, smoothness=CAST_FILLET * MM)

# ── 4. the barrels: one hole from the deck, patterned four ways ──────────────
# `Face.hole` returns the TOOL, not a cut solid, which is what lets this one
# column be subtracted from two different bodies below — from the crankcase
# cavity and from the water jacket, where in both cases it leaves the barrel
# standing — and never from the block itself. `through` reaches above the
# deck so the tool still covers the open-deck jacket pocket, whose top is
# proud of the deck.
barrel = deck.hole(
    barrel_radius,
    depth=(DECK_Z - BARREL_BOTTOM_Z) * MM,
    at=(FIRST_BORE_X * MM, 0.0),
    through=10.0 * MM,
)
barrels = LinearPattern(barrel, direction=[1.0, 0.0, 0.0], count=BORE_COUNT, spacing=bore_pitch)

# ── 5. crankcase cavity: cored around the barrels ────────────────────────────
# Two pieces because the outer form is two pieces: a lofted lower half that
# follows the skirt's flare inward at a constant 13 mm wall, and a prismatic
# upper half under the constant-width deck. They overlap by 4 mm in z so the
# join is a real intersection rather than two coincident planes.
# Both are cut back by the barrels, which is what leaves the barrel walls
# standing in the middle of an otherwise open crankcase.
skirt_void_top = PolygonProfile.rounded_rect(
    (BLOCK_LENGTH - 2 * WALL) * MM,
    (DECK_WIDTH - 2 * WALL) * MM,
    12.0 * MM,
    segments=2,
    # The pan rail's normal points down, so a negative offset lifts the plane
    # into the block: 33 mm up from the pan rail puts it halfway between the
    # cavity's own ends at +2 and -92.
    plane=pan_rail.plane(offset=-33.0 * MM),
    name="crankcase top",
)
skirt_void_foot = PolygonProfile.rounded_rect(
    (BLOCK_LENGTH - 2 * WALL) * MM,
    (PAN_RAIL_WIDTH - 2 * WALL) * MM,
    18.0 * MM,
    segments=2,
    name="crankcase foot",
)
# 94 mm tall, from +2 down to -92: it runs past the pan rail so the crankcase
# is open at the bottom rather than closed by a skin nothing would ever cut.
skirt_void = loft(skirt_void_top, skirt_void_foot, height=94.0 * MM)

bay_void_profile = PolygonProfile.rounded_rect(
    (BLOCK_LENGTH - 2 * WALL) * MM,
    (DECK_WIDTH - 2 * WALL) * MM,
    12.0 * MM,
    segments=2,
    plane=SketchPlane(origin=[0.0, 0.0, (CRANKCASE_CEILING_Z - 2.0) / 2.0 * MM]),
    name="bore bay",
)
bay_void = extrude(bay_void_profile, depth=(CRANKCASE_CEILING_Z + 2.0) * MM)

# Every boolean in this file states its blend, including the sharp ones.
# `smoothness` defaults to 0.1 — twenty millimetres at this scale — and a
# core tool blended by twenty millimetres is a core tool that has eaten the
# feature it was meant to define.
crankcase = Difference(
    Union(skirt_void, bay_void, smoothness=CAST_FILLET * MM), barrels, smoothness=0.0
)
hollow = Difference(blank, crankcase, smoothness=EDGE_BREAK * MM)

# ── 6. main bearing bulkheads: the one sketch with design freedom ────────────
# Sketch plane normal +X gives in-plane axes u = -Z, v = +Y, so profile x is
# world height measured DOWNWARD and profile y is world y. The pads at
# profile x = 0 are therefore on the crank axis, which is the main-cap
# parting plane: that is the modelling statement this whole feature makes.
bulkhead_pad_near = Vector2(value=[0.0, -70.0 * MM], free=True, name="bulkhead_pad_near")
bulkhead_pad_far = Vector2(value=[0.0, 70.0 * MM], free=True, name="bulkhead_pad_far")
bulkhead_crown_far = Vector2(value=[-74.0 * MM, 58.0 * MM], free=True, name="bulkhead_crown_far")
bulkhead_crown_near = Vector2(value=[-74.0 * MM, -58.0 * MM], free=True, name="bulkhead_crown_near")
bulkhead_profile = PolygonProfile(
    [bulkhead_pad_near, bulkhead_pad_far, bulkhead_crown_far, bulkhead_crown_near],
    plane=SketchPlane(origin=[FIRST_MAIN_X * MM, 0.0, 0.0], normal=[1.0, 0.0, 0.0]),
    name="main bulkhead",
)
bulkhead = extrude(bulkhead_profile, depth=bulkhead_thickness, material=cast_iron)

# What the constraints say is what the bulkhead IS: a parting pad lying flat
# on the crank axis, a named width across it, and a flat crown above. The
# web's taper — how far in the crown pulls from the pad — is left free, which
# is the freedom a bulkhead actually has once the saddle is fixed.
FixedConstraint(bulkhead_pad_near, [0.0, -70.0 * MM])
VerticalConstraint(bulkhead_pad_near, bulkhead_pad_far)
DistanceConstraint(bulkhead_pad_near, bulkhead_pad_far, saddle_pad_width)
VerticalConstraint(bulkhead_crown_near, bulkhead_crown_far)

bulkheads = LinearPattern(bulkhead, direction=[1.0, 0.0, 0.0], count=MAIN_COUNT, spacing=bore_pitch)

# The blend here is a real casting radius, not decoration: 4 mm against a
# 20 mm web is a fifth of its thickness, which is what a foundry would put
# in the corner where a bulkhead meets the skirt wall.
structure = Union(hollow, bulkheads, smoothness=CAST_FILLET * MM)

# ── 7. the water jacket: a void defined by what it is NOT ────────────────────
# The pocket inside the outer wall, minus the barrels, minus ten bolt
# columns. Nothing here models a boss: the bosses are the material the cut
# steps around, which is how a core box works and how the jacket stays one
# connected passage rather than ten pockets.
# The plane offset is a number, so `jacket_depth` moves the floor and not the
# whole pocket: half of any change goes down into the block, half goes up
# into air the deck already clears.
jacket_profile = PolygonProfile.rounded_rect(
    (BLOCK_LENGTH - 2 * WALL) * MM,
    (DECK_WIDTH - 2 * WALL) * MM,
    12.0 * MM,
    segments=2,
    # Half the nominal depth below the deck, less the 6 mm the pocket stands
    # proud of it: an extrusion straddles its plane, so this is what puts the
    # jacket's top above the deck face and makes the deck an open one.
    plane=deck.plane(offset=(6.0 - JACKET_DEPTH / 2.0) * MM),
    name="water jacket",
)
jacket_pocket = extrude(jacket_profile, depth=jacket_depth)

# One boss, mirrored across the block's longitudinal midplane and then walked
# along the bore pitch: two per bore boundary, ten in all, which is the head
# bolt count a four of this class actually uses.
bolt_boss = deck.hole(
    HEAD_BOLT_BOSS_RADIUS * MM,
    depth=140.0 * MM,
    at=(FIRST_MAIN_X * MM, -HEAD_BOLT_Y * MM),
    through=10.0 * MM,
)
bolt_boss_pair = Union(bolt_boss, Mirror(bolt_boss, "y"), smoothness=0.0)
bolt_bosses = LinearPattern(
    bolt_boss_pair, direction=[1.0, 0.0, 0.0], count=MAIN_COUNT, spacing=bore_pitch
)

jacket = Difference(jacket_pocket, barrels, bolt_bosses, smoothness=0.0)

# ── 8. the cuts ──────────────────────────────────────────────────────────────
# The bores share `bore_radius` with nothing else, so opening them eats into
# the barrel wall the fixed `barrel_radius` core left.
bore = deck.hole(
    bore_radius, depth=(DECK_Z - 46.0) * MM, at=(FIRST_BORE_X * MM, 0.0), through=6.0 * MM
)
bores = LinearPattern(bore, direction=[1.0, 0.0, 0.0], count=BORE_COUNT, spacing=bore_pitch)

# 132 mm, not deeper: the hole has to bottom out *inside* the jacket floor
# web, between the jacket at 90 and the crankcase ceiling at 70. A head bolt
# that broke through into the crankcase would drain the sump past its threads.
head_bolt = deck.hole(
    head_bolt_radius,
    depth=132.0 * MM,
    at=(FIRST_MAIN_X * MM, -HEAD_BOLT_Y * MM),
    through=6.0 * MM,
)
head_bolt_pair = Union(head_bolt, Mirror(head_bolt, "y"), smoothness=0.0)
head_bolts = LinearPattern(
    head_bolt_pair, direction=[1.0, 0.0, 0.0], count=MAIN_COUNT, spacing=bore_pitch
)

# One bore down the length of the block opens all five saddles at once, which
# is how a line bore is actually cut. Position, length and inclination are
# pinned; only the radius is a design variable.
crank_tunnel = Solid.cylinder(
    radius=main_saddle_radius,
    height=Scalar((BLOCK_LENGTH + 60.0) / 2.0 * MM, free=False),
    position=Vector([0.0, 0.0, 0.0], free=False),
    rotation=[Scalar(0.0, free=False), Scalar(math.pi / 2.0, free=False), Scalar(0.0, free=False)],
    name="crank_tunnel",
)

main_bolt = Solid.cylinder(
    radius=main_bolt_radius,
    height=Scalar(33.0 * MM, free=False),
    position=Vector([FIRST_MAIN_X * MM, -MAIN_BOLT_Y * MM, 27.0 * MM], free=False),
    name="main_bolt",
)
main_bolt_pair = Union(main_bolt, Mirror(main_bolt, "y"), smoothness=0.0)
main_bolts = LinearPattern(
    main_bolt_pair, direction=[1.0, 0.0, 0.0], count=MAIN_COUNT, spacing=bore_pitch
)

block = Difference(
    structure, jacket, bores, head_bolts, main_bolts, crank_tunnel, smoothness=EDGE_BREAK * MM
)
block.name = "block"

# ── 9. main bearing caps: rendered, not part of the casting ──────────────────
# Context, and the reason the crank axis reads as an axis in a screenshot.
# They are bored by the SAME tunnel solid the block is — one subtree, emitted
# once — which is what makes the cap bore and the saddle concentric by
# construction rather than by two matching numbers.
cap_profile = PolygonProfile(
    [
        [2.0 * MM, -50.0 * MM],
        [46.0 * MM, -50.0 * MM],
        [46.0 * MM, 50.0 * MM],
        [2.0 * MM, 50.0 * MM],
    ],
    plane=SketchPlane(origin=[FIRST_MAIN_X * MM, 0.0, 0.0], normal=[1.0, 0.0, 0.0]),
    name="main cap",
    free=False,
)
cap = extrude(cap_profile, depth=26.0 * MM, material=nodular_iron)
caps = Difference(
    LinearPattern(cap, direction=[1.0, 0.0, 0.0], count=MAIN_COUNT, spacing=bore_pitch),
    crank_tunnel,
    smoothness=0.0,
)

scene = Union(block, caps, smoothness=0.0)
satisfy_constraints(scene, steps=2)

# ── traceability: the iron volume as a function of the parameters ────────────
# Not an optimization — a proof that the casting is still differentiable end
# to end after ten booleans nested four deep. `tests/scenes/test_cylinder_block.py`
# finite-difference checks d(volume)/d(bore_radius) against this exact
# reverse-mode gradient, and the sign is the interesting part: opening the
# bores can only remove iron.
block_parameters, block_fixed, _ = extract_parameters(block)
block_sdf = functionalize(block)

_axes = [
    jnp.linspace(-1.10, 1.10, 25),
    jnp.linspace(-0.58, 0.58, 15),
    jnp.linspace(-0.42, 1.10, 21),
]
volume_cells = jnp.stack(jnp.meshgrid(*_axes, indexing="ij"), axis=-1).reshape(-1, 3)
volume_cell = float((2.20 / 24) * (1.16 / 14) * (1.52 / 20))


def block_volume(parameters):
    """Smoothed iron volume of the casting, differentiable in `parameters`."""
    sdf = block_sdf(parameters, block_fixed)
    return volume_cell * jnp.sum(jax.nn.sigmoid(-sdf(volume_cells) / 0.02))
