"""Parametric heat sink: finned extrusion, copper slug, press-fit bushings.

A compact power-module heat sink and the default tour of the toolchain: the
fin comb is one parameter-backed sketch profile extruded through a named
depth, the copper heat slug under the die is a revolved section, and two
steel bushings carry the mounting screws. A named SimMesh discretizes the
sink, the declared thermal study conducts the die's heat flux up into the
fins on it, and the single declared optimization at the bottom descends that
SAME simulation — mean temperature against a material-volume penalty —
differentiably, straight through the geometry the viewport renders.

Named design parameters:
  - ``fin_depth``: extrusion depth of the fin comb (along y)
  - ``base_width``: driving dimension across the base deck
  - ``bushing_spacing``: distance between the two mounting bushings

The comb sketch keeps exactly five meaningful design freedoms under its
constraints — fin depth, shared fin tip height, shared fin width, mirrored
outer-fin spacing, and deck thickness; everything else is a relation
(horizontal/vertical/equal/mirror) or pinned (base span, the slug's die
interface).
"""

import jax
import jax.numpy as jnp

from cadjoint import extract_parameters, functionalize
from cadjoint.constraints import (
    DistanceConstraint,
    EqualLengthConstraint,
    FixedConstraint,
    HorizontalConstraint,
    VerticalConstraint,
    satisfy_constraints,
)
from cadjoint.construction import PolygonProfile, SketchPlane, Solid, extrude, revolve
from cadjoint.fem import Dirichlet, HeatFlux, Nodes, SimMesh, ThermalStudy
from cadjoint.geometry import Scalar, Vector, Vector2
from cadjoint.optimize import Optimization
from cadjoint.render import Material
from cadjoint.sdf.boolean import Union

# ── design parameters ────────────────────────────────────────────────────────
# fin_depth is genuinely free (the optimizer moves it); the named scalars
# are driving dimensions: constraints hold the sketch to them, so editing a
# value here re-dimensions the part without freeing it to drift.
fin_depth = Scalar(1.2, free=True, name="fin_depth")
base_width = Scalar(1.8, name="base_width")
bushing_spacing = Scalar(1.56, name="bushing_spacing")

aluminum = Material(name="aluminum", color=[0.8, 0.82, 0.85], roughness=0.3, metallic=0.9)
copper = Material(name="copper", color=[0.9, 0.45, 0.22], roughness=0.18, metallic=0.95)
steel = Material(name="steel", color=[0.55, 0.57, 0.6], roughness=0.4, metallic=0.85)

# ── fin comb: base deck + three fins as one sketch profile ───────────────────
# Sketch plane normal +Y gives in-plane axes u = -X, v = +Z: profile y is
# world height. Extrusion spans ±fin_depth/2 around y = 0. Every vertex is a
# live sketch point — drag a fin tip in the viewport and rerun.
base_l = Vector2(value=[-0.9, 0.0], free=True, name="base_l")
base_r = Vector2(value=[0.9, 0.0], free=True, name="base_r")
deck_r = Vector2(value=[0.9, 0.18], free=True, name="deck_r")
fin1_root_r = Vector2(value=[0.68, 0.18], free=True, name="fin1_root_r")
fin1_tip_r = Vector2(value=[0.68, 0.85], free=True, name="fin1_tip_r")
fin1_tip_l = Vector2(value=[0.52, 0.85], free=True, name="fin1_tip_l")
fin1_root_l = Vector2(value=[0.52, 0.18], free=True, name="fin1_root_l")
fin2_root_r = Vector2(value=[0.08, 0.18], free=True, name="fin2_root_r")
fin2_tip_r = Vector2(value=[0.08, 0.85], free=True, name="fin2_tip_r")
fin2_tip_l = Vector2(value=[-0.08, 0.85], free=True, name="fin2_tip_l")
fin2_root_l = Vector2(value=[-0.08, 0.18], free=True, name="fin2_root_l")
fin3_root_r = Vector2(value=[-0.52, 0.18], free=True, name="fin3_root_r")
fin3_tip_r = Vector2(value=[-0.52, 0.85], free=True, name="fin3_tip_r")
fin3_tip_l = Vector2(value=[-0.68, 0.85], free=True, name="fin3_tip_l")
fin3_root_l = Vector2(value=[-0.68, 0.18], free=True, name="fin3_root_l")
deck_l = Vector2(value=[-0.9, 0.18], free=True, name="deck_l")
comb_profile = PolygonProfile(
    [
        base_l,
        base_r,
        deck_r,
        fin1_root_r,
        fin1_tip_r,
        fin1_tip_l,
        fin1_root_l,
        fin2_root_r,
        fin2_tip_r,
        fin2_tip_l,
        fin2_root_l,
        fin3_root_r,
        fin3_tip_r,
        fin3_tip_l,
        fin3_root_l,
        deck_l,
    ],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="fin comb",
)
sink = extrude(comb_profile, depth=fin_depth, material=aluminum)

# The comb is a production-style constrained sketch that still keeps real
# design freedom. Relations — horizontal/vertical squaring, equal fin
# widths, mirror symmetry about the base — shape HOW the comb may move;
# the pinned base span anchors it. Exactly four sketch freedoms survive
# (plus fin_depth): deck thickness (deck_r level), shared fin width,
# mirrored outer-fin spacing, and shared fin tip height. The optimizer
# below explores those and only those: every descent step is projected
# back onto this system (see cadjoint.optimize).
FixedConstraint(base_l, [-0.9, 0.0])
DistanceConstraint(base_l, base_r, base_width)
HorizontalConstraint(base_l, base_r)
VerticalConstraint(base_r, deck_r)
VerticalConstraint(base_l, deck_l)
HorizontalConstraint(deck_l, deck_r)
HorizontalConstraint(fin1_root_r, deck_r)
HorizontalConstraint(fin1_root_l, deck_r)
HorizontalConstraint(fin2_root_r, deck_r)
HorizontalConstraint(fin2_root_l, deck_r)
HorizontalConstraint(fin3_root_r, deck_r)
HorizontalConstraint(fin3_root_l, deck_r)
VerticalConstraint(fin1_root_r, fin1_tip_r)
VerticalConstraint(fin1_root_l, fin1_tip_l)
VerticalConstraint(fin2_root_r, fin2_tip_r)
VerticalConstraint(fin2_root_l, fin2_tip_l)
VerticalConstraint(fin3_root_r, fin3_tip_r)
VerticalConstraint(fin3_root_l, fin3_tip_l)
HorizontalConstraint(fin2_tip_r, fin2_tip_l)
HorizontalConstraint(fin2_tip_r, fin1_tip_r)
HorizontalConstraint(fin1_tip_r, fin1_tip_l)
HorizontalConstraint(fin2_tip_l, fin3_tip_r)
HorizontalConstraint(fin3_tip_r, fin3_tip_l)
EqualLengthConstraint(fin1_root_r, fin1_root_l, fin2_root_r, fin2_root_l)
EqualLengthConstraint(fin2_root_r, fin2_root_l, fin3_root_r, fin3_root_l)
EqualLengthConstraint(base_l, fin3_root_l, base_r, fin1_root_r)
EqualLengthConstraint(base_l, fin2_root_l, base_r, fin2_root_r)

# ── copper heat slug: revolved section under the die, screw bore on axis ─────
# Revolve spins the profile around the plane's local Y axis (world z here):
# profile x is radius, profile y runs along the axis. The slug presses into
# the deck from below; the die contacts its bottom face.
#
# Design rule: boundary-condition regions sit on pinned geometry. The slug
# bottom carries the die's heat-flux BC, so its outline is NOT free — the
# optimizer shapes fins and depth, never the chip interface. (The fin-top
# Dirichlet region below is anchored generously instead: its threshold sits
# far under the tips, so the height freedom cannot move the fins out of it.)
slug_bore_low = Vector2(value=[0.05, -0.18], name="slug_bore_low")
slug_rim_low = Vector2(value=[0.26, -0.18], name="slug_rim_low")
slug_rim_high = Vector2(value=[0.26, 0.04], name="slug_rim_high")
slug_bore_high = Vector2(value=[0.05, 0.04], name="slug_bore_high")
slug_profile = PolygonProfile(
    [slug_bore_low, slug_rim_low, slug_rim_high, slug_bore_high],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="slug section",
)
slug = revolve(slug_profile, material=copper)

# ── mounting bushings: fixed pattern, spacing tied by a constraint ───────────
bushing_a = Vector([0.78, 0.0, 0.1], free=True, name="bushing_a")
bushing_b = Vector([-0.78, 0.0, 0.1], free=True, name="bushing_b")
FixedConstraint(bushing_a, [0.78, 0.0, 0.1])
DistanceConstraint(bushing_a, bushing_b, bushing_spacing)
bush_a = Solid.cylinder(radius=0.07, height=0.12, position=bushing_a, material=steel, name="bush_a")
bush_b = Solid.cylinder(radius=0.07, height=0.12, position=bushing_b, material=steel, name="bush_b")

scene = Union(sink, slug, bush_a, bush_b, smoothness=0.03)
satisfy_constraints(scene, steps=2)

# ── simulation mesh: the sink volume on a named grid ─────────────────────────
# First-class meshing intent: the study below solves on it, the viewer
# inspects it (counts, bounds, element quality), and the optimization
# refreezes it as the design moves. method="tet10" is the quality path —
# boundary-conforming quadratic tets from the dual-contoured surface
# (method="hex" is the fast voxelize+snap alternative).
sink_mesh = SimMesh(
    name="sink-mesh",
    resolution=(18, 13, 11),
    bounds=(-1.05, -0.8, -0.3),
    size=(2.1, 1.6, 1.4),
    method="tet10",
)

# ── thermal study: die flux on the slug bottom, ambient at the fin field ─────
# Node selections are programmatic: the flux enters through the boundary
# faces of the slug's bottom disc; the upper fin field is held at ambient
# (an idealized convection sink).
heat_study = ThermalStudy(
    name="sink-conduction",
    conductivity=2.0,
    bcs=[
        HeatFlux(
            Nodes.halfspace([0.0, 0.0, -0.12], [0.0, 0.0, -1.0])
            & Nodes.sphere([0.0, 0.0, -0.18], 0.4),
            6.0,
        ),
        Dirichlet(Nodes.halfspace([0.0, 0.0, 0.45], [0.0, 0.0, 1.0]), 0.0),
    ],
    mesh=sink_mesh,
)

# The regularizer is a real reverse-mode derivative path through sketch
# points -> extrusion -> final SDF evaluation: the (smoothed) aluminum
# volume of the fin comb as a function of the free parameters above.
sink_parameters, sink_fixed, _ = extract_parameters(sink)
sink_sdf = functionalize(sink)

axes = [jnp.linspace(-1.0, 1.0, 15), jnp.linspace(-0.7, 0.7, 15), jnp.linspace(-0.05, 0.95, 15)]
cells = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
cell_volume = float((2.0 / 14) * (1.4 / 14) * (1.0 / 14))


def material_volume(parameters):
    sdf = sink_sdf(parameters, sink_fixed)
    return cell_volume * jnp.sum(jax.nn.sigmoid(-sdf(cells) / 0.03))


# The declared optimization closes the loop end to end: the thermal study
# above becomes the objective. Per step, the frozen simulation mesh follows
# the design differentiably (node positions re-projected through the traced
# SDF), the study solves on it, and the mean temperature descends against
# the material-volume regularizer while every update projects back onto the
# sketch constraints — run it from the viewer and the optimized part
# arrives with its temperature field attached, values written back here.
cool_sink = Optimization(
    name="cool-sink",
    study="sink-conduction",
    metric="mean",
    regularizer=material_volume,
    regularizer_weight=0.4,
    steps=12,
    learning_rate=0.01,
)
