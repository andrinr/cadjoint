"""Parametric L-bracket: base plate, bolt holes, tapered web, gusset rib.

A realistic mounting bracket built from the construction API. The base plate
and the vertical web are extrusions of parameter-backed sketch profiles, the
stiffening rib is a triangular gusset tying the web to the plate, and the two
bolt holes are subtracted cylinders whose positions are pinned by constraints.

Named design parameters (drive the FEM optimization in
``examples/fem_bracket_optimization.py``):
  - ``plate_thickness``: extrusion depth of the base plate
  - ``web_thickness``: extrusion depth of the vertical web
  - ``rib_height``: distance constraint from the rib's plate corner to its tip
"""

from jaxcad.constraints import DistanceConstraint, FixedConstraint, satisfy_constraints
from jaxcad.construction import PolygonProfile, SketchPlane, Solid, extrude
from jaxcad.fem import ElasticStudy, Fixed, Nodes, Traction
from jaxcad.geometry import Scalar, Vector, Vector2
from jaxcad.render import Material
from jaxcad.sdf.boolean import Difference, Union

# ── design parameters ────────────────────────────────────────────────────────
plate_thickness = Scalar(0.2, free=True, name="plate_thickness")
web_thickness = Scalar(0.16, free=True, name="web_thickness")
rib_height = Scalar(0.88, name="rib_height")
bolt_spacing = Scalar(1.4, name="bolt_spacing")

steel = Material(name="steel", color=[0.62, 0.66, 0.72], roughness=0.35, metallic=0.85)

# ── base plate: rectangle sketch extruded through plate_thickness ────────────
# The plate sits on z = 0; its mid-plane is the sketch plane at z = 0.1.
plate_bl = Vector2(value=[-1.2, -0.8], free=True, name="plate_bl")
plate_br = Vector2(value=[1.2, -0.8], free=True, name="plate_br")
plate_tr = Vector2(value=[1.2, 0.8], free=True, name="plate_tr")
plate_tl = Vector2(value=[-1.2, 0.8], free=True, name="plate_tl")
plate_profile = PolygonProfile(
    [plate_bl, plate_br, plate_tr, plate_tl],
    plane=SketchPlane(origin=[0.0, 0.0, 0.1], normal=[0.0, 0.0, 1.0]),
    name="plate",
)
plate = extrude(plate_profile, depth=plate_thickness, material=steel)

# ── vertical web: tapered wall on the back edge (y = -0.7) ───────────────────
# Sketch plane normal +Y gives in-plane axes u = -X, v = +Z: profile x runs
# along -world-x, profile y is world height. The root overlaps the plate.
web_root_right = Vector2(value=[-1.1, 0.0], free=True, name="web_root_right")
web_root_left = Vector2(value=[1.1, 0.0], free=True, name="web_root_left")
web_top_left = Vector2(value=[0.85, 1.2], free=True, name="web_top_left")
web_top_right = Vector2(value=[-0.85, 1.2], free=True, name="web_top_right")
web_profile = PolygonProfile(
    [web_root_right, web_root_left, web_top_left, web_top_right],
    plane=SketchPlane(origin=[0.0, -0.7, 0.0], normal=[0.0, 1.0, 0.0]),
    name="web",
)
web = extrude(web_profile, depth=web_thickness, material=steel)

# ── stiffening rib: triangular gusset from plate to web, centered in x ───────
# Sketch plane normal +X gives u = -Z, v = +Y: profile (x, y) = (-z, y) world.
rib_plate_corner = Vector2(value=[-0.02, 0.55], free=True, name="rib_plate_corner")
rib_web_corner = Vector2(value=[-0.02, -0.62], free=True, name="rib_web_corner")
rib_tip = Vector2(value=[-0.9, -0.62], free=True, name="rib_tip")
rib_profile = PolygonProfile(
    [rib_plate_corner, rib_web_corner, rib_tip],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0]),
    name="rib",
)
rib = extrude(rib_profile, depth=0.12, material=steel)

# The rib height is a named dimension: the tip must sit rib_height away from
# the web-side base corner, straight up the web.
DistanceConstraint(rib_web_corner, rib_tip, rib_height)

# ── bolt holes: fixed positions, spacing tied by a distance constraint ───────
bolt_left = Vector([-0.7, 0.35, 0.1], free=True, name="bolt_left")
bolt_right = Vector([0.7, 0.35, 0.1], free=True, name="bolt_right")
FixedConstraint(bolt_left, [-0.7, 0.35, 0.1])
DistanceConstraint(bolt_left, bolt_right, bolt_spacing)

hole_left = Solid.cylinder(radius=0.16, height=0.4, position=bolt_left, name="hole_left")
hole_right = Solid.cylinder(radius=0.16, height=0.4, position=bolt_right, name="hole_right")

# ── assembly: filleted union of the structure, sharp-ish bolt holes ──────────
body = Union(plate, web, rib, smoothness=0.05)
scene = Difference(body, hole_left, hole_right, smoothness=0.02)
satisfy_constraints(scene, steps=2)

# ── simulation study: bolt regions clamped, prying load on the web tip ───────
# Node selections are programmatic: a ball around each bolt hole is fixed,
# and the traction acts on the boundary faces spanned by the outer (-y) web
# wall above z = 1.0 (mirrors examples/fem_bracket_optimization.py).
pry_study = ElasticStudy(
    name="bracket-pry",
    resolution=(24, 17, 13),
    youngs=1000.0,
    poisson=0.3,
    bcs=[
        Fixed(Nodes.sphere([-0.7, 0.35, 0.1], 0.33) | Nodes.sphere([0.7, 0.35, 0.1], 0.33)),
        Traction(
            Nodes.halfspace([0.0, -0.7, 0.0], [0.0, -1.0, 0.0])
            & Nodes.halfspace([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
            (0.0, -2.0, 0.0),
        ),
    ],
    bounds=(-1.3, -0.95, -0.06),
    size=(2.6, 1.9, 1.42),
)
