# Reverse-engineering a scene from a STEP

Two tools and one shared primitive, for the job of turning somebody else's
production CAD file into a cadjoint scene and then finding out how close you
got.

| file | what it does |
| --- | --- |
| `step_probe.py` | reads a STEP, reports the face census, and derives the part's frame and dimensions |
| `sections.py` | draws the casting's silhouette and the model's on the same plane, and scores each plane |
| `overlay.py` | scores the whole scene against the STEP: IoU, where it differs, how far, and a picture |
| `render.py` | side-by-side orthographic render of both, on the CPU, lit identically |
| `mesh_query.py` | axis-aligned ray casting on a triangle soup, with no optional dependencies |

They live under `research/` and not in `cadjoint/` because they need
[OCP](https://github.com/CadQuery/OCP), OpenCascade's Python bindings — a
hundred-megabyte dependency that nothing in the package itself wants. Nothing
in `cadjoint/`, `scenes/` or `tests/` imports any of this; a scene built with
it still builds, renders and tests on a machine that has neither OCP nor the
STEP file.

```
pip install cadquery-ocp        # only to run these tools
```

## The idea

A production STEP is not a picture of a part, it is a **drawing**. Everything a
machinist dimensioned is still an analytic surface in the file: a bore is a
`GeomAbs_Cylinder` with an exact radius and an exact axis, a deck is a
`GeomAbs_Plane` with an exact offset. So the bore pitch is not something you
infer from a rendering — it is the distance between two axis lines, and it
comes out exact.

What the faces cannot tell you is anything the casting only *implies*: a wall
thickness, the height of a jacket floor, whether a saddle was line-bored or
left as cast. Those come from the second half of the tool: tessellate the
solid and cast axis-aligned rays through it, reading the solid intervals off
each line. `step_probe` marks which of the two a number came from, because a
radius off a B-rep face is exact and a thickness off a 0.35 mm tessellation is
not.

## Using it

```bash
# what is this part?
python -m research.reverse.step_probe "/path/to/part.stp" \
    --mesh-cache /tmp/part.npz --json /tmp/part.json

# how close is my scene to it?
python -m research.reverse.overlay --scene scenes.cylinder_block \
    --mesh-cache /tmp/part.npz --step-mm 3.0 --png /tmp/overlay.png
```

The overlay reads the world-to-scene transform from a `REFERENCE` dict on the
scene module, so a scene authored on a sane origin can be compared against a
CAD part parked wherever its author left it:

```python
REFERENCE = {
    "step": "/path/to/part.stp",
    "unit_mm": 200.0,                       # one scene unit
    "origin_mm": [826.4, 134.0, 210.03],    # the STEP point at the scene origin
    "axes": [[0, 1, 0], [1, 0, 0], [0, 0, 1]],   # scene x, y, z in world terms
}
```

## Sections, not the scalar

`overlay.py` tells you a model disagrees. It cannot tell you *how*, and the
scalar is a bad thing to optimise directly — it will happily reward a model
that is uniformly slightly fat over one that has the right shape.
`sections.py` is the tool that actually drives the work: it puts the two
silhouettes on the same axes at the heights where the part changes character,
and the answer reads off the picture.

```bash
python -m research.reverse.sections --scene scenes.cylinder_block \
    --mesh-cache /tmp/part.npz --png sections.png
```

Every edit to `scenes/cylinder_block.py` since it was first measured came from
looking at one of those panels. Two of them were reverted on the evidence of
the same table, which is the other half of what the tool is for: a feature
that stops paying says so.

## A picture without a GPU

`render.py` sphere-traces the scene and ray-casts the casting from the same
orthographic camera, and shades both from their own depth buffers so neither
is flattered:

```bash
python -m research.reverse.render --scene scenes.cylinder_block \
    --mesh-cache /tmp/part.npz --azimuth 35 --elevation 22 --png compare.png
```

## Reading the score

The overlay reports IoU **and a ceiling**: the IoU of the casting against
itself displaced by a millimetre and by two. That is not decoration. A lattice
IoU is dominated by boundary cells, and a cylinder block's mean wall is
`2V/A` ≈ 6.6 mm, so displacing the real part by 1 mm already costs it a fifth
of its own score:

```
IoU              0.3629
  ceiling        0.7919   the casting against itself displaced 1mm
  ceiling        0.6298   the casting against itself displaced 2mm
```

Read alongside the surface deviation — median |d| ≈ 2 mm, three quarters of
the surface within 5 mm — the two say the same thing from different
directions, and neither says it on its own.

`missing` and `added` are kept apart deliberately. Iron the model has and the
casting does not is a fillet you invented or a passage you failed to core;
iron the casting has and the model does not is a boss, a rib or a gallery
nobody drew. They call for opposite fixes and one number hides which.

## What is worth knowing before you use these

* **The parity test is a mitigation, not a proof.** `inside_lattice` decides a
  whole column of samples from one ray, jittered off the lattice by 1/π of a
  cell so a face of the model never lines up with a plane of rays. It reports
  how many columns came back with an odd crossing count; on the m15 block that
  is 1 of 15,860, which is the honest measure of how much to trust the rest.
* **Tessellation deflection matters more than lattice pitch.** At 0.35 mm the
  m15 block tessellates to 160k triangles whose volume is within 0.5% of the
  B-rep's exact 3776 cm³, and the mesh is watertight to 143 boundary edges out
  of 240k.
* **`trimesh.ray` is not used**, deliberately: it wants `rtree` for its bound
  tree and `pyembree` to be quick, and neither is in this environment. A
  uniform bucket grid over the two coordinates the ray does *not* travel along
  does the same job in thirty lines of numpy.
