# The B-rep axioms — simple solids with known answers, measured

Status: measurement report, 2026-09-02, on `refactor/code-quality` at
`04f9923`. Nothing under `cadjoint/` was changed. The battery is
`tests/brep/axioms.py` (the catalogue, the measurement, the renderer),
pinned by `tests/brep/test_axioms.py` (45 passed, 38 strict xfails in 3 min 47 s) and drawn in
`research/brep-axioms/` (`gallery.png` is the contact sheet, one
`<case>_<cells>[_off<offset>][_tol1cell].png` per extraction, the raw
numbers in `measurements.json`; the run also leaves picklable drawable data
in `cache/`, 38 MB, not kept in the repo, from which
`python -m tests.brep.axioms --rerender` redraws without re-extracting).
Companion notes: `research/brep-architecture.md` (what the graph is) and
`research/brep-edge-tracing.md` (how its edges should be traced; §6 below
maps the evidence here onto its classifier).

The user's direction, verbatim: *"for the edges its easy to find the edges
when two faces cross each other because then we can derive the edge based
on the cross product of the normals. other idea there is a bunch of points
and we do gradient descent along the surfaces and then see where they snap.
also if a user defines tangency between surfaces we could use this
information. But basically to solve this whole brep thing, we should start
to go about this more axiomatic. We should start with simple examples of
shape unions and see how the brep extraction work on them / look at them
visually etc."*

## 0. The one paragraph

`extract_brep` is exact on every transversal case in the battery — box,
overlapping boxes, cylinder on and through a box, bore, sphere on a box,
unequal Steinmetz, the concave extrusion, the oblique cut, a sphere tangent
to a face from inside — with faces, edges, vertices and Euler characteristic
right, every edge on its curve to 2·10⁻⁵ cell or better, every corner to
2·10⁻⁶ cell, at 32 and 64 cells and at every lattice offset. It is wrong,
and wrong in a way the lattice does not cure, on exactly three things: **two
patches that share a zero set** (coplanar faces, tangent cylinders, the
tangent points of equal Steinmetz cylinders), where the residual gate is
blind because both fields vanish along the spurious seam; **blend bands
narrower than about two cells**, which fragment into slivers; and
**blend bands wider than the geometry**, where a patch's *infinite* plane
crosses the displaced surface and `|f_patch| < tol` fires on points the
patch does not own. Of the user's three ideas, (a) the normal cross product
is not a marching direction on the failing cases — it is the *classifier*
that separates them from the passing ones, and it is the only signal in the
graph that does; (b) gradient-descent snapping detects convex edges from
outside and concave edges from inside with no false positives, and never
detects a tangent seam or a coplanar join; (c) declared tangency is the
right *input* to the classifier of `brep-edge-tracing.md` §8(c), and the
battery shows it is needed only where the classifier's second-order test
would otherwise sit on float noise.

## 1. The catalogue

Nineteen cases in `tests/brep/axioms.py`, each a scene built from
cadjoint's own primitives, transforms and booleans, with its textbook
B-rep: face count and kinds, edge count (open and closed), vertex count,
Euler characteristic, and — wherever it has a closed form — every edge
curve (segments, circles, the Steinmetz quartics `(r_b cos t, r_b sin t,
±√(r_a² − r_b² sin²t))`, the equal-radius V-curves `x = ±|z|`) and every
corner. The union-of-two-boxes edges are generated, not typed: clip each
box's twelve edges against the other box's open interior, add the
face-plane × face-plane segments where both faces are crossed.

| case | what it tests | F | E (closed) | V | χ |
|---|---|---|---|---|---|
| `box` | the base case | 6 | 12 | 8 | 2 |
| `boxes_overlap` | two boxes in general position, corner through three faces | 12 | 30 | 20 | 2 |
| `boxes_coplanar` | B's top and bottom in A's top and bottom planes | **10** (each shared plane is ONE 8-gon face) | 24 | 16 | 2 |
| `box_cyl_standing` | rim circle on the slab, cap circle | 8 | 14 (2) | 8 | 2 |
| `box_cyl_through` | two rims on the box, two cap rims | 10 | 16 (4) | 8 | 2 |
| `plate_bore` | genus 1 | 7 | 14 (2) | 8 | **0** |
| `sphere_box` | sphere crossing the top transversally | 7 | 13 (1) | 8 | 2 |
| `steinmetz` | r 0.3 ⟂ r 0.2: two closed quartics, **no vertex** | 7 | 6 (6) | 0 | 2 |
| `steinmetz_equal` | r = r: two ellipses crossing at two **tangent points** | 8 | 8 (4) | 2 | 2 |
| `cyl_tangent` | two r 0.3 cylinders touching along a line: antiparallel normals along the whole seam | 6 | 5 (4) + a tangent seam | 2 | 2 |
| `sphere_tangent_face` | sphere out of the top, tangent to +x from inside at one point | 7 | 13 (1) | 8 | 2 |
| `bracket_sharp` | wall on a slab, four concave creases | 11 | 24 | 16 | 2 |
| `fillet_{0.2,0.5,1,2,4}cell` | the same, `Union(..., smoothness=k)` at k = 0.2 … 4 cells of the 32-grid | see §2.2 | | | |
| `extruded_concave` | L profile, one concave corner | 8 | 18 | 12 | 2 |
| `oblique_cut` | box − box rotated 0.5 rad: a non-lattice-aligned cut face | 7 | 15 | 10 | 2 |

Euler characteristic is taken over open cells with compactly supported χ,
`χ = Σ_faces (2 − loops) − open edges + vertices`, so a face with a hole
counts 0, a full cylinder band 0, a closed rim 0 and a plate with a bore
comes out at 0 — the plain `V − E + F` is only right when every face is a
disk (`axioms.euler_characteristic`, checked by hand on the plate, the box
and the Steinmetz).

Per extraction the battery records: counts and kinds against the expected;
χ; for every known curve the largest distance of the matched extracted
edge's points from it (matching by mean distance, one extracted edge to one
curve, edges more than a cell from every curve counted *unmatched*) and its
coverage (fraction of the curve within half a cell of some extracted
segment, the polylines closed through their linked vertices); the largest
corner error and the number of extracted vertices more than half a cell
from every corner; a verdict per edge — `analytic`, `blend` (a neighbour
face is a blend), `refused` (residual above the overlay's 0.1-cell gate);
the minimum of `|n_a × n_b|` over the edge's samples (the user's marching
direction, measured); NaNs; and the DC and graph wall times separately.

Grid: a cube around the shape with a 0.33 margin, so a 32-cell grid puts a
0.052 cell on a unit part; offsets 0, 0.37 and 0.71 cells shift the lattice
for the seven `hard` cases; 64 cells doubles everything for the gallery.

## 2. Results

### 2.1 The case table, 32 cells (the tests) and 64 cells (the gallery)

Edge and vertex errors are in cells. `min cov` is the least-covered known
curve. `spur V / unmatched E` are extracted vertices and edges that match
nothing. Verdicts: `a` analytic, `b` blend-adjacent, `r` refused. `s` is
DC + graph wall time.

| case | cells | offset | F | E | V | χ | kinds | edge err | min cov | vertex err | spur V / unmatched E | verdicts | min sin θ | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| box | 32 | 0 | 6/6 | 12/12 | 8/8 | 2/2 | plane6 | 2.6e-07 | 1.000 | 2.6e-07 | 0/0 | a12 | 1.00 | 3.1 |
| boxes_overlap | 32 | 0 | 12/12 | 30/30 | 20/20 | 2/2 | plane12 | 9.5e-07 | 1.000 | 9.5e-07 | 0/0 | a30 | 1.00 | 4.8 |
| boxes_coplanar | 32 | 0 | 12/10 | 26/24 | 16/16 | 2/2 | plane12 | 6.1e-07 | 1.000 | 6.1e-07 | 0/2 | a26 | 0.00 | 4.4 |
| boxes_coplanar | 32 | 0.37 | 12/10 | 26/24 | 16/16 | 2/2 | plane12 | 6.1e-07 | 1.000 | 6.1e-07 | 0/2 | a26 | 0.00 | 4.4 |
| boxes_coplanar | 32 | 0.71 | 12/10 | 26/24 | 16/16 | 2/2 | plane12 | 6.1e-07 | 1.000 | 6.1e-07 | 0/2 | a26 | 0.00 | 4.1 |
| box_cyl_standing | 32 | 0 | 8/8 | 14/14 | 8/8 | 2/2 | plane7,cylinder1 | 5.4e-06 | 1.000 | 5.7e-08 | 0/0 | a14 | 1.00 | 4.8 |
| box_cyl_through | 32 | 0 | 10/10 | 16/16 | 8/8 | 2/2 | plane8,cylinder2 | 4.5e-06 | 1.000 | 1.9e-07 | 0/0 | a16 | 1.00 | 6.1 |
| plate_bore | 32 | 0 | 7/7 | 14/14 | 8/8 | 0/0 | plane6,cylinder1 | 5.1e-06 | 1.000 | 5.9e-07 | 0/0 | a14 | 1.00 | 4.8 |
| sphere_box | 32 | 0 | 7/7 | 13/13 | 8/8 | 2/2 | plane6,sphere1 | 8.2e-06 | 1.000 | 2.3e-07 | 0/0 | a13 | 0.93 | 4.0 |
| steinmetz | 32 | 0 | 7/7 | 6/6 | 0/0 | 2/2 | plane4,cylinder3 | 6.4e-06 | 1.000 | 0.0e+00 | 0/0 | a6 | 0.75 | 6.2 |
| steinmetz_equal | 32 | 0 | 7/8 | 6/8 | 0/2 | 2/2 | plane4,cylinder3 | 0.36 | 0.996 | ∞ | 0/0 | a6 | 0.13 | 6.1 |
| steinmetz_equal | 32 | 0.37 | 7/8 | 6/8 | 0/2 | 2/2 | plane4,cylinder3 | 5.1e-06 | 1.000 | ∞ | 0/2 | a6 | 0.07 | 6.8 |
| steinmetz_equal | 32 | 0.71 | 7/8 | 6/8 | 0/2 | 2/2 | plane4,cylinder3 | 5.1e-06 | 1.000 | ∞ | 0/2 | a6 | 0.09 | 3.6 |
| cyl_tangent | 32 | 0 | 6/6 | 8/5 | 4/2 | 2/2 | cylinder2,plane4 | 0.25 | 1.000 | 0.25 | 0/0 | a8 | 0.10 | 5.3 |
| cyl_tangent | 32 | 0.37 | 6/6 | 8/5 | 4/2 | 2/2 | cylinder2,plane4 | 0.76 | 1.000 | 0.50 | 4/0 | a8 | 0.07 | 5.4 |
| cyl_tangent | 32 | 0.71 | 6/6 | 8/5 | 4/2 | 2/2 | cylinder2,plane4 | 0.43 | 1.000 | 0.22 | 0/0 | a8 | 0.08 | 5.7 |
| sphere_tangent_face | 32 | 0 | 7/7 | 13/13 | 8/8 | 2/2 | plane6,sphere1 | 6.0e-06 | 1.000 | 2.3e-07 | 0/0 | a13 | 0.78 | 5.4 |
| sphere_tangent_face | 32 | 0.37 | 7/7 | 13/13 | 8/8 | 2/2 | plane6,sphere1 | 6.2e-06 | 1.000 | 2.3e-07 | 0/0 | a13 | 0.78 | 5.0 |
| sphere_tangent_face | 32 | 0.71 | 7/7 | 13/13 | 8/8 | 2/2 | plane6,sphere1 | 5.9e-06 | 1.000 | 2.3e-07 | 0/0 | a13 | 0.78 | 5.1 |
| bracket_sharp | 32 | 0 | 11/11 | 24/24 | 16/16 | 2/2 | plane11 | 1.0e-06 | 1.000 | 1.0e-06 | 0/0 | a24 | 1.00 | 6.0 |
| fillet_0.2cell | 32 | 0 | 17/13 | 37/26 | 23/16 | 2/2 | plane11,blend6 | 1.30 | 1.000 | 1.0e-06 | 11/0 | a20 b17 | 1.00 | 3.3 |
| fillet_0.2cell @1cell | 32 | 0 | 11/13 | 24/26 | 16/16 | 2/2 | plane11 | 0.13 | 1.000 | 1.0e-06 | 4/0 | a24 | 1.00 | 2.8 |
| fillet_0.5cell | 32 | 0 | 14/13 | 27/26 | 17/16 | None/2 | plane12,blend2 | 0.18 | 0.000 | 1.0e-06 | 5/6 | a20 b7 | 1.00 | 6.2 |
| fillet_0.5cell @1cell | 32 | 0 | 14/13 | 31/26 | 20/16 | 2/2 | plane14 | 0.22 | 1.000 | 1.0e-06 | 8/0 | a31 | 1.00 | 4.2 |
| fillet_0.5cell | 32 | 0.37 | 13/13 | 26/26 | 16/16 | 2/2 | plane11,blend2 | 1.0e-06 | 0.000 | 1.0e-06 | 4/6 | a20 b6 | 1.00 | 5.9 |
| fillet_0.5cell | 32 | 0.71 | 14/13 | 28/26 | 18/16 | 2/2 | plane11,blend3 | 0.98 | 0.000 | 1.0e-06 | 6/7 | a20 b8 | 1.00 | 5.1 |
| fillet_1cell | 32 | 0 | 15/13 | 33/28 | 22/18 | 2/2 | plane13,blend2 | 1.24 | 0.000 | 1.0e-06 | 10/10 | a21 b12 | 1.00 | 6.0 |
| fillet_1cell @1cell | 32 | 0 | 17/13 | 40/28 | 26/18 | 2/2 | plane17 | 0.97 | 0.857 | 1.0e-06 | 14/0 | a40 | 1.00 | 3.5 |
| fillet_2cell | 32 | 0 | 18/15 | 36/34 | 22/22 | 2/2 | plane16,blend2 | 1.18 | 0.000 | 6.56 | 12/12 | a20 b16 | 1.00 | 5.6 |
| fillet_2cell @1cell | 32 | 0 | 27/15 | 67/34 | 44/22 | None/2 | plane18,blend9 | 1.78 | 0.000 | 1.0e-06 | 32/40 | a21 b38 r8 | 1.00 | 3.6 |
| fillet_2cell | 32 | 0.37 | 16/15 | 33/34 | 20/22 | 2/2 | plane14,blend2 | 1.54 | 0.000 | 6.18 | 10/11 | a20 b13 | 1.00 | 6.0 |
| fillet_2cell | 32 | 0.71 | 25/15 | 48/34 | 29/22 | None/2 | plane22,blend3 | 1.77 | 0.000 | 4.08 | 19/23 | a20 b28 | 1.00 | 5.3 |
| fillet_4cell | 32 | 0 | 26/2 | 28/1 | 4/0 | 2/2 | blend2,plane24 | 1.02 | 0.000 | 15.43 | 4/22 | b28 | — | 3.7 |
| fillet_4cell @1cell | 32 | 0 | 67/2 | 182/1 | 118/0 | 2/2 | plane49,blend18 | 1.67 | 0.000 | 0.92 | 94/100 | a93 r15 b74 | 1.00 | 5.2 |
| extruded_concave | 32 | 0 | 8/8 | 18/18 | 12/12 | 2/2 | plane8 | 4.3e-07 | 1.000 | 4.3e-07 | 0/0 | a18 | 1.00 | 7.2 |
| oblique_cut | 32 | 0 | 7/7 | 15/15 | 10/10 | 2/2 | plane7 | 6.8e-07 | 1.000 | 5.4e-07 | 0/0 | a15 | 0.48 | 6.2 |
| oblique_cut | 32 | 0.37 | 7/7 | 15/15 | 10/10 | 2/2 | plane7 | 6.5e-07 | 1.000 | 4.5e-07 | 0/0 | a15 | 0.48 | 6.2 |
| oblique_cut | 32 | 0.71 | 7/7 | 15/15 | 10/10 | 2/2 | plane7 | 8.1e-07 | 1.000 | 4.5e-07 | 0/0 | a15 | 0.48 | 6.2 |
| box | 64 | 0 | 6/6 | 12/12 | 8/8 | 2/2 | plane6 | 5.1e-07 | 1.000 | 5.1e-07 | 0/0 | a12 | 1.00 | 3.8 |
| boxes_overlap | 64 | 0 | 12/12 | 30/30 | 20/20 | 2/2 | plane12 | 1.9e-06 | 1.000 | 1.9e-06 | 0/0 | a30 | 1.00 | 6.5 |
| boxes_coplanar | 64 | 0 | 12/10 | 26/24 | 16/16 | 2/2 | plane12 | 1.2e-06 | 1.000 | 1.2e-06 | 0/2 | a26 | 0.00 | 5.8 |
| boxes_coplanar | 64 | 0.37 | 12/10 | 26/24 | 16/16 | 2/2 | plane12 | 1.2e-06 | 1.000 | 1.2e-06 | 0/2 | a26 | 0.00 | 5.3 |
| boxes_coplanar | 64 | 0.71 | 12/10 | 26/24 | 16/16 | 2/2 | plane12 | 1.2e-06 | 1.000 | 1.2e-06 | 0/2 | a26 | 0.00 | 5.1 |
| box_cyl_standing | 64 | 0 | 8/8 | 14/14 | 8/8 | 2/2 | plane7,cylinder1 | 1.2e-05 | 1.000 | 1.1e-07 | 0/0 | a14 | 1.00 | 6.0 |
| box_cyl_through | 64 | 0 | 10/10 | 16/16 | 8/8 | 2/2 | plane8,cylinder2 | 9.7e-06 | 1.000 | 3.8e-07 | 0/0 | a16 | 1.00 | 7.3 |
| plate_bore | 64 | 0 | 7/7 | 14/14 | 8/8 | 0/0 | plane6,cylinder1 | 1.0e-05 | 1.000 | 1.2e-06 | 0/0 | a14 | 1.00 | 6.1 |
| sphere_box | 64 | 0 | 7/7 | 13/13 | 8/8 | 2/2 | plane6,sphere1 | 1.7e-05 | 1.000 | 4.6e-07 | 0/0 | a13 | 0.93 | 5.2 |
| steinmetz | 64 | 0 | 7/7 | 6/6 | 0/0 | 2/2 | plane4,cylinder3 | 1.3e-05 | 1.000 | 0.0e+00 | 0/0 | a6 | 0.75 | 7.1 |
| steinmetz_equal | 64 | 0 | 7/8 | 6/8 | 0/2 | 2/2 | plane4,cylinder3 | 0.36 | 0.996 | ∞ | 0/0 | a6 | 0.06 | 7.4 |
| steinmetz_equal | 64 | 0.37 | 7/8 | 6/8 | 0/2 | 2/2 | plane4,cylinder3 | 1.0e-05 | 0.996 | ∞ | 0/2 | a6 | 0.05 | 7.7 |
| steinmetz_equal | 64 | 0.71 | 6/8 | 5/8 | 0/2 | 2/2 | plane4,cylinder2 | 1.0e-05 | 0.996 | ∞ | 0/1 | a5 | 0.05 | 4.3 |
| cyl_tangent | 64 | 0 | 6/6 | 8/5 | 4/2 | 2/2 | cylinder2,plane4 | 0.50 | 0.086 | 0.41 | 0/0 | a8 | 0.09 | 5.9 |
| cyl_tangent | 64 | 0.37 | 6/6 | 8/5 | 4/2 | 2/2 | cylinder2,plane4 | 1.76 | 0.043 | 1.50 | 4/0 | a8 | 0.12 | 5.8 |
| cyl_tangent | 64 | 0.71 | 6/6 | 8/5 | 4/2 | 2/2 | cylinder2,plane4 | 1.43 | 1.000 | 0.84 | 4/0 | a8 | 0.08 | 5.8 |
| sphere_tangent_face | 64 | 0 | 7/7 | 13/13 | 8/8 | 2/2 | plane6,sphere1 | 1.2e-05 | 1.000 | 4.6e-07 | 0/0 | a13 | 0.78 | 5.2 |
| sphere_tangent_face | 64 | 0.37 | 7/7 | 13/13 | 8/8 | 2/2 | plane6,sphere1 | 1.2e-05 | 1.000 | 4.6e-07 | 0/0 | a13 | 0.78 | 5.4 |
| sphere_tangent_face | 64 | 0.71 | 7/7 | 13/13 | 8/8 | 2/2 | plane6,sphere1 | 1.2e-05 | 1.000 | 4.6e-07 | 0/0 | a13 | 0.78 | 5.7 |
| bracket_sharp | 64 | 0 | 11/11 | 24/24 | 16/16 | 2/2 | plane11 | 2.1e-06 | 1.000 | 2.1e-06 | 0/0 | a24 | 1.00 | 5.7 |
| fillet_0.2cell | 64 | 0 | 15/13 | 32/26 | 20/16 | 2/2 | plane11,blend4 | 1.16 | 0.000 | 2.1e-06 | 8/2 | a20 b12 | 1.00 | 5.2 |
| fillet_0.2cell @1cell | 64 | 0 | 11/13 | 24/26 | 16/16 | 2/2 | plane11 | 0.19 | 1.000 | 2.1e-06 | 4/0 | a24 | 1.00 | 3.1 |
| fillet_0.5cell | 64 | 0 | 13/13 | 26/26 | 16/16 | 2/2 | plane11,blend2 | 0.25 | 0.000 | 2.1e-06 | 4/6 | a20 b6 | 1.00 | 5.4 |
| fillet_0.5cell @1cell | 64 | 0 | 17/13 | 40/26 | 26/16 | 2/2 | plane17 | 1.08 | 0.895 | 2.1e-06 | 14/0 | a40 | 1.00 | 4.4 |
| fillet_0.5cell | 64 | 0.37 | 15/13 | 31/26 | 20/16 | 2/2 | plane13,blend2 | 0.13 | 0.000 | 2.1e-06 | 8/11 | a20 b11 | 1.00 | 6.1 |
| fillet_0.5cell | 64 | 0.71 | 13/13 | 26/26 | 16/16 | 2/2 | plane11,blend2 | 0.18 | 0.000 | 2.1e-06 | 4/6 | a20 b6 | 1.00 | 5.6 |
| fillet_1cell | 64 | 0 | 16/13 | 32/28 | 20/18 | 2/2 | plane14,blend2 | 1.34 | 0.000 | 2.1e-06 | 8/10 | a21 b11 | 1.00 | 5.5 |
| fillet_1cell @1cell | 64 | 0 | 24/13 | 59/28 | 38/18 | 2/2 | plane17,blend7 | 2.02 | 0.000 | 2.1e-06 | 26/31 | a25 b31 r3 | 1.00 | 3.6 |
| fillet_2cell | 64 | 0 | 29/15 | 56/34 | 34/22 | 2/2 | plane27,blend2 | 2.35 | 0.000 | 1.87 | 23/20 | a25 b31 | 1.00 | 6.1 |
| fillet_2cell @1cell | 64 | 0 | 34/15 | 88/34 | 57/22 | 2/2 | plane24,blend10 | 1.70 | 0.000 | 2.1e-06 | 41/44 | a39 r5 b44 | 1.00 | 4.6 |
| fillet_2cell | 64 | 0.37 | 40/15 | 78/34 | 43/22 | 2/2 | plane37,blend3 | 1.02 | 0.000 | 9.06 | 31/26 | a32 b46 | 1.00 | 5.8 |
| fillet_2cell | 64 | 0.71 | 26/15 | 46/34 | 24/22 | 2/2 | plane24,blend2 | 0.82 | 0.000 | 10.09 | 14/19 | a20 b26 | 1.00 | 4.4 |
| fillet_4cell | 64 | 0 | 38/2 | 42/1 | 11/0 | None/2 | blend3,plane35 | 1.00 | 0.000 | 31.18 | 11/40 | b41 a1 | 1.00 | 4.3 |
| fillet_4cell @1cell | 64 | 0 | 74/2 | 208/1 | 137/0 | 2/2 | plane48,blend26 | 2.04 | 0.000 | 3.46 | 127/170 | a82 b111 r15 | 1.00 | 5.6 |
| extruded_concave | 64 | 0 | 8/8 | 18/18 | 12/12 | 2/2 | plane8 | 8.6e-07 | 1.000 | 8.6e-07 | 0/0 | a18 | 1.00 | 7.7 |
| oblique_cut | 64 | 0 | 7/7 | 15/15 | 10/10 | 2/2 | plane7 | 1.4e-06 | 1.000 | 9.0e-07 | 0/0 | a15 | 0.48 | 6.7 |
| oblique_cut | 64 | 0.37 | 7/7 | 15/15 | 10/10 | 2/2 | plane7 | 1.9e-06 | 1.000 | 2.0e-06 | 0/0 | a15 | 0.48 | 6.2 |
| oblique_cut | 64 | 0.71 | 7/7 | 15/15 | 10/10 | 2/2 | plane7 | 1.9e-06 | 1.000 | 2.0e-06 | 0/0 | a15 | 0.48 | 6.8 |

Timings: 38 extractions at 32 cells took 194 s of DC + graph (5.1 s each,
of which the DC pass is 1.0–1.7 s); 38 at 64 cells took 214 s — the cost
is dominated by per-call JAX dispatch, not by cell count. No NaN anywhere.

### 2.2 The fillets

`smooth_min(a, b, k) = min(a, b) − h²/(16k)`, `h = max(4k − |a − b|, 0)`:
the surface leaves the crease by exactly `k`, but the *band* spans
`|a − b| < 4k`. On the bracket the wall's faces are 0.15, 0.25, 0.25 and
0.45 from the slab's edges and the wall stands 0.5 above the slab, so the
band reaches the +y edge at `4k > 0.15` (k ≈ 0.72 cell), −y and +x at
`4k > 0.25`, and at `4k = 0.83` (k = 4 cells) no point of the part is
outside it — a "four cell fillet" of this `smooth_min` is a global
deformation of a nineteen-cell part. The graph also splits every blend band
per *leaf* (`graph.py` step 1: a blend quad keeps its nearest leaf), so one
fillet ring is two blend faces with a closed mid-band edge between them.
The textbook answers per band, derived in `axioms.py:_FILLET_EXPECTED`:

| k (cells of 32) | band 4k | expected F | E (closed) | V | measured 32 | measured 64 (k is twice the cells there) |
|---|---|---|---|---|---|---|
| 0.2 | 0.04 (0.8 cell) | 13 = 11 plane + 2 blend | 26 (2) | 16 | **17 / 37 / 23**, 6 blend fragments | 15 / 32 / 20, 4 fragments |
| 0.5 | 0.10 (2 cells) | 13 | 26 (2) | 16 | 14 / 27 / 17, χ undefined; 13/26/16 and 14/28/18 at the other offsets | **13 / 26 / 16 exactly** (offset 0 and 0.71), 15/31/20 at 0.37 |
| 1 | 0.21 (4 cells) > 0.15 | 13 | 28 (1) | 18 | 15 / 33 / 22 | 16 / 32 / 20 |
| 2 | 0.41 (8 cells) > 0.25 | 15 = 13 plane + 2 blend | 34 (1) | 22 | 18 / 36 / 22; 16/33/20; 25/48/29 | 29 / 56 / 34; 40/78/43; 26/46/24 |
| 4 | 0.83 (16 cells) | 2 blend | 1 (1) | 0 | **26 / 28 / 4**: 24 plane islands of 1–18 quads | 38 / 42 / 11 |

Under the overlay's one-cell rule (`blend_tolerance` = one cell, the same
DC mesh): the 0.2-cell fillet comes back as the sharp bracket, 11/24/16,
every crease covered, the crease polylines 0.13 cell off the sharp crease
(0.19 at 64); the 0.5-cell fillet does **not** — 14/31/20 with three
slivers and two ambiguous vertices at 32, 17/40/26 at 64 — and above a cell
the rule produces 17/40/26, 27/67/44 (8 refused edges) and 67/182/118 (15
refused). The one-cell rule is right for fillets below about half a cell
and wrong above; the transition is not at one cell.

### 2.3 What the degenerate cases do, in detail

**`boxes_coplanar`** — 12/26/16 at every offset and resolution, never the
right 10/24/16. Ownership on the shared plane is a tie broken by leaf
order, so A's patch owns the plane for x < 0.5 and B's for x > 0.5, and the
region boundary at x = 0.5 becomes an edge between two *coplanar* patches.
Its two-field residual is exactly 0 — both fields vanish on the entire
plane — so the residual gate certifies it, and the seam's endpoints are
vertices with four incident faces. `min |n_a × n_b| = 0.0` along it, the
only number in the graph that says anything is wrong.
(`boxes_coplanar_64.png`)

**`cyl_tangent`** — F 6 is right; E 8/5 and V 4/2 are not. Dual contouring
bridges the sub-cell wedge along the contact line with a strip of quads, so
the boundary between the two cylinder faces runs down *both* sides of the
wedge, at y = ±0.015 (0.28 cell at 32, 0.41 cell at 64), and both chains
pass the residual gate (residual 4·10⁻⁴ = 0.008 cell) as analytic edges
with `|n_a × n_b|` = 0.07–0.12 — the two-field projection stops where the
Gram guard `λ_min > 10⁻² tr/2` trips, a few hundredths of a cell short of
the true seam. Cylinder B's rim circles are each broken by a one-point
chain where the rim touches cylinder A. The vertex error and the number of
spurious vertices change with the offset (0.25 / 0.50 / 0.22 cell; 0 / 4 /
0) and get *worse* at 64 cells (0.41 / 1.50 / 0.84): the finer the lattice,
the thinner the wedge it tries to fill. (`cyl_tangent_32_off0.37.png`)

**`steinmetz_equal`** — 7/6/0 for 8/8/2: cylinder A stays one face with
four loops, B splits in two, and the two V-shaped seams `x = ±|z|` come out
as single closed chains that miss their corner at the tangent point by 0.36
cell (offset 0; coverage 0.996) or 5·10⁻⁶ cell (the other offsets). Nothing
in the graph creates a vertex where four faces meet along two crossing
curves; `min |n_a × n_b|` on the seams is 0.05–0.13 and reaches 0 only at
the two points themselves. At 64 cells and offset 0.71 one seam is lost
entirely (6/5/0). (`steinmetz_equal_64_off0.71.png`)

**`sphere_tangent_face`** — right everywhere: 7/13/8, the rim to 1.2·10⁻⁵
cell, at every offset and resolution. A point of internal tangency with a
plane produces no ownership island because the sphere's field grows
quadratically off the point and the box's plane wins the argmin
immediately. The *first* version of this case had the rim 0.3 cell from the
box's +x edge and failed (7/15/10 at offset 0, 9/19/12 at 0.71, right at
0.37): that was a sub-cell strip of the top face, not the tangency, and it
is the same failure as the fillet slivers.

**`fillet_4cell`** — the 24 "plane" islands (1–18 quads, `fillet_4cell_64.png`,
the small orange loops on the slab's sides) are not flicker between two
owners of one point. On the displaced surface far from the wall, every
patch value is of order the displacement (0.03–0.08) — except where the
patch's *infinite* plane happens to cross the displaced surface, where
`|f_patch|` passes through zero on a curve. The blend test
`|f_owner| < tol` is a test on the field, not on the face, and it has
false positives wherever a patch's zero set extends past its face.

**`extruded_concave`**, **`oblique_cut`** — exact, 8/18/12 and 7/15/10, at
every offset; the concave corner and a cut plane at `sin θ = 0.48` to a
box face cost nothing. **`steinmetz`** (unequal radii) — 7/6/0 exact, the
quartics to 1.3·10⁻⁵ cell; a curved–curved edge with no closed form in the
graph is still placed exactly by the two-field kernel, which is what
`brep-architecture.md` §9.1 claimed.

### 2.4 Lattice sensitivity

| hard case | 32: offsets 0 / 0.37 / 0.71 | 64: offsets 0 / 0.37 / 0.71 |
|---|---|---|
| `boxes_coplanar` | 12/26/16 ×3 (wrong, stable) | 12/26/16 ×3 |
| `steinmetz_equal` | 7/6/0 ×3 (wrong, stable) | 7/6/0, 7/6/0, **6/5/0** |
| `cyl_tangent` | 6/8/4 ×3, vertex error 0.25 / 0.50 / 0.22 | 6/8/4 ×3, 0.41 / 1.50 / 0.84 |
| `sphere_tangent_face` | 7/13/8 ×3 | 7/13/8 ×3 |
| `fillet_0.5cell` | 14/27/17, 13/26/16, 14/28/18 | 13/26/16, 15/31/20, 13/26/16 |
| `fillet_2cell` | 18/36/22, 16/33/20, 25/48/29 | 29/56/34, 40/78/43, 26/46/24 |
| `oblique_cut` | 7/15/10 ×3 | 7/15/10 ×3 |

The coplanar and equal-Steinmetz failures are *stable*: the lattice does not
cause them and cannot fix them. The tangent-cylinder and fillet failures are
lattice-driven and do not improve with resolution, because what the lattice
sees (a wedge, a band) is a structure it fills with quads whose ownership
then flickers.

## 3. Failure taxonomy

1. **Coincident zero sets** (`boxes_coplanar`; the flush faces of any two
   stacked boxes). Ownership is a tie, the seam has zero residual, the Gram
   is rank 1 along the whole curve. The residual gate cannot see it; the
   normal-crossing test sees it exactly (`sin θ = 0`). Right answer: merge
   the faces before edges are built.
2. **Tangent seams** (`cyl_tangent`; a cylinder lying on a plate, a
   fillet's own G1 boundaries). Antiparallel or parallel normals along a
   curve; the two-field solve stops at the guard, the DC mesh fills the
   sub-cell wedge, the seam doubles, its ends become spurious vertices, and
   resolution makes it worse. Right answer: classify as `tangent`, place by
   a damped solve, never call it an edge, never give it a derivative.
3. **Tangent points / branch points** (`steinmetz_equal`). Two crossing
   seams through a point where four faces meet and `sin θ → 0` at the point
   only. No vertex is created because no mesh vertex touches three regions
   there; a seam may be lost at some offsets. Right answer: a `branch`
   record at the point and four half-edges.
4. **Sub-cell bands** (`fillet_0.2cell`, `fillet_0.5cell` at 32, the first
   `sphere_tangent_face`). Anything the DC pass resolves with one or two
   quads across its width fragments into slivers with unstable counts.
   Not a classification error — the features are real — but the lattice is
   the only witness and it is not a good one. Right answer: treat the DC
   chains as seeds only and trace the band's boundaries
   `f_a = 0 ∧ f_b = 4k` from the fields (`brep-edge-tracing.md` §2.2).
5. **Blend classification by field value** (`fillet_4cell`, and the
   ambiguous vertices of every fillet). `|f_owner| < tol` on the scene's
   zero set has false positives on the extensions of unbounded patch fields
   and a resolution-dependent threshold (export: a thousandth of the grid
   diagonal; overlay: one cell). Right answer: the band is known from the
   design — `|a − b| < 4k` — and should be read off the boolean, not
   measured on the surface.
6. **Corner sharing at blends / four-face vertices.** Every fillet case
   reports 5–14 ambiguous vertices; `steinmetz_equal` reports none where it
   needs two. Both come from "vertex = mesh vertex with three regions".
   Right answer: vertices from watch-field crossings on traced edges, keyed
   by patch triple (`brep-edge-tracing.md` §3).
7. **Not a failure: transversal geometry of any kind.** Concave, oblique,
   curved–curved, genus 1, sphere–plane tangency at a point — all exact and
   lattice-independent. The two-field and three-field kernels are not the
   problem.

## 4. The user's three ideas, against the evidence

### (a) `∇f_a × ∇f_b` as the marching direction

Measured as `min |n_a × n_b|` over every edge (`normal_crossing` in
`axioms.py`; it belongs next to `transversal()` in `brep/project.py`):

| | passing cases | `boxes_coplanar` | `cyl_tangent` | `steinmetz_equal` |
|---|---|---|---|---|
| min sin θ | 1.00 on box edges, 0.93 sphere rim, 0.75 Steinmetz, 0.48 oblique cut, 0.78 tangent-sphere rim | **0.00** on the spurious seam | 0.07–0.12 on both seam copies | 0.05–0.13 on the seams, 0 at the two points |

Every wrong edge in the battery has `sin θ < 0.13`; every right edge has
`sin θ ≥ 0.48`. The residual gate, by contrast, passes every wrong edge
(residual 0 on the coplanar seam, 0.008 cell on the tangent one). So the
cross product is not primarily a *direction* — as a direction it is fine
on the passing cases, where `trace_curves` already uses it, and it is
undefined exactly on the failing ones. It is the *classifier*: where it
vanishes the pair is coincident, tangent or at a branch point, and the
right response is to stop solving and name the state. What it implies for
marching: a tracer that steps along `∇f_a × ∇f_b` must (i) refuse to start
where `sin θ` is below a floor, (ii) hand the pair to a second-order test —
`M = H_b − λH_a` on the tangent plane, definite / rank-1 / indefinite /
zero for contact / tangent seam / branch / coincident, `brep-edge-tracing.md`
§8(c) — and (iii) place `tangent` seams by a damped (LM) solve with no IFT
derivative, since `(JJᵀ)⁻¹` does not exist there. `cyl_tangent` is the
case that shows why (iii) is not optional: the undamped kernel stops a
quarter cell short and the lattice then invents a second seam.

### (b) Scatter points, descend onto the surface, see where they snap

Measured directly (`axioms.snap_census`, 30 000 uniform seeds in the
padded bounding box, 60 damped one-field Newton steps, a landed point counts
as *on an edge* when it is within 2·10⁻³ of a known curve):

| case | seeds outside → on an edge | which edges | seeds inside → on an edge | which edges |
|---|---|---|---|---|
| `box` | 415 of 3668 (11.3 %) | all 12 | 1 of 3034 | — |
| `boxes_overlap` | 12.3 % | all 24 convex edges, **never** the 6 concave ones | 2.1 % | **only** the 6 concave ones |
| `bracket_sharp` | 12.0 % | the 20 convex edges | 2.0 % | only the 4 concave creases |
| `boxes_coplanar` | 10.2 % | 22 of 24 | 1.0 % | the 2 concave verticals at the join |
| `cyl_tangent` | 7.1 % | the 4 rims | 0.2 % (4 points) | the seam |

It works, and it works cleanly: a distance field's gradient carries an
exterior point to its nearest surface point, and the set of exterior points
whose nearest point is a convex edge has positive volume (the wedge), so
~11 % of exterior seeds land on convex edges, none on concave ones; from
inside it is the mirror image. There are no false positives. Two limits:
(i) it finds *convex-from-here* creases only, so a full census needs both
sides, and (ii) it cannot find what has no wedge — the coplanar join
attracts nothing (the two `vert` edges it hits are the concave verticals,
which are real), the tangent seam attracts 4 points in 2 513. As a detector
it is therefore a complement to (a), not a replacement: (a) classifies a
seam once you have a point on it, (b) supplies seed points on transversal
creases with a density proportional to the crease's exterior angle, which
is exactly what §5 of `brep-edge-tracing.md` wants from the lattice and
gets less reliably (the lattice seeds one component per DC-visible chain;
snapping seeds every crease with a wedge, sub-cell or not). It is cheap —
one batched one-field projection — and it is already the kernel's arity 1.

### (c) User-declared tangency as a hint

The battery's tangent cases are all *constructed* tangencies: equal radii,
centres 2r apart, a smooth union. In each the classifier's second-order test
is deciding on numbers that are exactly zero in exact arithmetic and float
noise in practice (`sin θ` on the tangent seam is 0.07–0.12 at the guard's
stopping point, not 0). A declaration removes that decision. Concretely,
three places where a declared relation is information the fields cannot
give: a `Union(..., smoothness=k)` *is* a declaration that every seam it
creates is G1 with band `4k` — the graph should read `k` off the boolean
(taxonomy item 5) instead of measuring `|f|`; a sketch tangency constraint
(`cadjoint.constraints`) between two profile edges is a declaration that
the extruded walls share a normal along a line, which pins their pair to
`tangent`; and "flush" placement of two bodies is a declaration of
`coincident`. What (c) does not do is replace the classifier — a tangency
that arises from the numbers (an optimizer driving two radii equal) must
still be detected — so the right shape is the one `brep-edge-tracing.md`
§8(c) sketches with one addition: `classify(a, b, x, k_blend,
declared=None)` where a declaration overrides the second-order test and is
*checked* against it (a declared tangency whose `M` is indefinite is a
modelling error worth reporting).

## 5. Recommended order of fixes

Ordered by how many battery cases each unlocks per line of code, with the
`brep-edge-tracing.md` section that specifies it.

1. **Put the normal-crossing test in the graph, and let it veto the
   residual gate** (§8(c) `crossing` vs the rest; ~20 lines in
   `_build_edges`, using `_patch_normals` which already exists). Every wrong
   edge in the battery has `sin θ < 0.13` and every right one `≥ 0.48`, so a
   floor at the overlay's `_TANGENT_FLOOR` = 0.1 separates them today. This
   turns the coplanar seams, the doubled tangent seam and the Steinmetz
   points from "analytic" into a named state and stops them reaching STEP
   and the mesher as edges. It fixes no topology on its own but makes every
   later fix testable: `boxes_coplanar`, `cyl_tangent`, `steinmetz_equal`
   `edges` xfails move from wrong to *classified*.
2. **Merge `coincident` faces before edges** (§3, §8(b) "merge faces of
   every coincident pair"). With (1) naming the pair, the fix is to give the
   two regions one key in `_components` (`graph.py` step 2). Unlocks
   `boxes_coplanar` completely (10/24/16); nothing else in the graph
   changes. One afternoon.
3. **Read the blend band off the boolean** (§2.2; taxonomy 5). Replace
   `|f_owner| < blend_tolerance` with "this quad's centroid has
   `|f_a − f_b| < 4k` for the smooth pair that made it", which is exact,
   threshold-free and resolution-independent, and trace the band's two
   boundaries as `f_a = 0 ∧ f_b − 4k = 0` with the existing two-field
   kernel. Removes the `fillet_4cell` islands, the export/overlay tolerance
   split, and gives every fillet its three exact curves. Unlocks
   `fillet_{1,2,4}cell` and the `@1cell` rows; with (5) also `0.2` and `0.5`.
4. **Damped solve for `tangent` seams, no derivative** (§7 LM row,
   `project.py damped=True`). Places the `cyl_tangent` seam on the contact
   line instead of a quarter cell off it, and stops the corner solve from
   inventing four vertices. Unlocks `cyl_tangent` up to the doubled chain.
5. **Watch-field termination and traced vertices** (§1.2, §3, §8(a)).
   Edges end where a third field changes sign; vertices are keyed by patch
   triple; `_build_vertices`' "mesh vertex with ≥ 3 faces" goes. Unlocks
   the two `steinmetz_equal` vertices (a `branch` record, §8(c)), the
   sub-cell fillet slivers (the DC chain is a seed, the trace is the edge),
   and the ambiguous-vertex counts on every blend. This is the large one;
   the battery gives it fourteen xfails to retire.
6. **Snap-seeding as a second witness** (idea (b); §5 of the memo is the
   interval census, which certifies completeness but is a new module). A
   batched one-field projection of scattered points, both sides, seeds every
   transversal crease with a wedge regardless of the lattice. Cheaper than
   the census and already in the kernel; the census remains the certificate
   for what neither the lattice nor the wedge can see (loops without a cell,
   tangencies).

Not recommended as a fix: raising the resolution (every failing case is
stable or worse at 64), lowering the residual gate (it is already blind on
the cases that matter), or tuning `blend_tolerance` (the transition it
approximates is at half a cell, not one, and (3) removes the need for it).

## 6. Reading the gallery

`research/brep-axioms/gallery.png` is the 64-cell contact sheet, one panel
per case at offset 0. Grey is the DC wireframe on the re-solved points;
navy the extracted analytic edges; orange blend-adjacent chains; red
refused ones; green the true curves behind them (pink a tangent seam, light
blue a crease a fillet replaced); green circles the true corners, navy dots
the solved vertices and red crosses the ambiguous ones. Where navy sits on
green and every green circle has a dot, the case passes. Look at
`boxes_coplanar_64.png` for the seam across the top face,
`cyl_tangent_32_off0.37.png` for the doubled seam and its four vertices,
`steinmetz_equal_64_off0.71.png` for the missing corners and the lost seam,
`fillet_0.5cell_64.png` for the textbook two-ring blend, and
`fillet_4cell_64.png` for the plane-extension islands.

## 7. Reproducing

```bash
.venv/bin/pytest tests/brep/test_axioms.py -q          # 45 passed, 38 xfailed, 3 min 47 s
PYTHONPATH=. .venv/bin/python -m tests.brep.axioms      # gallery, 32 + 64, ~8 min
PYTHONPATH=. .venv/bin/python -m tests.brep.axioms --rerender   # redraw from cache/
```

The snap experiment is `tests.brep.axioms.snap_census(case)`; §4(b)'s table
is its output for the five cases named there (`seed=0`, 30 000 seeds).
