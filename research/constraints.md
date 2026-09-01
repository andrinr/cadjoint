# Constraint system: final state and remaining vocabulary

> Compact note. This file was the pre-implementation research for the constraint
> system; everything it recommended has shipped. Full build history in git/PR #19.

## What shipped (see `cadjoint/constraints/`, `tests/constraints/`)

- **Solver**: `solve_constraints` runs `optimistix.LevenbergMarquardt`
  (implicit-adjoint differentiable through the solve); `satisfy_constraints`
  handles under-constrained projection; `project_to_manifold`,
  `make_manifold_projection`, and `make_bounds_projection` compose with optax.
- **Constraint types** (`cadjoint/constraints/types/`): distance, angle,
  parallel/perpendicular (vector- and edge-based), fixed, coincident,
  horizontal, vertical, equal-length, point-on-line — all plain differentiable
  residuals through one residual/solve path.
- **Numerical forms**: the singularity fixes recommended here (squared distance
  form, cosine angle form) are in the shipped residuals.

## Remaining constraint vocabulary (not yet implemented)

| Constraint | DOF removed | Residual |
|---|---|---|
| Collinear (3 points) | 1 | `(p2-p1) × (p3-p1) = 0` (scalar in 2D) |
| Midpoint | 2 | `p_mid - (p1+p2)/2 = 0` |
| Tangent (circle-line) | 1 | `dist(center, line)² - r² = 0` (squared form) |
| Tangent (circle-circle) | 1 | `‖c1-c2‖ - (r1±r2) = 0` |
| Concentric | 2 | `c1 - c2 = 0` |
| Symmetric | 2 | midpoint on axis + `(p2-p1) ⊥ axis` |
| Equal radius | 1 | `r1 - r2 = 0` |
| Point on circle | 1 | `‖p - center‖ - r = 0` |
| Coplanar (3D) | 1 | `(p - origin) · normal = 0` |

Also still open: per-constraint conflict identification (report which residuals
stay large in an over-constrained system) and connected-component decomposition
of the constraint graph.

## References

| Resource | URL |
|---|---|
| SolveSpace constraint reference | https://github.com/solvespace/solvespace (`src/system.cpp`) |
| FreeCAD planegcs | https://github.com/FreeCAD/FreeCAD/tree/master/src/Mod/Sketcher/App/planegcs |
| python-solvespace (validation oracle) | https://pypi.org/project/python-solvespace/ |
| optimistix docs | https://docs.kidger.site/optimistix/ |
