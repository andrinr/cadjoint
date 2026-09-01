# Improved Constraint System

> Research into richer constraint types, better solvers, and improved under/over-constrained handling for jaxCAD.

## Current system

| Component | What it does | Key limitation |
|---|---|---|
| `newton_raphson` | Python `for` loop + `jax.jacobian` + `lstsq` | Not JIT-compiled; no step-size control; diverges from bad initial guess |
| `solve_constraints` | Wraps NR with DOF check | Hard-raises `ValueError` for any non-exact system |
| Constraint types | Distance, Angle, Parallel, Perpendicular | Missing: Coincident, Fixed, Horizontal, Vertical, Tangent, etc. |
| DOF check | `total_dof == constraint_dof` required | No per-parameter DOF reporting; no conflict identification |

**Installed libraries** (`uv` environment):
- `optimistix 0.1.0` — Levenberg-Marquardt, Gauss-Newton, Dogleg, Newton+bounds
- `lineax 0.1.0` — JAX linear solvers (QR, LU, SVD, CG)
- `equinox 0.13.6` — JAX pytree utilities, `filter_jit`
- `scipy 1.17.0` — `least_squares(method='lm')` fallback

---

## Part 1: Richer constraint types

### The full vocabulary (professional CAD standard)

| Constraint | DOF removed | Residual |
|---|---|---|
| **Coincident** (p1 == p2) | 2–3 | `p1 - p2 = 0` |
| **Fixed** (pin to coordinate) | 2–3 | `p - p_fixed = 0` |
| **Horizontal** (line) | 1 | `end.y - start.y = 0` |
| **Vertical** (line) | 1 | `end.x - start.x = 0` |
| **Collinear** (3 points) | 1 | `(p2-p1) × (p3-p1) = 0` (scalar in 2D) |
| **Midpoint** | 2 | `p_mid - (p1+p2)/2 = 0` |
| **Equal length** | 1 | `‖p2-p1‖ - ‖p4-p3‖ = 0` |
| **Tangent** (circle-line) | 1 | `dist(center, line) - radius = 0` |
| **Tangent** (circle-circle) | 1 | `‖c1-c2‖ - (r1±r2) = 0` |
| **Concentric** | 2 | `c1 - c2 = 0` |
| **Symmetric** | 2 | midpoint on axis + (p2-p1) ⊥ axis |
| **Equal radius** | 1 | `r1 - r2 = 0` |
| **Point on line** | 1 | `(p - start) × direction = 0` |
| **Point on circle** | 1 | `‖p - center‖ - r = 0` |
| **Coplanar** (3D) | 1 | `(p - origin) · normal = 0` |

Reference implementation: [SolveSpace `src/system.cpp`](https://github.com/solvespace/solvespace/blob/master/src/system.cpp)

### Priority ordering for jaxCAD

**Tier 1 — Needed for any useful sketch:**

```python
# CoincidentConstraint: p1 == p2 (connect line endpoints)
class CoincidentConstraint(Constraint):
    def compute_residual(self, pv):
        return pv[self.p1.name] - pv[self.p2.name]   # shape (3,), 3 DOF
    def dof_reduction(self): return 3

# FixedConstraint: pin a point to a known location (eliminates rigid-body DOF)
class FixedConstraint(Constraint):
    def compute_residual(self, pv):
        return pv[self.point.name] - self.target       # shape (3,)
    def dof_reduction(self): return 3

# HorizontalConstraint
class HorizontalConstraint(Constraint):
    def compute_residual(self, pv):
        return pv[self.line.end.name][1] - pv[self.line.start.name][1]  # scalar
    def dof_reduction(self): return 1

# VerticalConstraint
class VerticalConstraint(Constraint):
    def compute_residual(self, pv):
        return pv[self.line.end.name][0] - pv[self.line.start.name][0]  # scalar
    def dof_reduction(self): return 1
```

**Tier 2 — High value for 2D sketch:**

```python
# PointOnLineConstraint: scalar cross product = 0
class PointOnLineConstraint(Constraint):
    def compute_residual(self, pv):
        a = pv[self.line.end.name] - pv[self.line.start.name]
        b = pv[self.point.name]    - pv[self.line.start.name]
        return a[0]*b[1] - a[1]*b[0]   # 2D cross product z-component
    def dof_reduction(self): return 1

# TangentConstraint(circle, line): dist(center, line) = radius
class TangentConstraint(Constraint):
    def compute_residual(self, pv):
        c  = pv[self.circle.center.name][:2]
        r  = pv[self.circle.radius.name]
        p1 = pv[self.line.start.name][:2]
        p2 = pv[self.line.end.name][:2]
        d = p2 - p1
        cross = d[0]*(p1[1]-c[1]) - d[1]*(p1[0]-c[0])
        return cross**2 / jnp.dot(d,d) - r**2   # squared to avoid sqrt singularity
    def dof_reduction(self): return 1

# EqualLengthConstraint
class EqualLengthConstraint(Constraint):
    def compute_residual(self, pv):
        l1 = jnp.linalg.norm(pv[self.line1.end.name] - pv[self.line1.start.name])
        l2 = jnp.linalg.norm(pv[self.line2.end.name] - pv[self.line2.start.name])
        return l1 - l2
    def dof_reduction(self): return 1
```

**Tier 3 — Advanced:**
- `SymmetricConstraint`, `CollinearConstraint`, `EqualRadiusConstraint`, `CoplanarConstraint` (3D)

### Numerical improvements to existing constraints

**`DistanceConstraint`** — current `‖p1-p2‖ - d` has a Jacobian singularity when `p1 == p2`. Fix:
```python
# Squared form: no sqrt, no singularity at d > 0
return jnp.dot(p1-p2, p1-p2) - d**2
```

**`AngleConstraint`** — current `arccos(cos_angle) - theta` has infinite derivative at 0° and 180°. Fix:
```python
# cos form: smooth everywhere, gradient well-conditioned
return jnp.dot(v1_norm, v2_norm) - jnp.cos(self.angle.value)
```

---

## Part 2: Better solvers

### Why the current Newton-Raphson fails

1. **No step-size control**: pure Newton steps can diverge far from solution.
2. **Not JIT-compiled**: the Python `for _ in range` + concrete `jnp.linalg.norm(r) < tol` check prevents XLA compilation.
3. **lstsq on every step**: SVD-based overkill for exactly-constrained square systems.
4. **No bounds enforcement**: `Parameter.bounds` is stored but never used.

### Drop-in replacement: `optimistix.LevenbergMarquardt`

LM interpolates between Gauss-Newton (fast near solution) and gradient descent (stable far away) via damping parameter λ. It handles:
- Singular/near-singular Jacobians (degenerate configurations).
- Over-constrained systems (finds least-squares solution).
- Bounded step via trust-region radius.

`optimistix` is **already installed** (v0.1.0). Tested and verified to work with jaxCAD's residual pattern.

```python
import optimistix as optx
import lineax as lx

def solve_lm(
    residual_fn,
    x0: Array,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = 256,
    throw: bool = True,
) -> tuple[Array, any]:
    """Solve residual_fn(x) ≈ 0 via Levenberg-Marquardt."""
    def _fn(x, args):
        return residual_fn(x)

    solver = optx.LevenbergMarquardt(
        rtol=rtol, atol=atol,
        linear_solver=lx.AutoLinearSolver(well_posed=False),
    )
    adjoint = optx.ImplicitAdjoint(
        linear_solver=lx.AutoLinearSolver(well_posed=False),
    )
    sol = optx.least_squares(
        _fn, solver, y0=x0, args=None,
        max_steps=max_steps,
        adjoint=adjoint,
        throw=throw,
    )
    return sol.value, sol.result
```

**`ImplicitAdjoint` is critical**: it makes differentiation through the solve O(n) (via the implicit function theorem) rather than O(n × max_steps). This means `jax.grad(downstream_loss)(constraint_parameters)` works efficiently — e.g., "how does the shape's volume change if I increase a distance constraint by ε?".

### JIT compatibility

`optimistix` uses `jax.lax.while_loop` internally — fully JIT-compatible. Wrap `solve_constraints` with `equinox.filter_jit`:

```python
import equinox as eqx

@eqx.filter_jit
def solve_constraints_jit(sdf, *, rtol=1e-6, atol=1e-6):
    ...
```

### Bounds handling

Current `Parameter.bounds` is stored but ignored. Two approaches:

**Variable transformation (recommended for LM):**
```python
def apply_bounds(x, lower, upper):
    """Map ℝ → (lower, upper) via sigmoid."""
    return lower + (upper - lower) * jax.nn.sigmoid(x)

def inverse_bounds(x, lower, upper):
    """Map (lower, upper) → ℝ via logit."""
    t = (x - lower) / (upper - lower)
    return jnp.log(t) - jnp.log(1.0 - t)

# Solve in unconstrained space, map back:
x0_unconstrained = inverse_bounds(x0_bounded, lower, upper)
x_solved = solve_lm(lambda t: residual_fn(apply_bounds(t, lower, upper)), x0_unconstrained)
x_bounded = apply_bounds(x_solved, lower, upper)
```

**Box-constrained Newton** (`optimistix.Newton` supports `lower`/`upper` options directly).

### Robust initialization

The current code passes initial parameter values as `x0`. Better strategies:

1. **Incremental constraint adding**: add constraints one at a time, solving after each. Dramatically better conditioning.
2. **Connected-component decomposition**: solve independent sub-systems separately (see Part 3).
3. **Random restarts**: retry with small perturbations when convergence fails.

### scipy fallback (non-JIT)

For debug/fallback, `scipy.optimize.least_squares(method='lm')` with a JAX-computed Jacobian:

```python
from scipy.optimize import least_squares as scipy_lm

def scipy_fallback(residual_fn, x0, *, tol=1e-6):
    result = scipy_lm(
        fun=lambda x: np.array(residual_fn(jnp.array(x))),
        x0=np.array(x0),
        jac=lambda x: np.array(jax.jacobian(residual_fn)(jnp.array(x))),
        method='lm', ftol=tol, xtol=tol,
    )
    return jnp.array(result.x), result.success
```

---

## Part 3: Under/over-constrained handling

### Current behavior

```python
if remaining > 0:
    raise ValueError(f"Under-constrained: {remaining} DOF remaining.")
if remaining < 0:
    raise ValueError(f"Over-constrained by {-remaining} equations.")
```

Hard gate before any solving. No information about *which* parameters or constraints are problematic.

### Better API: `ConstraintSolution` result object

```python
from dataclasses import dataclass, field
from enum import Enum

class ConstraintStatus(Enum):
    SOLVED            = "solved"
    UNDER_CONSTRAINED = "under_constrained"
    OVER_CONSTRAINED  = "over_constrained"
    FAILED            = "failed_convergence"

@dataclass
class ConstraintSolution:
    status: ConstraintStatus
    params: dict               # best-effort solved parameter values
    residual_norm: float
    free_dof: int              # remaining unconstrained DOF (0 = exactly constrained)
    violated_constraints: list # constraints with residual > tol
    message: str
```

### Under-constrained: solve anyway

Remove the DOF check. LM naturally finds the minimum-residual + minimum-norm solution. For under-determined systems (more unknowns than equations), LM finds the solution closest to `x0` — i.e., "minimum displacement from initial guess", which is exactly the right behavior for interactive use.

Optional: add soft regularization to pin unconstrained directions to initial values:
```python
def residual_regularized(x, x0, alpha=1e-3):
    return jnp.concatenate([
        constraint_residuals(x),
        alpha * (x - x0),          # soft pin to initial position
    ])
```

### Over-constrained: solve and report conflicts

Remove the check. LM minimizes `‖r‖²`. After solving, examine per-constraint residuals:

```python
r_chunks = split_residuals_by_constraint(residual_fn, x_solved, constraints)
violated = [(c, float(jnp.linalg.norm(r))) for c, r in r_chunks if ... > tol]
```

### Conflict detection via Jacobian rank

```python
J = build_constraint_jacobian(x, param_values)
U, s, Vt = jnp.linalg.svd(J, full_matrices=False)
rank = int(jnp.sum(s > 1e-10 * s[0]))
n_redundant = J.shape[0] - rank
# U[:, rank:] columns identify which constraint combinations are redundant
```

### Connected-component decomposition

Solve independent sub-systems separately for better performance and initialization:

```python
def get_connected_components(graph, param_list):
    """BFS on constraint-parameter bipartite graph."""
    param_names = {p.name for p in param_list}
    adj = {p.name: set() for p in param_list}
    for c in graph.constraints:
        fps = [p.name for p in c.get_parameters() if p.name in param_names]
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                adj[fps[i]].add(fps[j])
                adj[fps[j]].add(fps[i])
    visited, components = set(), []
    for start in param_names:
        if start not in visited:
            comp, queue = set(), [start]
            while queue:
                n = queue.pop()
                if n not in visited:
                    visited.add(n); comp.add(n)
                    queue.extend(adj[n] - visited)
            components.append(comp)
    return components
```

### Backward-compatible API

```python
def solve_constraints(sdf, *, tol=1e-6, max_steps=256, strict=False):
    """
    Args:
        strict: If True, raise ValueError for non-exactly-constrained systems
                (preserves old behavior). Default False: return ConstraintSolution.
    """
```

---

## Part 4: 2D sketch constraints

### Architecture question: 3D vectors in a 2D sketch

jaxCAD's `Vector` is always 3D. For 2D sketching, two options:

**Option A (near-term, no refactor)**: enforce z=0 with inline residuals per constraint. Each 2D constraint also zeroes the z-components of its parameters.

**Option B (recommended long-term)**: add a `Vector2` parameter type:
```python
@dataclass
class Vector2(Parameter):
    """2D point (x, y) for sketch geometry. 2 DOF each."""
    @property
    def xy(self) -> Array:
        return self.value   # shape (2,)
```
This gives correct DOF counting (2 DOF per point, not 3) and cleaner constraint residuals.

### Most important 2D constraints for jaxCAD's geometry

**For `Line(start: Vector, end: Vector)`:**
- `CoincidentConstraint(line1.end, line2.start)` — *most used constraint in practice*
- `HorizontalConstraint(line)`, `VerticalConstraint(line)`
- `EqualLengthConstraint(line1, line2)`
- `DistanceConstraint(line.start, line.end, length)` — already exists

**For `Circle(center: Vector, radius: Scalar, normal: Vector)`:**
- `ConcentricConstraint(c1, c2)` — 2 equations: `c1.center - c2.center = 0`
- `EqualRadiusConstraint(c1, c2)` — 1 equation: `c1.r - c2.r = 0`
- `PointOnCircleConstraint(p, circle)`
- `TangentConstraint(line, circle)` — see residual formula in Part 1

### How professional 2D sketch solvers work

| Solver | Architecture | Notes |
|---|---|---|
| **SolveSpace** (https://github.com/solvespace/solvespace) | Newton-Raphson + BFGS fallback, dense Jacobian | Open-source MIT, C++; Python bindings: `pip install python-solvespace` |
| **FreeCAD Sketcher** | `planegcs` solver (https://github.com/FreeCAD/FreeCAD/tree/master/src/Mod/Sketcher/App/planegcs), rank-revealing QR | 22 constraint types in `Constraints.h` |
| **Onshape** | D-Cubed 2D DCM (Siemens DISW), structural decomposition | Proprietary |

All three use Newton-Raphson (or variants) as the numerical core. The key investment is in:
1. **Incremental constraint solving** — add one constraint at a time.
2. **Real-time DOF counter** in the UI.
3. **Conflict visualization** — highlight over-constrained geometry in red.

### python-solvespace as a reference/fallback

`pip install python-solvespace` gives a Python API to the C++ SolveSpace solver:

```python
from python_solvespace import SolveSpaceSystem, Entity, Constraint, ResultFlag

sys = SolveSpaceSystem()
wp = sys.create_2d_base()
p1 = sys.add_point_2d(0, 0, wp)
p2 = sys.add_point_2d(2, 0, wp)
sys.distance(p1, p2, 2.0, wp)  # DistanceConstraint
result = sys.solve()
if result == ResultFlag.OKAY:
    print(sys.params(p2.params[0]), sys.params(p2.params[1]))
```

This can serve as a validation oracle for jaxCAD's constraint solver.

---

## Summary: recommended changes

### Priority 0 (highest impact)
1. **Replace `newton_raphson` with `optimistix.LevenbergMarquardt`** in `cadjoint/constraints/solve.py` — use `ImplicitAdjoint` for differentiability through the solve.
2. **Remove hard DOF check** — return `ConstraintSolution` with status field instead of raising.
3. **Add `CoincidentConstraint`** — the most-used constraint in any sketch workflow.
4. **Add `FixedConstraint`** — needed to eliminate rigid-body DOF and make sketches well-determined.

### Priority 1
5. **Fix `DistanceConstraint`** to use squared form (eliminate singularity at `p1 == p2`).
6. **Fix `AngleConstraint`** to use `cos` form (eliminate singularity at 0°/180°).
7. **Add `HorizontalConstraint`, `VerticalConstraint`**.
8. **Add `PointOnLineConstraint`, `PointOnCircleConstraint`**, `TangentConstraint(line, circle)`.
9. **Add bounds support** via sigmoid transformation in `solve_constraints`.

### Priority 2
10. Add `EqualLengthConstraint`, `MidpointConstraint`, `ConcentricConstraint`.
11. Add `Vector2` parameter type for 2D sketching.
12. Add connected-component decomposition in `ConstraintGraph`.
13. Add conflict identification (per-constraint residual reporting).

---

## Key references

| Resource | URL |
|---|---|
| optimistix docs | https://docs.kidger.site/optimistix/ |
| lineax docs | https://docs.kidger.site/lineax/ |
| SolveSpace source | https://github.com/solvespace/solvespace |
| FreeCAD planegcs | https://github.com/FreeCAD/FreeCAD/tree/master/src/Mod/Sketcher/App/planegcs |
| python-solvespace | https://pypi.org/project/python-solvespace/ |
| Nocedal & Wright "Numerical Optimization" | Ch. 4 (trust region), Ch. 10 (Levenberg-Marquardt) |
