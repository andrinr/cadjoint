# Constraints + Optimization via Automatic Differentiation

> Research into unifying constraint satisfaction, constrained optimization, and design sensitivity in jaxCAD using JAX autodiff.

## The three modes that need to be unified

| Mode | Problem | Current status |
|---|---|---|
| **1 — Constraint satisfaction** | Find x s.t. F(x, θ) = 0 | Works, but not JIT-compatible or differentiable |
| **2 — Constrained optimization** | min L(x) s.t. F(x, θ) = 0 | Partial — null-space GD with linearization drift |
| **3 — Design sensitivity** | d/dθ L(x*(θ)) | Not implemented |

The unification: replace `newton_raphson` with `optimistix.LevenbergMarquardt` + `ImplicitAdjoint`, thread constraint parameters through `residual_fn` as `args`. Modes 1–3 then work through the same computation graph. Estimated change: ~30 lines in `solve.py`.

---

## Environment (confirmed installed)

- `optimistix 0.1.0` — LM, Gauss-Newton, Dogleg, Newton + `ImplicitAdjoint`
- `lineax 0.1.0` — JAX linear solvers (`AutoLinearSolver`)
- `equinox 0.13.6` — `filter_jit`, pytree utilities
- `jaxopt` — **NOT installed** (superseded by `optimistix` for this use case)
- `optax` — **NOT installed** (add for outer-loop Adam optimizer)

---

## Part 1: Why `jax.grad` does not work through the current solver

The current `newton_raphson` in `solve.py` has two fundamental problems:

### Problem 1: Python `for` loop unrolls the gradient tape

```python
for _ in range(max_iter):   # ← Python loop = unrolled by JAX tracing
    J = jax.jacobian(residual_fn)(x)
    ...
```

JAX traces through Python loops by unrolling them. With `max_iter=100` and n=20 parameters, the backward pass includes 100 copies of the Jacobian computation. Memory is O(max_iter × n²). In practice this OOMs or produces a ≈100× slower backward pass.

### Problem 2: Early exit concretizes a traced value

```python
if jnp.linalg.norm(r) < tol:   # ← converts JAX array to Python bool
    return x
```

This forces concrete evaluation inside a JAX trace. Under `jax.grad` or `jax.jit`, JAX either raises `ConcretizationTypeError` or silently traces only one branch, producing wrong gradients.

**In practice:** calling `jax.grad(fn_using_solve_constraints)(theta)` either fails or produces incorrect gradients because the early-exit branch is not accounted for in the backward pass.

---

## Part 2: Implicit Differentiation (IFT) — the correct solution

### The mathematics

Given a constraint system `F(x, θ) = 0` where x are geometric parameters and θ are constraint parameters (distances, angles), the implicit function theorem gives:

```
dx*/dθ = -(∂F/∂x)⁻¹ (∂F/∂θ)
```

**Cost**: one linear solve of size n×n — O(n) regardless of Newton iterations.

In the reverse-mode VJP form used by `jax.grad`:
```
g_θ = -g_xᵀ (∂F/∂x)⁻¹ (∂F/∂θ)
```

where `g_x = dL/dx` is the upstream gradient. This is what `optimistix.ImplicitAdjoint` computes.

### `optimistix.ImplicitAdjoint` — drop-in for jaxCAD

```python
import optimistix as optx
import lineax as lx

def solve_constraints_differentiable(residual_fn, x0, theta):
    """
    Differentiably solve F(x, theta) = 0 with respect to theta.

    The 'args' parameter is what ImplicitAdjoint differentiates through.
    residual_fn signature must be: fn(x, args) -> residuals
    """
    solver = optx.LevenbergMarquardt(
        rtol=1e-8, atol=1e-8,
        linear_solver=lx.AutoLinearSolver(well_posed=False),
    )
    adjoint = optx.ImplicitAdjoint(
        linear_solver=lx.AutoLinearSolver(well_posed=False),
    )
    sol = optx.least_squares(
        residual_fn, solver,
        y0=x0,
        args=theta,         # ← IFT differentiates through theta
        max_steps=256,
        adjoint=adjoint,
        throw=False,
    )
    return sol.value, sol.result
```

**`lineax.AutoLinearSolver(well_posed=False)`** handles rank-deficient Jacobians (under-constrained systems) by falling back to a pseudoinverse solve.

### What this enables

```python
# Design sensitivity: how does volume change when a distance constraint changes?
d_volume_d_distance = jax.grad(
    lambda d: volume(solve_and_render(scene_with_distance_d))
)(2.0)

# Inverse design: fit constraint parameters to a target rendering
loss = lambda theta: render_loss(solve_and_render(scene, theta), target_image)
grad_theta = jax.grad(loss)(theta_init)

# Bilevel: outer loop optimizes constraint params, inner solves geometry
theta_opt = gradient_descent(jax.grad(lambda t: objective(solve(F, t)))(theta))
```

### Required change to `solve.py`

The `residual_fn` must accept constraint parameters as `args` instead of closing over `self.distance.value`:

```python
# Current (not differentiable w.r.t. constraint params):
def residual_fn(x_flat: Array) -> Array:
    ...
    return jnp.concatenate([
        jnp.atleast_1d(c.compute_residual(param_values))  # closes over c.distance.value
        for c in graph.constraints
    ])

# New (differentiable w.r.t. constraint params):
def residual_fn(x_flat: Array, constraint_params: Array) -> Array:
    ...
    return jnp.concatenate([
        jnp.atleast_1d(c.compute_residual_parametric(param_values, constraint_params[i]))
        for i, c in enumerate(graph.constraints)
    ])

theta = jnp.array([c.get_constraint_value() for c in graph.constraints])
x_solved, _ = solve_constraints_differentiable(residual_fn, x0, theta)
```

Each constraint needs a `compute_residual_parametric(self, param_values, constraint_value)` method where `constraint_value` is a JAX array (the distance, angle, etc.) passed from outside rather than read from `self`.

---

## Part 3: Null-space optimization — analysis and fixes

### What the current approach does

`graph.extract_free_dof(param_list)` computes the SVD of the constraint Jacobian J and returns:
- `null_space N ∈ ℝⁿˣ⁽ⁿ⁻ᶜ⁾` — right null space (directions that keep constraints satisfied to first order)
- `reduced_params α` — coordinates in the null space

Moving as `x(α) = x₀ + N·α` keeps `J · (N·α) = 0`, so constraints are satisfied **to first order**.

### The linearization drift problem

For nonlinear constraints (like `DistanceConstraint`), moving along `N·α` for large `|α|` accumulates constraint error:

```
F(x₀ + Nα) = O(|α|²)
```

For a distance constraint with `d=1`, after a gradient step of size ε=1.0, the distance error is ε²/(2d) = 0.5. After 20 steps this is significant.

### Fix: gradient step + Newton re-projection

```python
def constrained_gradient_step(x, loss_fn, residual_fn, lr):
    """Gradient descent step on constraint manifold via retraction."""
    # Step 1: gradient step in full parameter space
    grad = jax.grad(loss_fn)(x)
    x_trial = x - lr * grad

    # Step 2: project back to constraint manifold (one Newton step)
    J = jax.jacobian(residual_fn)(x_trial)
    r = residual_fn(x_trial)
    # Minimum-norm correction (pseudoinverse): Jᵀ(JJᵀ)⁻¹(-r)
    delta = J.T @ jnp.linalg.solve(J @ J.T, -r)
    return x_trial + delta
```

### Re-linearization schedule for null-space approach

```python
def optimize_on_manifold(loss_fn, residual_fn, param_list, n_steps=100,
                          relinearize_every=5, lr=0.01):
    """Null-space gradient descent with periodic re-linearization."""
    from cadjoint.constraints.graph import ConstraintGraph

    graph = ConstraintGraph.from_parameters(param_list)
    base_point = jnp.concatenate([p.xyz for p in param_list])
    _, null_space = graph.extract_free_dof(param_list)
    reduced = jnp.zeros(null_space.shape[1])  # start at zero offset from base_point

    for step in range(n_steps):
        def loss_reduced(alpha):
            full = base_point + null_space @ alpha
            return loss_fn(full)

        reduced = reduced - lr * jax.grad(loss_reduced)(reduced)

        if (step + 1) % relinearize_every == 0:
            # Move base_point to current full params, re-project to manifold
            full = base_point + null_space @ reduced
            J = jax.jacobian(residual_fn)(full)
            r = residual_fn(full)
            delta = J.T @ jnp.linalg.solve(J @ J.T, -r)
            base_point = full + delta  # now on manifold
            # Recompute null space at new base point
            _, s, Vt = jnp.linalg.svd(J, full_matrices=True)
            rank = int(jnp.sum(s > 1e-10))
            null_space = Vt.T[:, rank:]
            reduced = jnp.zeros(null_space.shape[1])

    return base_point + null_space @ reduced
```

---

## Part 4: Constrained optimization methods

### Method comparison for jaxCAD (5–20 params, equality constraints)

| Method | Constraint satisfaction | JAX-native | Conditioning | Recommended for |
|---|---|---|---|---|
| Null-space GD (current) | First-order only | Yes | Good | Interactive dragging, small steps |
| **Augmented Lagrangian** | Exact at convergence | Yes | Good | General objective optimization |
| **SQP** | Exact at each step | Yes | Excellent | Small exact problems (n ≤ 30) |
| Penalty `loss + λ‖r‖²` | Approximate | Yes | Poor (λ large) | Debugging / first pass only |
| Projected gradient | Exact (with re-projection) | Yes | Good | Online/interactive optimization |

### Augmented Lagrangian (recommended for general use)

```python
def augmented_lagrangian_step(x, lam, loss_fn, residual_fn, rho=10.0, lr=0.01, inner_steps=20):
    """One outer AL step: inner minimize + dual update."""
    def augmented(x):
        r = residual_fn(x)
        return loss_fn(x) + jnp.dot(lam, r) + (rho / 2) * jnp.dot(r, r)

    for _ in range(inner_steps):
        x = x - lr * jax.grad(augmented)(x)

    lam = lam + rho * residual_fn(x)
    return x, lam

# Outer loop
lam = jnp.zeros(n_constraints)
for outer_step in range(50):
    x, lam = augmented_lagrangian_step(x, lam, loss_fn, residual_fn)
    rho = min(rho * 1.1, 1e6)  # increase rho gradually
```

The dual variable `lam` corrects the first-order error, so ρ does not need to be increased aggressively (avoids the conditioning blow-up of pure penalty).

### SQP — exact quadratic convergence for small problems

For n ≤ 30 (typical jaxCAD scene), SQP solves the KKT system exactly at each step:

```python
def sqp_step(x, loss_fn, residual_fn):
    """One SQP step: solve KKT system for joint (x, λ) update."""
    g = jax.grad(loss_fn)(x)           # ∇L  (n,)
    H = jax.hessian(loss_fn)(x)        # ∇²L (n, n)
    J = jax.jacobian(residual_fn)(x)   # ∂F/∂x (c, n)
    r = residual_fn(x)                 # F(x) (c,)

    n, c = x.shape[0], r.shape[0]

    # KKT matrix and RHS
    kkt = jnp.block([[H, J.T], [J, jnp.zeros((c, c))]])
    rhs = jnp.concatenate([-g, -r])
    sol = jnp.linalg.solve(kkt, rhs)

    return x + sol[:n]   # quadratic convergence, exact constraints each step
```

---

## Part 5: Design sensitivity — end-to-end example

### The computation

Given:
- `x*(θ)` = point positions satisfying constraints with parameters θ (distances, angles)
- `L(x*(θ))` = scalar objective (volume, surface area, etc.)

```
dL/dθ = (∂L/∂x) · (dx*/dθ) = -(∂L/∂x) · (∂F/∂x)⁻¹ · (∂F/∂θ)
```

Computed as: 1 forward solve + 1 adjoint linear solve. O(n) total, independent of iteration count.

### Full working pattern

```python
import jax, jax.numpy as jnp, optimistix as optx, lineax as lx
from cadjoint import functionalize, extract_parameters
from cadjoint.geometry import Vector
from cadjoint.sdf import Sphere, Translate

def volume_of_sdf(sdf_fn, resolution=32):
    """Estimate SDF volume by signed voxel counting."""
    x = jnp.linspace(-3, 3, resolution)
    pts = jnp.stack(jnp.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = jax.vmap(sdf_fn)(pts) < 0
    return jnp.sum(inside) * (6.0 / resolution) ** 3

def make_residual_fn(param_list, graph):
    """Build residual_fn(x_flat, theta) -> residuals for IFT."""
    def fn(x_flat, theta):
        offset, pv = 0, {}
        for p in param_list:
            sz = p.value.size
            pv[p.name] = x_flat[offset:offset+sz]
            offset += sz
        return jnp.concatenate([
            jnp.atleast_1d(c.compute_residual_parametric(pv, theta[i]))
            for i, c in enumerate(graph.constraints)
        ])
    return fn

def sensitivity_demo():
    """Compute d(volume)/d(distance): how does shape volume change with constraint?"""
    anchor = Vector(jnp.array([0., 0., 0.]), free=False, name="anchor")
    p      = Vector(jnp.array([1., 0., 0.]), free=True,  name="p")
    scene  = Translate(Sphere(radius=0.5), offset=p)

    from cadjoint.constraints import DistanceConstraint
    from cadjoint.constraints.graph import ConstraintGraph
    c = DistanceConstraint(p, anchor, 2.0)

    param_list = [p]
    graph = ConstraintGraph.from_parameters(param_list)
    residual_fn = make_residual_fn(param_list, graph)

    solver  = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8,
                  linear_solver=lx.AutoLinearSolver(well_posed=False))
    adjoint = optx.ImplicitAdjoint(
                  linear_solver=lx.AutoLinearSolver(well_posed=False))

    def volume_at_distance(d):
        theta = jnp.array([d])
        x0 = p.value
        sol = optx.least_squares(residual_fn, solver, y0=x0,
                                 args=theta, max_steps=256,
                                 adjoint=adjoint, throw=False)
        # Rebuild solved_params and render
        free_params, fixed_params = extract_parameters(scene)
        solved = rebuild_params(sol.value, param_list, free_params)
        sdf_fn = functionalize(scene)(solved, fixed_params)
        return volume_of_sdf(sdf_fn)

    # This works after IFT is in place:
    d_vol_d_dist = jax.grad(volume_at_distance)(2.0)
    print(f"d(volume)/d(distance) at d=2.0: {d_vol_d_dist:.4f}")
```

---

## Part 6: Inverse design (bilevel optimization)

### Problem structure

```
min_θ   L(render(x*(θ)))          outer: optimize constraint parameters
s.t.    F(x*(θ), θ) = 0           inner: constraint solve
```

`ImplicitAdjoint` makes the inner solve differentiable w.r.t. θ, collapsing the bilevel problem into standard gradient descent on θ.

```python
def inverse_design(scene, target_image, n_outer=50, lr=0.01):
    """Find constraint parameters that produce a shape matching target_image."""
    free_params, fixed_params = extract_parameters(scene)
    param_list = deduplicated_free_params(free_params)
    graph = ConstraintGraph.from_parameters(param_list)

    residual_fn = make_residual_fn(param_list, graph)
    solver  = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8,
                  linear_solver=lx.AutoLinearSolver(well_posed=False))
    adjoint = optx.ImplicitAdjoint(
                  linear_solver=lx.AutoLinearSolver(well_posed=False))

    # Initial constraint params (all distance values as flat array)
    theta = jnp.array([c.get_constraint_value() for c in graph.constraints])
    x0    = jnp.concatenate([p.xyz for p in param_list])

    def outer_loss(theta):
        sol = optx.least_squares(residual_fn, solver,
                                 y0=x0, args=theta,
                                 max_steps=256, adjoint=adjoint, throw=False)
        solved = rebuild_params(sol.value, param_list, free_params)
        sdf_fn = functionalize(scene)(solved, fixed_params)
        rendered = render_to_array(sdf_fn)
        return jnp.mean((rendered - target_image)**2)

    grad_outer = jax.grad(outer_loss)

    for step in range(n_outer):
        g = grad_outer(theta)
        theta = theta - lr * g
        if step % 10 == 0:
            print(f"Step {step}: loss={outer_loss(theta):.4f}, theta={theta}")

    return theta
```

**Adding Adam for better outer-loop convergence** (`pip install optax`):
```python
import optax
optimizer = optax.adam(lr)
opt_state = optimizer.init(theta)

for step in range(n_outer):
    g = grad_outer(theta)
    updates, opt_state = optimizer.update(g, opt_state)
    theta = optax.apply_updates(theta, updates)
```

---

## Part 7: Bugs in `test_optimization.py`

### Bug 1 (critical) — line 49: missing `grad_fn` call

```python
# WRONG — parentheses are a no-op; grad = current_reduced (the parameters themselves)
grad =  (current_reduced)

# CORRECT
grad = grad_fn(current_reduced)
```

This causes the "optimization loop" to actually compute `current_reduced *= (1 - lr)` — exponential decay toward zero, not gradient descent. The test assertion `final_loss < initial_loss` accidentally passes if the initial point happens to be in the right direction from zero.

### Bug 2 — `reduced_params` initialization in `extract_free_dof`

In `graph.py` line ~249:
```python
reduced_params = null_space.T @ full_params_flat
```

With the parametrization `x(α) = base_point + N·α`, the correct initialization is α=0 (zero offset from base point). The current initialization gives a non-zero α that maps `base_point + N·(Nᵀ base_point)` ≠ `base_point` for most configurations.

Fix: return `jnp.zeros(null_space.shape[1])` as the initial reduced params, or document that `base_point` must be set to zero separately.

### Bug 3 — `compute_null_space` fails under JIT

In `graph.py` line ~171:
```python
rank = jnp.sum(s > tolerance)   # returns a traced int under jit
null_space = V[:, rank:]         # dynamic slice on traced int → ConcretizationTypeError
```

Fix: use `int(jnp.sum(...))` (eagerly materialize rank) or use `jnp.where` masking instead of dynamic slicing.

### Bug 4 — mixed-size parameter Jacobian assembly

In `graph.py` lines ~133–137, column offsets use `local_idx * param_dim`, assuming uniform parameter size. A constraint mixing `Vector` (3D) and `Scalar` (1D) parameters will have wrong column offsets.

Fix: track cumulative offsets per parameter rather than multiplying by a fixed `param_dim`.

---

## Summary: changes needed in `solve.py`

The core change to enable all three optimization modes:

```python
# New solve.py structure (≈30 line change)

import optimistix as optx
import lineax as lx

def solve_constraints(
    sdf, *,
    tol: float = 1e-6,
    max_steps: int = 256,
    constraint_params: Optional[Array] = None,  # NEW: external constraint values
    strict: bool = False,                        # NEW: False = return status, True = raise
) -> ConstraintSolution:

    # ... existing param extraction + graph building ...

    def residual_fn(x_flat, theta):
        # ... same as now but uses theta[i] for constraint i
        # instead of closing over c.distance.value
        ...

    theta = constraint_params or jnp.array([c.get_value() for c in graph.constraints])

    solver  = optx.LevenbergMarquardt(rtol=tol, atol=tol,
                  linear_solver=lx.AutoLinearSolver(well_posed=False))
    adjoint = optx.ImplicitAdjoint(
                  linear_solver=lx.AutoLinearSolver(well_posed=False))

    sol = optx.least_squares(
        residual_fn, solver,
        y0=x0, args=theta,
        max_steps=max_steps,
        adjoint=adjoint,
        throw=strict,
    )

    # Mode 1: x_solved satisfies F(x_solved, theta) = 0
    # Mode 3: jax.grad(loss)(theta) works via ImplicitAdjoint backward pass
    return ConstraintSolution(params=rebuild_params(sol.value, ...), ...)
```

---

## Library additions needed

| Library | Why | Install |
|---|---|---|
| `optax` | Adam/AdaGrad for outer design optimization loop | `pip install optax` |

No other new dependencies. `optimistix`, `lineax`, and `equinox` are already installed and cover everything needed for Modes 1–3.
