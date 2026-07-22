# jaxCAD Research: Improvement Directions

Three research areas for advancing jaxCAD beyond its current state.

---

## [1. GPU Rendering: GLSL / Warp](./rendering-gpu.md)

Improving interactive and batch forward rendering. Differentiable inverse
rendering is explicitly deferred; current work prioritizes runtime, fidelity,
and maintainable renderer structure.

| Approach | Use case | Effort |
|---|---|---|
| **GLSL + moderngl** | Interactive real-time visualization at 60fps | Low — add `to_glsl()` visitor to SDF tree |
| **NVIDIA Warp** | Fast batch rendering with early ray termination (~5–10× speedup) | Medium — needs code generator |
| **Slang + SlangPy** | Deferred inverse-rendering experiment | Out of current scope |

**Recommended first step**: Add `to_glsl()` to `SDF` base + subclasses, wire into `render_realtime()` via `moderngl`. Zero heavy dependencies; free parameters become GLSL uniforms (no recompilation on parameter change).

---

## [2. Improved Constraint System](./constraints.md)

Richer constraint types, a better solver, and graceful under/over-constrained handling.

### New constraint types (priority order)
1. `CoincidentConstraint(p1, p2)` — *most used; needed to connect lines*
2. `FixedConstraint(p, target)` — eliminates rigid-body DOF
3. `HorizontalConstraint`, `VerticalConstraint`
4. `PointOnLineConstraint`, `PointOnCircleConstraint`, `TangentConstraint(line, circle)`
5. `EqualLengthConstraint`, `MidpointConstraint`, `ConcentricConstraint`

### Solver upgrade
Replace the current Python `for` loop Newton-Raphson with **`optimistix.LevenbergMarquardt`** (already installed, v0.1.0). Key benefits:
- Handles ill-conditioned Jacobians (degenerate configurations).
- JIT-compatible (`jax.lax.while_loop` internally).
- **`ImplicitAdjoint`** makes differentiation *through* the constraint solve O(n) — enables `jax.grad(volume)(constraint_parameters)`.
- Naturally handles under- and over-constrained systems.

### DOF handling
Remove the hard `ValueError` gate. Return a `ConstraintSolution(status, params, residual_norm, free_dof, violated_constraints)` instead. Keep `strict=True` as a backward-compatible option.

### Numerical fixes
- `DistanceConstraint`: use squared form `‖p1-p2‖² - d² = 0` (removes sqrt singularity).
- `AngleConstraint`: use `cos` form `v1·v2 - cos(θ) = 0` (removes arccos singularity at 0°/180°).

---

## [3. Interactive UI: Draggable Objects](./interactive-ui.md)

Adding drag-to-manipulate interaction for jaxCAD objects in Python/Jupyter.

### The core loop (already supported by jaxCAD's architecture)
```python
p.value = new_value                  # user drags
solved  = solve_constraints(scene)   # snap to constraint manifold
image   = render(scene)              # re-render
```

### Recommended phases

| Phase | What | Tool | Effort |
|---|---|---|---|
| 1 | **2D sketch editor** with draggable constrained points | `ipympl` + matplotlib callbacks | 1–2 days |
| 2 | **Scalar parameter sliders** (radius, angle, etc.) | Marimo reactive sliders | ~half day |
| 3 | **3D drag handles** in Jupyter notebook | `anywidget` + three.js `DragControls` | 3–5 days |
| 4 | Shareable web app | FastAPI + three.js | 3–5 days |
| 5 | Desktop app with professional gizmo | `imgui-bundle` + ImGuizmo | 4–7 days |

The bridge from jaxCAD's parameter system to any UI framework is `extract_parameters(sdf)`, which already returns `{path: Parameter}` with `.name`, `.value`, `.free`, and `.bounds`.

---

## Dependencies audit

Libraries needed for each direction vs. what's already installed:

| Direction | Need to add | Already installed |
|---|---|---|
| GLSL rendering | `moderngl` | — |
| Warp rendering | `warp-lang` (NVIDIA GPU) | — |
| Slang rendering | `slangpy`, `torch` | — |
| LM solver | — | `optimistix`, `lineax`, `equinox` |
| 2D sketch UI | — | `matplotlib`, `ipympl` (via jupyter) |
| 3D drag handles | `anywidget` | `scikit-image` (for marching cubes) |
| pyvista picking | `pyvista` | — |
| Web app | `fastapi`, `uvicorn` | — |
