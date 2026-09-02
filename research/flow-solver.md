# A differentiable flow solver, and what it would take to cool a real heat sink

Status: working prototype + measured evaluation (2026-09-02). New code lives in
`cadjoint/flow/`, packaged as `cadjoint/fem/tesseracts/flow_brinkman/` and
registered as the `flow_solver` kind. Tests: `tests/flow/` (92 passing).

**Machine**: Apple M5 Max, 48 GB, macOS 25.4, CPython 3.14.5, jax 0.8.2 (CPU
backend), float64 throughout. A `jnp` add over 64 M elements sustains
**208 GB/s** on this box; that is the roofline every throughput number below is
measured against.

**Scene**: `scenes/starter.py` — the parametric finned heat sink, its copper
slug and two bushings, executed inside the capture registries and sampled
through `functionalize(sink)`.

---

## 0. The one-paragraph answer

A steady flow solve **can** join cadjoint's gradient path on the same terms the
thermal solve already does, and it can do so with no mesh anywhere in the loop.
The design enters as a smooth solid fraction `chi(x)` sampled from the scene
SDF on a fixed lattice, drives a Brinkman drag in a D3Q19 BGK lattice Boltzmann
step, and the gradient comes from the implicit function theorem at the
converged fixed point rather than from a tape. Against the starter sink this
reproduces finite differences on real design parameters to **3e-8** and agrees
with a converged unrolled tape to **5.6e-11**, while using **457 MB** where the
tape needs **21 GB**. The two things that had to be got right were not the
adjoint — they were the *shape of `chi`* (§3) and the *stability ceiling on
`omega`* (§4.3), both of which fail silently and expensively if guessed.

---

## 1. XLB: evaluated, and not used

The plan's first choice was [XLB](https://github.com/Autodesk/XLB), Autodesk's
JAX/Warp lattice Boltzmann library. It was installed and driven to a verdict
rather than dismissed on inspection.

### 1.1 It installs, but not against its own declared dependency ranges

`pip install xlb` resolves **xlb 0.3.1** (pure-Python wheel, `requires-python
>=3.11`) and pulls jax 0.11.1, jaxlib 0.11.1, numpy 2.5.2, warp-lang 1.17.0,
pyvista 0.48.4, vtk 9.6.2, trimesh 5.1.0, matplotlib 3.11.1, scipy 1.18.1 and
ruff 0.16.5. Two of those break it immediately:

| break | cause |
|---|---|
| `import xlb` raises `ImportError: cannot import name 'ScopedTimer' from 'warp.utils'` | `xlb/operator/stepper/ibm_stepper.py` imports a symbol removed from warp-lang after 1.10; xlb declares only `warp-lang>=1.10.0`. Pinning `warp-lang==1.10.0` fixes it. |
| `check_bc_overlaps` raises `ValueError: operands could not be broadcast together with shapes (0,) (5120,)` | `xlb/helper/check_boundary_overlaps.py:17` does `index_list[d] += bc.indices[d]`. Under NumPy ≥ 2 a `list += ndarray` goes through NumPy's broadcast path instead of `list.extend`. Passing boundary indices as Python lists avoids it. |

Both were worked around in an **isolated venv**, deliberately: xlb's
`ruff>=0.14.1` and `jax>=0.8.0` would have moved this repo's pinned ruff and
upgraded jax 0.8.2 → 0.11.1 in the shared checkout.

### 1.2 It is fast, and it is genuinely differentiable

32³ lid-driven cavity, JAX backend, `PrecisionPolicy.FP32FP32`, CPU:
**200 steps in 0.13 s = 52.2 MLUPS**.

Differentiability was verified rather than assumed (all against central
differences on the same problem, fp64, 16³):

| quantity | autodiff | finite difference | rel. err |
|---|---:|---:|---:|
| directional derivative w.r.t. populations | −2.24052992e+02 | −2.24052992e+02 | 1.7e-10 |
| `d/d(omega)` of one step | +3.91587320e-03 | +3.91594313e-03 | 1.8e-05 |
| `d/d(omega)` through a 20-step `lax.scan` | +1.86813474e-02 | +1.86814191e-02 | 3.8e-06 |

(An early NaN reading was my own error: XLB's stepper returns `(f_0, f_1)` and
the **second** element is the new state. Reading the first differentiates the
identity.)

### 1.3 Why it still is not the right vehicle here

**XLB has no per-cell force field.** Brinkman penalisation needs a drag
`alpha(x)·u` that varies cell by cell — that *is* the design. XLB's
`ForcedCollision` accepts a single `force_vector` of shape `(d,)` and asserts
`force_vector.shape[0] == velocity_set.d`; `ExactDifference` closes over that
one constant. Supporting a field means editing XLB's collision operators in
both the JAX and Warp backends, at which point the library is being forked, not
used.

Secondarily, XLB offers only the taped gradient. There is no fixed-point or
adjoint-at-convergence path, and §5 is the measurement of why that matters.

**Verdict: fall back to a minimal D3Q19 BGK in pure JAX** (`cadjoint/flow/`),
which is what the plan specified. XLB remains the right thing to revisit if a
per-cell forcing term ever lands upstream — its throughput is 3–4× ours (§7)
and its Warp backend is a GPU story we do not otherwise have.

---

## 2. What was built

```
cadjoint/flow/
  lattice.py    D3Q19 velocities, weights, bounce-back permutation (NumPy constants)
  domain.py     SDF -> chi on a fixed grid: FlowGrid, solid_fraction, profiles
  lbm.py        one differentiable step: collide + Brinkman drag, bounce-back, stream, BCs
  steady.py     pseudo-time march, and the IFT adjoint behind jax.custom_vjp
  objectives.py pressure drop and the convective proxy
  solver.py     FlowConfig / solve / convergence, and the step-closure cache
```

The chain, end to end, is one JAX expression:

```
sketch point -> SDF -> chi = profile(-f(x)/eps) -> alpha = alpha_max*chi
             -> LBM fixed point -> pressure drop / heat proxy
```

Nothing in it contours, meshes, or decides membership. The grid never moves;
the field on it does. That is the property a body-fitted mesh cannot offer, and
it is why `d(pressure drop)/d(fin tip)` exists at all.

---

## 3. The finding that mattered most: `chi` must have compact support

The obvious profile is `sigmoid(-f(x)/eps)` — it is exactly what
`scenes/starter.py` already uses for its `material_volume` regulariser, and
reusing it looked like consistency. It is wrong here, and wrong in a way that
looks like a solver bug.

A sigmoid never reaches zero. In the cells the geometry calls *clearly* open
(`f > 2 eps`) it still leaves `chi` up to **0.118**. `alpha_max` multiplies
that tail along with everything else, so at `alpha_max = 100` the middle of an
open fin channel carries a drag of **11.8** against a kinematic viscosity of
**0.0064**. Raising `alpha_max` to make the solid impermeable makes the fluid
porous at the same rate.

Measured on the starter sink at 32×64×32, pressure drop vs `alpha_max`:

| `alpha_max` | sigmoid | compact profile |
|---:|---:|---:|
| 1 | 2.12e-2 | 5.63e-3 |
| 5 | 5.02e-2 | 6.24e-3 |
| 20 | 1.22e-1 | 6.49e-3 |
| 100 | **diverged** | 6.64e-3 |
| 400 | **diverged** | 6.72e-3 |

The sigmoid column grows without bound and then the duct plugs, density climbs
monotonically and the march NaNs. The compact column *converges* — 1.1% over
the last factor of four. Leakage (mean `|u|` where `chi > 0.8`, as a fraction
of inlet speed) falls like `1/alpha`: 2.2e-2 → 6.1e-3 → 1.9e-3 → 5.0e-4 →
2.0e-4 → 4.3e-5 (α=2000) → 2.7e-6 (α=5e4).

Clamping to exact 0 and 1 costs nothing real: the derivative outside the band
*is* zero, because moving a surface already more than `epsilon` away does not
change that cell's occupancy. Every cell the design can influence sits inside
the band.

### 3.1 Which compact profile: the clamp's smoothness sets the FD order

Clamping puts a join at each band edge. How smooth that join is does not affect
whether the adjoint is *correct* — it affects whether it can be *checked*.
Measured on a box obstacle whose faces land exactly on cell centres (the worst
case; many cells sit precisely on a join), relative error between a central
difference and the adjoint:

| `h` | `smootherstep` (quintic, C²) | `smoothstep` (cubic, C¹) | `sigmoid` (C^∞) |
|---:|---:|---:|---:|
| 1e-2 | 2.57e-3 | 1.64e-1 | 2.57e-6 |
| 1e-3 | 2.59e-5 | 1.93e-2 | 2.43e-8 |
| 1e-4 | 2.60e-7 | 1.97e-3 | 1.11e-9 |
| 1e-5 | 2.90e-9 | 1.97e-4 | 1.29e-9 |

The cubic falls by ten per ten — first order, the signature of a jump in the
second derivative. The quintic falls by a hundred per ten, the second order a
central difference should have. `"smootherstep"` is therefore the default: one
extra multiply, and the gradient becomes checkable. All three remain available
(`cadjoint.flow.PROFILES`); `"sigmoid"` is kept for reproducing the
regulariser's field, where it is the right choice because a volume integral
never multiplies it by 200.

---

## 4. The penalised steady problem

### 4.1 Setup

Duct along `+Y`, inlet at `y = 0`, outlet at `y = NY−1`, halfway bounce-back on
the four lateral walls. Grid 32×64×32 = 65,536 cells over
`origin = (−1.05, −1.2, −0.30)`, `size = (2.1, 2.4, 1.35)` world units, which
leaves clear inlet and outlet runs with the sink blocking ~30% of the
mid-duct cross-section. `epsilon` = half a cell = 0.0235.

`inlet_speed` 0.02 lattice units, `Re` 100 against 32 cells → `nu` 0.0064,
`omega` 1.9260, Ma 0.035. `alpha_max` 200.

### 4.2 Results

| quantity | value |
|---|---:|
| cells | 65,536 |
| throughput | 13.7 MLUPS (6000 steps in 28.7 s) |
| steps to residual 1e-6 / 1e-7 | 2,800 / 4,200 |
| pressure drop | 6.610e-3 |
| empty-duct pressure drop | 5.279e-4 (the sink costs **12.5×**) |
| max \|u\| | 0.0845 (Ma 0.146) |
| density range | 0.9983 – 1.0249 (**2.7%**) |
| leak, mean \|u\| in `chi>0.8` / inlet | 2.8e-4 |

Residual history is monotone and geometric. The 2.7% density variation is the
number that says the answer may be *read* as incompressible; the first working
point tried (inlet 0.05, `alpha_max` 1, sigmoid) reached **39%**, which is not
a flow solution at all.

### 4.3 A stability ceiling on `omega`, now enforced up front

Coarsening the grid at fixed `Re` and fixed inlet speed *lowers* the viscosity
(`nu = u·L_cells/Re`), pushing `omega` toward 2 where BGK loses stability — and
the Brinkman drag brings it on sooner than the textbook limit. Measured on the
starter sink at `alpha_max = 200`:

| grid | `omega` | outcome |
|---|---:|---|
| 24×48×24 | 1.9440 | converges |
| 20×40×20 | 1.9531 | **NaN** |
| 16×32×16 | 1.9623 | **NaN** |

`FlowConfig.__post_init__` now refuses `omega > 1.95` with a message naming the
three ways out. A NaN an hour into a march is a much worse error message than a
`ValueError` at construction.

---

## 5. The adjoint: implicit, verified, and much cheaper

`R(f; theta) = T(f; theta) − f = 0` at convergence. With `A = dT/df`, the
reverse-mode rule solves `(I − A)ᵀ λ = g` matrix-free — `jax.vjp` of a *single*
step at the converged state supplies both `Aᵀ` and the parameter pullback — and
then `dJ/dtheta = λᵀ ∂T/∂theta`. Two solvers are offered because they fail
differently: `"gmres"` is fastest when it converges; `"fixed_point"` is
Richardson on the same system, so it converges exactly when the forward march
does. Disagreement between them means the forward had not converged.

### 5.1 Against finite differences, on real design parameters

Starter sink, 16×32×16, `Re` 30, `alpha_max` 200, `smootherstep`, adjoint via
GMRES. Parameters are the scene's own free `Scalar` / `Vector2` sketch handles;
`h = 1e-5`, central.

| parameter | objective | adjoint | finite difference | rel. err |
|---|---|---:|---:|---:|
| `fin_depth` | pressure drop | +1.6352066e-01 | +1.6352067e-01 | **3.2e-08** |
| `fin_depth` | heat proxy | +2.5522866e-04 | +2.5522857e-04 | 3.5e-07 |
| `fin1_tip_r.x` | pressure drop | +4.9243823e-05 | +4.9243665e-05 | 3.2e-06 |
| `fin1_tip_r.x` | heat proxy | +4.7585354e-07 | +4.7585354e-07 | 5.8e-09 |
| `fin1_tip_r.y` | pressure drop | +1.1359323e-04 | +1.1359316e-04 | 6.0e-07 |
| `fin1_tip_r.y` | heat proxy | +5.6017353e-07 | +5.6017352e-07 | 2.7e-08 |
| `fin2_tip_l.x` | pressure drop | −1.5604409e-02 | −1.5622686e-02 | 1.2e-03 |
| `fin2_tip_l.x` | heat proxy | −1.6280292e-05 | −1.6299516e-05 | 1.2e-03 |

At `h = 1e-4` the same rows read 5.2e-6, 3.1e-5, 5.3e-8, 2.0e-6, 6.6e-6,
3.9e-6, 1.1e-1, 1.1e-1 — so shrinking `h` by ten improves five of the eight by
between 87× and 349×, i.e. the second order a central difference should have.
The discrepancy is the difference scheme's truncation, not the adjoint. Two
rows behave otherwise, both for understood reasons:

- `fin1_tip_r.y` / pressure drop improves only 11×, because at 6.0e-7 it is
  approaching the same floor.
- `fin1_tip_r.x` / pressure drop *worsens*, 5.3e-8 → 3.2e-6. It was already at
  the round-off floor at `h = 1e-4`; halving the step further just amplifies
  cancellation in the difference of two nearly equal solves. This is the usual
  finite-difference U-curve, and it is evidence the adjoint is right rather
  than evidence it is wrong — the agreement at the *bottom* of the curve is
  5.3e-8.

`fin2_tip_l.x` is the slowest to converge (that fin face lands nearest a band
edge) but tracks second order cleanly, 90× per decade.

### 5.2 Against an unrolled tape, with memory

12×24×12 = 3,456 cells, `alpha_max` 1 (chosen so the march converges inside a
tape that fits), box obstacle, objective `pressure_drop + 50·heat_proxy`.

| gradient | value | rel. diff vs IFT | peak RSS | tape at 19·cells·8 B/step |
|---|---:|---:|---:|---:|
| **IFT** | +9.7368441121e+00 | — | **+457 MB** | — |
| unrolled 500 | +9.6497293665e+00 | 8.95e-03 | +1,486 MB | 263 MB |
| unrolled 1000 | +9.7385522073e+00 | 1.75e-04 | +2,917 MB | 525 MB |
| unrolled 2000 | +9.7368443369e+00 | 2.31e-08 | +5,232 MB | 1,051 MB |
| unrolled 4000 | +9.7368441116e+00 | **5.58e-11** | **+10,849 MB** | 2,101 MB |

The tape converges *to* the implicit answer as the march it tapes converges —
which is the cleanest possible statement that the two compute the same
derivative. Its memory grows linearly; the adjoint's does not. Note also that
the IFT gradient is a property of the converged state alone: `f0` correctly
receives a **zero** cotangent, where the tape's gradient depends on the initial
guess and on how many iterations were spent.

The regime where this matters is exactly the useful one. At `alpha_max = 200`
the march needs ~5,200 steps at 32×64×32, where a tape would want
**52 GB**; §7 extrapolates that to 5 PB at 128³.

---

## 6. Packaging

`cadjoint/fem/tesseracts/flow_brinkman/` follows the existing packages' layout.
The wire contract is unusually small: the solver plugins beside it ship a
*mesh*; this one ships one array.

- **Differentiable inputs**: `chi` (`(NX,NY,NZ)`), `inlet_velocity` (`(3,)`).
- **Outputs**: `pressure_drop`, `heat_transfer`, both `Differentiable` scalars.
- **Requirements**: `python-pip` provider with a single local `../../../..`.
  `cadjoint.flow` is pure JAX over a fixed array — no PETSc, no gmsh, no
  meshing — so unlike `thermal_jaxfem` it needs no conda and **no new
  `pyproject.toml` extra**.
- **Registration**: kind `flow_solver` added to `KINDS`, `flow_brinkman` to
  `BUILTIN_PACKAGES` and `BUILTIN_DEFAULTS`. That is the whole change to
  `cadjoint/plugins/`.

Loaded through `Tesseract.from_tesseract_api` and driven via
`plugin_for_kind("flow_solver")`, `apply` and `vector_jacobian_product` are
**bit-identical** to the in-process solve and gradient (max abs difference
0.0e+00 on the `chi` gradient; `tests/flow/test_plugin.py` asserts it to
`rel=1e-10`). Both sides run the same `custom_vjp`, so they cannot drift.

---

## 7. What this costs on a GPU

### 7.1 Measured scaling on CPU

| grid | cells | MLUPS | useful GB/s | one state |
|---|---:|---:|---:|---:|
| 16×32×16 | 8,192 | 14.4 | 4.4 | 1.2 MB |
| 24×48×24 | 27,648 | 15.7 | 4.8 | 4.2 MB |
| 32×64×32 | 65,536 | 15.7 | 4.8 | 10.0 MB |
| 40×80×40 | 128,000 | 12.0 | 3.7 | 19.5 MB |
| 48×96×48 | 221,184 | 11.9 | 3.6 | 33.6 MB |

Iterations to a 1e-7 residual grow close to linearly in the linear grid size:
3,300 (N=16), 4,100 (N=24), 5,200 (N=32) — about `119·N + 1,400`.

**The honest caveat**: 4.8 GB/s of useful traffic against 208 GB/s achievable
is **2.3% of roofline**. `stream()` is 19 separate `jnp.roll` calls plus a
`stack`, so each step makes ~19 full passes over the populations where a fused
pull-stream kernel makes one. The CPU numbers therefore understate what the
*algorithm* costs and overstate what *this implementation* costs.

### 7.2 Projection

Minimum traffic per cell-update is 19 populations × 8 B × (read + write) =
**304 B** (152 B in fp32). Extrapolating iterations as above:

| grid | cells | steps to 1e-7 | cell-updates |
|---|---:|---:|---:|
| 96³ | 884,736 | ~12,800 | 1.13e10 |
| 128³ | 2,097,152 | ~16,600 | 3.48e10 |

| scenario | 96³ | 128³ |
|---|---:|---:|
| this CPU, as measured (12 MLUPS) | ~16 min | ~48 min |
| A100 (2.0 TB/s) at today's 2.3% efficiency | ~75 s | ~230 s |
| A100 with a fused stream at 70% of bandwidth | **~2.5 s** | **~7.6 s** |
| the same in fp32 | ~1.2 s | ~3.8 s |

The middle row is the "port nothing, just move it" estimate and is the one to
plan against first; the third is what the algorithm is actually worth and needs
`stream()` rewritten as a single gather. Either way a forward solve at 128³ is
seconds, not minutes, which puts it inside an optimization loop.

**Adjoint memory is the constraint nobody expects.** One state is 134 MB at
96³ and 319 MB at 128³ — but GMRES holds `restart` Krylov vectors:

| solver | 96³ | 128³ |
|---|---:|---:|
| `"gmres"`, `adjoint_restart=40` | 5.4 GB | **12.8 GB** |
| `"fixed_point"` (3 states) | 0.4 GB | 1.0 GB |
| an unrolled tape | 1.7 PB | **5.3 PB** |

So `"gmres"` at 128³ wants an 80 GB card, `"fixed_point"` runs on anything, and
the tape is not a strategy at any budget. Lowering `adjoint_restart` trades
adjoint iterations for memory and is the first knob to reach for on a small
card.

---

## 8. Coupling this to the thermal study

Everything above computes a *flow*. The objective a user actually wants is
**junction temperature under real airflow** — today `scenes/starter.py` fakes
the convection with `Dirichlet(..., 0.0)` on the upper fin field, an idealized
infinite sink that makes taller fins look free.

### 8.1 The cheap coupling: a heat-transfer coefficient into a Robin condition

Replace the Dirichlet patch with a Robin (convective) boundary condition
`−k ∂T/∂n = h(x)·(T − T_ambient)`, where `h` comes from the converged flow.
The standard correlation for forced convection over a surface is
`Nu ∝ Re_x^{1/2} Pr^{1/3}`, i.e. `h ∝ sqrt(|u|)` locally; a defensible
discrete form on this grid is

```
h(x) = h0 * sqrt(|u(x)| / u_ref)     evaluated in the interface band
```

sampled where `|grad chi|` is large, then carried onto the FEM surface nodes.
Both factors are already differentiable: `|u|` through the flow adjoint,
`grad chi` through the SDF. The thermal solve gains a Robin term, which for
`thermal_jaxfem` is a surface mass matrix — jax-fem already assembles the
analogous flux term.

**Cost**: one flow solve per design step, plus the existing thermal solve. The
two adjoints compose through `chi` and `h` without either solver learning about
the other. This is the version to build first.

**What it does not capture**: the air heating up as it passes down the duct.
Downstream fins see warmer air, which is precisely the effect that sets the
useful fin count. A one-dimensional bulk-temperature march along `+Y`, driven
by the same `chi·|u|` integrand, recovers most of it for almost nothing.

### 8.2 The full coupling: conjugate heat transfer on the same lattice

Carry a temperature field on the flow grid and solve advection–diffusion
alongside the momentum solve — a second distribution function (a D3Q7 or D3Q19
scalar lattice) with the conductivity interpolated by `chi` between the solid's
and the air's. Both fields reach a joint fixed point; the IFT machinery in
`steady.py` already generalises to it, since `theta` and the state are both
pytrees.

This is the physically right answer — no correlation, no `h0` to calibrate, the
thermal boundary layer resolved rather than modelled — and it removes the FEM
mesh from the cooling objective entirely. Its costs are real: the scalar
lattice roughly doubles memory and step cost; the solid's thermal diffusivity
is orders of magnitude above the air's, so the coupled system is stiff and the
joint march converges far more slowly than the momentum one alone; and
resolving a thermal boundary layer needs finer cells than resolving the
momentum one at the same `Re·Pr`. It is the right target *after* §8.1 has shown
the loop closes.

### 8.3 The objective, and why both terms are needed

```
minimize   T_junction(theta)  +  w · pressure_drop(theta)
```

`pressure_drop` is not decoration. A sink that maximises surface contact with
moving air by filling the duct with metal strangles the fan driving it; without
the pressure term the optimizer walks straight into a solid block. The two
already pull in opposite directions in the measurements above — the starter
sink costs 12.5× the empty duct's drop — and the fin pitch that a real design
wants lives between them.

The `heat_transfer` proxy shipped today (`∫ chi·|u|`) is the cheap stand-in
until §8.1 lands. Its weakness is worth stating: with a compact `chi` and
`alpha_max = 200`, the drag has already killed `|u|` wherever `chi` is large,
so the integrand is carried by the thin band where both are moderate. It is a
usable descent direction, not a temperature.

---

## 9. What fought back

- **The sigmoid.** §3. It presented as a solver instability (`NaN` at
  `alpha_max ≥ 100`) and was actually a modelling error — the drag was being
  applied to the fluid. Chasing it as a stability bug would have wasted the
  whole exercise. The tell was density climbing *monotonically* rather than
  oscillating: the duct was plugging, not going unstable.
- **A tracer leak through an `lru_cache`.** `step_for` caches its step closure
  on the config (it must — `steady_populations` takes it as a `nondiff_argnum`
  and JAX hashes those by identity). The wall masks were built with `jnp`, so
  the *first* call inside a `jit` cached tracers and every later call leaked
  them. Fixed by building the masks in NumPy, which is also simply correct:
  they are fixed geometry that nothing differentiates.
  `tests/flow/test_solver.py` and `test_lbm.py` both guard it.
- **Coarse grids at fixed `Re`.** §4.3. Silently NaN, now a `ValueError`.
- **XLB's return convention.** `stepper(...)` returns `(f_0, f_1)` with the new
  state *second*; reading the first differentiates the identity and produces a
  confident, wrong verdict. Caught by finite differences.
- **The clamp's smoothness.** §3.1. A gradient that was exactly right looked
  wrong at 1e-3 against a central difference, purely because the cubic profile
  degrades FD to first order.

---

## 10. Where the code is

| file | what |
|---|---|
| `cadjoint/flow/` | the solver (6 modules, ~1,400 lines) |
| `cadjoint/fem/tesseracts/flow_brinkman/` | the Tesseract package |
| `cadjoint/plugins/registry.py` | `flow_solver` kind + built-in spec |
| `tests/flow/` | 92 tests: lattice, domain, lbm, steady, solver, plugin |

Verification: `pytest tests/flow -q` (92 passed),
`pytest tests/fem/test_tesseract_packaging.py -q` (28 passed),
`pytest tests/plugins -q` (54 passed, 1 skipped), ruff clean and formatted.
