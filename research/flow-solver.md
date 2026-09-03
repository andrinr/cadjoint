# A differentiable flow solver, and what it would take to cool a real heat sink

Status: working solver, coupled to a conjugate heat-transfer study and measured
(momentum 2026-09-02, coupling 2026-09-03). Code in `cadjoint/flow/`, packaged as
`cadjoint/fem/tesseracts/flow_brinkman/` and registered as the `flow_solver`
kind. Tests: `tests/flow/` (158 passing) plus
`tests/fem/test_flow_conjugate.py`.

**Read §8 first if you want the coupling**: what it solves, what it was checked
against (a closed form, a two-layer series resistance, an analytic
advection–diffusion column, a textbook Nusselt number, and finite differences
through both adjoints), and **§8.9, what is still not true** — the Reynolds
range, the absent turbulence model, the resolution a trustworthy answer needs
against the resolution the tests use, and how far the Brinkman smearing moves
the result.

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

## 8. Coupling this to the thermal study — built, and measured

§8.1 and §8.2 of the earlier draft set out two options: a heat-transfer
coefficient correlated from `|u|` into a Robin condition on the FEM study
(cheap, needs an `h0` to calibrate), or conjugate heat transfer on the flow
lattice (right, expected to be stiff and slow). **What was built is neither,
quite.** It is conjugate heat transfer on the same lattice, but the energy
equation is discretised by *finite volume and solved once by a Krylov method*
rather than marched as a second distribution function:

```
div(k grad T)  =  (rho cp)_f  u . grad T  -  q
```

with `k` interpolated linearly by `chi` between the air's and the solid's. The
reason for abandoning the second-lattice plan is the reason that plan was
listed as expensive in the first place. The steady energy equation is **linear
in T**, so pseudo-time marching it buys nothing and pays the full stiffness of a
conductivity ratio of several thousand. One matrix-free GMRES solve gets the
same answer in one shot, and its adjoint is `jax.lax.custom_linear_solve`'s
transposed solve — exact, implicit, and composing with the momentum solve's
fixed-point adjoint without either solver knowing the other exists.

Code: `cadjoint/flow/energy.py` (the solve), `cadjoint/flow/regions.py`
(selections on a lattice), `cadjoint/flow/study.py` (`FlowStudy`).

### 8.1 There is no interface condition, and that is the point

A two-domain formulation would have to find the metal–air interface, mesh it,
and match temperature and flux across it — the three things this project exists
not to do. A single-domain formulation with a variable coefficient satisfies
both conditions **by construction**: one unknown field admits no temperature
jump, and a conservative finite-volume flux is shared by the two cells either
side of a face.

The one place a choice is still made is how a *face* between an air cell and a
metal cell conducts. It is the **harmonic** mean of the two cell
conductivities, which is the exact series resistance of the two half-cells.
Measured against the analytic two-layer answer on a 24-cell column:

| `k_solid / k_fluid` | max abs error vs. series resistance |
|---|---|
| 10 | 3.1e-13 |
| 200 | 1.8e-14 |
| 8000 (aluminium in air) | 1.4e-13 |

An arithmetic mean would smear the step over a cell and report a flux wrong by
a factor approaching the ratio. This is the discrete form of "continuity of
heat flux at the interface", and it holds to machine precision.

### 8.2 The coupling is one-way, and in this model that is exact

The momentum solve's inputs are `chi`, the inlet velocity and a viscosity. Not
one of them is a function of temperature. Nothing the energy solve computes can
change the flow, so iterating between them to a fixed point would converge in
one pass *by definition* — one-way is not a first landing here, it is the whole
of the coupling this model contains.

What makes conjugate heat transfer genuinely two-way elsewhere is physics this
model does not carry: buoyancy, and temperature-dependent viscosity and
conductivity. The condition for ignoring buoyancy is a small **Richardson
number** `Ri = g beta dT L / U^2`, which `FlowStudy.richardson` computes from
the study's own numbers so it can be checked rather than assumed, and which
`FlowStudyResult.warnings()` reports above 0.1. On `scenes/duct_sink.py` it is
**3.0e-4** — buoyancy contributes under a thousandth of the momentum, and the
one-way reading is not measurably wrong. It would stop being so at a crawling
inlet speed or a large temperature rise, which is exactly when the warning
fires.

Temperature-dependent properties are simply absent. For air over 0–100 K of
rise, conductivity moves about 25% and viscosity about 25%; nothing here models
that, and nothing warns about it either.

### 8.3 The convective flux is `rho u`, and that decides whether energy is conserved

The energy equation transports `rho cp T`, so its flux is `rho cp u`. The
`rho` looked like a refinement and is not: lattice Boltzmann conserves *mass*
exactly in its streaming step, so `rho u` satisfies a discrete continuity
equation that `u` alone does not. Global energy balance as a fraction of
injected power, same solve, one multiply apart:

| lattice | flux = `u` | flux = `rho u` |
|---|---|---|
| 8x14x8 | 3.6e-2 | 8.9e-16 |
| 10x18x10 | 9.4e-3 | 1.6e-6 |

What is left with `u` alone is not round-off; it is the lattice's
compressibility error appearing as energy the duct creates or destroys.

The balance itself also had to be stated correctly before it could check
anything. Summing the discrete equations over every solved cell makes each
interior face cancel *exactly* — the coefficient `P` gives `E` and the flux `P`
sends `E` telescope whatever the mass flux is doing — so what remains is the
inlet plane, the outlet plane and the source. Counting only "mass flux times
inlet temperature" ignores both the conduction back out through the inlet and
the fact that the incoming air is advected at the first cell's temperature, and
reports a couple of percent of spurious imbalance on a duct where the scheme is
conserving to round-off. With both ends counted in full, the residual is
**7.7e-12 on `scenes/duct_sink.py`** and 1e-13 to 1e-15 on the test cases — for
any geometry, since the identity does not depend on one.

That makes the energy balance the sharpest available check on the whole
assembly, and it has since caught two separate faults that nothing else did
(§8.7).

### 8.4 Verification against answers the solver cannot produce

**Pure conduction, against a closed form and against `ThermalStudy`.** With the
inlet held still the momentum solve is skipped entirely (a zero inlet velocity
with a pressure-anchored outlet has rest as its fixed point, so this is an
equality, not an approximation) and the study solves `-k T'' = q`, `T(0) = 0`,
`T'(L) = 0`, whose answer is `(q/k)(Ly - y^2/2)`. Max relative error of the
lattice, against cells along the duct:

| cells | max rel. error | ratio |
|---|---|---|
| 8 | 3.92e-3 | |
| 16 | 9.78e-4 | 4.01 |
| 32 | 2.44e-4 | 4.01 |
| 64 | 6.10e-5 | 4.00 |

Exactly second order. The FEM `ThermalStudy` on the same problem is accurate to
**1e-14** — trilinear elements reproduce a quadratic solution exactly — so the
whole of the gap between the two solvers is the lattice's truncation, and the
honest form of "zero flow reproduces `ThermalStudy`" is: **2.4e-4 of the peak at
32 cells, falling as h².** Two discretisations of one problem never agree to
solver tolerance; they agree to the coarser one's error, and the useful claim is
the rate. `tests/fem/test_flow_conjugate.py`.

**Advection–diffusion, against the analytic column.** Patankar's exponential
blend is nodally exact for one-dimensional constant-coefficient
advection–diffusion at any cell Péclet number, and it is:

| scheme | max abs error vs. `(e^{Pe s/L} - 1)/(e^{Pe} - 1)`, cell Pe = 2 |
|---|---|
| exponential | 3.9e-16 |
| upwind | 1.98e-1 |

The upwind row is the false diffusion, not a bug: `A = 1` adds a numerical
conductivity `|u| h / 2`, which at cell Péclet 2 is the air's own conductivity
again. It is offered because it is what most codes do, and measured because the
difference is the argument for not using it.

**Nusselt number, against a textbook correlation.** The strongest external
check, because it exercises the convective half against something with no
connection to this code: fully developed laminar flow in a square duct with
isothermal walls has `Nu = 2.976`. Imposing the analytic square-duct velocity
profile (isolating the energy discretisation from the momentum one) and fitting
the exponential decay of the bulk temperature:

| cells across the duct | Nu | error |
|---|---|---|
| 6 | 2.463 | −17.2% |
| 10 | 2.755 | −7.4% |
| 14 | 2.857 | −4.0% |
| 22 | 2.927 | −1.65% |

Converging on the correlation at roughly second order. **This is the number to
quote when someone asks what resolution a trustworthy answer needs**: about 22
cells across a channel for a Nusselt number good to a couple of percent, and
`scenes/duct_sink.py` runs 12, where a heat-transfer coefficient is some 5% low.

### 8.5 The gradient survives the coupling

`d(mean temperature)/d(geometry)` runs through the momentum fixed point's
adjoint *and* the energy solve's transposed linear solve. Against a central
difference on a box in a duct, 10x18x10:

| step `h` | relative error | order |
|---|---|---|
| 1e-2 | 6.54e-2 | |
| 1e-3 | 6.39e-4 | 2.01 |
| 1e-4 | 7.05e-6 | 1.96 |
| 1e-5 | 4.59e-8 | 2.19 |

The difference converges on the adjoint at second order down to **4.6e-8**,
which is a central difference behaving exactly as it should on a smooth
objective. That the *rate* is 2 matters as much as the magnitude: agreement at
one step size can be agreement with a truncation error that happens to be
small.

**One direction behaves worse, and it is the finite difference's fault.** When
the design perturbation puts the geometry's surface exactly on a plane of cell
centres, the same check converges at first order instead:

| step `h` | relative error | order |
|---|---|---|
| 1e-2 | 2.09e-1 | |
| 1e-3 | 1.61e-2 | 1.11 |
| 1e-4 | 1.57e-3 | 1.01 |
| 1e-5 | 1.56e-4 | 1.00 |

It is still converging *to the adjoint* — a clean factor of ten per decade,
heading to zero — so the adjoint is the correct value and the objective simply
has a C1 kink at that configuration, from the compact profile's clamp joins
crossing the perturbation path. The practical consequence is for whoever
validates a new configuration: check a finite difference at two step sizes and
look at the rate, because a single ratio at a single `h` cannot tell a wrong
adjoint from a kinked difference. §3.1's smoothstep/smootherstep measurement is
the same effect seen from the other side.

### 8.6 What the study declares

`FlowStudy` is declared in a scene the way `ThermalStudy` is, registers itself
with `capture_studies`, and has `describe()`, boundary conditions and refusals
of the same shape:

```python
cooling = FlowStudy(
    name="duct-cooling",
    resolution=(14, 26, 14),
    bounds=(-0.70, -0.90, -0.50), size=(1.40, 1.80, 0.85),
    reynolds=25.0, conductivity_ratio=200.0,
    bcs=[Inlet(velocity=0.02, temperature=0.0), Outlet(), Walls(),
         HeatSource(Nodes.box([-0.14, -0.18, -0.40], [0.14, 0.18, -0.20]), power=1.0)],
)
```

`Inlet` drives the flow and sets the temperature everything is measured
against (`velocity=0.0` is legal and means the pure-conduction case). `Outlet`
takes no arguments, deliberately: a duct that exhausts to the room is fully
specified by the fact that it does. `Walls` is no-slip always and adiabatic
unless given a temperature. `HeatSource` and `HeldTemperature` are the
volumetric counterparts of `HeatFlux` and `Dirichlet`.

Selections reuse the mesh language (`Nodes.box`, `Nodes.sphere`,
`Nodes.halfspace`, `Nodes.cylinder`, composed with `& | ~`) read through its
`describe()` payload, so the two readings cannot drift. One semantic difference
is deliberate: on a lattice a selection is **volumetric**, every cell whose
centre satisfies the criterion, because the two things a flow study selects with
one — a heated region inside the solid, a block of cells held at a temperature —
are volumes. `Nodes.side` and `Nodes.predicate` are refused with the
alternative named: a lattice filled by an SDF has no boundary surface whose
extreme plane `side` could mean, and a predicate's callable never reaches the
serialized description.

### 8.6.1 What it took to make the viewer accept it

Declaring the study in a scene broke the viewer twice, in ways worth writing
down because both were about *scope* rather than about physics.

**A scene must not set jax flags.** The first version enabled
`jax_enable_x64` at the scene's module scope, because the solve needs float64
and the design parameters are built at import. `jax_enable_x64` is a
**process** setting, the viewer's compile worker imports the scene, and every
array afterwards was float64 — including the ones the WGSL backend has to
emit, and WebGPU has no `f64`. Merely *declaring* a flow study stopped the
scene from opening. The flip now lives in
`cadjoint/flow/precision.py::double_precision`, entered by `FlowStudy.solve`
and restored on the way out, following `cadjoint/fem/backends.py`. The
boundary is the same one that file documents and is worth repeating: **a
scope covers a forward solve and cannot cover a gradient**, because
`jax.grad` runs its transposed pass after the scope has closed and then
cannot materialise the float64 intermediates. So forward solves scope
themselves — which is all the viewer ever does — and callers who
differentiate enable x64 for their process, as `tests/flow/conftest.py` and
`scenes/duct_sink.py`'s `main` do.

**A boundary condition is not always a node selection.** The compile worker
serialised every BC as `bc.nodes.serializable`, which `ThermalStudy` and
`ElasticStudy` conditions satisfy and `Inlet`, `Outlet` and `Walls` do not —
they are planes of a duct, and carry a velocity or a wall temperature. The
fix is the seam rather than an attribute: `serializable` is a statement
*about* `describe()`, and `describe()` is the only shape every condition
shares, so the condition answers for itself. Changed:

| file | change |
|---|---|
| `cadjoint/viewer/_worker_declarations.py` | `bc.nodes.serializable` → `bc.serializable` (one line + docstring) |
| `cadjoint/viewer/_worker_payloads.py` | the same substitution on the solved-study path |
| `cadjoint/viewer/schema/payloads.py` | `StudyBc.nodes` optional; `velocity`/`temperature`/`power` added; `StudyPayload.kind` gains `"flow"` |
| `cadjoint/viewer/schema/payloads.d.ts` | regenerated (`python -m cadjoint.viewer.schema.emit`) |
| `cadjoint/fem/study.py` | `serializable` property on `Dirichlet`/`HeatFlux`/`Fixed`/`Traction`; `register_study` made public |

`StudyPayload.kind` is deliberately *wider* than
`cadjoint.enums.StudyKind`, which still holds two members. The enum is the
vocabulary the viewer can **create and edit** — it drives four tables in
`viewer/patch/studies.py` that say how to write a study's constructor — and
none of that exists for a flow study. What a scene declares is a wider
vocabulary than what the viewer can author, so the study serialises and
displays while reporting `editable: false`, which is true.

**Still outstanding on the viewer side**, and not attempted here:
`frontend/src/types.ts` and `frontend/src/studies.ts` still type a study's
kind as `"thermal" | "elastic"`, so the Studies window has no rendering for a
flow study's conditions; `STUDY_CALL_KINDS` in
`cadjoint/viewer/source_map/declarations.py` does not know `FlowStudy`, which
is what makes the declaration non-editable — and has one sharp edge worth
knowing: `_study_entries` aligns statements to studies by count, so a scene
declaring *both* a mesh study and a flow study would find the mesh study
marked non-editable too. No scene does that today; `scenes/duct_sink.py`
declares only the flow study.

### 8.7 Two faults the energy balance caught that nothing else did

Both returned a plausible number under a success flag, which is the failure
mode worth naming.

**Restarted GMRES stagnating on the conductivity ratio.** `restart = 30` on a
14x26x14 duct converges to 2.9e-12 at `conductivity_ratio = 50` and stalls at a
relative residual of **0.23** at 200 — reporting a peak temperature of **0.014
where the answer is 0.602**, a factor of 42, with no error raised. The subspace
size has to grow with the conductivity ratio; the default is now 60, which
converges at both, and `warnings()` reports a non-round-off energy balance as a
stagnated linear solve rather than as physics.

**bicgstab returning NaN, silently.** The predecessor formulation used
`jax.scipy.sparse.linalg.bicgstab`, which carries no breakdown guard. On a pure
*advection* column — an inlet temperature at one end, a held temperature at the
other, no source — the `rho` recurrence collapses and it returns `NaN` at every
tolerance from 1e-8 to 1e-12, with the same convergence flag it returns on
success. GMRES reaches 4.2e-16 of a dense reference on the same system, and is
no slower on the problems where bicgstab does converge.

A third fault of the same shape sits one layer out and is now guarded rather
than fixed: when the *momentum* march diverges it hands the energy solve a
velocity full of `NaN`, and GMRES answers a `NaN` operator with its zero
initial guess under a success flag — so the study reported a heat sink sitting
at exactly ambient. `solve_temperature` now poisons the result deliberately: a
`NaN` objective stops an optimizer, a plausible zero steers it.

### 8.8 The objective, and why both terms are needed

```
minimize   T_junction(theta)  +  w . pressure_drop(theta)
```

`pressure_drop` is not decoration. A sink that maximises contact with moving air
by filling the duct with metal strangles the fan driving it; without the
pressure term the optimizer walks into a solid block. On
`scenes/duct_sink.py` the sink costs **9.1x** the empty duct's drop
(9.78e-3 against 1.07e-3) while holding the die 0.602 above inlet air, and
`d(peak)/d(fin thickness)` is −0.52 against `d(pressure drop)/d(fin thickness)`
of +0.035 — the two terms genuinely pull opposite ways, which is what makes a
fin pitch fall out of the optimisation instead of a block.

The old `heat_transfer` proxy (`int chi|u|`) remains for callers that want a
scalar without an energy solve, but it is no longer the recommended objective:
a temperature is now available and a proxy for one is not needed.

### 8.9 What is still not true

Stated plainly, because every one of these would change an answer someone might
act on.

* **Turbulence is not modelled at all.** The solver is laminar BGK. A duct
  transitions around `Re ≈ 2300` and real forced-convection heat sinks run
  well above it, where mixing raises the heat-transfer coefficient several
  fold. Above 2300 this code reports the laminar answer, which *overstates*
  the temperature; `warnings()` says so, and that is the extent of the
  treatment.
* **The validated Reynolds range is roughly 5 to a few hundred.** The upper
  end is set by BGK stability rather than by physics, and the ceiling is a
  function of the lattice size, not only of `omega`: a 10x18x10 duct converges
  at `omega = 1.9417` while an 8x14x8 duct diverges at `omega = 1.8248`. The
  `OMEGA_CEILING = 1.95` guard is therefore necessary and *not* sufficient, and
  a coarse duct needs real margin. A diverged march is now visible rather than
  silent, but it is still the caller's job to avoid.
* **Resolution.** A Nusselt number good to a couple of percent needs about 22
  cells across a channel (§8.4). `scenes/duct_sink.py` uses 12 and the test
  suite uses 6 to 14. Nothing in the suite is at a resolution whose *absolute*
  numbers should be quoted; what the suite checks is rates, identities and
  gradients, all of which are resolution-independent claims.
* **Brinkman interface smearing.** The solid fraction transitions over about
  two cells by construction (`FlowGrid.suggested_epsilon`), and the penalised
  wall sits a penetration depth `sqrt(nu/alpha)` inside the true one. At
  `alpha_max = 200` the velocity leak into the solid is under 3e-4 of the inlet
  speed, but the pressure drop is still converging: it moves ~3% between
  `alpha_max = 200` and 50x that (§4.2), and only as `1/sqrt(alpha)`. So a
  pressure drop from this solver carries a few percent of penalisation bias on
  top of its discretisation error, and a *thermal* interface smeared over two
  cells puts the same order of uncertainty on the heat-transfer coefficient.
  This has not been separately quantified against a body-fitted reference,
  which is the obvious next measurement.
* **Compressibility.** Lattice Boltzmann is a low-Mach method and the inlet
  runs at Ma 0.035, so the compressibility error is under a tenth of a percent
  — provided the duct is not heavily blocked. It is not a fixed budget: the
  scene's first draft blocked 87% of every cross-section, which drove the
  density down 18% and *inverted the pressure drop*. Blockage is the thing to
  watch, and it is not obvious from the source.
* **Constant fluid properties**, per §8.2, and no radiation.

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
| `cadjoint/flow/energy.py` | the conjugate energy solve (finite volume + GMRES) |
| `cadjoint/flow/regions.py` | node selections resolved volumetrically on a lattice |
| `cadjoint/flow/study.py` | `FlowStudy` and its boundary conditions |
| `scenes/duct_sink.py` | the demonstration: a finned sink in a duct |
| `tests/flow/` | 158 tests: lattice, domain, lbm, steady, solver, plugin, energy, regions, study, scene |
| `tests/fem/test_flow_conjugate.py` | the still-inlet solve against `ThermalStudy` |

Verification: `pytest tests/flow -q` (158 passed, 87 s),
`pytest tests/fem/test_tesseract_packaging.py -q` (28 passed),
`pytest tests/plugins -q` (54 passed, 1 skipped), ruff clean and formatted.
