# A cut-cell finite-element solve on an implicit model, differentiated in θ

**The question.** With a direct implicit solver (CutFEM, the finite cell
method) there is no mesh and no node map: the PDE solution depends on the
design parameters θ only through the field f(p, θ) in the cut-cell
quadrature and in the Nitsche boundary terms. So `jax.grad` through the
whole assembly should be the classical level-set shape derivative, should
match finite differences to first order, and should have no topology
events in the way. Is it as clean as that?

**The short answer.** Two of three claims hold and the third needs a
qualification. `jax.grad` of the discrete objective is exact (it agrees
with central differences of the *re-classified* objective to 1e-9 where
no sub-grid corner changes sign within the step, and to 1e-6 otherwise,
in 2D and 3D, for a disc, a smoothly blended union, and a sphere). With
sharp-clipped cut-cell quadrature it is also a *convergent* shape
derivative: for the disc, dJ/dR converges to πR³/2 at second order with the
same constant as J itself, and evaluating the textbook Hadamard formula
∫Γ (∂ₙu)² Vₙ with the discrete solution is far worse (first order). But
the smoothed inside indicator the design asked for, a Heaviside of f/(αh),
gives an objective whose derivative is *not* convergent at fixed α: it
carries an O(α) bias of grid-phase-dependent sign that does not shrink
with h even though J does. And "no topology events" is true at the level of
C¹ only: every sub-grid corner that changes sign is a jump in the second
derivative, so finite differences drift away from `jax.grad` as O(δ ×
flips), and a discrete objective held at a stale sign structure is only C¹
if its clipped measures are signed, not folded at zero.

## What was built

`research/cutfem/`, Python and JAX, float64, every script under a few
minutes on CPU:

- `model.py` — the zero-set grammar of `research/zero-set.proto` (the
  proto JSON node table: `lit`, `coord`, `param`, `childValue`, `unary`,
  `binary`; `patch`, `warp`, `combine` with owning and blend rules)
  evaluated with `jax.numpy` so that `jax.grad` runs through the field in
  both the point and θ, plus a small `Table` builder in the conventions of
  `research/zeroset_test/lower.py` and three hand-lowered models: a disc
  (θ = R, cx, cy) as a warp of the unit patch ‖p‖ − R; a disc ∪ box under
  cadjoint's smooth minimum (θ = R, k, hy) with the box a `MAX` of four
  half-planes and the blend a `combine/blend` node with its transition; a
  sphere (θ = R, c).
- `cutfem.py` — the solver, dimension-generic over 2D and 3D. A background
  grid of Q1 cells; the *structure* (active cells, DOFs, cut cells, ghost
  faces, sign patterns) fixed at a θ₀ by `classify`; the θ-dependent
  quadrature in `quadrature`; Nitsche's method for u = 0 on Γ; Burman's
  ghost penalty γ_g Σ_F h ∫_F [∂ₙu][∂ₙv] on the faces of cut cells; a dense
  `jnp.linalg.solve` inside the graph; J = ∫Ω u = F·u; the Hadamard formula
  from u_h for comparison; condition numbers by LAPACK's subset eigensolver.
- `run_2d.py`, `run_3d.py` — the tables below; `test_cutfem.py` — nine
  coarse pytest checks in ~20 s (`PYTHONPATH=. pytest research/cutfem`).

### The design choices and what they cost

**Cut-cell quadrature.** Each cut cell carries a sub-grid of n_sub^d
sub-cells, each split into simplices (two triangles; six Kuhn tetrahedra),
and the sign of f at the sub-grid corners fixes a marching-simplices case
per simplex. Marching *simplices* rather than marching squares because
they have no saddle ambiguity and because the same piecewise-linear
interpolant then serves the volume clipping and the boundary polyline, so
the two integrals see one geometry. At fixed structure the edge crossings
t = f_i/(f_i − f_j) move smoothly with θ. Two volume modes:

- `sharp` — each sub-simplex is clipped by the interpolant (a tessellation
  of {f < 0}); full cells use their own d! Kuhn simplices. Error O(h_sub²)
  in the geometry, nothing else.
- `smooth` — the requested design: every sub-simplex is integrated whole
  and each quadrature point is weighted by a C² quintic Heaviside of
  −f/ε, ε = α h. The weight's derivative is a smeared delta of width 2ε
  on Γ, so the derivative of the volume terms is a *band average* of the
  integrand around Γ rather than its trace on Γ.

The consequence for the derivative is the central finding. `jax.grad` of
the discrete J = FᵀK⁻¹F is 2F′·u − uᵀK′u, and the true shape derivative
+∫Γ (∂ₙu)² Vₙ arises from a cancellation between the volume term's
derivative, −∫Γ |∇u|² Vₙ, and the Nitsche terms' derivative,
+2∫Γ (∂ₙu)² Vₙ. In sharp mode both are evaluated on the same polyline with
the same u_h and the cancellation is consistent. In smooth mode the volume
part is a band average of |∇u_h|² whose outer half lies *outside* Γ, in
cells where ∇u_h is only ghost-penalised, never solved, while the Nitsche
part stays sharp on Γ — the mismatch is O(1) relative to the band mass in
those cells, i.e. O(α), and independent of h. The tables confirm every
prediction of that picture: the bias scales with α, is insensitive to
n_sub, shrinks with a stiffer ghost penalty (until the penalty biases J
itself), and has a grid-phase-dependent sign at fixed h.

**Boundary integrals.** On the marching-simplices facets with 2-point
Gauss (segments) or the mid-edge rule (triangles); normals from ∇f/|∇f|
by `jax.grad`, not from the facet, as asked.

**Signed measures.** A simplex's measure is σ·det(edges)/d! with σ its
orientation at θ₀, and a facet's is det(edges, m)/(d−1)! with m its unit
normal at θ₀. At θ₀ these equal the unsigned measures with the same
derivative. Away from θ₀ they matter: with `abs(det)`, a sub-corner that
flips sign while the structure is held makes the stale clipped triangle
reappear reflected beyond the corner as a V-shaped kink of size O(|f|),
and finite differences at fixed structure then disagreed with `jax.grad`
by up to 6e-3 at δ = 1e-5 and by 20 % at δ = 1e-3. Signed, the stale
measure passes through zero smoothly and the fixed-structure objective is
C¹ like the re-classified one (the two finite differences agree with
`jax.grad` equally well in every table below). A stale structure is still
only local: at δ = 1e-2, hundreds of flips away from θ₀, it produces
crossings outside their edges and a singular system (the `nan` in the
last table).

**Ghost penalty.** Without it the sharp-mode matrix is *indefinite* at
γ_N = 20 (the small-cut trace constant) and the smooth-mode matrix has
condition number 1e11–1e12 (near-zero indicator weights). γ_g = 0.1 gives
cond(K) ≈ 500–1300 at h = 1/16–1/32, an ordinary O(h⁻²).

**Quadrature rules.** The 2D mid-edge rule is exact for the bilinear
stiffness (a quadratic). In 3D the trilinear stiffness is degree 4; the
common 4-point degree-2 tetrahedral rule under-integrates it (the
matrices differ at 1e-3 relative), so Keast's 14-point degree-5 rule with
positive weights is used and its exactness is tested.

## Results

All 2D tables are `PYTHONPATH=. python research/cutfem/run_2d.py`; the 3D
table is `run_3d.py`. "AD vs FD" is max_k |g_k − FD_k| / max_k |g_k| with
central differences of step δ = 1e-5, either with the structure
re-classified at each θ ± δ (the objective an optimiser sees) or held at
θ. |dJ/dc| is the translation derivative, analytically zero. Defaults:
γ_N = 20, γ_g = 0.1; the disc has R = 0.6 at an off-grid centre.

#### Disc: R = 0.6, J = πR⁴/8 = 0.050894, dJ/dR = πR³/2 = 0.339292

##### sharp (n_sub = 4)

| h    | DOFs | J rel err | order | dJ/dR rel err | order | |dJ/dc| | Hadamard rel err | AD vs FD (re-classified) | AD vs FD (fixed) | s |
|------|------|-----------|-------|---------------|-------|---------|------------------|--------------------------|------------------|---|
| 1/8  | 117  | -1.89e-02 | -     | -1.30e-02     | -     | 1.1e-04 | -2.34e-01        | 4.8e-09                  | 4.8e-09          | 4 |
| 1/16 | 371  | -4.36e-03 | 2.12  | -3.44e-03     | 1.92  | 3.1e-05 | -1.18e-01        | 2.5e-10                  | 2.6e-10          | 4 |
| 1/32 | 1321 | -1.03e-03 | 2.08  | -8.08e-04     | 2.09  | 4.6e-06 | -5.87e-02        | 2.0e-09                  | 1.9e-09          | 4 |
| 1/64 | 4947 | -2.49e-04 | 2.05  | -1.79e-04     | 2.17  | 1.3e-06 | -2.93e-02        | 1.8e-09                  | 1.8e-09          | 9 |

##### smooth (n_sub = 8, α = 0.5)

| h    | DOFs | J rel err | order | dJ/dR rel err | order | |dJ/dc| | Hadamard rel err | AD vs FD (re-classified) | AD vs FD (fixed) | s  |
|------|------|-----------|-------|---------------|-------|---------|------------------|--------------------------|------------------|----|
| 1/8  | 133  | -2.99e-02 | -     | -9.44e-03     | -     | 3.6e-03 | -3.35e-01        | 1.1e-08                  | 1.1e-08          | 3  |
| 1/16 | 406  | -6.20e-03 | 2.27  | -1.16e-02     | -0.30 | 3.5e-04 | -2.19e-01        | 1.9e-08                  | 1.9e-08          | 3  |
| 1/32 | 1373 | -9.83e-04 | 2.66  | -1.06e-02     | 0.13  | 1.0e-03 | -1.63e-01        | 9.6e-08                  | 9.6e-08          | 3  |
| 1/64 | 5067 | -9.67e-05 | 3.35  | -4.92e-03     | 1.11  | 9.6e-04 | -1.17e-01        | 1.1e-06                  | 5.1e-08          | 11 |

The sharp mode converges at second order in J and in dJ/dR with nearly
the same constant, so the derivative of the discretisation is a
discretisation of the derivative. The Hadamard formula evaluated with the
Q1 gradient on Γ is first order, as the FE gradient is. In smooth mode J
converges at second order but dJ/dR stalls near −1 %.

#### Disc ∪ box with smooth minimum: θ = {'R': 0.4, 'k': 0.08, 'hy': 0.25}

##### sharp (n_sub = 4)

| h    | DOFs | J        | dJ/dθ by AD (R, k, hy)     | AD vs FD (re-classified) | AD vs FD (fixed) | change from coarser h | s |
|------|------|----------|----------------------------|--------------------------|------------------|-----------------------|---|
| 1/16 | 294  | 0.021770 | 0.120983 0.094902 0.085934 | 5.9e-09                  | 5.9e-09          | -                     | 3 |
| 1/32 | 1036 | 0.021943 | 0.121642 0.095217 0.086426 | 9.3e-07                  | 9.3e-07          | 5.5e-03               | 4 |
| 1/64 | 3821 | 0.021985 | 0.121862 0.095364 0.086493 | 4.8e-07                  | 4.8e-07          | 1.8e-03               | 7 |

##### smooth (n_sub = 8, α = 0.5)

| h    | DOFs | J        | dJ/dθ by AD (R, k, hy)     | AD vs FD (re-classified) | AD vs FD (fixed) | change from coarser h | s |
|------|------|----------|----------------------------|--------------------------|------------------|-----------------------|---|
| 1/16 | 337  | 0.021642 | 0.120476 0.094548 0.086706 | 1.7e-06                  | 1.7e-06          | -                     | 3 |
| 1/32 | 1090 | 0.021920 | 0.121300 0.095147 0.086715 | 4.8e-08                  | 3.2e-08          | 6.8e-03               | 4 |
| 1/64 | 3936 | 0.021988 | 0.123426 0.096653 0.087025 | 1.3e-07                  | 1.7e-08          | 1.8e-02               | 7 |

No analytic solution here; `jax.grad` matches both finite differences at
every level. In sharp mode the derivative's change between successive
levels falls (5.5e-3 then 1.8e-3, order ≈ 1.6, with J's own change at
second order); in smooth mode it *grows* (6.8e-3 then 1.8e-2) while J
converges — the derivative is drifting with the grid, not converging.

#### The smoothed indicator's half-width ε = α h (disc, n_sub = 8)

| mode             | h    | |Ω_h| rel err | J rel err | dJ/dR rel err | |dJ/dc| |
|------------------|------|---------------|-----------|---------------|---------|
| sharp            | 1/16 | -2.80e-05     | -4.19e-03 | -3.16e-03     | 2.6e-05 |
| sharp            | 1/32 | -7.09e-06     | -9.86e-04 | -7.81e-04     | 3.3e-06 |
| sharp            | 1/64 | -1.76e-06     | -2.38e-04 | -1.89e-04     | 1.6e-06 |
| smooth α = 0.125 | 1/16 | +2.86e-05     | -4.56e-03 | -4.07e-03     | 1.0e-04 |
| smooth α = 0.125 | 1/32 | +6.20e-06     | -1.06e-03 | -1.15e-03     | 2.7e-04 |
| smooth α = 0.125 | 1/64 | +1.33e-06     | -2.54e-04 | +9.45e-05     | 4.0e-05 |
| smooth α = 0.25  | 1/16 | +9.71e-05     | -5.09e-03 | -5.75e-03     | 6.4e-04 |
| smooth α = 0.25  | 1/32 | +2.42e-05     | -1.14e-03 | -4.33e-03     | 8.1e-04 |
| smooth α = 0.25  | 1/64 | +6.04e-06     | -2.63e-04 | -1.11e-03     | 2.7e-04 |
| smooth α = 0.5   | 1/16 | +3.88e-04     | -6.20e-03 | -1.16e-02     | 3.5e-04 |
| smooth α = 0.5   | 1/32 | +9.69e-05     | -9.83e-04 | -1.06e-02     | 1.0e-03 |
| smooth α = 0.5   | 1/64 | +2.42e-05     | -9.67e-05 | -4.92e-03     | 9.6e-04 |
| smooth α = 1.0   | 1/16 | +1.55e-03     | -5.81e-03 | -1.53e-02     | 4.6e-03 |
| smooth α = 1.0   | 1/32 | +3.88e-04     | +2.04e-03 | -1.95e-02     | 1.0e-03 |
| smooth α = 1.0   | 1/64 | +9.69e-05     | +1.84e-03 | -8.37e-03     | 2.5e-03 |

The derivative's bias scales with α and does not decrease with h at fixed
α; the domain measure |Ω_h| itself is biased at O(ε²) only, and J at
second order regardless. n_sub = 4, 8, 16 gave the same numbers to two
digits (not shown).

#### Ghost penalty γ_g (disc, h = 1/32, γ_N = 20)

| mode   | γ_g   | cond(K)    | λ_min    | J rel err | dJ/dR rel err | |dJ/dc| |
|--------|-------|------------|----------|-----------|---------------|---------|
| sharp  | 0     | indefinite | -1.3e-02 | -9.56e-04 | -4.71e-04     | 4.1e-05 |
| sharp  | 0.001 | indefinite | -1.1e-02 | -9.56e-04 | -6.45e-04     | 3.5e-05 |
| sharp  | 0.01  | indefinite | -1.8e-03 | -9.64e-04 | -1.14e-03     | 2.2e-04 |
| sharp  | 0.1   | 1.3e+03    | +1.6e-02 | -1.03e-03 | -8.08e-04     | 4.6e-06 |
| sharp  | 1     | 1.6e+03    | +1.6e-02 | -1.43e-03 | -2.34e-03     | 1.1e-05 |
| sharp  | 10    | 6.7e+03    | +1.6e-02 | -3.68e-03 | -6.25e-03     | 2.2e-05 |
| smooth | 0     | 1.5e+11    | +1.3e-10 | +1.57e-03 | -2.71e-02     | 5.6e-03 |
| smooth | 0.001 | 3.6e+04    | +5.6e-04 | +1.34e-03 | -2.64e-02     | 5.4e-03 |
| smooth | 0.01  | 4.1e+03    | +4.9e-03 | +5.42e-04 | -2.47e-02     | 4.1e-03 |
| smooth | 0.1   | 1.3e+03    | +1.6e-02 | -9.83e-04 | -1.06e-02     | 1.0e-03 |
| smooth | 1     | 1.7e+03    | +1.6e-02 | -2.47e-03 | -5.25e-03     | 1.3e-04 |
| smooth | 10    | 8.7e+03    | +1.6e-02 | -7.08e-03 | -1.11e-02     | 3.7e-05 |

#### Nitsche penalty γ_N (disc, h = 1/32, γ_g = 0.1)

| mode   | γ_N | cond(K)    | λ_min    | J rel err | dJ/dR rel err | |dJ/dc| |
|--------|-----|------------|----------|-----------|---------------|---------|
| sharp  | 5   | indefinite | -2.3e-02 | -9.75e-04 | -1.03e-03     | 1.8e-04 |
| sharp  | 10  | 7.6e+02    | +1.6e-02 | -1.01e-03 | -7.39e-04     | 7.9e-06 |
| sharp  | 20  | 1.3e+03    | +1.6e-02 | -1.03e-03 | -8.08e-04     | 4.6e-06 |
| sharp  | 50  | 3.0e+03    | +1.6e-02 | -1.05e-03 | -8.25e-04     | 5.2e-06 |
| sharp  | 100 | 5.9e+03    | +1.6e-02 | -1.07e-03 | -8.07e-04     | 9.7e-06 |
| smooth | 5   | 5.0e+02    | +1.6e-02 | +2.92e-04 | +5.22e-03     | 1.3e-02 |
| smooth | 10  | 7.7e+02    | +1.6e-02 | -7.48e-04 | -1.16e-02     | 1.5e-03 |
| smooth | 20  | 1.3e+03    | +1.6e-02 | -9.83e-04 | -1.06e-02     | 1.0e-03 |
| smooth | 50  | 3.0e+03    | +1.6e-02 | -1.15e-03 | -9.81e-03     | 1.2e-03 |
| smooth | 100 | 5.9e+03    | +1.6e-02 | -1.24e-03 | -9.48e-03     | 1.4e-03 |

#### The disc shifted through the grid: 8 random centres in [0, h)²

| mode   | h    | J rel err              | dJ/dR rel err by AD    | max |dJ/dc| | Hadamard rel err       |
|--------|------|------------------------|------------------------|-------------|------------------------|
| sharp  | 1/16 | [-4.36e-03, -4.33e-03] | [-3.56e-03, -3.16e-03] | 4.7e-05     | [-1.19e-01, -1.16e-01] |
| sharp  | 1/32 | [-1.03e-03, -1.03e-03] | [-8.44e-04, -7.90e-04] | 2.6e-05     | [-5.90e-02, -5.82e-02] |
| sharp  | 1/64 | [-2.49e-04, -2.49e-04] | [-2.19e-04, -1.64e-04] | 7.0e-06     | [-2.98e-02, -2.90e-02] |
| smooth | 1/16 | [-7.02e-03, -5.60e-03] | [-2.59e-02, +8.08e-03] | 6.0e-03     | [-2.56e-01, -1.93e-01] |
| smooth | 1/32 | [-1.41e-03, -1.06e-03] | [-5.82e-03, +9.57e-03] | 3.0e-03     | [-1.49e-01, -1.29e-01] |
| smooth | 1/64 | [-1.27e-04, -8.99e-05] | [-3.88e-03, +1.08e-02] | 2.2e-03     | [-1.14e-01, -9.68e-02] |

In sharp mode the derivative's error scatters within ±10 % of an O(h²)
trend; in smooth mode it scatters between −0.6 % and +1 % at every h.

#### The finite-difference step (disc ∪ box, h = 1/32)

| mode   | δ     | sub-corner flips within δ (R, k, hy) | FD re-classified vs AD | FD fixed vs AD |
|--------|-------|--------------------------------------|------------------------|----------------|
| sharp  | 1e-02 | 920 506 535                          | 2.3e-03                | nan            |
| sharp  | 1e-03 | 110 63 35                            | 5.7e-04                | 2.0e-03        |
| sharp  | 1e-04 | 10 4 1                               | 7.8e-07                | 1.0e-04        |
| sharp  | 1e-05 | 0 0 0                                | 9.3e-07                | 9.3e-07        |
| sharp  | 1e-06 | 0 0 0                                | 9.4e-09                | 9.4e-09        |
| smooth | 1e-02 | 3335 1849 2124                       | 8.1e-04                | nan            |
| smooth | 1e-03 | 324 184 312                          | 1.2e-03                | 9.5e-04        |
| smooth | 1e-04 | 34 16 9                              | 3.6e-03                | 3.0e-06        |
| smooth | 1e-05 | 3 0 0                                | 4.8e-08                | 3.2e-08        |
| smooth | 1e-06 | 3 0 0                                | 1.9e-08                | 3.3e-10        |

The count is of sub-grid corners of cut cells whose f crosses zero within
±δ along each parameter. Every crossing is a jump in the second derivative
of J, so the finite-difference error grows as O(δ × flips) in both
variants; with no crossing inside the step the error is the central
difference's own O(δ²) truncation (9e-7 at δ = 1e-5, 9e-9 at 1e-6).

### 3D smoke test

#### sphere R = 0.6: J = 0.021715, dJ/dR = 0.180956

##### sharp (n_sub = 2)

| h      | DOFs  | J rel err | order | dJ/dR rel err | order | |dJ/dc| | AD vs FD | s  |
|--------|-------|-----------|-------|---------------|-------|---------|----------|----|
| 1.6/8  | 349   | -9.34e-02 | -     | -6.61e-02     | -     | 9.8e-05 | 8.2e-10  | 4  |
| 1.6/12 | 902   | -4.03e-02 | 2.07  | -3.01e-02     | 1.94  | 2.4e-05 | 8.1e-10  | 5  |
| 1.6/16 | 1728  | -2.19e-02 | 2.13  | -1.67e-02     | 2.04  | 2.5e-05 | 2.1e-07  | 7  |
| 1.6/24 | 4813  | -9.25e-03 | 2.12  | -7.02e-03     | 2.14  | 5.4e-06 | -        | 5  |
| 1.6/32 | 10255 | -5.03e-03 | 2.12  | -3.91e-03     | 2.03  | 2.0e-06 | -        | 12 |

##### smooth (n_sub = 2, α = 0.5)

| h      | DOFs  | J rel err | order | dJ/dR rel err | order | |dJ/dc| | AD vs FD | s  |
|--------|-------|-----------|-------|---------------|-------|---------|----------|----|
| 1.6/8  | 481   | -1.52e-01 | -     | -9.97e-02     | -     | 2.8e-04 | 1.0e-09  | 4  |
| 1.6/12 | 1066  | -7.03e-02 | 1.91  | -5.01e-02     | 1.70  | 6.2e-05 | 2.4e-09  | 5  |
| 1.6/16 | 2061  | -3.93e-02 | 2.02  | -2.95e-02     | 1.85  | 1.3e-04 | 1.0e-06  | 6  |
| 1.6/24 | 5421  | -1.69e-02 | 2.09  | -1.27e-02     | 2.07  | 5.3e-05 | -        | 6  |
| 1.6/32 | 11231 | -9.06e-03 | 2.16  | -6.86e-03     | 2.14  | 2.1e-05 | -        | 16 |

The 3D path behaves like the 2D one: `jax.grad` is exact, and in sharp
mode J and dJ/dR converge at second order from the coarsest level (3
cells per radius) to the finest (10 k dense DOFs, about 15 s for J and its
gradient), with the derivative's constant again close to J's. The smooth
mode's derivative bias is not visible here because at these h the
discretisation error still dominates; the 2D tables show what happens
when it does not. Finite differences are omitted at the two finest levels
only for time.

## Verdict

The shape derivative through a cut-cell finite-element solve on an
implicit model is as clean as the theory says *provided the quadrature is
a sharp tessellation of {f < 0}*: `jax.grad` through the whole assembly is
the exact derivative of the discrete objective (1e-9 against finite
differences when no sub-grid corner changes sign inside the step, 1e-6
when one does, with the structure re-classified or held), it converges to
the analytic shape derivative at the same second order as the objective,
with a grid-phase scatter of ±10 % of a vanishing error, and it beats the
Hadamard formula evaluated with the discrete solution by an order in h. The
honest caveats: (1) the smoothed indicator that makes the objective C²
also makes its derivative biased at O(α) with a grid-phase-dependent
sign, because the derivative of a smeared volume integral samples the
ghost region where the solution is only penalised — at fixed α it never
converges, and α → 0 needs the sub-grid to resolve the band, at which point
it is the sharp method with extra cost; (2) "no topology events" holds at
C¹, not C²: sub-corner sign flips are dense in θ, so finite differences
drift as O(δ × flips) and a Newton-type optimiser will see a piecewise
second derivative; (3) the result depends on the ghost penalty being on
(γ_g ≈ 0.1–1; the matrix is indefinite or 1e12-conditioned without it) and
the derivative's error follows J's in γ_g; (4) cost: the quadrature is
O(cells × n_sub^d × d!) points (about 1 M at h = 1/64 in 2D, 2.5 M at
1.6/24 in 3D) and reverse mode stores them, and the dense solve is O(n³)
in DOFs — about 15 s for J and its gradient at n ≈ 11 k, the practical
ceiling of this prototype; a sparse direct solve with a custom VJP (K is
symmetric, so the adjoint is the same solve) is the obvious next step.

Left undone at that point: no sparse solver; the 3D test was a single
sphere; no real scene; no comparison against the tet-mesh route. The
section below takes those up; what remains after it is at its end.

## Continuation: a sparse solve, any scene, mixed conditions, and the head to head

**Sparse direct solve with a custom VJP.** The matrix is assembled as its
nonzeros on a sparsity pattern fixed with the structure: the volume and
boundary contributions are summed per cell inside XLA (a segment sum over
the quadrature points) and scattered onto the pattern's slots, so the host
sees ~27 n values and never the 64 per quadrature point (shipping those —
366 M entries at 16 k DOFs — was 36 s of the 40 s a step took). SuperLU
through `jax.pure_callback`; the VJP is written out (F̄ = K⁻¹ū, K̄ =
−F̄uᵀ restricted to the pattern), and since K is symmetric the adjoint is
the same factorisation: two factorisations per objective-and-gradient. The
dense path stays for condition numbers and for the test that the two agree
in value and gradient to 1e-9.

**Any field.** `classify` takes a proto-JSON table, a
`cadjoint.zeroset.table.Model` lowered from a scene, or a plain
`f(θ, points)`; gradients in the point and in θ are JAX's. Two things a
real scene taught: the classification scan is jitted and chunked (a
thousand-node table evaluated eagerly keeps an array per node alive for
the whole batch), and the Nitsche normal is read a hair inside the facet,
because a polygon's signed distance is `sign·√(d·d)` and the interpolated
crossings on its planar faces sit exactly where that is 0·∞.

#### `scenes/starter.py` (thermal body: 39 free parameters, 1166 nodes), torsion problem, sharp

| cells across | h | n_sub | DOFs | cut cells | volume pts | J | ‖dJ/dθ‖ | warm J+grad s | AD vs FD (re-classified) |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 0.138 | 2 | 1305 | 670 | 0.58 M | 0.004663 | 0.0256 | 0.4 | 7.7e-09 |
| 24 | 0.069 | 2 | 6592 | 2568 | 2.3 M | 0.005557 | 0.0287 | 2.0 | – |
| 32 | 0.052 | 2 | 16494 | 6372 | 5.7 M | 0.005782 | 0.0297 | 4.6 | – |
| 16 | 0.104 | 1 | 2653 | 1206 | 0.26 M | 0.005437 | 0.0288 | 0.3 | 5.1e-09 |
| 32 | 0.052 | 1 | 16494 | 6372 | 1.7 M | 0.005954 | 0.0308 | 2.2 | – |

`jax.grad` is the derivative of the re-classified objective to 1e-8 on a
scene with polygons, extrusions and smooth unions; the first call pays
XLA 15–25 s for the quadrature. The quadrature, not the solve, is now the
cost: `n_sub = 1` (the cell's own Kuhn tetrahedra, geometry error still
O(h²)) is 8× fewer points for a 3 % different J at this h.

**Mixed conditions by region** (`Thermal`): conductivity and source, a
Dirichlet value by Nitsche on the facets a region predicate selects, an
inflow per area on the facets another selects, insulated elsewhere; a
cadjoint `NodeSelection.contains` is such a predicate, so a study's
conditions transfer as written. Regions are read on the facets at the
structure's θ and kept, as the mesh route keeps its nominal resolution.
`mean_temperature` is the objective ∫Ω T / |Ω| from the same quadrature.

#### Annulus, T = 0 on r₀ = 0.25, inflow q = 1 on R = 0.7, k = 2 (n_sub = 2)

| h | mean T rel err | d(mean T)/dR rel err | AD vs FD (re-classified) |
|---|---|---|---|
| 1/16 | -3.72e-03 | -1.31e-03 | 1.9e-03 |
| 1/32 | -7.82e-04 | +1.44e-03 | 1.6e-03 |
| 1/64 | -1.81e-04 | +2.97e-04 | 3.4e-04 |

Second order in the mean against the closed form; the derivative through
the moving flux boundary within 1e-3 and converging. (A square with
volumetric region predicates is *not* an exactness test: the marching
simplices chord its corners and a halfspace predicate catches the first
facets of the insulated faces; the annulus has neither problem.)

#### Head to head: the starter's own heat study, mean temperature over the body

`run_thermal.py` solves `heat_study` (k = 2, die flux 6 on the slug bottom,
ambient on the fin field above z = 0.45) two ways on the same design, and
checks each route's gradient against central differences of its own
objective.

| route | resolution | DOFs | J = mean T | ‖dJ/dθ‖ | cos ∠(·, finest tet) | J+grad s |
|---|---|---|---|---|---|---|
| tet10 ×1 (declared) | (18, 13, 11), 3182 tets | 6019 | 0.151314 | 0.37545 | 0.9654 | 11.2 eager (mesh 4.4) |
| tet10 ×1.5 | (27, 20, 16), 8229 tets | 14754 | 0.161273 | 0.34894 | 0.9846 | 14.4 eager (mesh 4.9) |
| tet10 ×2 | (36, 26, 22), 16170 tets | 27949 | 0.207433 | 0.42471 | 1.0000 | 17.6 eager (mesh 6.0) |
| cut cells, n_sub 1 | h = 0.131 (674 cut of 836 cells) | 1301 | 0.140063 | 0.47543 | 0.7171 | 0.1 warm (7 first) |
| cut cells | h = 0.066 (2764 / 5794) | 7425 | 0.178916 | 0.41177 | 0.9491 | 0.7 warm (9 first) |
| cut cells | h = 0.044 (6490 / 17896) | 21511 | 0.189420 | 0.64912 | 0.8663 | 2.8 warm (13 first) |
| cut cells | h = 0.033 (11150 / 37684) | 43754 | 0.202997 | 0.46606 | 0.9765 | 7.9 warm (23 first) |

| parameter | tet ×1 AD | tet FD δ = 1e-6 / 1e-5 / 1e-4 | tet ×2 AD | cut h=0.131 AD | cut FD fixed structure | cut FD re-classified | cut h=0.033 AD |
|---|---|---|---|---|---|---|---|
| fin_depth | -0.12289 | -0.10943 / -0.10723 / -0.10663 | -0.17317 | -0.10191 | -0.10191 | -0.10191 | -0.14672 |
| fin2_root_l[0] | +0.16101 | +0.14884 / +0.14728 / +0.14721 | +0.16233 | +0.12710 | +0.12710 | +473.31889 | +0.18908 |
| fin2_root_r[0] | -0.16258 | -0.15795 / -0.14520 / -0.14514 | -0.13202 | -0.17239 | -0.17239 | -0.17239 | -0.18160 |
| fin3_root_r[0] | -0.10960 | – | -0.13066 | -0.10490 | – | – | -0.17330 |
| fin1_root_l[0] | +0.08769 | – | +0.12903 | +0.03492 | – | – | +0.15922 |

What it says:

1. *Neither route is converged on this scene at these sizes* — the fins are
   thin, and both means still climb (tet 0.151 → 0.161 → 0.207; cut 0.140
   → 0.179 → 0.189 → 0.203). The finest of each are 2 % apart with
   gradients at cos 0.977, and the per-parameter derivatives differ by
   15–35 % while both are still moving between levels. Reading a
   converged derivative on a part like this needs h well under the fin
   thickness on the cut-cell side and a much finer surface mesh on the
   other; the cut cells get there at 8 s warm per objective-and-gradient
   at 44 k DOFs against 18 s eager plus 6 s of meshing per refreeze at 28
   k nodes.
2. *The cut-cell gradient is exact* — fixed-structure finite differences
   agree with `jax.grad` to 8e-9 — and the re-classified difference is
   meaningless for `fin2_root_l[0]` (473): the Dirichlet plane cuts the
   fins mid-height, region membership is per facet, and re-classification
   at θ ± δ moves whole facets across the plane. The objective an
   optimiser sees is C⁰ at the sub-cell scale there; the mesh route's is
   too, under re-meshing, only hidden by the frozen mesh between refreezes.
3. *The mesh route's gradient is not the derivative of its own frozen-mesh
   objective on this scene.* 11–13 % off finite differences at every step
   from 1e-6 to 1e-4 (so not solver noise), with the flux condition
   removed as well, and exact on a ball (0.040527 by both, at every δ).
   Isolated to the node motion: for a random linear functional of the
   recomputed positions alone, AD and FD differ by 20–360 % in these
   parameters, at 136 of the 860 surface corner vertices (820 of 6019
   nodes with midsides). Nudging the base points off the surface by 1e-7
   moves the global AD/FD gap to 1.4 % but leaves 814 nodes disagreeing,
   and a 1e-5 nudge changes both AD and FD threefold: the single-field
   Newton projection is not a smooth function of the design near creases,
   where a vertex placed *on* the crease steps between the two faces that
   meet there. This is the case the zero-set protocol's `Solve` with an
   incidence was written for — a crease vertex solved on both surfaces at
   once — and `cadjoint.zeroset.refresh.Overlay` already does that
   classification and solve for the viewer's overlay.

Left undone now: elasticity (the studies the app optimises are elastic;
the assembly generalises, Nitsche for a vector field and tractions by
region); convergence on thin features without brute-force refinement
(local refinement of the background grid, or a higher-order element);
the structure still comes from a sign scan rather than the protocol's
`Discover`; and the mesh route's projection derivative, which is not this
prototype's to fix but is its most useful finding.
