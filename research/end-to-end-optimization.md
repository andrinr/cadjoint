# End-to-end optimization: measured run record

Status: **runs end to end** (2026-09-01). The chain and the discrete/continuous
split are documented in `README.md` § "End-to-end optimization"; the showcase is
`examples/fem_bracket_optimization.py` on the L-bracket from `scenes/bracket.py`.
This note keeps what the README does not carry: the measured run record, the
box-bound rationale, and a known CalculiX deck fragility. Full build history in
git/PR #19.

Details beyond the README's summary:

- With the default backend the compliance term is the total squared
  displacement; with ccx it is classical compliance (`f . u`, twice the strain
  energy).
- `recompute_points` re-projects the snapped boundary vertices onto the traced
  SDF's zero set with total motion clamped to half a cell diagonal (the same
  clamp used at extraction); every `--remesh-every` steps (default 6) topology
  is re-extracted. The final reported number is evaluated on a freshly
  extracted mesh so it does not depend on the last frozen topology.

## Measured numbers

Default 30-step run (Apple Silicon CPU, jax-fem backend, 2026-09-01;
reproducible — two independent runs produced identical objective traces):

| quantity | initial | final | change |
|---|---|---|---|
| objective | 17.4416 | 4.6924 | -73.1% |
| compliance | 16.1642 | 3.0729 | -81.0% |
| mass | 1.2774 | 1.6195 | +26.8% |
| web_thickness | 0.1600 | 0.1666 | +4.1% |
| rib_height | 0.8800 | 1.1064 | +25.7% |
| plate_thickness | 0.2000 | 0.3000 (upper bound) | +50.0% |

- FD vs adjoint at the initial design (central differences, eps 1e-5):
  rel 2.3e-5 (`web_thickness`), 2.0e-3 (`rib_height`), 3.3e-7
  (`plate_thickness`).
- Mesh size at default resolution (30, 21, 16): 1714 hexes / 2735 nodes.
- Per-step timing: 4.7 s median warm eval (value + gradient), 11-42 s on
  re-extraction steps (retrace + XLA recompile). Some frozen-mesh phases run
  slower per eval (up to ~36 s) — the iterative solve needs more iterations
  when the thinned web conditions the system badly.
- Total wall time for the default 30-step run: ~9.5 min (360 s of
  objective/gradient evals plus FD check, meshing, and exports).
- `plate_thickness` runs to its upper bound and stays there (its raw gradient
  component remains ~-40); the projected gradient norm decays 272 -> 8.4.
- The remesh jumps are visible and material at this resolution: the step-12
  re-extraction raised the objective 6.02 -> 8.86 and the step-18 one lowered
  it 7.99 -> 5.53 — the web is only ~2 cells thick, so a one-cell-layer
  change in the loaded wall shifts the discrete compliance substantially.

## Box bounds and projection

| parameter | lower | upper |
|---|---|---|
| `web_thickness` | 0.12 | 0.26 |
| `rib_height` | 0.35 | 1.15 |
| `plate_thickness` | 0.14 | 0.30 |

The lower thickness bounds keep the thin walls above one grid cell, so
re-extraction never drops them from the topology; the upper bounds keep the part
inside the sampling lattice. Projection is a `jnp.clip` after
`optax.apply_updates`. The recorded *projected* gradient norm zeroes components
pushing against an active bound — that is the stationarity measure that should
shrink, while the raw norm can stay large at a bound-constrained optimum.

## Artifacts and guards

- `examples/output/` collects the convergence history CSV, the convergence
  figure (PNG), and before/after VTU exports for ParaView. The VTUs are per-run
  artifacts and gitignored via `examples/output/.gitignore`; CSV and PNG are
  commit-worthy.
- `examples/test_fem_bracket_optimization.py` is collected by a bare `pytest`
  run with no configuration, runs the smoke path in under a minute, and restores
  the x64 flag so the rest of the suite does not inherit it.
- `--smoke` runs 2 optimizer steps at resolution (14, 10, 8), asserts the
  objective descends, and asserts adjoint/FD agreement on the web-thickness
  component.

## CalculiX deck fragility (found here, worked around, not fixed)

The `--backend calculix` path works at both the smoke and the default
resolution (single eval + FD check verified: objective 1.3769 at (30, 21, 16),
web-thickness FD rel 2.3e-3 through the ccx `*SENSITIVITY` adjoint), but only
after the example flushes float-noise coordinates to exact zero before the
solve. The underlying issue sits in `cadjoint/fem/calculix.py`'s deck writer
(`write_elastic_deck`, `{x:.17g}` node fields): ccx's free-format reader
rejects any field wider than 20 characters, and a snapped node at ~1e-17 —
float noise on a coordinate plane — prints as 23 (also at risk: 17-significant-
digit negatives in (-0.1, -0.01), which print as 21). HEAD's validated
configuration passed by luck of its grid values. The proper fix (bounded-width
formatting in the deck writer) belongs to the fem layer, which is being
reworked concurrently; the example's `jnp.where(|points| < 1e-9, 0, points)`
shim is gradient-neutral because the affected coordinates are exactly zero for
every nearby design.
