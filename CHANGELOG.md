# Changelog

All notable changes to cadjoint are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) with the pre-1.0 rule spelled out
in [docs/releasing.md](https://github.com/andrinr/cadjoint/blob/main/docs/releasing.md):
until 1.0, a minor bump (0.x → 0.x+1) may break the API and a patch bump does
not.

## [Unreleased]

## [0.1.0]

First release on PyPI.

- Geometry: SDF primitives, smooth booleans, transforms and patterns;
  sketches on work planes extruded, revolved or lofted into solids.
- Constraints solved by projection, kept satisfied through optimization.
- Meshing by dual contouring; TET4/TET10 through TetGen, second-order tets
  through Gmsh, a HEX8 path; OBJ, STL and faceted STEP export.
- Thermal and linear-elastic studies through jax-fem or CalculiX, a
  lattice-Boltzmann flow prototype, and optax-driven shape optimization.
- Tesseract plugin interface for every non-JAX component, with prebuilt
  solver images on GHCR.
- The `cadjoint-viewer` playground, shipped prebuilt in the wheel.

[Unreleased]: https://github.com/andrinr/cadjoint/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/andrinr/cadjoint/releases/tag/v0.1.0
