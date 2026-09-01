"""Native (Rust) dual-contouring core behind the reference pipeline's API.

Array-in/array-out split of :mod:`cadjoint.meshing`: JAX evaluates the
implicit field (lattice sweep in :func:`sample_grid`, differentiable root
refinement in :func:`edge_hermite_data`) exactly as the reference does, and
the rayon-parallel Rust cdylib in ``native/`` does the heavy discrete and
linear-algebra work — crossing detection, manifold cell incidence, batched
QEF solves, and oriented dual faces. Discrete outputs are bit-identical to
the NumPy reference; continuous outputs agree to floating-point accuracy.

The gradient contract is untouched: crossing refinement (bisection + Newton
against the true SDF) stays in JAX, and the smooth Tikhonov QEF crosses
into Rust through a tesseract (``native/tesseract_api.py``) whose
``vector_jacobian_product`` endpoint serves the hand-derived linear-solve
VJP, composed into JAX autodiff via ``tesseract_jax.apply_tesseract``. An
optimization loop mirrors the reference recipe with native stages::

    values = sample_grid(compiled(free, fixed), grid)
    edges = find_crossing_edges_native(values)
    incidence = manifold_cell_incidence_native(edges, grid, values < 0.0)

    def loss(candidate_free):
        hermite = edge_hermite_data(compiled(candidate_free, fixed), grid, edges)
        vertices, _ = qef_vertices_native(hermite, incidence, grid)
        return mesh_loss(vertices, faces)

Requires the cdylib from ``cargo build --release --manifest-path
native/Cargo.toml``; every entry point raises an actionable error naming
that command when the library is missing. The tesseract-backed
:func:`qef_vertices_native` additionally requires jax x64 mode (the
tesseract schema is float64, matching the exactness contract).
"""

from __future__ import annotations

import ctypes
import os
import warnings
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array

from cadjoint.meshing.dual_contouring import Mesh, _averaged_normals
from cadjoint.meshing.edge_detection import (
    CrossingEdges,
    GridSpec,
    HermiteData,
    edge_hermite_data,
    sample_grid,
)
from cadjoint.meshing.features import CellIncidence

_BUILD_HINT = (
    "The native mesher cdylib is not available. Build it with:\n"
    "    cargo build --release --manifest-path native/Cargo.toml\n"
    "or point the CADJOINT_NATIVE_MESHER environment variable at the built library."
)
_TESSERACT_HINT = (
    "tesseract-core / tesseract-jax are not installed. "
    "Install the 'tesseract' extra: pip install cadjoint[tesseract]."
)
_NATIVE_DIR = Path(__file__).resolve().parents[2] / "native"


class _DcEdges(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint64),
        ("axis", ctypes.POINTER(ctypes.c_int8)),
        ("index", ctypes.POINTER(ctypes.c_int32)),
        ("start_inside", ctypes.POINTER(ctypes.c_uint8)),
    ]


class _DcIncidence(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint64),
        ("cells", ctypes.POINTER(ctypes.c_int32)),
        ("edge_ids", ctypes.POINTER(ctypes.c_int32)),
        ("counts", ctypes.POINTER(ctypes.c_int32)),
        ("error", ctypes.c_int32),
    ]


class _DcFaces(ctypes.Structure):
    _fields_ = [
        ("quad_count", ctypes.c_uint64),
        ("quads", ctypes.POINTER(ctypes.c_int32)),
        ("triangles", ctypes.POINTER(ctypes.c_int32)),
        ("skipped_boundary", ctypes.c_uint64),
    ]


def _library_candidates() -> list[Path]:
    override = os.environ.get("CADJOINT_NATIVE_MESHER")
    if override:
        return [Path(override)]
    release = _NATIVE_DIR / "target" / "release"
    return [
        release / "libcadjoint_native_mesher.dylib",
        release / "libcadjoint_native_mesher.so",
        release / "cadjoint_native_mesher.dll",
    ]


def native_available() -> bool:
    """Whether the native cdylib can be located (without loading it)."""
    return any(path.is_file() for path in _library_candidates())


@lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    """Load the cdylib and declare its ABI; raises with a build hint."""
    for path in _library_candidates():
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            break
    else:
        raise ImportError(_BUILD_HINT)

    u64, f64, i32 = ctypes.c_uint64, ctypes.c_double, ctypes.c_int32
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u8p = ctypes.POINTER(ctypes.c_uint8)

    lib.dc_abi_version.restype = ctypes.c_uint32
    lib.dc_abi_version.argtypes = []
    lib.dc_find_crossing_edges.restype = ctypes.POINTER(_DcEdges)
    lib.dc_find_crossing_edges.argtypes = [f64p, u64, u64, u64, f64]
    lib.dc_free_edges.restype = None
    lib.dc_free_edges.argtypes = [ctypes.POINTER(_DcEdges)]
    lib.dc_manifold_cell_incidence.restype = ctypes.POINTER(_DcIncidence)
    lib.dc_manifold_cell_incidence.argtypes = [u64, i8p, i32p, u8p, u64, u64, u64, u8p]
    lib.dc_free_incidence.restype = None
    lib.dc_free_incidence.argtypes = [ctypes.POINTER(_DcIncidence)]
    lib.dc_dual_faces.restype = ctypes.POINTER(_DcFaces)
    lib.dc_dual_faces.argtypes = [u64, i8p, i32p, u8p, u64, i32p, i32p, u64, u64, u64, f64p]
    lib.dc_free_faces.restype = None
    lib.dc_free_faces.argtypes = [ctypes.POINTER(_DcFaces)]
    lib.dc_qef_tikhonov.restype = i32
    lib.dc_qef_tikhonov.argtypes = [u64, f64p, f64p, u64, i32p, f64, f64p]
    lib.dc_qef_tikhonov_vjp.restype = i32
    lib.dc_qef_tikhonov_vjp.argtypes = [u64, f64p, f64p, u64, i32p, f64, f64p, f64p, f64p]
    lib.dc_sharp_qef.restype = i32
    lib.dc_sharp_qef.argtypes = [u64, f64p, f64p, u64, i32p, i32p, f64p, f64p, f64, f64p]

    if lib.dc_abi_version() != 1:
        raise ImportError(
            "The native mesher cdylib has an incompatible ABI version; rebuild it: "
            "cargo build --release --manifest-path native/Cargo.toml"
        )
    return lib


def _pointer(array: np.ndarray, kind) -> ctypes.POINTER:
    return array.ctypes.data_as(ctypes.POINTER(kind))


def _copy_out(pointer, count: int, dtype) -> np.ndarray:
    """Copy `count` items out of a Rust-owned buffer (safe for count == 0)."""
    if count == 0:
        return np.empty((0,), dtype=dtype)
    return np.ctypeslib.as_array(pointer, shape=(count,)).astype(dtype, copy=True)


def _edge_arrays(edges: CrossingEdges) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(edges.axis, dtype=np.int8),
        np.ascontiguousarray(edges.index, dtype=np.int32),
        np.ascontiguousarray(edges.start_inside, dtype=np.uint8),
    )


def find_crossing_edges_native(values: np.ndarray, *, level: float = 0.0) -> CrossingEdges:
    """Native drop-in for :func:`cadjoint.meshing.find_crossing_edges`.

    Bit-identical output (same edge order, indices, and orientation flags).
    """
    lattice = np.ascontiguousarray(values, dtype=np.float64)
    if lattice.ndim != 3 or any(count < 2 for count in lattice.shape):
        raise ValueError(
            f"values must be a lattice with at least two vertices per axis; received shape {lattice.shape}."
        )
    if not np.isfinite(lattice).all():
        raise ValueError("values must be finite; the lattice contains NaN or infinity.")
    if not np.isfinite(level):
        raise ValueError("level must be finite.")
    lib = _library()
    handle = lib.dc_find_crossing_edges(
        _pointer(lattice, ctypes.c_double), *lattice.shape, float(level)
    )
    if not handle:
        raise RuntimeError("Native crossing-edge detection failed.")
    try:
        result = handle.contents
        count = int(result.count)
        return CrossingEdges(
            axis=_copy_out(result.axis, count, np.int8),
            index=_copy_out(result.index, count * 3, np.int32).reshape((-1, 3)),
            start_inside=_copy_out(result.start_inside, count, np.uint8).astype(bool),
        )
    finally:
        lib.dc_free_edges(handle)


def manifold_cell_incidence_native(
    edges: CrossingEdges,
    grid: GridSpec,
    inside: np.ndarray,
) -> CellIncidence:
    """Native drop-in for :func:`cadjoint.meshing.manifold_cell_incidence`.

    Bit-identical output (same row order and per-row slot order).
    """
    inside_lattice = np.ascontiguousarray(inside, dtype=np.uint8)
    if inside_lattice.shape != grid.lattice_shape:
        raise ValueError(
            f"inside must be shaped {grid.lattice_shape}; received {inside_lattice.shape}."
        )
    if edges.count == 0:
        return CellIncidence(
            cells=np.empty((0, 3), dtype=np.int32),
            edge_ids=np.empty((0, 12), dtype=np.int32),
            counts=np.empty((0,), dtype=np.int32),
        )
    axis, index, start_inside = _edge_arrays(edges)
    lib = _library()
    handle = lib.dc_manifold_cell_incidence(
        edges.count,
        _pointer(axis, ctypes.c_int8),
        _pointer(index, ctypes.c_int32),
        _pointer(start_inside, ctypes.c_uint8),
        *grid.cells,
        _pointer(inside_lattice, ctypes.c_uint8),
    )
    if not handle:
        raise RuntimeError("Native cell-incidence construction failed.")
    try:
        result = handle.contents
        if result.error:
            raise ValueError(
                "inside is inconsistent with the edge set: a crossing edge's inside "
                "endpoint is not marked inside."
            )
        count = int(result.count)
        return CellIncidence(
            cells=_copy_out(result.cells, count * 3, np.int32).reshape((-1, 3)),
            edge_ids=_copy_out(result.edge_ids, count * 12, np.int32).reshape((-1, 12)),
            counts=_copy_out(result.counts, count, np.int32),
        )
    finally:
        lib.dc_free_incidence(handle)


def dual_faces_native(
    edges: CrossingEdges,
    incidence: CellIncidence,
    grid: GridSpec,
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Native drop-in for :func:`cadjoint.meshing.dual_contouring.dual_faces`."""
    axis, index, start_inside = _edge_arrays(edges)
    cells = np.ascontiguousarray(incidence.cells, dtype=np.int32)
    edge_ids = np.ascontiguousarray(incidence.edge_ids, dtype=np.int32)
    positions = np.ascontiguousarray(vertices, dtype=np.float64)
    lib = _library()
    handle = lib.dc_dual_faces(
        edges.count,
        _pointer(axis, ctypes.c_int8),
        _pointer(index, ctypes.c_int32),
        _pointer(start_inside, ctypes.c_uint8),
        incidence.count,
        _pointer(cells, ctypes.c_int32),
        _pointer(edge_ids, ctypes.c_int32),
        *grid.cells,
        _pointer(positions, ctypes.c_double),
    )
    if not handle:
        raise RuntimeError("Native dual-face construction failed.")
    try:
        result = handle.contents
        quad_count = int(result.quad_count)
        quads = _copy_out(result.quads, quad_count * 4, np.int32).reshape((-1, 4))
        triangles = _copy_out(result.triangles, quad_count * 6, np.int32).reshape((-1, 3))
        return quads, triangles, int(result.skipped_boundary)
    finally:
        lib.dc_free_faces(handle)


def sharp_qef_vertices_native(
    hermite: HermiteData,
    incidence: CellIncidence,
    grid: GridSpec,
    *,
    rcond: float = 5e-2,
) -> np.ndarray:
    """Native drop-in for :func:`cadjoint.meshing.dual_contouring.sharp_qef_vertices`."""
    if not 0 < rcond < 1:
        raise ValueError("rcond must be between 0 and 1.")
    points = np.ascontiguousarray(hermite.points, dtype=np.float64)
    normals = np.ascontiguousarray(hermite.unit_normals(), dtype=np.float64)
    return _qef_sharp_arrays(points, normals, incidence, grid, rcond)


def _qef_sharp_arrays(
    points: np.ndarray,
    normals: np.ndarray,
    incidence: CellIncidence,
    grid: GridSpec,
    rcond: float,
) -> np.ndarray:
    edge_ids = np.ascontiguousarray(incidence.edge_ids, dtype=np.int32)
    cells = np.ascontiguousarray(incidence.cells, dtype=np.int32)
    origin = np.asarray(grid.origin, dtype=np.float64)
    spacing = np.asarray(grid.spacing, dtype=np.float64)
    out = np.empty((incidence.count, 3), dtype=np.float64)
    status = _library().dc_sharp_qef(
        points.shape[0],
        _pointer(points, ctypes.c_double),
        _pointer(normals, ctypes.c_double),
        incidence.count,
        _pointer(edge_ids, ctypes.c_int32),
        _pointer(cells, ctypes.c_int32),
        _pointer(origin, ctypes.c_double),
        _pointer(spacing, ctypes.c_double),
        float(rcond),
        _pointer(out, ctypes.c_double),
    )
    if status != 0:
        raise RuntimeError("Native sharp QEF solve failed.")
    return out


def qef_forward_arrays(
    points: np.ndarray,
    normals: np.ndarray,
    edge_ids: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Unclamped Tikhonov QEF vertices from plain arrays (tesseract `apply`)."""
    points = np.ascontiguousarray(points, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    edge_ids = np.ascontiguousarray(edge_ids, dtype=np.int32)
    cell_count = edge_ids.shape[0]
    out = np.empty((cell_count, 3), dtype=np.float64)
    status = _library().dc_qef_tikhonov(
        points.shape[0],
        _pointer(points, ctypes.c_double),
        _pointer(normals, ctypes.c_double),
        cell_count,
        _pointer(edge_ids, ctypes.c_int32),
        float(regularization),
        _pointer(out, ctypes.c_double),
    )
    if status != 0:
        raise RuntimeError("Native Tikhonov QEF solve failed.")
    return out


def qef_vjp_arrays(
    points: np.ndarray,
    normals: np.ndarray,
    edge_ids: np.ndarray,
    regularization: float,
    cotangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hand-derived VJP of :func:`qef_forward_arrays` w.r.t. points/normals."""
    points = np.ascontiguousarray(points, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    edge_ids = np.ascontiguousarray(edge_ids, dtype=np.int32)
    cotangent = np.ascontiguousarray(cotangent, dtype=np.float64)
    points_bar = np.empty_like(points)
    normals_bar = np.empty_like(normals)
    status = _library().dc_qef_tikhonov_vjp(
        points.shape[0],
        _pointer(points, ctypes.c_double),
        _pointer(normals, ctypes.c_double),
        edge_ids.shape[0],
        _pointer(edge_ids, ctypes.c_int32),
        float(regularization),
        _pointer(cotangent, ctypes.c_double),
        _pointer(points_bar, ctypes.c_double),
        _pointer(normals_bar, ctypes.c_double),
    )
    if status != 0:
        raise RuntimeError("Native Tikhonov QEF VJP failed.")
    return points_bar, normals_bar


@lru_cache(maxsize=1)
def _qef_tesseract():
    """Load the packaged QEF tesseract locally (kept warm per process)."""
    _library()  # surface the build hint before any tesseract machinery
    try:
        from tesseract_core import Tesseract
    except ImportError as error:
        raise ImportError(_TESSERACT_HINT) from error
    return Tesseract.from_tesseract_api(str(_NATIVE_DIR / "tesseract_api.py"))


def qef_vertices_native(
    hermite: HermiteData,
    incidence: CellIncidence,
    grid: GridSpec,
    *,
    regularization: float = 1e-3,
) -> tuple[Array, Array]:
    """Differentiable drop-in for :func:`cadjoint.meshing.qef_vertices`.

    The Tikhonov solve runs in the Rust core behind the packaged tesseract:
    under ``jax.grad``, ``tesseract_jax.apply_tesseract`` dispatches to the
    ``vector_jacobian_product`` endpoint serving the hand-derived
    linear-solve VJP, so gradients w.r.t. the Hermite points and gradients
    (and through them any design parameters) match the reference JAX
    autodiff. Unit normalization and the final cell clamp stay in JAX so
    their subgradient semantics are identical to the reference.

    Requires jax x64 mode (the tesseract schema is float64).
    """
    import jax

    if not regularization > 0:
        raise ValueError("regularization must be positive.")
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "qef_vertices_native requires jax x64 mode; enable it with "
            'jax.config.update("jax_enable_x64", True).'
        )
    from tesseract_jax import apply_tesseract

    outputs = apply_tesseract(
        _qef_tesseract(),
        {
            "points": jnp.asarray(hermite.points, dtype=jnp.float64),
            "normals": jnp.asarray(hermite.unit_normals(), dtype=jnp.float64),
            "edge_ids": np.ascontiguousarray(incidence.edge_ids, dtype=np.int32),
            "regularization": np.asarray(regularization, dtype=np.float64),
        },
    )
    cell_min = jnp.asarray(
        np.asarray(grid.origin, dtype=np.float64)
        + incidence.cells * np.asarray(grid.spacing, dtype=np.float64)
    )
    vertices = jnp.clip(outputs["vertices"], cell_min, cell_min + jnp.asarray(grid.spacing))
    return vertices, _averaged_normals(hermite, incidence)


def extract_mesh_native(
    sdf: Callable[[Array], Array],
    grid: GridSpec,
    *,
    level: float = 0.0,
    bisection_iterations: int = 16,
    newton_steps: int = 1,
    regularization: float = 1e-3,
    sharp: bool = True,
    lipschitz: float | None = None,
) -> Mesh:
    """Native counterpart of :func:`cadjoint.meshing.extract_mesh`.

    Same signature and results: the SDF-evaluating stages (grid sampling,
    Hermite refinement) run in JAX exactly as the reference, while crossing
    detection, manifold incidence, QEF placement, and face construction run
    in the Rust core. Topology (faces, quads, cells, winding) is
    bit-identical to the reference; vertex positions agree to float64
    accuracy. With ``lipschitz`` set, octree-pruned detection stays on
    :mod:`cadjoint.meshing.adaptive` (it interleaves SDF evaluations) and
    the remaining stages go native.
    """
    if not regularization > 0:
        raise ValueError("regularization must be positive.")
    if lipschitz is None:
        values = sample_grid(sdf, grid)
        edges = find_crossing_edges_native(values, level=level)
        inside = (values - level) < 0
    else:
        from cadjoint.meshing.adaptive import sparse_crossing_edges

        edges, inside = sparse_crossing_edges(
            sdf, grid, level=level, lipschitz=lipschitz, return_inside=True
        )
    incidence = manifold_cell_incidence_native(edges, grid, inside)
    if edges.count == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return Mesh(
            vertices=jnp.asarray(empty),
            faces=np.empty((0, 3), dtype=np.int32),
            quads=np.empty((0, 4), dtype=np.int32),
            normals=jnp.asarray(empty.copy()),
            cells=np.empty((0, 3), dtype=np.int32),
        )
    hermite = edge_hermite_data(
        sdf,
        grid,
        edges,
        level=level,
        bisection_iterations=bisection_iterations,
        newton_steps=newton_steps,
    )
    normals = _averaged_normals(hermite, incidence)
    points = np.ascontiguousarray(hermite.points, dtype=np.float64)
    unit = np.ascontiguousarray(hermite.unit_normals(), dtype=np.float64)
    if sharp:
        placed = _qef_sharp_arrays(points, unit, incidence, grid, 5e-2)
    else:
        placed = qef_forward_arrays(points, unit, incidence.edge_ids, regularization)
        spacing = np.asarray(grid.spacing, dtype=np.float64)
        cell_min = np.asarray(grid.origin, dtype=np.float64) + incidence.cells * spacing
        placed = np.clip(placed, cell_min, cell_min + spacing)
    vertices = jnp.asarray(placed, dtype=normals.dtype)
    quads, faces, skipped_boundary = dual_faces_native(
        edges, incidence, grid, np.asarray(vertices, dtype=np.float64)
    )
    if skipped_boundary:
        warnings.warn(
            f"The isosurface crosses the extraction boundary on {skipped_boundary} "
            "grid edges; the returned mesh is open.",
            stacklevel=2,
        )
    return Mesh(
        vertices=vertices,
        faces=faces,
        quads=quads,
        normals=normals,
        cells=incidence.cells.copy(),
    )
