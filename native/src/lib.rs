//! C ABI of the native dual-contouring core.
//!
//! Consumed from Python via ctypes (`cadjoint/meshing/native.py`) and, for
//! the differentiable Tikhonov QEF, through the tesseract endpoints in
//! `native/tesseract_api.py`. Variable-size results (edges, incidence,
//! faces) are returned as heap-allocated handle structs the caller must
//! free with the matching `dc_free_*`; fixed-size results are written into
//! caller-allocated buffers.
//!
//! Every entry point is wrapped in `catch_unwind` so a bug can never unwind
//! across the FFI boundary; failures surface as null pointers / non-zero
//! status codes.

mod core;

use std::panic::{catch_unwind, AssertUnwindSafe};

fn leak_vec<T>(mut vec: Vec<T>) -> *mut T {
    vec.shrink_to_fit();
    let pointer = vec.as_mut_ptr();
    std::mem::forget(vec);
    pointer
}

/// # Safety
/// `pointer` must come from `leak_vec` with the same `length`.
unsafe fn free_vec<T>(pointer: *mut T, length: usize) {
    if !pointer.is_null() {
        drop(Vec::from_raw_parts(pointer, length, length));
    }
}

#[no_mangle]
pub extern "C" fn dc_abi_version() -> u32 {
    1
}

// ---------------------------------------------------------------------------
// Crossing edges
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct DcEdges {
    pub count: u64,
    pub axis: *mut i8,
    pub index: *mut i32,
    pub start_inside: *mut u8,
}

/// # Safety
/// `values` must point to `mx * my * mz` f64 lattice values (row-major).
#[no_mangle]
pub unsafe extern "C" fn dc_find_crossing_edges(
    values: *const f64,
    mx: u64,
    my: u64,
    mz: u64,
    level: f64,
) -> *mut DcEdges {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let values = std::slice::from_raw_parts(values, (mx * my * mz) as usize);
        let edges =
            core::find_crossing_edges(values, [mx as usize, my as usize, mz as usize], level);
        let count = edges.axis.len() as u64;
        Box::into_raw(Box::new(DcEdges {
            count,
            axis: leak_vec(edges.axis),
            index: leak_vec(edges.index),
            start_inside: leak_vec(edges.start_inside),
        }))
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `edges` must come from `dc_find_crossing_edges` and not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn dc_free_edges(edges: *mut DcEdges) {
    if edges.is_null() {
        return;
    }
    let boxed = Box::from_raw(edges);
    let count = boxed.count as usize;
    free_vec(boxed.axis, count);
    free_vec(boxed.index, count * 3);
    free_vec(boxed.start_inside, count);
}

// ---------------------------------------------------------------------------
// Manifold cell incidence
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct DcIncidence {
    pub count: u64,
    pub cells: *mut i32,
    pub edge_ids: *mut i32,
    pub counts: *mut i32,
    /// 0 ok; 1 inconsistent inside lattice (reference raises ValueError).
    pub error: i32,
}

/// # Safety
/// The edge arrays must hold `edge_count` entries (index: `3 * edge_count`);
/// `inside` must hold `(cx + 1) * (cy + 1) * (cz + 1)` bytes (row-major).
#[no_mangle]
pub unsafe extern "C" fn dc_manifold_cell_incidence(
    edge_count: u64,
    axis: *const i8,
    index: *const i32,
    start_inside: *const u8,
    cx: u64,
    cy: u64,
    cz: u64,
    inside: *const u8,
) -> *mut DcIncidence {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let count = edge_count as usize;
        let edges = core::Edges {
            axis: std::slice::from_raw_parts(axis, count).to_vec(),
            index: std::slice::from_raw_parts(index, count * 3).to_vec(),
            start_inside: std::slice::from_raw_parts(start_inside, count).to_vec(),
        };
        let lattice = ((cx + 1) * (cy + 1) * (cz + 1)) as usize;
        let inside = std::slice::from_raw_parts(inside, lattice);
        match core::manifold_cell_incidence(&edges, [cx as i64, cy as i64, cz as i64], inside) {
            Ok(incidence) => {
                let rows = incidence.counts.len() as u64;
                Box::into_raw(Box::new(DcIncidence {
                    count: rows,
                    cells: leak_vec(incidence.cells),
                    edge_ids: leak_vec(incidence.edge_ids),
                    counts: leak_vec(incidence.counts),
                    error: 0,
                }))
            }
            Err(()) => Box::into_raw(Box::new(DcIncidence {
                count: 0,
                cells: std::ptr::null_mut(),
                edge_ids: std::ptr::null_mut(),
                counts: std::ptr::null_mut(),
                error: 1,
            })),
        }
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `incidence` must come from `dc_manifold_cell_incidence`; free once.
#[no_mangle]
pub unsafe extern "C" fn dc_free_incidence(incidence: *mut DcIncidence) {
    if incidence.is_null() {
        return;
    }
    let boxed = Box::from_raw(incidence);
    let rows = boxed.count as usize;
    free_vec(boxed.cells, rows * 3);
    free_vec(boxed.edge_ids, rows * 12);
    free_vec(boxed.counts, rows);
}

// ---------------------------------------------------------------------------
// Dual faces
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct DcFaces {
    pub quad_count: u64,
    pub quads: *mut i32,
    pub triangles: *mut i32,
    pub skipped_boundary: u64,
}

/// # Safety
/// Edge arrays sized by `edge_count`, incidence arrays by `cell_count`
/// (cells: `3n`, edge_ids: `12n`), `vertices` holds `3 * cell_count` f64.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn dc_dual_faces(
    edge_count: u64,
    axis: *const i8,
    index: *const i32,
    start_inside: *const u8,
    cell_count: u64,
    cells: *const i32,
    edge_ids: *const i32,
    cx: u64,
    cy: u64,
    cz: u64,
    vertices: *const f64,
) -> *mut DcFaces {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let count = edge_count as usize;
        let rows = cell_count as usize;
        let edges = core::Edges {
            axis: std::slice::from_raw_parts(axis, count).to_vec(),
            index: std::slice::from_raw_parts(index, count * 3).to_vec(),
            start_inside: std::slice::from_raw_parts(start_inside, count).to_vec(),
        };
        let incidence = core::Incidence {
            cells: std::slice::from_raw_parts(cells, rows * 3).to_vec(),
            edge_ids: std::slice::from_raw_parts(edge_ids, rows * 12).to_vec(),
            counts: vec![0; rows],
        };
        let vertices = std::slice::from_raw_parts(vertices, rows * 3);
        let faces =
            core::dual_faces(&edges, &incidence, [cx as i64, cy as i64, cz as i64], vertices);
        let quad_count = (faces.quads.len() / 4) as u64;
        Box::into_raw(Box::new(DcFaces {
            quad_count,
            quads: leak_vec(faces.quads),
            triangles: leak_vec(faces.triangles),
            skipped_boundary: faces.skipped_boundary,
        }))
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `faces` must come from `dc_dual_faces`; free once.
#[no_mangle]
pub unsafe extern "C" fn dc_free_faces(faces: *mut DcFaces) {
    if faces.is_null() {
        return;
    }
    let boxed = Box::from_raw(faces);
    let quads = boxed.quad_count as usize;
    free_vec(boxed.quads, quads * 4);
    free_vec(boxed.triangles, quads * 6);
}

// ---------------------------------------------------------------------------
// QEF vertex placement
// ---------------------------------------------------------------------------

/// # Safety
/// `points`/`normals`: `3 * point_count` f64; `edge_ids`: `12 * cell_count`
/// i32; `out_vertices`: `3 * cell_count` f64 (caller-allocated).
#[no_mangle]
pub unsafe extern "C" fn dc_qef_tikhonov(
    point_count: u64,
    points: *const f64,
    normals: *const f64,
    cell_count: u64,
    edge_ids: *const i32,
    regularization: f64,
    out_vertices: *mut f64,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let samples = point_count as usize;
        let rows = cell_count as usize;
        core::qef_tikhonov(
            std::slice::from_raw_parts(points, samples * 3),
            std::slice::from_raw_parts(normals, samples * 3),
            std::slice::from_raw_parts(edge_ids, rows * 12),
            rows,
            regularization,
            std::slice::from_raw_parts_mut(out_vertices, rows * 3),
        );
    }));
    if result.is_ok() {
        0
    } else {
        1
    }
}

/// # Safety
/// As `dc_qef_tikhonov`; `cotangent`: `3 * cell_count` f64;
/// `out_points_bar` / `out_normals_bar`: `3 * point_count` f64 each.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn dc_qef_tikhonov_vjp(
    point_count: u64,
    points: *const f64,
    normals: *const f64,
    cell_count: u64,
    edge_ids: *const i32,
    regularization: f64,
    cotangent: *const f64,
    out_points_bar: *mut f64,
    out_normals_bar: *mut f64,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let samples = point_count as usize;
        let rows = cell_count as usize;
        core::qef_tikhonov_vjp(
            std::slice::from_raw_parts(points, samples * 3),
            std::slice::from_raw_parts(normals, samples * 3),
            std::slice::from_raw_parts(edge_ids, rows * 12),
            rows,
            regularization,
            std::slice::from_raw_parts(cotangent, rows * 3),
            std::slice::from_raw_parts_mut(out_points_bar, samples * 3),
            std::slice::from_raw_parts_mut(out_normals_bar, samples * 3),
        );
    }));
    if result.is_ok() {
        0
    } else {
        1
    }
}

/// # Safety
/// As `dc_qef_tikhonov`, plus `cells`: `3 * cell_count` i32 and
/// `origin`/`spacing`: 3 f64 each.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn dc_sharp_qef(
    point_count: u64,
    points: *const f64,
    normals: *const f64,
    cell_count: u64,
    edge_ids: *const i32,
    cells: *const i32,
    origin: *const f64,
    spacing: *const f64,
    rcond: f64,
    out_vertices: *mut f64,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let samples = point_count as usize;
        let rows = cell_count as usize;
        let origin = std::slice::from_raw_parts(origin, 3);
        let spacing = std::slice::from_raw_parts(spacing, 3);
        core::sharp_qef(
            std::slice::from_raw_parts(points, samples * 3),
            std::slice::from_raw_parts(normals, samples * 3),
            std::slice::from_raw_parts(edge_ids, rows * 12),
            std::slice::from_raw_parts(cells, rows * 3),
            rows,
            [origin[0], origin[1], origin[2]],
            [spacing[0], spacing[1], spacing[2]],
            rcond,
            std::slice::from_raw_parts_mut(out_vertices, rows * 3),
        );
    }));
    if result.is_ok() {
        0
    } else {
        1
    }
}
