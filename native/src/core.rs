//! Algorithms of the native dual-contouring core.
//!
//! Every function mirrors its Python reference in `cadjoint/meshing/*` down
//! to ordering: discrete outputs (edge sets, incidence rows, faces) are
//! bit-identical to the NumPy implementation, continuous outputs (QEF
//! vertices) agree to floating-point accuracy.

use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Crossing-edge detection (mirrors edge_detection.find_crossing_edges)
// ---------------------------------------------------------------------------

pub struct Edges {
    pub axis: Vec<i8>,
    pub index: Vec<i32>, // count * 3, edge start lattice coordinates
    pub start_inside: Vec<u8>,
}

/// Sign-change sweep over the lattice; axis-major then row-major order,
/// exactly like `np.argwhere` per axis in the reference.
pub fn find_crossing_edges(values: &[f64], m: [usize; 3], level: f64) -> Edges {
    let [mx, my, mz] = m;
    // Below ~64^3 lattice points the whole sweep is cheaper than rayon setup.
    let parallel = mx * my * mz >= 256 * 1024;
    let inside: Vec<bool> = if parallel {
        values.par_iter().map(|v| (v - level) < 0.0).collect()
    } else {
        values.iter().map(|v| (v - level) < 0.0).collect()
    };
    let at = |i: usize, j: usize, k: usize| inside[(i * my + j) * mz + k];

    let mut axis = Vec::new();
    let mut index = Vec::new();
    let mut start_inside = Vec::new();
    for a in 0..3usize {
        let limit = [
            if a == 0 { mx - 1 } else { mx },
            if a == 1 { my - 1 } else { my },
            if a == 2 { mz - 1 } else { mz },
        ];
        let step = [(a == 0) as usize, (a == 1) as usize, (a == 2) as usize];
        let sweep_slab = |i: usize| {
            let mut idx = Vec::new();
            let mut side = Vec::new();
            for j in 0..limit[1] {
                for k in 0..limit[2] {
                    let a_in = at(i, j, k);
                    let b_in = at(i + step[0], j + step[1], k + step[2]);
                    if a_in != b_in {
                        idx.extend_from_slice(&[i as i32, j as i32, k as i32]);
                        side.push(a_in as u8);
                    }
                }
            }
            (idx, side)
        };
        // Parallel over the outer (x) slabs; ordered collect keeps the
        // row-major candidate order of the reference.
        let slabs: Vec<(Vec<i32>, Vec<u8>)> = if parallel {
            (0..limit[0]).into_par_iter().map(sweep_slab).collect()
        } else {
            (0..limit[0]).map(sweep_slab).collect()
        };
        for (idx, side) in slabs {
            axis.resize(axis.len() + side.len(), a as i8);
            index.extend_from_slice(&idx);
            start_inside.extend_from_slice(&side);
        }
    }
    Edges { axis, index, start_inside }
}

// ---------------------------------------------------------------------------
// Manifold cell incidence (mirrors features.manifold_cell_incidence)
// ---------------------------------------------------------------------------

/// Lattice offsets from a cell to the start vertices of its twelve edges,
/// grouped by edge axis (`features._CELL_EDGE_OFFSETS`).
const CELL_EDGE_OFFSETS: [[[i64; 3]; 4]; 3] = [
    [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]],
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
];

pub struct Incidence {
    pub cells: Vec<i32>,    // count * 3
    pub edge_ids: Vec<i32>, // count * 12, padded with -1
    pub counts: Vec<i32>,
}

/// Connected components of a cell's inside corners: label each inside
/// corner with its id and propagate the minimum across the twelve corner
/// edges (corner ids differing in one bit) to a fixed point; outside
/// corners stay 8. Identical to the reference fixed point.
fn corner_labels(corner_inside: [bool; 8]) -> [u8; 8] {
    let mut labels = [8u8; 8];
    for c in 0..8 {
        if corner_inside[c] {
            labels[c] = c as u8;
        }
    }
    loop {
        let mut changed = false;
        for a in 0..8usize {
            for bit in [4usize, 2, 1] {
                if a & bit != 0 {
                    continue;
                }
                let b = a | bit;
                if corner_inside[a] && corner_inside[b] {
                    let merged = labels[a].min(labels[b]);
                    if labels[a] != merged || labels[b] != merged {
                        labels[a] = merged;
                        labels[b] = merged;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            return labels;
        }
    }
}

/// Group crossing edges by (cell, inside-corner component).
///
/// `inside` is the boolean lattice (shape `cells + 1` per axis, row-major).
/// Returns `Err(())` when a crossing edge's inside endpoint is not marked
/// inside (the reference raises ValueError).
pub fn manifold_cell_incidence(
    edges: &Edges,
    grid_cells: [i64; 3],
    inside: &[u8],
) -> Result<Incidence, ()> {
    let edge_count = edges.axis.len();
    if edge_count == 0 {
        return Ok(Incidence { cells: Vec::new(), edge_ids: Vec::new(), counts: Vec::new() });
    }
    let [cx, cy, cz] = grid_cells;
    let (my, mz) = ((cy + 1) as usize, (cz + 1) as usize);
    let lattice_inside =
        |i: i64, j: i64, k: i64| inside[(i as usize * my + j as usize) * mz + k as usize] != 0;

    // Candidate (cell, corner, edge) triples in reference order: axis-major
    // over the edge set, then the four neighbor-cell offsets.
    struct Candidate {
        cell_flat: i64,
        corner: u8,
        edge: i32,
    }
    let mut candidates: Vec<Candidate> = Vec::with_capacity(edge_count * 4);
    let strides = [cy * cz, cz, 1i64];
    for a in 0..3usize {
        for e in 0..edge_count {
            if edges.axis[e] as usize != a {
                continue;
            }
            let start = [
                edges.index[e * 3] as i64,
                edges.index[e * 3 + 1] as i64,
                edges.index[e * 3 + 2] as i64,
            ];
            let mut inward = [0i64; 3];
            if edges.start_inside[e] == 0 {
                inward[a] = 1;
            }
            for offsets in &CELL_EDGE_OFFSETS[a] {
                let cell = [start[0] - offsets[0], start[1] - offsets[1], start[2] - offsets[2]];
                if cell[0] < 0
                    || cell[1] < 0
                    || cell[2] < 0
                    || cell[0] >= cx
                    || cell[1] >= cy
                    || cell[2] >= cz
                {
                    continue;
                }
                let corner = [
                    start[0] + inward[0] - cell[0],
                    start[1] + inward[1] - cell[1],
                    start[2] + inward[2] - cell[2],
                ];
                candidates.push(Candidate {
                    cell_flat: cell[0] * strides[0] + cell[1] * strides[1] + cell[2],
                    corner: (corner[0] * 4 + corner[1] * 2 + corner[2]) as u8,
                    edge: e as i32,
                });
            }
        }
    }

    // Unique cells (ascending flat key) and their corner component labels.
    let mut unique_cells: Vec<i64> = candidates.iter().map(|c| c.cell_flat).collect();
    unique_cells.par_sort_unstable();
    unique_cells.dedup();
    let labels: Vec<[u8; 8]> = unique_cells
        .par_iter()
        .map(|&flat| {
            let cell = [flat / strides[0], (flat % strides[0]) / strides[1], flat % strides[1]];
            let mut corner_inside = [false; 8];
            for (id, item) in corner_inside.iter_mut().enumerate() {
                let (dx, dy, dz) = ((id >> 2) as i64 & 1, (id >> 1) as i64 & 1, id as i64 & 1);
                *item = lattice_inside(cell[0] + dx, cell[1] + dy, cell[2] + dz);
            }
            corner_labels(corner_inside)
        })
        .collect();

    // Group key = cell rank * 8 + component; stable order within a group is
    // the candidate order (sort on (key, original position)).
    let mut keyed: Vec<(i64, u32)> = Vec::with_capacity(candidates.len());
    for (position, candidate) in candidates.iter().enumerate() {
        let rank = unique_cells.binary_search(&candidate.cell_flat).expect("cell must exist");
        let component = labels[rank][candidate.corner as usize];
        if component == 8 {
            return Err(());
        }
        keyed.push(((rank as i64) * 8 + component as i64, position as u32));
    }
    keyed.par_sort_unstable();

    let mut cells = Vec::new();
    let mut edge_ids = Vec::new();
    let mut counts: Vec<i32> = Vec::new();
    let mut previous_key = i64::MIN;
    for &(key, position) in &keyed {
        if key != previous_key {
            previous_key = key;
            let flat = unique_cells[(key / 8) as usize];
            cells.extend_from_slice(&[
                (flat / strides[0]) as i32,
                ((flat % strides[0]) / strides[1]) as i32,
                (flat % strides[1]) as i32,
            ]);
            edge_ids.resize(edge_ids.len() + 12, -1);
            counts.push(0);
        }
        let row = counts.len() - 1;
        let slot = counts[row] as usize;
        edge_ids[row * 12 + slot] = candidates[position as usize].edge;
        counts[row] += 1;
    }
    Ok(Incidence { cells, edge_ids, counts })
}

// ---------------------------------------------------------------------------
// Dual faces (mirrors dual_contouring.dual_faces)
// ---------------------------------------------------------------------------

/// Neighbor cells around an axis-`a` edge as (du, dv) offsets on the two
/// other axes; counterclockwise seen from +a (`_QUAD_NEIGHBOR_OFFSETS`).
const QUAD_NEIGHBOR_OFFSETS: [[i64; 2]; 4] = [[1, 1], [0, 1], [0, 0], [1, 0]];

pub struct Faces {
    pub quads: Vec<i32>,     // quad_count * 4
    pub triangles: Vec<i32>, // quad_count * 2 * 3
    pub skipped_boundary: u64,
}

/// One oriented quad per interior crossing edge, triangulated along the
/// shorter diagonal; winding decided by the frozen `start_inside` flag.
pub fn dual_faces(
    edges: &Edges,
    incidence: &Incidence,
    grid_cells: [i64; 3],
    vertices: &[f64],
) -> Faces {
    let edge_count = edges.axis.len();
    let cell_count = incidence.counts.len();
    let strides = [grid_cells[1] * grid_cells[2], grid_cells[2], 1i64];

    // Per-edge registry of (cell key, incidence row): each edge belongs to
    // at most four cells and to exactly one row per cell.
    #[derive(Clone, Copy)]
    struct Registry {
        items: [(i64, i32); 4],
        len: u8,
    }
    let mut registry =
        vec![Registry { items: [(0, 0); 4], len: 0 }; edge_count];
    for row in 0..cell_count {
        let flat = incidence.cells[row * 3] as i64 * strides[0]
            + incidence.cells[row * 3 + 1] as i64 * strides[1]
            + incidence.cells[row * 3 + 2] as i64;
        for slot in 0..12 {
            let edge = incidence.edge_ids[row * 12 + slot];
            if edge >= 0 {
                let entry = &mut registry[edge as usize];
                // An edge touches at most four cells; ignore malformed extras.
                if (entry.len as usize) < entry.items.len() {
                    entry.items[entry.len as usize] = (flat, row as i32);
                    entry.len += 1;
                }
            }
        }
    }

    let quads_by_edge: Vec<Option<([i32; 4], bool)>> = (0..edge_count)
        .into_par_iter()
        .map(|e| {
            let a = edges.axis[e] as usize;
            let (u, v) = ((a + 1) % 3, (a + 2) % 3);
            let start = [
                edges.index[e * 3] as i64,
                edges.index[e * 3 + 1] as i64,
                edges.index[e * 3 + 2] as i64,
            ];
            if start[u] <= 0
                || start[u] >= grid_cells[u]
                || start[v] <= 0
                || start[v] >= grid_cells[v]
            {
                return Some(([0; 4], false)); // boundary marker, filtered below
            }
            let mut rows = [0i32; 4];
            for (corner, offset) in QUAD_NEIGHBOR_OFFSETS.iter().enumerate() {
                let mut cell = start;
                cell[u] -= offset[0];
                cell[v] -= offset[1];
                let flat = cell[0] * strides[0] + cell[1] * strides[1] + cell[2];
                let entry = &registry[e];
                let mut found = -1i32;
                for item in &entry.items[..entry.len as usize] {
                    if item.0 == flat {
                        found = item.1;
                        break;
                    }
                }
                if found < 0 {
                    return None; // incomplete quad: never emit (reference parity)
                }
                rows[corner] = found;
            }
            if edges.start_inside[e] == 0 {
                rows.reverse();
            }
            Some((rows, true))
        })
        .collect();

    let skipped_boundary =
        quads_by_edge.iter().filter(|q| matches!(q, Some((_, false)))).count() as u64;

    let mut quads = Vec::new();
    let mut triangles = Vec::new();
    let position = |r: i32| {
        let r = r as usize * 3;
        [vertices[r], vertices[r + 1], vertices[r + 2]]
    };
    for quad in quads_by_edge.iter().flatten() {
        let (rows, interior) = quad;
        if !interior {
            continue;
        }
        let [a, b, c, d] = *rows;
        quads.extend_from_slice(rows);
        let (pa, pb, pc, pd) = (position(a), position(b), position(c), position(d));
        let squared = |p: [f64; 3], q: [f64; 3]| {
            (p[0] - q[0]) * (p[0] - q[0])
                + (p[1] - q[1]) * (p[1] - q[1])
                + (p[2] - q[2]) * (p[2] - q[2])
        };
        if squared(pa, pc) <= squared(pb, pd) {
            triangles.extend_from_slice(&[a, b, c, a, c, d]);
        } else {
            triangles.extend_from_slice(&[a, b, d, b, c, d]);
        }
    }
    Faces { quads, triangles, skipped_boundary }
}

// ---------------------------------------------------------------------------
// QEF vertex placement (mirrors dual_contouring.qef_vertices / sharp_qef_vertices)
// ---------------------------------------------------------------------------

/// Gathered per-cell Hermite slots: masked unit normals and points, the
/// valid-slot mass point, and the valid count (clamped to 1). Mirrors
/// `_gather_incident` + `_masked_mean` exactly: mass-point statistics weigh
/// every real slot; normals additionally drop degenerate rows.
struct CellSlots {
    normals: [[f64; 3]; 12],
    points: [[f64; 3]; 12],
    valid: [bool; 12],
    mass_point: [f64; 3],
    count: f64,
}

fn gather_cell(
    points: &[f64],
    normals: &[f64],
    edge_ids: &[i32],
    cell: usize,
) -> CellSlots {
    let mut slots = CellSlots {
        normals: [[0.0; 3]; 12],
        points: [[0.0; 3]; 12],
        valid: [false; 12],
        mass_point: [0.0; 3],
        count: 0.0,
    };
    let mut valid_count = 0i64;
    for slot in 0..12 {
        let edge = edge_ids[cell * 12 + slot];
        let gather = edge.max(0) as usize * 3;
        let point = [points[gather], points[gather + 1], points[gather + 2]];
        slots.points[slot] = point;
        if edge >= 0 {
            slots.valid[slot] = true;
            valid_count += 1;
            for axis in 0..3 {
                slots.mass_point[axis] += point[axis];
            }
            let normal = [normals[gather], normals[gather + 1], normals[gather + 2]];
            let magnitude =
                normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
            if magnitude > 0.25 {
                slots.normals[slot] = normal;
            }
        }
    }
    slots.count = valid_count.max(1) as f64;
    for axis in 0..3 {
        slots.mass_point[axis] /= slots.count;
    }
    slots
}

/// Solve the SPD 3x3 system via Cholesky (A well conditioned by damping).
fn solve_spd3(a: &[[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let l00 = a[0][0].sqrt();
    let l10 = a[1][0] / l00;
    let l20 = a[2][0] / l00;
    let l11 = (a[1][1] - l10 * l10).sqrt();
    let l21 = (a[2][1] - l20 * l10) / l11;
    let l22 = (a[2][2] - l20 * l20 - l21 * l21).sqrt();
    let y0 = b[0] / l00;
    let y1 = (b[1] - l10 * y0) / l11;
    let y2 = (b[2] - l20 * y0 - l21 * y1) / l22;
    let x2 = y2 / l22;
    let x1 = (y1 - l21 * x2) / l11;
    let x0 = (y0 - l10 * x1 - l20 * x2) / l00;
    [x0, x1, x2]
}

/// Tikhonov QEF forward: `v = m + (sum n n^T + lambda*count*I)^-1 sum n (n.(p-m))`.
/// Returns the *unclamped* vertex; the caller (JAX) applies the cell clamp so
/// its subgradient semantics match the reference bit for bit.
pub fn qef_tikhonov(
    points: &[f64],
    normals: &[f64],
    edge_ids: &[i32],
    _cell_count: usize,
    regularization: f64,
    out_vertices: &mut [f64],
) {
    out_vertices
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(cell, out)| {
            let slots = gather_cell(points, normals, edge_ids, cell);
            let damping = regularization * slots.count;
            let mut a = [[0.0f64; 3]; 3];
            let mut rhs = [0.0f64; 3];
            for slot in 0..12 {
                let n = slots.normals[slot];
                let d = [
                    slots.points[slot][0] - slots.mass_point[0],
                    slots.points[slot][1] - slots.mass_point[1],
                    slots.points[slot][2] - slots.mass_point[2],
                ];
                let offset = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
                for i in 0..3 {
                    rhs[i] += n[i] * offset;
                    for j in 0..3 {
                        a[i][j] += n[i] * n[j];
                    }
                }
            }
            for i in 0..3 {
                a[i][i] += damping;
            }
            let x = solve_spd3(&a, rhs);
            for i in 0..3 {
                out[i] = slots.mass_point[i] + x[i];
            }
        });
}

/// Hand-derived VJP of `qef_tikhonov` w.r.t. the Hermite points and unit
/// normals (linear-solve differentiation; the gather/mask/mass-point chain
/// matches JAX reverse mode over the reference implementation).
pub fn qef_tikhonov_vjp(
    points: &[f64],
    normals: &[f64],
    edge_ids: &[i32],
    cell_count: usize,
    regularization: f64,
    cotangent: &[f64],
    out_points_bar: &mut [f64],
    out_normals_bar: &mut [f64],
) {
    let point_count = points.len() / 3;
    let threads = rayon::current_num_threads().max(1);
    let chunk = cell_count.div_ceil(threads).max(1);
    let ranges: Vec<(usize, usize)> = (0..cell_count)
        .step_by(chunk)
        .map(|start| (start, (start + chunk).min(cell_count)))
        .collect();

    let partials: Vec<(Vec<f64>, Vec<f64>)> = ranges
        .into_par_iter()
        .map(|(start, stop)| {
            let mut points_bar = vec![0.0f64; point_count * 3];
            let mut normals_bar = vec![0.0f64; point_count * 3];
            for cell in start..stop {
                let slots = gather_cell(points, normals, edge_ids, cell);
                let damping = regularization * slots.count;
                let mut a = [[0.0f64; 3]; 3];
                let mut rhs = [0.0f64; 3];
                for slot in 0..12 {
                    let n = slots.normals[slot];
                    let d = [
                        slots.points[slot][0] - slots.mass_point[0],
                        slots.points[slot][1] - slots.mass_point[1],
                        slots.points[slot][2] - slots.mass_point[2],
                    ];
                    let offset = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
                    for i in 0..3 {
                        rhs[i] += n[i] * offset;
                        for j in 0..3 {
                            a[i][j] += n[i] * n[j];
                        }
                    }
                }
                for i in 0..3 {
                    a[i][i] += damping;
                }
                let x = solve_spd3(&a, rhs);
                let vbar = [
                    cotangent[cell * 3],
                    cotangent[cell * 3 + 1],
                    cotangent[cell * 3 + 2],
                ];
                // v = m + x, x = A^-1 b (A symmetric):
                //   u = A^-1 vbar, bbar = u, Abar = -u x^T.
                let u = solve_spd3(&a, vbar);

                // Mass-point cotangent: direct term plus b's -sum n n^T u.
                let mut mbar = vbar;
                for slot in 0..12 {
                    let n = slots.normals[slot];
                    let nu = n[0] * u[0] + n[1] * u[1] + n[2] * u[2];
                    for i in 0..3 {
                        mbar[i] -= n[i] * nu;
                    }
                }
                for slot in 0..12 {
                    let edge = edge_ids[cell * 12 + slot];
                    if edge < 0 {
                        continue;
                    }
                    let n = slots.normals[slot];
                    let d = [
                        slots.points[slot][0] - slots.mass_point[0],
                        slots.points[slot][1] - slots.mass_point[1],
                        slots.points[slot][2] - slots.mass_point[2],
                    ];
                    let offset = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
                    let nu = n[0] * u[0] + n[1] * u[1] + n[2] * u[2];
                    let nx = n[0] * x[0] + n[1] * x[1] + n[2] * x[2];
                    let target = edge as usize * 3;
                    for i in 0..3 {
                        // b-term: u*(n.d) + (n.u)*d; A-term: (Abar+Abar^T) n.
                        normals_bar[target + i] +=
                            u[i] * offset + nu * d[i] - u[i] * nx - x[i] * nu;
                        // b-term n (n.u) plus the mass-point mean chain.
                        points_bar[target + i] += n[i] * nu + mbar[i] / slots.count;
                    }
                }
            }
            (points_bar, normals_bar)
        })
        .collect();

    out_points_bar.fill(0.0);
    out_normals_bar.fill(0.0);
    for (points_bar, normals_bar) in partials {
        for i in 0..point_count * 3 {
            out_points_bar[i] += points_bar[i];
            out_normals_bar[i] += normals_bar[i];
        }
    }
}

/// Jacobi eigendecomposition of a symmetric 3x3 matrix; returns
/// (eigenvalues, eigenvectors as columns), descending.
fn eigh3(matrix: &[[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = *matrix;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _sweep in 0..32 {
        let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if off < 1e-30 * (a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2] + 1e-300) {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let (vkp, vkq) = (v[k][p], v[k][q]);
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[j][j].partial_cmp(&a[i][i]).unwrap());
    let eigenvalues = [a[order[0]][order[0]], a[order[1]][order[1]], a[order[2]][order[2]]];
    let mut vectors = [[0.0f64; 3]; 3];
    for (rank, &source) in order.iter().enumerate() {
        for k in 0..3 {
            vectors[k][rank] = v[k][source];
        }
    }
    (eigenvalues, vectors)
}

/// Rank-escalating truncated-SVD QEF (mirrors `sharp_qef_vertices`): per
/// truncation rank the clamped candidate with the smallest actual QEF error
/// wins, ties preferring the higher rank.
#[allow(clippy::too_many_arguments)]
pub fn sharp_qef(
    points: &[f64],
    normals: &[f64],
    edge_ids: &[i32],
    cells: &[i32],
    _cell_count: usize,
    origin: [f64; 3],
    spacing: [f64; 3],
    rcond: f64,
    out_vertices: &mut [f64],
) {
    out_vertices
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(cell, out)| {
            let slots = gather_cell(points, normals, edge_ids, cell);
            // Gram matrix of the masked normal rows; its eigenpairs give the
            // singular values/right vectors of the 12x3 normal matrix.
            let mut gram = [[0.0f64; 3]; 3];
            let mut w = [0.0f64; 3]; // A^T offsets
            for slot in 0..12 {
                let n = slots.normals[slot];
                let d = [
                    slots.points[slot][0] - slots.mass_point[0],
                    slots.points[slot][1] - slots.mass_point[1],
                    slots.points[slot][2] - slots.mass_point[2],
                ];
                let offset = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
                for i in 0..3 {
                    w[i] += n[i] * offset;
                    for j in 0..3 {
                        gram[i][j] += n[i] * n[j];
                    }
                }
            }
            let (eigenvalues, vectors) = eigh3(&gram);
            let singular = [
                eigenvalues[0].max(0.0).sqrt(),
                eigenvalues[1].max(0.0).sqrt(),
                eigenvalues[2].max(0.0).sqrt(),
            ];
            let leading = singular[0].max(1e-30);
            let mut inverse = [0.0f64; 3];
            for j in 0..3 {
                if singular[j] > rcond * leading {
                    inverse[j] = 1.0 / singular[j].max(1e-30);
                }
            }
            // Per-direction solution coefficient: inverse_j * (u_j . f)
            // with u_j . f = (v_j . w) / s_j.
            let mut coefficient = [0.0f64; 3];
            for j in 0..3 {
                let vw = vectors[0][j] * w[0] + vectors[1][j] * w[1] + vectors[2][j] * w[2];
                coefficient[j] = inverse[j] * (vw / singular[j].max(1e-30));
            }

            let cell_min = [
                origin[0] + cells[cell * 3] as f64 * spacing[0],
                origin[1] + cells[cell * 3 + 1] as f64 * spacing[1],
                origin[2] + cells[cell * 3 + 2] as f64 * spacing[2],
            ];
            let mut best = [0.0f64; 3];
            let mut best_error = f64::INFINITY;
            for rank in [3usize, 2, 1] {
                let mut candidate = slots.mass_point;
                for j in 0..rank {
                    for i in 0..3 {
                        candidate[i] += vectors[i][j] * coefficient[j];
                    }
                }
                for i in 0..3 {
                    candidate[i] =
                        candidate[i].clamp(cell_min[i], cell_min[i] + spacing[i]);
                }
                let mut error = 0.0f64;
                for slot in 0..12 {
                    let n = slots.normals[slot];
                    let p = slots.points[slot];
                    let residual = n[0] * candidate[0] + n[1] * candidate[1] + n[2] * candidate[2]
                        - (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]);
                    error += residual * residual;
                }
                if rank == 3 || error < best_error {
                    best = candidate;
                    best_error = error;
                }
            }
            out.copy_from_slice(&best);
        });
}
