"""A zero-set backend's ``Solve`` on the GPU, and the classification behind a refresh.

One compute dispatch projects every seed onto its incident surfaces by
minimum-norm Newton, using the dual-number WGSL of :mod:`.wgsl`: each
surface returns its value and gradient, the incident ones form the k×3
system ``J_p``, and the step is ``-J_pᵀ (J_p J_pᵀ)⁻¹ f``.  θ is a storage
buffer, so a parameter change is a buffer write and one dispatch; that is
the refresh path.

A second dispatch, :meth:`Projector.classify`, is the part of ``Discover``
a refresh needs: given points already on the boundary it names the
surfaces each lies on, by the census's ownership rule (a patch where its
value is the model's, a band inside its rim, a rim or fold only with its
band).  It is not discovery — it finds nothing the caller did not place —
but it turns any extraction's points into a topology this solver can hold
fixed, and re-run after a solve it is the certificate's second half.

Float32 throughout: residuals bottom out near 1e-6 in field units, which
is the tolerance to ask for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cadjoint.zeroset.evaluate import census
from cadjoint.zeroset.table import Model
from cadjoint.zeroset.wgsl import emit_dual

__all__ = ["Projector", "Projection"]

_KIND = {"patch": 0, "band": 1, "rim": 2, "fold": 3}

_KERNELS = """
struct Seeds { data: array<vec4<f32>> };            // xyz, w unused
struct Incidence { data: array<vec4<u32>> };       // up to three surface ids, w = count
struct Points { data: array<vec4<f32>> };          // xyz, w = residual
struct Options { iterations: u32, tolerance: f32, count: u32, pad: u32 };
struct Meta { data: array<u32> };                  // per surface: kind | parent << 2
struct ClassifyOptions { count: u32, surfaces: u32, tolerance: f32, ownership: f32 };

@group(0) @binding(0) var<storage, read> seeds: Seeds;
@group(0) @binding(1) var<storage, read> incidence: Incidence;
@group(0) @binding(2) var<storage, read_write> out: Points;
@group(0) @binding(3) var<uniform> options: Options;

@group(2) @binding(0) var<storage, read> census_meta: Meta;
@group(2) @binding(1) var<storage, read_write> found: Incidence;
@group(2) @binding(2) var<uniform> copts: ClassifyOptions;

// One Newton step in the minimum-norm gauge, falling back to fewer
// constraints where the normals are dependent (two coplanar faces meeting,
// a third surface tangent to an edge): the point stays on what it can.
fn step1(f0: vec4<f32>) -> vec3<f32> {
    let n = f0.yzw;
    return -n * f0.x / max(dot(n, n), 1e-30);
}
fn step2(f0: vec4<f32>, f1: vec4<f32>) -> vec3<f32> {
    let n0 = f0.yzw; let n1 = f1.yzw;
    let a = dot(n0, n0); let b = dot(n0, n1); let c = dot(n1, n1);
    let det = a * c - b * b;
    if (det <= 1e-6 * a * c) { return step1(f0); }
    let l0 = (f0.x * c - f1.x * b) / det;
    let l1 = (a * f1.x - b * f0.x) / det;
    return -(n0 * l0 + n1 * l1);
}
fn step3(f0: vec4<f32>, f1: vec4<f32>, f2: vec4<f32>) -> vec3<f32> {
    // J has the gradients as rows; WGSL matrices are column-major
    let J = transpose(mat3x3<f32>(f0.yzw, f1.yzw, f2.yzw));
    let det = determinant(J);
    if (abs(det) <= 1e-6 * length(f0.yzw) * length(f1.yzw) * length(f2.yzw)) { return step2(f0, f1); }
    let f = -vec3<f32>(f0.x, f1.x, f2.x);
    let c0 = mat3x3<f32>(f, J[1], J[2]);
    let c1 = mat3x3<f32>(J[0], f, J[2]);
    let c2 = mat3x3<f32>(J[0], J[1], f);
    return vec3<f32>(determinant(c0), determinant(c1), determinant(c2)) / det;
}

fn residual_at(inc: vec4<u32>, p: vec3<f32>) -> f32 {
    var r = 0.0;
    if (inc.w >= 1u) { r = abs(surface(inc.x, p).x); }
    if (inc.w >= 2u) { r = max(r, abs(surface(inc.y, p).x)); }
    if (inc.w >= 3u) { r = max(r, abs(surface(inc.z, p).x)); }
    return r;
}

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= options.count) { return; }
    var p = seeds.data[i].xyz;
    let inc = incidence.data[i];
    if (inc.w == 0u) { out.data[i] = vec4<f32>(p, 0.0); return; }
    for (var it = 0u; it < options.iterations; it = it + 1u) {
        let f0 = surface(inc.x, p);
        var residual = abs(f0.x);
        var delta = vec3<f32>(0.0);
        if (inc.w == 1u) {
            delta = step1(f0);
        } else if (inc.w == 2u) {
            let f1 = surface(inc.y, p);
            residual = max(residual, abs(f1.x));
            delta = step2(f0, f1);
        } else {
            let f1 = surface(inc.y, p);
            let f2 = surface(inc.z, p);
            residual = max(residual, max(abs(f1.x), abs(f2.x)));
            delta = step3(f0, f1, f2);
        }
        if (residual < options.tolerance * 1e-2) { break; }
        p = p + delta;
    }
    out.data[i] = vec4<f32>(p, residual_at(inc, p));
}

// The surfaces a boundary point lies on: the (up to three) closest by
// |value| within `tolerance`, among those that own the point.
@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= copts.count) { return; }
    let p = seeds.data[i].xyz;
    let whole = field(p).x;
    // four slots, closest first, so a candidate the host will strike out
    // (a rim of a band that owns nothing) cannot crowd out a real one
    var ids = vec4<u32>(0xFFFFFFFFu);
    var errs = vec4<f32>(1e30);
    var band_ok = false;
    var band_id = 0u;
    for (var s = 0u; s < copts.surfaces; s = s + 1u) {
        let kind = census_meta.data[s] & 3u;
        let v = surface(s, p).x;
        let close = abs(v) <= copts.tolerance;
        var ok = false;
        if (kind == 0u) {
            ok = close && abs(v - whole) <= copts.ownership;
        } else if (kind == 1u) {
            band_id = s;
            band_ok = abs(v - whole) <= copts.ownership;
            // a band owns only strictly inside its rim, which follows it in
            // the census; on the rim itself a child's value is the model's
            var inside = true;
            if (s + 1u < copts.surfaces && (census_meta.data[s + 1u] & 3u) == 2u) {
                inside = surface(s + 1u, p).x > copts.ownership;
            }
            ok = close && band_ok && inside;
        } else {
            ok = close && band_ok && (census_meta.data[s] >> 2u) == band_id;
        }
        if (!ok) { continue; }
        var cid = s;
        var cerr = abs(v);
        if (cerr < errs.x) { let t = ids.x; ids.x = cid; cid = t; let e = errs.x; errs.x = cerr; cerr = e; }
        if (cerr < errs.y) { let t = ids.y; ids.y = cid; cid = t; let e = errs.y; errs.y = cerr; cerr = e; }
        if (cerr < errs.z) { let t = ids.z; ids.z = cid; cid = t; let e = errs.z; errs.z = cerr; cerr = e; }
        if (cerr < errs.w) { ids.w = cid; errs.w = cerr; }
    }
    found.data[i] = ids;
}
"""


@dataclass(frozen=True)
class Projection:
    points: np.ndarray  #: (N, 3)
    residual: np.ndarray  #: (N,) max |f_i| over the point's surfaces


_shared_device: Any = None


def _device() -> Any:
    """One wgpu device for the process: pipelines are per table, the device is not."""
    global _shared_device  # noqa: PLW0603 - a process-wide handle, created once
    if _shared_device is None:
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        _shared_device = adapter.request_device_sync()
    return _shared_device


class Projector:
    """A compiled ``Solve`` (and classification) for one table; θ and seeds vary per call."""

    def __init__(self, model: Model) -> None:
        import wgpu

        self.model = model
        self.surfaces = census(model)
        self.wgsl = emit_dual(model, theta_binding=(1, 0)) + _KERNELS
        self.device = d = _device()
        module = d.create_shader_module(code=self.wgsl)
        compute = wgpu.ShaderStage.COMPUTE
        ro, rw, uni = (
            wgpu.BufferBindingType.read_only_storage,
            wgpu.BufferBindingType.storage,
            wgpu.BufferBindingType.uniform,
        )

        def layout(*types):
            return d.create_bind_group_layout(
                entries=[
                    {"binding": i, "visibility": compute, "buffer": {"type": t}}
                    for i, t in enumerate(types)
                ]
            )

        self._layouts = [layout(ro, ro, rw, uni), layout(ro), layout(ro, rw, uni)]
        pipeline_layout = d.create_pipeline_layout(bind_group_layouts=self._layouts)
        self._project = d.create_compute_pipeline(
            layout=pipeline_layout, compute={"module": module, "entry_point": "project"}
        )
        self._classify = d.create_compute_pipeline(
            layout=pipeline_layout, compute={"module": module, "entry_point": "classify"}
        )
        # per surface: kind, and for a rim or fold the band it belongs to
        meta, band = [], 0
        for k, (kind, _warps, _what) in enumerate(self.surfaces):
            if kind == "band":
                band = k
            meta.append(_KIND[kind] | (band << 2))
        self._meta = np.asarray(meta or [0], np.uint32)

    # -- buffers -------------------------------------------------------------

    def _storage(self, data: np.ndarray, *, writable: bool = False) -> Any:
        import wgpu

        usage = wgpu.BufferUsage.STORAGE | (wgpu.BufferUsage.COPY_SRC if writable else 0)
        if writable:
            return self.device.create_buffer(size=int(data.nbytes), usage=usage)
        return self.device.create_buffer_with_data(data=data.tobytes(), usage=usage)

    def _uniform(self, words: list[float | int], kinds: str) -> Any:
        """A 16-byte uniform from mixed u32 (``u``) and f32 (``f``) words."""
        import wgpu

        raw = b"".join(
            np.asarray(w, np.uint32 if k == "u" else np.float32).tobytes()
            for w, k in zip(words, kinds)
        )
        return self.device.create_buffer_with_data(data=raw, usage=wgpu.BufferUsage.UNIFORM)

    def _group(self, index: int, *buffers: Any) -> Any:
        return self.device.create_bind_group(
            layout=self._layouts[index],
            entries=[
                {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                for i, b in enumerate(buffers)
            ],
        )

    def _theta_group(self, theta: Any) -> Any:
        data = np.asarray(theta, np.float32).ravel()
        if data.size != len(self.model.theta):
            raise ValueError(
                f"theta has {data.size} entries; the table has {len(self.model.theta)}"
            )
        if data.size == 0:
            data = np.zeros(1, np.float32)
        return self._group(1, self._storage(data))

    @staticmethod
    def _seed_data(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, np.float32).reshape(-1, 3)
        data = np.zeros((len(pts), 4), np.float32)
        data[:, :3] = pts
        return data

    def _dispatch(self, pipeline: Any, groups: list[tuple[int, Any]], n: int, out: Any) -> bytes:
        encoder = self.device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(pipeline)
        for index, group in groups:
            cpass.set_bind_group(index, group)
        cpass.dispatch_workgroups((n + 63) // 64)
        cpass.end()
        self.device.queue.submit([encoder.finish()])
        return self.device.queue.read_buffer(out)

    # -- the two dispatches --------------------------------------------------

    def solve(
        self,
        theta: Any,
        seeds: np.ndarray,
        incidence: list[list[int]],
        *,
        tolerance: float = 1e-6,
        iterations: int = 32,
    ) -> Projection:
        """Project every seed onto its incident surfaces.

        Args:
            theta: The design, length P.
            seeds: (N, 3) starting points, inside the solver's basin.
            incidence: For each seed, up to three surface ids in census
                order; an empty list leaves the point where it is.
            tolerance: Field units; the iteration stops early below a hundredth of it.
            iterations: Newton steps at most.
        """
        seed_data = self._seed_data(seeds)
        n = len(seed_data)
        inc_data = np.zeros((n, 4), np.uint32)
        for a, row in enumerate(incidence):
            inc_data[a, : len(row)] = row[:3]
            inc_data[a, 3] = min(len(row), 3)
        # the placeholders group 2 needs bound even though `project` never reads it
        out = self._storage(np.zeros((n, 4), np.float32), writable=True)
        found = self._storage(np.zeros((1, 4), np.uint32), writable=True)
        groups = [
            (
                0,
                self._group(
                    0,
                    self._storage(seed_data),
                    self._storage(inc_data),
                    out,
                    self._uniform([iterations, tolerance, n, 0], "ufuu"),
                ),
            ),
            (1, self._theta_group(theta)),
            (
                2,
                self._group(
                    2, self._storage(self._meta), found, self._uniform([0, 0, 0, 0], "uuff")
                ),
            ),
        ]
        raw = np.frombuffer(self._dispatch(self._project, groups, n, out), np.float32).reshape(n, 4)
        return Projection(
            points=raw[:, :3].astype(np.float64), residual=raw[:, 3].astype(np.float64)
        )

    def classify(
        self,
        theta: Any,
        points: np.ndarray,
        *,
        tolerance: float,
        ownership: float = 1e-5,
    ) -> list[list[int]]:
        """Which surfaces own each boundary point: up to four ids, closest first.

        Four rather than the three a solve takes, so a caller that strikes
        candidates out by a rule of its own (:mod:`.refresh`) still has
        three left at a corner.

        Args:
            theta: The design.
            points: (N, 3) points on (or within ``tolerance`` of) the boundary.
            tolerance: How far off a surface a point may be and still count.
            ownership: How far a patch's value may differ from the model's
                at the point and still own it (float32 noise, not geometry).
        """
        seed_data = self._seed_data(points)
        n = len(seed_data)
        out = self._storage(np.zeros((n, 4), np.float32), writable=True)
        found = self._storage(np.zeros((n, 4), np.uint32), writable=True)
        groups = [
            (
                0,
                self._group(
                    0,
                    self._storage(seed_data),
                    self._storage(np.zeros((1, 4), np.uint32)),
                    out,
                    self._uniform([0, 0, 0, 0], "ufuu"),
                ),
            ),
            (1, self._theta_group(theta)),
            (
                2,
                self._group(
                    2,
                    self._storage(self._meta),
                    found,
                    self._uniform([n, len(self.surfaces), tolerance, ownership], "uuff"),
                ),
            ),
        ]
        raw = np.frombuffer(self._dispatch(self._classify, groups, n, found), np.uint32).reshape(
            n, 4
        )
        return [row[row != 0xFFFFFFFF].tolist() for row in raw]
