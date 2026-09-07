"""A zero-set backend's ``Solve`` on the GPU.

One compute dispatch projects every seed onto its incident surfaces by
minimum-norm Newton, using the dual-number WGSL of :mod:`.wgsl`: each
surface returns its value and gradient, the incident ones form the k×3
system ``J_p``, and the step is ``-J_pᵀ (J_p J_pᵀ)⁻¹ f``.  θ is a storage
buffer, so a parameter change is a buffer write and one dispatch; that is
the refresh path.  Discovery is not here: it is the CPU's, or a real
backend's.

Float32 throughout: residuals bottom out near 1e-6 in field units, which
is the tolerance to ask for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cadjoint.zeroset.table import Model
from cadjoint.zeroset.wgsl import emit_dual

__all__ = ["Projector", "Projection"]

_KERNEL = """
struct Seeds { data: array<vec4<f32>> };            // xyz, w unused
struct Incidence { data: array<vec4<u32>> };       // up to three surface ids, w = count
struct Points { data: array<vec4<f32>> };          // xyz, w = residual
struct Options { iterations: u32, tolerance: f32, count: u32, pad: u32 };

@group(0) @binding(0) var<storage, read> seeds: Seeds;
@group(0) @binding(1) var<storage, read> incidence: Incidence;
@group(0) @binding(2) var<storage, read_write> out: Points;
@group(0) @binding(3) var<uniform> options: Options;

fn solve2(a: mat2x2<f32>, b: vec2<f32>) -> vec2<f32> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    return vec2<f32>(b.x * a[1][1] - b.y * a[0][1], a[0][0] * b.y - a[1][0] * b.x) / det;
}

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= options.count) { return; }
    var p = seeds.data[i].xyz;
    let inc = incidence.data[i];
    let k = inc.w;
    var residual = 0.0;
    for (var it = 0u; it < options.iterations; it = it + 1u) {
        let f0 = surface(inc.x, p);
        var step = vec3<f32>(0.0);
        residual = abs(f0.x);
        if (k == 1u) {
            let n = f0.yzw;
            step = -n * f0.x / max(dot(n, n), 1e-30);
        } else if (k == 2u) {
            let f1 = surface(inc.y, p);
            residual = max(residual, abs(f1.x));
            let n0 = f0.yzw; let n1 = f1.yzw;
            let g = mat2x2<f32>(vec2<f32>(dot(n0, n0), dot(n0, n1)), vec2<f32>(dot(n1, n0), dot(n1, n1)));
            let lambda = solve2(g, vec2<f32>(f0.x, f1.x));
            step = -(n0 * lambda.x + n1 * lambda.y);
        } else {
            let f1 = surface(inc.y, p);
            let f2 = surface(inc.z, p);
            residual = max(residual, max(abs(f1.x), abs(f2.x)));
            let J = mat3x3<f32>(f0.yzw, f1.yzw, f2.yzw);   // columns are the gradients
            // J^T is the k×3 system; a vertex is a full 3×3 solve of J^T step = -f
            let jt = transpose(J);
            let det = determinant(jt);
            let f = vec3<f32>(f0.x, f1.x, f2.x);
            // Cramer's rule
            let c0 = mat3x3<f32>(-f, jt[1], jt[2]);
            let c1 = mat3x3<f32>(jt[0], -f, jt[2]);
            let c2 = mat3x3<f32>(jt[0], jt[1], -f);
            step = vec3<f32>(determinant(c0), determinant(c1), determinant(c2)) / det;
        }
        p = p + step;
        if (residual < options.tolerance * 1e-2) { break; }
    }
    // the residual at the returned point
    let g0 = surface(inc.x, p);
    residual = abs(g0.x);
    if (k >= 2u) { residual = max(residual, abs(surface(inc.y, p).x)); }
    if (k >= 3u) { residual = max(residual, abs(surface(inc.z, p).x)); }
    out.data[i] = vec4<f32>(p, residual);
}
"""


@dataclass(frozen=True)
class Projection:
    points: np.ndarray  #: (N, 3)
    residual: np.ndarray  #: (N,) max |f_i| over the point's surfaces


class Projector:
    """A compiled ``Solve`` for one table; θ and seeds vary per call."""

    def __init__(self, model: Model) -> None:
        import wgpu

        self.model = model
        self.wgsl = emit_dual(model, theta_binding=(1, 0)) + _KERNEL
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = adapter.request_device_sync()
        module = self.device.create_shader_module(code=self.wgsl)
        storage = wgpu.BufferBindingType.read_only_storage
        self._layout0 = self.device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": storage}},
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]
        )
        self._layout1 = self.device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": storage}}
            ]
        )
        self.pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._layout0, self._layout1]
            ),
            compute={"module": module, "entry_point": "project"},
        )

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
            incidence: For each seed, one to three surface ids in census order.
            tolerance: Field units; the iteration stops early below a hundredth of it.
            iterations: Newton steps at most.
        """
        import wgpu

        n = len(seeds)
        seed_data = np.zeros((n, 4), np.float32)
        seed_data[:, :3] = seeds
        inc_data = np.zeros((n, 4), np.uint32)
        for a, row in enumerate(incidence):
            inc_data[a, : len(row)] = row
            inc_data[a, 3] = len(row)
        theta_data = np.asarray(theta, np.float32)
        if theta_data.size == 0:
            theta_data = np.zeros(1, np.float32)
        opts = np.array([iterations, 0, n, 0], np.uint32)
        opts_f = np.array([0, tolerance, 0, 0], np.float32)
        options = np.where(
            np.array([True, False, True, True]), opts.view(np.float32), opts_f
        ).tobytes()

        usage = wgpu.BufferUsage
        d = self.device
        b_seeds = d.create_buffer_with_data(data=seed_data.tobytes(), usage=usage.STORAGE)
        b_inc = d.create_buffer_with_data(data=inc_data.tobytes(), usage=usage.STORAGE)
        b_out = d.create_buffer(size=n * 16, usage=usage.STORAGE | usage.COPY_SRC)
        b_opts = d.create_buffer_with_data(data=options, usage=usage.UNIFORM)
        b_theta = d.create_buffer_with_data(data=theta_data.tobytes(), usage=usage.STORAGE)
        entry = lambda i, b: {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}  # noqa: E731
        g0 = d.create_bind_group(
            layout=self._layout0,
            entries=[entry(0, b_seeds), entry(1, b_inc), entry(2, b_out), entry(3, b_opts)],
        )
        g1 = d.create_bind_group(layout=self._layout1, entries=[entry(0, b_theta)])
        encoder = d.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, g0)
        cpass.set_bind_group(1, g1)
        cpass.dispatch_workgroups((n + 63) // 64)
        cpass.end()
        d.queue.submit([encoder.finish()])
        out = np.frombuffer(d.queue.read_buffer(b_out), np.float32).reshape(n, 4)
        return Projection(
            points=out[:, :3].astype(np.float64), residual=out[:, 3].astype(np.float64)
        )
