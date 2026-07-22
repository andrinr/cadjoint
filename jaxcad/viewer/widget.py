"""Interactive WebGPU SDF viewer for Jupyter notebooks.

Usage::

    from jaxcad.viewer import SDFViewer
    from jaxcad.sdf.primitives.sphere import Sphere
    from jaxcad.sdf.boolean.union import Union
    from jaxcad.sdf.transforms.affine.translate import Translate
    import jax.numpy as jnp

    scene = Union(
        Sphere(1.0),
        Translate(Sphere(0.5), offset=jnp.array([1.5, 0., 0.])),
    )
    SDFViewer(scene)

Controls
--------
- Left-drag   : orbit camera
- Scroll      : zoom
- Right-drag  : pan look-at point
- Double-click: reset camera
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Callable

from ..backends.wgsl.codegen import compile_sdf_to_wgsl

try:
    import anywidget
    import traitlets
except ImportError as exc:
    raise ImportError(
        "SDFViewer requires the optional viewer dependencies. "
        "Install them with: pip install 'jaxcad[viewer]'"
    ) from exc

# ── JavaScript (ES module) ────────────────────────────────────────────────────

_ESM = r"""
// ── shader builder ────────────────────────────────────────────────────────────
function buildShader(sdfCode) {
  return `
${sdfCode}

struct Uniforms {
  resolution   : vec4<f32>,
  camera_pos   : vec4<f32>,
  camera_target: vec4<f32>,
  light_dir    : vec4<f32>,
  bg_color     : vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

fn sdf_normal(p: vec3<f32>) -> vec3<f32> {
  let e = 0.001;
  return normalize(vec3<f32>(
    sdf(p + vec3<f32>( e, 0.0, 0.0)) - sdf(p + vec3<f32>(-e, 0.0, 0.0)),
    sdf(p + vec3<f32>(0.0,  e, 0.0)) - sdf(p + vec3<f32>(0.0, -e, 0.0)),
    sdf(p + vec3<f32>(0.0, 0.0,  e)) - sdf(p + vec3<f32>(0.0, 0.0, -e)),
  ));
}

fn trace(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
  var t = 0.01;
  for (var i = 0; i < 96; i++) {
    let d = sdf(ro + rd * t);
    if (d < 0.001) { return t; }
    if (t > 100.0) { return -1.0; }
    t += d;
  }
  return -1.0;
}

fn soft_shadow(ro: vec3<f32>, rd: vec3<f32>, k: f32) -> f32 {
  var res = 1.0; var t = 0.02;
  for (var i = 0; i < 24; i++) {
    let h = sdf(ro + rd * t);
    if (h < 0.001) { return 0.0; }
    res = min(res, k * h / t);
    t += h;
    if (t > 20.0) { break; }
  }
  return clamp(res, 0.0, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  let pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  return vec4<f32>(pos[vid], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let res = u.resolution.xy;
  let uv  = (frag.xy / res - 0.5) * vec2<f32>(res.x / res.y, -1.0);

  let cam = u.camera_pos.xyz;
  let tgt = u.camera_target.xyz;
  let fwd   = normalize(tgt - cam);
  let right = normalize(cross(fwd, vec3<f32>(0.0, 1.0, 0.0)));
  let up    = cross(right, fwd);

  let ro = cam;
  let rd = normalize(fwd + 1.5 * (uv.x * right + uv.y * up));

  let t  = trace(ro, rd);
  var col = u.bg_color.xyz;

  if (t >= 0.0) {
    let pos  = ro + rd * t;
    let nor  = sdf_normal(pos);
    let ldir = normalize(u.light_dir.xyz);
    let diff = max(dot(nor, ldir), 0.0);
    let sha  = soft_shadow(pos + 0.02 * nor, ldir, 8.0);
    let spec = pow(max(dot(reflect(-ldir, nor), -rd), 0.0), 32.0);
    col = vec3<f32>(0.027) + vec3<f32>(0.85) * diff * sha + vec3<f32>(0.4) * spec;
  }

  col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
  return vec4<f32>(col, 1.0);
}
`;
}

// ── widget entry point (must be synchronous) ──────────────────────────────────
export function render({ model, el }) {
  const controller = new AbortController();
  const { signal } = controller;
  const h = model.get('height');

  // ── DOM setup ─────────────────────────────────────────────────────────────
  const container = document.createElement('div');
  container.style.cssText =
    `position:relative;width:100%;height:${h}px;` +
    `background:#111;border-radius:4px;overflow:hidden;`;

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `display:block;width:100%;height:100%;cursor:grab;`;
  // pixel dimensions set by ResizeObserver below
  container.appendChild(canvas);

  const status = document.createElement('div');
  status.style.cssText =
    'position:absolute;top:6px;left:8px;color:#fff;' +
    'font:11px monospace;text-shadow:0 0 4px #000;pointer-events:none;' +
    'white-space:pre;';
  status.textContent = 'initialising WebGPU…';
  container.appendChild(status);

  el.appendChild(container);

  // ── async WebGPU initialisation ───────────────────────────────────────────
  async function init() {
    if (signal.aborted) return;
    if (!navigator.gpu) {
      status.style.color = '#f88';
      status.textContent = '⚠ WebGPU not supported\nUse Chrome 113+ or Edge 113+';
      return;
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (signal.aborted) return;
    if (!adapter) {
      status.style.color = '#f88';
      status.textContent = '⚠ No WebGPU adapter found';
      return;
    }
    const device  = await adapter.requestDevice();
    if (signal.aborted) {
      device.destroy();
      return;
    }
    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('Could not create a WebGPU canvas context');
    const format  = navigator.gpu.getPreferredCanvasFormat();

    function configureContext() {
      context.configure({ device, format, alphaMode: 'opaque' });
    }
    configureContext();

    // Uniform buffer: 5 × vec4<f32> = 80 bytes
    const uniformBuf = device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let pipeline  = null;
    let bindGroup = null;
    let pipelineRevision = 0;
    let shaderError = false;

    async function buildPipeline() {
      const revision = ++pipelineRevision;
      const code = buildShader(model.get('wgsl_sdf'));
      try {
        const module = device.createShaderModule({ code });
        const info   = await module.getCompilationInfo();
        const errors = info.messages.filter(m => m.type === 'error');
        if (errors.length) {
          if (revision !== pipelineRevision || signal.aborted) return false;
          shaderError = true;
          status.style.color = '#f88';
          status.textContent = '⚠ Shader error:\n' + errors[0].message;
          return false;
        }
        const nextPipeline = await device.createRenderPipelineAsync({
          layout: 'auto',
          vertex:   { module, entryPoint: 'vs_main' },
          fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
          primitive: { topology: 'triangle-list' },
        });
        if (revision !== pipelineRevision || signal.aborted) return false;
        const nextBindGroup = device.createBindGroup({
          layout: nextPipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: uniformBuf } }],
        });
        pipeline = nextPipeline;
        bindGroup = nextBindGroup;
        shaderError = false;
        status.style.color = '#aaa';
        return true;
      } catch (e) {
        if (revision !== pipelineRevision || signal.aborted) return false;
        shaderError = true;
        status.style.color = '#f88';
        status.textContent = '⚠ ' + e.message;
        return false;
      }
    }

    // ── orbit camera ─────────────────────────────────────────────────────────
    let az = 0.6, elev = 0.4, dist = 5.0;
    let cx = 0.0, cy = 0.0, cz = 0.0;

    function camPos() {
      return [
        cx + dist * Math.cos(elev) * Math.sin(az),
        cy + dist * Math.sin(elev),
        cz + dist * Math.cos(elev) * Math.cos(az),
      ];
    }

    function pushUniforms() {
      if (signal.aborted) return;
      const [px, py, pz] = camPos();
      const ld = model.get('light_dir');
      const bg = model.get('bg_color');
      device.queue.writeBuffer(uniformBuf, 0, new Float32Array([
        canvas.width, canvas.height, 0, 0,
        px, py, pz, 0,
        cx, cy, cz, 0,
        ld[0], ld[1], ld[2], 0,
        bg[0], bg[1], bg[2], 0,
      ]));
    }

    // ── render loop ───────────────────────────────────────────────────────────
    let pending = false;
    function schedule() {
      if (pending || signal.aborted) return;
      pending = true;
      requestAnimationFrame(frame);
    }

    function frame() {
      pending = false;
      if (!pipeline || signal.aborted) return;
      pushUniforms();
      if (!shaderError) {
        status.textContent =
          `az ${(az * 180 / Math.PI).toFixed(1)}°  ` +
          `el ${(elev * 180 / Math.PI).toFixed(1)}°  ` +
          `r ${dist.toFixed(2)}`;
      }

      const enc  = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear', clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1 },
          storeOp: 'store',
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
      device.queue.submit([enc.finish()]);
    }

    // ── mouse interaction ─────────────────────────────────────────────────────
    let dragging = false, rightBtn = false, lx = 0, ly = 0;

    canvas.addEventListener('mousedown', e => {
      dragging = true; rightBtn = e.button === 2;
      lx = e.clientX; ly = e.clientY;
      canvas.style.cursor = 'grabbing';
      e.preventDefault();
    }, { signal });
    canvas.addEventListener('contextmenu', e => e.preventDefault(), { signal });

    window.addEventListener('mousemove', e => {
      if (!dragging) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      if (rightBtn) {
        const pan = dx * dist * 0.002;
        cx -= pan * Math.cos(az);
        cz += pan * Math.sin(az);
        cy -= dy * dist * 0.002;
      } else {
        az   -= dx * 0.008;
        elev  = Math.max(-1.45, Math.min(1.45, elev + dy * 0.008));
      }
      schedule();
    }, { signal });
    window.addEventListener('mouseup', () => {
      dragging = false; canvas.style.cursor = 'grab';
    }, { signal });
    canvas.addEventListener('wheel', e => {
      dist = Math.max(0.1, Math.min(200, dist * (1 + e.deltaY * 0.001)));
      schedule(); e.preventDefault();
    }, { passive: false, signal });
    canvas.addEventListener('dblclick', () => {
      az = 0.6; elev = 0.4; dist = 5.0; cx = cy = cz = 0; schedule();
    }, { signal });

    // ── model observers ───────────────────────────────────────────────────────
    const onShaderChange = () => {
      status.textContent = 'recompiling…';
      buildPipeline().then(ok => { if (ok) schedule(); });
    };
    const onUniformChange = () => schedule();
    model.on('change:wgsl_sdf', onShaderChange);
    model.on('change:light_dir change:bg_color', onUniformChange);

    // ── responsive resize ─────────────────────────────────────────────────────
    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1;
      const w   = Math.max(1, Math.round(canvas.clientWidth  * dpr));
      const h   = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width  = w;
        canvas.height = h;
        configureContext();
        schedule();
      }
    });
    ro.observe(canvas);

    signal.addEventListener('abort', () => {
      ro.disconnect();
      model.off('change:wgsl_sdf', onShaderChange);
      model.off('change:light_dir change:bg_color', onUniformChange);
      device.destroy();
    }, { once: true });

    device.lost.then(info => {
      if (signal.aborted) return;
      status.style.color = '#f88';
      status.textContent = '⚠ WebGPU device lost: ' + info.message;
    });

    if (await buildPipeline()) schedule();
  }

  init().catch(e => {
    if (signal.aborted) return;
    status.style.color = '#f88';
    status.textContent = '⚠ ' + e.message;
  });

  return () => controller.abort();
}
"""

# ── Python widget ─────────────────────────────────────────────────────────────


class SDFViewer(anywidget.AnyWidget):
    """Interactive WebGPU SDF viewer for Jupyter notebooks.

    Compiles the SDF to WGSL via ``jax.export`` and renders it live in the
    browser using WebGPU.  Supports orbit camera (left-drag), zoom (scroll),
    pan (right-drag), and reset (double-click).

    Args:
        fn: JAX-traceable SDF callable ``(p: f32[3]) -> f32[]``.
        height: Canvas height in CSS pixels. The width fills its container.
        light_dir: World-space light direction, default ``[1, 2, 3]``.
        bg_color: Background RGB in [0, 1], default ``[0.05, 0.05, 0.1]``.
    """

    _esm = _ESM

    wgsl_sdf = traitlets.Unicode(default_value="").tag(sync=True)
    height = traitlets.Int(default_value=400, min=1).tag(sync=True)
    light_dir = traitlets.List(
        trait=traitlets.Float(), default_value=[1.0, 2.0, 3.0], minlen=3, maxlen=3
    ).tag(sync=True)
    bg_color = traitlets.List(
        trait=traitlets.Float(), default_value=[0.05, 0.05, 0.10], minlen=3, maxlen=3
    ).tag(sync=True)

    @traitlets.validate("light_dir")
    def _validate_light_dir(self, proposal):
        values = proposal["value"]
        if not all(math.isfinite(value) for value in values) or not any(values):
            raise traitlets.TraitError("light_dir must be a finite, non-zero 3D vector")
        return values

    @traitlets.validate("bg_color")
    def _validate_bg_color(self, proposal):
        values = proposal["value"]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise traitlets.TraitError("bg_color values must be finite and between 0 and 1")
        return values

    def __init__(
        self,
        fn: Callable,
        height: int = 400,
        light_dir: Sequence[float] = (1.0, 2.0, 3.0),
        bg_color: Sequence[float] = (0.05, 0.05, 0.10),
        **kwargs,
    ) -> None:
        super().__init__(
            wgsl_sdf=compile_sdf_to_wgsl(fn),
            height=height,
            light_dir=list(light_dir),
            bg_color=list(bg_color),
            **kwargs,
        )

    def update_scene(self, fn: Callable) -> None:
        """Recompile and hot-reload the scene without recreating the widget."""
        self.wgsl_sdf = compile_sdf_to_wgsl(fn)
