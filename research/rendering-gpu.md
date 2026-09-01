# GPU Rendering: GLSL / NVIDIA Warp / Slang

> Archived exploration. The active renderer roadmap is forward-only and
> prioritizes runtime, fidelity, and clean abstractions. The inverse-rendering
> and Slang sections below are retained as historical research, not planned work.

## Forward-rendering references

- [Soft shadows in raymarched SDFs](https://iquilezles.org/articles/rmshadows/)
  derives the `distance / travel` penumbra estimate and the two-sample closest
  approach used by the active forward renderer.
- [Binary search for SDF ray tracing](https://iquilezles.org/articles/binarysearchsdf/)
  is useful as a traversal reference, but its own evaluation found substantially
  more SDF evaluations than conventional sphere tracing. Keep early-exit sphere
  tracing as the default unless a scene-specific benchmark shows otherwise.
- [Smooth minimum](https://iquilezles.org/articles/smin/) documents normalized
  blend kernels, bounds inflation, and matching material blend factors. Use it
  as the reference for a focused CSG/material-blending pass.

## Current state

`cadjoint/render.py` has two paths:

| Method | How | Bottleneck |
|---|---|---|
| `render_raymarched` | `jax.lax.fori_loop` + `vmap`, 200×200 | All `max_steps` iterations always run (no early exit); seconds per frame on CPU |
| `render_marching_cubes` | JAX vmap grid eval → `skimage` CPU marching cubes | No interactivity; CPU-bound |

`functionalize.py` compiles the SDF tree to a pure function `(free_params, fixed_params) -> (point -> distance)` — the key abstraction for all three approaches.

---

## Approach 1: JAX tree → GLSL (most practical)

### What it is

Walk jaxCAD's Python SDF tree and emit a GLSL `float sceneSDF(vec3 p)` function. Inject it into a standard raymarching fragment shader, load with **moderngl**. No JAX compilation pipeline needed.

### Why it's the right first step

- Zero heavy new dependencies (`moderngl` is a thin OpenGL wrapper).
- Interactive 60fps at 512×512 on any GPU.
- The SDF tree already has the structure needed: each node has a known closed-form GLSL equivalent.
- Free parameters → GLSL uniforms: parameter changes update the uniform, no shader recompilation.

### Key implementation: `to_glsl()` visitor

Add a `to_glsl(p: str) -> str` method to each SDF class, returning a GLSL expression string for the SDF value at point named `p`.

```python
# cadjoint/sdf/primitives/sphere.py
def to_glsl(self, p: str = "p") -> str:
    r = float(self.params['radius'].value)
    return f"(length({p}) - {r:.6f})"

# cadjoint/sdf/boolean/union.py
def to_glsl(self, p: str = "p") -> str:
    children = [c.to_glsl(p) for c in self.sdfs]
    k = float(self.params.get('smoothness', 0.0))
    result = children[0]
    for expr in children[1:]:
        if k > 0:
            result = f"smin({result}, {expr}, {k:.6f})"
        else:
            result = f"min({result}, {expr})"
    return result

# cadjoint/sdf/transforms/affine/translate.py
def to_glsl(self, p: str = "p") -> str:
    o = self.params['offset'].xyz
    inner_p = f"({p} - vec3({o[0]:.6f},{o[1]:.6f},{o[2]:.6f}))"
    return self.sdf.to_glsl(inner_p)
```

Shader template:
```glsl
float smin(float a, float b, float k) {
    float h = max(k - abs(a-b), 0.0) / k;
    return min(a,b) - h*h*h*k*(1.0/6.0);
}

float sceneSDF(vec3 p) {
    return {SCENE_EXPR};
}
```

### moderngl integration (~30 lines)

```python
import moderngl
import numpy as np

ctx = moderngl.create_standalone_context()

prog = ctx.program(
    vertex_shader=FULLSCREEN_QUAD_VERT,
    fragment_shader=RAYMARCH_FRAG.format(scene_sdf=scene.to_glsl())
)

# Update free parameter as uniform — no recompilation needed
prog['u_sphere_radius'].value = float(radius_param.value)

fbo = ctx.simple_framebuffer((512, 512))
fbo.use()
# ... draw fullscreen quad, read pixels back as numpy array
```

### Handling free parameters as uniforms

Distinguish free vs. fixed at emit time:
```python
def to_glsl(self, p):
    if self.params['radius'].free:
        return f"(length({p}) - u_{self.params['radius'].name})"
    else:
        r = float(self.params['radius'].value)
        return f"(length({p}) - {r:.6f})"
```

Free parameters become `uniform float u_<name>;` declarations in the shader header. Changes to parameter values call `prog['u_radius'].value = ...` — no recompilation.

### Alternative: wgpu-py (WebGPU)

**wgpu-py** (https://github.com/pygfx/wgpu-py) + **pygfx** use WGSL instead of GLSL. Compile GLSL → SPIR-V via `glslang`, then load. Cross-platform including Metal on macOS. More setup but no vendor lock-in.

### Limitations

- Gradients through GLSL shaders do not flow back to JAX parameter space. Use for visualization only (not inverse rendering).
- Deep SDF trees can produce GLSL that exceeds driver inlining limits (~1000 ops). Use function-per-node architecture to mitigate.
- Smooth min requires a `smin()` helper in the shader preamble.

---

## Approach 2: NVIDIA Warp kernel raymarcher

### What it is

**NVIDIA Warp** (https://github.com/NVIDIA/warp, `pip install warp-lang`) is a Python DSL that compiles to CUDA (or C++). Unlike JAX's `fori_loop`, Warp kernels support `break` for early ray termination — the single biggest perf difference.

### Why Warp is faster than JAX `fori_loop`

| | JAX `fori_loop` | Warp `@wp.kernel` |
|---|---|---|
| Early exit | No — all `max_steps` run always | Yes — `break` on convergence |
| Memory layout | Batched: (H×W, 3) per step | One thread per pixel, no batching |
| Compilation | XLA trace | CUDA C++ JIT (~0.5–2s first run) |
| GPU req. | Any (CPU/GPU/TPU) | NVIDIA only |

For 512×512 with typical convergence in 10–30 of 64 max steps, Warp is ~5–10× faster than equivalent JAX code.

### JAX ↔ Warp interop (zero-copy via DLPack)

```python
import warp as wp
from jax.dlpack import to_dlpack
from jax.dlpack import from_dlpack as jax_from_dlpack

# JAX parameter array → Warp (zero-copy on GPU)
params_wp = wp.from_dlpack(to_dlpack(params_jax))

# Warp output → JAX
output_jax = jax_from_dlpack(output_wp.__dlpack__())
```

### Warp SDF kernel structure

```python
@wp.func
def sdf_sphere(p: wp.vec3, radius: float) -> float:
    return wp.length(p) - radius

@wp.func
def smooth_min(a: float, b: float, k: float) -> float:
    h = wp.max(k - wp.abs(a - b), 0.0) / k
    return wp.min(a, b) - h*h*h*k*(1.0/6.0)

@wp.func
def scene_sdf(p: wp.vec3, radius: float, translate: wp.vec3) -> float:
    d_sphere = sdf_sphere(p - translate, radius)
    return d_sphere

@wp.kernel
def raymarch_kernel(
    pixels: wp.array(dtype=wp.float32, ndim=2),
    camera_pos: wp.vec3,
    fwd: wp.vec3, right: wp.vec3, up: wp.vec3,
    fov: float,
    radius: float, translate: wp.vec3,   # ← JAX parameters passed in
    max_steps: int, eps: float,
):
    i, j = wp.tid()
    H, W = pixels.shape[0], pixels.shape[1]
    u = (float(j)/float(W) - 0.5) * fov
    v = (float(i)/float(H) - 0.5) * fov
    ray_dir = fwd + u*right + v*up
    ray_dir = ray_dir / wp.length(ray_dir)

    t = 0.0
    hit = False
    for _ in range(max_steps):            # ← real loop with break
        p = camera_pos + t * ray_dir
        d = scene_sdf(p, radius, translate)
        if wp.abs(d) < eps:
            hit = True
            break
        t += wp.max(d, eps*0.5) * 0.9

    if hit:
        # Phong shading (compute normal via finite differences)
        ...
        pixels[i, j] = 0.2 + 0.8 * diffuse
    else:
        pixels[i, j] = 0.0
```

**Code generation**: since Warp `@wp.func` does not support Python dispatch, a code generator is needed (analogous to the GLSL emitter) to convert the jaxCAD SDF tree to a flat Warp scene function at Python import time.

### Making Warp rendering differentiable from JAX

Use `custom_vjp` to bridge JAX ↔ Warp:

```python
from jax import custom_vjp

@custom_vjp
def warp_render(params: jax.Array) -> jax.Array:
    # Convert params to Warp, run kernel, return pixel array via DLPack
    ...

def warp_render_fwd(params):
    pixels = warp_render(params)
    return pixels, params  # residuals for backward

def warp_render_bwd(params, g):
    # Run wp.Tape backward with pixel gradient g
    tape = wp.Tape()
    with tape.record():
        pixels_wp = run_warp_kernel(params)
    tape.backward(grads={pixels_wp: wp_from_dlpack(to_dlpack(g))})
    return (jax_from_dlpack(tape.gradients[params_wp].__dlpack__()),)

warp_render.defvjp(warp_render_fwd, warp_render_bwd)

# Now jax.grad works through Warp:
grad = jax.grad(lambda p: jnp.sum(warp_render(p)**2))(params)
```

### Key libraries

- Warp: https://github.com/NVIDIA/warp (`pip install warp-lang`)
- Warp autodiff: https://nvidia.github.io/warp/autodiff.html
- Isaac Lab (JAX+Warp example): https://github.com/isaac-sim/IsaacLab

### Limitations

- **NVIDIA GPU required** — no CPU fallback for raymarching.
- Warp `@wp.func` is statically typed — the SDF scene must be a flat function, not a Python class tree. Need a code generator.
- `wp.Tape` gradient support is incomplete for some ops (dynamic indexing).
- Warp kernels compile at first launch (~0.5–2s).

---

## Approach 3: Differentiable rendering with NVIDIA Slang

### What it is

**Slang** (https://github.com/shader-slang/slang, https://shader-slang.com/) is a shading language (HLSL superset) with:
- First-class differentiable programming via `[Differentiable]` attribute
- Compilation targets: GLSL, HLSL, SPIR-V, CUDA, Metal, CPU
- Python bindings: **SlangPy** (included in Slang package)

This is the right choice for **inverse rendering**: optimizing SDF parameters from rendered images.

### SDF ops in Slang are differentiable out of the box

```hlsl
[Differentiable]
float sdSphere(float3 p, float radius) { return length(p) - radius; }

[Differentiable]
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
    return lerp(b, a, h) - k*h*(1.0-h);
}

[Differentiable]
float sceneSDF(float3 p, float radius, float3 translate, float smoothness) {
    float d1 = sdSphere(p - translate, radius);
    float d2 = sdBox(p, float3(1.0));
    return smin(d1, d2, smoothness);
}
```

Slang auto-generates `bwd_diff(sceneSDF)`, giving `d_loss/d_radius`, `d_loss/d_translate`, etc. directly from an image gradient.

### SlangPy usage

SlangPy integrates with PyTorch autograd (not JAX directly):

```python
import slangpy, torch

module = slangpy.load_module("cadjoint_sdf.slang")

# PyTorch tensors with grad tracking
radius = torch.tensor(1.0, requires_grad=True)
translate = torch.zeros(3, requires_grad=True)

image = module.raymarch(
    resolution=(512, 512),
    radius=radius,
    translate=translate,
)
loss = ((image - target_image)**2).mean()
loss.backward()
print(radius.grad)   # d_loss/d_radius via Slang AD
```

### JAX + SlangPy integration loop

Bridge via DLPack (zero-copy GPU transfer):

```python
from jax.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

def render_slang(jax_params: dict) -> jnp.ndarray:
    torch_params = {
        k: from_dlpack(to_dlpack(v)).requires_grad_(True)
        for k, v in jax_params.items()
    }
    image = slang_module.raymarch(resolution=(512,512), **torch_params)
    return image, torch_params

# Optimization loop: JAX handles geometry constraints,
# Slang handles photometric loss
for step in range(1000):
    image_t, torch_params = render_slang(params)
    loss = ((image_t - target_torch)**2).mean()
    loss.backward()

    # Grads back to JAX
    grads = {k: jnp.array(v.grad.cpu().numpy()) for k, v in torch_params.items()}
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
```

### Differentiability at boolean boundaries

| Op | Differentiable? | Notes |
|---|---|---|
| `smin` (smooth union) | Yes, everywhere | Use this by default |
| `min` (hard union) | Subgradient at boundary | Gradient flows through closer surface |
| `max` (intersection/difference) | Subgradient at boundary | Same caveat |

jaxCAD already defaults to smooth min (`smoothness=0.1`) — the correct choice for differentiable optimization.

**Sphere-tracing loop gradient**: the step count is discrete, so the loop is not differentiable in general. Two solutions:
1. **Fixed-step marching** (easiest): run exactly `N` steps with no break — fully differentiable. This is what jaxCAD's `fori_loop` already does.
2. **Implicit differentiation** (research-grade): treat the hit point as an implicit function of parameters. Slang supports custom derivative definitions for this.

### Key libraries

- Slang: https://github.com/shader-slang/slang
- SlangPy docs: https://shader-slang.com/slang/user-guide/a1-04-slangpy.html
- Falcor differentiable renderer: https://github.com/NVIDIAGameWorks/Falcor
- Mitsuba 3 / drjit (comparable system): https://github.com/mitsuba-renderer/mitsuba3

### Limitations

- SlangPy targets PyTorch — JAX interop via DLPack adds a layer.
- Requires writing `.slang` files (separate from the Python SDF tree). A `to_slang()` visitor is needed (same effort as `to_glsl()`).
- NVIDIA GPU required for CUDA target.
- Windows/Linux only (no macOS Metal for CUDA kernels as of 2025).
- Slang compilation at Python import: ~1–5 seconds; cacheable.

---

## Comparison and recommended order

| | GLSL + moderngl | NVIDIA Warp | Slang |
|---|---|---|---|
| **Primary use** | Interactive visualization | Fast batch rendering | Differentiable inverse rendering |
| **Autodiff through render** | No | Via `wp.Tape` + `custom_vjp` | Native, PyTorch autograd |
| **GPU requirement** | Any OpenGL GPU | NVIDIA only | NVIDIA only |
| **Implementation effort** | Low | Medium | Medium–High |
| **Render speed** | ~60fps at 512×512 | ~30–60fps at 512×512 | ~10–30fps at 512×512 |
| **New dependencies** | `moderngl` | `warp-lang` | `slangpy` + PyTorch |

**Recommended order:**

1. **GLSL + moderngl** — Add `to_glsl()` to `SDF` base class and each subclass. Wire into `render_realtime()` in `render.py`. Gives interactive visualization immediately.

2. **Warp** — When high-resolution batch rendering or performance is needed. Write a Warp code generator mirroring the GLSL one. Bridge back to JAX via DLPack + `custom_vjp`.

3. **Slang** — Deferred. Revisit only if inverse rendering becomes an explicit project goal.

---

## Appendix: JAX IR export (for completeness)

JAX does expose its compiled representation:

```python
import jax
# Lower to StableHLO MLIR
lowered = jax.jit(sdf).lower(jnp.zeros(3))
stablehlo_text = lowered.compiler_ir(dialect='stablehlo')
```

**IREE** (https://github.com/openxla/iree, `pip install iree-compiler iree-runtime`) can compile StableHLO to Vulkan SPIR-V, enabling GPU execution without NVIDIA dependency. However, IREE produces compute kernels — not GLSL fragment shaders with a sphere-tracing loop. The GLSL tree-walk approach is more practical for visualization.
