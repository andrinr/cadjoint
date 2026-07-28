# WebGPU SDF path tracing

## Goal

Add an optional progressive path-tracing mode to the browser playground while
preserving the existing fast deterministic preview. Geometry and material
selection use the same JAX → StableHLO → WGSL scene compiler; the renderer is a
hand-written WGSL transport runtime around those generated functions.

## Primary references

- Kajiya's [rendering equation](https://doi.org/10.1145/15886.15902) defines
  path tracing as a Monte Carlo solution to surface light transport.
- PBRT's [path-tracing chapter](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/Path_Tracing)
  and [improved integrator](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer)
  motivate BSDF sampling, next-event estimation, throughput tracking, and
  Russian roulette.
- Veach and Guibas'
  [multiple importance sampling](https://doi.org/10.1145/218380.218498)
  combines BSDF and light sampling with substantially lower variance when both
  distributions can generate the same paths.
- Walter et al.'s
  [microfacet reflection and refraction model](https://diglib.eg.org/items/590e957c-92d6-4d8f-9c4c-c23ec106ecda/full)
  supplies the GGX distribution used for rough metallic reflection.
- Jarzynski and Olano's
  [GPU hash evaluation](https://jcgt.org/published/0009/03/02/) supports using a
  compact PCG hash for independent per-pixel sample streams.
- Inigo Quilez's
  [binary-search SDF intersections](https://iquilezles.org/articles/binarysearchsdf/)
  motivates bracketing sign changes and refining them rather than accepting a
  small positive distance as a surface hit.
- The [WebGPU specification](https://gpuweb.github.io/gpuweb/) requires
  compatible texture usages within a render pass. Progressive accumulation
  therefore uses distinct read and write textures and swaps them each frame.

## First-mode design

Each animation frame traces one jittered camera path per pixel and folds it into
a running linear-HDR average:

1. Sphere-trace the generated `sdf(point)` until its sign changes, then refine
   the bracket with seven binary-search steps. Near misses remain misses even
   when they pass within the surface epsilon.
2. Read `material_base(point)` and `material_optics(point)`.
3. Add next-event lighting from a sampled finite sun when the material has a
   non-delta opaque component.
4. Continue with one sampled event:
   - cosine-weighted Lambertian / importance-sampled GGX for opaque surfaces;
   - perfect reflection for explicit mirror reflectivity;
   - Schlick-Fresnel reflection or Snell refraction for glass.
5. Add environment radiance on a miss.
6. Apply Russian roulette after three bounces.

Two `rgba16float` textures ping-pong between sampled input and render
attachment. A separate presentation pass applies ACES tone mapping and gamma.
Accumulation resets after camera, viewport, scene, or rendering-mode changes.
Quality presets scale resolution, bounce depth, total accumulation, and the
number of finite-sun visibility samples per surface hit.

## Deliberate boundaries

- The finite sun is sampled only by next-event estimation and is not included
  in environment radiance, so the current light and BSDF strategies do not
  overlap. Add power-heuristic MIS with the first emissive area light or
  importance-sampled environment map.
- The initial material model has no emission property. Environment and
  directional lighting still produce multi-bounce indirect illumination, but
  emissive geometry belongs in a material-schema PR.
- The browser mode caps accumulation to avoid wasting frames after half-float
  precision stops producing useful improvements.
- Denoising and biased firefly clamping are intentionally omitted. They should
  be evaluated against a high-sample reference before becoming defaults.
