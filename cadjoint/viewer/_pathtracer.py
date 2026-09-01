"""Progressive WebGPU path-tracing shaders for the browser playground."""

from __future__ import annotations

from cadjoint.viewer._camera_wgsl import inject_camera

_SCENE_MARKER = "__CADJOINT_SCENE_CODE__"

_PATH_TRACER_TEMPLATE = r"""__CADJOINT_SCENE_CODE__

const PI: f32 = 3.141592653589793;
const HIT_EPSILON: f32 = 0.0005;
const MAX_DISTANCE: f32 = 100.0;
const MAX_TRACE_STEPS: u32 = 160u;
const MAX_PATH_BOUNCES: u32 = 8u;
const MAX_SHADOW_SAMPLES: u32 = 4u;
const DISPLAY_SHADOWS: u32 = 1u;
const DISPLAY_REFLECTIONS: u32 = 2u;
const DISPLAY_FLAT: u32 = 4u;
const DISPLAY_HIDE_SOLID: u32 = 8u;
const DISPLAY_HARD_SHADOWS: u32 = 16u;
const HARD_SHADOW_FLOOR: f32 = 0.45;

struct Uniforms {
  resolution   : vec4<f32>,
  camera_pos   : vec4<f32>,
  camera_target: vec4<f32>,
  light_dir    : vec4<f32>,
  bg_color     : vec4<f32>,
  path_settings: vec4<f32>,
  display      : vec4<f32>,
};

struct TraceHit {
  distance: f32,
  hit: bool,
};

struct BsdfSample {
  direction: vec3<f32>,
  weight: vec3<f32>,
  valid: bool,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var previous_accumulation: texture_2d<f32>;

__CADJOINT_CAMERA__

fn display_flag(flag: u32) -> bool {
  return (u32(max(u.display.z, 0.0)) & flag) != 0u;
}

fn pcg_hash(input: u32) -> u32 {
  let state = input * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn random_f32(state: ptr<function, u32>) -> f32 {
  *state = pcg_hash(*state);
  return f32(*state) * (1.0 / 4294967296.0);
}

fn signs_differ(a: f32, b: f32) -> bool {
  return (a < 0.0 && b >= 0.0) || (a > 0.0 && b <= 0.0);
}

fn refine_sign_crossing(
  origin: vec3<f32>,
  direction: vec3<f32>,
  near_distance: f32,
  far_distance: f32,
  near_surface_distance: f32,
) -> f32 {
  var near = near_distance;
  var far = far_distance;
  var near_sdf = near_surface_distance;
  for (var refinement = 0u; refinement < 7u; refinement += 1u) {
    let middle = 0.5 * (near + far);
    let middle_sdf = sdf(origin + direction * middle);
    let same_side = (near_sdf < 0.0) == (middle_sdf < 0.0);
    if (same_side) {
      near = middle;
      near_sdf = middle_sdf;
    } else {
      far = middle;
    }
  }
  return 0.5 * (near + far);
}

fn trace_scene(origin: vec3<f32>, direction: vec3<f32>) -> TraceHit {
  var distance = 0.0;
  var surface_distance = sdf(origin);
  for (var step = 0u; step < MAX_TRACE_STEPS; step += 1u) {
    let advance = max(abs(surface_distance) * 0.9, HIT_EPSILON * 0.5);
    let next_distance = distance + advance;
    if (next_distance >= MAX_DISTANCE) {
      break;
    }
    let next_surface_distance = sdf(origin + direction * next_distance);
    if (signs_differ(surface_distance, next_surface_distance)) {
      let refined_distance = refine_sign_crossing(
        origin,
        direction,
        distance,
        next_distance,
        surface_distance,
      );
      return TraceHit(refined_distance, true);
    }
    distance = next_distance;
    surface_distance = next_surface_distance;
  }
  return TraceHit(MAX_DISTANCE, false);
}

fn visible_to_directional_light(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
  var distance = 0.0;
  var surface_distance = sdf(origin);
  if (surface_distance <= 0.0) {
    return 0.0;
  }
  for (var step = 0u; step < 96u; step += 1u) {
    let advance = max(surface_distance * 0.9, HIT_EPSILON * 0.5);
    let next_distance = distance + advance;
    if (next_distance >= 50.0) {
      break;
    }
    let next_surface_distance = sdf(origin + direction * next_distance);
    if (next_surface_distance <= 0.0) {
      return 0.0;
    }
    distance = next_distance;
    surface_distance = next_surface_distance;
  }
  return 1.0;
}

fn sdf_normal(position: vec3<f32>) -> vec3<f32> {
  let e = 0.00075;
  return safe_normalize(vec3<f32>(
    sdf(position + vec3<f32>( e, 0.0, 0.0)) -
      sdf(position + vec3<f32>(-e, 0.0, 0.0)),
    sdf(position + vec3<f32>(0.0,  e, 0.0)) -
      sdf(position + vec3<f32>(0.0, -e, 0.0)),
    sdf(position + vec3<f32>(0.0, 0.0,  e)) -
      sdf(position + vec3<f32>(0.0, 0.0, -e)),
  ));
}

// The viewport ground — see the identical function in `_webgpu.py` for why it
// is flat. Keep the two in sync: the preview and the accumulated path trace
// are the same picture at two sample counts, and a background that changes
// when the trace starts reads as a bug.
fn environment_radiance(direction: vec3<f32>) -> vec3<f32> {
  return max(u.bg_color.xyz, vec3<f32>(0.001));
}

fn orthonormal_basis(normal: vec3<f32>) -> mat3x3<f32> {
  let helper = select(
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(1.0, 0.0, 0.0),
    abs(normal.z) > 0.999,
  );
  let tangent = safe_normalize(cross(helper, normal));
  let bitangent = cross(normal, tangent);
  return mat3x3<f32>(tangent, bitangent, normal);
}

fn sample_cosine_hemisphere(
  normal: vec3<f32>,
  random_sample: vec2<f32>,
) -> vec3<f32> {
  let radius = sqrt(random_sample.x);
  let phi = 2.0 * PI * random_sample.y;
  let local = vec3<f32>(
    radius * cos(phi),
    radius * sin(phi),
    sqrt(max(0.0, 1.0 - random_sample.x)),
  );
  return safe_normalize(orthonormal_basis(normal) * local);
}

fn sample_sun_direction(
  center_direction: vec3<f32>,
  random_sample: vec2<f32>,
) -> vec3<f32> {
  let angular_radius = 0.025;
  let cos_theta = mix(cos(angular_radius), 1.0, random_sample.x);
  let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  let phi = 2.0 * PI * random_sample.y;
  let local = vec3<f32>(
    sin_theta * cos(phi),
    sin_theta * sin(phi),
    cos_theta,
  );
  return safe_normalize(orthonormal_basis(center_direction) * local);
}

fn sample_ggx_half_vector(
  normal: vec3<f32>,
  alpha: f32,
  random_sample: vec2<f32>,
) -> vec3<f32> {
  let alpha_squared = alpha * alpha;
  let phi = 2.0 * PI * random_sample.x;
  let cos_theta = sqrt(
    max(0.0, (1.0 - random_sample.y) /
      (1.0 + (alpha_squared - 1.0) * random_sample.y)),
  );
  let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  let local = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
  return safe_normalize(orthonormal_basis(normal) * local);
}

fn ggx_distribution(alpha: f32, normal_dot_half: f32) -> f32 {
  let alpha_squared = alpha * alpha;
  let denominator =
    normal_dot_half * normal_dot_half * (alpha_squared - 1.0) + 1.0;
  return alpha_squared / max(PI * denominator * denominator, 1e-7);
}

fn geometry_schlick(normal_dot_direction: f32, roughness: f32) -> f32 {
  let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
  return normal_dot_direction /
    max(normal_dot_direction * (1.0 - k) + k, 1e-6);
}

fn fresnel_schlick_rgb(cosine: f32, f0: vec3<f32>) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_dielectric(cosine: f32, ior: f32) -> f32 {
  let r0_base = (1.0 - ior) / (1.0 + ior);
  let r0 = r0_base * r0_base;
  return r0 + (1.0 - r0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

fn evaluate_opaque_bsdf(
  base_color: vec3<f32>,
  roughness: f32,
  metallic: f32,
  normal: vec3<f32>,
  outgoing: vec3<f32>,
  incoming: vec3<f32>,
) -> vec3<f32> {
  let normal_dot_incoming = max(dot(normal, incoming), 0.0);
  let normal_dot_outgoing = max(dot(normal, outgoing), 0.0);
  if (normal_dot_incoming <= 0.0 || normal_dot_outgoing <= 0.0) {
    return vec3<f32>(0.0);
  }

  let half_vector = safe_normalize(outgoing + incoming);
  let normal_dot_half = max(dot(normal, half_vector), 0.0);
  let outgoing_dot_half = max(dot(outgoing, half_vector), 0.0);
  let perceptual_roughness = clamp(roughness, 0.04, 1.0);
  let alpha = perceptual_roughness * perceptual_roughness;
  let distribution = ggx_distribution(alpha, normal_dot_half);
  let geometry =
    geometry_schlick(normal_dot_incoming, perceptual_roughness) *
    geometry_schlick(normal_dot_outgoing, perceptual_roughness);
  let f0 = mix(vec3<f32>(0.04), base_color, metallic);
  let fresnel = fresnel_schlick_rgb(outgoing_dot_half, f0);
  let specular = distribution * geometry * fresnel /
    max(4.0 * normal_dot_incoming * normal_dot_outgoing, 1e-6);
  let diffuse =
    (vec3<f32>(1.0) - fresnel) * (1.0 - metallic) * base_color / PI;
  return diffuse + specular;
}

fn opaque_bsdf_pdf(
  roughness: f32,
  metallic: f32,
  normal: vec3<f32>,
  outgoing: vec3<f32>,
  incoming: vec3<f32>,
) -> f32 {
  let normal_dot_incoming = max(dot(normal, incoming), 0.0);
  if (normal_dot_incoming <= 0.0) {
    return 0.0;
  }
  let specular_probability = mix(0.25, 0.85, metallic);
  let diffuse_pdf = normal_dot_incoming / PI;
  let half_vector = safe_normalize(outgoing + incoming);
  let normal_dot_half = max(dot(normal, half_vector), 0.0);
  let outgoing_dot_half = max(abs(dot(outgoing, half_vector)), 1e-6);
  let alpha = clamp(roughness, 0.04, 1.0);
  let distribution = ggx_distribution(alpha * alpha, normal_dot_half);
  let specular_pdf = distribution * normal_dot_half / (4.0 * outgoing_dot_half);
  return mix(diffuse_pdf, specular_pdf, specular_probability);
}

fn sample_opaque_bsdf(
  base_color: vec3<f32>,
  roughness: f32,
  metallic: f32,
  normal: vec3<f32>,
  outgoing: vec3<f32>,
  random_lobe: f32,
  random_sample: vec2<f32>,
) -> BsdfSample {
  let specular_probability = mix(0.25, 0.85, metallic);
  var incoming: vec3<f32>;
  if (random_lobe < specular_probability) {
    let alpha = clamp(roughness, 0.04, 1.0);
    let half_vector = sample_ggx_half_vector(
      normal,
      alpha * alpha,
      random_sample,
    );
    incoming = reflect(-outgoing, half_vector);
  } else {
    incoming = sample_cosine_hemisphere(normal, random_sample);
  }

  let normal_dot_incoming = max(dot(normal, incoming), 0.0);
  let pdf = opaque_bsdf_pdf(
    roughness,
    metallic,
    normal,
    outgoing,
    incoming,
  );
  if (normal_dot_incoming <= 0.0 || pdf <= 1e-7) {
    return BsdfSample(incoming, vec3<f32>(0.0), false);
  }
  let bsdf = evaluate_opaque_bsdf(
    base_color,
    roughness,
    metallic,
    normal,
    outgoing,
    incoming,
  );
  return BsdfSample(incoming, bsdf * normal_dot_incoming / pdf, true);
}

fn trace_path(
  ray_origin: vec3<f32>,
  ray_direction: vec3<f32>,
  random_state: ptr<function, u32>,
) -> vec3<f32> {
  var origin = ray_origin;
  var direction = ray_direction;
  var radiance = vec3<f32>(0.0);
  var throughput = vec3<f32>(1.0);
  var eta_scale = 1.0;
  let configured_bounces = min(u32(u.path_settings.y), MAX_PATH_BOUNCES);
  var configured_shadow_samples = clamp(
    u32(u.path_settings.z),
    1u,
    MAX_SHADOW_SAMPLES,
  );
  let shadows_enabled = display_flag(DISPLAY_SHADOWS);
  let hard_shadows = display_flag(DISPLAY_HARD_SHADOWS);
  if (!shadows_enabled || hard_shadows) {
    configured_shadow_samples = 1u;
  }

  for (var bounce = 0u; bounce < MAX_PATH_BOUNCES; bounce += 1u) {
    if (bounce >= configured_bounces) {
      break;
    }

    let intersection = trace_scene(origin, direction);
    if (!intersection.hit) {
      // Backplate and light dome are the same colour but not the same
      // intensity. A camera ray that misses has to land on exactly the paper
      // the viewport declares; a *bounce* ray that misses is the studio's
      // fill light, and at paper's own radiance a white part lit by a white
      // dome converges on the ground and the silhouette dissolves. The dome
      // is therefore half the backplate — the standard backplate/HDRI split,
      // and the only knob that keeps a light part legible on a light ground
      // without touching the BSDF.
      let dome = select(0.5, 1.0, bounce == 0u);
      radiance += throughput * environment_radiance(direction) * dome;
      break;
    }

    let position = origin + direction * intersection.distance;
    let outward_normal = sdf_normal(position);
    let front_face = dot(direction, outward_normal) < 0.0;
    let normal = select(-outward_normal, outward_normal, front_face);
    let outgoing = -direction;
    let base = material_base(position);
    let optics = material_optics(position);
    let base_color = clamp(base.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let roughness = clamp(base.w, 0.04, 1.0);
    let metallic = clamp(optics.x, 0.0, 1.0);
    let xray = clamp(u.display.w, 0.0, 1.0);
    let facing = 1.0 - abs(dot(normal, direction));
    let xray_alpha = mix(1.0, mix(0.12, 0.95, facing * facing), xray);
    let opacity = clamp(optics.y, 0.0, 1.0) * xray_alpha;
    let ior = max(optics.z, 1.0001);
    let reflectivity = select(
      0.0,
      clamp(optics.w, 0.0, 1.0),
      display_flag(DISPLAY_REFLECTIONS),
    );

    if (display_flag(DISPLAY_FLAT)) {
      let light_direction = safe_normalize(u.light_dir.xyz);
      let normal_dot_light = max(dot(normal, light_direction), 0.0);
      var visibility = 1.0;
      if (shadows_enabled) {
        visibility = visible_to_directional_light(
          position + normal * (HIT_EPSILON * 6.0),
          light_direction,
        );
        if (hard_shadows) {
          visibility = mix(HARD_SHADOW_FLOOR, 1.0, visibility);
        }
      }
      let flat_color =
        base_color * mix(0.35, 1.0, normal_dot_light * visibility);
      return mix(environment_radiance(direction), flat_color, opacity);
    }

    if (
      opacity > 0.0 &&
      reflectivity < 1.0
    ) {
      let light_radiance =
        vec3<f32>(1.0, 0.92, 0.82) * max(u.light_dir.w, 0.0);
      var direct_lighting = vec3<f32>(0.0);
      for (
        var shadow_sample = 0u;
        shadow_sample < MAX_SHADOW_SAMPLES;
        shadow_sample += 1u
      ) {
        if (shadow_sample >= configured_shadow_samples) {
          break;
        }
        var light_direction = safe_normalize(u.light_dir.xyz);
        if (shadows_enabled && !hard_shadows) {
          light_direction = sample_sun_direction(
            light_direction,
            vec2<f32>(random_f32(random_state), random_f32(random_state)),
          );
        }
        let normal_dot_light = max(dot(normal, light_direction), 0.0);
        if (normal_dot_light > 0.0) {
          var visibility = 1.0;
          if (shadows_enabled) {
            visibility = visible_to_directional_light(
              position + normal * (HIT_EPSILON * 6.0),
              light_direction,
            );
            if (hard_shadows) {
              visibility = mix(HARD_SHADOW_FLOOR, 1.0, visibility);
            }
          }
          let direct_bsdf = evaluate_opaque_bsdf(
            base_color,
            roughness,
            metallic,
            normal,
            outgoing,
            light_direction,
          );
          direct_lighting +=
            direct_bsdf * light_radiance * normal_dot_light * visibility;
        }
      }
      radiance += throughput * opacity * (1.0 - reflectivity) *
        direct_lighting / f32(configured_shadow_samples);
    }

    let transparent_event = random_f32(random_state) >= opacity;
    if (transparent_event) {
      let cosine = max(dot(outgoing, normal), 0.0);
      let fresnel = fresnel_schlick_dielectric(cosine, ior);
      let eta = select(ior, 1.0 / ior, front_face);
      let transmitted_sine_squared =
        eta * eta * max(0.0, 1.0 - cosine * cosine);
      let total_internal_reflection = transmitted_sine_squared >= 1.0;
      if (
        total_internal_reflection ||
        random_f32(random_state) < fresnel
      ) {
        direction = reflect(direction, normal);
      } else {
        direction = safe_normalize(refract(direction, normal, eta));
        throughput *= eta * eta;
        eta_scale /= eta * eta;
        if (!front_face) {
          throughput *= base_color;
        }
      }
    } else {
      let mirror_event = random_f32(random_state) < reflectivity;
      if (mirror_event) {
        direction = reflect(direction, normal);
        throughput *= mix(vec3<f32>(1.0), base_color, metallic);
      } else {
        let sample = sample_opaque_bsdf(
          base_color,
          roughness,
          metallic,
          normal,
          outgoing,
          random_f32(random_state),
          vec2<f32>(random_f32(random_state), random_f32(random_state)),
        );
        if (!sample.valid) {
          break;
        }
        direction = sample.direction;
        throughput *= sample.weight;
      }
    }

    direction = safe_normalize(direction);
    let offset_sign = select(-1.0, 1.0, dot(direction, normal) >= 0.0);
    origin = position + normal * (HIT_EPSILON * 6.0 * offset_sign);

    if (bounce >= 2u) {
      let roulette_throughput = throughput * eta_scale;
      let survival_probability = clamp(
        max(
          roulette_throughput.x,
          max(roulette_throughput.y, roulette_throughput.z),
        ),
        0.05,
        0.95,
      );
      if (random_f32(random_state) >= survival_probability) {
        break;
      }
      throughput /= survival_probability;
    }
  }
  return max(radiance, vec3<f32>(0.0));
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
  let positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_path_trace(
  @builtin(position) fragment: vec4<f32>,
) -> @location(0) vec4<f32> {
  let pixel = vec2<i32>(fragment.xy);
  let sample_index = u32(u.path_settings.x);
  var random_state = pcg_hash(
    (u32(pixel.x) * 1973u) ^
    (u32(pixel.y) * 9277u) ^
    (sample_index * 26699u) ^
    0x68bc21ebu
  );
  let jitter = vec2<f32>(
    random_f32(&random_state) - 0.5,
    random_f32(&random_state) - 0.5,
  );

  let resolution = u.resolution.xy;
  let uv = ((fragment.xy + jitter) / resolution - 0.5) *
    vec2<f32>(resolution.x / resolution.y, -1.0);
  let ray = primary_ray(
    uv,
    u.camera_pos.xyz,
    u.camera_target.xyz,
    u.display.x,
    u.display.y,
  );
  let camera = ray.origin;
  let ray_direction = ray.direction;
  var sample_radiance = environment_radiance(ray_direction);
  if (!display_flag(DISPLAY_HIDE_SOLID)) {
    sample_radiance = trace_path(camera, ray_direction, &random_state);
  }

  let previous = textureLoad(previous_accumulation, pixel, 0).xyz;
  let sample_count = f32(sample_index);
  let accumulated =
    (previous * sample_count + sample_radiance) / (sample_count + 1.0);
  return vec4<f32>(accumulated, 1.0);
}
"""

WGSL_PRESENT_TEMPLATE = r"""
@group(0) @binding(0) var accumulated_radiance: texture_2d<f32>;

fn aces_tone_map(color: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp(
    (color * (a * color + vec3<f32>(b))) /
      (color * (c * color + vec3<f32>(d)) + vec3<f32>(e)),
    vec3<f32>(0.0),
    vec3<f32>(1.0),
  );
}

@vertex
fn vs_present(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
  let positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_present(@builtin(position) fragment: vec4<f32>) -> @location(0) vec4<f32> {
  let pixel = vec2<i32>(fragment.xy);
  let linear_color = textureLoad(accumulated_radiance, pixel, 0).xyz;
  let display_color = pow(
    aces_tone_map(linear_color),
    vec3<f32>(1.0 / 2.2),
  );
  return vec4<f32>(display_color, 1.0);
}
"""


WGSL_PATH_TRACER_TEMPLATE = inject_camera(_PATH_TRACER_TEMPLATE)


def build_path_tracer_shader(scene_code: str) -> str:
    """Embed compiled scene distance and material functions in the path tracer."""
    if _SCENE_MARKER in scene_code:
        raise ValueError(f"Scene source cannot contain the reserved marker {_SCENE_MARKER!r}")
    return WGSL_PATH_TRACER_TEMPLATE.replace(_SCENE_MARKER, scene_code, 1)
