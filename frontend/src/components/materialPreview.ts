/**
 * A material's swatch, rendered rather than painted.
 *
 * The swatch is a small part, not a ball: a flange with a raised boss, a
 * through-bore and four bolt holes. A sphere shows one highlight and one
 * terminator; a part with flats, fillets, a bore and edges shows what a
 * material does to a CAD solid, which is the question the library answers.
 *
 * It is ray-marched on the CPU into a canvas, once per material appearance,
 * and cached by that appearance. At 96 px square it is ~9k rays, a few
 * milliseconds; the browser's downscale to the 48 px tile is the anti-alias.
 * The shading is the viewport's in miniature: GGX specular with Schlick
 * Fresnel over a metallic/roughness base, a key and fill light, a studio
 * environment for the reflections, and the part standing on the viewport's
 * own ground with a soft contact shadow, so a swatch reads as the part will.
 */

import { VIEWPORT_BACKGROUND } from "../tokens";

export interface PreviewAppearance {
  color: readonly [number, number, number];
  roughness: number;
  metallic: number;
  opacity: number;
}

type V3 = [number, number, number];

const SIZE = 96;
const cache = new Map<string, string>();

const clamp = (x: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x));
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const toLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toSrgb = (c: number) => (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);
const norm = (v: V3): V3 => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Smooth minimum and its subtraction twin, as in cadjoint's own unions. */
function smin(a: number, b: number, k: number): number {
  const h = clamp(0.5 + (0.5 * (b - a)) / k);
  return mix(b, a, h) - k * h * (1 - h);
}
function smax(a: number, b: number, k: number): number {
  return -smin(-a, -b, k);
}

/** A capped cylinder about +Y, rounded by `r` at its rims. */
function cylinder(x: number, y: number, z: number, radius: number, half: number, r: number): number {
  const dx = Math.hypot(x, z) - radius + r;
  const dy = Math.abs(y) - half + r;
  return Math.min(Math.max(dx, dy), 0) + Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) - r;
}

const BOLT = 0.47;
const BOLTS: [number, number][] = [0, 1, 2, 3].map((i) => {
  const a = Math.PI / 4 + (i * Math.PI) / 2;
  return [Math.cos(a) * BOLT, Math.sin(a) * BOLT];
});

/** The part: flange, filleted boss, bore, four bolt holes. Y up, ground at -0.3. */
function part(x: number, y: number, z: number): number {
  const flange = cylinder(x, y + 0.21, z, 0.66, 0.09, 0.03);
  const boss = cylinder(x, y + 0.02, z, 0.32, 0.2, 0.04);
  let d = smin(flange, boss, 0.07);
  d = smax(d, -cylinder(x, y, z, 0.15, 0.5, 0), 0.025);
  for (const [bx, bz] of BOLTS) d = smax(d, -cylinder(x - bx, y + 0.21, z - bz, 0.065, 0.2, 0), 0.015);
  return d;
}

const GROUND_Y = -0.3;

function normal(p: V3): V3 {
  const e = 0.002;
  return norm([
    part(p[0] + e, p[1], p[2]) - part(p[0] - e, p[1], p[2]),
    part(p[0], p[1] + e, p[2]) - part(p[0], p[1] - e, p[2]),
    part(p[0], p[1], p[2] + e) - part(p[0], p[1], p[2] - e),
  ]);
}

/** Penumbra soft shadow toward `l`, 1 = lit. */
function shadow(p: V3, l: V3): number {
  let t = 0.02;
  let lit = 1;
  for (let i = 0; i < 40 && t < 3; i++) {
    const d = part(p[0] + l[0] * t, p[1] + l[1] * t, p[2] + l[2] * t);
    if (d < 0.0005) return 0;
    lit = Math.min(lit, (10 * d) / t);
    t += clamp(d, 0.01, 0.2);
  }
  return clamp(lit);
}

function occlusion(p: V3, n: V3): number {
  let occ = 0;
  for (let i = 1; i <= 4; i++) {
    const h = 0.03 * i;
    occ += (h - part(p[0] + n[0] * h, p[1] + n[1] * h, p[2] + n[2] * h)) / 2 ** i;
  }
  return clamp(1 - 4 * occ);
}

const KEY = norm([-0.55, 0.8, 0.45]);
const FILL = norm([0.7, 0.35, -0.4]);

/** The studio the reflections see: a bright softbox overhead, a dim floor. */
function environment(r: V3): number {
  const sky = mix(0.32, 1.25, clamp(r[1] * 0.5 + 0.5) ** 1.6);
  const softbox = Math.max(0, dot(r, KEY)) ** 24 * 5;
  return r[1] < -0.05 ? 0.22 : sky + softbox;
}

function ggx(n: V3, v: V3, l: V3, roughness: number): { spec: number; nl: number } {
  const h = norm([v[0] + l[0], v[1] + l[1], v[2] + l[2]]);
  const nl = Math.max(dot(n, l), 0);
  const nv = Math.max(dot(n, v), 1e-4);
  const nh = Math.max(dot(n, h), 0);
  const a = roughness * roughness;
  const a2 = a * a;
  const dd = nh * nh * (a2 - 1) + 1;
  const D = a2 / (Math.PI * dd * dd);
  const k = ((roughness + 1) ** 2) / 8;
  const G = (nl / (nl * (1 - k) + k)) * (nv / (nv * (1 - k) + k));
  return { spec: (D * G) / Math.max(4 * nl * nv, 1e-4), nl };
}

function filmic(x: number): number {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

function render(appearance: PreviewAppearance, data: Uint8ClampedArray): void {
  const albedo = appearance.color.map((c) => toLinear(clamp(c))) as V3;
  const metallic = clamp(appearance.metallic);
  const roughness = clamp(appearance.roughness, 0.05, 1);
  const opacity = clamp(appearance.opacity, 0.16, 1);
  const ground = VIEWPORT_BACKGROUND.map(toLinear) as V3;
  const f0 = albedo.map((c) => mix(0.04, c, metallic)) as V3;

  // Orthographic camera, 30° above the floor, turned 30° off the axis.
  const elev = (30 * Math.PI) / 180;
  const azim = (30 * Math.PI) / 180;
  const dir = norm([-Math.cos(elev) * Math.sin(azim), -Math.sin(elev), -Math.cos(elev) * Math.cos(azim)]);
  const right = norm([Math.cos(azim), 0, -Math.sin(azim)]);
  const up: V3 = [
    right[1] * dir[2] - right[2] * dir[1],
    right[2] * dir[0] - right[0] * dir[2],
    right[0] * dir[1] - right[1] * dir[0],
  ];
  const view = 0.8;
  const v: V3 = [-dir[0], -dir[1], -dir[2]];

  for (let py = 0; py < SIZE; py++) {
    for (let px = 0; px < SIZE; px++) {
      const sx = ((px + 0.5) / SIZE) * 2 - 1;
      const sy = 1 - ((py + 0.5) / SIZE) * 2 - 0.02;
      const o: V3 = [
        right[0] * sx * view + up[0] * sy * view - dir[0] * 3,
        right[1] * sx * view + up[1] * sy * view - dir[1] * 3,
        right[2] * sx * view + up[2] * sy * view - dir[2] * 3,
      ];
      let t = 0;
      let hit = false;
      for (let i = 0; i < 90 && t < 6; i++) {
        const d = part(o[0] + dir[0] * t, o[1] + dir[1] * t, o[2] + dir[2] * t);
        if (d < 0.0008) {
          hit = true;
          break;
        }
        t += d * 0.9;
      }
      // The floor behind and under the part, with its contact shadow.
      const tg = (GROUND_Y - o[1]) / dir[1];
      const g: V3 = [o[0] + dir[0] * tg, GROUND_Y, o[2] + dir[2] * tg];
      const lit = mix(0.72, 1, shadow(g, KEY));
      const back: V3 = [ground[0] * lit, ground[1] * lit, ground[2] * lit];
      let rgb: V3 = back;

      if (hit && t < tg) {
        const p: V3 = [o[0] + dir[0] * t, o[1] + dir[1] * t, o[2] + dir[2] * t];
        const n = normal(p);
        const ao = occlusion(p, n);
        const nv = Math.max(dot(n, v), 1e-4);
        const fres = (1 - nv) ** 5;
        const out: V3 = [0, 0, 0];
        for (const [l, power, shadowed] of [
          [KEY, 2.4, true],
          [FILL, 0.55, false],
        ] as [V3, number, boolean][]) {
          const { spec, nl } = ggx(n, v, l, roughness);
          if (nl <= 0) continue;
          const s = shadowed ? mix(0.15, 1, shadow(p, l)) : 1;
          for (let c = 0; c < 3; c++) {
            const F = f0[c] + (1 - f0[c]) * fres;
            out[c] += (((1 - F) * (1 - metallic) * albedo[c]) / Math.PI + F * spec) * nl * power * s;
          }
        }
        const rd = 2 * dot(n, v);
        const r = norm([rd * n[0] - v[0], rd * n[1] - v[1], rd * n[2] - v[2]]);
        const env = mix(environment(r), 0.62, roughness * roughness);
        const hemi = mix(0.3, 0.75, n[1] * 0.5 + 0.5);
        for (let c = 0; c < 3; c++) {
          const Fr = f0[c] + (Math.max(1 - roughness, f0[c]) - f0[c]) * fres;
          out[c] += (Fr * env + (1 - metallic) * albedo[c] * hemi * 0.55) * ao;
        }
        rgb = [mix(back[0], out[0], opacity), mix(back[1], out[1], opacity), mix(back[2], out[2], opacity)];
        rgb = rgb.map((c) => filmic(c * 1.15)) as V3;
      }

      const k = (py * SIZE + px) * 4;
      for (let c = 0; c < 3; c++) data[k + c] = Math.round(toSrgb(clamp(rgb[c])) * 255);
      data[k + 3] = 255;
    }
  }
}

const keyOf = (a: PreviewAppearance) =>
  [...a.color, a.roughness, a.metallic, a.opacity].map((x) => x.toFixed(3)).join(",");

/**
 * The swatch as a data URL, or null where there is no canvas (tests, a
 * worker). Cached per appearance, so a library re-render costs nothing and
 * dragging a roughness slider renders only the values it passes through.
 */
export function materialPreviewUrl(appearance: PreviewAppearance): string | null {
  const key = keyOf(appearance);
  const hit = cache.get(key);
  if (hit) return hit;
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const context = canvas.getContext("2d");
  if (!context) return null;
  const image = context.createImageData(SIZE, SIZE);
  render(appearance, image.data);
  context.putImageData(image, 0, 0);
  const url = canvas.toDataURL("image/png");
  if (cache.size > 64) cache.clear();
  cache.set(key, url);
  return url;
}
