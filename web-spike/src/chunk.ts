/**
 * Evaluate a field (or its gradient) over many points in fixed-size chunks.
 *
 * A fused scene kernel's intermediates scale with the batch, and jax-js's
 * Wasm heap is 32-bit (4 GiB): a 144-node scene over a 550k-point lattice
 * exceeds it in one call (and its gradient program does at 64k points). Chunks keep the batch bounded, and the last one is
 * padded (with its own first point) so jit sees a single shape.
 */
import { numpy as np } from "@jax-js/jax";
import type { Pts } from "./sdf";

export const CHUNK = Number(globalThis.process?.env?.JAXJS_CHUNK ?? 1 << 14);

/** Run `run` over (xs, ys, zs) in chunks, concatenating each of its outputs. */
export function chunked(
  xs: Float32Array, ys: Float32Array, zs: Float32Array,
  run: (p: Pts) => Float32Array[],
): Float32Array[] {
  const n = xs.length;
  if (n <= CHUNK) return run({ x: np.array(xs), y: np.array(ys), z: np.array(zs) });
  let outs: Float32Array[] | null = null;
  for (let start = 0; start < n; start += CHUNK) {
    const m = Math.min(CHUNK, n - start);
    const take = (a: Float32Array) => {
      const c = new Float32Array(CHUNK).fill(a[start]);
      c.set(a.subarray(start, start + m));
      return np.array(c);
    };
    const parts = run({ x: take(xs), y: take(ys), z: take(zs) });
    outs ??= parts.map(() => new Float32Array(n));
    parts.forEach((p, i) => outs![i].set(p.subarray(0, m), start));
  }
  return outs!;
}

/** Field values at interleaved points [x0, y0, z0, x1, ...]. */
export function sample(field: (p: Pts) => np.Array, flat: Float32Array): Float32Array {
  const n = flat.length / 3;
  const xs = new Float32Array(n), ys = new Float32Array(n), zs = new Float32Array(n);
  for (let i = 0; i < n; i++) { xs[i] = flat[3 * i]; ys[i] = flat[3 * i + 1]; zs[i] = flat[3 * i + 2]; }
  return chunked(xs, ys, zs, (p) => [field(p).dataSync() as Float32Array])[0];
}
