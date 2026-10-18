/**
 * The parameter buffer's packing rules, and the module cache in front of it.
 *
 * Both are pure enough to test without a GPU: `packParameters` is arithmetic
 * over a slot table, and `ShaderModuleCache` only needs an object that
 * answers `createShaderModule` and `getCompilationInfo`. What is being
 * pinned here is the contract the WGSL backend generates against — one
 * `vec4<f32>` slot per parameter, `.x`/`.xy`/`.xyz` swizzles decided at
 * compile time — because a packing bug does not fail loudly. It silently
 * draws a different part.
 */

import { describe, expect, it, vi } from "vitest";
import {
  OPEN_BOX_HALF,
  PARAMETER_SLOT_BYTES,
  ShaderModuleCache,
  packParameters,
  sameLayout,
  type ShaderProgramPayload,
} from "../src/viewer/shaderProgram";

/** A three-slot program: a scalar, a vec3 and a second scalar. */
const program: ShaderProgramPayload = {
  group: 3,
  binding: 0,
  buffer_bytes: 48,
  parameters: [
    { name: "fin_depth", offset: 0, components: 1, value: [2.5], free: true },
    { name: "box.size", offset: 16, components: 3, value: [1, 2, 3], free: false },
    { name: "bore", offset: 32, components: 1, value: [0.75], free: true },
  ],
};

describe("packParameters", () => {
  it("puts each parameter at its own 16-byte slot", () => {
    const packed = packParameters(program);
    expect(packed.length).toBe(12);
    expect(packed[0]).toBeCloseTo(2.5);
    expect([packed[4], packed[5], packed[6]]).toEqual([1, 2, 3]);
    expect(packed[8]).toBeCloseTo(0.75);
  });

  it("leaves the unused floats of a slot zeroed", () => {
    const packed = packParameters(program);
    // A scalar reads .x; the other three floats of its slot are padding and
    // must not carry the neighbouring parameter's value.
    expect([packed[1], packed[2], packed[3]]).toEqual([0, 0, 0]);
    expect(packed[7]).toBe(0);
  });

  it("substitutes an override of the right width", () => {
    const packed = packParameters(program, { fin_depth: [9] });
    expect(packed[0]).toBe(9);
    // Everything else keeps the program's own value.
    expect(packed[8]).toBeCloseTo(0.75);
  });

  it("ignores an override of the wrong width rather than corrupting the slot", () => {
    // The swizzle is fixed at compile time, so five floats cannot go into a
    // one-float slot; writing them would spill into the next parameter.
    const packed = packParameters(program, { fin_depth: [1, 2, 3, 4, 5] });
    expect(packed[0]).toBeCloseTo(2.5);
    expect([packed[4], packed[5], packed[6]]).toEqual([1, 2, 3]);
  });

  it("writes NaN for a component the scene never set", () => {
    // JSON has no NaN, so an unset material property travels as null. The
    // literal form inlines a NaN into the same slot; this must match it.
    const unset: ShaderProgramPayload = {
      ...program,
      parameters: [
        { name: "material.density", offset: 0, components: 1, value: [null], free: false },
        ...program.parameters.slice(1),
      ],
    };
    const packed = packParameters(unset);
    expect(Number.isNaN(packed[0])).toBe(true);
    expect(packed[8]).toBeCloseTo(0.75);
  });

  it("opens a culling box whose parameters a drag has overridden, and only that box", () => {
    const boxed: ShaderProgramPayload = {
      group: 3,
      binding: 0,
      buffer_bytes: 6 * 16,
      parameters: [
        { name: "r", offset: 0, components: 1, value: [0.5], free: true },
        { name: "w", offset: 16, components: 1, value: [0.2], free: true },
        { name: "__cull0.center", offset: 32, components: 3, value: [1, 2, 3], free: false, depends: ["r"] },
        { name: "__cull0.half", offset: 48, components: 3, value: [0.5, 0.5, 0.5], free: false, depends: ["r"] },
        { name: "__cull1.center", offset: 64, components: 3, value: [0, 0, 0], free: false, depends: ["w"] },
        { name: "__cull1.half", offset: 80, components: 3, value: [0.2, 0.2, 0.2], free: false, depends: ["w"] },
      ],
    };
    const at = (packed: Float32Array, offset: number) => Array.from(packed.slice(offset / 4, offset / 4 + 3));
    const rest = packParameters(boxed);
    expect(at(rest, 48)).toEqual([0.5, 0.5, 0.5]);
    const dragging = packParameters(boxed, { r: [0.9] });
    expect(at(dragging, 32)).toEqual([1, 2, 3]);
    expect(at(dragging, 48)).toEqual([OPEN_BOX_HALF, OPEN_BOX_HALF, OPEN_BOX_HALF].map(Math.fround));
    // The other box rests on nothing the drag touched, and keeps culling.
    expect(at(dragging, 80).map((v) => Number(v.toFixed(6)))).toEqual([0.2, 0.2, 0.2]);
  });

  it("allocates at least one slot for a program with no parameters", () => {
    const empty: ShaderProgramPayload = { ...program, buffer_bytes: 0, parameters: [] };
    // WebGPU rejects a zero-sized uniform buffer.
    expect(packParameters(empty).byteLength).toBe(PARAMETER_SLOT_BYTES);
  });
});

describe("sameLayout", () => {
  it("accepts two programs differing only in their values", () => {
    const moved: ShaderProgramPayload = {
      ...program,
      parameters: program.parameters.map((slot) => ({ ...slot, value: [0, 0, 0] })),
    };
    expect(sameLayout(program, moved)).toBe(true);
  });

  it("rejects a renamed, resized, remapped or rebound parameter", () => {
    const rename = { ...program.parameters[0], name: "other" };
    expect(
      sameLayout(program, { ...program, parameters: [rename, ...program.parameters.slice(1)] }),
    ).toBe(false);
    expect(sameLayout(program, { ...program, buffer_bytes: 64 })).toBe(false);
    expect(sameLayout(program, { ...program, binding: 1 })).toBe(false);
    expect(
      sameLayout(program, { ...program, parameters: program.parameters.slice(0, 2) }),
    ).toBe(false);
  });

  it("treats two literal-form scenes as the same layout", () => {
    // Both null: there is no buffer either way, so an identical source is a
    // values-only edit that changed nothing.
    expect(sameLayout(null, null)).toBe(true);
    expect(sameLayout(program, null)).toBe(false);
  });
});

/** A device stub that counts the modules it is asked to compile. */
function fakeDevice() {
  const createShaderModule = vi.fn(({ code }: { code: string }) => ({
    code,
    getCompilationInfo: async () => ({ messages: [] }),
  }));
  return { createShaderModule } as unknown as GPUDevice & {
    createShaderModule: ReturnType<typeof vi.fn>;
  };
}

describe("ShaderModuleCache", () => {
  it("compiles a source once and serves it thereafter", async () => {
    const device = fakeDevice();
    const cache = new ShaderModuleCache();
    const first = await cache.get(device, "fn main() {}", "a");
    const second = await cache.get(device, "fn main() {}", "a");
    expect(second).toBe(first);
    expect(device.createShaderModule).toHaveBeenCalledTimes(1);
    expect(cache.hits).toBe(1);
    expect(cache.misses).toBe(1);
  });

  it("compiles a different source separately", async () => {
    const device = fakeDevice();
    const cache = new ShaderModuleCache();
    await cache.get(device, "a", "a");
    await cache.get(device, "b", "b");
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(cache.hits).toBe(0);
  });

  it("serves an undo from the cache", async () => {
    // The access pattern the cache exists for: edit, then undo back onto a
    // shader this session already compiled.
    const device = fakeDevice();
    const cache = new ShaderModuleCache();
    await cache.get(device, "v1", "v1");
    await cache.get(device, "v2", "v2");
    await cache.get(device, "v1", "v1");
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(cache.hits).toBe(1);
  });

  it("evicts least-recently-used past its capacity", async () => {
    const device = fakeDevice();
    const cache = new ShaderModuleCache(2);
    await cache.get(device, "a", "a");
    await cache.get(device, "b", "b");
    await cache.get(device, "a", "a"); // refreshes a, so b is now the oldest
    await cache.get(device, "c", "c"); // evicts b
    expect(cache.size).toBe(2);
    await cache.get(device, "b", "b");
    expect(device.createShaderModule).toHaveBeenCalledTimes(4);
  });

  it("throws on a source with compilation errors and does not cache it", async () => {
    const device = {
      createShaderModule: vi.fn(() => ({
        getCompilationInfo: async () => ({
          messages: [{ type: "error", lineNum: 3, linePos: 7, message: "unknown identifier" }],
        }),
      })),
    } as unknown as GPUDevice;
    const cache = new ShaderModuleCache();
    await expect(cache.get(device, "bad", "scene")).rejects.toThrow("unknown identifier");
    expect(cache.size).toBe(0);
  });
});
