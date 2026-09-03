/**
 * The scene shader's parameter buffer, and the module cache in front of it.
 *
 * The compile worker emits WGSL in one of two forms. In the *literal* form
 * every design parameter is a float constant in the source, so moving one
 * slider by 0.05 rewrites three lines of a three-megabyte module and the
 * browser recompiles all of it — 440 ms of `getCompilationInfo` and 520 ms of
 * pipeline creation for `scenes/motor_shield.py`, per edit. In the *uniform*
 * form the parameters are read out of a `@group(3)` uniform block instead:
 * the source is byte-identical for every value of every parameter, and an
 * edit is a `writeBuffer` of a few hundred bytes.
 *
 * That only pays off if the frontend can *tell* the two cases apart, which
 * is what this module is for:
 *
 * - `packParameters` turns the payload's slot table into the `Float32Array`
 *   the buffer wants — one `vec4<f32>` slot per parameter, which is the only
 *   element type a WGSL uniform array carries without per-field alignment
 *   rules, so a scalar occupies 16 bytes and leaves 12 of them padding.
 * - `ShaderModuleCache` keys compiled `GPUShaderModule`s by their source, so
 *   a topology edit that lands back on a shader this session already saw —
 *   an undo, a redo, a toggle flipped twice — skips `createShaderModule`
 *   entirely.
 *
 * The renderer's own short-circuit sits above both: when a new payload's
 * shader sources are *identical* to the installed ones it never reaches this
 * module's cache either, because it never asks for a module at all.
 */

/** One design parameter's slot, mirroring `ShaderParameter` in the schema. */
export interface ShaderParameterSlot {
  name: string;
  /** Byte offset into the buffer; always a multiple of `PARAMETER_SLOT_BYTES`. */
  offset: number;
  /** How many of the slot's four floats the shader reads (1-4). */
  components: number;
  /**
   * The parameter's value, `null` per component that is not finite.
   *
   * JSON has no NaN, and NaNs are common here: a material that never
   * states its density or Young's modulus carries one in each. The null
   * is a transport detail, not a substitution — `packParameters` writes a
   * NaN back into the buffer, because a NaN is what the literal form
   * inlines into the same slot and the two forms must draw the same image.
   */
  value: (number | null)[];
  free: boolean;
}

/** The uniform contract for a compiled scene, mirroring `ShaderProgram`. */
export interface ShaderProgramPayload {
  group: number;
  binding: number;
  buffer_bytes: number;
  /**
   * Byte offset of a reserved slot the shader reads wherever it needs a NaN.
   *
   * WGSL has no NaN literal that survives const-evaluation — Chromium's
   * compiler rejects the bit-pattern bitcast with "value nan cannot be
   * represented as 'f32'" — so the module loads one from the buffer
   * instead, and this is the one value the packer writes that no parameter
   * owns.
   */
  nan_offset?: number;
  /**
   * Byte offset of the reserved slot holding the bounding-box cull margin.
   *
   * Every skip test inside the *generated* module reads
   * `box_distance(p, bounds) >= threshold + margin`, so writing
   * `CULL_MARGIN_OFF` here makes every test false and the shader falls back
   * to evaluating every leaf. That is what makes culling a render toggle
   * rather than a recompile: the tests are in the scene shader, which reads
   * no uniform but this buffer.
   */
  cull_margin_offset?: number | null;
  parameters: ShaderParameterSlot[];
}

/**
 * The cull margin that leaves culling on.
 *
 * World-unit slack on every skip test, so float rounding in the box distance
 * can never flip a test the exact arithmetic would not. Mirrors `CULL_MARGIN`
 * in `cadjoint/backends/wgsl/_culling.py`.
 */
export const CULL_MARGIN_ON = 1e-4;

/**
 * The cull margin that switches culling off.
 *
 * Infinity, so no box test can ever pass and no operand is ever skipped. The
 * image is identical either way — measured at zero changed pixels on every
 * shipped scene — and the cost is 2.0x to 2.4x the frame.
 */
export const CULL_MARGIN_OFF = Number.POSITIVE_INFINITY;

/** Bytes per parameter: one `vec4<f32>`. Mirrors `PARAMETER_SLOT_BYTES`. */
export const PARAMETER_SLOT_BYTES = 16;

/**
 * The parameter values, packed for upload.
 *
 * `overrides` replaces a parameter's value by name, which is what a handle
 * drag does at frame rate while the server patch is still in flight. An
 * override of the wrong width is ignored rather than trusted: a slot's
 * `components` is fixed at compile time (it decided the swizzle in the
 * generated source), so writing five floats into a three-float slot would
 * corrupt the next parameter rather than fail loudly.
 *
 * @param program The compiled scene's uniform contract.
 * @param overrides Values to substitute, keyed by parameter name.
 * @param cullMargin `CULL_MARGIN_ON` or `CULL_MARGIN_OFF`; see the field.
 * @returns A `float32` array of `buffer_bytes / 4` elements, padding zeroed.
 */
export function packParameters(
  program: ShaderProgramPayload,
  overrides?: Readonly<Record<string, readonly number[]>>,
  cullMargin: number = CULL_MARGIN_ON,
): Float32Array {
  const packed = new Float32Array(Math.max(program.buffer_bytes, PARAMETER_SLOT_BYTES) / 4);
  for (const slot of program.parameters) {
    const override = overrides?.[slot.name];
    const value = override && override.length === slot.components ? override : slot.value;
    for (let i = 0; i < slot.components; i += 1) {
      const component = value[i];
      // `null` is a value the scene never set; the literal form spells the
      // same slot as a NaN, so this one does too.
      packed[slot.offset / 4 + i] = component === null || component === undefined
        ? Number.NaN
        : component;
    }
  }
  // The reserved slots: the module reads its NaN from one and the margin
  // every bounding-box skip test is compared against from the other.
  if (program.nan_offset !== undefined) packed[program.nan_offset / 4] = Number.NaN;
  // `null` and absent both mean the program reserved no margin slot — a
  // literal-form shader, or one built before the toggle existed. Writing
  // anywhere on that basis would land on a real parameter.
  if (program.cull_margin_offset != null) {
    packed[program.cull_margin_offset / 4] = cullMargin;
  }
  return packed;
}

/** Whether two payloads describe the same buffer, slot for slot. */
export function sameLayout(
  a: ShaderProgramPayload | null | undefined,
  b: ShaderProgramPayload | null | undefined,
): boolean {
  if (!a || !b) return a === b || (!a && !b);
  if (a.group !== b.group || a.binding !== b.binding) return false;
  if (a.buffer_bytes !== b.buffer_bytes) return false;
  if (a.parameters.length !== b.parameters.length) return false;
  return a.parameters.every((slot, index) => {
    const other = b.parameters[index];
    return (
      slot.name === other.name &&
      slot.offset === other.offset &&
      slot.components === other.components
    );
  });
}

/**
 * Compiled shader modules, keyed by their own source.
 *
 * Bounded, because the keys *are* the sources and a scene's two shaders are
 * six megabytes of string: an unbounded cache over an afternoon of editing
 * would hold every intermediate shape the part passed through. Eviction is
 * least-recently-used, which is the right rule for the access pattern that
 * makes this worth having — undo/redo and a toggle flipped back walk a short
 * way into the recent past and no further.
 */
export class ShaderModuleCache {
  private entries = new Map<string, GPUShaderModule>();
  /** Modules served without touching the GPU. */
  hits = 0;
  /** Modules that had to be created and validated. */
  misses = 0;

  constructor(private readonly capacity = 8) {}

  /**
   * The module for this source, compiled if it is not already held.
   *
   * @param device The GPU device.
   * @param code WGSL source.
   * @param label Debug label, used only on a miss.
   * @returns The compiled module.
   * @throws If the source has compilation errors.
   */
  async get(device: GPUDevice, code: string, label: string): Promise<GPUShaderModule> {
    const held = this.entries.get(code);
    if (held) {
      // Re-insert so the map's iteration order stays least-recent-first.
      this.entries.delete(code);
      this.entries.set(code, held);
      this.hits += 1;
      return held;
    }
    this.misses += 1;
    const module = device.createShaderModule({ code, label });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) {
      throw new Error(
        errors.map((m) => `${label} ${m.lineNum}:${m.linePos} ${m.message}`).join("\n"),
      );
    }
    this.entries.set(code, module);
    while (this.entries.size > this.capacity) {
      const oldest = this.entries.keys().next();
      if (oldest.done) break;
      this.entries.delete(oldest.value);
    }
    return module;
  }

  /** How many modules are held. */
  get size(): number {
    return this.entries.size;
  }

  /** Forget everything, which a device loss requires. */
  clear(): void {
    this.entries.clear();
  }
}
