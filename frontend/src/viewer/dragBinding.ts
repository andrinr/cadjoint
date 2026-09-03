/**
 * Joining a handle to the uniform slot behind it.
 *
 * The compile payload arrives in two halves that describe the same numbers.
 * `ShaderProgram` says where each *free* design parameter sits in the
 * `@group(3)` buffer the scene's shaders read; the construction payload says
 * which handle moves which value, and — since the parameter bindings landed —
 * which named parameter backs it. Put the two together and a drag can answer
 * a pointer move with a `writeBuffer` instead of a source rewrite, a worker
 * round trip and a pipeline rebuild.
 *
 * The join is deliberately suspicious of itself. A binding is a claim about
 * the *source* ("this value is the free parameter `fin_depth`"), and the
 * program is the truth about the *shader*: a parameter can be free, named and
 * still have no slot, because the SDF never used it. A primitive at rest is
 * the everyday case — an identity rotation is not built into the tree at all,
 * so its three angle parameters fold away and eight of the starter's handles
 * are "declared but absent". So every name is checked against the slot table,
 * with its width, before anything is written.
 *
 * Three states come out of that, and the viewport draws the first differently
 * from the other two:
 *
 * - `free` — bound, and the shader has the slot: the drag is live.
 * - `fixed` — no binding: a literal in the source, and moving it is a
 *   recompile by design.
 * - `unbound` — bound, but the shader has no such slot (or is in the literal
 *   form entirely): also a recompile, and worth telling apart from `fixed`
 *   because the source *does* name a parameter.
 *
 * Everything here is pure: payloads in, plain values out. The renderer owns
 * the buffer; this module only decides what to put in it.
 */

import type { ConstructionTransform, ConstructionVertex, ParameterBinding } from "../types";
import type { ShaderProgramPayload } from "./shaderProgram";

/** Whether a value can be dragged through the uniform buffer, and why not. */
export type BindingState = "free" | "fixed" | "unbound";

/** Parameter values to write, keyed by name — what `setParameterOverrides` takes. */
export type ParameterOverrides = Record<string, number[]>;

/** Whether the program really has this slot, at this width. */
function hasSlot(program: ShaderProgramPayload | null, binding: ParameterBinding): boolean {
  return (program?.parameters ?? []).some(
    (slot) => slot.name === binding.name && slot.components === binding.components,
  );
}

/**
 * How a value's bindings stand against the installed shader.
 *
 * All or nothing: one component whose parameter the shader dropped makes the
 * whole value `unbound`, because writing the rest would leave the image
 * disagreeing with the source it is supposed to be previewing.
 */
export function bindingState(
  bindings: readonly ParameterBinding[] | null | undefined,
  program: ShaderProgramPayload | null,
): BindingState {
  if (!bindings || bindings.length === 0) return "fixed";
  return bindings.every((binding) => hasSlot(program, binding)) ? "free" : "unbound";
}

/** How a sketch vertex handle stands: `free` means dragging it is live. */
export function vertexState(
  vertex: Pick<ConstructionVertex, "binding">,
  program: ShaderProgramPayload | null,
): BindingState {
  return bindingState(vertex.binding ? [vertex.binding] : null, program);
}

/** How one of a primitive's gizmo arguments stands — `position`, `radius`, … */
export function transformState(
  transform: Pick<ConstructionTransform, "bindings"> | null | undefined,
  argument: string,
  program: ShaderProgramPayload | null,
): BindingState {
  return bindingState(transform?.bindings?.[argument], program);
}

/**
 * The buffer writes that show this value now, or null to take the slow path.
 *
 * @param bindings The parameters covering the value, in component order.
 * @param value The dragged value: a vertex's `[u, v]`, a position's `[x, y, z]`,
 *   a radius's `[r]`.
 * @param program The installed uniform contract.
 * @returns Overrides by parameter name, or null when anything at all does not
 *   line up — an unbound name, a width that disagrees, a non-finite number.
 *   Null is the caller's instruction to emit a patch and wait, not a hint.
 */
export function overridesFor(
  bindings: readonly ParameterBinding[] | null | undefined,
  value: readonly number[],
  program: ShaderProgramPayload | null,
): ParameterOverrides | null {
  if (bindingState(bindings, program) !== "free") return null;
  const overrides: ParameterOverrides = {};
  for (const binding of bindings!) {
    const slice =
      binding.index === null || binding.index === undefined
        ? value.slice(0, binding.components)
        : [value[binding.index]];
    if (slice.length !== binding.components) return null;
    if (slice.some((component) => !Number.isFinite(component))) return null;
    overrides[binding.name] = slice as number[];
  }
  return overrides;
}

/**
 * The value a gizmo drag writes for one argument, flattened to components.
 *
 * `position` and `rotation` are already triples; a dimension is either a
 * triple (a box's `size`) or a single number (a radius, a height).
 */
export function argumentValue(
  value: number | readonly number[] | undefined,
): number[] | null {
  if (typeof value === "number") return [value];
  if (Array.isArray(value)) return [...value];
  return null;
}
