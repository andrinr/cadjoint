/**
 * Which handles drag live, and which cost a recompile.
 *
 * The classification is the whole decision the drag path makes and the whole
 * meaning of the filled-versus-hollow handle in the viewport, so it is worth
 * pinning on its own: it is pure, it has three outcomes, and two of them are
 * easy to conflate.
 *
 * - `free` — the payload names a parameter and the installed shader has its
 *   slot. The pointer move is a `writeBuffer`.
 * - `fixed` — the payload names nothing, because the value is a literal in
 *   the program text. A recompile, by design.
 * - `unbound` — the payload names a parameter the shader does *not* have. It
 *   is not a mistake and it is not rare: a primitive at rest has three free
 *   rotation parameters, and an identity rotation is never built into the SDF
 *   at all, so all three fold away. It has to read as a recompile, and it has
 *   to be told apart from `fixed`, because the source really does name them.
 *
 * The tests below also pin the refusal that matters most: a value whose
 * components are only *partly* bound writes nothing. Half a transform in the
 * buffer would draw a solid the release is not about to produce.
 */

import { describe, expect, it } from "vitest";
import {
  argumentValue,
  bindingState,
  overridesFor,
  transformState,
  vertexState,
} from "../src/viewer/dragBinding";
import { HANDLE_STRIDE, packConstructionOverlay } from "../src/viewer/overlayGeometry";
import type { ShaderProgramPayload } from "../src/viewer/shaderProgram";
import type { ConstructionNode, ConstructionTransform, ConstructionVertex } from "../src/types";

/** A program in the shape the starter compiles to: a point, a vector, a scalar. */
const program: ShaderProgramPayload = {
  group: 3,
  binding: 0,
  buffer_bytes: 64,
  parameters: [
    { name: "fin1_tip_r", offset: 0, components: 2, value: [0.75, 0.85], free: true },
    { name: "bushing_a", offset: 16, components: 3, value: [0.78, 0, 0.1], free: true },
    { name: "head_a_radius", offset: 32, components: 1, value: [0.062], free: true },
    { name: "board_rx", offset: 48, components: 1, value: [0], free: true },
  ],
};

const vertex = (binding: ConstructionVertex["binding"]): Pick<ConstructionVertex, "binding"> => ({
  binding,
});

const transform = (
  bindings: ConstructionTransform["bindings"],
): Pick<ConstructionTransform, "bindings"> => ({ bindings });

describe("bindingState", () => {
  it("calls a value with no binding fixed", () => {
    expect(bindingState(null, program)).toBe("fixed");
    expect(bindingState([], program)).toBe("fixed");
  });

  it("calls a bound value the shader has free", () => {
    expect(bindingState([{ name: "bushing_a", components: 3 }], program)).toBe("free");
  });

  it("calls a bound value the shader dropped unbound", () => {
    expect(bindingState([{ name: "board_ry", components: 1 }], program)).toBe("unbound");
  });

  it("does not accept a slot of a different width under the same name", () => {
    expect(bindingState([{ name: "bushing_a", components: 2 }], program)).toBe("unbound");
  });

  it("is unbound in the literal form, where there is no program at all", () => {
    expect(bindingState([{ name: "bushing_a", components: 3 }], null)).toBe("unbound");
  });

  it("refuses a value only some of whose components are bound", () => {
    const rotation = [
      { name: "board_rx", components: 1, index: 0 },
      { name: "board_ry", components: 1, index: 1 },
      { name: "board_rz", components: 1, index: 2 },
    ];
    expect(bindingState(rotation, program)).toBe("unbound");
  });
});

describe("vertexState", () => {
  it("reads a free sketch point off its binding", () => {
    expect(vertexState(vertex({ name: "fin1_tip_r", components: 2 }), program)).toBe("free");
  });

  it("reads a literal sketch point as fixed", () => {
    expect(vertexState(vertex(null), program)).toBe("fixed");
    expect(vertexState(vertex(undefined), program)).toBe("fixed");
  });

  it("reads a named point the shader never used as unbound", () => {
    expect(vertexState(vertex({ name: "slug_rim_low", components: 2 }), program)).toBe("unbound");
  });
});

describe("transformState", () => {
  const bindings = {
    position: [{ name: "bushing_a", components: 3 }],
    radius: [{ name: "head_a_radius", components: 1 }],
    rotation: [
      { name: "board_rx", components: 1, index: 0 },
      { name: "board_ry", components: 1, index: 1 },
      { name: "board_rz", components: 1, index: 2 },
    ],
  };

  it("classifies each argument on its own", () => {
    expect(transformState(transform(bindings), "position", program)).toBe("free");
    expect(transformState(transform(bindings), "radius", program)).toBe("free");
    expect(transformState(transform(bindings), "rotation", program)).toBe("unbound");
  });

  it("calls an argument the payload does not mention fixed", () => {
    expect(transformState(transform(bindings), "height", program)).toBe("fixed");
    expect(transformState(transform({}), "position", program)).toBe("fixed");
    // A sketch plane's origin: a transform that binds nothing at all.
    expect(transformState(null, "origin", program)).toBe("fixed");
  });
});

describe("overridesFor", () => {
  it("writes a whole vector under one name", () => {
    expect(
      overridesFor([{ name: "bushing_a", components: 3 }], [0.5, 0.1, 0.2], program),
    ).toEqual({ bushing_a: [0.5, 0.1, 0.2] });
  });

  it("takes only the components a parameter covers", () => {
    // A sketch vertex is dragged in plane coordinates; the trailing world
    // coordinate a caller might pass is not part of the parameter.
    expect(
      overridesFor([{ name: "fin1_tip_r", components: 2 }], [0.7, 0.9, 0.0], program),
    ).toEqual({ fin1_tip_r: [0.7, 0.9] });
  });

  it("indexes into the value when a parameter drives one component", () => {
    expect(
      overridesFor([{ name: "board_rx", components: 1, index: 0 }], [1.5, 0, 0], program),
    ).toEqual({ board_rx: [1.5] });
  });

  it("returns null rather than writing part of a value", () => {
    const rotation = [
      { name: "board_rx", components: 1, index: 0 },
      { name: "board_ry", components: 1, index: 1 },
    ];
    expect(overridesFor(rotation, [0.2, 0.3, 0.0], program)).toBeNull();
  });

  it("returns null for an unbound value, a fixed one, and no program", () => {
    expect(overridesFor(null, [1, 2, 3], program)).toBeNull();
    expect(overridesFor([{ name: "nope", components: 3 }], [1, 2, 3], program)).toBeNull();
    expect(overridesFor([{ name: "bushing_a", components: 3 }], [1, 2, 3], null)).toBeNull();
  });

  it("refuses a value that is too short for its slot", () => {
    expect(overridesFor([{ name: "bushing_a", components: 3 }], [1, 2], program)).toBeNull();
  });

  it("refuses a non-finite component rather than poisoning the buffer", () => {
    expect(
      overridesFor([{ name: "bushing_a", components: 3 }], [1, Number.NaN, 3], program),
    ).toBeNull();
  });
});

describe("argumentValue", () => {
  it("flattens a scalar dimension and copies a vector one", () => {
    expect(argumentValue(0.5)).toEqual([0.5]);
    expect(argumentValue([1, 2, 3])).toEqual([1, 2, 3]);
    expect(argumentValue(undefined)).toBeNull();
  });
});

/**
 * The mark and the decision are the same function.
 *
 * The handle instance's last float is what the fragment shader fills or
 * hollows, and the drag path asks `vertexState` whether to write the buffer.
 * If those two ever came from different code the viewport could promise a
 * live drag it does not give, so this asserts they are one answer.
 */
describe("packConstructionOverlay marks handles", () => {
  const node = (vertices: ConstructionVertex["binding"][]) =>
    ({
      id: "profile_0",
      kind: "profile",
      editable: true,
      edges: [],
      faces: [],
      constraints: [],
      operators: [],
      vertices: vertices.map((binding, index) => ({
        stableId: null,
        name: `v${index}`,
        free: binding !== null,
        uv: [0, 0],
        world: [index, 0, 0],
        span: null,
        binding,
      })),
    }) as unknown as ConstructionNode;

  it("fills exactly the handles a drag can write live", () => {
    const profile = node([
      { name: "fin1_tip_r", components: 2 },
      null,
      { name: "slug_rim_low", components: 2 },
    ]);
    const { handles } = packConstructionOverlay([profile], null, null, program);
    const floats = HANDLE_STRIDE / 4;
    expect(handles.length).toBe(3 * floats);
    const fill = [0, 1, 2].map((index) => handles[index * floats + floats - 1]);
    expect(fill).toEqual([1, 0, 0]);
    // …and it is the same answer the drag path acts on.
    expect(
      profile.vertices.map((vertex) => (vertexState(vertex, program) === "free" ? 1 : 0)),
    ).toEqual(fill);
  });

  it("hollows every handle in the literal form, where nothing is live", () => {
    const profile = node([{ name: "fin1_tip_r", components: 2 }]);
    const { handles } = packConstructionOverlay([profile], null, null, null);
    expect(handles[HANDLE_STRIDE / 4 - 1]).toBe(0);
  });
});
