/**
 * The demo's parameter path: values → scene graph → uniform contract → buffer,
 * and values → construction tree, on the shipped bracket.
 */
import { beforeAll, describe, expect, it } from "vitest";
import { sceneProgram } from "../src/scene/shaders.ts";
import { loadScene } from "../src/static/bundle.ts";
import {
  acceptValues,
  bindingValues,
  constructionAt,
  constructionBounds,
  editedValues,
  graphAt,
  handleLock,
  shippedValues,
} from "../src/static/parameters.ts";
import type { DemoScene } from "../src/types";
import { packParameters } from "../src/viewer/shaderProgram.ts";

let bracket: DemoScene;
beforeAll(async () => {
  bracket = await loadScene("bracket");
});

const slot = (program: ReturnType<typeof sceneProgram>["program"], name: string) =>
  program.parameters.find((s) => s.name === name)!;

describe("values to the uniform buffer", () => {
  it("a free parameter's new value changes its slot and not the module", () => {
    const shipped = sceneProgram(bracket.scene_graph);
    const values = acceptValues(bracket.parameters, shippedValues(bracket.parameters), { plate_thickness: [0.3] });
    const edited = sceneProgram(graphAt(bracket.scene_graph, values));
    expect(edited.wgsl).toBe(shipped.wgsl);
    expect(edited.program.buffer_bytes).toBe(shipped.program.buffer_bytes);
    expect(slot(edited.program, "plate_thickness").value).toEqual([0.3]);
    expect(slot(edited.program, "web_thickness").value).toEqual(slot(shipped.program, "web_thickness").value);
  });

  it("a sketch vertex reaches the graph's vertex list and its slot", () => {
    const values = acceptValues(bracket.parameters, shippedValues(bracket.parameters), { plate_tr: [1.5, 0.9] });
    expect(slot(sceneProgram(graphAt(bracket.scene_graph, values)).program, "plate_tr").value).toEqual([1.5, 0.9]);
  });

  it("an override packs into the slot's floats, and opens the culling boxes built from it", () => {
    const { program } = sceneProgram(bracket.scene_graph);
    const packed = packParameters(program, { web_thickness: [0.25] });
    expect(packed[slot(program, "web_thickness").offset / 4]).toBeCloseTo(0.25, 6);
    const box = program.parameters.find((s) => s.name.endsWith(".half") && s.depends?.includes("web_thickness"))!;
    expect(packed[box.offset / 4]).toBeGreaterThan(1e6);
  });

  it("every listed parameter is a free slot of the shader the page builds", () => {
    const { program } = sceneProgram(bracket.scene_graph);
    for (const parameter of bracket.parameters) {
      const s = slot(program, parameter.name);
      expect(s?.free, parameter.name).toBe(true);
      expect(s.components).toBe(parameter.components);
    }
  });
});

describe("accepting values", () => {
  it("refuses a constraint-driven parameter and clamps to the range", () => {
    const shipped = shippedValues(bracket.parameters);
    const next = acceptValues(bracket.parameters, shipped, { rib_tip: [0, 0], plate_thickness: [99] });
    expect(next.rib_tip).toEqual(shipped.rib_tip);
    const thickness = bracket.parameters.find((p) => p.name === "plate_thickness")!;
    expect(next.plate_thickness).toEqual([thickness.upper[0]]);
    expect(Object.keys(editedValues(bracket.parameters, next))).toEqual(["plate_thickness"]);
  });

  it("ignores non-finite and wrongly sized proposals", () => {
    const shipped = shippedValues(bracket.parameters);
    const next = acceptValues(bracket.parameters, shipped, { plate_tr: [Number.NaN, 1], plate_bl: [1] });
    expect(next.plate_tr).toEqual(shipped.plate_tr);
    expect(next.plate_bl).toEqual(shipped.plate_bl);
  });

  it("splits a handle's value over its bindings", () => {
    expect(
      bindingValues(
        [
          { name: "rx", components: 1, index: 0 },
          { name: "rz", components: 1, index: 2 },
        ],
        [0.1, 0.2, 0.3],
      ),
    ).toEqual({ rx: [0.1], rz: [0.3] });
    expect(bindingValues([{ name: "p", components: 3, index: null }], [1, 2, 3])).toEqual({ p: [1, 2, 3] });
  });
});

describe("the construction tree at the values", () => {
  it("moves a bound vertex and the outline through it", () => {
    const values = { ...shippedValues(bracket.parameters), plate_tr: [1.5, 0.9] };
    const plate = constructionAt(bracket.construction, bracket.parameters, values).find((n) => n.name === "plate")!;
    const tr = plate.vertices.find((v) => v.name === "plate_tr")!;
    expect(tr.uv).toEqual([1.5, 0.9]);
    expect(tr.world[0]).toBeCloseTo(1.5, 6);
    expect(tr.world[1]).toBeCloseTo(0.9, 6);
    expect(plate.edges.some(([a, b]) => a === tr.world || b === tr.world)).toBe(true);
  });

  it("re-places a primitive whose dimensions are bound", () => {
    const values = { ...shippedValues(bracket.parameters), hole_left_radius: [0.32] };
    const shipped = bracket.construction.find((n) => n.name === "hole_left")!;
    const moved = constructionAt(bracket.construction, bracket.parameters, values).find((n) => n.name === "hole_left")!;
    expect(moved.transform!.dimensions.radius).toBeCloseTo(0.32, 6);
    const reach = (edges: typeof shipped.edges) =>
      Math.max(...edges.flat().map((p) => Math.hypot(p[0] - shipped.transform!.position[0], p[1] - shipped.transform!.position[1])));
    expect(reach(moved.edges)).toBeCloseTo(2 * reach(shipped.edges), 4);
  });

  it("draws a constraint-driven vertex as held, where it ships", () => {
    const nodes = constructionAt(bracket.construction, bracket.parameters, shippedValues(bracket.parameters));
    const rib = nodes.find((n) => n.name === "rib")!;
    expect(rib.vertices.find((v) => v.name === "rib_tip")!.free).toBe(false);
    expect(rib.vertices.find((v) => v.name === "rib_plate_corner")!.free).toBe(true);
  });

  it("bounds the tree with a margin", () => {
    const bounds = constructionBounds(bracket.construction, 0)!;
    expect(bounds.half[0]).toBeCloseTo(1.3, 5);
    expect(constructionBounds([])).toBeNull();
  });
});

describe("which handles drag", () => {
  const program = () => sceneProgram(bracket.scene_graph).program;
  const node = (name: string) => bracket.construction.find((n) => n.name === name);

  it("a free vertex drags; a held one says what holds it", () => {
    const rib = node("rib")!;
    const index = (name: string) => rib.vertices.findIndex((v) => v.name === name);
    expect(handleLock(rib, index("rib_plate_corner"), bracket.parameters, program())).toBeNull();
    expect(handleLock(rib, index("rib_tip"), bracket.parameters, program())).toMatch(/DistanceConstraint/);
  });

  it("a gizmo argument drags only when every binding behind it is free and in the shader", () => {
    expect(handleLock(node("hole_left"), "radius", bracket.parameters, program())).toBeNull();
    expect(handleLock(node("hole_right"), "position", bracket.parameters, program())).toMatch(/held by/);
    // An identity rotation folds away, so its angles have no slot.
    expect(handleLock(node("hole_left"), "rotation", bracket.parameters, program())).toMatch(/not in this scene's shader/);
    expect(handleLock(node("hole_left"), "position", bracket.parameters, null)).toMatch(/not in this scene's shader/);
  });
});
