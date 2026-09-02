/**
 * The named edits the panes can make to the program.
 *
 * Every viewer and panel action is one `/patch` operation, and the op names
 * and their payload keys are the app's half of that contract. Collecting the
 * wrappers here keeps the shell's JSX free of request literals and gives the
 * contract one place to read: if the server renames an argument, it changes
 * on one line.
 *
 * Each wrapper returns the queued promise from `applyPatch`, so callers that
 * need to sequence edits (place a sketch, then set its normal) can await it.
 * Nothing here touches the network directly.
 *
 * How an operation names its target
 * ---------------------------------
 * Every addressable request takes either `id` — the payload entry's stable
 * id — or `line`, and the server resolves whichever it is given. A line is a
 * position, and inserting a statement renumbers every position after it, so
 * two edits queued back to back can disagree about what "line 91" meant. A
 * stable id does not move. Both are sent (`addressing`): the id is the real
 * address, the line is the fallback for an entry the identity table could not
 * name, and neither call sites nor signatures had to change to get it —
 * the shell already knows which node lives at the line it was handed.
 */

import { nodeById, nodes, selection } from "../state";
import type { ConstraintKind, SketchPlaneReference } from "../types";

export type VertexPatchOp = "set_vertex" | "insert_vertex" | "delete_vertex";

export interface PatchOperations {
  patch: (op: VertexPatchOp, line: number, index: number, xy?: [number, number]) => Promise<void>;
  setValue: (
    line: number,
    name: string,
    argument: string,
    value: number | number[],
  ) => Promise<void>;
  deleteObject: (line: number) => Promise<void>;
  addPrimitive: (
    kind: string,
    position: [number, number, number],
    dimensions: Record<string, number | number[]>,
  ) => Promise<void>;
  addMaterial: () => Promise<void>;
  assignMaterial: (line: number, material: string) => Promise<void>;
  addSketch: (origin: [number, number, number]) => Promise<void>;
  /** Re-plant an existing sketch on a face, or on a tangent plane. */
  setSketchPlane: (line: number, reference: SketchPlaneReference) => Promise<void>;
  addConstraint: (
    line: number,
    kind: ConstraintKind,
    indices: number[],
    value?: number | number[],
  ) => Promise<void>;
  deleteConstraint: (line: number, index: number) => Promise<void>;
  setConstraintValue: (line: number, index: number, value: number) => Promise<void>;
  addExtrusion: (line: number) => Promise<void>;
  addRevolution: (line: number) => Promise<void>;
  addLoft: (lineA: number, lineB: number) => Promise<void>;
  solveSketch: (
    line: number,
    method: "newton" | "adam" | "sgd",
    iterations: number,
  ) => Promise<void>;
  /** Extrude the selected sketch — shared by the rail and the sketch panel. */
  extrudeSelection: () => void;
  revolveSelection: () => void;
}

export function createPatchOperations(
  applyPatch: (body: Record<string, unknown>) => Promise<void>,
): PatchOperations {
  /** Name the construction entry declared at *line*, by id and by line. */
  const addressing = (line: number): { id?: string; line: number } => {
    const stable = nodes().find((node) => node.line === line)?.stableId;
    return stable ? { id: stable, line } : { line };
  };

  const patch = (
    op: VertexPatchOp,
    line: number,
    index: number,
    xy?: [number, number],
  ) => applyPatch({ op, ...addressing(line), index, xy });

  const setValue = (
    line: number,
    name: string,
    argument: string,
    value: number | number[],
  ) => applyPatch({ op: "set_value", ...addressing(line), name, argument, value });

  const deleteObject = (line: number) => applyPatch({ op: "delete_object", ...addressing(line) });

  const addPrimitive = (
    kind: string,
    position: [number, number, number],
    dimensions: Record<string, number | number[]>,
  ) => applyPatch({ op: "add_primitive", kind, position, dimensions });

  const addMaterial = () =>
    applyPatch({
      op: "add_material",
      color: [0.32, 0.72, 0.86],
      roughness: 0.35,
      metallic: 0,
      opacity: 1,
      ior: 1.45,
      reflectivity: 0,
    });

  const assignMaterial = (line: number, material: string) =>
    applyPatch({ op: "assign_material", ...addressing(line), material });

  const addSketch = (origin: [number, number, number]) =>
    applyPatch({ op: "add_sketch", origin });

  const setSketchPlane = (line: number, reference: SketchPlaneReference) =>
    applyPatch({ op: "set_sketch_plane", ...addressing(line), reference });

  const addConstraint = (
    line: number,
    kind: ConstraintKind,
    indices: number[],
    value?: number | number[],
  ) => applyPatch({ op: "add_constraint", ...addressing(line), kind, indices, value });

  const deleteConstraint = (line: number, index: number) =>
    applyPatch({ op: "delete_constraint", ...addressing(line), index });

  const setConstraintValue = (line: number, index: number, value: number) =>
    applyPatch({ op: "set_constraint_value", ...addressing(line), index, value });

  const addExtrusion = (line: number) =>
    applyPatch({ op: "add_extrusion", ...addressing(line), depth: 0.5 });

  const addRevolution = (line: number) =>
    applyPatch({ op: "add_revolution", ...addressing(line), offset: 0 });

  /** Extrude the selected sketch — shared by the rail and the sketch panel. */
  const extrudeSelection = () => {
    const active = selection();
    const node = active && nodeById(active.nodeId);
    if (node?.kind === "profile" && node.line !== null) {
      void addExtrusion(node.line);
    }
  };

  const revolveSelection = () => {
    const active = selection();
    const node = active && nodeById(active.nodeId);
    if (node?.kind === "profile" && node.line !== null) {
      void addRevolution(node.line);
    }
  };

  const addLoft = (lineA: number, lineB: number) =>
    applyPatch({
      op: "add_loft",
      id_a: addressing(lineA).id,
      id_b: addressing(lineB).id,
      line_a: lineA,
      line_b: lineB,
      height: 1.0,
    });

  const solveSketch = (
    line: number,
    method: "newton" | "adam" | "sgd",
    iterations: number,
  ) => applyPatch({ op: "solve_sketch", ...addressing(line), method, iterations });

  return {
    patch,
    setValue,
    deleteObject,
    addPrimitive,
    addMaterial,
    assignMaterial,
    addSketch,
    setSketchPlane,
    addConstraint,
    deleteConstraint,
    setConstraintValue,
    addExtrusion,
    addRevolution,
    addLoft,
    solveSketch,
    extrudeSelection,
    revolveSelection,
  };
}
