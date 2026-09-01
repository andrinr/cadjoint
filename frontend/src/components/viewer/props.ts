/**
 * The viewer pane's contract with the app shell.
 *
 * Every edit the viewport can make travels back out through one of these
 * callbacks and lands in the Python source; the pane never mutates a private
 * scene graph. The interaction modules (tools, keyboard) act on the same
 * object, so the contract lives here rather than inside the component that
 * happens to render it.
 */

import type { JSX } from "solid-js";
import type { ConstraintKind } from "../../types";
import type { DisplaySettings } from "../../viewer/renderer";
import type { Renderer } from "../../viewer/renderer";

export interface ViewerPaneProps {
  renderer: Renderer;
  /** View-only settings, including construction annotation visibility. */
  display: DisplaySettings;
  /** Overlays rendered on top of the canvas (tool rail, ViewCube). */
  overlay?: JSX.Element;
  /** Apply a patch operation to the source and recompile. */
  onPatch: (
    op: "set_vertex" | "insert_vertex" | "delete_vertex",
    line: number,
    index: number,
    xy?: [number, number],
  ) => Promise<void>;
  /** Rewrite a primitive's placement keyword. */
  onSetValue: (
    line: number,
    name: string,
    argument: string,
    value: number | number[],
  ) => Promise<void>;
  /** Insert a new solid into the program. */
  onAddPrimitive: (
    kind: string,
    position: [number, number, number],
    dimensions: Record<string, number | number[]>,
  ) => Promise<void>;
  /** Insert a standalone sketch at a world-space origin. */
  onAddSketch: (origin: [number, number, number]) => Promise<void>;
  /** Attach a source-level constraint to sketch vertices. */
  onAddConstraint: (
    line: number,
    kind: ConstraintKind,
    indices: number[],
    value?: number | number[],
  ) => Promise<void>;
  /** Loft two named sketches, identified by their source lines. */
  onAddLoft: (lineA: number, lineB: number) => Promise<void>;
  /** Remove a whole construction object from the program. */
  onDeleteObject: (line: number) => Promise<void>;
  /** Assign a named Python material to the object under a drop. */
  onAssignMaterial: (line: number, material: string) => Promise<void>;
}
