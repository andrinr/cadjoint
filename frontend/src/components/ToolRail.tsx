/**
 * Vertical tool rail, the way modelling applications arrange their tools.
 *
 * Grouped top to bottom: what a click selects, what a click creates, and how a
 * selection is transformed. Icons carry the meaning; every button keeps a
 * tooltip with its keyboard shortcut.
 */

import { Dynamic } from "solid-js/web";
import { For, Show, type Component } from "solid-js";
import {
  gizmoMode,
  selection,
  selectionMode,
  setGizmoMode,
  setSelectionMode,
  setTool,
  tool,
} from "../state";
import type { GizmoMode, SelectionMode, ToolMode } from "../types";
import {
  BoxIcon,
  CylinderIcon,
  MoveIcon,
  ObjectSelectIcon,
  PolygonIcon,
  RotateIcon,
  SphereIcon,
  TrashIcon,
  VertexSelectIcon,
} from "./icons";

const MODES: { key: SelectionMode; label: string; hint: string; icon: Component }[] = [
  { key: "object", label: "Object", hint: "Select whole objects  (1)", icon: ObjectSelectIcon },
  { key: "vertex", label: "Vertex", hint: "Select sketch vertices  (2)", icon: VertexSelectIcon },
];

const CREATE: { key: ToolMode; label: string; hint: string; icon: Component }[] = [
  { key: "polygon", label: "Polygon", hint: "Add sketch vertices  (P)", icon: PolygonIcon },
  { key: "box", label: "Box", hint: "Place a box  (B)", icon: BoxIcon },
  { key: "sphere", label: "Sphere", hint: "Place a sphere  (S)", icon: SphereIcon },
  { key: "cylinder", label: "Cylinder", hint: "Place a cylinder  (C)", icon: CylinderIcon },
];

const TRANSFORMS: { key: GizmoMode; label: string; hint: string; icon: Component }[] = [
  { key: "translate", label: "Move", hint: "Move along an axis  (G)", icon: MoveIcon },
  { key: "rotate", label: "Rotate", hint: "Rotate about an axis  (R)", icon: RotateIcon },
];

export interface ToolRailProps {
  onDelete: () => void;
}

export function ToolRail(props: ToolRailProps) {
  /** A whole object is selected, so the transform gizmo applies. */
  const transformable = () => selection() !== null && selection()!.vertexIndex === null;

  return (
    <nav class="tool-rail" aria-label="Tools">
      <For each={MODES}>
        {(mode) => (
          <button
            type="button"
            class={tool() === "select" && selectionMode() === mode.key ? "active" : ""}
            onClick={() => {
              setTool("select");
              setSelectionMode(mode.key);
            }}
            title={mode.hint}
            aria-label={mode.label}
            data-testid={`mode-${mode.key}`}
          >
            <Dynamic component={mode.icon} />
          </button>
        )}
      </For>

      <hr />

      <For each={CREATE}>
        {(entry) => (
          <button
            type="button"
            class={tool() === entry.key ? "active" : ""}
            onClick={() => setTool(tool() === entry.key ? "select" : entry.key)}
            title={entry.hint}
            aria-label={entry.label}
            data-testid={`tool-${entry.key}`}
          >
            <Dynamic component={entry.icon} />
          </button>
        )}
      </For>

      <hr />

      <For each={TRANSFORMS}>
        {(entry) => (
          <button
            type="button"
            class={transformable() && gizmoMode() === entry.key ? "active" : ""}
            disabled={!transformable()}
            onClick={() => setGizmoMode(entry.key)}
            title={
              transformable() ? entry.hint : `${entry.label} — select an object first`
            }
            aria-label={entry.label}
            data-testid={`gizmo-${entry.key}`}
          >
            <Dynamic component={entry.icon} />
          </button>
        )}
      </For>

      <hr />

      <button
        type="button"
        class="danger"
        disabled={selection() === null}
        onClick={props.onDelete}
        title="Delete the selection  (Del)"
        aria-label="Delete"
        data-testid="delete-selection"
      >
        <TrashIcon />
      </button>

      <Show when={transformable()}>
        <span class="rail-hint" data-testid="rail-hint">
          {gizmoMode() === "translate" ? "move" : "turn"}
        </span>
      </Show>
    </nav>
  );
}
