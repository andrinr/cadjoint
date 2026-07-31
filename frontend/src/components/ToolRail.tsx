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
  nodeById,
  selection,
  selectionMode,
  setGizmoMode,
  setSelection,
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
  PointAddIcon,
  PolygonIcon,
  RotateIcon,
  ScaleIcon,
  SphereIcon,
  TrashIcon,
  VertexSelectIcon,
} from "./icons";

const MODES: { key: SelectionMode; label: string; hint: string; icon: Component }[] = [
  { key: "object", label: "Object", hint: "Select whole objects  (1)", icon: ObjectSelectIcon },
  { key: "vertex", label: "Vertex", hint: "Select sketch vertices  (2)", icon: VertexSelectIcon },
];

const CREATE: { key: ToolMode; label: string; hint: string; icon: Component }[] = [
  { key: "sketch", label: "Sketch", hint: "Place a new 2D sketch", icon: PolygonIcon },
  { key: "polygon", label: "Point", hint: "Add a point to a sketch  (P)", icon: PointAddIcon },
  { key: "box", label: "Box", hint: "Place a box  (B)", icon: BoxIcon },
  { key: "sphere", label: "Sphere", hint: "Place a sphere  (S)", icon: SphereIcon },
  { key: "cylinder", label: "Cylinder", hint: "Place a cylinder  (C)", icon: CylinderIcon },
];

const TRANSFORMS: { key: GizmoMode; label: string; hint: string; icon: Component }[] = [
  { key: "translate", label: "Move", hint: "Move along an axis  (G)", icon: MoveIcon },
  { key: "rotate", label: "Rotate", hint: "Rotate about an axis  (R)", icon: RotateIcon },
  { key: "scale", label: "Scale", hint: "Scale along an axis", icon: ScaleIcon },
];

export interface ToolRailProps {
  onDelete: () => void;
}

export function ToolRail(props: ToolRailProps) {
  /** A whole object is selected, so the transform gizmo applies. */
  const transformable = () => selection() !== null && selection()!.vertexIndex === null;
  const supports = (mode: GizmoMode) => {
    if (!transformable()) return false;
    const node = nodeById(selection()!.nodeId);
    if (!node?.transform) return false;
    if (mode === "rotate") return node.transform.canRotate;
    if (mode === "scale") return node.kind !== "profile";
    return true;
  };
  const activeTransform = () => (supports(gizmoMode()) ? gizmoMode() : "translate");

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
              const active = selection();
              if (mode.key === "object" && active && active.vertexIndex !== null) {
                setSelection({ nodeId: active.nodeId, vertexIndex: null });
                if (nodeById(active.nodeId)?.kind === "profile") {
                  setGizmoMode("translate");
                }
              }
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
            class={supports(entry.key) && gizmoMode() === entry.key ? "active" : ""}
            disabled={!supports(entry.key)}
            onClick={() => setGizmoMode(entry.key)}
            title={
              supports(entry.key)
                ? entry.hint
                : entry.key === "scale" && transformable()
                  ? "Scale — select a solid primitive"
                  : `${entry.label} — select a compatible object first`
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
          {activeTransform() === "translate"
            ? "move"
            : activeTransform() === "rotate"
              ? "turn"
              : "size"}
        </span>
      </Show>
    </nav>
  );
}
