/**
 * Vertical tool rail: mode switcher on top, grouped tool clusters below.
 *
 * The switcher scopes the rail to an editing mode (model / sketch /
 * simulate), the way Blender's mode dropdown scopes its tool set. Tools sit
 * in expandable clusters: a parent icon shows the group's last-used child
 * and expands — on hover dwell or click — into a horizontal flyout of the
 * children, like Photoshop's nested tools. Flyouts close on selection,
 * Escape, or a mouse-leave delay; the last-used choice persists.
 *
 * The open/close rules live in `../flyout`, the mode filtering in
 * `../editingMode` — both pure and unit tested.
 */

import { Dynamic } from "solid-js/web";
import { For, Show, createEffect, createSignal, onCleanup, onMount, type Component } from "solid-js";
import {
  busy,
  editingMode,
  gizmoMode,
  nodeById,
  pendingLoft,
  selection,
  selectionMode,
  setEditingMode,
  setGizmoMode,
  setPendingLoft,
  setSelection,
  setSelectionMode,
  setSketchPlane,
  setTool,
  sketchPlane,
  tool,
} from "../state";
import {
  createToolVisibleInMode,
  groupVisibleInMode,
  type EditingMode,
} from "../editingMode";
import { FlyoutController, loadLastUsed, persistLastUsed } from "../flyout";
import { SKETCH_PLANE_CHOICES } from "../sketchPlanes";
import { isEdgeConstraintTool } from "../constraints";
import type { GizmoMode, SelectionMode, ToolMode } from "../types";
import {
  BoxIcon,
  CoincidentIcon,
  CylinderIcon,
  DistanceIcon,
  ExtrudeIcon,
  HorizontalIcon,
  LoftIcon,
  ModelModeIcon,
  MoveIcon,
  ObjectSelectIcon,
  ParallelIcon,
  PerpendicularIcon,
  PointAddIcon,
  PolygonIcon,
  RevolveIcon,
  RotateIcon,
  ScaleIcon,
  SimulateModeIcon,
  SketchModeIcon,
  SphereIcon,
  TrashIcon,
  VertexSelectIcon,
  VerticalIcon,
} from "./icons";

const EDIT_MODES: { key: EditingMode; label: string; hint: string; icon: Component }[] = [
  { key: "model", label: "Model", hint: "Model mode — solids and transforms  (M cycles)", icon: ModelModeIcon },
  { key: "sketch", label: "Sketch", hint: "Sketch mode — 2D profiles and constraints  (M cycles)", icon: SketchModeIcon },
  { key: "simulate", label: "Simulate", hint: "Simulate mode — FEM setup  (M cycles)", icon: SimulateModeIcon },
];

const SELECT_MODES: { key: SelectionMode; label: string; hint: string; icon: Component }[] = [
  { key: "object", label: "Object", hint: "Select whole objects  (1)", icon: ObjectSelectIcon },
  { key: "vertex", label: "Vertex", hint: "Select sketch vertices  (2)", icon: VertexSelectIcon },
];

interface ToolEntry {
  key: string;
  label: string;
  hint: string;
  icon: Component;
  testid: string;
}

const CREATE: ToolEntry[] = [
  { key: "sketch", label: "Sketch", hint: "Place a new 2D sketch", icon: PolygonIcon, testid: "tool-sketch" },
  { key: "polygon", label: "Point", hint: "Add a point to a sketch  (P)", icon: PointAddIcon, testid: "tool-polygon" },
  { key: "box", label: "Box", hint: "Place a box  (B)", icon: BoxIcon, testid: "tool-box" },
  { key: "sphere", label: "Sphere", hint: "Place a sphere  (S)", icon: SphereIcon, testid: "tool-sphere" },
  { key: "cylinder", label: "Cylinder", hint: "Place a cylinder  (C)", icon: CylinderIcon, testid: "tool-cylinder" },
];

const MODIFY: ToolEntry[] = [
  { key: "extrude", label: "Extrude", hint: "Extrude the selected sketch", icon: ExtrudeIcon, testid: "modify-extrude" },
  { key: "revolve", label: "Revolve", hint: "Revolve the selected sketch", icon: RevolveIcon, testid: "modify-revolve" },
  { key: "loft", label: "Loft", hint: "Loft the selected sketch to a second one", icon: LoftIcon, testid: "modify-loft" },
];

const TRANSFORMS: ToolEntry[] = [
  { key: "translate", label: "Move", hint: "Move along an axis  (G)", icon: MoveIcon, testid: "gizmo-translate" },
  { key: "rotate", label: "Rotate", hint: "Rotate about an axis  (R)", icon: RotateIcon, testid: "gizmo-rotate" },
  { key: "scale", label: "Scale", hint: "Scale along an axis", icon: ScaleIcon, testid: "gizmo-scale" },
];

const ANNOTATE: ToolEntry[] = [
  { key: "distance", label: "Distance", hint: "Choose two points to preserve their distance", icon: DistanceIcon, testid: "annotate-distance" },
  { key: "horizontal", label: "Horizontal", hint: "Choose two points to keep level", icon: HorizontalIcon, testid: "annotate-horizontal" },
  { key: "vertical", label: "Vertical", hint: "Choose two points to keep stacked", icon: VerticalIcon, testid: "annotate-vertical" },
  { key: "coincident", label: "Coincident", hint: "Choose two points to join", icon: CoincidentIcon, testid: "annotate-coincident" },
  { key: "parallel", label: "Parallel", hint: "Choose two edges to keep parallel", icon: ParallelIcon, testid: "annotate-parallel" },
  { key: "perpendicular", label: "Perpendicular", hint: "Choose two edges to keep perpendicular", icon: PerpendicularIcon, testid: "annotate-perpendicular" },
];

/** Known children per group, for validating persisted last-used entries. */
const GROUP_CHILDREN: Record<string, string[]> = {
  create: CREATE.map((entry) => entry.key),
  modify: MODIFY.map((entry) => entry.key),
  transform: TRANSFORMS.map((entry) => entry.key),
  annotate: ANNOTATE.map((entry) => entry.key),
};

export interface ToolRailProps {
  onDelete: () => void;
  /** Extrude the selected sketch (App resolves the profile and patches). */
  onExtrude: () => void;
  onRevolve: () => void;
}

export function ToolRail(props: ToolRailProps) {
  const [openGroup, setOpenGroup] = createSignal<string | null>(null);
  const flyouts = new FlyoutController(setOpenGroup);
  const [lastUsed, setLastUsed] = createSignal<Record<string, string>>(
    loadLastUsed(GROUP_CHILDREN),
  );

  const rememberChoice = (group: string, key: string) => {
    const next = { ...lastUsed(), [group]: key };
    setLastUsed(next);
    persistLastUsed(next);
    flyouts.select();
  };

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

  /** Selected sketch profile eligible for the modify operators. */
  const modifyProfile = () => {
    const active = selection();
    const node = active && nodeById(active.nodeId);
    return node?.kind === "profile" && node.line !== null ? node : null;
  };
  const hasOperator = () => (modifyProfile()?.operators.length ?? 0) > 0;
  const modifyEnabled = (key: string) => {
    const profile = modifyProfile();
    if (!profile || busy()) return false;
    if (key === "extrude") {
      return !profile.operators.some((item) => item.kind === "extrude");
    }
    return !hasOperator();
  };

  const activateCreate = (key: string) => {
    setTool(tool() === key ? "select" : (key as ToolMode));
  };

  const activateAnnotate = (key: string) => {
    if (!isEdgeConstraintTool(key as ToolMode)) setSelectionMode("vertex");
    setTool(tool() === key ? "select" : (key as ToolMode));
  };

  const activateModify = (key: string) => {
    const profile = modifyProfile();
    if (!profile) return;
    if (key === "extrude") props.onExtrude();
    else if (key === "revolve") props.onRevolve();
    else if (key === "loft") {
      if (pendingLoft()?.nodeId === profile.id) {
        setPendingLoft(null);
        return;
      }
      setSelectionMode("object");
      setPendingLoft({ nodeId: profile.id, line: profile.line! });
    }
  };

  // A mode switch drops tools that no longer exist in the new mode and
  // closes any flyout so the rail cannot show a stale expansion.
  createEffect(() => {
    const mode = editingMode();
    flyouts.dismiss();
    const active = tool();
    if (active === "select") return;
    const available =
      mode !== "simulate" &&
      (CREATE.some((entry) => entry.key === active)
        ? createToolVisibleInMode(active, mode)
        : mode === "sketch");
    if (!available) setTool("select");
  });

  onMount(() => {
    // Capture phase: when a flyout is open, Escape closes only the flyout.
    // Without consuming the event, the viewer's global Escape would also
    // clear the selection and drop the user back to model mode.
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || openGroup() === null) return;
      event.stopPropagation();
      flyouts.dismiss();
    };
    document.addEventListener("keydown", onKeyDown, true);
    onCleanup(() => document.removeEventListener("keydown", onKeyDown, true));
  });

  /** One expandable cluster: parent shows the last-used child. */
  const group = (
    id: string,
    label: string,
    entries: ToolEntry[],
    hooks: {
      isVisible?: (entry: ToolEntry) => boolean;
      isActive: (entry: ToolEntry) => boolean;
      isEnabled?: (entry: ToolEntry) => boolean;
      hint?: (entry: ToolEntry) => string;
      activate: (entry: ToolEntry) => void;
    },
  ) => {
    const visible = () =>
      entries.filter((entry) => hooks.isVisible?.(entry) ?? true);
    const parentEntry = () => {
      const shown = visible();
      return (
        shown.find((entry) => entry.key === lastUsed()[id]) ?? shown[0] ?? entries[0]
      );
    };
    const anyActive = () => visible().some((entry) => hooks.isActive(entry));
    return (
      <div
        class="tool-group"
        classList={{ open: openGroup() === id }}
        onPointerEnter={() => flyouts.pointerEnter(id)}
        onPointerLeave={() => flyouts.pointerLeave(id)}
      >
        <button
          type="button"
          class="tool-group-parent"
          classList={{ active: anyActive() }}
          onClick={() => flyouts.toggle(id)}
          title={`${label} — ${parentEntry().label} (click to expand)`}
          aria-label={`${label} tools`}
          aria-haspopup="menu"
          aria-expanded={openGroup() === id}
          data-testid={`tool-group-${id}`}
        >
          <Dynamic component={parentEntry().icon} />
          <i class="tool-group-caret" aria-hidden="true" />
        </button>
        <div class="tool-flyout" role="menu" aria-label={`${label} tools`}>
          <For each={visible()}>
            {(entry) => (
              <button
                type="button"
                role="menuitem"
                classList={{ active: hooks.isActive(entry) }}
                disabled={!(hooks.isEnabled?.(entry) ?? true)}
                onClick={() => {
                  hooks.activate(entry);
                  rememberChoice(id, entry.key);
                }}
                title={hooks.hint?.(entry) ?? entry.hint}
                aria-label={entry.label}
                data-testid={entry.testid}
              >
                <Dynamic component={entry.icon} />
              </button>
            )}
          </For>
        </div>
      </div>
    );
  };

  return (
    <nav class="tool-rail" aria-label="Tools">
      <div class="mode-switch" role="group" aria-label="Editing mode">
        <For each={EDIT_MODES}>
          {(mode) => (
            <button
              type="button"
              classList={{ active: editingMode() === mode.key }}
              onClick={() => setEditingMode(mode.key)}
              title={mode.hint}
              aria-label={`${mode.label} mode`}
              aria-pressed={editingMode() === mode.key}
              data-testid={`editmode-${mode.key}`}
            >
              <Dynamic component={mode.icon} />
            </button>
          )}
        </For>
      </div>

      <Show when={groupVisibleInMode("select", editingMode())}>
        <hr />
        <For each={SELECT_MODES}>
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
      </Show>

      <Show when={groupVisibleInMode("create", editingMode())}>
        <hr />
        {group("create", "Create", CREATE, {
          isVisible: (entry) => createToolVisibleInMode(entry.key, editingMode()),
          isActive: (entry) => tool() === entry.key,
          activate: (entry) => activateCreate(entry.key),
        })}
      </Show>

      <Show when={groupVisibleInMode("modify", editingMode())}>
        {group("modify", "Modify", MODIFY, {
          isActive: (entry) =>
            entry.key === "loft" && pendingLoft()?.nodeId === modifyProfile()?.id,
          isEnabled: (entry) => modifyEnabled(entry.key),
          hint: (entry) =>
            modifyProfile() === null
              ? `${entry.label} — select a sketch first`
              : entry.hint,
          activate: (entry) => activateModify(entry.key),
        })}
      </Show>

      <Show when={groupVisibleInMode("transform", editingMode())}>
        {group("transform", "Transform", TRANSFORMS, {
          isActive: (entry) =>
            supports(entry.key as GizmoMode) && gizmoMode() === entry.key,
          isEnabled: (entry) => supports(entry.key as GizmoMode),
          hint: (entry) =>
            supports(entry.key as GizmoMode)
              ? entry.hint
              : entry.key === "scale" && transformable()
                ? "Scale — select a solid primitive"
                : `${entry.label} — select a compatible object first`,
          activate: (entry) => setGizmoMode(entry.key as GizmoMode),
        })}
      </Show>

      <Show when={groupVisibleInMode("annotate", editingMode())}>
        {group("annotate", "Annotate", ANNOTATE, {
          isActive: (entry) => tool() === entry.key,
          isEnabled: () => !busy(),
          activate: (entry) => activateAnnotate(entry.key),
        })}
      </Show>

      <Show when={groupVisibleInMode("delete", editingMode())}>
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
      </Show>

      <Show when={tool() === "sketch"}>
        <div class="plane-picks" role="group" aria-label="Sketch plane">
          <span class="plane-picks-label">plane</span>
          <For each={SKETCH_PLANE_CHOICES}>
            {(choice) => (
              <button
                type="button"
                classList={{ active: sketchPlane() === choice.key }}
                onClick={() => setSketchPlane(choice.key)}
                title={choice.hint}
                data-testid={`sketch-plane-${choice.key}`}
              >
                {choice.label}
              </button>
            )}
          </For>
        </div>
      </Show>

      <Show when={transformable() && editingMode() !== "simulate"}>
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
