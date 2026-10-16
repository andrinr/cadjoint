/**
 * Vertical tool rail: grouped tool clusters, scoped by the editing mode.
 *
 * The groups are the menu bar's: Create, Shape, Transform, Constrain — one
 * name per command in every home. What a tool *does* lives in
 * `../shell/toolActions`, shared with the menus and the keys; the rail only
 * decides how it looks. Every item carries its word at 1280px and wider
 * (design language I2), the icon alone below that.
 *
 * The mode switcher itself lives in the toolbar (`ModeSwitcher`); the rail
 * only shows the clusters the active mode offers, and disappears entirely in
 * tool-less modes (Simulate, Render). Tools sit in expandable clusters: a
 * parent icon shows the group's last-used child and expands — on hover dwell
 * or click — into a horizontal flyout of the children, like Photoshop's
 * nested tools. Flyouts close on selection, Escape, or a mouse-leave delay;
 * the last-used choice persists.
 *
 * The open/close rules live in `../flyout`, the mode filtering in
 * `../editingMode` — both pure and unit tested.
 */

import { Dynamic } from "solid-js/web";
import { For, Show, createEffect, createSignal, onCleanup, onMount, type Component } from "solid-js";
import {
  editingMode,
  gizmoMode,
  pendingLoft,
  selection,
  selectionMode,
  setGizmoMode,
  setSketchPlane,
  setTool,
  sketchPlane,
  tool,
} from "../state";
import {
  createToolVisibleInMode,
  groupVisibleInMode,
  railGroupsForMode,
} from "../editingMode";
import { FlyoutController, loadLastUsed, persistLastUsed } from "../flyout";
import { SKETCH_PLANE_CHOICES } from "../sketchPlanes";
import { shortcutLabel } from "../shortcuts";
import {
  constraintBlocked,
  fixBlocked,
  gizmoBlocked,
  modifyProfile,
  setSelectKind,
  shapeBlocked,
  supportsGizmo,
  toggleConstraintTool,
  toggleCreateTool,
  toggleLoft,
  transformable,
} from "../shell/toolActions";
import type { GizmoMode, SelectionMode, ToolMode } from "../types";
import { FixIcon } from "./FixIcon";
import {
  BoxIcon,
  CoincidentIcon,
  CylinderIcon,
  DistanceIcon,
  ExtrudeIcon,
  FaceSketchIcon,
  HorizontalIcon,
  LoftIcon,
  MoveIcon,
  ObjectSelectIcon,
  ParallelIcon,
  PerpendicularIcon,
  PointAddIcon,
  PolygonIcon,
  RevolveIcon,
  RotateIcon,
  ScaleIcon,
  SphereIcon,
  TrashIcon,
  VertexSelectIcon,
  VerticalIcon,
} from "./icons";

/** "Move along an axis  (G)" — the hint, with the key the table binds. */
const withKey = (hint: string, command: string) => {
  const key = shortcutLabel(command);
  return key ? `${hint}  (${key})` : hint;
};

const SELECT_MODES: { key: SelectionMode; label: string; hint: string; icon: Component }[] = [
  { key: "object", label: "Object", hint: withKey("Select whole objects", "select-object"), icon: ObjectSelectIcon },
  { key: "vertex", label: "Vertex", hint: withKey("Select sketch vertices", "select-vertex"), icon: VertexSelectIcon },
];

interface ToolEntry {
  key: string;
  label: string;
  hint: string;
  icon: Component;
  testid: string;
}

const CREATE: ToolEntry[] = [
  { key: "sketch", label: "Sketch", hint: withKey("Place a new 2D sketch", "tool-sketch"), icon: PolygonIcon, testid: "tool-sketch" },
  { key: "face", label: "Sketch on face", hint: "Place a sketch on a solid's face", icon: FaceSketchIcon, testid: "tool-face" },
  { key: "polygon", label: "Point", hint: withKey("Add a point to a sketch", "tool-polygon"), icon: PointAddIcon, testid: "tool-polygon" },
  { key: "box", label: "Box", hint: withKey("Place a box", "tool-box"), icon: BoxIcon, testid: "tool-box" },
  { key: "sphere", label: "Sphere", hint: withKey("Place a sphere", "tool-sphere"), icon: SphereIcon, testid: "tool-sphere" },
  { key: "cylinder", label: "Cylinder", hint: withKey("Place a cylinder", "tool-cylinder"), icon: CylinderIcon, testid: "tool-cylinder" },
];

// Test ids keep their old `modify-` / `annotate-` prefixes: the end-to-end
// specs address them, and a test id is not a name anybody reads.
const SHAPE: ToolEntry[] = [
  { key: "extrude", label: "Extrude", hint: withKey("Extrude the selected sketch", "shape-extrude"), icon: ExtrudeIcon, testid: "modify-extrude" },
  { key: "revolve", label: "Revolve", hint: "Revolve the selected sketch", icon: RevolveIcon, testid: "modify-revolve" },
  { key: "loft", label: "Loft", hint: "Loft the selected sketch to a second one", icon: LoftIcon, testid: "modify-loft" },
];

const TRANSFORMS: ToolEntry[] = [
  { key: "translate", label: "Move", hint: withKey("Move along an axis", "gizmo-translate"), icon: MoveIcon, testid: "gizmo-translate" },
  { key: "rotate", label: "Rotate", hint: withKey("Rotate about an axis", "gizmo-rotate"), icon: RotateIcon, testid: "gizmo-rotate" },
  { key: "scale", label: "Scale", hint: withKey("Scale along an axis", "gizmo-scale"), icon: ScaleIcon, testid: "gizmo-scale" },
];

const CONSTRAIN: ToolEntry[] = [
  { key: "distance", label: "Distance", hint: "Choose two points to preserve their distance", icon: DistanceIcon, testid: "annotate-distance" },
  { key: "horizontal", label: "Horizontal", hint: "Choose two points to keep level", icon: HorizontalIcon, testid: "annotate-horizontal" },
  { key: "vertical", label: "Vertical", hint: "Choose two points to keep stacked", icon: VerticalIcon, testid: "annotate-vertical" },
  { key: "coincident", label: "Coincident", hint: "Choose two points to join", icon: CoincidentIcon, testid: "annotate-coincident" },
  { key: "parallel", label: "Parallel", hint: "Choose two edges to keep parallel", icon: ParallelIcon, testid: "annotate-parallel" },
  { key: "perpendicular", label: "Perpendicular", hint: "Choose two edges to keep perpendicular", icon: PerpendicularIcon, testid: "annotate-perpendicular" },
  // An act, not a tool: it anchors the point that is already selected.
  { key: "fix", label: "Fix", hint: "Anchor the selected point where it is", icon: FixIcon, testid: "annotate-fix" },
];

/** Known children per group, for validating persisted last-used entries. */
const GROUP_CHILDREN: Record<string, string[]> = {
  create: CREATE.map((entry) => entry.key),
  modify: SHAPE.map((entry) => entry.key),
  transform: TRANSFORMS.map((entry) => entry.key),
  annotate: CONSTRAIN.map((entry) => entry.key),
};

export interface ToolRailProps {
  onDelete: () => void;
  /** Extrude the selected sketch (App resolves the profile and patches). */
  onExtrude: () => void;
  onRevolve: () => void;
  /** Anchor the selected sketch point. */
  onFix: () => void;
}

/** The icon, and the word that rides with it at 1280px and wider. */
const Face = (props: { icon: Component; word: string }) => (
  <>
    <Dynamic component={props.icon} />
    <span class="rail-word">{props.word}</span>
  </>
);

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

  const activeTransform = () => (supportsGizmo(gizmoMode()) ? gizmoMode() : "translate");

  const activateShape = (key: string) => {
    if (!modifyProfile()) return;
    if (key === "extrude") props.onExtrude();
    else if (key === "revolve") props.onRevolve();
    else if (key === "loft") toggleLoft();
  };

  // A mode switch drops tools that no longer exist in the new mode and
  // closes any flyout so the rail cannot show a stale expansion.
  createEffect(() => {
    const mode = editingMode();
    flyouts.dismiss();
    const active = tool();
    if (active === "select") return;
    const available =
      railGroupsForMode(mode).length > 0 &&
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
          <Face icon={parentEntry().icon} word={label} />
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
                <Face icon={entry.icon} word={entry.label} />
              </button>
            )}
          </For>
        </div>
      </div>
    );
  };

  return (
    // Tool-less modes (Simulate) hide the rail entirely — their dock panel is
    // the whole workspace, and an empty rail would read as broken.
    <Show when={railGroupsForMode(editingMode()).length > 0}>
    <nav class="tool-rail" aria-label="Tools">
      <Show when={groupVisibleInMode("select", editingMode())}>
        <For each={SELECT_MODES}>
          {(mode) => (
            <button
              type="button"
              class={tool() === "select" && selectionMode() === mode.key ? "active" : ""}
              onClick={() => setSelectKind(mode.key)}
              title={mode.hint}
              aria-label={mode.label}
              data-testid={`mode-${mode.key}`}
            >
              <Face icon={mode.icon} word={mode.label} />
            </button>
          )}
        </For>
      </Show>

      <Show when={groupVisibleInMode("create", editingMode())}>
        <hr />
        {group("create", "Create", CREATE, {
          isVisible: (entry) => createToolVisibleInMode(entry.key, editingMode()),
          isActive: (entry) => tool() === entry.key,
          activate: (entry) => toggleCreateTool(entry.key as ToolMode),
        })}
      </Show>

      <Show when={groupVisibleInMode("modify", editingMode())}>
        {group("modify", "Shape", SHAPE, {
          isActive: (entry) =>
            // Both sides are undefined when nothing is pending and no profile is
            // selected, which lit the group as current beside Object.
            entry.key === "loft" &&
            pendingLoft() !== null &&
            pendingLoft()?.nodeId === modifyProfile()?.id,
          isEnabled: (entry) => shapeBlocked(entry.key as "extrude") === null,
          hint: (entry) => {
            const reason = shapeBlocked(entry.key as "extrude");
            return reason ? `${entry.label} — ${reason.toLowerCase()}` : entry.hint;
          },
          activate: (entry) => activateShape(entry.key),
        })}
      </Show>

      <Show when={groupVisibleInMode("transform", editingMode())}>
        {group("transform", "Transform", TRANSFORMS, {
          isActive: (entry) =>
            supportsGizmo(entry.key as GizmoMode) && gizmoMode() === entry.key,
          isEnabled: (entry) => supportsGizmo(entry.key as GizmoMode),
          hint: (entry) => {
            const reason = gizmoBlocked(entry.key as GizmoMode);
            return reason ? `${entry.label} — ${reason.toLowerCase()}` : entry.hint;
          },
          activate: (entry) => setGizmoMode(entry.key as GizmoMode),
        })}
      </Show>

      <Show when={groupVisibleInMode("annotate", editingMode())}>
        {group("annotate", "Constrain", CONSTRAIN, {
          isActive: (entry) => tool() === entry.key,
          isEnabled: (entry) =>
            (entry.key === "fix" ? fixBlocked() : constraintBlocked()) === null,
          hint: (entry) => {
            const reason = entry.key === "fix" ? fixBlocked() : constraintBlocked();
            return reason ? `${entry.label} — ${reason.toLowerCase()}` : entry.hint;
          },
          activate: (entry) =>
            entry.key === "fix" ? props.onFix() : toggleConstraintTool(entry.key as ToolMode),
        })}
      </Show>

      <Show when={groupVisibleInMode("delete", editingMode())}>
        <hr />
        <button
          type="button"
          class="danger"
          disabled={selection() === null}
          onClick={props.onDelete}
          title={withKey("Delete the selection", "edit-delete")}
          aria-label="Delete"
          data-testid="delete-selection"
        >
          <Face icon={TrashIcon} word="Delete" />
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
    </Show>
  );
}
