/**
 * Object tree: a purely visual scene-composition view.
 *
 * READ-ONLY by design — every row is derived from what the compiler reports
 * (`nodes()`, `elements()` and the evaluated `sceneTree()` in state), and the
 * panel never writes back: clicking a row only selects (`setSelection`,
 * `setInspected`, and for a boolean `setTreeFocus`, which lights its solids
 * in the viewport) and hovering only highlights (`setHover`), exactly the
 * signals the viewport uses. All editing still happens through the source or
 * the viewport tools.
 */

import { Dynamic } from "solid-js/web";
import { For, Show, createMemo, createSignal, type Component } from "solid-js";
import {
  elements,
  hover,
  inspected,
  nodes,
  sceneTree,
  selection,
  setHover,
  setInspected,
  setSelection,
  setTreeFocus,
  sceneName,
  treeFocus,
} from "../state";
import { buildSceneTree, visibleRows, type SceneTreeRow } from "../objectTree";
import {
  BoxIcon,
  CylinderIcon,
  ObjectSelectIcon,
  PolygonIcon,
  SphereIcon,
} from "./icons";
import { PanelHeader } from "./ui";

const KIND_ICONS: Record<SceneTreeRow["kind"], Component | null> = {
  scene: ObjectSelectIcon,
  profile: PolygonIcon,
  box: BoxIcon,
  sphere: SphereIcon,
  cylinder: CylinderIcon,
  operator: null,
  boolean: ObjectSelectIcon,
  transform: null,
  sdf: null,
  group: null,
};

export function ObjectTree() {
  const [collapsed, setCollapsed] = createSignal<ReadonlySet<string>>(new Set());
  const rows = createMemo(() =>
    buildSceneTree(nodes(), elements(), sceneName() ?? "scene.py", sceneTree()),
  );
  const shown = createMemo(() => visibleRows(rows(), collapsed()));

  const toggleGroup = (key: string) => {
    const next = new Set(collapsed());
    if (next.has(key)) next.delete(key);
    else next.add(key);
    setCollapsed(next);
  };

  // One selection (guide F2): a row is either the selected object or the
  // inspected element, never both, so each pick clears the other.
  const selectRow = (row: SceneTreeRow) => {
    if (row.nodeId !== null) {
      setInspected(null);
      setSelection({ nodeId: row.nodeId, vertexIndex: null });
      return;
    }
    // A boolean or a pattern has no outline of its own: it points the
    // properties window at its element, and lights every solid under it.
    if ((row.kind === "boolean" || row.kind === "transform") && row.itemId !== null) {
      setSelection(null);
      setInspected(row.elementId);
      setTreeFocus(row.itemId);
      return;
    }
    // An operator has no geometry to select either; it points the
    // properties window at its element instead.
    if (row.elementId !== null) {
      setSelection(null);
      setInspected(row.elementId);
    }
  };

  const isActive = (row: SceneTreeRow) => {
    if (row.nodeId !== null) return selection()?.nodeId === row.nodeId;
    const focus = treeFocus();
    if (focus !== null) return row.itemId === focus;
    return row.elementId !== null && inspected() === row.elementId;
  };

  /** Roving arrow-key navigation over the visible rows. */
  const onRowKeyDown = (event: KeyboardEvent, row: SceneTreeRow, index: number) => {
    const focusRow = (target: number) => {
      const buttons = (event.currentTarget as HTMLElement)
        .closest(".object-tree-rows")
        ?.querySelectorAll<HTMLElement>(".object-tree-row");
      buttons?.[target]?.focus();
    };
    if (event.key === "ArrowDown") {
      event.preventDefault();
      focusRow(Math.min(index + 1, shown().length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      focusRow(Math.max(index - 1, 0));
    } else if (event.key === "ArrowLeft" && row.group && !collapsed().has(row.key)) {
      event.preventDefault();
      toggleGroup(row.key);
    } else if (event.key === "ArrowRight" && row.group && collapsed().has(row.key)) {
      event.preventDefault();
      toggleGroup(row.key);
    }
  };

  return (
    <aside class="object-tree-panel" data-testid="object-tree-panel">
        <PanelHeader kicker="read-only" title="Object tree" />
        <div class="object-tree-rows" role="tree" aria-label="Scene composition">
          <For each={shown()}>
            {(row, index) => (
              <div
                class="object-tree-row"
                classList={{
                  active: isActive(row),
                  hovered: row.nodeId !== null && hover()?.nodeId === row.nodeId,
                  operator: row.kind === "operator",
                  boolean: row.kind === "boolean" || row.kind === "transform",
                  reference: row.reference,
                  heading: row.kind === "group",
                }}
                role="treeitem"
                aria-level={row.depth + 1}
                aria-selected={isActive(row)}
                aria-expanded={row.group ? !collapsed().has(row.key) : undefined}
                tabIndex={index() === 0 ? 0 : -1}
                style={{ "padding-left": `calc(var(--inset) + var(--unit) * ${4 * row.depth})` }}
                onClick={() => selectRow(row)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    selectRow(row);
                  } else {
                    onRowKeyDown(event, row, index());
                  }
                }}
                onMouseEnter={() =>
                  row.nodeId !== null &&
                  setHover({ nodeId: row.nodeId, vertexIndex: null })
                }
                onMouseLeave={() => row.nodeId !== null && setHover(null)}
                data-testid={`tree-row-${row.testKey}`}
                data-item={row.itemId ?? undefined}
              >
                <Show when={row.group} fallback={<i class="tree-caret-space" />}>
                  <button
                    type="button"
                    class="tree-caret"
                    tabIndex={-1}
                    aria-hidden="true"
                    onClick={(event) => {
                      event.stopPropagation();
                      toggleGroup(row.key);
                    }}
                  >
                    {collapsed().has(row.key) ? "▸" : "▾"}
                  </button>
                </Show>
                <Show when={KIND_ICONS[row.kind]}>
                  {(icon) => (
                    <i class="tree-icon" aria-hidden="true">
                      <Dynamic component={icon()} />
                    </i>
                  )}
                </Show>
                <span class="tree-label">{row.label}</span>
                <Show when={row.detail}>
                  <small class="tree-detail">{row.detail}</small>
                </Show>
                <Show when={row.uses > 1}>
                  <b
                    class="tree-badge tree-uses"
                    title={`Shared: read by ${row.uses} operands — one object, not ${row.uses}`}
                    data-testid={`tree-uses-${row.testKey}`}
                  >
                    ×{row.uses}
                  </b>
                </Show>
                <Show when={row.constraintCount > 0}>
                  <b
                    class="tree-badge"
                    title={`${row.constraintCount} constraints`}
                  >
                    {row.constraintCount}
                  </b>
                </Show>
                <Show when={row.material}>
                  <small class="tree-material">{row.material}</small>
                </Show>
              </div>
            )}
          </For>
        </div>
    </aside>
  );
}
