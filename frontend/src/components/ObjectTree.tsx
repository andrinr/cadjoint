/**
 * Object tree: a purely visual scene-composition view.
 *
 * READ-ONLY by design — every row is derived from the construction payload the
 * compiler already reports (`nodes()` in state), and the panel never writes
 * back: clicking a row only selects (`setSelection`) and hovering only
 * highlights (`setHover`), exactly the signals the viewport uses. All editing
 * still happens through the source or the viewport tools.
 */

import { Dynamic } from "solid-js/web";
import { For, Show, createMemo, createSignal, type Component } from "solid-js";
import { hover, nodes, selection, setHover, setSelection } from "../state";
import { windowManager } from "../windows/manager";
import { buildSceneTree, visibleRows, type SceneTreeRow } from "../objectTree";
import {
  BoxIcon,
  CylinderIcon,
  ObjectSelectIcon,
  PolygonIcon,
  SphereIcon,
} from "./icons";

const KIND_ICONS: Record<SceneTreeRow["kind"], Component | null> = {
  scene: ObjectSelectIcon,
  profile: PolygonIcon,
  box: BoxIcon,
  sphere: SphereIcon,
  cylinder: CylinderIcon,
  operator: null,
};

export function ObjectTree() {
  const [collapsed, setCollapsed] = createSignal<ReadonlySet<string>>(new Set());
  const rows = createMemo(() => buildSceneTree(nodes()));
  const shown = createMemo(() => visibleRows(rows(), collapsed()));

  const toggleGroup = (key: string) => {
    const next = new Set(collapsed());
    if (next.has(key)) next.delete(key);
    else next.add(key);
    setCollapsed(next);
  };

  const selectRow = (row: SceneTreeRow) => {
    if (row.nodeId === null) return;
    setSelection({ nodeId: row.nodeId, vertexIndex: null });
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
        <header>
          {/* Kicker first, title second — the same order every dock panel
              header uses, so the eyebrow line is always the top line. */}
          <span>
            <small>read-only</small>
            Object tree
          </span>
          <button
            type="button"
            class="object-tree-close"
            onClick={() => windowManager()?.minimise("objects")}
            title="Park the object tree in the tray"
            aria-label="Park the object tree in the tray"
            data-testid="object-tree-close"
          >
            —
          </button>
        </header>
        <div class="object-tree-rows" role="tree" aria-label="Scene composition">
          <For each={shown()}>
            {(row, index) => (
              <div
                class="object-tree-row"
                classList={{
                  active: row.nodeId !== null && selection()?.nodeId === row.nodeId,
                  hovered: row.nodeId !== null && hover()?.nodeId === row.nodeId,
                  operator: row.kind === "operator",
                }}
                role="treeitem"
                aria-level={row.depth + 1}
                aria-selected={row.nodeId !== null && selection()?.nodeId === row.nodeId}
                aria-expanded={row.group ? !collapsed().has(row.key) : undefined}
                tabIndex={index() === 0 ? 0 : -1}
                style={{ "padding-left": `${6 + row.depth * 14}px` }}
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
                data-testid={`tree-row-${row.key}`}
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
