/**
 * The workspace: a dock, a tray for what has been parked, and the rules that
 * tie both to the editing mode.
 *
 * What a mode means now
 * --------------------
 * Modes no longer force windows on and off. A mode owns a *layout*: the
 * arrangement you were last in while you were in that mode, seeded from a
 * default the first time. Entering Simulate restores your Simulate desk;
 * going back to Model restores the Model desk exactly as you left it, down to
 * which windows were floating and which were parked in the tray. Opening the
 * Simulate window while in Model mode is allowed — it simply becomes part of
 * what Model remembers. That is strictly more honest than hiding windows a
 * mode "does not own": the user asked for free-form windows, and a mode that
 * kept yanking them away would be fighting them.
 *
 * Persistence is per mode and best-effort. A record that cannot be read, or
 * that names a window this build no longer has, is discarded in favour of the
 * defaults rather than half-applied.
 */

import { createEffect, createSignal, For, on, onCleanup, onMount, Show } from "solid-js";
import type { JSX } from "solid-js";
import type { EditingMode } from "../editingMode";
import { editingMode, setPanels, setPanelVisibilityHandler, type PanelVisibility } from "../state";
import { ProcessesPanel } from "../components/ProcessesPanel";
import { createDock, type Dock } from "./dock";
import { publishWindowManager, type WindowManager } from "./manager";
import {
  DEFAULT_LAYOUTS,
  isPermanent,
  WINDOW_DEFS,
  WINDOW_IDS,
  windowTitle,
  type WindowId,
} from "./panels";
import {
  defaultModeWindows,
  minimisedWindows,
  reduceModeWindows,
  type ModeWindows,
  type WindowStates,
} from "./windowState";
import {
  isRestorableLayout,
  readWorkspace,
  writeWorkspace,
  type Workspace,
} from "./workspace";
import { DOCK_REBUILT_EVENT } from "./events";
import "./windows.css";

export interface WindowLayoutProps {
  /** The Solid tree for one window, rendered into the dock's container. */
  renderWindow: (id: WindowId) => JSX.Element;
}

/** The Window menu's older, coarser vocabulary, kept working. */
const PANEL_KEY_TO_WINDOW: Record<keyof PanelVisibility, WindowId> = {
  editor: "editor",
  objectTree: "objects",
  materials: "materials",
  sketch: "sketch",
};

export function WindowLayout(props: WindowLayoutProps) {
  let host!: HTMLDivElement;
  let dock: Dock | undefined;
  let activeMode: EditingMode = editingMode();
  /** Guards the reconciliation that runs while we are rebuilding the dock. */
  let rebuilding = false;

  const storage = typeof localStorage === "undefined" ? undefined : localStorage;
  let workspace: Workspace = readWorkspace(storage);

  const [states, setStates] = createSignal<WindowStates>(workspace.windows);
  /** Bumped whenever a window's dock location may have changed. */
  const [locationRevision, setLocationRevision] = createSignal(0);

  const modeWindows = (): ModeWindows => states()[activeMode];

  const putStatuses = (next: ModeWindows) => {
    setStates((current) => ({ ...current, [activeMode]: next }));
  };

  let persistHandle: ReturnType<typeof setTimeout> | undefined;
  const persistNow = () => {
    persistHandle = undefined;
    if (!dock) return;
    workspace = {
      ...workspace,
      layouts: { ...workspace.layouts, [activeMode]: dock.save() },
      windows: states(),
    };
    writeWorkspace(storage, workspace);
  };
  /** Drags fire a layout event per frame; one write per settle is plenty. */
  const persist = () => {
    if (persistHandle !== undefined) clearTimeout(persistHandle);
    persistHandle = setTimeout(persistNow, 250);
  };

  /** Rebuild a mode's default arrangement from scratch. */
  const buildDefault = (mode: EditingMode) => {
    if (!dock) return;
    dock.clear();
    for (const step of DEFAULT_LAYOUTS[mode]) dock.open(step.id, step);
    putStatuses(defaultModeWindows(mode));
  };

  /**
   * Finish a rebuild on the next frame.
   *
   * Two things can only be done once the grid has settled. Column widths,
   * because `initialWidth` is a hint the grid honours only when there is room
   * at insertion time, so the last window added would otherwise squeeze the
   * earlier ones back to equal columns. And a `layout()`, because the
   * overlay-rendered viewport is positioned against its group's rectangle, and
   * a restored grid has not measured itself yet — without this the viewport
   * covers the whole dock after a mode switch.
   */
  const settleRebuild = (mode: EditingMode, applySizes: boolean) => {
    requestAnimationFrame(() => {
      if (!dock) return;
      if (applySizes) {
        for (const step of DEFAULT_LAYOUTS[mode]) {
          if (step.size === undefined) continue;
          const horizontal = step.direction === "left" || step.direction === "right";
          dock.resize(step.id, horizontal ? { width: step.size } : { height: step.size });
        }
      }
      dock.layout();
      window.dispatchEvent(new Event(DOCK_REBUILT_EVENT));
    });
  };

  /**
   * Bring the statuses back in line with what the dock actually holds.
   *
   * A restored layout is the authority on what is docked; a window absent
   * from it is parked only if the record said so, and closed otherwise.
   */
  const reconcile = (stored: ModeWindows) => {
    if (!dock) return;
    const next = { ...stored };
    for (const id of WINDOW_IDS) {
      if (dock.has(id)) next[id] = "open";
      else if (next[id] === "open") next[id] = isPermanent(id) ? "open" : "closed";
    }
    putStatuses(next);
  };

  const applyMode = (mode: EditingMode) => {
    if (!dock) return;
    let fromDefaults = true;
    rebuilding = true;
    try {
      const stored = workspace.layouts[mode];
      if (isRestorableLayout(stored)) {
        try {
          dock.restore(stored!);
          fromDefaults = false;
          reconcile(workspace.windows[mode] ?? defaultModeWindows(mode));
        } catch {
          // A layout the library refuses is not worth debugging at runtime.
          buildDefault(mode);
        }
      } else {
        buildDefault(mode);
      }
      // The viewport is furniture: a record that lost it gets it back.
      if (!dock.has("viewport")) dock.open("viewport");
      // Windows this mode does not show gave up their place in the grid; now
      // they give up their Solid roots too.
      dock.sweep();
    } finally {
      rebuilding = false;
    }
    setLocationRevision((revision) => revision + 1);
    settleRebuild(mode, fromDefaults);
  };

  const openWindow = (id: WindowId) => {
    dock?.open(id);
    putStatuses(reduceModeWindows(modeWindows(), { kind: "open", id }));
    setLocationRevision((revision) => revision + 1);
    persist();
  };

  const closeWindow = (id: WindowId) => {
    if (isPermanent(id)) return;
    dock?.detach(id);
    putStatuses(reduceModeWindows(modeWindows(), { kind: "close", id }));
    setLocationRevision((revision) => revision + 1);
    persist();
  };

  const minimiseWindow = (id: WindowId) => {
    const next = reduceModeWindows(modeWindows(), { kind: "minimise", id });
    if (next === modeWindows()) return;
    dock?.detach(id);
    putStatuses(next);
    setLocationRevision((revision) => revision + 1);
    persist();
  };

  const resetLayout = () => {
    rebuilding = true;
    try {
      buildDefault(activeMode);
      dock?.sweep();
    } finally {
      rebuilding = false;
    }
    settleRebuild(activeMode, true);
    workspace = {
      ...workspace,
      layouts: { ...workspace.layouts, [activeMode]: undefined },
      windows: states(),
    };
    setLocationRevision((revision) => revision + 1);
    persist();
  };

  const manager: WindowManager = {
    windows: WINDOW_DEFS.map((def) => ({
      id: def.id,
      title: def.title,
      permanent: def.permanent === true,
    })),
    status: (id) => states()[activeMode][id],
    isFloating: (id) => {
      locationRevision();
      return dock?.isFloating(id) ?? false;
    },
    minimised: () => minimisedWindows(states()[activeMode]),
    open: openWindow,
    close: closeWindow,
    minimise: minimiseWindow,
    restore: openWindow,
    toggle: (id) => (states()[activeMode][id] === "open" ? closeWindow(id) : openWindow(id)),
    float: (id) => {
      if (!dock?.has(id)) openWindow(id);
      dock?.float(id);
      setLocationRevision((revision) => revision + 1);
      persist();
    },
    dock: (id) => {
      dock?.dock(id);
      setLocationRevision((revision) => revision + 1);
      persist();
    },
    resetLayout,
  };

  onMount(() => {
    dock = createDock(host, {
      // The process monitor is the one window the shell does not have to
      // wire up: it talks to `/api/jobs` and to the job store directly, so
      // it is rendered here rather than threaded through the app's
      // `renderWindow`, which is a map of *scene* panels.
      renderWindow: (id) => (id === "processes" ? <ProcessesPanel /> : props.renderWindow(id)),
      onClosed: (id) => {
        if (rebuilding) return;
        putStatuses(reduceModeWindows(modeWindows(), { kind: "close", id }));
        setLocationRevision((revision) => revision + 1);
        persist();
      },
      onMinimise: (id) => minimiseWindow(id),
      onToggleFloat: (id) => {
        if (dock?.isFloating(id)) dock.dock(id);
        else dock?.float(id);
        setLocationRevision((revision) => revision + 1);
        persist();
      },
      onLayoutChanged: () => {
        if (rebuilding) return;
        setLocationRevision((revision) => revision + 1);
        persist();
      },
    });
    // Adopt the container's size before laying anything out: the library's own
    // ResizeObserver has not fired yet on the first frame, and a grid that
    // still thinks it is zero-wide silently drops the column widths.
    // Adopt the container's size before laying anything out: the library's own
    // ResizeObserver has not fired on the first frame, and a grid that still
    // thinks it is zero-wide silently drops the column widths.
    dock.layout();
    applyMode(activeMode);
    publishWindowManager(manager);
    setPanelVisibilityHandler((key: keyof PanelVisibility, visible: boolean) => {
      const id = PANEL_KEY_TO_WINDOW[key];
      if (visible) openWindow(id);
      else closeWindow(id);
    });

    onCleanup(() => {
      if (persistHandle !== undefined) clearTimeout(persistHandle);
      persistNow();
      setPanelVisibilityHandler(null);
      publishWindowManager(null);
      dock?.dispose();
      dock = undefined;
    });
  });

  // Mirror the dock into the coarse `panels()` record the menu bar reads.
  createEffect(() => {
    const current = states()[activeMode];
    setPanels({
      editor: current.editor === "open",
      objectTree: current.objects === "open",
      materials: current.materials === "open",
      sketch: current.sketch === "open",
    });
  });

  // A mode change saves the desk you are leaving and lays out the one you enter.
  createEffect(
    on(
      editingMode,
      (mode) => {
        if (!dock || mode === activeMode) return;
        if (persistHandle !== undefined) clearTimeout(persistHandle);
        persistNow();
        activeMode = mode;
        applyMode(mode);
        persist();
      },
      { defer: true },
    ),
  );

  return (
    <div class="win-workspace" data-testid="window-workspace">
      <div class="win-dock-host" ref={host} data-testid="window-dock" />
      <Show when={manager.minimised().length > 0}>
        <div class="win-tray" data-testid="window-tray" role="toolbar" aria-label="Minimised windows">
          <span class="win-tray-label">Parked</span>
          <For each={manager.minimised()}>
            {(id) => (
              <button
                type="button"
                class="win-tray-item"
                data-testid={`window-restore-${id}`}
                title={`Restore ${windowTitle(id)}`}
                onClick={() => openWindow(id)}
              >
                {windowTitle(id)}
              </button>
            )}
          </For>
        </div>
      </Show>
    </div>
  );
}
