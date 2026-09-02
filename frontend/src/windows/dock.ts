/**
 * The dock: a `dockview` grid whose panels hold Solid trees.
 *
 * Ownership is the whole story here. `dockview` owns the geometry — groups,
 * tabs, splitters, floating windows — and hands us a bare `<div>` per panel.
 * We own what goes inside it: one Solid root per panel, created when the panel
 * is created and disposed when the panel is disposed, so closing a window
 * tears down its effects instead of leaking them into the next reload.
 *
 * Two details are load-bearing:
 *
 *   - Solid does not care whether its DOM is in the document. A panel in a
 *     background tab keeps its subscriptions and keeps writing into a detached
 *     node, so it is already current when its tab is next selected. Nothing
 *     needs to re-run on activation.
 *
 *   - The viewport is rendered with `renderer: "always"`, which parks its
 *     element in dockview's overlay container and repositions it over whatever
 *     group owns it. The WebGPU canvas is therefore *never* re-parented: not
 *     when it is dragged to a new split, not when it is tabbed behind another
 *     window, not when its group is floated. Re-parenting a canvas is what
 *     loses a GPU context, and the cheapest way to survive that is to not do
 *     it. Its size still changes, which `ViewerPane`'s `ResizeObserver`
 *     already reports to the renderer.
 */

import { DockviewComponent, type DockviewApi, type IContentRenderer } from "dockview";
import { render } from "solid-js/web";
import type { JSX } from "solid-js";
import {
  FALLBACK_PLACEMENTS,
  isPermanent,
  tabTestId,
  windowTitle,
  WINDOW_IDS,
  type OpenPlacement,
  type WindowId,
} from "./panels";
import type { DockLayout } from "./workspace";

export interface DockCallbacks {
  /** A panel's Solid content, by window id. */
  renderWindow: (id: WindowId) => JSX.Element;
  /** The user closed a window through the tab's × or the context menu. */
  onClosed: (id: WindowId) => void;
  /** The user parked a window from its group header. */
  onMinimise: (id: WindowId) => void;
  /** The user asked for the group to float, or to go back into the grid. */
  onToggleFloat: (id: WindowId) => void;
  /** Any geometry change worth persisting (move, resize, tab, float). */
  onLayoutChanged: () => void;
}

export interface Dock {
  readonly api: DockviewApi;
  /** Add a window, or focus it when it is already in the dock. */
  open(id: WindowId, placement?: OpenPlacement): void;
  /** Remove a window from the dock without telling the caller it closed. */
  detach(id: WindowId): void;
  has(id: WindowId): boolean;
  /** Set the extent of a window's group along one axis, in CSS pixels. */
  resize(id: WindowId, size: { width?: number; height?: number }): void;
  /** Lift a window's group out of the grid into a floating window. */
  float(id: WindowId): void;
  /** Put a floating window back into the grid. */
  dock(id: WindowId): void;
  isFloating(id: WindowId): boolean;
  focus(id: WindowId): void;
  save(): DockLayout;
  restore(layout: DockLayout): void;
  clear(): void;
  /** Dispose the roots of windows the last rebuild left out of the dock. */
  sweep(): void;
  layout(): void;
  dispose(): void;
}

interface WindowRoot {
  element: HTMLElement;
  dispose: () => void;
}

export function createDock(container: HTMLElement, callbacks: DockCallbacks): Dock {
  /**
   * One mounted Solid tree per window, kept alive across dock rebuilds.
   *
   * Switching editing modes tears the whole grid down and lays a new one out.
   * If that disposed each panel's root, the viewport would get a *new* canvas
   * every time — and a new canvas means a new WebGPU context, which is
   * exactly the failure this design exists to avoid. So a rebuild hands the
   * same element back, and only a genuine close (or unmounting the dock)
   * disposes the root and its effects.
   */
  const roots = new Map<string, WindowRoot>();
  /** Windows we are removing ourselves; their close must not read as a user close. */
  const detaching = new Set<string>();
  /** Non-zero while the grid is being rebuilt, so roots survive the churn. */
  let preserving = 0;
  let suppressLayoutEvents = 0;

  const component = new DockviewComponent(container, {
    className: "win-dock",
    // The library reads its whole appearance out of CSS variables; this
    // names the class that binds them to the shell's tokens, and turns off
    // the two decorations the design system does not have (a gap between
    // groups, and a curved indicator around the active tab).
    theme: {
      name: "cadjoint",
      className: "win-theme",
      colorScheme: "light",
      gap: 0,
      dndOverlayMounting: "absolute",
      dndPanelOverlay: "group",
      dndTabIndicator: "line",
      tabGroupIndicator: "none",
    },
    disableFloatingGroups: false,
    floatingGroupBounds: "boundedWithinViewport",
    floatingGroupDragHandle: "titlebar",
    singleTabMode: "default",
    noPanelsOverlay: "emptyGroup",
    defaultTabComponent: "window-tab",
    // Pointer-driven DnD: HTML5 drag images are unusable over a WebGPU canvas
    // (the drag image is captured from the compositor and comes out blank),
    // and pointer events are what every other gesture in the viewport uses.
    dndStrategy: "pointer",
    // A tab is a mono label and, for everything but the viewport, a close
    // control. Rendering it ourselves is what lets the viewport refuse to be
    // closed at all — the library has no per-panel "closable" — and gives
    // every tab a stable test hook.
    createTabComponent: ({ id }) => {
      const element = document.createElement("div");
      element.className = "win-tab";
      element.dataset.testid = `window-tab-${id}`;
      const label = document.createElement("span");
      label.className = "win-tab-label";
      // The four simulation windows were a tab strip inside one panel until
      // they became windows; the label keeps that strip's id so the control
      // the audit tool and the e2e suite click still raises the window.
      const alias = tabTestId(id);
      if (alias) label.dataset.testid = alias;
      element.appendChild(label);
      let close: HTMLButtonElement | undefined;
      if (!isPermanent(id)) {
        close = document.createElement("button");
        close.type = "button";
        close.className = "win-tab-close";
        close.textContent = "×";
        close.title = "Close this window";
        close.setAttribute("aria-label", "Close this window");
        close.dataset.testid = `window-close-${id}`;
        // The tab itself is a drag handle; a press on the × must not start one.
        close.addEventListener("pointerdown", (event) => event.stopPropagation());
        close.addEventListener("mousedown", (event) => event.stopPropagation());
        close.addEventListener("click", (event) => {
          event.stopPropagation();
          api.getPanel(id)?.api.close();
        });
        element.appendChild(close);
      }
      return {
        element,
        init: (params) => {
          label.textContent = params.title ?? windowTitle(id);
          element.title = label.textContent;
        },
        update: (event) => {
          const title = (event.params as { title?: string }).title;
          if (typeof title === "string") label.textContent = title;
        },
      };
    },
    createRightHeaderActionComponent: (group) => {
      // The chrome the library does not offer: park a window, and lift its
      // group out of the grid (or put it back). Plain DOM — these are two
      // buttons that read one panel, not a place for a reactive root.
      const element = document.createElement("div");
      element.className = "win-head-actions";
      const button = (
        label: string,
        title: string,
        testid: string,
        action: (id: WindowId) => void,
      ) => {
        const node = document.createElement("button");
        node.type = "button";
        node.className = "win-head-button";
        node.textContent = label;
        node.title = title;
        node.setAttribute("aria-label", title);
        node.dataset.testid = testid;
        node.addEventListener("click", () => {
          const id = group.activePanel?.id as WindowId | undefined;
          if (id) action(id);
        });
        element.appendChild(node);
        return node;
      };
      const floatButton = button("◱", "Float this group", "window-float", (id) =>
        callbacks.onToggleFloat(id),
      );
      const minimiseButton = button("—", "Minimise this window", "window-minimise", (id) =>
        callbacks.onMinimise(id),
      );
      const refresh = () => {
        const id = group.activePanel?.id;
        const permanent = typeof id === "string" && isPermanent(id);
        minimiseButton.disabled = permanent;
        minimiseButton.hidden = permanent;
        const floating = group.api.location.type === "floating";
        floatButton.textContent = floating ? "◰" : "◱";
        floatButton.title = floating ? "Dock this group" : "Float this group";
        floatButton.setAttribute("aria-label", floatButton.title);
      };
      const listener = group.model.onDidActivePanelChange(refresh);
      return {
        element,
        init: refresh,
        dispose: () => listener.dispose(),
      };
    },
    createComponent: (options): IContentRenderer => {
      let root = roots.get(options.id);
      if (!root) {
        const element = document.createElement("div");
        element.className = "win-body";
        element.dataset.window = options.name;
        // `render` returns the disposer, which is the only thing that
        // unsubscribes this window's effects.
        const dispose = render(
          () => callbacks.renderWindow(options.name as WindowId),
          element,
        );
        root = { element, dispose };
        roots.set(options.id, root);
      }
      return {
        element: root.element,
        init: () => undefined,
        dispose: () => {
          if (preserving > 0) return;
          roots.get(options.id)?.dispose();
          roots.delete(options.id);
        },
      };
    },
  });

  const api = component.api;

  const panelFor = (id: WindowId) => api.getPanel(id);

  api.onDidRemovePanel((panel) => {
    if (detaching.has(panel.id)) return;
    if (WINDOW_IDS.includes(panel.id as WindowId)) callbacks.onClosed(panel.id as WindowId);
  });

  api.onDidLayoutChange(() => {
    if (suppressLayoutEvents > 0) return;
    callbacks.onLayoutChanged();
  });

  /** Run a rebuild: no layout events, and every root survives it. */
  const quietly = (work: () => void) => {
    suppressLayoutEvents += 1;
    preserving += 1;
    try {
      work();
    } finally {
      suppressLayoutEvents -= 1;
      preserving -= 1;
    }
  };

  const addPanel = (id: WindowId, placement: OpenPlacement) => {
    const reference = placement.reference ? panelFor(placement.reference) : undefined;
    const horizontal = placement.direction === "left" || placement.direction === "right";
    api.addPanel({
      id,
      component: id,
      title: windowTitle(id),
      // Only the viewport is overlay-rendered; see the module comment.
      renderer: id === "viewport" ? "always" : "onlyWhenVisible",
      ...(placement.inactive ? { inactive: true } : {}),
      ...(placement.size !== undefined
        ? horizontal
          ? { initialWidth: placement.size }
          : { initialHeight: placement.size }
        : {}),
      ...(reference
        ? { position: { referencePanel: reference.id, direction: placement.direction } }
        : {}),
    });
  };

  return {
    api,

    open(id, placement) {
      const existing = panelFor(id);
      if (existing) {
        existing.api.setActive();
        return;
      }
      addPanel(id, placement ?? FALLBACK_PLACEMENTS[id] ?? {});
    },

    detach(id) {
      const panel = panelFor(id);
      if (!panel) return;
      detaching.add(id);
      try {
        panel.api.close();
      } finally {
        detaching.delete(id);
      }
    },

    has(id) {
      return panelFor(id) !== undefined;
    },

    resize(id, size) {
      panelFor(id)?.api.group.api.setSize(size);
    },

    float(id) {
      const panel = panelFor(id);
      if (!panel || panel.api.location.type === "floating") return;
      // A window lifted off the sheet lands on the sheet's margin, never on
      // the model. The default desks put the editor in a 460px left column,
      // so a 380px window at x = 40 clears the viewport's own tool rail at
      // every audited width; the previous 120 put its right edge on the rail
      // and the float stole the clicks meant for the tools underneath.
      api.addFloatingGroup(panel, {
        position: { top: 96, left: 40 },
        width: 380,
        height: 420,
      });
    },

    dock(id) {
      const panel = panelFor(id);
      if (!panel || panel.api.location.type !== "floating") return;
      // Somewhere in the grid to land, that is not the group being moved —
      // floating the viewport and docking it back would otherwise ask the
      // library to move a group beside itself, which it rejects.
      const anchor = api.panels.find(
        (candidate) => candidate.id !== id && candidate.api.location.type === "grid",
      );
      panel.api.moveTo(
        anchor
          ? { group: anchor.api.group, position: "right" }
          : // Nothing is docked at all: give it a fresh group in the grid.
            { group: api.addGroup() },
      );
    },

    isFloating(id) {
      return panelFor(id)?.api.location.type === "floating";
    },

    focus(id) {
      panelFor(id)?.api.setActive();
    },

    save() {
      return api.toJSON() as unknown as DockLayout;
    },

    restore(layout) {
      quietly(() => {
        api.fromJSON(layout as never);
      });
    },

    clear() {
      quietly(() => {
        for (const id of WINDOW_IDS) {
          if (isPermanent(id)) continue;
          const panel = panelFor(id);
          if (!panel) continue;
          detaching.add(id);
          try {
            panel.api.close();
          } finally {
            detaching.delete(id);
          }
        }
        api.clear();
      });
    },

    sweep() {
      for (const [id, root] of [...roots]) {
        if (api.getPanel(id)) continue;
        root.dispose();
        roots.delete(id);
      }
    },

    layout() {
      const rect = container.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) api.layout(rect.width, rect.height);
    },

    dispose() {
      for (const root of roots.values()) root.dispose();
      roots.clear();
      component.dispose();
    },
  };
}
