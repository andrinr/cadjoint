/**
 * The Window menu's contract with the dock.
 *
 * `WindowLayout` publishes one of these on mount. Everything a menu, a
 * keyboard shortcut or a test needs to drive the dock goes through it, so the
 * chrome never reaches into `dockview` — or into our own bookkeeping — itself.
 *
 * It is a signal rather than a context because the menu bar sits *above* the
 * dock in the tree and must not be re-parented under it.
 */

import { createSignal } from "solid-js";
import type { WindowId } from "./panels";
import type { WindowStatus } from "./windowState";

export interface WindowManager {
  /** Every window this build has, in menu order. */
  readonly windows: readonly { id: WindowId; title: string; permanent: boolean }[];
  /** Reactive: the window's state in the current mode. */
  status: (id: WindowId) => WindowStatus;
  /** Reactive: whether the window's group has been lifted out of the grid. */
  isFloating: (id: WindowId) => boolean;
  /** Reactive: which windows are parked in the tray right now. */
  minimised: () => WindowId[];
  open: (id: WindowId) => void;
  close: (id: WindowId) => void;
  minimise: (id: WindowId) => void;
  /** Bring a minimised or closed window back where it was. */
  restore: (id: WindowId) => void;
  /** Open when closed or parked, close when open. */
  toggle: (id: WindowId) => void;
  float: (id: WindowId) => void;
  dock: (id: WindowId) => void;
  /** Throw away this mode's arrangement and rebuild its default. */
  resetLayout: () => void;
}

export const [windowManager, setWindowManager] = createSignal<WindowManager | null>(null);

/**
 * The same object on `window`, for the end-to-end tests and the console.
 *
 * The Window menu is a real menu with real buttons and the tests drive those;
 * this exists for the operations a menu does not have a button for yet
 * (`resetLayout`, `float`) so a test does not have to reach into `dockview`.
 */
declare global {
  interface Window {
    __cadjointWindows?: WindowManager;
  }
}

export function publishWindowManager(manager: WindowManager | null): void {
  setWindowManager(manager);
  if (typeof window !== "undefined") {
    if (manager) window.__cadjointWindows = manager;
    else delete window.__cadjointWindows;
  }
}
