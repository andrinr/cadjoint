/**
 * Flyout state machine for the grouped tool rail.
 *
 * One flyout can be open at a time. A parent icon opens its horizontal
 * flyout on click or after a short hover dwell; the flyout closes when a
 * child is chosen, on Escape, or after a mouse-leave delay. Each group
 * remembers its last-used child (shown on the parent, like Blender and
 * Photoshop), and that memory persists across sessions.
 *
 * Pure logic with injected timers so it can be unit tested without a DOM.
 */

export interface FlyoutTimers {
  set(callback: () => void, delayMs: number): number;
  clear(id: number): void;
}

const realTimers: FlyoutTimers = {
  set: (callback, delayMs) => window.setTimeout(callback, delayMs),
  clear: (id) => window.clearTimeout(id),
};

/** Hover dwell before a flyout opens on its own. */
export const FLYOUT_HOVER_OPEN_MS = 160;
/** Grace period after the pointer leaves before the flyout closes. */
export const FLYOUT_LEAVE_CLOSE_MS = 350;

export class FlyoutController {
  private open: string | null = null;
  private hoverTimer: number | null = null;
  private leaveTimer: number | null = null;

  constructor(
    private readonly onChange: (open: string | null) => void,
    private readonly timers: FlyoutTimers = realTimers,
  ) {}

  openGroup(): string | null {
    return this.open;
  }

  /** Click on a parent: toggle its flyout, closing any other. */
  toggle(group: string): void {
    this.cancelTimers();
    this.setOpen(this.open === group ? null : group);
  }

  /** Pointer entered a group; arm the hover-open dwell. */
  pointerEnter(group: string): void {
    this.cancelLeave();
    if (this.open === group) return;
    // An already-open sibling glides straight over, like menu bars do.
    if (this.open !== null) {
      this.cancelHover();
      this.setOpen(group);
      return;
    }
    this.cancelHover();
    this.hoverTimer = this.timers.set(() => {
      this.hoverTimer = null;
      this.setOpen(group);
    }, FLYOUT_HOVER_OPEN_MS);
  }

  /** Pointer left the group; close after a forgiving delay. */
  pointerLeave(group: string): void {
    this.cancelHover();
    if (this.open !== group) return;
    this.cancelLeave();
    this.leaveTimer = this.timers.set(() => {
      this.leaveTimer = null;
      if (this.open === group) this.setOpen(null);
    }, FLYOUT_LEAVE_CLOSE_MS);
  }

  /** A child was chosen: the flyout closes at once. */
  select(): void {
    this.cancelTimers();
    this.setOpen(null);
  }

  /** Escape (or a mode change) closes everything immediately. */
  dismiss(): void {
    this.cancelTimers();
    this.setOpen(null);
  }

  private setOpen(open: string | null): void {
    if (this.open === open) return;
    this.open = open;
    this.onChange(open);
  }

  private cancelHover(): void {
    if (this.hoverTimer !== null) {
      this.timers.clear(this.hoverTimer);
      this.hoverTimer = null;
    }
  }

  private cancelLeave(): void {
    if (this.leaveTimer !== null) {
      this.timers.clear(this.leaveTimer);
      this.leaveTimer = null;
    }
  }

  private cancelTimers(): void {
    this.cancelHover();
    this.cancelLeave();
  }
}

export const LAST_USED_STORAGE_KEY = "cadjoint.railLastUsed.v1";

export interface StringStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function defaultStorage(): StringStorage | undefined {
  try {
    return typeof localStorage === "undefined" ? undefined : localStorage;
  } catch {
    return undefined;
  }
}

/**
 * Load the per-group last-used child map, dropping anything unrecognised.
 *
 * `valid` maps a group to its known children, so a stale entry from an older
 * build cannot select a tool that no longer exists.
 */
export function loadLastUsed(
  valid: Record<string, readonly string[]>,
  storage: StringStorage | undefined = defaultStorage(),
): Record<string, string> {
  try {
    const raw = storage?.getItem(LAST_USED_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const result: Record<string, string> = {};
    for (const [group, child] of Object.entries(parsed)) {
      if (typeof child === "string" && valid[group]?.includes(child)) {
        result[group] = child;
      }
    }
    return result;
  } catch {
    return {};
  }
}

export function persistLastUsed(
  lastUsed: Record<string, string>,
  storage: StringStorage | undefined = defaultStorage(),
): void {
  try {
    storage?.setItem(LAST_USED_STORAGE_KEY, JSON.stringify(lastUsed));
  } catch {
    // Persistence is a convenience only.
  }
}
