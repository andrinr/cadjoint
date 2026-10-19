/**
 * The browser storage the app persists its conveniences to: the editing mode,
 * the rail's last-used tools, render presets, job references, the window
 * layout.
 *
 * Every caller takes the storage as a parameter, so tests pass a plain object
 * and a browser without storage passes null. Persistence is never load-bearing:
 * a read that fails falls back to defaults and a write that fails is dropped.
 */

/** The part of the Web Storage API the app uses. */
export interface KeyValueStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/**
 * `localStorage`, or null where there is none.
 *
 * Private-mode browsers can throw on the property access itself, not only on
 * reads and writes, so the lookup is guarded too.
 */
export function browserStorage(): KeyValueStorage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}
