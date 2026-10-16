/**
 * The command registry: one implementation per command, every home calls it.
 *
 * The menu bar, the keyboard and (where it is the same act) the rail all run
 * a command by id, so a command does the same thing from wherever it is
 * reached. Implementations are registered by whoever owns the state they act
 * on — the shell for patches, camera and windows, the menu bar for its own
 * dialogs — and the registry is a module singleton so a key handler deep in
 * the viewport can run one without the shell threading a callback to it.
 *
 * `blocked` says why a command cannot run right now (a sentence the menu puts
 * in the item's title), or null when it can. Mode scoping is separate
 * (`menuModel.ts`): the menu disables an item outside its modes, while a key
 * may still carry the command into the mode it needs, as B has always
 * switched to Model mode to place a box.
 */

import { createSignal } from "solid-js";

export interface CommandImpl {
  run: () => void;
  /** Reactive: null when the command can run, else the reason it cannot. */
  blocked?: () => string | null;
  /** Reactive: for a toggle, whether it is on. */
  checked?: () => boolean;
}

const registry = new Map<string, CommandImpl>();
const [version, setVersion] = createSignal(0);

/** Register implementations; returns the function that removes them again. */
export function registerCommands(impls: Record<string, CommandImpl>): () => void {
  for (const [id, impl] of Object.entries(impls)) registry.set(id, impl);
  setVersion((value) => value + 1);
  return () => {
    for (const [id, impl] of Object.entries(impls)) {
      if (registry.get(id) === impl) registry.delete(id);
    }
    setVersion((value) => value + 1);
  };
}

/** Reactive lookup. */
export function command(id: string): CommandImpl | undefined {
  version();
  return registry.get(id);
}

/** Why `id` cannot run (reactive), or null when it can. */
export function commandBlocked(id: string): string | null {
  const impl = command(id);
  if (!impl) return "Not available yet";
  return impl.blocked?.() ?? null;
}

/** Run `id` if it is registered and not blocked; true when it ran. */
export function runCommand(id: string): boolean {
  const impl = registry.get(id);
  if (!impl || (impl.blocked?.() ?? null) !== null) return false;
  impl.run();
  return true;
}
