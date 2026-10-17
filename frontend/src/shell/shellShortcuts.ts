/**
 * The window-level ⌘ chords the shell owns; bare keys belong to the viewport
 * (`components/viewer/keyboard.ts`). Both dispatch from `shortcuts.ts`
 * through the command registry.
 *
 * Every matched chord is `preventDefault`ed, runnable or not, so ⌘S never
 * falls through to the browser's Save Page. Inside a text field ⌘Z / ⇧⌘Z are
 * CodeMirror's history, and ⌘↵ in the editor is its own Run binding. Escape
 * closes the WGSL dialog in the capture phase so the viewer's Escape ladder
 * does not also fire.
 */

import { onCleanup, onMount, type Accessor } from "solid-js";
import { matchShortcut, typingTarget } from "../shortcuts";
import { runCommand } from "./commands";

export interface ShellShortcutOptions {
  /** Whether the generated-WGSL dialog is open. */
  wgslOpen: Accessor<boolean>;
  closeWgsl: () => void;
}

/** Commands whose chord a text surface keeps for itself. */
const TEXT_OWNS = new Set(["edit-undo", "edit-redo"]);

export function createShellShortcuts(options: ShellShortcutOptions): void {
  onMount(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const shortcut = matchShortcut(event, "app");
      if (!shortcut) return;
      const typing = typingTarget();
      if (typing && TEXT_OWNS.has(shortcut.command)) return;
      if (shortcut.command === "program-run" && typing?.closest(".cm-editor")) return;
      event.preventDefault();
      runCommand(shortcut.command);
    };
    window.addEventListener("keydown", onKeyDown);
    onCleanup(() => window.removeEventListener("keydown", onKeyDown));

    // Escape closes the WGSL dialog. Capture phase, so the viewer's global
    // Escape (clear selection, reset mode) does not also fire underneath.
    const onEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || !options.wgslOpen()) return;
      event.stopPropagation();
      options.closeWgsl();
    };
    document.addEventListener("keydown", onEscape, true);
    onCleanup(() => document.removeEventListener("keydown", onEscape, true));
  });
}
