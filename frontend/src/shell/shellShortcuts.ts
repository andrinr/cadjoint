/**
 * The window-level keys the shell owns, as opposed to the viewport's.
 *
 * Two rules, both about staying out of the way: Ctrl/Cmd+Z is undo/redo
 * everywhere *except* inside a text field, where CodeMirror's own history is
 * the right one; and Escape closes the WGSL dialog in the capture phase, so
 * the viewer's global Escape — which cancels one thing per press, down the
 * ladder in `components/viewer/escape.ts` — does not also fire underneath it.
 *
 * Tool and mode keys are not here — they belong to the viewport and live in
 * `shortcuts.ts` with the Help dialog's table.
 */

import { onCleanup, onMount, type Accessor } from "solid-js";

export interface ShellShortcutOptions {
  undo: () => void;
  redo: () => void;
  /** Whether the generated-WGSL dialog is open. */
  wgslOpen: Accessor<boolean>;
  closeWgsl: () => void;
}

export function createShellShortcuts(options: ShellShortcutOptions): void {
  onMount(() => {
    // Undo/redo shortcuts, kept away from the editor: while typing there,
    // CodeMirror's own history owns Ctrl/Cmd+Z. The menu items always work.
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== "z") return;
      const target = document.activeElement;
      const typing =
        target &&
        (target.tagName === "TEXTAREA" ||
          target.tagName === "INPUT" ||
          target.closest(".cm-editor"));
      if (typing) return;
      event.preventDefault();
      if (event.shiftKey) options.redo();
      else options.undo();
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
