/**
 * The window-level ⌘ keys the shell owns, as opposed to the viewport's.
 *
 * Every chord is a row of `shortcuts.ts` and runs its command through the
 * registry (`commands.ts`), so ⌘S here and File › Save are the same call.
 * Every matched chord is `preventDefault`ed — whether or not the command can
 * run right now — so ⌘S never falls through to the browser's Save Page and
 * ⌘O never to its file picker.
 *
 * Two keys stay out of the way of the code editor: ⌘Z / ⇧⌘Z are CodeMirror's
 * own history inside any text field, and ⌘↵ inside the editor is the
 * editor's own Run binding (so one press is one run). Escape closes the WGSL
 * dialog in the capture phase, so the viewer's global Escape — which cancels
 * one thing per press, down the ladder in `components/viewer/escape.ts` —
 * does not also fire underneath it.
 *
 * Bare-key tool and mode shortcuts belong to the viewport and are dispatched
 * from `components/viewer/keyboard.ts`, from the same table.
 */

import { onCleanup, onMount, type Accessor } from "solid-js";
import { matchShortcut } from "../shortcuts";
import { runCommand } from "./commands";

export interface ShellShortcutOptions {
  /** Whether the generated-WGSL dialog is open. */
  wgslOpen: Accessor<boolean>;
  closeWgsl: () => void;
}

/** Focus is in a text field or the code editor. */
function typingTarget(): Element | null {
  const target = document.activeElement;
  if (!target) return null;
  return target.tagName === "TEXTAREA" ||
    target.tagName === "INPUT" ||
    target.closest(".cm-editor")
    ? target
    : null;
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
