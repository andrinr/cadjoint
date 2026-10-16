/**
 * Window-level keys the viewport claims.
 *
 * These listeners are global rather than canvas-scoped because the shortcuts
 * have to work while the pointer is anywhere over the app, so every handler
 * starts by asking whether the user is typing — the editor and any text field
 * own their keys first. Escape is deliberately the weakest claim: overlays
 * close themselves in the capture phase, and only a bare viewport Escape backs
 * the editing state out.
 *
 * The pane keeps the state these mutate (the pending constraint pick, whether
 * space is held); this module only decides what a key means.
 */

import {
  bcPickArmed,
  editingMode,
  nodeById,
  pendingLoft,
  selection,
  setBcPickArmed,
  setEditingMode,
  setPendingLoft,
  setSelection,
  setSimProbe,
  setTool,
  simProbe,
  tool,
} from "../../state";
import { matchShortcut } from "../../shortcuts";
import { runCommand } from "../../shell/commands";
import { escapeLevel } from "./escape";
import type { ViewerPaneProps } from "./props";

/**
 * True while focus is in the code editor or another field that takes keys.
 *
 * A number box, a select and a text input own their letters, digits,
 * Backspace and space as much as the editor does: B typed into a file name
 * is a B, not a box.
 */
function isTypingTarget(): boolean {
  const target = document.activeElement as HTMLElement | null;
  return Boolean(
    target &&
      (target.tagName === "TEXTAREA" ||
        target.tagName === "INPUT" ||
        target.tagName === "SELECT" ||
        target.isContentEditable ||
        target.closest(".cm-editor")),
  );
}

export interface ViewerKeyboardContext {
  props: ViewerPaneProps;
  /** Escape abandons a half-finished two-click constraint flow. */
  clearPendingConstraint: () => void;
  /** Whether that flow is half-finished, for Escape's first rung. */
  pendingConstraintActive: () => boolean;
  /** Whether a *command* gesture — a handle or gizmo drag, a BC rectangle —
   * is in flight. Orbiting and panning are not commands. */
  gestureActive: () => boolean;
  /** Abandon that gesture, restoring the value the drag started from. */
  cancelGesture: () => void;
  /** Held space turns any drag into a pan, as other 3D viewports do. */
  setPanHeld: (held: boolean) => void;
}

export function createViewerKeyboard(context: ViewerKeyboardContext) {
  const onKeyDown = (event: KeyboardEvent) => {
    const typing = isTypingTarget();
    if (event.key === "Escape") {
      // Overlays own their Escape (dialogs, menus, popovers, flyouts close
      // themselves in the capture phase); only a bare viewport Escape backs
      // the editing state out.
      const overlayOpen = document.querySelector(
        ".dialog-backdrop, .menu-dropdown, .tool-group.open",
      );
      if (!overlayOpen) {
        // One rung per press: see `escape.ts` for why this is a ladder and
        // not the single flat clear it used to be.
        const level = escapeLevel({
          gesture: context.gestureActive(),
          pendingConstraint: context.pendingConstraintActive(),
          pendingLoft: pendingLoft() !== null,
          toolArmed: tool() !== "select",
          bcPickArmed: bcPickArmed(),
          selection: selection() !== null,
          simProbe: simProbe() !== null,
          awayFromModel: editingMode() !== "model",
        });
        // Only claim the key when there is actually something to cancel, so
        // a press with nothing pending stays available to anything else.
        if (level !== null) event.preventDefault();
        if (level === "gesture") {
          context.cancelGesture();
        } else if (level === "pending") {
          context.clearPendingConstraint();
          setPendingLoft(null);
        } else if (level === "tool") {
          setTool("select");
          setBcPickArmed(false);
        } else if (level === "selection") {
          setSelection(null);
          setSimProbe(null);
        } else if (level === "mode") {
          setEditingMode("model");
        }
      }
    }
    if (!typing) {
      // The bare-key table shared with the menu bar and the Help sheet
      // (src/shortcuts.ts); each key runs its command through the registry,
      // so K here and Create › Sketch are one call. Placement keys for solid
      // primitives imply model mode, so they keep working from any mode
      // without leaving a hidden tool armed (see shell/toolActions.ts).
      const shortcut = matchShortcut(event, "viewport");
      if (shortcut) {
        event.preventDefault();
        runCommand(shortcut.command);
        return;
      }
    }
    const active = selection();
    if ((event.key === "Delete" || event.key === "Backspace") && active) {
      if (typing) return;
      const node = nodeById(active.nodeId);
      if (node?.editable && node.line !== null) {
        event.preventDefault();
        if (active.vertexIndex !== null) {
          void context.props.onPatch("delete_vertex", node.line, active.vertexIndex);
        } else {
          void context.props.onDeleteObject(node.line);
        }
        setSelection(null);
      }
    }
  };

  const onPanKey = (event: KeyboardEvent) => {
    if (event.code !== "Space") return;
    if (isTypingTarget()) return;
    // Stop the page scrolling or a focused button firing.
    event.preventDefault();
    context.setPanHeld(event.type === "keydown");
  };

  return { onKeyDown, onPanKey };
}
