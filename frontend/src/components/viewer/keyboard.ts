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
  cycleMode,
  editingMode,
  nodeById,
  pendingLoft,
  selection,
  setBcPickArmed,
  setEditingMode,
  setGizmoMode,
  setPendingLoft,
  setSelection,
  setSelectionMode,
  setSimProbe,
  setTool,
  simProbe,
  tool,
} from "../../state";
import { VIEWER_TOOL_KEYS, type ViewerToolAction } from "../../shortcuts";
import { escapeLevel } from "./escape";
import type { ViewerPaneProps } from "./props";

/** True while focus is in the code editor or another text surface. */
function isTypingTarget(): boolean {
  const target = document.activeElement;
  return Boolean(
    target && (target.tagName === "TEXTAREA" || target.closest(".cm-editor")),
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
    if (!typing && !event.metaKey && !event.ctrlKey && !event.altKey) {
      // Key→action table shared with the Help dialog (src/shortcuts.ts).
      // Placement shortcuts for solid primitives imply model mode, so the
      // keys keep working from any mode without leaving a hidden tool armed.
      const actions: Record<ViewerToolAction, () => void> = {
        "select-object": () => (setTool("select"), setSelectionMode("object")),
        "select-vertex": () => (setTool("select"), setSelectionMode("vertex")),
        "tool-polygon": () => setTool("polygon"),
        "tool-box": () => (setEditingMode("model"), setTool("box")),
        "tool-sphere": () => (setEditingMode("model"), setTool("sphere")),
        "tool-cylinder": () => (setEditingMode("model"), setTool("cylinder")),
        "gizmo-translate": () => setGizmoMode("translate"),
        "gizmo-rotate": () => setGizmoMode("rotate"),
        "cycle-mode": () => cycleMode(event.shiftKey ? -1 : 1),
      };
      const action = VIEWER_TOOL_KEYS[event.key.toLowerCase()];
      if (action) {
        event.preventDefault();
        actions[action]();
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
