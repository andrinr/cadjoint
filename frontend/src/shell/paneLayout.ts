/**
 * The editor/viewport splitter and the grid it drives.
 *
 * One preference — how wide the code pane is — expressed three ways: a
 * signal while dragging, a `grid-template-columns` string for the panes, and
 * a browser-local entry so it survives a reload. Null means "the stylesheet
 * default", which is what a double-click on the splitter restores.
 *
 * Layout persistence is best-effort: a private window or a full quota must
 * never break the viewer, so every storage call is guarded.
 */

import { createSignal, type Accessor } from "solid-js";
import { panels } from "../state";

const EDITOR_WIDTH_KEY = "cadjoint.editorWidth.v1";
const EDITOR_MIN = 280;
const VIEWER_MIN = 360;

export interface PaneLayout {
  editorWidth: Accessor<number | null>;
  resizing: Accessor<boolean>;
  /** Store the width (or clear it) and adopt it. */
  persistEditorWidth: (width: number | null) => void;
  /** The panes' `grid-template-columns`, collapsing with the editor. */
  paneColumns: () => string;
  onSplitterDown: (event: PointerEvent) => void;
  /** `ref` for the panes element the drag measures against. */
  setPanesElement: (element: HTMLElement) => void;
}

export function createPaneLayout(): PaneLayout {
  const storedWidth = Number(localStorage.getItem(EDITOR_WIDTH_KEY));
  const [editorWidth, setEditorWidth] = createSignal<number | null>(
    Number.isFinite(storedWidth) && storedWidth >= EDITOR_MIN ? storedWidth : null,
  );
  const [resizing, setResizing] = createSignal(false);
  let panesElement: HTMLElement | undefined;

  const persistEditorWidth = (width: number | null) => {
    setEditorWidth(width);
    try {
      if (width === null) localStorage.removeItem(EDITOR_WIDTH_KEY);
      else localStorage.setItem(EDITOR_WIDTH_KEY, String(Math.round(width)));
    } catch {
      // Layout persistence is best-effort only.
    }
  };

  const paneColumns = () => {
    if (!panels().editor) return "44px 0px 1fr";
    const width = editorWidth();
    const editor =
      width === null
        ? "minmax(320px, 42%)"
        : `min(${Math.round(width)}px, calc(100% - ${VIEWER_MIN + 6}px))`;
    return `${editor} 6px 1fr`;
  };

  const onSplitterDown = (event: PointerEvent) => {
    if (!panels().editor || !panesElement) return;
    const splitter = event.currentTarget as HTMLElement;
    splitter.setPointerCapture(event.pointerId);
    setResizing(true);
    const bounds = panesElement.getBoundingClientRect();
    const move = (moveEvent: PointerEvent) => {
      const width = Math.min(
        Math.max(moveEvent.clientX - bounds.left, EDITOR_MIN),
        bounds.width - VIEWER_MIN - 6,
      );
      setEditorWidth(width);
    };
    const up = () => {
      splitter.removeEventListener("pointermove", move);
      splitter.removeEventListener("pointerup", up);
      splitter.removeEventListener("pointercancel", up);
      setResizing(false);
      persistEditorWidth(editorWidth());
    };
    splitter.addEventListener("pointermove", move);
    splitter.addEventListener("pointerup", up);
    splitter.addEventListener("pointercancel", up);
  };

  return {
    editorWidth,
    resizing,
    persistEditorWidth,
    paneColumns,
    onSplitterDown,
    setPanesElement: (element: HTMLElement) => {
      panesElement = element;
    },
  };
}
