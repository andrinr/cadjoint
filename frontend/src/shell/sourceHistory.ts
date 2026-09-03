/**
 * Undo/redo over the program text, and the shell actions that replace it.
 *
 * The playground's source of truth is the Python source, so "undo" is just
 * "restore an earlier text and recompile". `history.ts` owns the snapshot
 * stack; this module wires it to Solid so the menu's enabled state tracks it,
 * and holds the four actions that swap the whole program — undo, redo, a
 * fresh scene, and a file opened from disk. They all end the same way:
 * adopt the text, drop the selection, recompile.
 *
 * The actions need `run()`, which in turn commits to the history, so the two
 * halves are created separately: the store first, the compile cycle over it,
 * then the actions over both.
 */

import { createMemo, createSignal, type Accessor } from "solid-js";
import { SourceHistory } from "../history";
import { dirty, setSceneName, setSelection, setSource, source } from "../state";

export interface SourceHistoryStore {
  /** Adopt `text` as the newest committed state. */
  commit: (text: string) => void;
  /** Re-read the stack's depth after a step that bypassed `commit`. */
  bump: () => void;
  canUndo: Accessor<boolean>;
  canRedo: Accessor<boolean>;
  /** Step back one snapshot, or null at the beginning of history. */
  stepBack: () => string | null;
  /** Step forward one snapshot, or null at the end of history. */
  stepForward: () => string | null;
}

/**
 * The snapshot stack, with a version signal so `canUndo`/`canRedo` are
 * reactive over a plain (non-reactive) class instance.
 */
export function createSourceHistory(): SourceHistoryStore {
  const history = new SourceHistory();
  const [version, setVersion] = createSignal(0);
  const bump = () => setVersion((current) => current + 1);
  const commit = (text: string) => {
    history.commit(text);
    bump();
  };
  const canUndo = createMemo(() => (version(), history.canUndo()));
  const canRedo = createMemo(() => (version(), history.canRedo()));
  return {
    commit,
    bump,
    canUndo,
    canRedo,
    stepBack: () => history.undo(),
    stepForward: () => history.redo(),
  };
}

export interface SourceActionsOptions {
  history: SourceHistoryStore;
  /** Recompile the adopted source; the compile cycle's `run`. */
  run: () => Promise<void>;
  /** The starter program a new scene resets to. */
  example: Accessor<string>;
}

export interface SourceActions {
  undo: () => void;
  redo: () => void;
  newScene: () => void;
  adoptScene: (name: string, text: string) => void;
}

/** The four actions that replace the whole program and recompile. */
export function createSourceActions(options: SourceActionsOptions): SourceActions {
  const undo = () => {
    // Capture typed-but-unrun edits first so redo can return to them.
    options.history.commit(source());
    const previous = options.history.stepBack();
    options.history.bump();
    if (previous === null) return;
    setSource(previous);
    setSelection(null);
    void options.run();
  };

  const redo = () => {
    const next = options.history.stepForward();
    options.history.bump();
    if (next === null) return;
    setSource(next);
    setSelection(null);
    void options.run();
  };

  /** Reset to the starter example, asking before unsaved work is lost. */
  const newScene = () => {
    if (dirty() && !window.confirm("Discard unsaved changes to the current scene?")) {
      return;
    }
    setSceneName(null);
    setSource(options.example());
    setSelection(null);
    void options.run();
  };

  /** Adopt a scene file loaded through File → Open. */
  const adoptScene = (name: string, text: string) => {
    setSceneName(name);
    setSource(text);
    setSelection(null);
    void options.run();
  };

  return { undo, redo, newScene, adoptScene };
}
