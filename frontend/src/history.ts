/**
 * Bounded undo/redo history over source snapshots.
 *
 * The playground's source of truth is the program text, so undo is simply
 * "restore an earlier text". Snapshots are committed at meaningful moments —
 * each run and each viewer patch — rather than per keystroke; CodeMirror's own
 * history covers typing while the editor is focused.
 */

export const HISTORY_LIMIT = 100;

export class SourceHistory {
  private past: string[] = [];
  private future: string[] = [];
  private current: string | null = null;

  constructor(private readonly limit = HISTORY_LIMIT) {}

  /**
   * Adopt `source` as the newest committed state.
   *
   * A no-op when the text is unchanged, so calling this on every run cannot
   * flood the stack. Committing a genuinely new state clears the redo branch,
   * as editing after an undo does in any editor.
   */
  commit(source: string): void {
    if (this.current === null) {
      this.current = source;
      return;
    }
    if (source === this.current) return;
    this.past.push(this.current);
    if (this.past.length > this.limit) this.past.shift();
    this.future = [];
    this.current = source;
  }

  /** Step back one snapshot, or return null at the beginning of history. */
  undo(): string | null {
    if (this.past.length === 0 || this.current === null) return null;
    this.future.push(this.current);
    this.current = this.past.pop()!;
    return this.current;
  }

  /** Step forward one snapshot, or return null at the end of history. */
  redo(): string | null {
    if (this.future.length === 0 || this.current === null) return null;
    this.past.push(this.current);
    this.current = this.future.pop()!;
    return this.current;
  }

  canUndo(): boolean {
    return this.past.length > 0;
  }

  canRedo(): boolean {
    return this.future.length > 0;
  }

  /** Number of undo steps currently held (bounded by the limit). */
  get depth(): number {
    return this.past.length;
  }
}
