/**
 * The smallest single replacement that turns one text into another.
 *
 * Source arrives from outside the editor — a `/patch` response, a session
 * start, the optimizer adopting its result — and used to be adopted by
 * replacing the whole document. CodeMirror treats that as "everything
 * changed": the selection collapses to the start and the view scrolls to
 * line 1, so releasing a dragged handle yanked the editor away from the
 * literal it had just rewritten. A patch changes a few characters in one
 * literal; expressed as that change, the editor keeps its place and maps
 * the selection across it like any other edit.
 *
 * Returns `null` when the texts are equal.
 */
export function minimalChange(
  before: string,
  after: string,
): { from: number; to: number; insert: string } | null {
  if (before === after) return null;
  let start = 0;
  const limit = Math.min(before.length, after.length);
  while (start < limit && before.charCodeAt(start) === after.charCodeAt(start)) start += 1;
  let endBefore = before.length;
  let endAfter = after.length;
  while (
    endBefore > start &&
    endAfter > start &&
    before.charCodeAt(endBefore - 1) === after.charCodeAt(endAfter - 1)
  ) {
    endBefore -= 1;
    endAfter -= 1;
  }
  return { from: start, to: endBefore, insert: after.slice(start, endAfter) };
}
