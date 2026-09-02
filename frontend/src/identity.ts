/**
 * How a request names the declaration it edits.
 *
 * Every addressable `/patch` operation accepts either `id` — the stable id
 * the compile payload publishes for that entry — or the older positional
 * handle: a line for a construction node, an index for a study, mesh or
 * optimization. The server resolves whichever it is given.
 *
 * The id is the better address, and for one reason: a position moves. Adding
 * a study renumbers every study after it, and inserting a statement renumbers
 * every line below it, so two edits queued back to back can disagree about
 * what index 1 meant. A stable id does not move.
 *
 * Both are sent. The id is omitted rather than sent as null when the identity
 * table could not name an entry — a declaration inside a loop, say — which
 * leaves the positional handle doing exactly the job it did before, and keeps
 * a request body free of a key that says nothing.
 */

/** `{ id }` when the entry has a stable id, and nothing when it does not. */
export function byId(entry: { stableId?: string | null }): { id?: string } {
  return entry.stableId ? { id: entry.stableId } : {};
}
