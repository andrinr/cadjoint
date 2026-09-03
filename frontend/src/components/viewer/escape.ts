/**
 * What one press of Escape cancels.
 *
 * Escape used to clear *everything* at once — the pending constraint pick, a
 * half-finished loft, the probe, the selection, the armed tool and the
 * editing mode, in one statement. That is not a cancel, it is a reset: a user
 * who armed the loft tool by accident lost their selection and their mode
 * along with it, and there was no way to back out of one step at a time.
 *
 * So Escape cancels exactly one thing per press, most specific first, and the
 * user walks back out the way they walked in. The rungs, in order:
 *
 * 1. **A gesture in flight** — a drag on a handle or a gizmo, a rubber-band
 *    BC rectangle. The drag is abandoned, which means the value goes back to
 *    what it was before the pointer went down: the wireframe preview drops
 *    *and* the live parameter overrides drop, or the solid would keep showing
 *    a value the source never received.
 * 2. **A half-finished command** — a constraint waiting for its second
 *    entity, a loft waiting for its second sketch.
 * 3. **An armed tool** — back to `select`. A BC pick is armed the same way
 *    and disarms here, and so does a sketch tool waiting for its click:
 *    "armed and waiting" is the whole of its pending state.
 * 4. **The selection** — including a simulation probe readout, which is the
 *    same kind of thing: something shown because the user pointed at it.
 * 5. **The mode** — back to model, which is what the hint bar has always
 *    promised ("Esc returns to model"). That promise stays true, it just
 *    stops being the *first* thing Escape does.
 *
 * When nothing on the ladder applies, Escape does nothing at all rather than
 * something invisible.
 */

/** Everything Escape can cancel, as a flat snapshot. */
export interface EscapeState {
  /** A pointer gesture that is a *command* — a handle drag, a gizmo drag, a
   * BC rectangle. Orbiting and panning are not commands and are not here. */
  gesture: boolean;
  pendingConstraint: boolean;
  pendingLoft: boolean;
  /** A tool other than `select` is armed. */
  toolArmed: boolean;
  /** The BC picker is armed, which is an armed tool by another name. */
  bcPickArmed: boolean;
  selection: boolean;
  simProbe: boolean;
  /** The editing mode is not `model`. */
  awayFromModel: boolean;
}

/** Which rung a press lands on, or null when there is nothing to cancel. */
export type EscapeLevel = "gesture" | "pending" | "tool" | "selection" | "mode";

/**
 * The one thing this press cancels.
 *
 * @param state What is currently cancellable.
 * @returns The rung, or null when Escape should do nothing.
 */
export function escapeLevel(state: EscapeState): EscapeLevel | null {
  if (state.gesture) return "gesture";
  if (state.pendingConstraint || state.pendingLoft) return "pending";
  if (state.toolArmed || state.bcPickArmed) return "tool";
  if (state.selection || state.simProbe) return "selection";
  if (state.awayFromModel) return "mode";
  return null;
}
