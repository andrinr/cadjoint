/**
 * What the toolbar's running-work chip says, given everything that is running.
 *
 * Split out of `Toolbar.tsx` because it is the answer to a question the user
 * asked out loud — *"if there is multiple running processes, are they all
 * shown there?"* — and an answer worth stating in a test rather than only in
 * pixels. Nothing here knows about Solid or the DOM: it takes the job
 * registry's running set plus the app's own `busy` clock and returns the
 * sentence.
 *
 * ### The rules
 *
 * 1. **The compile is not read off the registry alone.** It is registered like
 *    everything else, but the poll is up to a second behind and the app is
 *    already behind its own code before the request is even sent — the
 *    debounce window, the worker spawn. `busy` is the exact answer to "is the
 *    picture on screen the picture of the code", so the compile row exists
 *    from the edit, and adopts the registry's job id as soon as one appears.
 *    That is the case the deleted in-viewport indicator used to cover.
 * 2. **A lint is never a chip.** It fires on a pause in typing, which would
 *    make the chip blink several times a minute while you write code.
 * 3. **The compile leads.** It is the one job that decides whether what is on
 *    screen is the program in the editor; everything else is work you started
 *    on purpose and can watch in the Processes window.
 * 4. **Nothing running is hidden.** The chip names one job — the one its ×
 *    stops — and then says how many more there are. The Processes window
 *    lists them in full: one vocabulary, two depths.
 */

import type { JobKind, RunningJob } from "../jobs";

/**
 * The kinds worth a chip beside the compile: work you start and then look
 * away from. `warmup` is here because the session's mesh warm-up genuinely
 * occupies the machine for as long as it runs, and the honest answer to "what
 * is running" includes it.
 */
export const CHIP_KINDS: ReadonlySet<JobKind> = new Set<JobKind>([
  "simulate",
  "optimize",
  "mesh",
  "mesh_inspect",
  "export",
  "warmup",
]);

/**
 * The rows of the chip, most important first.
 *
 * @param running The registry's running jobs, newest first.
 * @param compilingSinceMs When the app stopped agreeing with its own source,
 *   by the client clock, or 0 when it agrees. The compile's seconds are
 *   counted from here rather than from the worker's start, because that is
 *   the interval the person waiting actually experiences.
 * @param nowMs The current client time.
 */
export function chipJobs(
  running: readonly RunningJob[],
  compilingSinceMs: number,
  nowMs: number,
): RunningJob[] {
  const others = running.filter((job) => job.kind !== "compile" && CHIP_KINDS.has(job.kind));
  if (!compilingSinceMs) return others;
  const live = running.find((job) => job.kind === "compile");
  return [
    {
      // Empty until the poll brings the id back; the × stays boxed and dead
      // for that moment rather than appearing under the pointer.
      id: live?.id ?? "",
      kind: "compile",
      // No name: the registry labels an unnamed job with its source hash, and
      // a name arriving a second late would grow the chip under the pointer.
      name: "",
      elapsed_s: Math.max(0, (nowMs - compilingSinceMs) / 1_000),
    },
    ...others,
  ];
}

/**
 * What the × will stop, said in full.
 *
 * Never "cancel this job": the button stops exactly one of what may be
 * several running things, so it has to say which. Before the registry has
 * named the compile there is nothing to stop yet, and it says that instead.
 */
export function cancelLabel(job: RunningJob | undefined): string {
  if (!job) return "";
  if (!job.id) return `Starting ${job.kind}…`;
  return `Cancel ${job.kind}${job.name ? ` ${job.name}` : ""}`;
}

/** `+2 more`, or "" when the chip already names everything that is running. */
export function othersLabel(count: number): string {
  return count > 0 ? `+${count} more` : "";
}
