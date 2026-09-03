/**
 * "The newest edit wins": one scheduler, for work only the last answer of
 * which is wanted.
 *
 * A compile of the motor shield costs twenty-five seconds and a whole core.
 * Two drags in quick succession must therefore not mean fifty seconds of
 * waiting with the *older* geometry on screen — which is what a "remember to
 * run again afterwards" latch produces. What is wanted is the opposite: the
 * second edit replaces the first, the first is stopped where it stands, and
 * its answer is discarded whether or not the stop landed in time.
 *
 * Those are two separate properties and this module keeps them separate:
 *
 * - **Correctness — the revision guard.** Every run carries a monotonically
 *   increasing revision. A run whose revision is no longer the newest is not
 *   current, and a caller that checks `token.current()` before writing
 *   anything can never apply a superseded answer. This holds with no server,
 *   no cancellation and no cooperation from the network.
 * - **Performance — supersession.** A run that stops being current has its
 *   registered `stop` called once, so the work behind it (a worker process, a
 *   socket) can be killed rather than left burning a core in competition with
 *   the run the user is actually waiting for.
 *
 * And one comfort: **coalescing.** A burst of requests inside `debounceMs`
 * starts one run, of the last request's work — a drag that patches on every
 * release, or a panel sequence of constraint → satisfy → extrude, is one
 * compile of the final program rather than three of three intermediate ones.
 *
 * ### Why no request can be lost
 *
 * The failure this exists to avoid is an edit that is cancelled and then never
 * re-run, leaving the viewport permanently behind the code. It cannot happen,
 * for four reasons that are each visible in `request` and `start` below:
 *
 * 1. `request` always leaves `pending` non-null with a timer armed for it.
 * 2. `start` — the only thing that clears `pending` — always begins that work.
 * 3. `revision` increases only in `request`, and the work `start` begins
 *    carries the revision of the newest request made so far.
 * 4. A run therefore stops being current only because a *newer* request
 *    exists, and by (1) that newer request has its own armed timer.
 *
 * So in any finite burst the last request is started (by 1–3) and, having no
 * successor, is never superseded (by 4): its answer is the one that lands.
 * `test/supersede.test.ts` states each of these as a test.
 *
 * Nothing here touches Solid, the network or the DOM: the timers are
 * injectable and the work is a callback, which is what makes the rules above
 * testable in milliseconds rather than in compiles.
 */

/** The handle one run is given: its place in the order, and its kill switch. */
export interface RunToken {
  /** Monotonic request number. Higher is newer. */
  readonly revision: number;
  /** Whether this is still the newest run — false once superseded. */
  current: () => boolean;
  /**
   * Register how to stop this run's work; called at most once, and
   * immediately if the run has already been superseded.
   */
  onSupersede: (stop: () => void) => void;
}

/** The work of one run. Errors are the caller's to handle; see `request`. */
export type Work = (token: RunToken) => Promise<void>;

/** Just enough of the timer API to be swapped out in a test. */
export interface Timers {
  setTimeout: (fn: () => void, ms: number) => unknown;
  clearTimeout: (handle: unknown) => void;
}

export interface SupersedeOptions {
  /**
   * Milliseconds of quiet before a requested run starts.
   *
   * Zero starts on the next turn of the event loop, which still coalesces a
   * synchronous burst but not a burst spread over pointer events.
   */
  debounceMs?: number;
  timers?: Timers;
}

export interface Superseding {
  /**
   * Queue *work* as the newest run.
   *
   * The promise settles when the work finishes **or** when a newer request
   * replaces it, whichever comes first, and never rejects — a caller
   * serialized behind it (the patch queue) must advance either way, and a
   * superseded run has no answer worth waiting for. Errors thrown by *work*
   * are swallowed here precisely because the guard means a caller has to
   * handle its own failures inside the run, where it can still ask whether it
   * is the run whose failure is worth reporting.
   */
  request: (work: Work) => Promise<void>;
  /** The newest revision handed out. */
  revision: () => number;
  /** Whether a run is scheduled or in flight and still current. */
  active: () => boolean;
}

const DEFAULT_TIMERS: Timers = {
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};

export function createSuperseding(options: SupersedeOptions = {}): Superseding {
  const debounceMs = options.debounceMs ?? 0;
  const timers = options.timers ?? DEFAULT_TIMERS;

  let revision = 0;
  /** The newest requested run that has not started yet. */
  let pending: { revision: number; work: Work; settle: () => void } | null = null;
  let timer: unknown = null;
  /** The run in flight, while it is still the newest one. */
  let running: { revision: number; stop: (() => void) | null; settle: () => void } | null =
    null;

  const disarm = (): void => {
    if (timer === null) return;
    timers.clearTimeout(timer);
    timer = null;
  };

  /** Resolve exactly once, however the run ends. */
  const once = (settle: () => void): (() => void) => {
    let done = false;
    return () => {
      if (done) return;
      done = true;
      settle();
    };
  };

  /**
   * Retire the run a newer request has just replaced.
   *
   * Two things happen to it and neither is "wait for it": its caller is
   * released (a queue serialized behind a superseded compile must advance to
   * the edit that replaced it, not sit out the twenty-five seconds), and the
   * work behind it is stopped. The run's own answer, if it still arrives, is
   * dropped by its revision guard rather than by anything here.
   */
  const supersedeCurrent = (): void => {
    if (pending) {
      // Never started, so there is nothing to kill.
      const stale = pending;
      pending = null;
      disarm();
      stale.settle();
    }
    if (running) {
      const stale = running;
      running = null;
      const stop = stale.stop;
      stale.stop = null;
      stop?.();
      stale.settle();
    }
  };

  const start = (): void => {
    timer = null;
    const entry = pending;
    pending = null;
    if (!entry) return;
    const live = {
      revision: entry.revision,
      stop: null as (() => void) | null,
      settle: once(entry.settle),
    };
    running = live;
    const token: RunToken = {
      revision: entry.revision,
      current: () => entry.revision === revision,
      onSupersede: (stop) => {
        // Registered after a supersession — which cannot happen while the
        // work runs synchronously up to its first await, but is cheap to be
        // right about — means stop now rather than never.
        if (entry.revision !== revision) {
          stop();
          return;
        }
        live.stop = stop;
      },
    };
    void (async () => {
      try {
        await entry.work(token);
      } catch {
        // A run's failure is its own business: it still has `token.current()`
        // and can report what it likes. Here it only means "over".
      } finally {
        if (running === live) running = null;
        live.settle();
      }
    })();
  };

  const request = (work: Work): Promise<void> => {
    revision += 1;
    supersedeCurrent();
    const mine = revision;
    return new Promise<void>((resolve) => {
      pending = { revision: mine, work, settle: resolve };
      disarm();
      timer = timers.setTimeout(start, debounceMs);
    });
  };

  return {
    request,
    revision: () => revision,
    active: () => pending !== null || running !== null,
  };
}
