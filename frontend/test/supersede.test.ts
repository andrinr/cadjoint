/**
 * "The newest edit wins": the ordering rule the compile cycle is built on.
 *
 * The bug this answers was reported as *"there is a fundamental problem with
 * chaining operations"*: a second drag arriving mid-compile used to be latched
 * and run **after** the first finished, so two edits of a twenty-five-second
 * scene were a fifty-second wait showing geometry two edits old. What is
 * asserted here is the replacement, in the three properties it has to have:
 *
 * - a superseded run can never write anything (`current()` is false for it),
 *   which holds with no server and no cancellation at all;
 * - a superseded run's work is stopped, once, so it is not still burning a
 *   core in competition with the run that replaced it;
 * - a burst coalesces into one run, and — the property that matters most —
 *   **the last request of any burst always runs**. An edit that is cancelled
 *   and never re-run would leave the viewport permanently behind the code,
 *   which is a worse bug than the one being fixed.
 *
 * The timers are injected, so these are milliseconds of test rather than
 * seconds of waiting on a real clock.
 */

import { describe, expect, it } from "vitest";
import { createSuperseding, type RunToken, type Timers } from "../src/shell/supersede";

/** A clock the test advances by hand. */
function fakeTimers(): Timers & { flush: () => void; armed: () => number } {
  let pending: (() => void)[] = [];
  return {
    setTimeout: (fn) => {
      pending.push(fn);
      return fn;
    },
    clearTimeout: (handle) => {
      pending = pending.filter((fn) => fn !== handle);
    },
    flush: () => {
      const due = pending;
      pending = [];
      for (const fn of due) fn();
    },
    armed: () => pending.length,
  };
}

/** A promise a test resolves when it likes, standing in for a compile. */
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}

/** Let every already-resolved promise callback run. */
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("the revision guard", () => {
  it("keeps a run current while nothing newer has been asked for", async () => {
    const timers = fakeTimers();
    const latest = createSuperseding({ timers });
    let seen: RunToken | null = null;
    const done = latest.request(async (token) => {
      seen = token;
    });
    timers.flush();
    await done;
    expect(seen!.revision).toBe(1);
    expect(seen!.current()).toBe(true);
  });

  it("drops a late answer from a superseded run, cancel or no cancel", async () => {
    const timers = fakeTimers();
    const latest = createSuperseding({ timers });
    const slow = deferred<string>();
    const applied: string[] = [];

    // The first run is in flight and never told to stop — no cancel endpoint,
    // no abort, nothing but the guard.
    const first = latest.request(async (token) => {
      const answer = await slow.promise;
      if (!token.current()) return;
      applied.push(answer);
    });
    timers.flush();

    const second = latest.request(async (token) => {
      if (!token.current()) return;
      applied.push("second");
    });
    timers.flush();
    await second;

    // Now the superseded compile finally answers.
    slow.resolve("first");
    await first;
    await settle();

    expect(applied).toEqual(["second"]);
  });

  it("hands out strictly increasing revisions", () => {
    const timers = fakeTimers();
    const latest = createSuperseding({ timers });
    const seen: number[] = [];
    for (let index = 0; index < 5; index += 1) {
      void latest.request(async (token) => {
        seen.push(token.revision);
      });
      timers.flush();
    }
    expect(seen).toEqual([1, 2, 3, 4, 5]);
    expect(latest.revision()).toBe(5);
  });
});

describe("supersession", () => {
  it("stops the work of the run it replaces, exactly once", async () => {
    const timers = fakeTimers();
    const latest = createSuperseding({ timers });
    const slow = deferred<void>();
    let stops = 0;

    const first = latest.request(async (token) => {
      token.onSupersede(() => {
        stops += 1;
      });
      await slow.promise;
    });
    timers.flush();
    expect(stops).toBe(0);

    void latest.request(async () => {});
    expect(stops).toBe(1);

    // A third request must not stop the first a second time.
    void latest.request(async () => {});
    timers.flush();
    slow.resolve();
    await first;
    expect(stops).toBe(1);
  });

  it("releases a caller waiting on a superseded run immediately", async () => {
    const timers = fakeTimers();
    const latest = createSuperseding({ timers });
    const slow = deferred<void>();
    let released = false;

    const first = latest.request(async () => {
      await slow.promise;
    });
    timers.flush();
    void first.then(() => {
      released = true;
    });
    await settle();
    expect(released).toBe(false);

    // The patch queue is serialized behind this promise: a second edit must
    // not have to sit out the compile it just replaced.
    void latest.request(async () => {});
    await settle();
    expect(released).toBe(true);

    slow.resolve();
  });

  it("stops a run whose kill switch is registered after it was replaced", async () => {
    const timers = fakeTimers();
    const latest = createSuperseding({ timers });
    let stopped = false;
    const slow = deferred<void>();
    const first = latest.request(async (token) => {
      await slow.promise;
      token.onSupersede(() => {
        stopped = true;
      });
    });
    timers.flush();
    void latest.request(async () => {});
    slow.resolve();
    await first;
    expect(stopped).toBe(true);
  });

  it("does not stop a run that finished before anything replaced it", async () => {
    const timers = fakeTimers();
    const latest = createSuperseding({ timers });
    let stops = 0;
    const first = latest.request(async (token) => {
      token.onSupersede(() => {
        stops += 1;
      });
    });
    timers.flush();
    await first;
    void latest.request(async () => {});
    expect(stops).toBe(0);
  });
});

describe("coalescing", () => {
  it("runs one compile for a burst, of the last request's work", async () => {
    const timers = fakeTimers();
    const latest = createSuperseding({ debounceMs: 150, timers });
    const ran: string[] = [];
    for (const text of ["a", "b", "c", "d"]) {
      void latest.request(async () => {
        ran.push(text);
      });
    }
    // One timer armed, not four: every request replaced the previous one's.
    expect(timers.armed()).toBe(1);
    timers.flush();
    await settle();
    expect(ran).toEqual(["d"]);
  });

  it("never loses the last request of a burst, however long the burst", async () => {
    const timers = fakeTimers();
    const latest = createSuperseding({ debounceMs: 150, timers });
    const ran: number[] = [];
    for (let index = 0; index < 1_000; index += 1) {
      void latest.request(async () => {
        ran.push(index);
      });
    }
    timers.flush();
    await settle();
    expect(ran).toEqual([999]);
    expect(latest.active()).toBe(false);
  });

  it("leaves a request armed until it has actually run", () => {
    const timers = fakeTimers();
    const latest = createSuperseding({ debounceMs: 150, timers });
    void latest.request(async () => {});
    // The invariant behind "no edit is ever dropped": a request always leaves
    // a timer armed for itself, and the only thing that disarms it is the run
    // it starts.
    expect(timers.armed()).toBe(1);
    expect(latest.active()).toBe(true);
    timers.flush();
    expect(timers.armed()).toBe(0);
  });

  it("keeps running after a request whose work threw", async () => {
    const timers = fakeTimers();
    const latest = createSuperseding({ timers });
    const failed = latest.request(async () => {
      throw new Error("compile blew up");
    });
    timers.flush();
    await expect(failed).resolves.toBeUndefined();
    expect(latest.active()).toBe(false);

    let ran = false;
    const next = latest.request(async () => {
      ran = true;
    });
    timers.flush();
    await next;
    expect(ran).toBe(true);
  });
});
