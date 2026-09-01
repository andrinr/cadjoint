import { describe, expect, it } from "vitest";
import {
  FLYOUT_HOVER_OPEN_MS,
  FLYOUT_LEAVE_CLOSE_MS,
  FlyoutController,
  loadLastUsed,
  persistLastUsed,
  type FlyoutTimers,
} from "../src/flyout";

/** Deterministic timer queue the tests can advance by hand. */
function fakeTimers() {
  let nextId = 1;
  const pending = new Map<number, { at: number; run: () => void }>();
  let now = 0;
  const timers: FlyoutTimers = {
    set: (callback, delay) => {
      const id = nextId++;
      pending.set(id, { at: now + delay, run: callback });
      return id;
    },
    clear: (id) => void pending.delete(id),
  };
  const advance = (ms: number) => {
    now += ms;
    for (const [id, timer] of [...pending]) {
      if (timer.at <= now) {
        pending.delete(id);
        timer.run();
      }
    }
  };
  return { timers, advance, pendingCount: () => pending.size };
}

function controller() {
  const events: (string | null)[] = [];
  const { timers, advance, pendingCount } = fakeTimers();
  const flyouts = new FlyoutController((open) => events.push(open), timers);
  return { flyouts, events, advance, pendingCount };
}

describe("flyout state machine", () => {
  it("click toggles a group open and closed", () => {
    const { flyouts } = controller();
    flyouts.toggle("create");
    expect(flyouts.openGroup()).toBe("create");
    flyouts.toggle("create");
    expect(flyouts.openGroup()).toBeNull();
  });

  it("clicking another parent moves the flyout there", () => {
    const { flyouts } = controller();
    flyouts.toggle("create");
    flyouts.toggle("transform");
    expect(flyouts.openGroup()).toBe("transform");
  });

  it("opens on hover only after the dwell delay", () => {
    const { flyouts, advance } = controller();
    flyouts.pointerEnter("create");
    expect(flyouts.openGroup()).toBeNull();
    advance(FLYOUT_HOVER_OPEN_MS - 1);
    expect(flyouts.openGroup()).toBeNull();
    advance(1);
    expect(flyouts.openGroup()).toBe("create");
  });

  it("a touch-and-leave before the dwell never opens", () => {
    const { flyouts, advance, pendingCount } = controller();
    flyouts.pointerEnter("create");
    flyouts.pointerLeave("create");
    advance(FLYOUT_HOVER_OPEN_MS + 10);
    expect(flyouts.openGroup()).toBeNull();
    expect(pendingCount()).toBe(0);
  });

  it("glides between groups instantly while one is open", () => {
    const { flyouts, advance } = controller();
    flyouts.toggle("create");
    flyouts.pointerEnter("transform");
    expect(flyouts.openGroup()).toBe("transform");
    advance(1000);
    expect(flyouts.openGroup()).toBe("transform");
  });

  it("closes after the mouse-leave delay, not immediately", () => {
    const { flyouts, advance } = controller();
    flyouts.toggle("create");
    flyouts.pointerLeave("create");
    expect(flyouts.openGroup()).toBe("create");
    advance(FLYOUT_LEAVE_CLOSE_MS - 1);
    expect(flyouts.openGroup()).toBe("create");
    advance(1);
    expect(flyouts.openGroup()).toBeNull();
  });

  it("returning before the leave delay keeps the flyout open", () => {
    const { flyouts, advance } = controller();
    flyouts.toggle("create");
    flyouts.pointerLeave("create");
    flyouts.pointerEnter("create");
    advance(FLYOUT_LEAVE_CLOSE_MS + 10);
    expect(flyouts.openGroup()).toBe("create");
  });

  it("selection and Escape close at once and cancel timers", () => {
    const first = controller();
    first.flyouts.toggle("create");
    first.flyouts.select();
    expect(first.flyouts.openGroup()).toBeNull();

    const second = controller();
    second.flyouts.pointerEnter("create");
    second.flyouts.dismiss();
    second.advance(FLYOUT_HOVER_OPEN_MS + 10);
    expect(second.flyouts.openGroup()).toBeNull();
    expect(second.pendingCount()).toBe(0);
  });

  it("reports every transition to the subscriber exactly once", () => {
    const { flyouts, events } = controller();
    flyouts.toggle("create");
    flyouts.toggle("create");
    flyouts.dismiss(); // Already closed: no event.
    expect(events).toEqual(["create", null]);
  });
});

describe("last-used persistence", () => {
  const VALID = { create: ["sketch", "box"], transform: ["translate", "scale"] };

  function memoryStorage(initial: string | null = null) {
    let value = initial;
    return {
      getItem: () => value,
      setItem: (_key: string, next: string) => {
        value = next;
      },
    };
  }

  it("round-trips the per-group choice", () => {
    const storage = memoryStorage();
    persistLastUsed({ create: "box", transform: "scale" }, storage);
    expect(loadLastUsed(VALID, storage)).toEqual({ create: "box", transform: "scale" });
  });

  it("drops unknown groups and stale children", () => {
    const storage = memoryStorage(
      JSON.stringify({ create: "warp-drive", legacy: "sketch", transform: "scale" }),
    );
    expect(loadLastUsed(VALID, storage)).toEqual({ transform: "scale" });
  });

  it("tolerates corrupt and missing storage", () => {
    expect(loadLastUsed(VALID, memoryStorage("{not json"))).toEqual({});
    expect(loadLastUsed(VALID, memoryStorage())).toEqual({});
    expect(loadLastUsed(VALID, undefined)).toEqual({});
  });
});
