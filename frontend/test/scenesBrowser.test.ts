import { describe, expect, it } from "vitest";
import { byId } from "../src/identity";
import {
  createQueue,
  firstLine,
  formatFileSize,
  formatModified,
} from "../src/thumbnails";

describe("the thumbnail queue", () => {
  it("runs jobs one at a time, in order", async () => {
    const queue = createQueue();
    const order: string[] = [];
    /** Resolvers, so each job can be held open deliberately. */
    const gates: (() => void)[] = [];
    const job = (name: string) => () =>
      new Promise<string>((resolve) => {
        order.push(`start ${name}`);
        gates.push(() => {
          order.push(`end ${name}`);
          resolve(name);
        });
      });

    /** Let every already-queued microtask run. */
    const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

    const first = queue.push(job("a"));
    const second = queue.push(job("b"));
    await flush();

    // The second job has not started: one at a time is the whole point.
    expect(order).toEqual(["start a"]);
    expect(queue.pending()).toBe(2);

    gates[0]();
    expect(await first).toBe("a");
    await flush();
    expect(order).toEqual(["start a", "end a", "start b"]);

    gates[1]();
    expect(await second).toBe("b");
    expect(order).toEqual(["start a", "end a", "start b", "end b"]);
    await flush();
    expect(queue.pending()).toBe(0);
  });

  it("keeps going after a job throws", async () => {
    const queue = createQueue();
    const failed = queue.push(async () => {
      throw new Error("no WebGPU");
    });
    await expect(failed).rejects.toThrow("no WebGPU");
    await expect(queue.push(async () => "next")).resolves.toBe("next");
  });
});

describe("placeholder text", () => {
  it("takes the first non-empty line of a compile error", () => {
    expect(firstLine("SyntaxError: bad token\n  File scene.py, line 4\n")).toBe(
      "SyntaxError: bad token",
    );
    expect(firstLine("\n\n  Traceback\nmore")).toBe("Traceback");
  });

  it("has something to say about nothing", () => {
    expect(firstLine(null)).toBe("This scene did not compile.");
    expect(firstLine("   ")).toBe("This scene did not compile.");
  });
});

describe("card metadata", () => {
  it("formats the listing's ISO stamp, and refuses junk", () => {
    expect(formatModified("2026-03-04T10:11:12Z")).toMatch(/2026/);
    expect(formatModified(null)).toBe("–");
    expect(formatModified("not a date")).toBe("–");
  });

  it("formats file sizes on the same scale the monitor uses", () => {
    expect(formatFileSize(940)).toBe("940 B");
    expect(formatFileSize(8_737)).toBe("8.7 kB");
    expect(formatFileSize(1_400_000)).toBe("1.4 MB");
    expect(formatFileSize(undefined)).toBe("–");
  });
});

describe("addressing a declaration", () => {
  it("names an entry by its stable id when it has one", () => {
    expect(byId({ stableId: "study:sink-conduction" })).toEqual({
      id: "study:sink-conduction",
    });
  });

  it("says nothing rather than null when the identity table could not", () => {
    // A key that says nothing is worse than an absent key: the positional
    // handle beside it is doing the work, and `id: null` reads like a bug.
    expect(byId({ stableId: null })).toEqual({});
    expect(byId({})).toEqual({});
  });
});
