/**
 * The compile cycle's ordering: what lands on screen after chained edits.
 *
 * `supersede.test.ts` states the rule in the abstract; this states it in the
 * app's own terms, over the real `createCompileCycle` with only the network
 * and the renderer replaced. The claims are the ones the user's report turns
 * into: two edits in quick succession show the **second** one's geometry, the
 * first one's worker is killed rather than merely ignored, a burst of patches
 * is one compile of the final program, and the app never ends a burst busy or
 * showing a program that is not the one in the editor.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { createRoot, createSignal } from "solid-js";
import { deferred, settle } from "./helpers";

const compile = vi.fn();
const mesh = vi.fn();
const patch = vi.fn();
const cancelClientJob = vi.fn(async (_clientId: string) => 1);

vi.mock("../src/api", () => ({
  compile: (...args: unknown[]) => compile(...args),
  mesh: (...args: unknown[]) => mesh(...args),
  patch: (...args: unknown[]) => patch(...args),
}));

vi.mock("../src/jobs", () => ({
  cancelClientJob: (...args: unknown[]) => cancelClientJob(...(args as [string])),
  nextRequestId: () => `req-${(requestIds += 1)}`,
  watchJobs: () => {
    watchers.hold();
    return watchers.release;
  },
}));

/**
 * The job poller, counted rather than stubbed away.
 *
 * A compile has to hold a watcher for as long as its request is out: the
 * registry's own id arrives on a 1 Hz poll, and a poll that stops before the
 * worker is registered leaves the chip unable to name — or stop — the very
 * compile worth stopping. Balanced release is the property, so the counter
 * is asserted rather than merely accepted.
 */
const watchers = {
  held: 0,
  peak: 0,
  hold(): void {
    watchers.held += 1;
    watchers.peak = Math.max(watchers.peak, watchers.held);
  },
  release(): void {
    watchers.held -= 1;
  },
};

let requestIds = 0;

const { createCompileCycle } = await import("../src/shell/compileCycle");
const { busy, meshEdges, nodes, source, setSource, setViewerError, status, viewerError } =
  await import("../src/state");

/** A compile response naming the program it came from. */
function answer(name: string) {
  return {
    ok: true,
    sdf: name,
    preview_shader: `preview:${name}`,
    path_shader: `path:${name}`,
    present_shader: `present:${name}`,
    construction: [
      {
        id: name,
        name,
        kind: "solid",
        line: null,
        vertices: [],
        edges: [],
        spans: {},
      },
    ],
    relations: [],
    solver_runs: [],
    materials: [],
    mesh_edges: null,
    output: "",
  };
}


const DEBOUNCE = 20;

/** The cycle, with the two things it talks to stubbed out. */
function cycle(display: () => { showMeshEdges: boolean; showMeshWireframe: boolean } = () => ({
  showMeshEdges: false,
  showMeshWireframe: false,
})) {
  return createRoot(() =>
    createCompileCycle({
      renderer: { setShaders: async () => {} } as never,
      history: {
        commit: () => {},
        bump: () => {},
        canUndo: () => false,
        canRedo: () => false,
        stepBack: () => null,
        stepForward: () => null,
      },
      display: display as never,
      debounceMs: DEBOUNCE,
    }),
  );
}

beforeEach(() => {
  compile.mockReset();
  mesh.mockReset();
  patch.mockReset();
  cancelClientJob.mockClear();
  watchers.held = 0;
  watchers.peak = 0;
  setSource("");
  setViewerError("");
});

describe("chained edits", () => {
  it("shows the second edit's geometry, not the first's", async () => {
    const slow = deferred<ReturnType<typeof answer>>();
    compile.mockImplementationOnce(() => slow.promise);
    compile.mockImplementationOnce(async () => answer("second"));

    const app = cycle();
    setSource("first");
    const first = app.run();
    await settle(DEBOUNCE + 10);

    setSource("second");
    const second = app.run();
    await settle(DEBOUNCE + 10);
    await second;

    // The first compile only now comes back — the worst case for a design
    // that trusted cancellation to be timely.
    slow.resolve(answer("first"));
    await first;
    await settle();

    expect(nodes().map((node) => node.id)).toEqual(["second"]);
    expect(source()).toBe("second");
    expect(busy()).toBe(false);
  });

  it("kills the superseded worker rather than only ignoring its answer", async () => {
    const slow = deferred<ReturnType<typeof answer>>();
    compile.mockImplementationOnce(() => slow.promise);
    compile.mockImplementationOnce(async () => answer("second"));

    const app = cycle();
    setSource("first");
    void app.run();
    await settle(DEBOUNCE + 10);
    const firstClientId = compile.mock.calls[0][1].clientId as string;
    expect(firstClientId).toBeTruthy();

    setSource("second");
    await app.run();

    expect(cancelClientJob).toHaveBeenCalledWith(firstClientId);
    // And the request this side is holding open was dropped too.
    expect((compile.mock.calls[0][1].signal as AbortSignal).aborted).toBe(true);
    slow.resolve(answer("first"));
  });

  it("compiles once for a burst of edits, of the last one", async () => {
    compile.mockImplementation(async (text: string) => answer(text));
    const app = cycle();

    for (const text of ["a", "b", "c", "d"]) {
      setSource(text);
      void app.run();
    }
    await settle(DEBOUNCE + 20);

    expect(compile).toHaveBeenCalledTimes(1);
    expect(compile.mock.calls[0][0]).toBe("d");
    expect(nodes().map((node) => node.id)).toEqual(["d"]);
  });

  it("ends a burst with the editor's program compiled and the app idle", async () => {
    compile.mockImplementation(async (text: string) => {
      await settle(5);
      return answer(text);
    });
    const app = cycle();

    // Edits spread over several debounce windows, the way a person drags.
    for (const text of ["a", "b", "c"]) {
      setSource(text);
      void app.run();
      await settle(DEBOUNCE + 5);
    }
    await settle(DEBOUNCE + 40);

    // The failure this must never introduce: an edit cancelled and never
    // re-run, leaving the viewport permanently behind the code.
    expect(nodes().map((node) => node.id)).toEqual([source()]);
    expect(busy()).toBe(false);
    expect(status().kind).toBe("ready");
  });

  it("says it is busy from the edit, not from the request", async () => {
    compile.mockImplementation(async (text: string) => answer(text));
    const app = cycle();
    setSource("a");
    const done = app.run();
    // Still inside the debounce window: nothing has been sent, and the app
    // already says the picture is not the picture of the code. The saying is
    // `busy` — one indicator, the toolbar's chip, reads it.
    expect(compile).not.toHaveBeenCalled();
    expect(busy()).toBe(true);
    // And the status line is silent rather than saying the same thing twice.
    expect(status().text).toBe("");
    await done;
    expect(busy()).toBe(false);
    expect(status().text).toBe("Scene compiled");
  });

  it("keeps the job poller alive for exactly as long as the request is out", async () => {
    const slow = deferred<ReturnType<typeof answer>>();
    compile.mockImplementationOnce(() => slow.promise);

    const app = cycle();
    setSource("a");
    const done = app.run();
    await settle(DEBOUNCE + 10);

    // The chip can only name the compile, and only offer its ×, once the
    // registry's poll has returned the job id. One nudge at send time races
    // the registration and can lose; a held watcher cannot.
    expect(watchers.held).toBe(1);

    slow.resolve(answer("a"));
    await done;
    // And an idle playground goes quiet again rather than polling forever.
    expect(watchers.held).toBe(0);
  });

  it("stops watching even when the compile fails", async () => {
    compile.mockImplementationOnce(async () => {
      throw new Error("boom");
    });
    const app = cycle();
    setSource("a");
    await app.run();
    expect(watchers.peak).toBe(1);
    expect(watchers.held).toBe(0);
    expect(status().kind).toBe("error");
  });

  it("does not report a superseded compile's failure", async () => {
    const doomed = deferred<never>();
    compile.mockImplementationOnce(() => doomed.promise);
    compile.mockImplementationOnce(async () => answer("second"));

    const app = cycle();
    setSource("first");
    const first = app.run();
    await settle(DEBOUNCE + 10);
    setSource("second");
    await app.run();

    doomed.resolve(new Error("aborted") as never);
    await first.catch(() => undefined);
    await settle();

    expect(status().kind).toBe("ready");
    expect(nodes().map((node) => node.id)).toEqual(["second"]);
  });

  it("keeps a patch queue moving instead of making it wait out a compile", async () => {
    const slow = deferred<ReturnType<typeof answer>>();
    compile.mockImplementationOnce(() => slow.promise);
    compile.mockImplementation(async (text: string) => answer(text));
    patch.mockImplementation(async (body: { source: string }) => ({
      ok: true,
      source: `${body.source}+`,
    }));

    const app = cycle();
    setSource("p");
    void app.applyPatch({ op: "one" });
    await settle(DEBOUNCE + 10);

    // The second edit is issued while the first compile is still in flight;
    // it must go out at once, from the source the first patch produced.
    void app.applyPatch({ op: "two" });
    await settle(DEBOUNCE + 20);

    expect(patch).toHaveBeenCalledTimes(2);
    expect(patch.mock.calls[1][0].source).toBe("p+");
    expect(source()).toBe("p++");
    slow.resolve(answer("p+"));
    await settle(DEBOUNCE + 20);
    expect(nodes().map((node) => node.id)).toEqual(["p++"]);
  });
});

describe("the feature-edge overlay", () => {
  const EDGES = { sharp: [[0, 0, 0, 1, 0, 0]], wire: [], resolution: 32, edges: "graph" };

  /** A cycle with the overlay switch under the test's control, switched on. */
  function withOverlay() {
    const [display, setDisplay] = createSignal({ showMeshEdges: true, showMeshWireframe: false });
    return { app: cycle(display), setDisplay };
  }

  it("asks again when the same program compiles a second time", async () => {
    compile.mockImplementation(async (text: string) => answer(text));
    mesh.mockImplementation(async () => ({ ok: true, mesh_edges: EDGES }));
    const { app } = withOverlay();
    setSource("same");
    await app.run();
    await settle();
    expect(mesh).toHaveBeenCalledTimes(1);
    expect(meshEdges()).not.toBeNull();

    // Run again with no edit: the compile clears the overlay, so it has to
    // be asked for again rather than found already "requested".
    await app.run();
    await settle();
    expect(mesh).toHaveBeenCalledTimes(2);
    expect(meshEdges()).not.toBeNull();
  });

  it("reports a failed extraction, and the switch tries again", async () => {
    compile.mockImplementation(async (text: string) => answer(text));
    mesh.mockImplementationOnce(async () => {
      throw new Error("Server returned a non-JSON response (504).");
    });
    mesh.mockImplementationOnce(async () => ({ ok: true, mesh_edges: EDGES }));
    const { app, setDisplay } = withOverlay();
    setSource("part");
    await app.run();
    await settle();
    expect(viewerError()).toContain("Feature edges unavailable");
    expect(viewerError()).toContain("504");
    expect(meshEdges()).toBeNull();

    setDisplay({ showMeshEdges: false, showMeshWireframe: false });
    setDisplay({ showMeshEdges: true, showMeshWireframe: false });
    await settle();
    expect(mesh).toHaveBeenCalledTimes(2);
    expect(meshEdges()).not.toBeNull();
    expect(viewerError()).toBe("");
  });

  it("reports a refusal from the server, not only a network failure", async () => {
    compile.mockImplementation(async (text: string) => answer(text));
    mesh.mockImplementation(async () => ({ ok: false, error: "mesh extraction timed out" }));
    const { app } = withOverlay();
    setSource("part");
    await app.run();
    await settle();
    expect(viewerError()).toBe("Feature edges unavailable: mesh extraction timed out");
  });

  it("asks again when a failed compile aborted the extraction in flight", async () => {
    compile.mockImplementationOnce(async (text: string) => answer(text));
    compile.mockImplementationOnce(async () => ({ ok: false, error: "SyntaxError" }));
    const first = deferred<unknown>();
    mesh.mockImplementationOnce(
      (_source: string, options: { signal: AbortSignal }) =>
        new Promise((resolve, reject) => {
          options.signal.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          );
          void first.promise.then(resolve);
        }),
    );
    mesh.mockImplementationOnce(async () => ({ ok: true, mesh_edges: EDGES }));
    const { app } = withOverlay();
    setSource("good");
    await app.run();
    await settle();
    expect(mesh).toHaveBeenCalledTimes(1);

    // A broken edit compiles, fails, and never publishes: the program on
    // screen is still "good", and it still wants its edges.
    setSource("bad");
    await app.run();
    await settle();
    expect(mesh).toHaveBeenCalledTimes(2);
    expect(mesh.mock.calls[1][0]).toBe("good");
    expect(meshEdges()).not.toBeNull();
    expect(viewerError()).toBe("");
  });
});
