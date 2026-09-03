import { createHash } from "node:crypto";

import { expect, test, type Page } from "@playwright/test";

/**
 * Chained edits, against a server that is really slow to answer.
 *
 * The report was *"there is a fundamental problem with chaining operations…
 * when I drag an object twice, before the first one compiled, that should
 * cancel the first drag and just apply the second one"*. Unit tests pin the
 * ordering rule; this pins the whole loop, on the real server, with two edits
 * eight seconds apart in compile time and one second apart in wall time:
 *
 * - the **second** program is what ends up on screen, never the first;
 * - the first program's compile is **cancelled** in the registry, not merely
 *   ignored — its worker had a whole core and it does not keep it;
 * - and the app is not left stale: it settles idle, showing the program that
 *   is in the editor.
 *
 * The scenes are deliberately trivial and made slow with `time.sleep`, so the
 * test measures the ordering rather than the geometry kernel.
 */

/** A program whose compile takes *seconds*, with a solid named for it. */
function slowScene(name: string, seconds: number): string {
  return [
    "import time",
    "",
    "from cadjoint.construction import Solid",
    "",
    `time.sleep(${seconds})`,
    `scene = Solid.box(size=[1, 1, 0.5], position=[0, 0, 0], name="${name}")`,
    "",
  ].join("\n");
}

const FIRST = slowScene("first_edit", 8);
const SECOND = slowScene("second_edit", 0);

/** The digest the server stamps on a job, computed the same way it does. */
function sourceHash(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

interface JobSummary {
  kind: string;
  status: string;
  source_hash: string | null;
  /** Wall seconds the job actually lived — the measure of a real kill. */
  elapsed_s: number;
}

async function jobs(page: Page): Promise<JobSummary[]> {
  const session = (await (await page.request.get("/api/session")).json()) as { token: string };
  const snapshot = (await (
    await page.request.get("/api/jobs", { headers: { "X-Cadjoint-Token": session.token } })
  ).json()) as { jobs: JobSummary[] };
  return snapshot.jobs;
}

/** Replace the whole document without compiling it. */
async function setSource(page: Page, source: string) {
  await page.evaluate((text) => {
    type EditorLike = {
      view?: { state: { doc: { length: number } }; dispatch: (spec: unknown) => void };
    };
    const content = document.querySelector("[data-testid=editor] .cm-content") as
      | (HTMLElement & { cmView?: EditorLike; cmTile?: EditorLike })
      | null;
    const view = content?.cmView?.view ?? content?.cmTile?.view;
    if (!view) throw new Error("no CodeMirror view");
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
  }, source);
}

/**
 * Wait until the app agrees with its own source again.
 *
 * The seam under the toolbar is up for exactly as long as it does not — the
 * debounce window included, which matters here because a burst spends its
 * whole life in that window. The status line is *not* the signal: while work
 * is in flight it deliberately says nothing, so that running work has one
 * indicator rather than two.
 */
async function waitForIdle(page: Page) {
  await expect(page.getByTestId("status")).not.toHaveText(/^(|Starting…)$/, {
    timeout: 180_000,
  });
  await expect(page.getByTestId("toolbar-busy")).toHaveCount(0, { timeout: 180_000 });
}

test("the newest edit wins, and the one it replaced is cancelled", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto("/");
  await waitForIdle(page);

  // The first edit. Its compile sleeps for eight seconds, which is the whole
  // point: it is still running when the second edit arrives.
  await setSource(page, FIRST);
  await page.getByTestId("run").click();

  // The app says so immediately — the indicator is up before the request has
  // even been sent, because the picture is already out of date.
  await expect(page.getByTestId("toolbar-busy")).toBeVisible();
  await expect(page.getByTestId("job-chip")).toContainText("compile");

  // Wait until this compile's own worker is genuinely running, so the cancel
  // has something to kill rather than racing an unregistered request.
  await expect
    .poll(
      async () =>
        (await jobs(page)).some(
          (job) =>
            job.kind === "compile" &&
            job.status === "running" &&
            job.source_hash === sourceHash(FIRST),
        ),
      { timeout: 60_000 },
    )
    .toBe(true);

  // The second edit, while the first is still compiling.
  await setSource(page, SECOND);
  await page.getByTestId("run").click();

  await waitForIdle(page);

  // 1. What landed is the second program.
  await expect(page.getByTestId("object-tree-panel")).toContainText("second_edit");
  await expect(page.getByTestId("object-tree-panel")).not.toContainText("first_edit");

  // 2. The first compile was stopped, not waited out.
  const registry = await jobs(page);
  const first = registry.find(
    (job) => job.kind === "compile" && job.source_hash === sourceHash(FIRST),
  );
  expect(first, "the first edit's compile should be in the registry").toBeDefined();
  expect(first!.status).toBe("cancelled");
  // Measured, not asserted by name: the worker slept for eight seconds and
  // the job did not live that long, so it was killed rather than abandoned to
  // finish an answer nobody wanted. That difference is a whole core and about
  // a gigabyte, for as long as the compile it was competing with runs.
  console.log(`superseded compile lived ${first!.elapsed_s.toFixed(2)}s of an 8s sleep`);
  expect(first!.elapsed_s).toBeLessThan(6);

  const second = registry.find(
    (job) => job.kind === "compile" && job.source_hash === sourceHash(SECOND),
  );
  expect(second!.status).toBe("done");

  // 3. And the app is not left stale: idle, nothing outstanding, and the Run
  //    button back to plain "Run" — the dot it wears while the document is
  //    ahead of the last compile is gone.
  await expect(page.getByTestId("toolbar-busy")).toHaveCount(0);
  await expect(page.getByTestId("run")).not.toContainText("•");
  // Nothing of the compile pipeline is still burning a core. Deliberately
  // narrowed to compiles: the session's mesh warm-up is a tracked job of its
  // own that outlives a compile on purpose, and it is exactly the case that
  // makes the chip say "+1 more" rather than pretend one thing is running.
  expect(
    registry.filter((job) => job.kind === "compile" && job.status === "running"),
  ).toEqual([]);
});

test("a burst of edits compiles once, for the last of them", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto("/");
  await waitForIdle(page);

  const before = (await jobs(page)).filter((job) => job.kind === "compile").length;
  const texts = [0, 1, 2, 3, 4].map((index) => slowScene(`burst_${index}`, 0));

  // Five edits in one turn of the event loop: a handle dragged back and
  // forth, or a panel emitting a chain of patches. Driven from inside the
  // page rather than over the wire, because a round trip per click would put
  // them further apart than any coalescing window should reach.
  await page.evaluate((sources) => {
    type EditorLike = {
      view?: { state: { doc: { length: number } }; dispatch: (spec: unknown) => void };
    };
    const content = document.querySelector("[data-testid=editor] .cm-content") as
      | (HTMLElement & { cmView?: EditorLike; cmTile?: EditorLike })
      | null;
    const view = content?.cmView?.view ?? content?.cmTile?.view;
    if (!view) throw new Error("no CodeMirror view");
    const run = document.querySelector("[data-testid=run]") as HTMLButtonElement;
    for (const text of sources) {
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
      run.click();
    }
  }, texts);

  await waitForIdle(page);

  await expect(page.getByTestId("object-tree-panel")).toContainText("burst_4");
  const after = (await jobs(page)).filter((job) => job.kind === "compile");
  // One compile, not five: the burst coalesced rather than starting (and
  // then killing) four programs nobody asked to see.
  expect(after.length - before).toBe(1);
  expect(after[0].source_hash).toBe(sourceHash(texts[4]));
  expect(after[0].status).toBe("done");
});
