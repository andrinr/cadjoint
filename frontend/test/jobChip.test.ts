/**
 * The one running-work indicator, and what it says when several things run.
 *
 * There used to be two: a chip in the top bar, backed by the job registry and
 * cancellable, and a "COMPILING" overlay in the viewport driven by `busy`.
 * The chip survives, so it has to cover the case the overlay covered — a
 * compile the registry has not registered yet — and it has to answer the
 * question that came with the request: *"if there is multiple running
 * processes, are they all shown there?"*
 */

import { describe, expect, it } from "vitest";
import { CHIP_KINDS, cancelLabel, chipJobs, othersLabel } from "../src/components/jobChip";
import type { RunningJob } from "../src/jobs";

const NOW = 1_700_000_000_000;

function job(kind: RunningJob["kind"], id: string, name = ""): RunningJob {
  return { id, kind, name, elapsed_s: 3 };
}

describe("what the chip is for", () => {
  it("shows a compile the registry has not heard of yet", () => {
    // The case the deleted viewport overlay covered: the edit has been made,
    // the debounce has not elapsed, no job exists. The app is behind its own
    // code and must say so.
    const rows = chipJobs([], NOW - 2_000, NOW);
    expect(rows).toHaveLength(1);
    expect(rows[0].kind).toBe("compile");
    expect(rows[0].id).toBe("");
    expect(rows[0].elapsed_s).toBeCloseTo(2);
  });

  it("adopts the registry's id for that compile as soon as one appears", () => {
    const rows = chipJobs([job("compile", "job-000007")], NOW - 2_000, NOW);
    expect(rows[0].id).toBe("job-000007");
    // Still the client's clock, which starts at the edit rather than at the
    // worker — that is the interval the person is counting.
    expect(rows[0].elapsed_s).toBeCloseTo(2);
  });

  it("shows nothing for a compile that has landed", () => {
    expect(chipJobs([], 0, NOW)).toEqual([]);
  });

  it("never chips a lint", () => {
    expect(CHIP_KINDS.has("lint")).toBe(false);
    expect(chipJobs([job("lint", "job-1")], 0, NOW)).toEqual([]);
  });

  it("never shows the same compile twice", () => {
    // The registry row and the busy clock are two views of one compile.
    const rows = chipJobs([job("compile", "job-1")], NOW, NOW);
    expect(rows.filter((row) => row.kind === "compile")).toHaveLength(1);
  });
});

describe("several things running at once", () => {
  const running = [
    job("optimize", "job-9", "shield_mass"),
    job("simulate", "job-8", "static_load"),
    job("warmup", "job-7"),
  ];

  it("accounts for every running job, naming one and counting the rest", () => {
    const rows = chipJobs(running, NOW - 1_000, NOW);
    // Four things are running; none is silently dropped.
    expect(rows).toHaveLength(4);
    expect(othersLabel(rows.length - 1)).toBe("+3 more");
  });

  it("leads with the compile, because it decides whether the picture is current", () => {
    expect(chipJobs(running, NOW, NOW)[0].kind).toBe("compile");
  });

  it("leads with the newest job when nothing is compiling", () => {
    const rows = chipJobs(running, 0, NOW);
    expect(rows.map((row) => row.kind)).toEqual(["optimize", "simulate", "warmup"]);
    expect(othersLabel(rows.length - 1)).toBe("+2 more");
  });

  it("says nothing extra when the chip already names everything", () => {
    expect(othersLabel(0)).toBe("");
    expect(chipJobs([job("export", "job-3", "shield.stl")], 0, NOW)).toHaveLength(1);
  });
});

describe("what the × stops", () => {
  it("names the one job it will cancel, never 'this job'", () => {
    // With four things running, an unqualified "cancel" is a question, not a
    // label.
    expect(cancelLabel(job("optimize", "job-9", "shield_mass"))).toBe(
      "Cancel optimize shield_mass",
    );
    expect(cancelLabel(job("compile", "job-2"))).toBe("Cancel compile");
  });

  it("says a compile is still starting rather than offering a dead ×", () => {
    expect(cancelLabel(chipJobs([], NOW, NOW)[0])).toBe("Starting compile…");
  });

  it("says nothing when there is nothing to stop", () => {
    expect(cancelLabel(undefined)).toBe("");
  });
});
