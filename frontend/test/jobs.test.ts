/**
 * The client half of the job registry.
 *
 * These are the rules a panel depends on when it stores four fields instead
 * of a megabyte of solved mesh: that the hash it stores is the same digest
 * the server computed, that a stored record survives every shape of
 * corruption, that "the newest result for this document" means exactly that,
 * and that the readouts the monitor draws are formatted once, here, rather
 * than four times in four panels.
 */

import { describe, expect, it } from "vitest";
import {
  elapsedOf,
  emptyJobRefs,
  findRunningJob,
  forgetJobRef,
  formatBytes,
  formatDuration,
  formatPercent,
  isPending,
  isStale,
  jobLabel,
  jobRefFor,
  newestMatching,
  parseJobRefs,
  rememberJobRef,
  sceneKey,
  setRequestedJob,
  sourceHash,
  takeRequestedJob,
  writeJobRefs,
  readJobRefs,
  type JobRef,
  type JobSummary,
} from "../src/jobs";

const REF: JobRef = {
  job_id: "job-000007",
  source_hash: "a".repeat(64),
  kind: "simulate",
  fields: { kind: "study", name: "bar" },
};

function job(over: Partial<JobSummary> = {}): JobSummary {
  return {
    job_id: "job-000001",
    kind: "simulate",
    status: "done",
    fields: { name: "bar" },
    source_hash: "a".repeat(64),
    source_bytes: 120,
    submitted_at: 0,
    started_at: 0,
    finished_at: 1,
    elapsed_s: 1,
    pid: 4242,
    ok: true,
    error: null,
    progress: null,
    cpu_percent: 0,
    rss_bytes: 0,
    peak_cpu_percent: 0,
    peak_rss_bytes: 0,
    cpu_seconds: 0,
    sampling: "psutil",
    samples: 2,
    result_available: true,
    result_bytes: 10,
    ...over,
  };
}

describe("source hashing", () => {
  it("is the same sha256 the server computes", async () => {
    expect(await sourceHash("abc")).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });

  it("is stable for one text and moves for any edit", async () => {
    const text = "scene = Sphere(0.5)\n";
    expect(await sourceHash(text)).toBe(await sourceHash(text));
    expect(await sourceHash(text)).not.toBe(await sourceHash(`${text} `));
  });

  it("calls a result stale only when both hashes are known and differ", () => {
    expect(isStale({ source_hash: "a" }, "b")).toBe(true);
    expect(isStale({ source_hash: "a" }, "a")).toBe(false);
    // An unknown hash on either side is not evidence of anything.
    expect(isStale({ source_hash: null }, "b")).toBe(false);
    expect(isStale({ source_hash: "a" }, null)).toBe(false);
  });
});

describe("the stored record", () => {
  it("gives an unsaved buffer its own slot", () => {
    expect(sceneKey(null)).toBe("(untitled)");
    expect(sceneKey("   ")).toBe("(untitled)");
    expect(sceneKey("part.py")).toBe("part.py");
  });

  it("round-trips through storage", () => {
    const store: Record<string, string> = {};
    const storage = {
      getItem: (key: string) => store[key] ?? null,
      setItem: (key: string, value: string) => {
        store[key] = value;
      },
    };
    writeJobRefs(storage, rememberJobRef(emptyJobRefs(), "part.py", REF));
    expect(jobRefFor(readJobRefs(storage), "part.py", "simulate")).toEqual(REF);
  });

  it("keeps one reference per kind per scene", () => {
    let refs = rememberJobRef(emptyJobRefs(), "a.py", REF);
    refs = rememberJobRef(refs, "a.py", { ...REF, job_id: "job-000009" });
    refs = rememberJobRef(refs, "b.py", { ...REF, job_id: "job-000011" });
    expect(jobRefFor(refs, "a.py", "simulate")?.job_id).toBe("job-000009");
    expect(jobRefFor(refs, "b.py", "simulate")?.job_id).toBe("job-000011");
    expect(jobRefFor(refs, "a.py", "optimize")).toBeNull();
  });

  it("forgets one kind without touching the others", () => {
    const inspect: JobRef = { ...REF, job_id: "job-000002", kind: "mesh_inspect" };
    let refs = rememberJobRef(rememberJobRef(emptyJobRefs(), "a.py", REF), "a.py", inspect);
    refs = forgetJobRef(refs, "a.py", "simulate");
    expect(jobRefFor(refs, "a.py", "simulate")).toBeNull();
    expect(jobRefFor(refs, "a.py", "mesh_inspect")).toEqual(inspect);
    // Forgetting what is not there returns the same object, so no write.
    expect(forgetJobRef(refs, "a.py", "simulate")).toBe(refs);
  });

  it("never throws on a corrupt record, and keeps nothing it cannot trust", () => {
    expect(parseJobRefs(null)).toEqual(emptyJobRefs());
    expect(parseJobRefs("{ not json")).toEqual(emptyJobRefs());
    expect(parseJobRefs("[]")).toEqual(emptyJobRefs());
    // A record from a future version means nothing to this build.
    expect(parseJobRefs(JSON.stringify({ version: 99, scenes: { a: {} } }))).toEqual(
      emptyJobRefs(),
    );
    const messy = parseJobRefs(
      JSON.stringify({
        version: 1,
        scenes: {
          "a.py": {
            simulate: REF,
            optimize: { job_id: 7 },
            mesh: null,
            lint: { job_id: "job-3", kind: "lint" },
          },
          "b.py": "nonsense",
        },
      }),
    );
    expect(jobRefFor(messy, "a.py", "simulate")).toEqual(REF);
    expect(jobRefFor(messy, "a.py", "optimize")).toBeNull();
    expect(jobRefFor(messy, "b.py", "simulate")).toBeNull();
    // A reference with no hash is still usable: it is simply never stale.
    expect(jobRefFor(messy, "a.py", "lint")).toEqual({
      job_id: "job-3",
      kind: "lint",
      source_hash: null,
      fields: {},
    });
  });

  it("survives storage that refuses to answer", () => {
    const broken = {
      getItem: () => {
        throw new Error("private mode");
      },
      setItem: () => {
        throw new Error("quota");
      },
    };
    expect(readJobRefs(broken)).toEqual(emptyJobRefs());
    expect(() => writeJobRefs(broken, emptyJobRefs())).not.toThrow();
    expect(readJobRefs(undefined)).toEqual(emptyJobRefs());
  });
});

describe("matching a job to the document", () => {
  const hash = "a".repeat(64);

  it("takes the newest finished result for this exact document", () => {
    const jobs = [
      job({ job_id: "job-9", kind: "mesh" }),
      job({ job_id: "job-8", source_hash: "b".repeat(64) }),
      job({ job_id: "job-7", status: "running" }),
      job({ job_id: "job-6", result_available: false }),
      job({ job_id: "job-5" }),
      job({ job_id: "job-4" }),
    ];
    // The listing is newest first, so the first legitimate hit wins.
    expect(newestMatching(jobs, "simulate", hash)?.job_id).toBe("job-5");
    expect(newestMatching(jobs, "optimize", hash)).toBeNull();
    // Without a hash there is no such thing as "for this document".
    expect(newestMatching(jobs, "simulate", null)).toBeNull();
  });

  it("finds the running job a panel can offer to cancel", () => {
    const jobs = [
      job({ job_id: "job-9", status: "running", kind: "mesh" }),
      job({ job_id: "job-8", status: "running", source_hash: "b".repeat(64) }),
      job({ job_id: "job-7", status: "running" }),
    ];
    expect(findRunningJob(jobs, "simulate", hash)?.job_id).toBe("job-7");
    // A run whose source the panel does not know is still that panel's run.
    expect(findRunningJob([job({ status: "running", source_hash: null })], "simulate", hash))
      .not.toBeNull();
    expect(findRunningJob(jobs, "optimize", hash)).toBeNull();
  });

  it("counts queued work as pending", () => {
    expect(isPending(job({ status: "queued" }))).toBe(true);
    expect(isPending(job({ status: "running" }))).toBe(true);
    expect(isPending(job({ status: "done" }))).toBe(false);
    expect(isPending(job({ status: "cancelled" }))).toBe(false);
  });

  it("labels a row by what the request asked for", () => {
    expect(jobLabel(job({ fields: { name: "bar" } }))).toBe("bar");
    expect(jobLabel({ kind: "warmup", fields: { mode: "mesh" }, source_hash: null })).toBe("mesh");
    // Nameless work is named by the document it ran on, not by its own kind
    // repeated beside its kind chip.
    expect(jobLabel({ kind: "compile", fields: {}, source_hash: "a11b805c9".repeat(7) })).toBe(
      "#a11b805c",
    );
    expect(jobLabel({ kind: "compile", fields: {}, source_hash: null })).toBe("compile");
  });
});

describe("readouts", () => {
  it("formats bytes at three significant figures", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(940)).toBe("940 B");
    expect(formatBytes(12_400)).toBe("12.4 kB");
    expect(formatBytes(469_008_384)).toBe("469 MB");
    expect(formatBytes(51_539_607_552)).toBe("51.5 GB");
    expect(formatBytes(null)).toBe("–");
  });

  it("formats durations coarser the longer they ran", () => {
    expect(formatDuration(0.42)).toBe("0.4s");
    expect(formatDuration(9.8)).toBe("9.8s");
    expect(formatDuration(34.2)).toBe("34s");
    expect(formatDuration(134)).toBe("2m 14s");
    expect(formatDuration(undefined)).toBe("–");
  });

  it("formats CPU as a whole percentage that may exceed one core", () => {
    expect(formatPercent(97.4)).toBe("97%");
    expect(formatPercent(312)).toBe("312%");
    expect(formatPercent(NaN)).toBe("–");
  });

  it("extrapolates a running job's elapsed between polls, and freezes it", () => {
    expect(elapsedOf(job({ status: "running", elapsed_s: 4 }), 0.75)).toBeCloseTo(4.75);
    // A finished job's number is final: the clock must not run past it.
    expect(elapsedOf(job({ status: "done", elapsed_s: 4 }), 0.75)).toBe(4);
  });
});

describe("the cross-panel request", () => {
  it("is honoured once, and only by the panel it is addressed to", () => {
    setRequestedJob(REF);
    expect(takeRequestedJob("optimize")).toBeNull();
    expect(takeRequestedJob("simulate")).toEqual(REF);
    expect(takeRequestedJob("simulate")).toBeNull();
  });
});
