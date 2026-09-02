/**
 * The client's half of the server's job registry.
 *
 * Every request that costs real time — compile, mesh, mesh inspection, solve,
 * optimization run, export, lint, the startup warm-up — is registered on the server as
 * a *job* and keeps its result after the response is read. That turns three
 * things that used to be impossible into ordinary bookkeeping:
 *
 *   - **A result outlives the panel that asked for it.** A panel stores four
 *     small fields ({@link JobRef}) instead of a megabyte of mesh, and asks
 *     for the payload again by id when it comes back. Switching modes tears
 *     the Simulate window's Solid root down; the solve it was showing is not
 *     lost, it is one fetch away.
 *   - **The machine is visible.** One 1 Hz poll of `/api/jobs` feeds the
 *     Processes window: what is running, what it costs, what it cost.
 *   - **A run can be stopped**, because the job id is the handle the cancel
 *     endpoint takes.
 *
 * Everything here is either pure (hashing, matching, formatting, the stored
 * record) or the one shared poller. The poller is reference-counted on
 * purpose: it runs while somebody is looking (the Processes window) or while
 * somebody is waiting (a panel that needs the id of its in-flight solve so it
 * can offer Cancel), and stops the moment neither is true.
 */

import { createSignal } from "solid-js";
import * as api from "./api";

/** The kinds of work the server registers. */
export type JobKind =
  | "compile"
  | "mesh"
  | "mesh_inspect"
  | "simulate"
  | "optimize"
  | "export"
  | "lint"
  | "warmup";

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

/** The per-step figures an optimize job mirrors out of its stream. */
export interface JobProgress {
  step?: number;
  steps?: number;
  objective?: number;
  grad_norm?: number;
  elapsed?: number;
}

/** One resource sample of a running worker, relative to the job's start. */
export interface JobSample {
  t: number;
  cpu_percent: number;
  rss_bytes: number;
}

/** A job as `GET /api/jobs` lists it: everything but the result and samples. */
export interface JobSummary {
  job_id: string;
  kind: JobKind;
  status: JobStatus;
  fields: Record<string, string | number | boolean>;
  source_hash: string | null;
  source_bytes: number;
  submitted_at: number;
  started_at: number | null;
  finished_at: number | null;
  elapsed_s: number;
  pid: number | null;
  ok: boolean | null;
  error: string | null;
  progress: JobProgress | null;
  cpu_percent: number;
  rss_bytes: number;
  peak_cpu_percent: number;
  peak_rss_bytes: number;
  cpu_seconds: number;
  sampling: "psutil" | "rusage" | "none";
  samples: number;
  result_available: boolean;
  result_bytes: number;
}

export interface JobDetail extends JobSummary {
  sample_series: JobSample[];
  progress_events: JobProgress[];
}

export interface JobTotals {
  running: number;
  cpu_percent: number;
  rss_bytes: number;
  uptime_s: number;
  server: { cpu_percent: number; rss_bytes: number; pid: number };
  host: { cpu_count: number | null; mem_total: number | null; mem_available: number | null };
  sampling: "psutil" | "rusage";
}

export interface JobStore {
  jobs: number;
  max_jobs: number;
  max_lint_jobs: number;
  result_bytes: number;
  max_result_bytes: number;
  evicted_jobs: number;
  evicted_results: number;
}

export interface JobsSnapshot {
  ok: boolean;
  jobs: JobSummary[];
  totals: JobTotals;
  store: JobStore;
}

/**
 * What a panel stores instead of a result.
 *
 * Four fields, all short: the id to fetch by, the hash of the program the
 * result describes (so a panel can tell a current result from one the last
 * edit invalidated), and enough of the request to label the row before the
 * payload has arrived.
 */
export interface JobRef {
  job_id: string;
  source_hash: string | null;
  kind: JobKind;
  fields: Record<string, string | number | boolean>;
}

/**
 * What registration adds to an endpoint's response.
 *
 * The server's contract did not change when jobs arrived: every endpoint
 * answers exactly as before, with the id of the job that ran it added, and
 * `error_kind: "cancelled"` when somebody stopped that job (HTTP 409). Both
 * are optional so a response from an older server still types.
 */
export interface JobStamped {
  job_id?: string;
  error_kind?: string;
}

// ── source hashing ─────────────────────────────────────────────────────────

/**
 * The sha256 of a program's text, hex, lowercase.
 *
 * The same digest the server computes for `source_hash`, so a stored result
 * can be matched against the document currently in the editor without asking
 * the server what it thinks the document is.
 */
export async function sourceHash(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

/** Whether a result describes a program that is no longer in the editor. */
export function isStale(ref: Pick<JobRef, "source_hash">, currentHash: string | null): boolean {
  if (!ref.source_hash || !currentHash) return false;
  return ref.source_hash !== currentHash;
}

// ── the stored record ──────────────────────────────────────────────────────

export const JOB_REFS_STORAGE_KEY = "cadjoint.jobs.v1";
export const JOB_REFS_VERSION = 1;

/** Job refs per scene, per kind: `{ [scene]: { simulate: ref, … } }`. */
export type JobRefsByScene = Record<string, Partial<Record<JobKind, JobRef>>>;

export interface JobRefs {
  version: number;
  scenes: JobRefsByScene;
}

export function emptyJobRefs(): JobRefs {
  return { version: JOB_REFS_VERSION, scenes: {} };
}

/** The storage key for a scene; an unsaved buffer is its own slot. */
export function sceneKey(name: string | null): string {
  return name && name.trim() ? name : "(untitled)";
}

function parseRef(value: unknown): JobRef | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (typeof record.job_id !== "string" || !record.job_id) return null;
  if (typeof record.kind !== "string") return null;
  const hash = record.source_hash;
  return {
    job_id: record.job_id,
    kind: record.kind as JobKind,
    source_hash: typeof hash === "string" ? hash : null,
    fields:
      record.fields && typeof record.fields === "object"
        ? (record.fields as JobRef["fields"])
        : {},
  };
}

/** Read a stored record, tolerating every shape of corruption. */
export function parseJobRefs(raw: string | null | undefined): JobRefs {
  if (!raw) return emptyJobRefs();
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return emptyJobRefs();
  }
  if (!parsed || typeof parsed !== "object") return emptyJobRefs();
  const record = parsed as Record<string, unknown>;
  if (record.version !== JOB_REFS_VERSION) return emptyJobRefs();
  const scenes = record.scenes;
  if (!scenes || typeof scenes !== "object") return emptyJobRefs();
  const next = emptyJobRefs();
  for (const [scene, kinds] of Object.entries(scenes as Record<string, unknown>)) {
    if (!kinds || typeof kinds !== "object") continue;
    const bucket: Partial<Record<JobKind, JobRef>> = {};
    for (const [kind, value] of Object.entries(kinds as Record<string, unknown>)) {
      const ref = parseRef(value);
      if (ref) bucket[kind as JobKind] = ref;
    }
    if (Object.keys(bucket).length > 0) next.scenes[scene] = bucket;
  }
  return next;
}

export function rememberJobRef(refs: JobRefs, scene: string, ref: JobRef): JobRefs {
  return {
    ...refs,
    scenes: { ...refs.scenes, [scene]: { ...refs.scenes[scene], [ref.kind]: ref } },
  };
}

export function forgetJobRef(refs: JobRefs, scene: string, kind: JobKind): JobRefs {
  const bucket = refs.scenes[scene];
  if (!bucket || !bucket[kind]) return refs;
  const next = { ...bucket };
  delete next[kind];
  return { ...refs, scenes: { ...refs.scenes, [scene]: next } };
}

export function jobRefFor(refs: JobRefs, scene: string, kind: JobKind): JobRef | null {
  return refs.scenes[scene]?.[kind] ?? null;
}

/** The minimal storage surface, so tests can pass a plain object. */
export interface RefStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function readJobRefs(storage: RefStorage | undefined): JobRefs {
  if (!storage) return emptyJobRefs();
  try {
    return parseJobRefs(storage.getItem(JOB_REFS_STORAGE_KEY));
  } catch {
    return emptyJobRefs();
  }
}

export function writeJobRefs(storage: RefStorage | undefined, refs: JobRefs): void {
  if (!storage) return;
  try {
    storage.setItem(JOB_REFS_STORAGE_KEY, JSON.stringify(refs));
  } catch {
    // Result persistence is a convenience; a full quota must not break a solve.
  }
}

/** The browser's storage, or nothing when it is unavailable (private mode). */
export function refStorage(): RefStorage | undefined {
  return typeof localStorage === "undefined" ? undefined : localStorage;
}

/** Store one job reference for a scene, in one call. */
export function saveJobRef(scene: string, ref: JobRef): void {
  const storage = refStorage();
  writeJobRefs(storage, rememberJobRef(readJobRefs(storage), scene, ref));
}

/** Drop one job reference for a scene, in one call. */
export function dropJobRef(scene: string, kind: JobKind): void {
  const storage = refStorage();
  writeJobRefs(storage, forgetJobRef(readJobRefs(storage), scene, kind));
}

/** Read one job reference for a scene, in one call. */
export function loadJobRef(scene: string, kind: JobKind): JobRef | null {
  return jobRefFor(readJobRefs(refStorage()), scene, kind);
}

// ── matching a job to the document ─────────────────────────────────────────

/** The newest finished job of *kind* whose result still describes *hash*. */
export function newestMatching(
  jobs: readonly JobSummary[],
  kind: JobKind,
  hash: string | null,
): JobSummary | null {
  if (!hash) return null;
  for (const job of jobs) {
    // The listing is newest first, so the first hit is the one to take.
    if (job.kind !== kind) continue;
    if (job.status !== "done" || !job.result_available) continue;
    if (job.source_hash !== hash) continue;
    return job;
  }
  return null;
}

/** The job currently running for *kind*, so a panel can offer Cancel. */
export function findRunningJob(
  jobs: readonly JobSummary[],
  kind: JobKind,
  hash: string | null,
): JobSummary | null {
  for (const job of jobs) {
    if (job.kind !== kind || job.status !== "running") continue;
    if (hash && job.source_hash && job.source_hash !== hash) continue;
    return job;
  }
  return null;
}

/** Whether a job is still going, in the two states that mean "not yet". */
export function isPending(job: Pick<JobSummary, "status">): boolean {
  return job.status === "queued" || job.status === "running";
}

/**
 * What names a job in a list.
 *
 * A solve, an inspection and an optimization are asked for by name, and that
 * name is what the user is waiting on. A compile, a lint or a warm-up has no
 * name — repeating its kind beside its own chip would say nothing — so it is
 * identified by the document it ran on, the first eight hex of the same
 * source hash the staleness check uses. Two compiles of the same text are
 * then visibly the same work, and one after an edit is visibly not.
 */
export function jobLabel(job: Pick<JobSummary, "kind" | "fields" | "source_hash">): string {
  const name = job.fields?.name ?? job.fields?.mode;
  if (typeof name === "string" && name) return name;
  if (job.source_hash) return `#${job.source_hash.slice(0, 8)}`;
  return job.kind;
}

/**
 * Seconds a job has been running, counted against the client's clock.
 *
 * A running job's `elapsed_s` is a second old by the time it is drawn, so the
 * row would tick in visible jumps. Extrapolating from `started_at` — a server
 * wall-clock stamp — against the poll it arrived with keeps the readout
 * smooth without pretending the two clocks agree.
 */
export function elapsedOf(job: JobSummary, sinceSample: number): number {
  if (!isPending(job)) return job.elapsed_s;
  return job.elapsed_s + Math.max(0, sinceSample);
}

// ── formatting ─────────────────────────────────────────────────────────────

/** "1.2 GB", "480 MB", "12 kB" — three significant figures, never more. */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return "–";
  if (bytes < 1_000) return `${Math.round(bytes)} B`;
  const units = ["kB", "MB", "GB", "TB"];
  let value = bytes / 1_000;
  let unit = 0;
  while (value >= 1_000 && unit < units.length - 1) {
    value /= 1_000;
    unit += 1;
  }
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

/** "0.4s", "9.8s", "34s", "2m 14s" — coarser the longer it ran. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "–";
  if (seconds < 10) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds - minutes * 60)}s`;
}

/** "97%" — CPU is reported per core, so it can exceed 100. */
export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "–";
  return `${Math.round(value)}%`;
}

// ── the shared 1 Hz poller ─────────────────────────────────────────────────

/** How many total-CPU samples the load sparkline keeps. */
export const CPU_HISTORY = 60;
export const POLL_INTERVAL_MS = 1_000;

const [snapshot, setSnapshot] = createSignal<JobsSnapshot | null>(null);
const [snapshotAt, setSnapshotAt] = createSignal(0);
const [cpuHistory, setCpuHistory] = createSignal<number[]>([]);
const [pollError, setPollError] = createSignal("");

/** The last `GET /api/jobs` payload, or null before the first poll. */
export { snapshot as jobsSnapshot, cpuHistory, pollError };

/** `performance.now()` of the last poll, for smooth elapsed readouts. */
export { snapshotAt as jobsSnapshotAt };

let watchers = 0;
let timer: ReturnType<typeof setTimeout> | undefined;
let inFlight = false;

/**
 * Whether the next tick is worth making.
 *
 * Two reasons, and the second is the one that matters for the viewport's
 * running-job chip: somebody is *looking* (the Processes window is open, or a
 * panel is waiting for the id of its in-flight request), or something is
 * *running*. A job outlives the window that started it — a solve keeps going
 * while you edit in Model mode — so the poll has to outlive it too, and a
 * poll that only ran while a monitor was open would leave the chip frozen on
 * the last thing it happened to see.
 *
 * Both conditions false means no requests at all: an idle playground is
 * silent, which is the property this predicate exists to preserve.
 */
function pollingWanted(): boolean {
  if (watchers > 0) return true;
  return (snapshot()?.jobs ?? []).some(isPending);
}

/** Queue the next tick, unless one is already queued. */
function schedule(): void {
  if (timer !== undefined) return;
  timer = setTimeout(() => {
    timer = undefined;
    void refreshJobs();
  }, POLL_INTERVAL_MS);
}

/**
 * Fetch one snapshot now, and keep going while there is a reason to.
 *
 * This is the whole loop: every tick is one of these, and each one decides
 * whether there will be another. Calling it from outside (after a cancel, or
 * to start watching a job that has just been submitted) therefore does not
 * only refresh — it starts a poll that runs itself until the work is done.
 */
export async function refreshJobs(): Promise<void> {
  if (inFlight) return;
  inFlight = true;
  try {
    const next = await api.listJobs();
    setSnapshot(next);
    setSnapshotAt(now());
    setPollError("");
    setCpuHistory((history) =>
      [...history, next.totals?.cpu_percent ?? 0].slice(-CPU_HISTORY),
    );
  } catch (error) {
    setPollError(error instanceof Error ? error.message : String(error));
  } finally {
    inFlight = false;
    if (pollingWanted()) schedule();
  }
}

function now(): number {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}

/**
 * Nudge the poller: refresh now, and keep polling while work is running.
 *
 * For a caller that has just started something the registry will know about
 * but does not itself want to watch — a compile, say. Cheap by design: a
 * poll already running is not doubled, and one that finds nothing pending
 * stops after a single request.
 */
export function pokeJobs(): void {
  void refreshJobs();
}

/**
 * Start polling, and stop when the returned function is called.
 *
 * Reference-counted: the Processes window subscribes while it is open, and a
 * panel waiting on an in-flight request subscribes until it has the job id it
 * needs. Releasing the last watcher does not necessarily stop the poll —
 * `pollingWanted` keeps it alive while a job is still running, so a chip in
 * the viewport stays live after the window that started the work is gone.
 */
export function watchJobs(): () => void {
  watchers += 1;
  if (watchers === 1) void refreshJobs();
  else schedule();
  let released = false;
  return () => {
    if (released) return;
    released = true;
    watchers -= 1;
    if (!pollingWanted() && timer !== undefined) {
      clearTimeout(timer);
      timer = undefined;
    }
  };
}

/** Whether anything is polling right now (for tests and diagnostics). */
export function jobsWatchers(): number {
  return watchers;
}

/** One line of "something is running", for a chip that is not a window. */
export interface RunningJob {
  id: string;
  kind: JobKind;
  /** What the user asked for, or the document hash for unnamed work. */
  name: string;
  elapsed_s: number;
  /** Optimization runs only: step, total, objective so far. */
  progress?: JobProgress;
}

/**
 * What is running right now, for anyone who wants to say so.
 *
 * The Processes window is the full instrument; this is the one sentence
 * version of it, so a viewport chip (or a status bar, or a tool rail badge)
 * can show that the machine is busy without knowing anything about the job
 * registry. It reads the same 1 Hz snapshot the window does — there is one
 * poll in this app, not one per reader — and `elapsed_s` is extrapolated
 * against the client clock so a readout built on it counts up smoothly
 * instead of stepping once a second.
 *
 * It is an accessor rather than a signal because the derivation is cheap and
 * the source of truth is the snapshot: subscribe to this and you subscribe
 * to the poll.
 */
export function runningJobs(): RunningJob[] {
  const snap = snapshot();
  if (!snap) return [];
  const since = Math.max(0, (now() - snapshotAt()) / 1_000);
  return snap.jobs.filter(isPending).map((job) => ({
    id: job.job_id,
    kind: job.kind,
    name: jobLabel(job),
    elapsed_s: elapsedOf(job, since),
    ...(job.progress ? { progress: job.progress } : {}),
  }));
}

/**
 * Kill a running job by id.
 *
 * Re-exported here so a caller that reads {@link runningJobs} has both halves
 * of the contract in one import: what is running, and how to stop it. The
 * request that started the work ends on its own; this only asks for the kill,
 * and a job that had already finished answers 409, which is not worth
 * reporting to anyone.
 */
export async function cancelJob(jobId: string): Promise<boolean> {
  const stopped = await api.cancelJob(jobId);
  await refreshJobs();
  return stopped;
}

// ── the cross-panel request ────────────────────────────────────────────────

/**
 * A result the user asked to re-open from the Processes window.
 *
 * The Processes window does not know how to render a solved field or a
 * trajectory; the panels that do are mounted somewhere else, and possibly not
 * mounted at all yet. So the click publishes a request here, switches to the
 * mode whose desk holds that panel, and the panel consumes it when it can —
 * setting this back to null so it is honoured exactly once.
 */
export const [requestedJob, setRequestedJob] = createSignal<JobRef | null>(null);

/**
 * A job row somebody asked the monitor to open, by id.
 *
 * The other half of "show this failure in Processes": the window that made
 * the request knows the id, the monitor knows how to draw the row, and
 * neither has to know about the other. The monitor expands that row and
 * clears this, so the request is honoured exactly once.
 */
export const [focusedJob, setFocusedJob] = createSignal<string | null>(null);

/** Take the pending request when it is for *kind*, else leave it alone. */
export function takeRequestedJob(kind: JobKind): JobRef | null {
  const pending = requestedJob();
  if (!pending || pending.kind !== kind) return null;
  setRequestedJob(null);
  return pending;
}

// ── waiting on a job that is still running ─────────────────────────────────

/**
 * Fetch a job's result, waiting out a job that has not finished yet.
 *
 * A stored reference can point at work that is still in flight — the panel
 * was closed mid-solve, or the page was reloaded while the worker ran. The
 * server answers 409 for that, which is not an error but a "not yet", so
 * this polls until the job settles and then reads the payload once.
 *
 * `stopped` is checked between attempts so an unmounting panel stops
 * waiting; a caller that never stops is capped by `attempts`.
 */
export async function awaitJobResult<T>(
  jobId: string,
  options: { stopped?: () => boolean; attempts?: number; intervalMs?: number } = {},
): Promise<api.JobResultOutcome<T>> {
  const attempts = options.attempts ?? 600;
  const interval = options.intervalMs ?? POLL_INTERVAL_MS;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (options.stopped?.()) return { state: "error", message: "stopped" };
    const outcome = await api.fetchJobResult<T>(jobId);
    if (outcome.state !== "pending") return outcome;
    await new Promise((resolve) => setTimeout(resolve, interval));
  }
  return { state: "error", message: "That job is still running." };
}

/** The newest finished job of *kind* for this document, fetched fresh. */
export async function findMatchingJob(
  kind: JobKind,
  hash: string | null,
): Promise<JobSummary | null> {
  if (!hash) return null;
  try {
    const snap = await api.listJobs();
    return newestMatching(snap.jobs ?? [], kind, hash);
  } catch {
    return null;
  }
}
