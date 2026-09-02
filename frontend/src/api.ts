/**
 * Typed client for the playground server.
 *
 * The session token is fetched once at startup. A cross-origin page cannot read
 * that response, so requiring the token on writes keeps them same-origin only.
 */

import type { JobDetail, JobsSnapshot, JobStamped } from "./jobs";
import {
  parseOptimizeStreamLine,
  splitStreamBuffer,
  type OptimizeProgress,
} from "./optimize";
import type {
  CompileResponse,
  CompleteResponse,
  LintResponse,
  MeshInspectResponse,
  MeshResponse,
  OptimizeRequest,
  OptimizeResponse,
  PatchOperation,
  PatchResponse,
  SceneListResponse,
  SceneLoadResponse,
  SceneSaveResponse,
  SessionResponse,
  SimulateRequest,
  SimulateResponse,
  SignatureResponse,
  SimulateStudyRequest,
} from "./types";

let token = "";

/**
 * Resolves once the session token is in hand.
 *
 * Nothing that needs the token may run before `startSession` has answered,
 * and some of it now does run early: a panel remounting after a reload asks
 * the job registry for the result it was showing, and the process monitor
 * starts polling, both while the app's own startup request is still in
 * flight. Without this gate those requests are refused with a 403 and the
 * restored result silently never arrives.
 */
let markSessionReady: () => void = () => {};
const sessionReady = new Promise<void>((resolve) => {
  markSessionReady = resolve;
});

/** Await the session token; every job endpoint below does. */
export function whenSessionReady(): Promise<void> {
  return sessionReady;
}

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(`Server returned a non-JSON response (${response.status}).`);
  }
}

/** Fetch the session token and starter program. */
export async function startSession(): Promise<SessionResponse> {
  const response = await fetch("/api/session");
  if (!response.ok) throw new Error(`Could not start a session (${response.status}).`);
  const session = await readJson<SessionResponse>(response);
  token = session.token;
  markSessionReady();
  return session;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Cadjoint-Token": token },
    body: JSON.stringify(body),
  });
  return readJson<T>(response);
}

/** Execute the program and compile its scene to WGSL. */
export async function compile(source: string): Promise<CompileResponse & JobStamped> {
  return post<CompileResponse & JobStamped>("/compile", { source });
}

/**
 * Apply one edit to the program text.
 *
 * Operations differ in shape — a vertex edit needs an index, a placement edit
 * needs an argument name — so the body is passed through as given.
 */
export async function patch(
  body: { source: string; op?: PatchOperation | string } & Record<string, unknown>,
): Promise<PatchResponse> {
  return post<PatchResponse>("/patch", body);
}

/**
 * Extract the dual-contour mesh edges for the current program.
 *
 * Split out of `/compile` because it dominates the compile round-trip; the
 * viewer only asks while a mesh overlay is actually displayed.
 */
export async function mesh(source: string): Promise<MeshResponse & JobStamped> {
  return post<MeshResponse & JobStamped>("/api/mesh", { source });
}

/**
 * Mesh the scene into hexahedra and run (or just probe) a FEM simulation.
 *
 * Errors come back in the body — including `error_kind: "fem_unavailable"`
 * when the optional jax-fem extra is missing — so callers can render them.
 */
export async function simulate(
  body: SimulateRequest,
): Promise<SimulateResponse & JobStamped> {
  return post<SimulateResponse & JobStamped>("/api/simulate", body);
}

/** Run a study declared in the program; the declaration owns mesh and BCs. */
export async function simulateStudy(
  body: SimulateStudyRequest,
): Promise<SimulateResponse & JobStamped> {
  return post<SimulateResponse & JobStamped>("/api/simulate", body);
}

/** Build a declared SimMesh and return its quality report + surface. */
export async function meshInspect(
  source: string,
  name: string,
): Promise<MeshInspectResponse & JobStamped> {
  return post<MeshInspectResponse & JobStamped>("/api/mesh_inspect", { source, name });
}

/**
 * Run a declared optimization through the differentiable pipeline.
 *
 * The endpoint may stream chunked NDJSON — per-step `progress` lines, then
 * one `done` line carrying the classic response; `onProgress` fires per
 * step. A non-streaming server (or an error body) ships a single JSON
 * document, which the reader falls back to parsing whole. On success the
 * response's `source` carries the optimized parameter literals written
 * back — the caller adopts it exactly like a patch response.
 */
export async function optimize(
  body: OptimizeRequest,
  onProgress?: (progress: OptimizeProgress) => void,
  onJob?: (jobId: string) => void,
): Promise<OptimizeResponse & JobStamped> {
  const response = await fetch("/api/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Cadjoint-Token": token },
    body: JSON.stringify(body),
  });
  if (!response.body) return readJson<OptimizeResponse & JobStamped>(response);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let full = "";
  let done: (OptimizeResponse & JobStamped) | null = null;
  for (;;) {
    const { value, done: finished } = await reader.read();
    const chunk = decoder.decode(value ?? new Uint8Array(0), { stream: !finished });
    buffer += chunk;
    full += chunk;
    const { lines, rest } = splitStreamBuffer(buffer);
    buffer = rest;
    if (finished && buffer.trim().length > 0) lines.push(buffer.trim());
    for (const line of lines) {
      // Every event of a registered run carries its job id, from the first
      // progress line onward — which is the only way a caller can learn the
      // handle for cancelling a run that has not finished yet.
      if (onJob) {
        try {
          const raw = JSON.parse(line) as { job_id?: unknown };
          if (typeof raw?.job_id === "string") onJob(raw.job_id);
        } catch {
          // Not JSON, or not an object: the parser below reports it.
        }
      }
      const event = parseOptimizeStreamLine(line);
      if (event?.kind === "progress") {
        const { kind: _kind, ...progress } = event;
        onProgress?.(progress);
      } else if (event?.kind === "done") {
        done = event.response;
      }
    }
    if (finished) break;
  }
  if (done) return done;
  try {
    return JSON.parse(full) as OptimizeResponse & JobStamped;
  } catch {
    throw new Error(`Server returned a non-JSON response (${response.status}).`);
  }
}

/**
 * Static-analyse the program: ruff, plus the last compile traceback.
 *
 * The whole document goes over the wire, so callers must debounce — the
 * editor drives this from `linter()`'s idle delay, never from a keystroke.
 * Lines are 1-based and columns 0-based; see `LintDiagnostic`.
 */
export async function lint(source: string): Promise<LintResponse> {
  return post<LintResponse>("/api/lint", { source });
}

/** Completions at a caret (1-based `line`, 0-based `column`). */
export async function complete(
  source: string,
  line: number,
  column: number,
): Promise<CompleteResponse> {
  return post<CompleteResponse>("/api/complete", { source, line, column });
}

/** The signature of the call the caret sits inside, if it sits inside one. */
export async function signature(
  source: string,
  line: number,
  column: number,
): Promise<SignatureResponse> {
  return post<SignatureResponse>("/api/signature", { source, line, column });
}

/** List saved scene files in the server's `scenes` workspace. */
export async function listScenes(): Promise<SceneListResponse> {
  const response = await fetch("/api/scenes");
  return readJson<SceneListResponse>(response);
}

/** Read one saved scene file. */
export async function loadScene(name: string): Promise<SceneLoadResponse> {
  return post<SceneLoadResponse>("/api/scenes/load", { name });
}

/** Write one scene file into the server's `scenes` workspace. */
export async function saveScene(
  name: string,
  source: string,
): Promise<SceneSaveResponse> {
  return post<SceneSaveResponse>("/api/scenes/save", { name, source });
}

// ── the job registry ───────────────────────────────────────────────────────

/**
 * Every registered job, newest first, plus the live load totals.
 *
 * Cheap enough to poll at 1 Hz: the server answers it from memory and the
 * payload carries no results, only summaries. See `jobs.ts` for the poller
 * that calls this and the panels that read it.
 */
export async function listJobs(): Promise<JobsSnapshot> {
  await sessionReady;
  const response = await fetch("/api/jobs", { headers: { "X-Cadjoint-Token": token } });
  if (!response.ok) throw new Error(`Could not list jobs (${response.status}).`);
  return readJson<JobsSnapshot>(response);
}

/** One job with its resource samples and, for an optimize run, its steps. */
export async function jobDetail(jobId: string): Promise<JobDetail | null> {
  await sessionReady;
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
    headers: { "X-Cadjoint-Token": token },
  });
  if (!response.ok) return null;
  const body = await readJson<{ ok: boolean; job?: JobDetail }>(response);
  return body.job ?? null;
}

/** What `fetchJobResult` found: the payload, or why there is none. */
export type JobResultOutcome<T> =
  | { state: "ok"; payload: T }
  /** Still queued or running — poll the job and ask again when it finishes. */
  | { state: "pending" }
  /** The job is gone, or its payload was evicted: the work must be redone. */
  | { state: "gone" }
  | { state: "error"; message: string };

/**
 * The payload a job produced, byte-identical to the response its request got.
 *
 * The three refusals are all ordinary states of a stored result rather than
 * failures, so they are returned rather than thrown: 409 while the job is
 * still running, 404 for an id this server has never had (a reload against a
 * restarted playground), 410 for a job whose payload has been evicted.
 */
export async function fetchJobResult<T>(jobId: string): Promise<JobResultOutcome<T>> {
  await sessionReady;
  let response: Response;
  try {
    response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/result`, {
      headers: { "X-Cadjoint-Token": token },
    });
  } catch (error) {
    return { state: "error", message: error instanceof Error ? error.message : String(error) };
  }
  if (response.ok) return { state: "ok", payload: await readJson<T>(response) };
  if (response.status === 409) return { state: "pending" };
  if (response.status === 404 || response.status === 410) return { state: "gone" };
  return { state: "error", message: `Could not read that result (${response.status}).` };
}

/**
 * Kill a running job's worker process.
 *
 * The request that started the work ends on its own with
 * `error: "cancelled"`; this only asks for the kill. A 409 means the job had
 * already finished, which is not worth reporting to the user.
 */
export async function cancelJob(jobId: string): Promise<boolean> {
  await sessionReady;
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Cadjoint-Token": token },
    body: "{}",
  });
  return response.ok;
}

/** Drop every finished job from the registry; running work is kept. */
export async function clearJobs(): Promise<void> {
  await sessionReady;
  await fetch("/api/jobs/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Cadjoint-Token": token },
    body: "{}",
  });
}
