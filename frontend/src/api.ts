/**
 * Typed client for the playground server.
 *
 * The session token is fetched once at startup. A cross-origin page cannot read
 * that response, so requiring the token on writes keeps them same-origin only.
 */

import {
  parseOptimizeStreamLine,
  splitStreamBuffer,
  type OptimizeProgress,
} from "./optimize";
import type {
  CompileResponse,
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
  SimulateStudyRequest,
} from "./types";

let token = "";

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
export async function compile(source: string): Promise<CompileResponse> {
  return post<CompileResponse>("/compile", { source });
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
export async function mesh(source: string): Promise<MeshResponse> {
  return post<MeshResponse>("/api/mesh", { source });
}

/**
 * Mesh the scene into hexahedra and run (or just probe) a FEM simulation.
 *
 * Errors come back in the body — including `error_kind: "fem_unavailable"`
 * when the optional jax-fem extra is missing — so callers can render them.
 */
export async function simulate(body: SimulateRequest): Promise<SimulateResponse> {
  return post<SimulateResponse>("/api/simulate", body);
}

/** Run a study declared in the program; the declaration owns mesh and BCs. */
export async function simulateStudy(
  body: SimulateStudyRequest,
): Promise<SimulateResponse> {
  return post<SimulateResponse>("/api/simulate", body);
}

/** Build a declared SimMesh and return its quality report + surface. */
export async function meshInspect(
  source: string,
  name: string,
): Promise<MeshInspectResponse> {
  return post<MeshInspectResponse>("/api/mesh_inspect", { source, name });
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
): Promise<OptimizeResponse> {
  const response = await fetch("/api/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Cadjoint-Token": token },
    body: JSON.stringify(body),
  });
  if (!response.body) return readJson<OptimizeResponse>(response);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let full = "";
  let done: OptimizeResponse | null = null;
  for (;;) {
    const { value, done: finished } = await reader.read();
    const chunk = decoder.decode(value ?? new Uint8Array(0), { stream: !finished });
    buffer += chunk;
    full += chunk;
    const { lines, rest } = splitStreamBuffer(buffer);
    buffer = rest;
    if (finished && buffer.trim().length > 0) lines.push(buffer.trim());
    for (const line of lines) {
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
    return JSON.parse(full) as OptimizeResponse;
  } catch {
    throw new Error(`Server returned a non-JSON response (${response.status}).`);
  }
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
