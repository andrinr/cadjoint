/**
 * Typed client for the playground server.
 *
 * The session token is fetched once at startup. A cross-origin page cannot read
 * that response, so requiring the token on writes keeps them same-origin only.
 */

import type {
  CompileResponse,
  PatchOperation,
  PatchResponse,
  SessionResponse,
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
    headers: { "Content-Type": "application/json", "X-Jaxcad-Token": token },
    body: JSON.stringify(body),
  });
  return readJson<T>(response);
}

/** Execute the program and compile its scene to WGSL. */
export async function compile(source: string): Promise<CompileResponse> {
  return post<CompileResponse>("/compile", { source });
}

/** Rewrite a sketch vertex literal in the program text. */
export async function patch(
  source: string,
  op: PatchOperation,
  line: number,
  index: number,
  xy?: [number, number],
): Promise<PatchResponse> {
  return post<PatchResponse>("/patch", { source, op, line, index, xy });
}
