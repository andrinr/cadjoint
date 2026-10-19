/**
 * Fetching the derived B-rep for the inspector, lazily.
 *
 * The same shape as the mesh-edge overlay's lazy fetch in `compileCycle.ts`,
 * kept in its own module because it shares nothing with it but the trigger:
 * while `display.showBrep` is on, every newly compiled program is sent to
 * `/api/brep` once, a newer compile cancels an older request, and the answer
 * is dropped if the program it describes is no longer the one on screen.
 *
 * Without the private tier the server says so (`brep.available === false`);
 * that is recorded as `unavailable`, which turns the mode off and greys the
 * switch with the server's sentence — the inspector never errors for a
 * missing install.
 */

import { createEffect, type Accessor } from "solid-js";
import * as api from "../api";
import { cancelClientJob, nextRequestId, watchJobs } from "../jobs";
import {
  brepStatus,
  setBrepHover,
  setBrepMessage,
  setBrepPayload,
  setBrepStatus,
} from "../state";
import type { DisplaySettings } from "../viewer/renderer";

export interface BrepCycleOptions {
  display: Accessor<DisplaySettings>;
  compiledSource: Accessor<string | null>;
  /** Turn the mode off (the private tier is missing). */
  disable: () => void;
}

export function createBrepCycle(options: BrepCycleOptions): void {
  let requestedFor: string | null = null;
  let inFlight: { clientId: string; controller: AbortController } | null = null;

  const cancel = () => {
    if (!inFlight) return;
    inFlight.controller.abort();
    void cancelClientJob(inFlight.clientId);
    inFlight = null;
  };

  createEffect(() => {
    const wanted = options.display().showBrep === true;
    const compiled = options.compiledSource();
    if (!wanted) {
      // Off: keep the last answer — back on over the same program, it is
      // still the right one and costs nothing. Only a request cut short is
      // forgotten, so that turning back on asks again.
      if (inFlight) {
        cancel();
        requestedFor = null;
      }
      setBrepHover(null);
      return;
    }
    if (compiled === null || requestedFor === compiled || brepStatus() === "unavailable") return;
    cancel();
    requestedFor = compiled;
    const clientId = nextRequestId();
    const controller = new AbortController();
    const mine = { clientId, controller };
    inFlight = mine;
    // The old B-rep describes a program that is gone: drawing it while the
    // new one derives would show faces the scene no longer has. The solid
    // stands in until the answer arrives.
    setBrepPayload(null);
    setBrepHover(null);
    setBrepStatus("loading");
    setBrepMessage(null);
    const unwatch = watchJobs();
    void api
      .brep(compiled, { clientId, signal: controller.signal })
      .then((result) => {
        if (options.compiledSource() !== compiled || inFlight !== mine) return;
        const payload = result.brep ?? null;
        if (!result.ok) {
          setBrepStatus("error");
          setBrepMessage(lastLine(result.error) ?? "The B-rep request failed.");
          return;
        }
        if (payload && !payload.available) {
          setBrepStatus("unavailable");
          setBrepMessage(payload.reason ?? "The private tier (diff-brep) is not installed.");
          setBrepPayload(null);
          options.disable();
          return;
        }
        if (payload?.error) {
          setBrepStatus("error");
          setBrepMessage(payload.error);
          setBrepPayload(null);
          return;
        }
        setBrepHover(null);
        setBrepPayload(payload);
        setBrepStatus("ready");
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted || inFlight !== mine) return;
        setBrepStatus("error");
        setBrepMessage(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        unwatch();
        if (inFlight === mine) inFlight = null;
      });
  });
}

/** The last non-empty line of a traceback: the exception, not the frames. */
function lastLine(text: string | undefined): string | null {
  if (!text) return null;
  const lines = text.trim().split("\n").filter((line) => line.trim());
  return lines.length ? lines[lines.length - 1].trim() : null;
}
