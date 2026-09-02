/**
 * Top bar: brand, the mode switcher, the running-work chip, source controls,
 * status — and the render-settings popover anchored to the eye icon.
 * Rendering (presets, shading, quality, annotations) is orthogonal to what
 * you are editing, so the popover opens from any mode rather than being a
 * mode itself.
 */

import { Show, createEffect, createSignal, onCleanup } from "solid-js";
import { busy, dirty, nodeById, selection, status } from "../state";
import {
  cancelJob,
  jobsSnapshot,
  runningJobs,
  watchJobs,
  type JobKind,
  type RunningJob,
} from "../jobs";
import { windowManager } from "../windows/manager";
import { ModeSwitcher } from "./ModeSwitcher";
import { RenderPanel, type RenderPanelProps } from "./RenderPanel";
import { CodeIcon, DisplayIcon, PlayIcon, ResetIcon } from "./icons";

/** The kinds worth a chip: work you start and then look away from. */
const CHIP_KINDS = new Set<JobKind>(["simulate", "optimize", "mesh", "mesh_inspect", "export"]);

/** How long a failure keeps the dot after the job has gone. */
const FAILURE_LINGER_MS = 6_000;

/** `0:42`, `12:07`, `1:03:20` — the shortest form that is not ambiguous. */
function elapsedLabel(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const s = String(whole % 60).padStart(2, "0");
  const minutes = Math.floor(whole / 60);
  if (minutes < 60) return `${minutes}:${s}`;
  const m = String(minutes % 60).padStart(2, "0");
  return `${Math.floor(minutes / 60)}:${m}:${s}`;
}

/** `step 3/12` for an optimization; nothing for work with no steps to count. */
function progressLabel(job: RunningJob): string | null {
  const { step, steps } = job.progress ?? {};
  if (typeof step !== "number") return null;
  return typeof steps === "number" ? `step ${step}/${steps}` : `step ${step}`;
}

/**
 * "Something is running", in the one place the eye is already resting.
 *
 * The request was for an indicator in the viewport and then, on seeing it, for
 * it to be up beside the mode switcher instead — which is right: the modes are
 * where you say what you are doing, and this says what the machine is doing
 * about it. It is a *chip*, not a panel: kind, name, and a clock counting up,
 * enough to know a solve is alive without opening the monitor. Pressing it
 * opens the monitor; the × kills the job.
 *
 * It occupies no space at all when nothing is running — no reserved slot, no
 * empty box — because a permanent widget that is blank nine tenths of the time
 * teaches the eye to stop looking at that spot, which is the opposite of what
 * an indicator is for.
 *
 * The only colour in it is the dot: `--ok` while work is alive, `--danger` for
 * a few seconds after something failed, and nothing else here is anything but
 * ink on paper. A failure is stamped against the *client* clock when a job is
 * first seen in that state, rather than read off `finished_at`, so the linger
 * is six seconds of the reader's time and not six seconds of a server clock
 * that may not agree with it.
 */
function JobChip() {
  // The chip is a reader of the job poll in its own right: it is on screen in
  // every mode, including the ones where the Processes window is not mounted,
  // so it keeps the poll alive rather than borrowing the window's.
  onCleanup(watchJobs());

  const [tick, setTick] = createSignal(0);
  const [failedAt, setFailedAt] = createSignal(0);
  const seen = new Set<string>();

  // A quarter-second ticker, so `elapsed_s` — extrapolated against the client
  // clock — reads as a clock counting up rather than a number stepping once a
  // second. It runs only while there is something to count.
  const ticker = setInterval(() => setTick((value) => value + 1), 250);
  onCleanup(() => clearInterval(ticker));

  createEffect(() => {
    for (const job of jobsSnapshot()?.jobs ?? []) {
      if (!CHIP_KINDS.has(job.kind) || seen.has(job.job_id)) continue;
      if (job.status !== "failed") continue;
      seen.add(job.job_id);
      setFailedAt(Date.now());
    }
  });

  /**
   * The work this chip is for, which is not all of it.
   *
   * A compile and a lint are jobs in the registry, and neither belongs here:
   * the status line two controls to the right already says "JAX compiling…"
   * and the Run button is already disabled, so a third mark saying the same
   * thing is noise — and a lint fires on a pause in typing, which would make
   * the chip blink several times a minute while you write code. What is left
   * is the work you start and then go and do something else during: a solve,
   * an optimization, a mesh, an export. The Processes window lists every job; this is the
   * one sentence, and it is about the ones worth interrupting for.
   */
  const jobs = () => {
    tick();
    return runningJobs().filter((job) => CHIP_KINDS.has(job.kind));
  };
  const lead = () => jobs()[0];
  const recentlyFailed = () => tick() >= 0 && Date.now() - failedAt() < FAILURE_LINGER_MS;

  return (
    <Show when={jobs().length > 0 || recentlyFailed()}>
      <div
        class="job-chip"
        classList={{ failed: jobs().length === 0 || recentlyFailed() }}
        data-testid="job-chip"
      >
        <button
          type="button"
          class="job-chip-open"
          onClick={() => windowManager()?.open("processes")}
          title="Open the process monitor"
          data-testid="job-chip-open"
        >
          <i class="dot" aria-hidden="true" />
          <Show when={jobs().length > 1}>
            <b>{jobs().length} running</b>
          </Show>
          <Show
            when={lead()}
            fallback={<b>failed</b>}
          >
            {(job) => (
              <>
                <b>{job().kind}</b>
                <span>{job().name}</span>
                <Show when={progressLabel(job())}>
                  <span>{progressLabel(job())}</span>
                </Show>
                <time>{elapsedLabel(job().elapsed_s)}</time>
              </>
            )}
          </Show>
        </button>
        <Show when={lead()}>
          {(job) => (
            <button
              type="button"
              class="job-chip-cancel"
              onClick={() => void cancelJob(job().id)}
              title={`Cancel ${job().kind} ${job().name}`}
              aria-label="Cancel this job"
              data-testid="job-chip-cancel"
            >
              ×
            </button>
          )}
        </Show>
      </div>
    </Show>
  );
}

export interface ToolbarProps {
  onRun: () => void;
  onReset: () => void;
  onShowWgsl: () => void;
  wgslReady: boolean;
  /** Everything the render-settings popover forwards to RenderPanel. */
  render: RenderPanelProps;
}

export function Toolbar(props: ToolbarProps) {
  const [renderOpen, setRenderOpen] = createSignal(false);
  let anchor: HTMLDivElement | undefined;

  // Escape closes the popover in the capture phase, so the viewer's own
  // Escape handling (clear selection, return to model) never sees the key.
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape" && renderOpen()) {
      event.preventDefault();
      event.stopPropagation();
      setRenderOpen(false);
    }
  };
  // Clicking anywhere outside the anchor dismisses like any popover.
  const onPointerDown = (event: PointerEvent) => {
    if (renderOpen() && anchor && !anchor.contains(event.target as Node)) {
      setRenderOpen(false);
    }
  };
  window.addEventListener("keydown", onKeyDown, true);
  window.addEventListener("pointerdown", onPointerDown, true);
  onCleanup(() => {
    window.removeEventListener("keydown", onKeyDown, true);
    window.removeEventListener("pointerdown", onPointerDown, true);
  });

  return (
    <header class="toolbar">
      <div class="brand">
        <span class="mark">cj</span>
        <span>CADJOINT</span>
      </div>

      <ModeSwitcher />
      <JobChip />

      <div class="spacer" />

      <Show when={selection()}>
        <span class="selection-chip" data-testid="selection-chip">
          {selection()!.vertexIndex === null
            ? (nodeById(selection()!.nodeId)?.name ?? "solid")
            : `vertex ${selection()!.vertexIndex}`}
        </span>
      </Show>

      <span class={`status ${status().kind}`} data-testid="status">
        <i class="dot" />
        {status().text}
      </span>

      <div class="render-popover-anchor" ref={anchor}>
        <button
          type="button"
          class={`icon ${renderOpen() ? "active" : ""}`}
          onClick={() => setRenderOpen(!renderOpen())}
          title="Render settings — presets, shading, and quality"
          aria-label="Render settings"
          aria-expanded={renderOpen()}
          data-testid="display-options"
        >
          <DisplayIcon />
        </button>
        <Show when={renderOpen()}>
          <div class="render-popover" data-testid="render-popover">
            <RenderPanel {...props.render} />
          </div>
        </Show>
      </div>
      <button
        type="button"
        class="icon"
        onClick={props.onShowWgsl}
        disabled={!props.wgslReady}
        title="Show the generated WGSL"
        aria-label="Generated WGSL"
      >
        <CodeIcon />
      </button>
      <button
        type="button"
        class="icon"
        onClick={props.onReset}
        title="Reset to the starter program"
        aria-label="Reset"
      >
        <ResetIcon />
      </button>
      <button
        type="button"
        class="primary"
        onClick={props.onRun}
        disabled={busy()}
        data-testid="run"
      >
        <PlayIcon />
        {busy() ? "Compiling…" : dirty() ? "Run •" : "Run"}
      </button>
    </header>
  );
}
