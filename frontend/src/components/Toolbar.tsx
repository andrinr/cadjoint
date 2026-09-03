/**
 * Top bar: brand, the mode switcher, the running-work chip, source controls,
 * status — and the render-settings popover anchored to the eye icon.
 * Rendering (presets, shading, quality, annotations) is orthogonal to what
 * you are editing, so the popover opens from any mode rather than being a
 * mode itself.
 */

import { Show, createEffect, createSignal, onCleanup, untrack } from "solid-js";
import { busy, dirty, nodeById, selection, status } from "../state";
import {
  cancelJob,
  jobsSnapshot,
  runningJobs,
  watchJobs,
  type RunningJob,
} from "../jobs";
import { CHIP_KINDS, cancelLabel, chipJobs, othersLabel } from "./jobChip";
import { windowManager } from "../windows/manager";
import { ModeSwitcher } from "./ModeSwitcher";
import { RenderPanel, type RenderPanelProps } from "./RenderPanel";
import { CodeIcon, DisplayIcon, PlayIcon, ResetIcon } from "./icons";

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
   * When the app stopped agreeing with its own source, by the client clock.
   *
   * Zero when it agrees. Kept here rather than derived from the job registry
   * because the registry cannot see the debounce window before the request,
   * and because this is the number the user is counting: seconds since *my
   * edit*, not seconds since the worker got round to it.
   */
  const [compilingSince, setCompilingSince] = createSignal(0);
  createEffect(() => {
    const working = busy();
    setCompilingSince(working ? untrack(compilingSince) || Date.now() : 0);
  });

  /**
   * The work this chip is for, which is not all of it.
   *
   * A lint does not belong here: it fires on a pause in typing, which would
   * make the chip blink several times a minute while you write code. A
   * compile does, and used to be excluded on the grounds that the status line
   * and a disabled Run button already said it — which was true and still left
   * the longest wait in the app as the quietest thing on screen. What is here
   * now is everything the machine is doing that a person is waiting on: the
   * compile first, because it is the one that decides whether the picture is
   * current, then the work you start and go away from — a solve, an
   * optimization, a mesh, an export. The Processes window lists every job;
   * this is the one sentence about the ones worth interrupting for.
   */
  /**
   * The work this chip is for, which is not all of it — see `./jobChip`.
   *
   * `tick()` is read for its reactivity alone: the compile's clock runs off
   * the client's own `Date.now()`, and without the quarter-second ticker in
   * the dependency list the seconds would only move when the registry poll
   * happened to change something.
   */
  const jobs = (): RunningJob[] => {
    tick();
    return chipJobs(runningJobs(), compilingSince(), Date.now());
  };
  /**
   * The one named on the chip when several things are running.
   *
   * `jobs()` puts the compile first and the registry's newest-first order
   * after it, and the lead is simply the head of that. A compile leads
   * because it is the one job that decides whether the picture on screen is
   * the picture of the code; everything else is work you started on purpose
   * and can watch in the monitor.
   */
  const lead = () => jobs()[0];
  /** How many are running but not named here. Never hidden — see `others`. */
  const others = () => Math.max(0, jobs().length - 1);
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
          title={
            others() > 0
              ? `${jobs().length} jobs running — open the process monitor`
              : "Open the process monitor"
          }
          data-testid="job-chip-open"
        >
          <i class="dot" aria-hidden="true" />
          <Show
            when={lead()}
            fallback={<b>failed</b>}
          >
            {(job) => (
              <>
                <b>{job().kind}</b>
                <Show when={job().name}>
                  <span>{job().name}</span>
                </Show>
                <Show when={progressLabel(job())}>
                  <span>{progressLabel(job())}</span>
                </Show>
                <time>{elapsedLabel(job().elapsed_s)}</time>
              </>
            )}
          </Show>
          {/*
            Everything running is accounted for, and the chip is still one
            sentence: it names the job it can stop and then says how many more
            there are, rather than showing one and quietly hiding the rest.
            The count is a link into the Processes window, which is the thing
            that lists them in full — one vocabulary, two depths.
          */}
          <Show when={others() > 0}>
            <span data-testid="job-chip-others">{othersLabel(others())}</span>
          </Show>
        </button>
        <Show when={lead()}>
          {(job) => (
            <button
              type="button"
              class="job-chip-cancel"
              // A compile is on this chip from the moment of the edit, which
              // is up to a poll before the registry can name the job. The
              // button keeps its box and goes dead for that second rather
              // than appearing under the pointer once the id turns up.
              disabled={!job().id}
              onClick={() => void cancelJob(job().id)}
              // Never "cancel this job" when there is more than one: the ×
              // stops the one the chip names, and it says which that is.
              title={cancelLabel(job())}
              aria-label={cancelLabel(job())}
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

      {/*
        The status line is a *result* readout: what the last piece of work
        produced, what the renderer is drawing, what failed. It used to say
        "JAX compiling…" as well, which put the same machine state on the
        same bar twice — the chip to the left already names the compile,
        counts its seconds and can stop it. So while work is in flight this
        yields: the element stays (its slot is not a hole that opens and
        closes) and it says nothing, because it has nothing settled to say.
      */}
      <span class={`status ${status().kind}`} data-testid="status">
        <Show when={status().text}>
          <i class="dot" />
          {status().text}
        </Show>
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
      {/* Run is a verb, not an indicator: it no longer relabels itself
          "Compiling…" — the chip says that, once. It is refused only while
          pressing it would do nothing, which is a compile of text that is
          already compiling; over edited text a Run mid-compile is a
          legitimate thing to ask for and supersedes the one running. */}
      <button
        type="button"
        class="primary"
        onClick={props.onRun}
        disabled={busy() && !dirty()}
        data-testid="run"
      >
        <PlayIcon />
        {dirty() ? "Run •" : "Run"}
      </button>

      {/*
        The seam.

        The chip says what is running and the status says so in words, but
        both are small marks in a bar the eye has learned to skip. This is the
        one indicator that cannot be missed and still costs nothing: the rule
        that separates the chrome from the work below it, doubled to 2px and
        filled with the accent while the picture on screen is not the picture
        of the code. It is the language's own vocabulary — a rule, not a box;
        the accent as a ground, never as a mark; 2px reserved for *active*
        state (§5) — and it reserves no space, because it is drawn on the
        border that is there either way.
      */}
      <Show when={busy()}>
        <div class="toolbar-busy" data-testid="toolbar-busy" aria-hidden="true">
          <i />
        </div>
      </Show>
    </header>
  );
}
