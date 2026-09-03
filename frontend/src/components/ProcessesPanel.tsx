/**
 * The process monitor: what this playground is running, what it ran, and what
 * the machine is paying for it.
 *
 * A CAD session here is mostly waiting — a mesh is five seconds, a solve is
 * nine, an optimization is minutes — and until now none of that waiting was
 * visible. The work happened inside a subprocess, in a panel you may have
 * navigated away from, with no way to tell a slow solve from a hung one and
 * no way to stop either. This window is the instrument that closes that gap,
 * and it is deliberately shaped like the rest of the sheet: three zones, mono
 * readouts, no colour that is not already in the system.
 *
 *   RUNNING  what is executing right now, with a cancel per row
 *   HISTORY  what finished, newest first; a click re-opens its result
 *   LOAD     the totals, a minute of worker CPU, and the store's budget
 *
 * It owns no scene state. Everything it draws comes from one 1 Hz poll of
 * `/api/jobs` (shared through `jobs.ts`, and only while this window is open
 * or a panel is waiting on a job), and the per-job sample series is fetched
 * only for a row somebody expanded.
 */

import { For, Show, createEffect, createSignal, onCleanup } from "solid-js";
import * as api from "../api";
import {
  cancelJob,
  cpuHistory,
  elapsedOf,
  focusedJob,
  formatBytes,
  formatDuration,
  formatPercent,
  isPending,
  jobLabel,
  jobsSnapshot,
  jobsSnapshotAt,
  pollError,
  refreshJobs,
  setFocusedJob,
  setRequestedJob,
  watchJobs,
  type JobDetail,
  type JobSummary,
} from "../jobs";
import { editingMode, setEditingMode } from "../state";
import { windowManager } from "../windows/manager";
import type { WindowId } from "../windows/panels";
import { Section, Sparkline, Stat, StatRow } from "./ui";
import "./processes.css";

/** Kinds whose stored result a panel elsewhere knows how to re-open. */
const REPLAYABLE = new Set(["simulate", "mesh_inspect", "optimize"]);

/** How often the elapsed readouts re-render between polls. */
const TICK_MS = 250;

/**
 * The chip each kind wears, borrowed from the study and mesh chips.
 *
 * Only the two kinds a user *asked* for are filled — a solve and an
 * optimization are the work; a compile, a lint and the startup warm-up are
 * the machine keeping up with the editor, and a monitor full of accent
 * cells would say they were all equally worth looking at.
 */
function kindClass(kind: string): string {
  if (kind === "simulate" || kind === "optimize") return "sim-kind-thermal";
  if (kind === "mesh" || kind === "mesh_inspect" || kind === "export") return "sim-kind-mesh";
  return "";
}

export function ProcessesPanel() {
  const [expanded, setExpanded] = createSignal<string | null>(null);
  const [detail, setDetail] = createSignal<JobDetail | null>(null);
  const [now, setNow] = createSignal(clock());
  const [cancelling, setCancelling] = createSignal<string[]>([]);

  function clock(): number {
    return typeof performance === "undefined" ? Date.now() : performance.now();
  }

  // Polling runs while this window is open, and stops with it. The ticker is
  // separate and cheap: it only advances the running rows' elapsed readouts
  // between polls, so a five-second solve counts up instead of stepping.
  onCleanup(watchJobs());
  const ticker = setInterval(() => setNow(clock()), TICK_MS);
  onCleanup(() => clearInterval(ticker));

  const jobs = (): JobSummary[] => jobsSnapshot()?.jobs ?? [];
  const running = () => jobs().filter(isPending);
  const history = () => jobs().filter((job) => !isPending(job));
  const totals = () => jobsSnapshot()?.totals ?? null;
  const store = () => jobsSnapshot()?.store ?? null;
  /** Seconds since the poll the rows were drawn from. */
  const since = () => Math.max(0, (now() - jobsSnapshotAt()) / 1_000);
  const approximate = () => totals()?.sampling === "rusage";

  // An expanded row's sample series is a second request, so it is made only
  // for the row that is open and re-made on every poll while it runs.
  createEffect(() => {
    const id = expanded();
    jobsSnapshot();
    if (!id) {
      setDetail(null);
      return;
    }
    void api.jobDetail(id).then((found) => {
      if (expanded() === id) setDetail(found);
    });
  });

  const toggle = (id: string) => setExpanded((current) => (current === id ? null : id));

  // A window that showed a failure and offered "show in Processes" names the
  // row it meant; open it rather than leaving the user to find it.
  createEffect(() => {
    const wanted = focusedJob();
    if (!wanted) return;
    setFocusedJob(null);
    setExpanded(wanted);
    void refreshJobs();
  });

  const cancel = async (job: JobSummary) => {
    setCancelling((ids) => [...ids, job.job_id]);
    try {
      await cancelJob(job.job_id);
    } finally {
      setCancelling((ids) => ids.filter((id) => id !== job.job_id));
    }
  };

  /**
   * Re-open a finished job's result in the panel that can draw it.
   *
   * This window knows nothing about solved fields or trajectories, and the
   * panel that does may not even be mounted. So the click publishes the
   * reference, puts the user in the mode whose desk holds that panel, and
   * raises the window; the panel consumes the request when it mounts.
   */
  const reopen = (job: JobSummary) => {
    setRequestedJob({
      job_id: job.job_id,
      source_hash: job.source_hash,
      kind: job.kind,
      fields: job.fields ?? {},
    });
    // Deterministic rather than clever: each kind of result has exactly one
    // window that knows how to draw it — a solved field in Results, a
    // generated mesh in Meshes, a trajectory in Optimize — so the click goes
    // to the desk that holds that window. Guessing at the mode the user is
    // already in would sometimes publish a request to a window that is not
    // mounted, and the click would appear to do nothing.
    const target: WindowId =
      job.kind === "optimize" ? "optimize" : job.kind === "mesh_inspect" ? "meshes" : "results";
    const mode = job.kind === "optimize" ? "model" : "simulate";
    if (editingMode() !== mode) setEditingMode(mode);
    // The desk is rebuilt by an effect on the mode; raise the window after it.
    setTimeout(() => windowManager()?.open(target), 0);
  };

  const sampleValues = (): number[] => {
    const series = detail()?.sample_series ?? [];
    return series.map((sample) => sample.cpu_percent);
  };

  const rowSamples = (job: JobSummary) => (
    <Show when={expanded() === job.job_id}>
      <div class="proc-detail" data-testid={`processes-detail-${job.job_id}`}>
        <Show
          when={sampleValues().length > 1}
          fallback={
            <p class="sim-help">
              {approximate()
                ? "Per-sample CPU needs psutil on the server."
                : "No samples yet — this job finished inside one sample interval."}
            </p>
          }
        >
          <Sparkline
            values={sampleValues()}
            ariaLabel={`CPU of ${jobLabel(job)}`}
            testId={`processes-samples-${job.job_id}`}
          />
        </Show>
        <StatRow>
          <Stat label="pid" value={job.pid ?? "–"} />
          <Stat label="cpu time" value={formatDuration(job.cpu_seconds)} />
          <Stat label="peak cpu" value={formatPercent(job.peak_cpu_percent)} />
          <Stat label="peak rss" value={formatBytes(job.peak_rss_bytes)} />
          <Stat label="samples" value={job.samples} />
        </StatRow>
        <Show when={job.error}>
          <p class="sim-error">{job.error}</p>
        </Show>
      </div>
    </Show>
  );

  return (
    <aside class="sim-panel proc-panel" data-testid="processes-panel">
      <header>
        <span>
          <small>SERVER</small>
          Processes
        </span>
      </header>

      <Section
        title="Running"
        count={running().length}
        testId="processes-running"
      >
        <Show
          when={running().length > 0}
          fallback={
            <p class="sim-help" data-testid="processes-running-empty">
              Nothing is running — the server is idle.
            </p>
          }
        >
          <ul class="proc-list">
            <For each={running()}>
              {(job) => (
                <li class="proc-row" data-testid={`processes-job-${job.job_id}`}>
                  <div class="proc-line">
                    <button
                      type="button"
                      class="proc-open"
                      onClick={() => toggle(job.job_id)}
                      title="Show this job's resource samples"
                      data-testid={`processes-expand-${job.job_id}`}
                    >
                      <span class={`sim-kind ${kindClass(job.kind)}`}>{job.kind}</span>
                      <strong>{jobLabel(job)}</strong>
                    </button>
                    <span class="proc-figures">
                      <b data-testid={`processes-elapsed-${job.job_id}`}>
                        {formatDuration(elapsedOf(job, since()))}
                      </b>
                      <Show when={!approximate()}>
                        <b>{formatPercent(job.cpu_percent)}</b>
                      </Show>
                      <b>{formatBytes(job.rss_bytes)}</b>
                    </span>
                    <button
                      type="button"
                      class="sim-delete"
                      disabled={cancelling().includes(job.job_id)}
                      onClick={() => void cancel(job)}
                      title={`Cancel this ${job.kind} job`}
                      aria-label={`Cancel ${jobLabel(job)}`}
                      data-testid={`processes-cancel-${job.job_id}`}
                    >
                      ×
                    </button>
                  </div>
                  <Show when={job.progress?.step !== undefined}>
                    <p class="proc-progress" data-testid={`processes-progress-${job.job_id}`}>
                      step {job.progress!.step}/{job.progress!.steps ?? "?"}
                      <Show when={job.progress?.objective !== undefined}>
                        {" · objective "}
                        <b>{job.progress!.objective!.toPrecision(4)}</b>
                      </Show>
                    </p>
                  </Show>
                  {rowSamples(job)}
                </li>
              )}
            </For>
          </ul>
        </Show>
      </Section>

      <Section title="History" count={history().length} testId="processes-history">
        <Show
          when={history().length > 0}
          fallback={
            <p class="sim-help" data-testid="processes-history-empty">
              Nothing has finished yet this session.
            </p>
          }
        >
          <ul class="proc-list">
            <For each={history()}>
              {(job) => (
                <li class="proc-row" data-testid={`processes-job-${job.job_id}`}>
                  <div class="proc-line">
                    <button
                      type="button"
                      class="proc-open"
                      onClick={() =>
                        REPLAYABLE.has(job.kind) && job.result_available
                          ? reopen(job)
                          : toggle(job.job_id)
                      }
                      title={
                        REPLAYABLE.has(job.kind) && job.result_available
                          ? "Open this result again"
                          : "Show this job's resource samples"
                      }
                      data-testid={`processes-open-${job.job_id}`}
                    >
                      <span class={`sim-kind ${kindClass(job.kind)}`}>{job.kind}</span>
                      <strong>{jobLabel(job)}</strong>
                    </button>
                    <span class="proc-figures">
                      <b>{formatDuration(job.elapsed_s)}</b>
                      <Show when={!approximate()}>
                        <b>{formatPercent(job.peak_cpu_percent)}</b>
                      </Show>
                      <b>{formatBytes(job.peak_rss_bytes)}</b>
                      <i
                        class="proc-status"
                        data-status={job.status}
                        data-testid={`processes-status-${job.job_id}`}
                      >
                        {job.status}
                      </i>
                    </span>
                    <button
                      type="button"
                      class="sim-delete"
                      onClick={() => toggle(job.job_id)}
                      title="Show this job's resource samples"
                      aria-label={`Details of ${jobLabel(job)}`}
                      data-testid={`processes-expand-${job.job_id}`}
                    >
                      ▾
                    </button>
                  </div>
                  {rowSamples(job)}
                </li>
              )}
            </For>
          </ul>
        </Show>
      </Section>

      <Section
        title="Load"
        count={totals()?.host.cpu_count ? `${totals()!.host.cpu_count} cores` : ""}
        testId="processes-load"
        actions={
          <button
            type="button"
            class="sim-add-inline"
            onClick={() => void api.clearJobs().then(refreshJobs)}
            title="Drop every finished job from the registry"
            data-testid="processes-clear"
          >
            Clear
          </button>
        }
      >
        <Show when={cpuHistory().length > 1}>
          <Sparkline
            values={cpuHistory()}
            ariaLabel="Total worker CPU over the last minute"
            testId="processes-cpu"
          />
        </Show>
        <StatRow testId="processes-totals">
          <Stat label="running" value={totals()?.running ?? 0} />
          <Stat label="worker cpu" value={formatPercent(totals()?.cpu_percent)} />
          <Stat label="worker rss" value={formatBytes(totals()?.rss_bytes)} />
          <Stat label="server rss" value={formatBytes(totals()?.server.rss_bytes)} />
          <Stat label="free" value={formatBytes(totals()?.host.mem_available)} />
          <Stat label="uptime" value={formatDuration(totals()?.uptime_s)} />
        </StatRow>
        <Show when={store()}>
          {(current) => (
            <p class="proc-budget" data-testid="processes-budget">
              {current().jobs}/{current().max_jobs} jobs ·{" "}
              {formatBytes(current().result_bytes)}/{formatBytes(current().max_result_bytes)} ·{" "}
              {current().evicted_results} results evicted
            </p>
          )}
        </Show>
        <Show when={approximate()}>
          <p class="sim-note" data-testid="processes-degraded">
            psutil is not installed on the server, so CPU and memory are
            totals over all worker processes rather than per-job samples.
            Install it with <code> pip install cadjoint[viewer]</code>.
          </p>
        </Show>
        <Show when={pollError()}>
          <p class="sim-error" data-testid="processes-error">
            {pollError()}
          </p>
        </Show>
      </Section>
    </aside>
  );
}
