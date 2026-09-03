/**
 * The optimization card list — shared by the Model-mode Optimize panel and
 * the Simulate panel's Optimize tab, so optimization always sits next to the
 * geometry and the simulation it drives.
 *
 * `Optimization(...)` declarations come from the compile payload; the cards
 * edit their steps/learning_rate through /patch and run them through
 * POST /api/optimize. A successful run returns the program with the
 * optimized literals written back, adopted exactly like a patch response;
 * a study-backed run additionally returns the optimized design's solved
 * field, which is published to the Results tab via the shared
 * optimizeSimulate signal.
 *
 * The replay player animates the run's trajectory by substituting each
 * snapshot's literals client-side and ghost-compiling: the editor text and
 * viewport morph frame by frame at the honest pace of a real compile, and
 * nothing lands in undo history — only the adopted final source does.
 */

import { For, Show, createEffect, createSignal, onCleanup, onMount } from "solid-js";
import * as api from "../api";
import {
  awaitJobResult,
  dropJobRef,
  findRunningJob,
  isStale,
  jobsSnapshot,
  loadJobRef,
  requestedJob,
  saveJobRef,
  sceneKey,
  sourceHash,
  takeRequestedJob,
  watchJobs,
  type JobRef,
} from "../jobs";
import {
  deleteOptimizationRequest,
  optimizeRequest,
  playbackFrames,
  setOptimizationValueRequest,
} from "../optimize";
import { formatScalar } from "../simulation";
import {
  busy,
  optimizations,
  optimizeRun,
  sceneName,
  setOptimizeAutoPlay,
  setOptimizePlayer,
  setOptimizeRun,
  setOptimizeSimulate,
  source,
  type OptimizeRunState,
} from "../state";
import { TrajectoryPlayer } from "./TrajectoryPlayer";
import { Card, CardHeader, CardList, NumberField, Sparkline } from "./ui";
import type { OptimizationPayload, OptimizeResponse } from "../types";

export interface OptimizeCardsProps {
  /** Serialized /patch queue owned by the app shell. */
  onPatch: (body: Record<string, unknown>) => Promise<void>;
  /** Adopt server-produced source like a patch response (commit + rerun). */
  onAdoptSource: (source: string) => Promise<void>;
  /** Compile-and-render a transient program without committing it. */
  onGhostCompile: (source: string) => Promise<boolean>;
}

const formatValue = (value: number | number[] | undefined): string => {
  if (value === undefined) return "–";
  if (Array.isArray(value)) return `[${value.map((item) => formatScalar(item)).join(", ")}]`;
  return formatScalar(value);
};

/** The objective label: "mean(sink-conduction)" for study-backed runs. */
export function objectiveLabel(optimization: OptimizationPayload): string {
  if (optimization.study) {
    return `${optimization.metric ?? "objective"}(${optimization.study})`;
  }
  // A study-backed declaration may state no objective expression at all —
  // the study's metric is the objective — so the field can be null.
  return optimization.objective ?? "objective";
}

/** Parameter rows shown before the list collapses behind "+N more". */
const PARAMETER_PREVIEW = 6;

/** What the streaming run has reported so far. */
interface LiveRun {
  step: number;
  steps: number;
  /** Objective per received progress event, for the growing sparkline. */
  objectives: number[];
  /** Seconds since the run started, per the server. */
  elapsed: number | null;
}

/** "34s", "3.4s" — coarse above ten seconds, one decimal below. */
const formatSeconds = (value: number): string =>
  `${value >= 10 ? value.toFixed(0) : value.toFixed(1)}s`;

export function OptimizeCards(props: OptimizeCardsProps) {
  const [running, setRunning] = createSignal<string | null>(null);
  const [error, setError] = createSignal("");
  /** Live progress of the in-flight run (streaming /api/optimize only). */
  const [live, setLive] = createSignal<LiveRun | null>(null);
  /** Optimization names whose full parameter list is expanded. */
  const [expanded, setExpanded] = createSignal<string[]>([]);
  /**
   * The running run's job id: the handle Cancel needs.
   *
   * It arrives from whichever source speaks first — every NDJSON line the
   * run streams carries it, and the shared job poll finds it about a second
   * in (see the effect below, which matters because a study-backed run's
   * first line can be a minute away).
   */
  const [jobId, setJobId] = createSignal<string | null>(null);
  /** The reference to the displayed run, for staleness and for storage. */
  const [heldRef, setHeldRef] = createSignal<JobRef | null>(null);
  const [documentHash, setDocumentHash] = createSignal<string | null>(null);
  /** A stored run the server no longer has. */
  const [expired, setExpired] = createSignal("");

  const scene = () => sceneKey(sceneName());
  let unmounted = false;
  onCleanup(() => {
    unmounted = true;
  });

  createEffect(() => {
    const text = source();
    void sourceHash(text).then((hash) => setDocumentHash(hash));
  });

  /** Whether the displayed run descended a program that has since changed. */
  const stale = (): boolean => {
    const ref = heldRef();
    return ref ? isStale(ref, documentHash()) : false;
  };

  /**
   * Adopt a stored optimize payload as the displayed run.
   *
   * Everything the player needs is in the response the run already
   * returned — history, trajectory, initial and final parameters, and the
   * solved field of a study-backed run — so replaying an old run costs one
   * fetch and no optimizer steps. The program text is deliberately *not*
   * adopted: restoring a view must never rewrite the editor.
   */
  const adoptRun = (name: string, payload: OptimizeResponse): boolean => {
    if (!payload.ok || !payload.source) return false;
    const declared = optimizations().find((entry) => entry.name === name);
    setOptimizeRun({
      name,
      source: payload.source,
      history: payload.history ?? [],
      trajectory: payload.trajectory ?? [],
      parameters: payload.parameters ?? {},
      initial: payload.initial ?? {},
      study: declared?.study ?? null,
    });
    setOptimizePlayer({
      frame: Math.max(playbackFrames((payload.trajectory ?? []).length).length - 1, 0),
      playing: false,
    });
    const simulate = payload.simulate;
    if (simulate?.mesh) {
      setOptimizeSimulate({
        name,
        field: simulate.field ?? "field",
        mesh: simulate.mesh,
        result: simulate.result ?? null,
        meshInfo: simulate.mesh_info ?? null,
      });
    }
    return true;
  };

  /** Fetch one run's payload by job id and show it. */
  const restore = async (ref: JobRef) => {
    const outcome = await awaitJobResult<OptimizeResponse>(ref.job_id, {
      stopped: () => unmounted,
    });
    if (unmounted) return;
    if (outcome.state === "gone") {
      dropJobRef(scene(), "optimize");
      setExpired("That run is no longer stored on the server — run it again.");
      return;
    }
    if (outcome.state !== "ok") return;
    const name =
      typeof ref.fields?.name === "string" ? ref.fields.name : (optimizations()[0]?.name ?? "");
    if (!adoptRun(name, outcome.payload)) return;
    setExpired("");
    setHeldRef(ref);
    saveJobRef(scene(), ref);
  };

  // A run survives a mode switch in memory; this is what makes it survive a
  // reload, and what lets the Processes window hand an older run back.
  onMount(() => {
    if (optimizeRun()) return;
    const stored = loadJobRef(scene(), "optimize");
    if (stored) void restore(stored);
  });

  createEffect(() => {
    requestedJob();
    const ref = takeRequestedJob("optimize");
    if (ref) void restore(ref);
  });

  /**
   * Arm Cancel from the registry, not only from the stream.
   *
   * A study-backed run's first NDJSON line does not arrive until the first
   * step has finished, and on a cold cache that is the better part of a
   * minute — precisely the minute in which somebody realises they meant to
   * change something first. So while a run is in flight the card also
   * watches the shared job poll and takes the id from there; whichever
   * source speaks first arms the button.
   */
  createEffect(() => {
    if (running() === null) return;
    onCleanup(watchJobs());
  });

  createEffect(() => {
    if (running() === null || jobId() !== null) return;
    const snap = jobsSnapshot();
    const found = snap ? findRunningJob(snap.jobs, "optimize", documentHash()) : null;
    if (found) setJobId(found.job_id);
  });

  const run = async (optimization: OptimizationPayload) => {
    // Retire the previous run first: any mounted player unmounts and stops.
    setOptimizeRun(null);
    setOptimizeAutoPlay(false);
    setRunning(optimization.name);
    setJobId(null);
    setExpired("");
    setLive({ step: 0, steps: optimization.steps, objectives: [], elapsed: null });
    setError("");
    const posted = source();
    try {
      const response = await api.optimize(
        optimizeRequest(posted, optimization.name),
        // Streaming servers report each step; the card shows the objective
        // descending live. (A non-streaming server sends no events and the
        // bar stays indeterminate until the single response lands.)
        (progress) => {
          setLive((current) => ({
            step: progress.step,
            steps: progress.steps > 0 ? progress.steps : optimization.steps,
            objectives: [...(current?.objectives ?? []), progress.objective],
            elapsed: progress.elapsed,
          }));
        },
        // Stamped on every streamed event, so Cancel is live from step one.
        (id) => setJobId(id),
      );
      if (response.error_kind === "cancelled") {
        // Stopping a run is a decision, not a failure: no notice, no toast.
        return;
      }
      if (!response.ok || !response.source) {
        setError(response.error ?? "The optimization failed.");
        return;
      }
      setOptimizeRun({
        name: optimization.name,
        source: response.source,
        history: response.history ?? [],
        trajectory: response.trajectory ?? [],
        parameters: response.parameters ?? {},
        initial: response.initial ?? {},
        study: optimization.study ?? null,
      });
      setOptimizePlayer({
        frame: Math.max(playbackFrames((response.trajectory ?? []).length).length - 1, 0),
        playing: false,
      });
      // A study-backed run ends with the optimized design's solved field —
      // publish it so the Simulate panel's Results tab shows the end state.
      const simulate = response.simulate;
      if (simulate?.mesh) {
        setOptimizeSimulate({
          name: optimization.name,
          field: simulate.field ?? "field",
          mesh: simulate.mesh,
          result: simulate.result ?? null,
          meshInfo: simulate.mesh_info ?? null,
        });
      }
      // The run is now a stored result: the id and the hash of the program
      // it descended are all a remount needs to bring the trajectory back.
      if (response.job_id) {
        const ref: JobRef = {
          job_id: response.job_id,
          source_hash: await sourceHash(posted),
          kind: "optimize",
          fields: { name: optimization.name },
        };
        setHeldRef(ref);
        saveJobRef(scene(), ref);
      }
      // The optimizer is a patch layer: adopt its source like any edit.
      await props.onAdoptSource(response.source);
      // Queue exactly one unprompted replay: the geometry morphing from the
      // initial design to the optimized one IS the result. Whichever
      // trajectory player is mounted where the user lands (this card, or
      // the Results tab for study-backed runs) consumes it.
      if ((response.trajectory ?? []).length > 1) setOptimizeAutoPlay(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(null);
      setJobId(null);
      setLive(null);
    }
  };

  /** Kill the running optimizer; its own request answers with `cancelled`. */
  const cancel = async () => {
    const id = jobId();
    if (id) await api.cancelJob(id);
  };

  const patch = async (body: Record<string, unknown>) => {
    setError("");
    await props.onPatch(body);
  };

  const historyFor = (name: string): OptimizeRunState | null => {
    const current = optimizeRun();
    return current && current.name === name ? current : null;
  };

  return (
    <>
      <Show
        when={optimizations().length > 0}
        fallback={
          <p class="sim-help" data-testid="optimize-empty">
            No optimizations declared — add an Optimization(...) to the program.
          </p>
        }
      >
        <CardList testId="optimize-list">
          <For each={optimizations()}>
            {(optimization) => (
              <Card testId={`optimize-${optimization.name}`}>
                <CardHeader
                  kind={optimization.method}
                  kindClass="opt-kind"
                  name={optimization.name}
                  onDelete={() => void patch(deleteOptimizationRequest(optimization))}
                  deleteTitle="Delete this optimization from the code"
                  deleteAriaLabel={`Delete optimization ${optimization.name}`}
                  deleteTestId={`optimize-delete-${optimization.name}`}
                />

                <p class="opt-objective">
                  minimize <code>{objectiveLabel(optimization)}</code>
                </p>

                <Show
                  when={optimization.editable}
                  fallback={
                    <p class="sim-note">Defined dynamically in code — edit it there.</p>
                  }
                >
                  <div class="sim-args">
                    <NumberField
                      label="steps"
                      min="1"
                      step="1"
                      value={optimization.steps}
                      disabled={running() !== null}
                      onCommit={(value) =>
                        void patch(
                          setOptimizationValueRequest(
                            optimization,
                            "steps",
                            Math.round(value),
                          ),
                        )
                      }
                      testId={`optimize-steps-${optimization.name}`}
                    />
                    <NumberField
                      label="learning rate"
                      step="0.01"
                      value={optimization.learning_rate}
                      disabled={running() !== null}
                      onCommit={(value) =>
                        void patch(
                          setOptimizationValueRequest(
                            optimization,
                            "learning_rate",
                            value,
                          ),
                        )
                      }
                      testId={`optimize-lr-${optimization.name}`}
                    />
                  </div>
                </Show>

                <Show
                  when={running() === optimization.name}
                  fallback={
                    <button
                      type="button"
                      class="sim-run"
                      disabled={running() !== null || busy()}
                      onClick={() => void run(optimization)}
                      title="Run this optimization through the differentiable path"
                      data-testid={`optimize-run-${optimization.name}`}
                    >
                      Run
                    </button>
                  }
                >
                  <button
                    type="button"
                    class="sim-run"
                    disabled={jobId() === null}
                    onClick={() => void cancel()}
                    title="Stop this run and free the worker"
                    data-testid={`optimize-cancel-${optimization.name}`}
                  >
                    {jobId() ? "Cancel" : "Optimizing…"}
                  </button>
                </Show>

                {/* Live progress while this card's run is in flight. */}
                <Show when={running() === optimization.name ? live() : null}>
                  {(current) => (
                    <div
                      class="opt-live"
                      data-testid={`optimize-progress-${optimization.name}`}
                    >
                      <div
                        class="opt-progress"
                        classList={{ indeterminate: current().objectives.length === 0 }}
                        role="progressbar"
                        aria-valuemin="0"
                        aria-valuemax={current().steps}
                        aria-valuenow={current().step}
                      >
                        <i
                          style={{
                            width: `${Math.min(
                              100,
                              (100 * current().step) / Math.max(current().steps, 1),
                            ).toFixed(1)}%`,
                          }}
                        />
                      </div>
                      <div class="opt-live-stats">
                        <span data-testid={`optimize-progress-step-${optimization.name}`}>
                          step {current().step}/{current().steps}
                        </span>
                        <Show
                          when={current().elapsed !== null && current().step > 0}
                          fallback={<span>working…</span>}
                        >
                          <span>
                            {formatSeconds(current().elapsed!)}
                            {" · "}
                            {formatSeconds(current().elapsed! / current().step)}/step
                          </span>
                        </Show>
                        <Show when={current().objectives.length > 0}>
                          <span>
                            objective{" "}
                            <b>
                              {formatScalar(
                                current().objectives[current().objectives.length - 1],
                              )}
                            </b>
                          </span>
                        </Show>
                      </div>
                      <Show when={current().objectives.length > 1}>
                        <Sparkline
                          values={current().objectives}
                          ariaLabel="Objective so far"
                          testId={`optimize-live-${optimization.name}`}
                        />
                      </Show>
                    </div>
                  )}
                </Show>

                <Show when={historyFor(optimization.name)}>
                  {(current) => (
                    <div class="opt-result" data-testid={`optimize-result-${optimization.name}`}>
                      {/* The shared trajectory player (sparkline + cursor +
                          scrubber); runs without a replayable trajectory
                          fall back to the plain history sparkline. */}
                      <Show
                        when={current().trajectory.length > 1}
                        fallback={
                          <Sparkline
                            values={current().history.map((entry) => entry.objective)}
                            ariaLabel="Objective history"
                            testId={`optimize-history-${optimization.name}`}
                          />
                        }
                      >
                        <TrajectoryPlayer
                          onGhostCompile={props.onGhostCompile}
                          sparkTestId={`optimize-history-${optimization.name}`}
                          fieldNote={Boolean(optimization.study)}
                        />
                      </Show>
                      <Show when={stale()}>
                        <span class="sim-kind" data-testid={`optimize-stale-${optimization.name}`}>
                          stale · source changed
                        </span>
                      </Show>
                      <div class="opt-summary">
                        <span>
                          objective{" "}
                          <b>
                            {formatScalar(current().history[0]?.objective ?? NaN)}
                            {" → "}
                            {formatScalar(
                              current().history[current().history.length - 1]?.objective ?? NaN,
                            )}
                          </b>
                        </span>
                        <span>{current().history.length} steps</span>
                      </div>
                    </div>
                  )}
                </Show>

                {/* The free parameters, after a run with initial→final
                    values. Long lists collapse so the sparkline and the
                    player stay in reach without scrolling. */}
                <ul class="opt-parameters" data-testid={`optimize-parameters-${optimization.name}`}>
                  <For
                    each={
                      expanded().includes(optimization.name)
                        ? optimization.parameters
                        : optimization.parameters.slice(0, PARAMETER_PREVIEW)
                    }
                  >
                    {(parameter) => {
                      const runResult = () => historyFor(optimization.name);
                      return (
                        <li>
                          <code>{parameter}</code>
                          <Show when={runResult()}>
                            {(current) => (
                              <span class="opt-parameter-values">
                                {formatValue(current().initial[parameter])}
                                <i>→</i>
                                {formatValue(current().parameters[parameter])}
                              </span>
                            )}
                          </Show>
                        </li>
                      );
                    }}
                  </For>
                </ul>
                <Show when={optimization.parameters.length > PARAMETER_PREVIEW}>
                  <button
                    type="button"
                    class="opt-parameters-toggle"
                    onClick={() =>
                      setExpanded((names) =>
                        names.includes(optimization.name)
                          ? names.filter((name) => name !== optimization.name)
                          : [...names, optimization.name],
                      )
                    }
                    data-testid={`optimize-parameters-toggle-${optimization.name}`}
                  >
                    {expanded().includes(optimization.name)
                      ? "Show fewer parameters"
                      : `+${optimization.parameters.length - PARAMETER_PREVIEW} more parameters`}
                  </button>
                </Show>
              </Card>
            )}
          </For>
        </CardList>
      </Show>

      <Show when={expired()}>
        <p class="sim-note" data-testid="optimize-expired">
          {expired()}
        </p>
      </Show>
      <Show when={error()}>
        <p class="sim-error" data-testid="optimize-error">
          {error()}
        </p>
      </Show>
    </>
  );
}
