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

import { For, Show, createMemo, createSignal, onCleanup } from "solid-js";
import * as api from "../api";
import {
  advancePlayer,
  deleteOptimizationRequest,
  frameObjective,
  optimizeRequest,
  playbackFrames,
  setOptimizationValueRequest,
  sparklineCursorX,
  sparklinePoints,
  startPlayer,
  substituteParameters,
  type PlayerState,
} from "../optimize";
import { formatScalar } from "../simulation";
import { busy, optimizations, setOptimizeSimulate, source } from "../state";
import type {
  OptimizationPayload,
  OptimizeHistoryEntry,
  OptimizeTrajectoryEntry,
} from "../types";

export interface OptimizeCardsProps {
  /** Serialized /patch queue owned by the app shell. */
  onPatch: (body: Record<string, unknown>) => Promise<void>;
  /** Adopt server-produced source like a patch response (commit + rerun). */
  onAdoptSource: (source: string) => Promise<void>;
  /** Compile-and-render a transient program without committing it. */
  onGhostCompile: (source: string) => Promise<boolean>;
}

/** A completed run, kept for the summary block and the replay player. */
interface RunResult {
  name: string;
  /** The adopted program with the final literals written back. */
  source: string;
  history: OptimizeHistoryEntry[];
  trajectory: OptimizeTrajectoryEntry[];
  parameters: Record<string, number | number[]>;
  initial: Record<string, number | number[]>;
}

const SPARK_WIDTH = 220;
const SPARK_HEIGHT = 44;
/** Replay pace: one ghost compile per frame, plus a beat to look at it. */
const FRAME_MILLISECONDS = 1_500;

const parse = (raw: string): number | null => {
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
};

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
  return optimization.objective;
}

/** Parameter rows shown before the list collapses behind "+N more". */
const PARAMETER_PREVIEW = 6;

export function OptimizeCards(props: OptimizeCardsProps) {
  const [running, setRunning] = createSignal<string | null>(null);
  const [error, setError] = createSignal("");
  const [result, setResult] = createSignal<RunResult | null>(null);
  const [player, setPlayer] = createSignal<PlayerState>({ frame: 0, playing: false });
  /** Optimization names whose full parameter list is expanded. */
  const [expanded, setExpanded] = createSignal<string[]>([]);

  const frames = createMemo(() => {
    const run = result();
    return run ? playbackFrames(run.trajectory.length) : [];
  });

  // Ghost compiles are serialized: scrubbing queues at most one frame, and a
  // new request simply replaces the queued one until the compile in flight
  // finishes — the slider stays responsive while the render honestly lags.
  // "final" restores the exact adopted source instead of a substitution.
  let replayBusy = false;
  let queuedFrame: number | "final" | null = null;

  const renderFrame = async (frameIndex: number | "final") => {
    const run = result();
    if (!run) return;
    if (frameIndex === "final") {
      await props.onGhostCompile(run.source);
      return;
    }
    const frameList = frames();
    if (frameList.length === 0) return;
    const entry = run.trajectory[frameList[Math.min(frameIndex, frameList.length - 1)]];
    if (!entry) return;
    await props.onGhostCompile(substituteParameters(run.source, entry.parameters));
  };

  const showFrame = (frameIndex: number | "final") => {
    queuedFrame = frameIndex;
    if (replayBusy) return;
    replayBusy = true;
    void (async () => {
      while (queuedFrame !== null) {
        const next = queuedFrame;
        queuedFrame = null;
        await renderFrame(next);
      }
      replayBusy = false;
    })();
  };

  let playTimer: ReturnType<typeof setInterval> | undefined;

  const stopPlayback = (restoreFinal: boolean) => {
    if (playTimer !== undefined) {
      clearInterval(playTimer);
      playTimer = undefined;
    }
    setPlayer((state) => ({ ...state, playing: false }));
    if (restoreFinal && result()) {
      setPlayer({ frame: Math.max(frames().length - 1, 0), playing: false });
      showFrame("final");
    }
  };

  const play = () => {
    const count = frames().length;
    if (count === 0) return;
    const started = startPlayer(player(), count);
    setPlayer(started);
    showFrame(started.frame);
    if (playTimer !== undefined) clearInterval(playTimer);
    playTimer = setInterval(() => {
      const next = advancePlayer(player(), frames().length);
      setPlayer(next);
      if (!next.playing) {
        stopPlayback(true);
        return;
      }
      showFrame(next.frame);
    }, FRAME_MILLISECONDS);
  };

  const scrub = (frameIndex: number) => {
    stopPlayback(false);
    setPlayer({ frame: frameIndex, playing: false });
    showFrame(frameIndex);
  };

  onCleanup(() => {
    // Leaving the mode mid-replay must not strand a ghost frame on screen.
    const run = result();
    const mid = playTimer !== undefined || player().frame < frames().length - 1;
    if (playTimer !== undefined) clearInterval(playTimer);
    if (run && mid) showFrame("final");
  });

  const run = async (optimization: OptimizationPayload) => {
    stopPlayback(false);
    setRunning(optimization.name);
    setError("");
    try {
      const response = await api.optimize(optimizeRequest(source(), optimization.name));
      if (!response.ok || !response.source) {
        setError(response.error ?? "The optimization failed.");
        return;
      }
      setResult({
        name: optimization.name,
        source: response.source,
        history: response.history ?? [],
        trajectory: response.trajectory ?? [],
        parameters: response.parameters ?? {},
        initial: response.initial ?? {},
      });
      setPlayer({
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
      // The optimizer is a patch layer: adopt its source like any edit.
      await props.onAdoptSource(response.source);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(null);
    }
  };

  const patch = async (body: Record<string, unknown>) => {
    setError("");
    await props.onPatch(body);
  };

  const historyFor = (name: string): RunResult | null => {
    const current = result();
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
        <ul class="sim-studies" data-testid="optimize-list">
          <For each={optimizations()}>
            {(optimization) => (
              <li class="sim-study" data-testid={`optimize-${optimization.name}`}>
                <div class="sim-study-head">
                  <span class="sim-kind opt-kind">{optimization.method}</span>
                  <strong>{optimization.name}</strong>
                  <button
                    type="button"
                    class="sim-delete"
                    onClick={() => void patch(deleteOptimizationRequest(optimization))}
                    title="Delete this optimization from the code"
                    aria-label={`Delete optimization ${optimization.name}`}
                    data-testid={`optimize-delete-${optimization.name}`}
                  >
                    ×
                  </button>
                </div>

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
                    <label>
                      <span>steps</span>
                      <input
                        type="number"
                        min="1"
                        step="1"
                        value={optimization.steps}
                        disabled={running() !== null}
                        onChange={(event) => {
                          const value = parse(event.currentTarget.value);
                          if (value !== null) {
                            void patch(
                              setOptimizationValueRequest(optimization, "steps", Math.round(value)),
                            );
                          }
                        }}
                        data-testid={`optimize-steps-${optimization.name}`}
                      />
                    </label>
                    <label>
                      <span>learning rate</span>
                      <input
                        type="number"
                        step="0.01"
                        value={optimization.learning_rate}
                        disabled={running() !== null}
                        onChange={(event) => {
                          const value = parse(event.currentTarget.value);
                          if (value !== null) {
                            void patch(
                              setOptimizationValueRequest(optimization, "learning_rate", value),
                            );
                          }
                        }}
                        data-testid={`optimize-lr-${optimization.name}`}
                      />
                    </label>
                  </div>
                </Show>

                <button
                  type="button"
                  class="sim-run"
                  disabled={running() !== null || busy()}
                  onClick={() => void run(optimization)}
                  title="Run this optimization through the differentiable path"
                  data-testid={`optimize-run-${optimization.name}`}
                >
                  {running() === optimization.name ? "Optimizing…" : "Run"}
                </button>

                <Show when={historyFor(optimization.name)}>
                  {(current) => (
                    <div class="opt-result" data-testid={`optimize-result-${optimization.name}`}>
                      <div class="opt-spark">
                        <svg
                          viewBox={`0 0 ${SPARK_WIDTH} ${SPARK_HEIGHT}`}
                          preserveAspectRatio="none"
                          role="img"
                          aria-label="Objective history"
                          data-testid={`optimize-history-${optimization.name}`}
                        >
                          <polyline
                            points={sparklinePoints(
                              current().history.map((entry) => entry.objective),
                              SPARK_WIDTH,
                              SPARK_HEIGHT,
                            )}
                          />
                          <Show when={frames().length > 0}>
                            <line
                              class="opt-cursor"
                              x1={sparklineCursorX(
                                frames()[Math.min(player().frame, frames().length - 1)] ?? 0,
                                current().trajectory.length || current().history.length,
                                SPARK_WIDTH,
                              )}
                              y1="0"
                              x2={sparklineCursorX(
                                frames()[Math.min(player().frame, frames().length - 1)] ?? 0,
                                current().trajectory.length || current().history.length,
                                SPARK_WIDTH,
                              )}
                              y2={SPARK_HEIGHT}
                            />
                          </Show>
                        </svg>
                      </div>
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

                      <Show when={current().trajectory.length > 1}>
                        <div class="opt-player" data-testid="optimize-player">
                          <button
                            type="button"
                            onClick={() => (player().playing ? stopPlayback(false) : play())}
                            title={
                              player().playing
                                ? "Pause the replay"
                                : "Replay the optimization in the viewport"
                            }
                            data-testid="optimize-play"
                          >
                            {player().playing ? "❚❚" : "▶"}
                          </button>
                          <input
                            type="range"
                            min="0"
                            max={Math.max(frames().length - 1, 0)}
                            step="1"
                            value={Math.min(player().frame, Math.max(frames().length - 1, 0))}
                            onInput={(event) => scrub(Number(event.currentTarget.value))}
                            title="Scrub through the optimization steps"
                            data-testid="optimize-scrub"
                          />
                          <span class="opt-frame-label">
                            step {current().trajectory[
                              frames()[Math.min(player().frame, frames().length - 1)] ?? 0
                            ]?.step ?? 0}
                            {" · "}
                            {formatScalar(
                              frameObjective(
                                current().trajectory,
                                frames()[Math.min(player().frame, frames().length - 1)] ?? 0,
                              ) ?? NaN,
                            )}
                          </span>
                        </div>
                      </Show>
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
              </li>
            )}
          </For>
        </ul>
      </Show>

      <Show when={error()}>
        <p class="sim-error" data-testid="optimize-error">
          {error()}
        </p>
      </Show>
    </>
  );
}
