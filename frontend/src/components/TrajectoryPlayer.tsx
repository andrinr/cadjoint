/**
 * The shared optimization trajectory player: sparkline with a moving
 * cursor + highlighted point, play/pause, scrubber, and a step/objective
 * readout.
 *
 * Run and cursor state live in state.ts (`optimizeRun` / `optimizePlayer`),
 * so the player can mount wherever the user lands after a run — the
 * Optimize card, or the Simulate panel's Results tab for study-backed runs
 * — and always shows the same position. Replay renders intermediate
 * parameter snapshots by substituting the literals client-side and
 * ghost-compiling, at the honest pace of a real compile; only the adopted
 * final source ever touches undo history. A pending one-shot auto-play
 * (`optimizeAutoPlay`) is consumed by whichever instance is mounted, so a
 * finished run replays itself once right where the user is looking.
 */

import { Show, createEffect, createMemo, onCleanup } from "solid-js";
import {
  advancePlayer,
  frameObjective,
  playbackFrames,
  sparklineCursorPoint,
  sparklineCursorX,
  sparklinePoints,
  startPlayer,
  substituteParameters,
} from "../optimize";
import { formatScalar } from "../simulation";
import {
  optimizeAutoPlay,
  optimizePlayer,
  optimizeRun,
  setOptimizeAutoPlay,
  setOptimizePlayer,
} from "../state";

const SPARK_WIDTH = 220;
const SPARK_HEIGHT = 44;
/** Replay pace: one ghost compile per frame, plus a beat to look at it. */
const FRAME_MILLISECONDS = 1_500;

export interface TrajectoryPlayerProps {
  /** Compile-and-render a transient program without committing it. */
  onGhostCompile: (source: string) => Promise<boolean>;
  /** data-testid for the sparkline svg (hosts keep their historical ids). */
  sparkTestId?: string;
  /** Study-backed runs: the replay morphs geometry, the field stays final. */
  fieldNote?: boolean;
  /** First intermediate frame shown (e.g. hand the viewport to the scene). */
  onReplayStart?: () => void;
  /** Rested back on the final design (e.g. restore the mesh view). */
  onReplayEnd?: () => void;
}

export function TrajectoryPlayer(props: TrajectoryPlayerProps) {
  const frames = createMemo(() => {
    const run = optimizeRun();
    return run ? playbackFrames(run.trajectory.length) : [];
  });
  /** Objective per trajectory entry — the sparkline the cursor rides on. */
  const values = createMemo(
    () => optimizeRun()?.trajectory.map((entry) => entry.objective) ?? [],
  );
  const frameIndex = () =>
    frames()[Math.min(optimizePlayer().frame, frames().length - 1)] ?? 0;

  // Ghost compiles are serialized: scrubbing queues at most one frame, and
  // a new request replaces the queued one until the compile in flight
  // finishes — the slider stays responsive, the render honestly lags.
  let replayBusy = false;
  let queuedFrame: number | "final" | null = null;
  let replayActive = false;

  const beginReplay = () => {
    if (replayActive) return;
    replayActive = true;
    props.onReplayStart?.();
  };
  const endReplay = () => {
    if (!replayActive) return;
    replayActive = false;
    props.onReplayEnd?.();
  };

  const renderFrame = async (frame: number | "final") => {
    const run = optimizeRun();
    if (!run) return;
    if (frame === "final") {
      await props.onGhostCompile(run.source);
      return;
    }
    const entry = run.trajectory[frames()[Math.min(frame, frames().length - 1)]];
    if (!entry) return;
    await props.onGhostCompile(substituteParameters(run.source, entry.parameters));
  };

  const showFrame = (frame: number | "final") => {
    if (frame === "final") endReplay();
    else beginReplay();
    queuedFrame = frame;
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
    setOptimizePlayer((state) => ({ ...state, playing: false }));
    if (restoreFinal && optimizeRun()) {
      setOptimizePlayer({ frame: Math.max(frames().length - 1, 0), playing: false });
      showFrame("final");
    }
  };

  const play = () => {
    const count = frames().length;
    if (count === 0) return;
    const started = startPlayer(optimizePlayer(), count);
    setOptimizePlayer(started);
    showFrame(started.frame);
    if (playTimer !== undefined) clearInterval(playTimer);
    playTimer = setInterval(() => {
      const next = advancePlayer(optimizePlayer(), frames().length);
      setOptimizePlayer(next);
      if (!next.playing) {
        stopPlayback(true);
        return;
      }
      showFrame(next.frame);
    }, FRAME_MILLISECONDS);
  };

  const scrub = (frame: number) => {
    stopPlayback(false);
    setOptimizePlayer({ frame, playing: false });
    // The last frame is the final design: show the exact adopted source.
    showFrame(frame >= frames().length - 1 ? "final" : frame);
  };

  // A finished run queues exactly one unprompted replay; the mounted
  // player consumes it, so the animation happens where the user landed.
  createEffect(() => {
    if (!optimizeAutoPlay()) return;
    setOptimizeAutoPlay(false);
    if (frames().length > 1) play();
  });

  onCleanup(() => {
    // Leaving the host mid-replay must not strand a ghost frame on screen.
    const mid =
      playTimer !== undefined || optimizePlayer().frame < frames().length - 1;
    if (playTimer !== undefined) clearInterval(playTimer);
    setOptimizePlayer((state) => ({ ...state, playing: false }));
    if (optimizeRun() && mid) showFrame("final");
    else endReplay();
  });

  const cursorPoint = () =>
    sparklineCursorPoint(values(), frameIndex(), SPARK_WIDTH, SPARK_HEIGHT);
  const lastStep = () => {
    const trajectory = optimizeRun()?.trajectory ?? [];
    return trajectory[trajectory.length - 1]?.step ?? 0;
  };

  return (
    <Show when={optimizeRun() && frames().length > 1}>
      <div class="opt-player-strip">
        <div class="opt-spark">
          <svg
            viewBox={`0 0 ${SPARK_WIDTH} ${SPARK_HEIGHT}`}
            preserveAspectRatio="none"
            role="img"
            aria-label="Objective history"
            data-testid={props.sparkTestId ?? "optimize-trajectory"}
          >
            <polyline points={sparklinePoints(values(), SPARK_WIDTH, SPARK_HEIGHT)} />
            <line
              class="opt-cursor"
              x1={sparklineCursorX(frameIndex(), values().length, SPARK_WIDTH)}
              y1="0"
              x2={sparklineCursorX(frameIndex(), values().length, SPARK_WIDTH)}
              y2={SPARK_HEIGHT}
            />
            <Show when={cursorPoint()}>
              {(point) => (
                <circle class="opt-cursor-point" cx={point().x} cy={point().y} r="2.6" />
              )}
            </Show>
          </svg>
        </div>
        <div class="opt-player" data-testid="optimize-player">
          <button
            type="button"
            class="opt-replay"
            onClick={() => (optimizePlayer().playing ? stopPlayback(false) : play())}
            title={
              optimizePlayer().playing
                ? "Pause the replay"
                : "Replay the optimization in the viewport"
            }
            data-testid="optimize-play"
          >
            {optimizePlayer().playing ? "❚❚ Pause" : "▶ Replay"}
          </button>
          <input
            type="range"
            min="0"
            max={Math.max(frames().length - 1, 0)}
            step="1"
            value={Math.min(optimizePlayer().frame, Math.max(frames().length - 1, 0))}
            onInput={(event) => scrub(Number(event.currentTarget.value))}
            title="Scrub through the optimization steps"
            data-testid="optimize-scrub"
          />
          <span class="opt-frame-label" data-testid="optimize-frame-label">
            step {optimizeRun()?.trajectory[frameIndex()]?.step ?? 0}/{lastStep()}
            {" · "}
            {formatScalar(
              frameObjective(optimizeRun()?.trajectory ?? [], frameIndex()) ?? NaN,
            )}
          </span>
        </div>
        <small class="opt-player-hint" data-testid="optimize-player-hint">
          drag to scrub the optimization path
          {props.fieldNote
            ? " · the geometry morphs; the field shown is the final design's"
            : ""}
        </small>
      </div>
    </Show>
  );
}
