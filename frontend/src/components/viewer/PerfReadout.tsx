/**
 * The viewport's bottom-right readout: what drawing the picture costs (FPS,
 * CPU per frame, resolution, scene size).
 *
 * Sampled twice a second and replaced, never tweened (guide M3). An on-demand
 * renderer that is not drawing says `idle`, not `0`.
 */

import { Show, createSignal, onCleanup } from "solid-js";
import { nodes, simView } from "../../state";
import type { Renderer } from "../../viewer/renderer";

const SAMPLE_MS = 500;

function Row(props: { label: string; value: string; unit?: string }) {
  return (
    <>
      <dt>{props.label}</dt>
      <dd>
        {props.value}
        <Show when={props.unit}>
          <small> {props.unit}</small>
        </Show>
      </dd>
    </>
  );
}

export function PerfReadout(props: { renderer: Renderer; show: boolean }) {
  const [stats, setStats] = createSignal(props.renderer.frameStats);
  const timer = window.setInterval(() => setStats(props.renderer.frameStats), SAMPLE_MS);
  onCleanup(() => clearInterval(timer));

  const mesh = () => {
    const view = simView();
    if (!view) return null;
    return {
      elements: view.info?.elements ?? null,
      triangles: Math.floor(view.payload.indices.length / 3),
    };
  };

  return (
    <Show when={props.show}>
      <dl class="title-block perf-readout" data-testid="perf-readout">
        <Row label="FPS" value={stats().fps === null ? "idle" : String(stats().fps)} />
        <Row
          label="CPU"
          value={stats().cpuMs === null ? "—" : stats().cpuMs!.toFixed(1)}
          unit={stats().cpuMs === null ? undefined : "ms/frame"}
        />
        <Row label="RES" value={`${stats().width} × ${stats().height}`} unit="px" />
        <Row label="NODES" value={String(nodes().length)} />
        <Show when={mesh()}>
          {(m) => (
            <>
              <Row label="ELEMENTS" value={m().elements === null ? "—" : m().elements!.toLocaleString("en-US")} />
              <Row label="TRIS" value={m().triangles.toLocaleString("en-US")} />
            </>
          )}
        </Show>
      </dl>
    </Show>
  );
}
