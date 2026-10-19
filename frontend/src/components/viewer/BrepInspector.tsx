/**
 * The B-rep inspector's card: what the extraction found, and what is under
 * the pointer.
 *
 * A title block like the scale readout's — caps label, value, one hairline
 * between rows — pinned to the viewport's top-left while the inspector is
 * on. Three parts: the summary (counts, the path that produced the B-rep and
 * why the exact one refused), the edge legend (each kind a switch, so a
 * kind can be taken out to see what is under it, plus the off-surface
 * highlight), and the readout of the face, edge or vertex the pointer is on.
 *
 * Achromatic like every other viewport annotation, except the legend's
 * swatches: they are the key to the edge tones, and a key has to show them.
 */

import { For, Show, createMemo, type JSX } from "solid-js";
import { brepModel, edgeLegend, offSurface, type BrepModel } from "../../brep";
import {
  brepHiddenKinds,
  brepHover,
  brepMessage,
  brepOffSurfaceOnly,
  brepPayload,
  brepStatus,
  setBrepHiddenKinds,
  setBrepOffSurfaceOnly,
} from "../../state";
import { cssColor } from "../../simColors";
import { BREP_OFF_SURFACE, CHROME } from "../../tokens";

/** A length in model units, to the precision the payload carries. */
const length = (value: number | null | undefined): string => {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  return Math.abs(value) < 1e-3 ? value.toExponential(1) : value.toFixed(4);
};

function Row(props: { label: string; children: JSX.Element; testId?: string }) {
  return (
    <>
      <dt>{props.label}</dt>
      <dd data-testid={props.testId}>{props.children}</dd>
    </>
  );
}

/** `{plane: 4, blend: 2}` as "4 plane · 2 blend", largest first. */
const breakdown = (counts: Record<string, number>): string =>
  Object.entries(counts)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([name, count]) => `${count} ${name}`)
    .join(" · ");

function Summary(props: { model: BrepModel }) {
  const d = () => props.model.diagnostics;
  return (
    <>
      <Row label="Path" testId="brep-path">
        {d().exact ? "Exact · analytic" : `Fallback · ${d().source.replace("_", " ")}`}
      </Row>
      <Show when={!d().exact && d().fallback_reason}>
        <Row label="Why" testId="brep-fallback-reason">
          <span title={d().fallback_reason ?? undefined}>{d().fallback_reason}</span>
        </Row>
      </Show>
      <Row label="Faces" testId="brep-faces">
        {d().faces} <small>{breakdown(d().surfaces)}</small>
      </Row>
      <Row label="Edges" testId="brep-edges">
        {d().edges}
      </Row>
      <Row label="Vertices" testId="brep-vertices">
        {d().vertices}
      </Row>
      <Show when={d().feature_curves}>
        <Row label="Curves">{d().feature_curves}</Row>
      </Show>
    </>
  );
}

function Legend(props: { model: BrepModel }) {
  const entries = createMemo(() => edgeLegend(props.model));
  const toggle = (kind: string, on: boolean) => {
    const next = new Set(brepHiddenKinds());
    if (on) next.delete(kind);
    else next.add(kind);
    setBrepHiddenKinds(next);
  };
  const d = () => props.model.diagnostics;
  return (
    <div class="brep-legend" data-testid="brep-legend">
      <For each={entries()}>
        {(entry) => (
          <label class="brep-legend-row" title={`${entry.count} ${entry.kind} edges`}>
            <input
              type="checkbox"
              checked={!brepHiddenKinds().has(entry.kind)}
              onChange={(event) => toggle(entry.kind, event.currentTarget.checked)}
              data-testid={`brep-kind-${entry.kind}`}
            />
            <span class="brep-swatch" style={{ background: cssColor(entry.rgb) }} />
            <span>{entry.label}</span>
            <small>{entry.count}</small>
          </label>
        )}
      </For>
      <label
        class="brep-legend-row brep-off-surface"
        title={`Edges whose points lie more than ${length(d().off_surface_tolerance)} off the solid (|f| of the scene): a curve under a smooth blend, drawn where the blend has rounded the corner away.`}
      >
        <input
          type="checkbox"
          checked={brepOffSurfaceOnly()}
          onChange={(event) => setBrepOffSurfaceOnly(event.currentTarget.checked)}
          data-testid="brep-off-surface"
        />
        <span
          class="brep-swatch"
          style={{ background: CHROME[BREP_OFF_SURFACE] }}
        />
        <span>Highlight off-surface</span>
        <small data-testid="brep-off-surface-count">{d().off_surface_edges}</small>
      </label>
    </div>
  );
}

function Readout(props: { model: BrepModel }) {
  const target = createMemo(() => {
    const hover = brepHover();
    if (!hover) return null;
    if (hover.kind === "face") {
      const face = props.model.faces.find((item) => item.id === hover.id);
      return face ? ({ kind: "face", face } as const) : null;
    }
    if (hover.kind === "edge") {
      const edge = props.model.edges.find((item) => item.id === hover.id);
      return edge ? ({ kind: "edge", edge } as const) : null;
    }
    const vertex = props.model.vertices.find((item) => item.id === hover.id);
    return vertex ? ({ kind: "vertex", vertex } as const) : null;
  });
  return (
    <Show
      when={target()}
      fallback={
        <Row label="Pointer">
          <small>Hover a face, edge or vertex</small>
        </Row>
      }
    >
      {(item) => {
        const value = item();
        if (value.kind === "face") {
          const face = value.face;
          return (
            <>
              <Row label="Face" testId="brep-readout">
                #{face.id} · {face.surface}
                <Show when={face.surface === "faceted" && face.kind !== "freeform"}>
                  <small> ({face.kind})</small>
                </Show>
              </Row>
              <Row label="Patch">
                {face.patch < 0 ? "none (blend)" : `#${face.patch}`}
                <small>
                  {" "}
                  · leaf #{face.leaf}
                  {face.leaf_type ? ` ${face.leaf_type}` : ""}
                </small>
              </Row>
            </>
          );
        }
        if (value.kind === "edge") {
          const edge = value.edge;
          return (
            <>
              <Row label="Edge" testId="brep-readout">
                {edge.id < 0 ? "curve" : `#${edge.id}`} · {edge.kind}
                <Show when={edge.state !== edge.kind}>
                  <small> ({edge.state})</small>
                </Show>
              </Row>
              <Row label="Faces">
                {edge.faces.map((face) => (face < 0 ? "—" : `#${face}`)).join(" | ")}
                <small> · patches {edge.patches.map((p) => (p < 0 ? "blend" : `#${p}`)).join(" | ")}</small>
              </Row>
              <Row label="Off solid">
                <span classList={{ "brep-alert": offSurface(edge, props.model) }}>
                  {length(edge.off_surface)}
                </span>
                <small> · residual {length(edge.residual)}</small>
              </Row>
            </>
          );
        }
        const vertex = value.vertex;
        return (
          <>
            <Row label="Vertex" testId="brep-readout">
              #{vertex.id} · {vertex.kind}
            </Row>
            <Row label="At">{vertex.point.map((c) => c.toFixed(4)).join(", ")}</Row>
            <Row label="Faces">{vertex.faces.map((face) => `#${face}`).join(" ")}</Row>
          </>
        );
      }}
    </Show>
  );
}

export function BrepInspector(props: { show: boolean }) {
  const model = createMemo(() => brepModel(brepPayload()));
  return (
    <Show when={props.show}>
      <div class="brep-inspector" data-testid="brep-inspector">
        <dl class="title-block brep-card">
          <Row label="B-rep" testId="brep-status">
            {brepStatus() === "loading"
              ? "Deriving…"
              : brepStatus() === "error"
                ? "Extraction failed"
                : brepStatus() === "unavailable"
                  ? "Not installed"
                  : model()
                    ? "Derived"
                    : "Waiting for a compile"}
          </Row>
          <Show when={brepMessage()}>
            <Row label="Note" testId="brep-message">
              <span title={brepMessage() ?? undefined}>{brepMessage()}</span>
            </Row>
          </Show>
          <Show when={model()}>
            {(m) => (
              <>
                <Summary model={m()} />
                <Readout model={m()} />
              </>
            )}
          </Show>
        </dl>
        <Show when={model()}>{(m) => <Legend model={m()} />}</Show>
      </div>
    </Show>
  );
}
