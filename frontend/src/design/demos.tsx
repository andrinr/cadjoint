/**
 * The functional rules, as things to try.
 *
 * A rule about time, keys or commitment cannot be shown by a still picture,
 * so each one here is a small working model of the behaviour the playground
 * owes. They share nothing with the app's state: they are specimens, built
 * from the same primitives the guide defines.
 */

import { For, Show, createSignal, onCleanup, type JSX } from "solid-js";
import { Select } from "../components/ui/Select";
import { Cross } from "./kit";

// ── A2 / A9: a panel edits a literal, and the edit propagates ─────────────

/**
 * A panel field and the source line it writes. Committing sequences the
 * propagation: the literal lights first, the geometry follows 260ms later.
 */
export function LiteralDemo() {
  const [committed, setCommitted] = createSignal(0.85);
  const [draft, setDraft] = createSignal("0.85");
  const [lit, setLit] = createSignal(false);
  const [geometry, setGeometry] = createSignal(0.85);
  const invalid = () => !Number.isFinite(Number(draft())) || draft().trim() === "";
  let timers: number[] = [];
  onCleanup(() => timers.forEach(clearTimeout));
  const commit = () => {
    if (invalid()) return;
    const value = Number(draft());
    if (value === committed()) return;
    timers.forEach(clearTimeout);
    setCommitted(value);
    setLit(true);
    timers = [
      window.setTimeout(() => setGeometry(value), 260),
      window.setTimeout(() => setLit(false), 900),
    ];
  };
  return (
    <div class="dl-stack">
      <div class="dl-field">
        <label for="dl-fin-depth">fin depth</label>
        <input
          id="dl-fin-depth"
          class={invalid() ? "dl-input invalid" : "dl-input"}
          value={draft()}
          onInput={(event) => setDraft(event.currentTarget.value)}
          onKeyDown={(event) => event.key === "Enter" && commit()}
          onBlur={commit}
        />
        <small>m</small>
      </div>
      <pre class="dl-code">
        <div>
          <span>120</span>
          <span>
            fin1_tip_r = Vector2(value=[0.75, <span class={lit() ? "lit lit-on" : "lit"}>{committed()}</span>],{" "}
            <span class="free">free=True</span>)
          </span>
        </div>
      </pre>
      <div class="dl-rows">
        <div>
          <span>geometry · fin height</span>
          <b>{geometry().toFixed(2)}</b>
          <small>m</small>
        </div>
      </div>
      <div class="dl-bar" style={{ width: `${Math.min(100, Math.max(0, geometry() * 100))}%` }} />
    </div>
  );
}

// ── A10: every action answers ─────────────────────────────────────────────

const STAGES = ["COMPILE", "MESH", "ASSEMBLE", "SOLVE", "ADJOINT"] as const;
type JobState = "idle" | "running" | "done" | "cancelled" | "failed";

/** A solve as an annunciator strip: every stage allocated, a stop control. */
export function JobDemo(props: { fail?: boolean }) {
  const [state, setState] = createSignal<JobState>("idle");
  const [stage, setStage] = createSignal(-1);
  const [residual, setResidual] = createSignal<number | null>(null);
  let timer: number | undefined;
  onCleanup(() => clearInterval(timer));
  const run = () => {
    clearInterval(timer);
    setState("running");
    setStage(0);
    setResidual(null);
    let tick = 0;
    timer = window.setInterval(() => {
      tick += 1;
      const next = Math.floor(tick / 4);
      if (props.fail && next === 3) {
        clearInterval(timer);
        setState("failed");
        return;
      }
      if (next >= STAGES.length) {
        clearInterval(timer);
        setState("done");
        setStage(STAGES.length);
        return;
      }
      setStage(next);
      if (next === 3) setResidual(10 ** -(tick % 4 + 2) * 3.1);
    }, 220);
  };
  const stop = () => {
    clearInterval(timer);
    setState("cancelled");
  };
  const cellClass = (index: number) => {
    if (state() === "failed" && index === 3) return "failed";
    if (state() === "running" && index === stage()) return "current";
    if (index < stage() || state() === "done") return "done";
    return "";
  };
  const line = () => {
    switch (state()) {
      case "idle":
        return "ready";
      case "running":
        return residual() === null ? STAGES[stage()].toLowerCase() : `residual ${residual()!.toExponential(1)}`;
      case "done":
        return "done · 2.6 s";
      case "cancelled":
        return `cancelled at ${STAGES[stage()].toLowerCase()}`;
      case "failed":
        return "failed at solve";
    }
  };
  return (
    <div class="dl-stack">
      <div class="dl-stages">
        <For each={STAGES}>{(name, index) => <span class={cellClass(index())}>{name}</span>}</For>
        <output>{line()}</output>
        <Show
          when={state() === "running"}
          fallback={
            <button type="button" class="dl-btn primary" onClick={run}>
              {state() === "idle" ? "Run study" : "Run again"}
            </button>
          }
        >
          <button type="button" class="dl-btn" onClick={stop}>
            Stop
          </button>
        </Show>
      </div>
      <Show when={state() === "failed"}>
        <div class="dl-notice error">
          <b>Error</b>
          <p>
            Solve diverged: <code>sink-conduction</code> has no fixed temperature, so the
            system is singular. Add a <code>Dirichlet</code> condition on any face.
          </p>
          <span />
          <details>
            <summary>Traceback</summary>
            <pre>LinAlgError: Singular matrix (cadjoint/fem/solve.py:212)</pre>
          </details>
        </div>
      </Show>
    </div>
  );
}

// ── A11: Escape cancels one thing per press ──────────────────────────────

const RUNGS = [
  { name: "gesture in flight", detail: "drag on a handle" },
  { name: "half-finished command", detail: "distance: pick 2nd point" },
  { name: "armed tool", detail: "polygon" },
  { name: "selection", detail: "P9" },
  { name: "mode", detail: "sketch" },
] as const;

/** The escape ladder: build up state, then walk back out one press at a time. */
export function EscapeDemo() {
  const [held, setHeld] = createSignal<boolean[]>([true, true, true, true, true]);
  const [last, setLast] = createSignal("Focus this plate and press Esc");
  const top = () => held().findIndex((value) => value);
  const escape = () => {
    const index = top();
    if (index < 0) {
      setLast("Nothing to cancel: Esc does nothing");
      return;
    }
    setHeld(held().map((value, i) => (i === index ? false : value)));
    setLast(index === 4 ? "Cancelled the mode: back to Model" : `Cancelled the ${RUNGS[index].name}`);
  };
  return (
    <div
      class="dl-stack"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          escape();
        }
      }}
    >
      <ol class="dl-ladder">
        <For each={RUNGS}>
          {(rung, index) => (
            <li classList={{ held: held()[index()], next: index() === top() }}>
              <span>{index() + 1}</span>
              <span>{rung.name}</span>
              <code>{held()[index()] ? rung.detail : "—"}</code>
            </li>
          )}
        </For>
      </ol>
      <div class="dl-row-of">
        <button type="button" class="dl-btn h-sm" onClick={escape}>
          Esc
        </button>
        <button type="button" class="dl-btn quiet h-sm" onClick={() => setHeld([true, true, true, true, true])}>
          Reset
        </button>
        <span class="dl-status">{last()}</span>
      </div>
    </div>
  );
}

// ── Direct manipulation: preview, commit on release, Esc abandons ────────

/** A detented machinist's rule: dragging previews, release writes the source. */
export function DragDemo() {
  const MIN = 0;
  const MAX = 1;
  const DETENT = 0.05;
  const [source, setSource] = createSignal(0.45);
  const [preview, setPreview] = createSignal<number | null>(null);
  const [writes, setWrites] = createSignal(0);
  let track: HTMLDivElement | undefined;
  const at = (event: PointerEvent) => {
    const box = track!.getBoundingClientRect();
    const t = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    return Math.round((MIN + t * (MAX - MIN)) / DETENT) * DETENT;
  };
  const shown = () => preview() ?? source();
  const ticks = Array.from({ length: 21 }, (_, i) => i);
  const abandon = () => setPreview(null);
  return (
    <div
      class="dl-stack"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Escape" && preview() !== null) {
          event.preventDefault();
          abandon();
        }
      }}
    >
      <div
        ref={track}
        class={preview() === null ? "dl-track" : "dl-track dragging"}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          event.currentTarget.parentElement?.focus();
          setPreview(at(event));
        }}
        onPointerMove={(event) => preview() !== null && setPreview(at(event))}
        onPointerUp={() => {
          const value = preview();
          if (value === null) return;
          setPreview(null);
          if (value !== source()) {
            setSource(Number(value.toFixed(2)));
            setWrites(writes() + 1);
          }
        }}
      >
        <For each={ticks}>
          {(tick) => <i classList={{ major: tick % 5 === 0 }} style={{ left: `${tick * 5}%` }} />}
        </For>
        <Show when={preview() !== null}>
          <b class="origin" style={{ left: `${source() * 100}%` }} />
        </Show>
        <b style={{ left: `${shown() * 100}%` }} />
      </div>
      <div class="dl-rows">
        <div>
          <span>{preview() === null ? "value" : "preview · release to write · Esc to abandon"}</span>
          <b>{shown().toFixed(2)}</b>
          <small>m</small>
        </div>
      </div>
      <pre class="dl-code">
        <div>
          <span>58</span>
          <span>
            bushing_spacing = Scalar(value={source().toFixed(2)}, <span class="free">free=True</span>)
          </span>
        </div>
      </pre>
      <span class="dl-status">
        source writes: {writes()} · undo steps: {writes()}
      </span>
    </div>
  );
}

// ── Inputs: commit on Enter or blur, only when valid ─────────────────────

export function InputDemo() {
  const [value, setValue] = createSignal(0.75);
  const [draft, setDraft] = createSignal("0.75");
  const invalid = () => draft().trim() === "" || !Number.isFinite(Number(draft()));
  const commit = () => {
    if (!invalid()) setValue(Number(draft()));
  };
  return (
    <div class="dl-stack">
      <div class="dl-field">
        <label for="dl-width">base width</label>
        <input
          id="dl-width"
          class={invalid() ? "dl-input invalid" : "dl-input"}
          value={draft()}
          onInput={(event) => setDraft(event.currentTarget.value)}
          onKeyDown={(event) => event.key === "Enter" && commit()}
          onBlur={commit}
        />
        <small>m</small>
      </div>
      <div class="dl-field">
        <label>tip width</label>
        <input class="dl-input" readonly value="0.5 * base_w" title="Written as an expression: edit it in the source" />
        <small>m</small>
      </div>
      <div class="dl-field">
        <label>fillet</label>
        <input class="dl-input" disabled value="0.02" title="Fillets are fixed while a loft depends on this edge" />
        <small>m</small>
      </div>
      <span class="dl-status">
        committed: {value()} {invalid() ? "· invalid entry kept as typed, nothing written" : ""}
      </span>
    </div>
  );
}

// ── Undo replaces confirmation ────────────────────────────────────────────

const CONSTRAINTS = ["fix · P1", "distance · P1–P2 = 1.8", "horiz · P1–P2", "vert · P2–P3", "equal · P1P15–P2P4"];

export function UndoDemo() {
  const [items, setItems] = createSignal<string[]>([...CONSTRAINTS]);
  const [removed, setRemoved] = createSignal<{ item: string; index: number } | null>(null);
  const remove = (index: number) => {
    const list = items();
    setRemoved({ item: list[index], index });
    setItems(list.filter((_, i) => i !== index));
  };
  const undo = () => {
    const last = removed();
    if (!last) return;
    const list = [...items()];
    list.splice(last.index, 0, last.item);
    setItems(list);
    setRemoved(null);
  };
  return (
    <div class="dl-stack">
      <div class="dl-row-of">
        <For each={items()}>
          {(item, index) => (
            <span class="dl-chip">
              {item}
              <button type="button" title={`Remove ${item}`} onClick={() => remove(index())}>
                <Cross />
              </button>
            </span>
          )}
        </For>
      </div>
      <Show when={removed()}>
        {(last) => (
          <div class="dl-row-of">
            <span class="dl-status">removed {last().item}</span>
            <button type="button" class="dl-btn h-sm" onClick={undo}>
              Undo <kbd>⌘Z</kbd>
            </button>
          </div>
        )}
      </Show>
    </div>
  );
}

// ── One selection ─────────────────────────────────────────────────────────

const OBJECTS = [
  { name: "fin comb", kind: "sketch", line: 132 },
  { name: "slug section", kind: "sketch", line: 203 },
  { name: "bush_a", kind: "cylinder", line: 211 },
  { name: "bush_b", kind: "cylinder", line: 214 },
];

/** Tree, viewport and editor showing one selection; select in any of them. */
export function SelectionDemo() {
  const [selected, setSelected] = createSignal(0);
  return (
    <div class="dl-examples">
      <div class="dl-panel">
        <nav>
          <button type="button" class="active">Objects</button>
        </nav>
        <div>
          <div class="dl-menu flush">
            <For each={OBJECTS}>
              {(object, index) => (
                <button
                  type="button"
                  class={selected() === index() ? "selected" : ""}
                  onClick={() => setSelected(index())}
                >
                  <span>{object.kind === "sketch" ? "◇" : "○"}</span>
                  <span>{object.name}</span>
                  <kbd>line {object.line}</kbd>
                </button>
              )}
            </For>
          </div>
        </div>
      </div>
      <div class="dl-stack">
        <div class="dl-viewport">
          <svg viewBox="0 0 320 120" role="img" aria-label="Selection in the viewport">
            <For each={OBJECTS}>
              {(_, index) => (
                <rect
                  x={16 + index() * 76}
                  y="30"
                  width="60"
                  height="60"
                  fill={selected() === index() ? "var(--ink)" : "var(--ink)"}
                  fill-opacity={selected() === index() ? "0.14" : "0.05"}
                  stroke={selected() === index() ? "var(--ink)" : "var(--ink-3)"}
                  stroke-width={selected() === index() ? 2 : 1}
                  style={{ cursor: "pointer" }}
                  onClick={() => setSelected(index())}
                />
              )}
            </For>
          </svg>
        </div>
        <pre class="dl-code">
          <For each={OBJECTS}>
            {(object, index) => (
              <div>
                <span>{object.line}</span>
                <span>
                  <span class={selected() === index() ? "lit lit-on" : "lit"}>{object.name.replace(" ", "_")}</span> = …
                </span>
              </div>
            )}
          </For>
        </pre>
      </div>
    </div>
  );
}

// ── A4: never by hue alone ────────────────────────────────────────────────

export function GreyscaleDemo(props: { children: JSX.Element }) {
  const [grey, setGrey] = createSignal(false);
  return (
    <div class="dl-stack">
      <div style={{ filter: grey() ? "grayscale(1)" : "none" }}>{props.children}</div>
      <div class="dl-row-of">
        <button type="button" class={grey() ? "dl-btn h-sm pressed" : "dl-btn h-sm"} onClick={() => setGrey(!grey())}>
          {grey() ? "Show colour" : "View in greyscale"}
        </button>
      </div>
    </div>
  );
}

// ── Motion: three durations, named by meaning ────────────────────────────

export function DurationDemo() {
  const [on, setOn] = createSignal([false, false, false]);
  const flip = (index: number) => setOn(on().map((value, i) => (i === index ? !value : value)));
  const rows = [
    { token: "dur-fast", ms: 90, meaning: "acknowledgement" },
    { token: "dur-base", ms: 160, meaning: "state" },
    { token: "dur-slow", ms: 260, meaning: "propagation" },
  ];
  return (
    <div class="dl-stack">
      <For each={rows}>
        {(row, index) => (
          <div class="dl-field" style={{ "grid-template-columns": "calc(var(--unit) * 36) 1fr calc(var(--unit) * 36)" }}>
            <label>
              {row.token} · {row.ms} ms
            </label>
            <div style={{ height: "calc(var(--unit) * 3)", background: "var(--surface-bar-alt)" }}>
              <div
                style={{
                  height: "100%",
                  width: on()[index()] ? "100%" : "0%",
                  background: "var(--ink)",
                  transition: `width var(--${row.token}) var(--ease)`,
                }}
              />
            </div>
            <button type="button" class="dl-btn h-sm" onClick={() => flip(index())}>
              {row.meaning}
            </button>
          </div>
        )}
      </For>
    </div>
  );
}

/** Digits replace instantly; a tweened counter shows values that never were. */
export function TweenDemo(props: { tween: boolean }) {
  const [value, setValue] = createSignal(411.79);
  const [shown, setShown] = createSignal(411.79);
  let frame = 0;
  onCleanup(() => cancelAnimationFrame(frame));
  const next = () => {
    const target = value() > 412 ? 398.21 : 412.8;
    setValue(target);
    if (!props.tween) {
      setShown(target);
      return;
    }
    const from = shown();
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / 600);
      setShown(from + (target - from) * t);
      if (t < 1) frame = requestAnimationFrame(step);
    };
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(step);
  };
  return (
    <div class="dl-stack">
      <div class="dl-rows">
        <div>
          <span>peak temperature</span>
          <b>{shown().toFixed(2)}</b>
          <small>K</small>
        </div>
      </div>
      <div class="dl-row-of">
        <button type="button" class="dl-btn h-sm" onClick={next}>
          New result
        </button>
      </div>
    </div>
  );
}

// ── Modes and tools ──────────────────────────────────────────────────────

const MODES = ["model", "sketch", "simulate"] as const;
const TOOLS: Record<(typeof MODES)[number], string[]> = {
  model: ["select", "box", "sphere", "cylinder", "move"],
  sketch: ["select", "polygon", "distance", "horizontal", "vertical"],
  simulate: ["select", "pick face", "probe"],
};
const HINTS: Record<string, string> = {
  select: "Click to select · Drag to orbit",
  box: "Click to place a box",
  sphere: "Click to place a sphere",
  cylinder: "Click to place a cylinder",
  move: "Drag the gizmo to move the selection",
  polygon: "Click to add a vertex · Enter closes",
  distance: "Pick two points to constrain their distance",
  horizontal: "Pick an edge to make it horizontal",
  vertical: "Pick an edge to make it vertical",
  "pick face": "Click faces to add them to the boundary condition",
  probe: "Hover the field to read it",
};

export function ModeDemo() {
  const [mode, setMode] = createSignal<(typeof MODES)[number]>("sketch");
  const [tool, setTool] = createSignal("polygon");
  return (
    <div class="dl-stack">
      <div class="dl-seg">
        <For each={MODES}>
          {(name) => (
            <button
              type="button"
              class={mode() === name ? "active" : ""}
              onClick={() => {
                setMode(name);
                setTool("select");
              }}
            >
              {name}
            </button>
          )}
        </For>
      </div>
      <div class="dl-row-of">
        <For each={TOOLS[mode()]}>
          {(name) => (
            <button
              type="button"
              class={tool() === name ? "dl-btn h-sm armed" : "dl-btn h-sm"}
              onClick={() => setTool(name)}
            >
              {name}
            </button>
          )}
        </For>
      </div>
      <div class="dl-row-of">
        <span class="dl-tag">{mode()}</span>
        <span class="dl-hint">{HINTS[tool()]}</span>
      </div>
    </div>
  );
}

// ── K9: the playground's own Select, live ─────────────────────────────────

export function SelectSpecimen() {
  const [material, setMaterial] = createSignal("aluminium_6061");
  return (
    <div class="dl-field">
      <label>material</label>
      <Select
        value={material()}
        options={[
          { value: "aluminium_6061" },
          { value: "copper_c11000" },
          { value: "steel_1018" },
          { value: "titanium_ti6al4v" },
          { value: "fr4" },
        ]}
        onChange={setMaterial}
      />
      <small />
    </div>
  );
}
