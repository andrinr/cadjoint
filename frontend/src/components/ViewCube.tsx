/**
 * The navigation cube: a chamfered cube that is also the whole view rose.
 *
 * Twenty-six click targets, one per standard view — six faces, twelve bevels
 * (the 45° edge views) and eight corner triangles (the isometric octants) —
 * drawn as projected SVG polygons rather than as CSS 3D planes, so a corner is
 * clickable exactly where the corner is drawn and the whole solid is in the
 * orthographic projection the viewport itself defaults to. The geometry and
 * the projection live in `viewer/navCube.ts`; what is here is the drawing and
 * the interaction. This follows FreeCAD's navigation cube, which is where the
 * idiom comes from.
 *
 * Around it, one square:
 *   · four quarter-turn controls on the flanks, each a quarter turn about one
 *     of the camera's own screen axes — up from Front reaches Top, and again
 *     reaches Back. Each is drawn as a quarter arc ending in an arrowhead: the
 *     arc *is* ninety degrees of turn, and the head says which way. FreeCAD
 *     draws plain triangles here, and a triangle reads as a nudge;
 *   · an axis triad at the lower left, in red/green/blue. That is the one
 *     place saturated hue is allowed in this chrome, and it is allowed because
 *     it is not chrome: X-red, Y-green, Z-blue is a reading of orientation
 *     that every CAD user already has, and drawing it achromatic would make it
 *     three identical grey lines;
 *   · the projection toggle at the lower right, the corner opposite the triad.
 *
 * Two things FreeCAD's cube has that this one does not, both deliberately.
 * There is no dot opening a view menu, because there is no view menu. And
 * there are no roll arrows, because this camera has no roll: its up vector is
 * pinned to world +Z (see `cameraBasis` in `viewer/math.ts`), so a roll
 * control would be a button that either does nothing or lies about what the
 * viewport is doing. A control with nothing behind it is worse than a gap.
 *
 * ── Direction is not projection ──────────────────────────────────────────
 * Nothing on the cube changes the projection. *Isometric* names a direction, a
 * 1:1:1 line through the scene; *orthographic* names a projection, parallel
 * rays. The corner facets choose the first; the glyph in the corner chooses
 * the second. There is no ISO button, because the eight corners are the isometric
 * directions, stated eight ways instead of one.
 */

import { For, Show, onCleanup, onMount } from "solid-js";
import {
  FACETS,
  TRIAD,
  cameraBasisFor,
  dot,
  facetVisible,
  frontFacet,
  project,
  stepView,
  type Basis,
  type Facet,
  type TurnAxis,
} from "../viewer/navCube";
import { VIEW_PRESETS, sameView } from "../viewer/display";
import type { Projection } from "../viewer/math";
import { VIEW_KEYS } from "../shortcuts";
import { PITCH_LIMIT } from "./viewer/camera";
import { OrthographicIcon, PerspectiveIcon } from "./icons";

/** SVG units per cube unit. The cube's half-edge is 1. */
const SCALE = 34;
/** Half the viewBox. The body diagonal is √3 ≈ 1.73 cube units. */
const EXTENT = 92;
/**
 * How far from the centre the four quarter-turn controls sit.
 *
 * Close enough to read as part of the widget rather than as four marks
 * floating around it. The cube reaches `SCALE` (34) at a face and about 59
 * at a corner, but the controls sit on the flanks, where the silhouette is
 * the face and its chamfer — so 66 clears the cube with room to spare while
 * a control's hit square (half-side `TURN_HIT`) still stops short of it.
 */
const TURN_RADIUS = 66;
/** Radius of the quarter arc each turn control is drawn with. */
const TURN_ARC = 12;
/** Length of each arm of the arrowhead, measured along an axis. */
const TURN_HEAD = 5;
/** Half the side of a turn control's (invisible) hit square. */
const TURN_HIT = 15;

/** Pointer travel, in pixels, below which the gesture counts as a click. */
const CLICK_SLOP = 4;
const ORBIT_SPEED = 0.011;

const path = (points: [number, number][]): string =>
  points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");

/**
 * The four quarter-turn controls, one per flank.
 *
 * Each turns the camera a quarter turn toward the side of the screen it sits
 * on, from any standpoint — `stepView` in `navCube.ts` has the rule and the
 * argument for what happens at the poles. None is ever disabled: a greyed
 * control was read as a broken one.
 *
 * The glyph is drawn once, for the top flank, and the other three are that
 * drawing placed by the transform here: down is up mirrored through the
 * horizontal, right is up turned a quarter clockwise, left is right mirrored
 * through the vertical. Mirrored pairs rather than four rotated copies,
 * because four arcs all sweeping the same way around the cube read as a roll
 * ring, and this camera has no roll. As pairs, each glyph says "turn toward
 * this side" and its opposite says the reverse.
 */
const TURNS = [
  {
    id: "up",
    axis: "elevation" as const,
    turns: 1 as const,
    transform: `translate(0 ${-TURN_RADIUS})`,
    label: "Turn up",
  },
  {
    id: "right",
    axis: "azimuth" as const,
    turns: 1 as const,
    transform: `translate(${TURN_RADIUS} 0) rotate(90)`,
    label: "Turn right",
  },
  {
    id: "down",
    axis: "elevation" as const,
    turns: -1 as const,
    transform: `translate(0 ${TURN_RADIUS}) scale(1 -1)`,
    label: "Turn down",
  },
  {
    id: "left",
    axis: "azimuth" as const,
    turns: -1 as const,
    transform: `translate(${-TURN_RADIUS} 0) scale(-1 1) rotate(90)`,
    label: "Turn left",
  },
];

/**
 * The turn glyph, drawn for the top flank: a quarter arc that comes in from
 * the left, sweeps up through ninety degrees and ends in an open arrowhead
 * pointing up — the direction the control turns toward. The arc runs from
 * the bottom of its circle to the right of it, and the circle's centre is
 * placed so the whole mark sits centred on the flank.
 */
const TURN_GLYPH = (() => {
  const tip: [number, number] = [3, -TURN_ARC / 2];
  const centre: [number, number] = [tip[0] - TURN_ARC, tip[1]];
  const tail: [number, number] = [centre[0], centre[1] + TURN_ARC];
  return [
    `M ${tail[0]} ${tail[1]} A ${TURN_ARC} ${TURN_ARC} 0 0 0 ${tip[0]} ${tip[1]}`,
    `M ${tip[0] - TURN_HEAD} ${tip[1] + TURN_HEAD} L ${tip[0]} ${tip[1]} L ${tip[0] + TURN_HEAD} ${tip[1] + TURN_HEAD}`,
  ].join(" ");
})();

/** True while focus is in the code editor or another text surface. */
function isTypingTarget(): boolean {
  const target = document.activeElement;
  return Boolean(
    target &&
      (target.tagName === "TEXTAREA" ||
        target.tagName === "INPUT" ||
        target.closest(".cm-editor")),
  );
}

const titleCase = (key: string) =>
  key
    .split("-")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join("-");

/** What a facet's tooltip says, including the far side it also offers. */
function facetTitle(facet: Facet): string {
  const name =
    facet.rank === "face"
      ? `${titleCase(facet.key)} view`
      : facet.rank === "edge"
        ? `${titleCase(facet.key)} — 45° edge view`
        : `Isometric — ${facet.key.split("-").join(", ")}`;
  return `${name}  ·  Shift-click for ${titleCase(facet.opposite)}`;
}

export interface ViewCubeProps {
  yaw: number;
  pitch: number;
  projection: Projection;
  onPreset: (key: string) => void;
  onProjection: (projection: Projection) => void;
  /** Drag the cube to orbit, as in other 3D viewports. */
  onOrbit: (yaw: number, pitch: number) => void;
}

export function ViewCube(props: ViewCubeProps) {
  let dragging = false;
  let travelled = 0;
  let lastX = 0;
  let lastY = 0;

  const basis = (): Basis => cameraBasisFor(props.yaw, props.pitch);

  /**
   * The visible facets, furthest first.
   *
   * The solid is convex and back facets are culled, so the survivors never
   * overlap; the order only decides which hairline is drawn over which.
   */
  const visible = () => {
    const view = basis();
    return FACETS.filter((facet) => facetVisible(facet, view)).sort(
      (a, b) => dot(a.normal, view.direction) - dot(b.normal, view.direction),
    );
  };

  const outline = (facet: Facet) =>
    path(
      facet.polygon.map((point) => {
        const [x, y] = project(point, basis());
        return [x * SCALE, y * SCALE] as [number, number];
      }),
    );

  /**
   * Which preset the camera is standing on, decided by the angles.
   *
   * Derived rather than remembered, for the same reason the graticule's VIEW
   * field is: after an orbit the last facet pressed is no longer where the
   * camera is, and a widget that is also a readout may not lie about that.
   */
  const isActive = (key: string) => sameView(props.yaw, props.pitch, VIEW_PRESETS[key]);

  const front = () => frontFacet(basis());

  /** The label's own basis, projected — this is what foreshortens the type. */
  const labelTransform = (facet: Facet) => {
    const view = basis();
    const [cx, cy] = project(facet.centre, view);
    const [rx, ry] = project(facet.labelRight, view);
    const [ux, uy] = project(facet.labelUp, view);
    // SVG's local +y runs down the page while `labelUp` runs up the world.
    const m = [rx * SCALE, ry * SCALE, -ux * SCALE, -uy * SCALE].map((value) =>
      value.toFixed(3),
    );
    return `matrix(${m.join(" ")} ${(cx * SCALE).toFixed(2)} ${(cy * SCALE).toFixed(2)})`;
  };

  const onPointerDown = (event: PointerEvent) => {
    dragging = true;
    travelled = 0;
    lastX = event.clientX;
    lastY = event.clientY;
  };

  const onPointerMove = (event: PointerEvent) => {
    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    travelled += Math.abs(dx) + Math.abs(dy);
    lastX = event.clientX;
    lastY = event.clientY;
    // Capture only once this is really a drag: capturing on pointerdown would
    // retarget the click and a facet would never receive it.
    if (travelled > CLICK_SLOP) {
      const element = event.currentTarget as unknown as HTMLElement;
      if (!element.hasPointerCapture(event.pointerId)) {
        element.setPointerCapture(event.pointerId);
      }
    }
    props.onOrbit(
      props.yaw - dx * ORBIT_SPEED,
      Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, props.pitch + dy * ORBIT_SPEED)),
    );
  };

  const onPointerUp = (event: PointerEvent) => {
    dragging = false;
    const element = event.currentTarget as unknown as HTMLElement;
    if (element.hasPointerCapture(event.pointerId)) {
      element.releasePointerCapture(event.pointerId);
    }
  };

  /**
   * A facet click only snaps the view when the pointer barely moved.
   *
   * A held modifier takes the far side. Twenty of the twenty-six facets are
   * culled at any moment and a facet you cannot see is a view you cannot ask
   * for — from anywhere above the floor there is simply no BOTTOM to press —
   * so the antipode is one press away instead of an orbit away.
   *
   * The modifier is **Shift**, and not Ctrl, which is what the same meaning is
   * bound to on the view keys (Ctrl+7 is Blender's Bottom). macOS is why: the
   * system claims Control-click as the secondary click, and the browser
   * delivers a `contextmenu` where the page expected a `click` — measured, not
   * assumed. A modifier that works on the keyboard and silently does nothing
   * under the pointer on a third of the machines running this is not a
   * modifier. Ctrl is accepted as well, so the habit costs nothing on the
   * platforms where it does arrive.
   */
  const choose = (facet: Facet, event: MouseEvent) => {
    if (travelled > CLICK_SLOP) return;
    props.onPreset(event.shiftKey || event.ctrlKey ? facet.opposite : facet.key);
  };

  const toggleProjection = () =>
    props.onProjection(
      props.projection === "orthographic" ? "perspective" : "orthographic",
    );

  const turn = (axis: TurnAxis, turns: 1 | -1) => {
    if (travelled > CLICK_SLOP) return;
    const next = stepView(props.yaw, props.pitch, axis, turns);
    props.onOrbit(next.yaw, next.pitch);
  };

  /**
   * The view keys, claimed at the window.
   *
   * They live with the widget rather than with the viewport's other shortcuts
   * because this is what they operate: the same two callbacks the facets call.
   * Modifier discipline matches the rest of the app — Meta and Alt are the
   * system's and the browser's, and anything typed into a text surface is the
   * text surface's.
   */
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.metaKey || event.altKey || isTypingTarget()) return;
    const binding = VIEW_KEYS[event.code];
    if (!binding) return;
    if (binding.projection) {
      event.preventDefault();
      toggleProjection();
      return;
    }
    if (binding.reverse) {
      event.preventDefault();
      // The other side of the same line: the octant mirrored through the
      // orbit target, which is what Blender's 9 does.
      props.onOrbit(props.yaw + Math.PI, -props.pitch);
      return;
    }
    const key = event.ctrlKey ? binding.opposite : binding.preset;
    if (!key) return;
    event.preventDefault();
    props.onPreset(key);
  };

  onMount(() => {
    window.addEventListener("keydown", onKeyDown);
    onCleanup(() => window.removeEventListener("keydown", onKeyDown));
  });

  return (
    <div class="view-cube" data-testid="view-cube">
      <svg
        class="cube-stage"
        viewBox={`${-EXTENT} ${-EXTENT} ${EXTENT * 2} ${EXTENT * 2}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        role="group"
        aria-label="View cube"
      >
        <title>Drag to orbit. Click a face, bevel or corner to snap.</title>

        {/* The quarter-turn controls, outside the cube on the four flanks.
            The glyph is a hairline, so an invisible square under it is what
            takes the press. */}
        <For each={TURNS}>
          {(control) => (
            <g
              class="cube-turn"
              transform={control.transform}
              onClick={() => turn(control.axis, control.turns)}
              data-testid={`cube-turn-${control.id}`}
            >
              <title>{control.label} — a quarter turn</title>
              <rect
                class="cube-turn-hit"
                x={-TURN_HIT}
                y={-TURN_HIT}
                width={TURN_HIT * 2}
                height={TURN_HIT * 2}
              />
              <path class="cube-turn-glyph" d={TURN_GLYPH} />
            </g>
          )}
        </For>

        {/* The solid itself, back facets already culled. */}
        <For each={visible()}>
          {(facet) => (
            <polygon
              class={`cube-facet ${facet.rank}`}
              classList={{
                active: isActive(facet.key),
                front: front()?.key === facet.key,
              }}
              points={outline(facet)}
              onClick={(event) => choose(facet, event)}
              data-testid={`view-${facet.key}`}
              data-front={front()?.key === facet.key ? "true" : "false"}
            >
              <title>{facetTitle(facet)}</title>
            </polygon>
          )}
        </For>

        {/* Face names, foreshortened onto their own facets. */}
        <For each={visible().filter((facet) => facet.rank === "face")}>
          {(facet) => (
            <text
              class="cube-label"
              classList={{ active: isActive(facet.key) }}
              transform={labelTransform(facet)}
              font-size="0.26"
              text-anchor="middle"
              dominant-baseline="central"
              data-testid={`cube-label-${facet.key}`}
            >
              {facet.label}
            </text>
          )}
        </For>

        {/* The world axes, at the lower left, turning with the cube. */}
        <g class="cube-triad" transform={`translate(${-EXTENT + 22} ${EXTENT - 22})`}>
          <For each={TRIAD}>
            {(entry) => {
              const tip = (): [number, number] => {
                const [x, y] = project(entry.axis, basis());
                return [x * 24, y * 24];
              };
              return (
                <>
                  <line
                    class={`cube-axis ${entry.token}`}
                    x1="0"
                    y1="0"
                    x2={tip()[0].toFixed(2)}
                    y2={tip()[1].toFixed(2)}
                  />
                  <text
                    class={`cube-axis-label ${entry.token}`}
                    x={(tip()[0] * 1.32).toFixed(2)}
                    y={(tip()[1] * 1.32).toFixed(2)}
                    text-anchor="middle"
                    dominant-baseline="central"
                  >
                    {entry.label}
                  </text>
                </>
              );
            }}
          </For>
        </g>
      </svg>

      {/* The projection toggle: a glyph, not a facet, because it is not a
          direction. It sits in the stage's lower-right corner — the flanks
          are the turn controls, the corners are free, the triad has the
          lower-left — so the two camera readouts bracket the down control
          and the widget stays one square, rather than a square with a button
          dangling under it, which is where this used to be and where it read
          as orphaned. FreeCAD puts its own cube glyph in the same corner. */}
      <button
        type="button"
        class="cube-projection"
        classList={{ active: props.projection === "orthographic" }}
        onClick={toggleProjection}
        title={
          props.projection === "orthographic"
            ? "Orthographic — click for perspective  (5)"
            : "Perspective — click for orthographic  (5)"
        }
        aria-label="Toggle projection"
        data-testid="projection-toggle"
      >
        <Show when={props.projection === "orthographic"} fallback={<PerspectiveIcon />}>
          <OrthographicIcon />
        </Show>
      </button>
    </div>
  );
}
