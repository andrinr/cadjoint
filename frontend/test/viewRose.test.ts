/**
 * The view rose: twenty-six standard views, and the one place they are named.
 *
 * Two things are held here. The *table* — `VIEW_PRESETS` — has to contain
 * every direction exactly once, at the exact angles those directions imply,
 * and it has to call all eight corners ISO because that is what an isometric
 * direction is called in any octant. And the *cube* has to agree with it: the
 * facet turned toward you must be the facet the camera is standing on, for
 * every direction and not merely for the one that was looked at while
 * building it. That second claim is the one the widget shipped wrong — its
 * CSS pitch rotation doubled and mirrored the elevation, so a camera above the
 * floor was shown the BOTTOM face — and it is invisible at pitch 0, which is
 * how it survived.
 */

import { describe, expect, it } from "vitest";
import {
  VIEW_PRESETS,
  anglesForDirection,
  matchViewPreset,
  sameView,
  viewPresetName,
} from "../src/viewer/display";
import {
  CHAMFER,
  FACETS,
  cameraBasisFor,
  cross,
  dot,
  facetVisible,
  frontFacet,
  project,
  type Vec3,
} from "../src/viewer/navCube";
import { octant } from "../src/viewer/graticule";

/** The camera offset the app builds from a yaw/pitch, in a Z-up world. */
const offsetOf = (yaw: number, pitch: number): [number, number, number] => [
  Math.cos(pitch) * Math.sin(yaw),
  -Math.cos(pitch) * Math.cos(yaw),
  Math.sin(pitch),
];

const sign = (value: number) => (value > 1e-9 ? 1 : value < -1e-9 ? -1 : 0);

describe("the preset table", () => {
  it("holds all twenty-six directions, plus the iso alias", () => {
    // 6 faces + 12 edges + 8 corners = 26, and `iso` names one of them twice.
    expect(Object.keys(VIEW_PRESETS)).toHaveLength(27);
    const directions = new Set<string>();
    for (const preset of Object.values(VIEW_PRESETS)) {
      directions.add(offsetOf(preset.yaw, preset.pitch).map(sign).join(","));
    }
    expect(directions.size).toBe(26);
  });

  it("puts every preset on the direction its name spells", () => {
    for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
      if (name === "iso") continue;
      const direction = offsetOf(preset.yaw, preset.pitch).map(sign) as [
        number,
        number,
        number,
      ];
      expect(viewPresetName(direction), name).toBe(name);
    }
  });

  it("keeps iso on the +X−Y+Z corner the session opens on", () => {
    expect(octant(VIEW_PRESETS.iso.yaw, VIEW_PRESETS.iso.pitch)).toBe("+X−Y+Z");
    expect(VIEW_PRESETS.iso).toEqual({ ...anglesForDirection(1, -1, 1), label: "ISO" });
    // A true 1:1:1 direction, not angles that merely look isometric: the
    // elevation of an isometric standpoint is atan(1/√2).
    expect(VIEW_PRESETS.iso.pitch).toBeCloseTo(Math.atan(1 / Math.SQRT2), 12);
  });

  it("gives the two polar views no azimuth to get wrong", () => {
    // `atan2(0, -0)` is π, not 0, so generating these naively yaws them half a
    // turn — which showed up as an upside-down TOP label on the cube.
    expect(VIEW_PRESETS.top.yaw).toBe(0);
    expect(VIEW_PRESETS.bottom.yaw).toBe(0);
  });

  it("matches a camera to a view, and refuses one that has orbited off", () => {
    expect(matchViewPreset(VIEW_PRESETS.front.yaw, VIEW_PRESETS.front.pitch)).toBe("front");
    expect(matchViewPreset(VIEW_PRESETS.front.yaw + 0.4, 0)).toBeNull();
    // Yaw wraps: a camera one full turn round is on the same view.
    expect(sameView(VIEW_PRESETS.right.yaw + 2 * Math.PI, 0, VIEW_PRESETS.right)).toBe(true);
    // Looking straight down, yaw is a spin about the view axis.
    expect(sameView(1.234, Math.PI / 2, VIEW_PRESETS.top)).toBe(true);
  });
});

describe("the chamfered cube's facets", () => {
  it("has one facet per standard view, in three ranks", () => {
    expect(FACETS).toHaveLength(26);
    const ranks = FACETS.map((facet) => facet.rank);
    expect(ranks.filter((rank) => rank === "face")).toHaveLength(6);
    expect(ranks.filter((rank) => rank === "edge")).toHaveLength(12);
    expect(ranks.filter((rank) => rank === "corner")).toHaveLength(8);
    expect(new Set(FACETS.map((facet) => facet.key)).size).toBe(26);
    for (const facet of FACETS) expect(VIEW_PRESETS[facet.key], facet.key).toBeDefined();
  });

  it("names each facet after the direction its own normal points", () => {
    for (const facet of FACETS) {
      const direction = facet.normal.map(sign) as [number, number, number];
      expect(viewPresetName(direction), facet.key).toBe(facet.key);
      // And the preset it snaps to is on that same direction.
      const preset = VIEW_PRESETS[facet.key];
      expect(offsetOf(preset.yaw, preset.pitch).map(sign), facet.key).toEqual([
        ...direction,
      ]);
    }
  });

  it("draws a closed, planar polygon on each facet's own plane", () => {
    for (const facet of FACETS) {
      expect(facet.polygon.length, facet.key).toBe(
        facet.rank === "corner" ? 3 : 4,
      );
      // Every vertex is the same distance along the normal — that is what
      // makes it a facet of a convex solid rather than a bent quad.
      const plane = dot(facet.polygon[0], facet.normal);
      for (const vertex of facet.polygon) {
        expect(dot(vertex, facet.normal), facet.key).toBeCloseTo(plane, 9);
      }
      expect(dot(facet.centre, facet.normal), facet.key).toBeCloseTo(plane, 9);
      // Wound about the outward normal, so a renderer that culls by winding
      // and one that culls by normal agree.
      const a = facet.polygon[1].map(
        (value, axis) => value - facet.polygon[0][axis],
      ) as unknown as Vec3;
      const b = facet.polygon[2].map(
        (value, axis) => value - facet.polygon[1][axis],
      ) as unknown as Vec3;
      expect(dot(cross(a, b), facet.normal), facet.key).toBeGreaterThan(0);
    }
  });

  it("puts the three ranks at the distances a chamfer implies", () => {
    // A cube of half-edge 1 cut back by `CHAMFER` on each half-edge: faces at
    // 1, bevels at (2 − 2k)/√2, corners at (3 − 4k)/√3. Getting the corner
    // plane wrong is not subtle — it floats the triangle off the solid.
    const at = (rank: string) =>
      FACETS.filter((facet) => facet.rank === rank).map((facet) =>
        dot(facet.centre, facet.normal),
      );
    for (const distance of at("face")) expect(distance).toBeCloseTo(1, 9);
    for (const distance of at("edge")) {
      expect(distance).toBeCloseTo((2 - 2 * CHAMFER) / Math.SQRT2, 9);
    }
    for (const distance of at("corner")) {
      expect(distance).toBeCloseTo((3 - 4 * CHAMFER) / Math.sqrt(3), 9);
    }
    // The chamfer has to actually cut something, or there are no edge and
    // corner targets to click.
    expect(CHAMFER).toBeGreaterThan(0);
    expect(CHAMFER).toBeLessThan(0.5);
  });

  it("shares every corner vertex with the three facets around it", () => {
    // The chamfer only reads as a chamfer if the solid is closed: every
    // vertex of a corner triangle has to also be a vertex of the bevel and
    // the face beside it, or there is a wedge of nothing between them.
    const key = (point: Vec3) => point.map((value) => value.toFixed(6)).join(",");
    const vertices = new Map<string, number>();
    for (const facet of FACETS) {
      for (const point of facet.polygon) {
        vertices.set(key(point), (vertices.get(key(point)) ?? 0) + 1);
      }
    }
    // 24 vertices — each has exactly one coordinate at ±1, so it belongs to
    // exactly one face — and four facets meet at every one of them: that
    // face, the two bevels either side, and the corner triangle.
    expect(vertices.size).toBe(24);
    for (const [point, count] of vertices) expect(count, point).toBe(4);
  });

  it("points each facet's Shift-click at the view straight through the cube", () => {
    // Twenty of the twenty-six facets are turned away at any moment, so a
    // facet you cannot see is a view you could not otherwise ask for. The
    // modifier takes the far side, the way Ctrl+7 is Bottom to 7's Top —
    // which is only true if `opposite` really is the antipode.
    const byKey = new Map(FACETS.map((facet) => [facet.key, facet]));
    for (const facet of FACETS) {
      const other = byKey.get(facet.opposite);
      expect(other, facet.key).toBeDefined();
      expect(other!.rank, facet.key).toBe(facet.rank);
      // Antipodal normals, and the relation is its own inverse.
      for (const axis of [0, 1, 2]) {
        expect(other!.normal[axis], `${facet.key}.${axis}`).toBeCloseTo(
          -facet.normal[axis],
          12,
        );
      }
      expect(other!.opposite, facet.key).toBe(facet.key);
    }
    expect(byKey.get("top")!.opposite).toBe("bottom");
    expect(byKey.get("front")!.opposite).toBe("back");
    expect(byKey.get("front-right-top")!.opposite).toBe("back-left-bottom");
  });

  it("exactly doubles what the cube can reach from any standpoint", () => {
    // The modifier earns its place by never being redundant: a facet and its
    // antipode are never both on the near side of a convex solid, so Shift
    // reaches twenty views a plain click cannot — including, from anywhere above
    // the floor, BOTTOM, which is otherwise unclickable at every standpoint
    // the camera is allowed to occupy.
    for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
      const basis = cameraBasisFor(preset.yaw, preset.pitch);
      const shown = FACETS.filter((facet) => facetVisible(facet, basis));
      const plain = new Set(shown.map((facet) => facet.key));
      const shifted = new Set(shown.map((facet) => facet.opposite));
      for (const key of shifted) expect(plain.has(key), `${name} ${key}`).toBe(false);
      expect(plain.size + shifted.size, name).toBe(2 * shown.length);
    }
    // From anywhere actually *above* the floor — which is where the camera
    // spends its life — BOTTOM is one Shift-click away. (Dead-on FRONT is the
    // boundary case: the top face is edge-on there, visible from neither
    // side, so it is not clickable and neither is its antipode.)
    for (let yaw = -Math.PI; yaw < Math.PI; yaw += 0.23) {
      for (const pitch of [0.05, 0.4, 0.95, 1.45]) {
        const label = `yaw ${yaw.toFixed(2)} pitch ${pitch}`;
        const basis = cameraBasisFor(yaw, pitch);
        const reachable = FACETS.filter((facet) => facetVisible(facet, basis)).map(
          (facet) => facet.opposite,
        );
        expect(reachable, label).toContain("bottom");
      }
    }
  });

  it("keeps every face's label basis square to its own face", () => {
    for (const facet of FACETS.filter((entry) => entry.rank === "face")) {
      expect(dot(facet.labelRight, facet.normal), facet.key).toBeCloseTo(0, 9);
      expect(dot(facet.labelUp, facet.normal), facet.key).toBeCloseTo(0, 9);
      expect(dot(facet.labelRight, facet.labelUp), facet.key).toBeCloseTo(0, 9);
      expect(facet.label, facet.key).toBe(facet.key.toUpperCase());
    }
  });

  it("stands every face's label up in the view that face names", () => {
    // Squareness is not orientation: a basis can be perfectly orthonormal and
    // still be turned half a circle, which is how BOTTOM shipped reading
    // "WOTTO8" from directly below. Project each label's own basis from the
    // standpoint the face is named after and require the plain identity —
    // `labelRight` along screen +x, `labelUp` along screen −y, since SVG's y
    // runs down the page. Positive determinant alone would let a 180° through.
    for (const facet of FACETS.filter((entry) => entry.rank === "face")) {
      const preset = VIEW_PRESETS[facet.key];
      const basis = cameraBasisFor(preset.yaw, preset.pitch);
      const [rx, ry] = project(facet.labelRight, basis);
      const [ux, uy] = project(facet.labelUp, basis);
      expect(rx, `${facet.key} right.x`).toBeCloseTo(1, 9);
      expect(ry, `${facet.key} right.y`).toBeCloseTo(0, 9);
      expect(ux, `${facet.key} up.x`).toBeCloseTo(0, 9);
      expect(uy, `${facet.key} up.y`).toBeCloseTo(-1, 9);
    }
  });
});

describe("the cube agrees with the camera", () => {
  const expectedFacet = (yaw: number, pitch: number) => {
    const direction = offsetOf(yaw, pitch);
    return FACETS.reduce((best, facet) =>
      dot(facet.normal, direction) > dot(best.normal, direction) ? facet : best,
    );
  };

  it("turns the camera's own facet to the front at all twenty-six views", () => {
    for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
      const basis = cameraBasisFor(preset.yaw, preset.pitch);
      expect(frontFacet(basis)!.key, name).toBe(
        expectedFacet(preset.yaw, preset.pitch).key,
      );
      // …and it is the facet the preset is named after.
      if (name !== "iso") expect(frontFacet(basis)!.key, name).toBe(name);
    }
  });

  it("holds over a sweep of free standpoints", () => {
    for (let yaw = -Math.PI; yaw < Math.PI; yaw += 0.17) {
      for (let pitch = -1.45; pitch <= 1.45; pitch += 0.11) {
        const label = `yaw ${yaw.toFixed(2)} pitch ${pitch.toFixed(2)}`;
        expect(frontFacet(cameraBasisFor(yaw, pitch))!.key, label).toBe(
          expectedFacet(yaw, pitch).key,
        );
      }
    }
  });

  it("shows exactly the facets the camera is outside of", () => {
    for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
      const basis = cameraBasisFor(preset.yaw, preset.pitch);
      const direction = offsetOf(preset.yaw, preset.pitch);
      const shown = FACETS.filter((facet) => facetVisible(facet, basis))
        .map((facet) => facet.key)
        .sort();
      const facing = FACETS.filter((facet) => dot(facet.normal, direction) > 1e-6)
        .map((facet) => facet.key)
        .sort();
      expect(shown, name).toEqual(facing);
      // A convex solid seen from outside never shows more than half of it.
      expect(shown.length, name).toBeLessThan(FACETS.length);
      expect(shown.length, name).toBeGreaterThan(0);
    }
  });

  it("shows TOP from above and BOTTOM from below", () => {
    const facing = (name: string) => {
      const preset = VIEW_PRESETS[name];
      return frontFacet(cameraBasisFor(preset.yaw, preset.pitch))!.label;
    };
    expect(facing("top")).toBe("TOP");
    expect(facing("bottom")).toBe("BOTTOM");
    // The standpoint from the screenshot that exposed the bug: above the
    // floor, so never BOTTOM whatever the azimuth.
    const corner = VIEW_PRESETS["front-left-top"];
    const basis = cameraBasisFor(corner.yaw, corner.pitch);
    const shown = FACETS.filter(
      (facet) => facetVisible(facet, basis) && facet.rank === "face",
    )
      .map((facet) => facet.key)
      .sort();
    expect(shown).toEqual(["front", "left", "top"]);
  });

  it("projects a corner standpoint to a regular hexagon", () => {
    // The signature of a true isometric direction under a parallel
    // projection, and the reason the widget carries no CSS perspective: the
    // cube's eight vertices project onto six points at one radius.
    const preset = VIEW_PRESETS.iso;
    const basis = cameraBasisFor(preset.yaw, preset.pitch);
    const radii = new Set<string>();
    for (const sx of [-1, 1]) {
      for (const sy of [-1, 1]) {
        for (const sz of [-1, 1]) {
          const [x, y] = project([sx, sy, sz], basis);
          radii.add(Math.hypot(x, y).toFixed(6));
        }
      }
    }
    // Six on the silhouette at one radius, two at the centre (the near and
    // far corners, which project on top of each other).
    expect(radii.size).toBe(2);
    expect([...radii].map(Number).sort((a, b) => a - b)[0]).toBeCloseTo(0, 9);
  });
});

