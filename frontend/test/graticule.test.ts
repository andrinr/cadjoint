/**
 * The graticule's two contracts.
 *
 * 1. The gain is a *measurement*: one division is worth a definite number of
 *    millimetres, derived from the camera and the scene's declared unit, and
 *    it says so honestly — off the 1-2-5 ladder, or under a projection where
 *    the scale is not uniform over the frame, the reading is marked
 *    uncalibrated. A grid whose readout can be wrong is decoration.
 * 2. The tones are *furniture*: measured against paper they must land inside
 *    a band, not merely above a floor. Too weak and the grid is invisible;
 *    too strong and it competes with the field it exists to sit behind.
 */

import { describe, expect, it } from "vitest";
import {
  DIVISIONS,
  MM_PER_UNIT,
  distanceForGain,
  formatGain,
  gainFor,
  gainOf,
  isDetented,
  nearestDetent,
  octant,
  stepDetent,
  viewLabel,
  worldHeightFor,
} from "../src/viewer/graticule";
import { FOV_SCALE, orthoHeightFor } from "../src/viewer/math";
import { VIEW_PRESETS } from "../src/viewer/display";
import { detentZoomCamera, MAX_DISTANCE, MIN_DISTANCE } from "../src/components/viewer/camera";
import { contrastRatio } from "../src/simColors";
import { CHROME, GRATICULE_TONES, hexToRgb } from "../src/tokens";

const PAPER = hexToRgb(CHROME["surface-viewport"]);

describe("the 1-2-5 ladder", () => {
  it("snaps to the nearest rung on a log scale", () => {
    expect(nearestDetent(10)).toBe(10);
    expect(nearestDetent(11)).toBe(10);
    expect(nearestDetent(9)).toBe(10);
    expect(nearestDetent(862.5)).toBe(1000);
    expect(nearestDetent(3)).toBe(2);
    expect(nearestDetent(3.5)).toBe(5);
    expect(nearestDetent(0.07)).toBe(0.05);
    expect(nearestDetent(7000)).toBe(5000);
  });

  it("only ever returns a 1, 2 or 5 followed by zeros", () => {
    for (const sample of [0.013, 0.4, 7.7, 63, 480, 3300, 91_000]) {
      const rung = nearestDetent(sample);
      const mantissa = rung / 10 ** Math.round(Math.log10(rung / 5) + Math.log10(5));
      expect([1, 2, 5], `${sample} -> ${rung}`).toContain(Math.round(mantissa));
    }
  });

  it("refuses a scale that is not a positive number", () => {
    expect(nearestDetent(0)).toBeNaN();
    expect(nearestDetent(-4)).toBeNaN();
    expect(nearestDetent(Number.NaN)).toBeNaN();
  });

  it("steps a full rung even from a rung it is already sitting on", () => {
    expect(stepDetent(10, 1)).toBe(20);
    expect(stepDetent(10, -1)).toBe(5);
    expect(stepDetent(5, -1)).toBe(2);
    expect(stepDetent(2, -1)).toBe(1);
    expect(stepDetent(1, -1)).toBeCloseTo(0.5, 12);
    expect(stepDetent(1, 1)).toBe(2);
  });

  it("steps to the next rung in the direction of travel when off-ladder", () => {
    expect(stepDetent(862.5, 1)).toBe(1000);
    expect(stepDetent(862.5, -1)).toBe(500);
    expect(stepDetent(1.4, 1)).toBe(2);
    expect(stepDetent(1.4, -1)).toBe(1);
  });

  it("walks a full decade in three steps, both ways", () => {
    let up = 10;
    for (let step = 0; step < 3; step++) up = stepDetent(up, 1);
    expect(up).toBe(100);
    let down = 100;
    for (let step = 0; step < 3; step++) down = stepDetent(down, -1);
    expect(down).toBeCloseTo(10, 9);
  });

  it("reports detent only on the ladder itself", () => {
    expect(isDetented(10)).toBe(true);
    expect(isDetented(1000)).toBe(true);
    expect(isDetented(0.2)).toBe(true);
    expect(isDetented(862.5)).toBe(false);
    expect(isDetented(11)).toBe(false);
  });
});

describe("the scale derivation", () => {
  it("divides the framed world height into eight square divisions", () => {
    // 1 world unit = 1 m, the only length the repository declares
    // (cadjoint/meshing/export.py stamps SI_UNIT($,.METRE.)).
    expect(MM_PER_UNIT).toBe(1000);
    expect(gainFor(8)).toBe(1000);
    expect(gainFor(DIVISIONS * 0.01)).toBeCloseTo(10, 9);
  });

  it("round-trips gain, world height and orbit distance", () => {
    for (const mm of [75, 100, 862.5, 1000, 11_250]) {
      expect(worldHeightFor(mm)).toBeCloseTo((mm / MM_PER_UNIT) * DIVISIONS, 12);
      expect(gainFor(worldHeightFor(mm))).toBeCloseTo(mm, 9);
      expect(orthoHeightFor(distanceForGain(mm))).toBeCloseTo(worldHeightFor(mm), 9);
      expect(distanceForGain(mm)).toBeCloseTo(worldHeightFor(mm) / FOV_SCALE, 12);
    }
  });

  it("reads the default framing as an uncalibrated 862.5 mm per division", () => {
    // The renderer's starting camera: distance 4.6, FOV_SCALE 1.5.
    const gain = gainOf(4.6, "orthographic");
    expect(gain.mm).toBeCloseTo(862.5, 9);
    expect(gain.calibrated).toBe(false);
  });

  it("is calibrated on the ladder under an orthographic camera", () => {
    const gain = gainOf(distanceForGain(1000), "orthographic");
    expect(gain.mm).toBeCloseTo(1000, 9);
    expect(gain.calibrated).toBe(true);
  });

  it("is never calibrated under perspective, where the scale varies with depth", () => {
    const gain = gainOf(distanceForGain(1000), "perspective");
    expect(gain.mm).toBeCloseTo(1000, 9);
    expect(gain.calibrated).toBe(false);
  });

  it("covers the whole zoom range with rungs", () => {
    const closest = gainOf(MIN_DISTANCE, "orthographic").mm;
    const furthest = gainOf(MAX_DISTANCE, "orthographic").mm;
    expect(closest).toBeCloseTo(75, 6);
    expect(furthest).toBeCloseTo(11_250, 6);
    const reachable = [100, 200, 500, 1000, 2000, 5000, 10_000];
    for (const rung of reachable) {
      const distance = distanceForGain(rung);
      expect(distance, `${rung} mm/div`).toBeGreaterThanOrEqual(MIN_DISTANCE);
      expect(distance, `${rung} mm/div`).toBeLessThanOrEqual(MAX_DISTANCE);
    }
  });
});

describe("the gain readout", () => {
  it("prints three significant figures and switches unit at a metre", () => {
    expect(formatGain(100, true)).toMatchObject({ value: "100", unit: "mm" });
    expect(formatGain(10, true)).toMatchObject({ value: "10.0", unit: "mm" });
    expect(formatGain(1000, true)).toMatchObject({ value: "1.00", unit: "m" });
    expect(formatGain(11_250, true)).toMatchObject({ value: "11.3", unit: "m" });
  });

  it("prefixes an uncalibrated scale with >, as a 2465 does", () => {
    expect(formatGain(862.5, false).text).toBe(">863");
    expect(formatGain(1000, false).text).toBe(">1.00");
    expect(formatGain(1000, true).text).toBe("1.00");
  });

  it("states an absent scale rather than printing a wrong one", () => {
    expect(formatGain(0, true).text).toBe("—");
    expect(formatGain(Number.NaN, true).text).toBe("—");
  });
});

describe("the view readout", () => {
  it("names a standard view and reports FREE once orbited off it", () => {
    for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
      expect(viewLabel(preset.yaw, preset.pitch)).toBe(name.toUpperCase());
    }
    expect(viewLabel(VIEW_PRESETS.front.yaw + 0.4, 0)).toBe("FREE");
  });

  it("spells the octant the camera stands in", () => {
    expect(octant(VIEW_PRESETS.iso.yaw, VIEW_PRESETS.iso.pitch)).toBe("+X+Y+Z");
    expect(octant(VIEW_PRESETS.front.yaw, VIEW_PRESETS.front.pitch)).toBe("+Z");
    expect(octant(VIEW_PRESETS.left.yaw, VIEW_PRESETS.left.pitch)).toBe("−X");
    expect(octant(VIEW_PRESETS.top.yaw, VIEW_PRESETS.top.pitch)).toBe("+Y");
  });
});

describe("detented zoom", () => {
  const camera = { yaw: 0.75, pitch: 0.32, distance: 4.6, target: [0, 0, 0] as const };

  it("lands on the ladder from an off-ladder start", () => {
    const out = detentZoomCamera({ ...camera }, -100);
    expect(gainOf(out.distance, "orthographic").calibrated).toBe(true);
    expect(gainOf(out.distance, "orthographic").mm).toBeCloseTo(500, 6);
  });

  it("walks one rung per notch, in the wheel's direction", () => {
    let out = detentZoomCamera({ ...camera }, 100);
    expect(gainOf(out.distance, "orthographic").mm).toBeCloseTo(1000, 6);
    out = detentZoomCamera(out, 100);
    expect(gainOf(out.distance, "orthographic").mm).toBeCloseTo(2000, 6);
    out = detentZoomCamera(out, -100);
    expect(gainOf(out.distance, "orthographic").mm).toBeCloseTo(1000, 6);
  });

  it("refuses a step that would be clamped off the ladder", () => {
    const far = detentZoomCamera({ ...camera, distance: MAX_DISTANCE }, 100);
    expect(far.distance).toBe(MAX_DISTANCE);
    const near = detentZoomCamera({ ...camera, distance: MIN_DISTANCE }, -100);
    expect(near.distance).toBe(MIN_DISTANCE);
  });

  it("leaves the rest of the camera alone", () => {
    const out = detentZoomCamera({ ...camera }, 100);
    expect(out.yaw).toBe(camera.yaw);
    expect(out.pitch).toBe(camera.pitch);
    expect(out.target).toBe(camera.target);
  });
});

describe("the graticule reads as structure, not as content", () => {
  it("holds every tone inside the furniture band on paper", () => {
    for (const tone of GRATICULE_TONES) {
      const ratio = contrastRatio(hexToRgb(CHROME[tone]), PAPER);
      // The floor is what survives the render: the framebuffer is smaller
      // than the CSS box under the quality budget, so a hairline is resampled
      // across ~1.25 pixels and loses roughly 0.2 of its ratio in the process.
      expect(ratio, `${tone} on paper`).toBeGreaterThan(1.6);
      // The ceiling is the 3:1 a meaningful mark owes, held well clear: this
      // one is deliberately not meaningful, and past ~2.8 the grid starts to
      // survive being attended away from, which is the one thing it must not do.
      expect(ratio, `${tone} on paper`).toBeLessThan(2.8);
    }
  });

  it("keeps the three weights separable and in order", () => {
    const ratios = GRATICULE_TONES.map((tone) =>
      contrastRatio(hexToRgb(CHROME[tone]), PAPER),
    );
    for (let index = 1; index < ratios.length; index++) {
      expect(ratios[index], GRATICULE_TONES[index]).toBeGreaterThan(ratios[index - 1] + 0.15);
    }
  });

  it("stays quieter than the ink drawn on top of it", () => {
    const ink = contrastRatio(hexToRgb(CHROME["viewport-ink-2"]), PAPER);
    for (const tone of GRATICULE_TONES) {
      expect(contrastRatio(hexToRgb(CHROME[tone]), PAPER), tone).toBeLessThan(ink);
    }
  });
});
