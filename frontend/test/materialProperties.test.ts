/**
 * The material inspector's physical half.
 *
 * Two things are worth asserting and neither is visible in a screenshot: that
 * a property the program does not state is reported as *unstated* rather than
 * as a zero — the inspector's whole offer on such a row is "start stating it",
 * which needs the three states told apart — and that an SI magnitude is
 * printed in a form a reader can compare, a Young's modulus and a thermal
 * expansion being eleven orders of magnitude apart with only one notation
 * surviving both.
 */

import { describe, expect, it } from "vitest";
import {
  PHYSICAL_PROPERTIES,
  formatPhysical,
  hasPhysicalBlock,
  physicalRows,
  statedRows,
} from "../src/materialProperties";
import type { MaterialDefinition } from "../src/types";

const UNITS = {
  density: "kg/m^3",
  conductivity: "W/(m*K)",
  specific_heat: "J/(kg*K)",
  youngs_modulus: "Pa",
  poisson_ratio: "-",
  thermal_expansion: "1/K",
  yield_strength: "Pa",
};

function material(overrides: Partial<MaterialDefinition> = {}): MaterialDefinition {
  return {
    id: "material_0",
    stableId: null,
    name: "aluminum",
    line: 56,
    editable: true,
    color: [0.8, 0.82, 0.85],
    roughness: 0.3,
    metallic: 0.9,
    opacity: 1,
    ior: 1,
    reflectivity: 0,
    spans: {},
    ...overrides,
  };
}

describe("printing an SI magnitude", () => {
  it("keeps a human-sized number plain", () => {
    expect(formatPhysical(2700)).toBe("2700");
    expect(formatPhysical(0.33)).toBe("0.33");
    expect(formatPhysical(237)).toBe("237");
    expect(formatPhysical(0)).toBe("0");
  });

  it("goes exponential where plain notation becomes digit-counting", () => {
    expect(formatPhysical(7e10)).toBe("7e10");
    expect(formatPhysical(2.31e-5)).toBe("2.31e-5");
    expect(formatPhysical(-1.5e8)).toBe("-1.5e8");
  });

  it("refuses to print a non-number as one", () => {
    expect(formatPhysical(Number.NaN)).toBe("—");
    expect(formatPhysical(Number.POSITIVE_INFINITY)).toBe("—");
  });
});

describe("the rows the inspector draws", () => {
  it("draws nothing at all when the server sends no physical block", () => {
    expect(hasPhysicalBlock(material())).toBe(false);
    expect(physicalRows(material())).toEqual([]);
  });

  it("offers every property, unstated, for a material that states none", () => {
    const declared = material({
      physical: Object.fromEntries(PHYSICAL_PROPERTIES.map(({ key }) => [key, null])),
      units: UNITS,
    });
    expect(hasPhysicalBlock(declared)).toBe(true);
    // Seven rows, all of them an offer rather than a reading. A row that is
    // not drawn is an offer nobody can take.
    expect(physicalRows(declared)).toHaveLength(PHYSICAL_PROPERTIES.length);
    expect(physicalRows(declared).every((row) => row.state === "unstated")).toBe(true);
    expect(physicalRows(declared).every((row) => row.value === null)).toBe(true);
    expect(statedRows(declared)).toEqual([]);
  });

  it("reads the stated properties, in reading order, with their units", () => {
    const declared = material({
      physical: {
        density: 2700,
        conductivity: 237,
        specific_heat: null,
        youngs_modulus: 6.9e10,
        poisson_ratio: 0.33,
        thermal_expansion: null,
        yield_strength: null,
      },
      units: UNITS,
      free: { density: true },
    });
    expect(statedRows(declared)).toEqual([
      {
        key: "density",
        label: "Density",
        state: "stated",
        value: 2700,
        text: "2700",
        unit: "kg/m^3",
        free: true,
      },
      {
        key: "conductivity",
        label: "Conductivity",
        state: "stated",
        value: 237,
        text: "237",
        unit: "W/(m*K)",
        free: false,
      },
      {
        key: "youngs_modulus",
        label: "Young's modulus",
        state: "stated",
        value: 6.9e10,
        text: "6.9e10",
        unit: "Pa",
        free: false,
      },
      // A dimensionless ratio prints no unit rather than the payload's "-".
      {
        key: "poisson_ratio",
        label: "Poisson ratio",
        state: "stated",
        value: 0.33,
        text: "0.33",
        unit: "",
        free: false,
      },
    ]);
  });

  it("tells an expression apart from an absence", () => {
    // Both arrive as a null. A keyword whose value is `thickness * 2` has a
    // span into the source and is not ours to overwrite; a keyword that is
    // simply not there has no span, and is the row the inspector can offer to
    // start. Reading both as "absent" would put a number box over an
    // expression and silently flatten it on the first commit.
    const declared = material({
      physical: { density: null, conductivity: null },
      units: UNITS,
      spans: { density: [10, 20] },
    });
    const rows = new Map(physicalRows(declared).map((row) => [row.key, row]));
    expect(rows.get("density")!.state).toBe("opaque");
    expect(rows.get("conductivity")!.state).toBe("unstated");
    expect(rows.get("density")!.text).toBe("—");
  });
});
