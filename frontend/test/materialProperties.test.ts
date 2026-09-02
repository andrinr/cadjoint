/**
 * The material inspector's physical half.
 *
 * Two things are worth asserting and neither is visible in a screenshot: that
 * a property the program does not state is *left out* rather than shown as
 * zero, and that an SI magnitude is printed in a form a reader can compare —
 * a Young's modulus and a thermal expansion are eleven orders of magnitude
 * apart and only one notation survives both.
 */

import { describe, expect, it } from "vitest";
import {
  PHYSICAL_PROPERTIES,
  formatPhysical,
  hasPhysicalBlock,
  physicalRows,
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

  it("draws a section, and no rows, for a material that states none", () => {
    const stated = material({
      physical: Object.fromEntries(PHYSICAL_PROPERTIES.map(({ key }) => [key, null])),
      units: UNITS,
    });
    expect(hasPhysicalBlock(stated)).toBe(true);
    expect(physicalRows(stated)).toEqual([]);
  });

  it("lists only the stated properties, in reading order, with their units", () => {
    const stated = material({
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
    });
    expect(physicalRows(stated)).toEqual([
      { key: "density", label: "Density", value: "2700", unit: "kg/m^3" },
      { key: "conductivity", label: "Conductivity", value: "237", unit: "W/(m*K)" },
      { key: "youngs_modulus", label: "Young's modulus", value: "6.9e10", unit: "Pa" },
      // A dimensionless ratio prints no unit rather than the payload's "-".
      { key: "poisson_ratio", label: "Poisson ratio", value: "0.33", unit: "" },
    ]);
  });
});
