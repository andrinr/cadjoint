/**
 * The editor's analysis adapters held to numbers.
 *
 * The server speaks 1-based lines and 0-based columns; CodeMirror speaks
 * document offsets. Every bug this file exists to catch looks the same in the
 * product — a squiggle under the wrong word, a completion that replaces one
 * character too many — and none of them is visible in a screenshot, so the
 * conversion is asserted rather than eyeballed.
 */

import { Text } from "@codemirror/state";
import { describe, expect, it } from "vitest";
import {
  COMPLETION_VALID_FOR,
  LINT_DELAY_MS,
  activeSignature,
  completionFrom,
  diagnosticRange,
  fixChanges,
  fixIsApplicable,
  offsetOf,
  orderDiagnostics,
  positionOf,
  signatureSegments,
  toCompletions,
  toDiagnostics,
} from "../src/editor/intelligence";
import type { LintDiagnostic, SignatureInfo } from "../src/types";

const doc = Text.of(["import cadjoint", "", "value = wide", "end"]);

/** A diagnostic with the boring fields filled in. */
function diagnostic(overrides: Partial<LintDiagnostic> = {}): LintDiagnostic {
  return {
    from_line: 3,
    from_col: 8,
    to_line: 3,
    to_col: 12,
    severity: "warning",
    message: "Undefined name `wide`",
    code: "F821",
    source: "ruff",
    url: null,
    fix: null,
    ...overrides,
  };
}

describe("line/column ↔ offset", () => {
  it("resolves a position as doc.line(n).from + column", () => {
    expect(offsetOf(doc, 1, 0)).toBe(0);
    expect(offsetOf(doc, 1, 6)).toBe(6);
    // Line 3 starts after "import cadjoint\n" (16) and "\n" (1).
    expect(offsetOf(doc, 3, 0)).toBe(17);
    expect(offsetOf(doc, 3, 8)).toBe(25);
  });

  it("clamps a column past the end of its line", () => {
    expect(offsetOf(doc, 2, 40)).toBe(doc.line(2).from);
    expect(offsetOf(doc, 3, 999)).toBe(doc.line(3).to);
  });

  it("clamps a line past the end of the document", () => {
    expect(offsetOf(doc, 99, 0)).toBe(doc.line(doc.lines).from);
    expect(offsetOf(doc, 0, 0)).toBe(0);
  });

  it("round-trips through positionOf", () => {
    for (const offset of [0, 5, 16, 17, 25, doc.length]) {
      const { line, column } = positionOf(doc, offset);
      expect(offsetOf(doc, line, column)).toBe(offset);
    }
  });

  it("reports a 1-based line and a 0-based column", () => {
    expect(positionOf(doc, 0)).toEqual({ line: 1, column: 0 });
    expect(positionOf(doc, 25)).toEqual({ line: 3, column: 8 });
  });
});

describe("diagnostic ranges", () => {
  it("spans exactly the reported characters", () => {
    expect(diagnosticRange(doc, diagnostic())).toEqual({ from: 25, to: 29 });
    expect(doc.sliceString(25, 29)).toBe("wide");
  });

  it("widens a zero-width range so the squiggle is visible", () => {
    const range = diagnosticRange(doc, diagnostic({ to_col: 8 }));
    expect(range.to - range.from).toBe(1);
  });

  it("pulls a zero-width range at the end of a line back onto the line", () => {
    const range = diagnosticRange(doc, diagnostic({ from_col: 12, to_col: 12 }));
    expect(range).toEqual({ from: 28, to: 29 });
  });

  it("never inverts a range whose ends arrive out of order", () => {
    const range = diagnosticRange(doc, diagnostic({ from_col: 12, to_col: 8 }));
    expect(range.from).toBeLessThanOrEqual(range.to);
  });
});

describe("ruff fixes", () => {
  const fix = {
    message: "Remove unused import",
    applicability: "safe" as const,
    edits: [{ from_line: 1, from_col: 0, to_line: 2, to_col: 0, content: "" }],
  };

  it("resolves edits against the current document", () => {
    expect(fixChanges(doc, fix.edits)).toEqual([{ from: 0, to: 16, insert: "" }]);
  });

  it("offers safe and unsafe fixes but never a display-only one", () => {
    expect(fixIsApplicable(diagnostic({ fix }))).toBe(true);
    expect(fixIsApplicable(diagnostic({ fix: { ...fix, applicability: "unsafe" } }))).toBe(true);
    expect(fixIsApplicable(diagnostic({ fix: { ...fix, applicability: "display" } }))).toBe(false);
    expect(fixIsApplicable(diagnostic())).toBe(false);
  });

  it("attaches an applicable fix as the diagnostic's one action", () => {
    const apply = () => undefined;
    const [item] = toDiagnostics(
      doc,
      { ok: true, diagnostics: [diagnostic({ fix })] },
      { applyFix: () => apply },
    );
    expect(item.actions).toHaveLength(1);
    expect(item.actions![0].name).toBe("Remove unused import");
    expect(item.actions![0].apply).toBe(apply);
  });

  it("leaves a display-only fix without a button", () => {
    const [item] = toDiagnostics(
      doc,
      { ok: true, diagnostics: [diagnostic({ fix: { ...fix, applicability: "display" } })] },
      { applyFix: () => () => undefined },
    );
    expect(item.actions).toBeUndefined();
  });
});

describe("diagnostic ordering", () => {
  const runtime = diagnostic({
    from_line: 4,
    from_col: 0,
    to_line: 4,
    to_col: 3,
    severity: "warning",
    source: "runtime",
    code: "NameError",
  });

  it("puts the compile traceback first, wherever it sits in the file", () => {
    const ordered = orderDiagnostics([diagnostic({ severity: "error" }), runtime]);
    expect(ordered[0].source).toBe("runtime");
  });

  it("shows the traceback as an error whatever the payload called it", () => {
    const [first] = toDiagnostics(doc, { ok: true, diagnostics: [runtime] });
    expect(first.severity).toBe("error");
    expect(first.markClass).toBe("cm-lint-runtime");
  });

  it("then sorts by severity, then by position", () => {
    const ordered = orderDiagnostics([
      diagnostic({ severity: "info", from_line: 1 }),
      diagnostic({ severity: "error", from_line: 3 }),
      diagnostic({ severity: "warning", from_line: 2 }),
      diagnostic({ severity: "error", from_line: 1 }),
    ]);
    expect(ordered.map((item) => [item.severity, item.from_line])).toEqual([
      ["error", 1],
      ["error", 3],
      ["warning", 2],
      ["info", 1],
    ]);
  });

  it("carries the rule code through as the diagnostic's source label", () => {
    const [item] = toDiagnostics(doc, { ok: true, diagnostics: [diagnostic()] });
    expect(item.source).toBe("F821");
    expect(item.markClass).toBe("cm-lint-ruff");
  });

  it("shows nothing at all when the analyser could not run", () => {
    expect(toDiagnostics(doc, { ok: false, error: "no ruff" })).toEqual([]);
  });

  it("waits CodeMirror's own idle delay rather than a keystroke", () => {
    expect(LINT_DELAY_MS).toBe(750);
  });
});

describe("completions", () => {
  const response = {
    ok: true,
    from_line: 3,
    from_column: 8,
    truncated: false,
    completions: [
      { label: "origin", type: "property", detail: "param", info: "param origin=(0,0,0)", apply: "origin=" },
      { label: "normal", type: "property", detail: "param", info: null, apply: "normal=" },
      { label: "abs", type: "function", detail: "function", info: null, apply: "abs" },
    ],
  };

  it("replaces from the start of the typed prefix", () => {
    expect(completionFrom(doc, response)).toBe(25);
    expect(completionFrom(doc, { ok: false })).toBeNull();
  });

  it("keeps the server's keyword arguments in front, in order", () => {
    const options = toCompletions(response);
    expect(options.map((option) => option.label)).toEqual(["origin", "normal", "abs"]);
    expect(options[0].boost).toBe(99);
    expect(options[1].boost).toBe(98);
    expect(options[2].boost).toBeUndefined();
  });

  it("carries label, type, detail and apply text through unchanged", () => {
    const [first] = toCompletions(response);
    expect(first).toMatchObject({
      label: "origin",
      type: "property",
      detail: "param",
      apply: "origin=",
    });
  });

  it("only builds a documentation panel for the entries that have one", () => {
    const options = toCompletions(response, (text) => text);
    expect(options[0].info).toBe("param origin=(0,0,0)");
    expect(options[1].info).toBeUndefined();
  });

  it("lets a typed word filter locally instead of refetching", () => {
    expect(COMPLETION_VALID_FOR.test("SketchPlane")).toBe(true);
    expect(COMPLETION_VALID_FOR.test("plane.on")).toBe(true);
    expect(COMPLETION_VALID_FOR.test("")).toBe(true);
    expect(COMPLETION_VALID_FOR.test("plane(")).toBe(false);
  });
});

describe("signature help", () => {
  const signature: SignatureInfo = {
    name: "SketchPlane",
    label: "SketchPlane(origin, normal, x_axis=None)",
    active_parameter: 1,
    parameters: [
      { name: "origin", label: "origin" },
      { name: "normal", label: "normal" },
      { name: "x_axis", label: "x_axis=None" },
    ],
    documentation: "A plane a sketch is drawn on.",
  };

  it("splits the call into runs and marks only the active parameter", () => {
    const segments = signatureSegments(signature);
    expect(segments.map((segment) => segment.text).join("")).toBe(
      "SketchPlane(origin, normal, x_axis=None)",
    );
    expect(segments.filter((segment) => segment.active).map((s) => s.text)).toEqual(["normal"]);
  });

  it("marks nothing when the caret sits between calls", () => {
    const segments = signatureSegments({ ...signature, active_parameter: null });
    expect(segments.some((segment) => segment.active)).toBe(false);
  });

  it("does not confuse two parameters sharing a name prefix", () => {
    const segments = signatureSegments({
      ...signature,
      active_parameter: 2,
      parameters: [
        { name: "x", label: "x" },
        { name: "x_axis", label: "x_axis" },
        { name: "x_axis_2", label: "x_axis_2" },
      ],
    });
    expect(segments.filter((segment) => segment.active).map((s) => s.text)).toEqual(["x_axis_2"]);
  });

  it("shows the resolved overload, and nothing when there is none", () => {
    expect(activeSignature({ ok: true, signatures: [signature] })).toBe(signature);
    expect(activeSignature({ ok: true, signatures: [] })).toBeNull();
    expect(activeSignature({ ok: false })).toBeNull();
  });
});
