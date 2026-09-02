/**
 * Editor intelligence, as pure adapters.
 *
 * The three analysis endpoints (`/api/lint`, `/api/complete`,
 * `/api/signature`) all speak one coordinate convention — **1-based lines,
 * 0-based columns** — and CodeMirror speaks document offsets. Everything that
 * converts between the two, and everything that reshapes a server payload
 * into a CodeMirror value, lives here: no DOM, no network, no editor
 * instance, so `test/editorIntelligence.test.ts` can hold the conversion to
 * numbers instead of trusting it.
 *
 * The one rule worth stating twice, because an off-by-one here draws a
 * squiggle under the wrong word: `offset = doc.line(line).from + column`.
 */

import type { Text } from "@codemirror/state";
import type { Action, Diagnostic } from "@codemirror/lint";
import type { Completion } from "@codemirror/autocomplete";
import type {
  CompleteResponse,
  LintDiagnostic,
  LintFixEdit,
  LintResponse,
  SignatureInfo,
} from "../types";

/**
 * How long the editor waits, idle, before it asks the server anything.
 *
 * Every request carries the *whole* document, so a per-keystroke analysis
 * would ship the program a hundred times a minute. 750 ms is CodeMirror's own
 * default for `linter()` and it is reused here rather than invented.
 */
export const LINT_DELAY_MS = 750;

/**
 * The signature tooltip's debounce.
 *
 * Shorter than the lint delay on purpose: the tooltip answers "what goes in
 * this argument", asked by typing `(`, and an answer three quarters of a
 * second later arrives after the user has already guessed. It is still a
 * debounce — one request per pause in typing, never one per keystroke.
 */
export const SIGNATURE_DELAY_MS = 250;

/** Filter deciding which completions may be reused while typing locally. */
export const COMPLETION_VALID_FOR = /^[\w.]*$/;

/** Jedi's kind for a keyword argument, which the popup keeps at the top. */
const KEYWORD_ARGUMENT = "param";

/**
 * Document offset of a (1-based line, 0-based column) position.
 *
 * Both halves are clamped: the caret can legitimately sit one keystroke ahead
 * of the text the server last analysed, and a diagnostic must never be able
 * to throw a `RangeError` out of the editor's update cycle.
 */
export function offsetOf(doc: Text, line: number, column: number): number {
  const number = Math.min(Math.max(Math.trunc(line), 1), doc.lines);
  const target = doc.line(number);
  return target.from + Math.min(Math.max(Math.trunc(column), 0), target.length);
}

/** The inverse: a 1-based line and 0-based column for a document offset. */
export function positionOf(doc: Text, offset: number): { line: number; column: number } {
  const clamped = Math.min(Math.max(offset, 0), doc.length);
  const line = doc.lineAt(clamped);
  return { line: line.number, column: clamped - line.from };
}

/**
 * The character range one diagnostic marks.
 *
 * Ruff reports zero-width ranges for a few rules and CodeMirror draws nothing
 * under an empty range, so a collapsed span is widened by one character (or
 * pulled back one, at the end of a line) — a squiggle that cannot be seen is
 * the same as no diagnostic at all.
 */
export function diagnosticRange(
  doc: Text,
  item: Pick<LintDiagnostic, "from_line" | "from_col" | "to_line" | "to_col">,
): { from: number; to: number } {
  const from = offsetOf(doc, item.from_line, item.from_col);
  const to = Math.max(from, offsetOf(doc, item.to_line, item.to_col));
  if (to > from) return { from, to };
  const line = doc.lineAt(from);
  if (from < line.to) return { from, to: from + 1 };
  return { from: Math.max(line.from, from - 1), to: from };
}

/** The edits of a ruff fix, as CodeMirror changes against the current text. */
export function fixChanges(
  doc: Text,
  edits: readonly LintFixEdit[],
): { from: number; to: number; insert: string }[] {
  return edits.map((edit) => {
    const { from, to } = {
      from: offsetOf(doc, edit.from_line, edit.from_col),
      to: offsetOf(doc, edit.to_line, edit.to_col),
    };
    return { from: Math.min(from, to), to: Math.max(from, to), insert: edit.content };
  });
}

/**
 * Whether a fix is something the user can actually apply.
 *
 * Ruff's `display` applicability means "here is what a fix would look like",
 * not "this is safe to run" — those are shown as prose in the tooltip and
 * never as a button.
 */
export const fixIsApplicable = (item: LintDiagnostic): boolean =>
  item.fix !== null && item.fix.applicability !== "display" && item.fix.edits.length > 0;

/**
 * Diagnostics in the order the tooltip and the panel should list them.
 *
 * A `runtime` diagnostic is the traceback of a compile that actually failed
 * on this exact text: it names a line that is provably wrong, where every
 * ruff finding is a suspicion. So it sorts first regardless of position, then
 * errors, then warnings, then info, then document order.
 */
const SEVERITY_RANK: Record<string, number> = { error: 0, warning: 1, info: 2 };

export function orderDiagnostics(items: readonly LintDiagnostic[]): LintDiagnostic[] {
  return [...items].sort((a, b) => {
    if ((a.source === "runtime") !== (b.source === "runtime")) {
      return a.source === "runtime" ? -1 : 1;
    }
    const severity = (SEVERITY_RANK[a.severity] ?? 3) - (SEVERITY_RANK[b.severity] ?? 3);
    if (severity !== 0) return severity;
    return a.from_line - b.from_line || a.from_col - b.from_col;
  });
}

/**
 * Hooks the editor supplies so this module stays free of the DOM.
 *
 * `renderMessage` builds the tooltip body (the message, its rule code, the
 * "learn more" link, and the note a display-only fix leaves behind);
 * `applyFix` runs a fix's edits as one transaction. Both are optional, and
 * without them the adapter still produces valid, testable diagnostics.
 */
export interface DiagnosticHooks {
  renderMessage?: (item: LintDiagnostic) => Diagnostic["renderMessage"];
  applyFix?: (item: LintDiagnostic) => Action["apply"];
}

/**
 * The server's diagnostics as CodeMirror's, runtime failure first.
 *
 * `markClass` carries the source through to the decoration so a traceback
 * squiggle can be told apart from a lint squiggle of the same severity.
 */
export function toDiagnostics(
  doc: Text,
  response: LintResponse,
  hooks: DiagnosticHooks = {},
): Diagnostic[] {
  if (!response.ok || !response.diagnostics) return [];
  return orderDiagnostics(response.diagnostics).map((item) => {
    const { from, to } = diagnosticRange(doc, item);
    const diagnostic: Diagnostic = {
      from,
      to,
      // A traceback is not a suspicion, whatever the payload calls it.
      severity: item.source === "runtime" ? "error" : item.severity,
      source: item.code,
      message: item.message,
      markClass: `cm-lint-${item.source}`,
    };
    const render = hooks.renderMessage?.(item);
    if (render) diagnostic.renderMessage = render;
    const apply = fixIsApplicable(item) ? hooks.applyFix?.(item) : undefined;
    if (apply) diagnostic.actions = [{ name: item.fix!.message, apply }];
    return diagnostic;
  });
}

/**
 * The completion popup's options, keyword arguments kept in front.
 *
 * Jedi is asked for the enclosing call's parameters first and the server
 * preserves that partition; CodeMirror re-sorts by match score, so the order
 * has to be restated as a `boost` or it is lost the moment a prefix is typed.
 * Only the parameters are boosted — everything else keeps CodeMirror's own
 * ranking, which is better than a stale server order once the user is typing.
 */
export function toCompletions(
  response: CompleteResponse,
  info?: (item: string) => Completion["info"],
): Completion[] {
  if (!response.ok || !response.completions) return [];
  let keywordRank = 0;
  return response.completions.map((item) => {
    const completion: Completion = {
      label: item.label,
      type: item.type,
      detail: item.detail,
      apply: item.apply,
    };
    if (item.detail === KEYWORD_ARGUMENT) completion.boost = 99 - keywordRank++;
    if (item.info && info) completion.info = info(item.info);
    return completion;
  });
}

/** Where the popup's replacement starts, or null when the answer is unusable. */
export function completionFrom(doc: Text, response: CompleteResponse): number | null {
  if (!response.ok || response.from_line === undefined || response.from_column === undefined) {
    return null;
  }
  return offsetOf(doc, response.from_line, response.from_column);
}

/** One run of a signature's rendered label: a parameter, or the text between. */
export interface SignatureSegment {
  text: string;
  /** True for the parameter the caret is currently filling in. */
  active: boolean;
}

/**
 * A signature split into the runs a tooltip prints.
 *
 * Built from `name` and the parameter labels rather than by searching jedi's
 * rendered `label` for a substring: two parameters can share a name prefix
 * (`x`, `x_axis`), and a substring search would embolden the wrong one.
 */
export function signatureSegments(signature: SignatureInfo): SignatureSegment[] {
  const segments: SignatureSegment[] = [{ text: `${signature.name}(`, active: false }];
  signature.parameters.forEach((parameter, index) => {
    if (index > 0) segments.push({ text: ", ", active: false });
    segments.push({ text: parameter.label, active: index === signature.active_parameter });
  });
  segments.push({ text: ")", active: false });
  return segments;
}

/**
 * The signature to show, or null when the caret is not inside a call.
 *
 * Jedi can report several overloads; the first is the one it resolved the
 * call to, and a tooltip that stacks them all is a wall of text over the code
 * the user is trying to read.
 */
export function activeSignature(response: { ok: boolean; signatures?: SignatureInfo[] }):
  | SignatureInfo
  | null {
  if (!response.ok || !response.signatures || response.signatures.length === 0) return null;
  return response.signatures[0];
}
