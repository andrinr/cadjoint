/**
 * The editor's three analysis extensions, wired to the playground server.
 *
 * `intelligenceExtensions()` returns everything the CodeMirror instance needs
 * to lint, complete, and describe the call the caret sits inside. The
 * conversions all live next door in `intelligence.ts` — pure, and unit tested;
 * what is left here is the parts that need a `view`, a network round trip, or
 * the DOM: the lint source, the completion override, the signature tooltip's
 * state field, and the theme that paints all three from design tokens.
 *
 * ── The one budget that matters ──────────────────────────────────────────
 * Every request carries the *whole* document. That is why nothing here fires
 * on a keystroke: `linter()` runs on its own idle delay, the completion
 * source is reused locally while the typed prefix still matches `validFor`,
 * and the signature tooltip is debounced behind a pause in typing. A stale
 * answer is dropped rather than shown — each request stamps a token and the
 * response is discarded when a newer one has already gone out.
 */

import {
  autocompletion,
  closeCompletion,
  completionStatus,
  type Completion,
  type CompletionResult,
} from "@codemirror/autocomplete";
import { lintGutter, lintKeymap, linter, type Diagnostic } from "@codemirror/lint";
import { StateEffect, StateField, type Extension } from "@codemirror/state";
import {
  EditorView,
  closeHoverTooltips,
  hasHoverTooltips,
  keymap,
  showTooltip,
  type Tooltip,
  type ViewUpdate,
} from "@codemirror/view";
import { ViewPlugin } from "@codemirror/view";
import * as api from "../api";
import type { LintDiagnostic, SignatureInfo } from "../types";
import {
  COMPLETION_VALID_FOR,
  LINT_DELAY_MS,
  SIGNATURE_DELAY_MS,
  activeSignature,
  completionFrom,
  fixChanges,
  positionOf,
  signatureSegments,
  toCompletions,
  toDiagnostics,
} from "./intelligence";

/** `document.createElement` with a class, because this file does it a lot. */
function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// ── lint ────────────────────────────────────────────────────────────────────

/**
 * The tooltip body for one diagnostic.
 *
 * Beyond the message: the note a display-only fix leaves (ruff can describe a
 * change it will not make), and the rule's documentation as a plain link.
 */
const renderMessage = (item: LintDiagnostic) => (): Node => {
  const root = element("div", "cm-diagnostic-body");
  root.appendChild(element("span", "cm-diagnostic-message", item.message));
  if (item.fix && item.fix.applicability === "display") {
    root.appendChild(element("p", "cm-diagnostic-note", item.fix.message));
  }
  if (item.url) {
    const link = element("a", "cm-diagnostic-link", "learn more");
    link.href = item.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    root.appendChild(link);
  }
  return root;
};

/**
 * Apply a ruff fix as one transaction.
 *
 * The edits are re-resolved against the document as it stands when the button
 * is pressed, not as it stood when the diagnostic was produced: a fix is a
 * list of line/column replacements, and one transaction keeps them atomic in
 * the undo history — accepting a fix is a single ⌘Z.
 */
const applyFix = (item: LintDiagnostic) => (view: EditorView) => {
  const changes = fixChanges(view.state.doc, item.fix!.edits);
  if (changes.length > 0) view.dispatch({ changes });
};

/** Ask the server to lint the current document. */
async function lintSource(view: EditorView): Promise<readonly Diagnostic[]> {
  try {
    const response = await api.lint(view.state.doc.toString());
    return toDiagnostics(view.state.doc, response, { renderMessage, applyFix });
  } catch {
    // A linter that cannot be reached goes quiet; it never takes the editor
    // down with it, and it never puts a network error in the gutter.
    return [];
  }
}

// ── completion ──────────────────────────────────────────────────────────────

/** The documentation panel beside the popup, for the head of the list. */
const completionInfo = (text: string): Completion["info"] => () =>
  element("pre", "cm-completionInfo-doc", text);

/** Jedi completions at the caret, replacing CodeMirror's own sources. */
async function completeSource(context: {
  state: EditorView["state"];
  pos: number;
  explicit: boolean;
}): Promise<CompletionResult | null> {
  const { line, column } = positionOf(context.state.doc, context.pos);
  let response;
  try {
    response = await api.complete(context.state.doc.toString(), line, column);
  } catch {
    return null;
  }
  const from = completionFrom(context.state.doc, response);
  if (from === null) return null;
  const options = toCompletions(response, completionInfo);
  if (options.length === 0) return null;
  return { from, options, validFor: COMPLETION_VALID_FOR };
}

// ── signature help ──────────────────────────────────────────────────────────

interface SignatureState {
  pos: number;
  signature: SignatureInfo;
}

const setSignature = StateEffect.define<SignatureState | null>();

/** The tooltip DOM: the call, its active parameter in bold, and its prose. */
function signatureTooltip(state: SignatureState): Tooltip {
  return {
    pos: state.pos,
    above: true,
    create: () => {
      const dom = element("div", "cm-signature-tooltip");
      const label = element("div", "cm-signature-label");
      for (const segment of signatureSegments(state.signature)) {
        const span = element(
          "span",
          segment.active ? "cm-signature-param cm-signature-param-active" : "cm-signature-param",
          segment.text,
        );
        label.appendChild(span);
      }
      dom.appendChild(label);
      if (state.signature.documentation) {
        dom.appendChild(element("pre", "cm-signature-doc", state.signature.documentation));
      }
      return { dom };
    },
  };
}

/**
 * The signature currently shown.
 *
 * An edit maps the anchor forward rather than dropping it: the tooltip is
 * answering a question the user is still typing the answer to, and blinking it
 * out between every keystroke and the next response is worse than a position
 * that is briefly a character stale.
 */
const signatureField = StateField.define<SignatureState | null>({
  create: () => null,
  update(value, transaction) {
    let next = value;
    if (next && transaction.docChanged) {
      next = { ...next, pos: transaction.changes.mapPos(next.pos) };
    }
    for (const effect of transaction.effects) {
      if (effect.is(setSignature)) next = effect.value;
    }
    return next;
  },
  provide: (field) =>
    showTooltip.from(field, (value) => (value ? signatureTooltip(value) : null)),
});

/**
 * Ask what call the caret sits inside, on a pause.
 *
 * The trigger is deliberately "the caret moved or the text changed" rather
 * than a `(`/`,` keymap: typing `(` opens a call, typing `,` advances the
 * active parameter, and arrowing back into an earlier argument changes it
 * too — one debounced query answers all three, and the server returns an
 * empty list whenever the caret is not inside a call at all.
 */
const signaturePlugin = ViewPlugin.fromClass(
  class {
    private timer: ReturnType<typeof setTimeout> | undefined;
    /** Monotonic request stamp, so a slow answer cannot overwrite a fast one. */
    private stamp = 0;

    constructor(private readonly view: EditorView) {}

    update(update: ViewUpdate): void {
      if (!update.docChanged && !update.selectionSet) return;
      clearTimeout(this.timer);
      this.timer = setTimeout(() => void this.request(), SIGNATURE_DELAY_MS);
    }

    private async request(): Promise<void> {
      // Nothing is asked for while the editor is not the thing being used.
      // The tooltip answers a question about the caret, and a caret in an
      // unfocused editor is not a question anyone is asking.
      if (!this.view.hasFocus) return;
      const state = this.view.state;
      const head = state.selection.main.head;
      const { line, column } = positionOf(state.doc, head);
      const stamp = ++this.stamp;
      let response;
      try {
        response = await api.signature(state.doc.toString(), line, column);
      } catch {
        return;
      }
      if (stamp !== this.stamp) return;
      const signature = activeSignature(response);
      const current = this.view.state.field(signatureField, false) ?? null;
      if (!signature && !current) return;
      this.view.dispatch({
        effects: setSignature.of(signature ? { pos: head, signature } : null),
      });
    }

    destroy(): void {
      clearTimeout(this.timer);
    }
  },
);

// ── dismissal ───────────────────────────────────────────────────────────────

/**
 * Put every floating surface away when the editor stops being used.
 *
 * The three of them have three different lifetimes, and only one of them used
 * to be right. The completion popup closes itself on blur (autocompletion's
 * `closeOnBlur`, on by default). The lint hover already had a `mouseleave` on
 * the editor's own element — but the editor's element is not the *pane*: the
 * header, the hint and the console below it are all outside it, so a pointer
 * moving down through them left the editor without ever leaving it, and the
 * tooltip stayed. And the signature tooltip is not a hover at all: it is a
 * `StateField`, shown because of where the caret is, so nothing about the
 * pointer or the focus was ever going to close it. Clicking the viewport left
 * a call signature floating over a pane nobody was typing in.
 *
 * So dismissal is stated once, for all three, on the two events that actually
 * mean "not editing any more": focus leaving the editor, and the pointer
 * leaving the whole pane.
 */
function dismissTooltips(view: EditorView): void {
  const effects = [];
  if (view.state.field(signatureField, false)) effects.push(setSignature.of(null));
  if (hasHoverTooltips(view.state)) effects.push(closeHoverTooltips);
  if (effects.length > 0) view.dispatch({ effects });
  // Not an effect: the completion state lives behind its own transaction.
  if (completionStatus(view.state) !== null) closeCompletion(view);
}

const dismissOnLeave = ViewPlugin.fromClass(
  class {
    private readonly pane: HTMLElement;
    private readonly leave: () => void;

    constructor(private readonly view: EditorView) {
      // The pane, not the editor: see above. Falling back to the editor's own
      // element keeps this working in a test harness that mounts it bare.
      this.pane = view.dom.closest(".pane") ?? view.dom;
      this.leave = () => dismissTooltips(this.view);
      this.pane.addEventListener("pointerleave", this.leave);
    }

    destroy(): void {
      this.pane.removeEventListener("pointerleave", this.leave);
    }
  },
);

/** Focus leaving the editor is the other end of the same rule. */
const dismissOnBlur = EditorView.domEventHandlers({
  blur(_event, view) {
    dismissTooltips(view);
    return false;
  },
});

// ── paint ───────────────────────────────────────────────────────────────────

/**
 * Everything the three features draw, on design tokens.
 *
 * Two rules shape this block. **Severity is carried by shape as well as by
 * hue** — a filled square, a triangle, a hollow square in the gutter; a wavy
 * rule, a wavy rule, a dotted rule underneath — because a colour-blind reader
 * and a greyscale screenshot both have to be able to tell an error from a
 * hint. And **the marks are tokens, never literals**: CodeMirror's own lint
 * theme paints with `#d11` and `orange` baked into SVG data URIs, so both the
 * gutter markers and the underlines are replaced outright rather than tinted.
 */
const intelligenceTheme = EditorView.theme({
  // Gutter markers: replace the packaged SVG images with token-coloured shapes.
  ".cm-gutter-lint": { width: "1.2em" },
  ".cm-gutter-lint .cm-gutterElement": { padding: "0 var(--space-1)" },
  ".cm-lint-marker": {
    content: "none",
    width: "9px",
    height: "9px",
    margin: "auto",
    backgroundColor: "transparent",
  },
  ".cm-lint-marker-error": { backgroundColor: "var(--danger)" },
  ".cm-lint-marker-warning": {
    backgroundColor: "var(--warn)",
    clipPath: "polygon(50% 0%, 100% 100%, 0% 100%)",
  },
  ".cm-lint-marker-info": {
    backgroundColor: "transparent",
    border: "1.5px solid var(--ink-3)",
  },

  // Underlines: a text decoration rather than a background image, so the
  // colour can be a custom property at all.
  ".cm-lintRange": {
    backgroundImage: "none",
    paddingBottom: "0",
    textDecorationLine: "underline",
    textDecorationSkipInk: "none",
    textUnderlineOffset: "3px",
  },
  ".cm-lintRange-error": {
    textDecorationStyle: "wavy",
    textDecorationColor: "var(--danger)",
  },
  ".cm-lintRange-warning": {
    textDecorationStyle: "wavy",
    textDecorationColor: "var(--warn)",
  },
  ".cm-lintRange-info": {
    textDecorationStyle: "dotted",
    textDecorationThickness: "2px",
    textDecorationColor: "var(--ink-3)",
  },
  // The traceback of a compile that actually failed, drawn heavier than the
  // suspicions around it.
  ".cm-lint-runtime": { textDecorationThickness: "2px" },
  ".cm-lintRange-active": { backgroundColor: "var(--surface-inset-hover)" },

  ".cm-tooltip-lint": {
    backgroundColor: "var(--surface-float)",
    border: "1px solid var(--rule-strong)",
    borderRadius: "var(--radius)",
    color: "var(--ink)",
    maxWidth: "44em",
  },
  ".cm-diagnostic": {
    padding: "var(--space-3) var(--space-4)",
    font: "var(--weight-regular) var(--text-sm) / var(--leading-snug) var(--font-sans)",
  },
  ".cm-diagnostic-error": { borderLeft: "3px solid var(--danger)" },
  ".cm-diagnostic-warning": { borderLeft: "3px solid var(--warn)" },
  ".cm-diagnostic-info": { borderLeft: "3px solid var(--ink-3)" },
  ".cm-diagnostic-note": {
    margin: "var(--space-2) 0 0",
    color: "var(--ink-2)",
  },
  ".cm-diagnostic-link": {
    display: "inline-block",
    marginTop: "var(--space-2)",
    color: "var(--info)",
  },
  ".cm-diagnosticSource": {
    color: "var(--ink-3)",
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-2xs)",
    letterSpacing: "var(--tracking-2xs)",
    opacity: "1",
  },
  ".cm-diagnosticAction": {
    backgroundColor: "var(--accent)",
    color: "var(--ink-on-accent)",
    borderRadius: "var(--radius)",
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-2xs)",
    padding: "var(--space-1) var(--space-3)",
  },
  ".cm-panel.cm-panel-lint": {
    backgroundColor: "var(--surface-panel)",
    borderTop: "1px solid var(--rule-strong)",
  },

  ".cm-tooltip.cm-tooltip-autocomplete": {
    backgroundColor: "var(--surface-float)",
    border: "1px solid var(--rule-strong)",
    borderRadius: "var(--radius)",
  },
  ".cm-tooltip.cm-tooltip-autocomplete > ul > li": {
    padding: "var(--space-1) var(--space-3)",
    font: "var(--weight-regular) var(--text-sm) / var(--leading-tight) var(--font-mono)",
    color: "var(--ink)",
  },
  ".cm-tooltip.cm-tooltip-autocomplete > ul > li[aria-selected]": {
    backgroundColor: "var(--accent)",
    color: "var(--ink-on-accent)",
  },
  ".cm-completionDetail": {
    color: "var(--ink-3)",
    fontSize: "var(--text-2xs)",
    fontStyle: "normal",
    marginLeft: "var(--space-3)",
  },
  ".cm-completionInfo": {
    backgroundColor: "var(--surface-float)",
    border: "1px solid var(--rule-strong)",
    borderRadius: "var(--radius)",
    color: "var(--ink-2)",
    padding: "var(--space-3) var(--space-4)",
    maxWidth: "34em",
    // A docstring is a hint, not the manual: bounded and scrolled, or a long
    // one becomes a column of prose down the whole window.
    maxHeight: "22em",
    overflowY: "auto",
  },
  ".cm-completionInfo-doc": {
    margin: "0",
    whiteSpace: "pre-wrap",
    font: "var(--weight-regular) var(--text-xs) / var(--leading-snug) var(--font-mono)",
  },

  ".cm-signature-tooltip": {
    backgroundColor: "var(--surface-float)",
    border: "1px solid var(--rule-strong)",
    borderRadius: "var(--radius)",
    color: "var(--ink-2)",
    padding: "var(--space-3) var(--space-4)",
    maxWidth: "44em",
  },
  ".cm-signature-label": {
    font: "var(--weight-regular) var(--text-sm) / var(--leading-tight) var(--font-mono)",
    color: "var(--ink-2)",
  },
  ".cm-signature-param-active": {
    color: "var(--ink)",
    fontWeight: "var(--weight-bold)",
  },
  ".cm-signature-doc": {
    margin: "var(--space-3) 0 0",
    whiteSpace: "pre-wrap",
    font: "var(--weight-regular) var(--text-xs) / var(--leading-snug) var(--font-mono)",
    color: "var(--ink-3)",
    maxHeight: "16em",
    overflow: "auto",
  },
});

/**
 * Lint, completion and signature help, ready to drop into the editor.
 *
 * Ordered as they are read: the gutter and its diagnostics, then the popup,
 * then the tooltip, then the paint that unifies them.
 */
export function intelligenceExtensions(): Extension[] {
  return [
    lintGutter(),
    linter(lintSource, { delay: LINT_DELAY_MS }),
    keymap.of(lintKeymap),
    autocompletion({ override: [completeSource] }),
    signatureField,
    signaturePlugin,
    dismissOnLeave,
    dismissOnBlur,
    intelligenceTheme,
  ];
}
