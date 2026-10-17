/**
 * CodeMirror editor pane.
 *
 * Besides editing, this is the other half of viewer/code parity: selecting a
 * sketch vertex in the 3D view highlights and scrolls to the exact characters
 * that define it.
 */

import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags } from "@lezer/highlight";
import { EditorState, StateEffect, StateField, type Extension } from "@codemirror/state";
import {
  Decoration,
  EditorView,
  keymap,
  lineNumbers,
  type DecorationSet,
  MatchDecorator,
  ViewPlugin,
  type ViewUpdate,
} from "@codemirror/view";
import { createEffect, onCleanup, onMount } from "solid-js";
import { intelligenceExtensions, textRole } from "../editor/extensions";
import { minimalChange } from "../editor/minimalChange";
import type { FocusSpan } from "../editorFocus";
import { consoleText, sceneName, setDirty, setSource, source } from "../state";

/** Highlight the character span of the selected sketch vertex. */
const setHighlight = StateEffect.define<{ from: number; to: number } | null>();

const highlightMark = Decoration.mark({ class: "cm-vertex-highlight" });

const highlightField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(value, transaction) {
    let next = value.map(transaction.changes);
    for (const effect of transaction.effects) {
      if (effect.is(setHighlight)) {
        next = effect.value
          ? Decoration.set([highlightMark.range(effect.value.from, effect.value.to)])
          : Decoration.none;
      }
    }
    return next;
  },
  provide: (field) => EditorView.decorations.from(field),
});

/**
 * Syntax colours for the editor.
 *
 * CodeMirror ships no highlighting unless a style is installed — the language
 * package only supplies the parser — so this defines one on the app's palette
 * rather than pulling in the default light-leaning theme.
 *
 * Every colour is a design token (src/tokens.ts, mirrored into styles.css), so
 * the source pane and the panels around it are painted from one system, and
 * every one of them clears WCAG AA on --surface-panel.
 *
 * The mapping is achromatic — hue is meaning, and syntax has none to carry —
 * so ink level alone gives the program's shape: values at full ink;
 * declarations, keywords and strings one step down; argument names,
 * punctuation and comments two steps down.
 */
const highlightStyle = HighlightStyle.define([
  // The values are the subject (A1): numbers and booleans are the literals a
  // panel, a drag or the optimiser rewrites, so they alone take full ink.
  { tag: [tags.number, tags.bool, tags.null], color: "var(--ink)" },
  // The program's structure — what it declares and the words it is built
  // from — one ink step down.
  { tag: tags.definition(tags.variableName), color: "var(--ink-2)" },
  { tag: tags.function(tags.variableName), color: "var(--ink-2)" },
  { tag: [tags.className, tags.typeName, tags.namespace], color: "var(--ink-2)" },
  { tag: [tags.keyword, tags.controlKeyword, tags.moduleKeyword], color: "var(--ink-2)" },
  { tag: tags.self, color: "var(--ink-2)", fontStyle: "italic" },
  { tag: [tags.string, tags.special(tags.string)], color: "var(--ink-2)" },
  // Argument names, punctuation and comments recede, so `density=8940.0`
  // reads as its number.
  { tag: [tags.propertyName, tags.attributeName], color: "var(--ink-3)" },
  { tag: tags.variableName, color: "var(--ink)" },
  { tag: [tags.operator, tags.punctuation, tags.separator, tags.paren, tags.squareBracket, tags.brace], color: "var(--ink-3)" },
  { tag: tags.comment, color: "var(--ink-3)", fontStyle: "italic" },
]);

/** A view plugin that marks every match of `regexp` with `className`. */
function markMatches(regexp: RegExp, className: string) {
  const matcher = new MatchDecorator({ regexp, decoration: Decoration.mark({ class: className }) });
  return ViewPlugin.fromClass(
    class {
      decorations: DecorationSet;
      constructor(view: EditorView) {
        this.decorations = matcher.createDeco(view);
      }
      update(update: ViewUpdate) {
        this.decorations = matcher.updateDeco(update, this.decorations);
      }
    },
    { decorations: (plugin) => plugin.decorations },
  );
}

/**
 * The editor's one hue besides diagnostics: `free=True` in the free-parameter
 * blue (token `param`), the literal a drag or the optimiser may rewrite.
 */
const freeParameters = markMatches(/\bfree\s*=\s*True\b/g, "cm-free-param");

/**
 * Keyword-argument names (`density=`) recede to ink-3. The Python grammar
 * tags them as plain variables, so they are found by shape: a name glued to a
 * single `=` after `(`, `,` or indentation (an assignment has spaces).
 */
const keywordArguments = markMatches(/(?<=[(,]\s*|^\s+)([A-Za-z_]\w*)(?==(?!=))/gm, "cm-kwarg");

const theme = EditorView.theme(
  {
    "&": {
      height: "100%",
      ...textRole("body"),
      backgroundColor: "var(--surface-panel)",
    },
    ".cm-scroller": { font: "inherit" },
    ".cm-content": { caretColor: "var(--ink)" },
    ".cm-gutters": {
      backgroundColor: "var(--surface-panel)",
      border: "none",
      color: "var(--ink-3)",
    },
    ".cm-activeLine": { backgroundColor: "var(--surface-bar-alt)" },
    ".cm-kwarg, .cm-kwarg *": { color: "var(--ink-3)" },
    ".cm-free-param, .cm-free-param *": { color: "var(--param)" },
    ".cm-activeLineGutter": { backgroundColor: "transparent" },
    ".cm-vertex-highlight": {
      // The edited span: the accent as a veil behind the type, never an edge.
      backgroundColor: "rgba(var(--mode-accent-rgb, var(--accent-rgb)), var(--alpha-veil))",
      borderRadius: "var(--radius)",
    },
    "&.cm-focused": { outline: "none" },
  },
  // The pane is paper, like everything else in this build; CodeMirror uses
  // this only to decide which of its own defaults to reach for.
  { dark: false },
);

export interface EditorPaneProps {
  /** Character span to reveal and highlight, or null to clear. */
  highlight: FocusSpan | null;
  onRun: () => void;
  /** Collapse the pane to its slim rail (same state as Window → Editor). */
  onCollapse?: () => void;
}

export function EditorPane(props: EditorPaneProps) {
  let host!: HTMLDivElement;
  let view: EditorView | undefined;

  onMount(() => {
    const extensions: Extension[] = [
      lineNumbers(),
      history(),
      python(),
      syntaxHighlighting(highlightStyle),
      keywordArguments,
      freeParameters,
      highlightField,
      theme,
      // ruff diagnostics, jedi completion, and signature help — all three
      // read the program without running it (src/editor/extensions.ts).
      ...intelligenceExtensions(),
      keymap.of([
        {
          key: "Mod-Enter",
          preventDefault: true,
          run: () => {
            props.onRun();
            return true;
          },
        },
        indentWithTab,
        ...defaultKeymap,
        ...historyKeymap,
      ]),
      EditorView.updateListener.of((update) => {
        if (!update.docChanged) return;
        const text = update.state.doc.toString();
        // Guard against feeding our own programmatic updates back as edits.
        if (text !== source()) {
          setSource(text);
          setDirty(true);
        }
      }),
    ];

    view = new EditorView({
      state: EditorState.create({ doc: source(), extensions }),
      parent: host,
    });

    onCleanup(() => view?.destroy());
  });

  // Adopt source changes that came from elsewhere (session start, /patch),
  // as the smallest edit that produces them: a whole-document replacement
  // sends the view back to line 1, and a patch is a few characters in one
  // literal the user is looking at (see `minimalChange`).
  createEffect(() => {
    const text = source();
    if (!view) return;
    const change = minimalChange(view.state.doc.toString(), text);
    if (change) view.dispatch({ changes: change });
  });

  /**
   * Reveal and highlight whatever the viewport selected.
   *
   * Two judgements beyond "scroll there":
   *
   * - **A statement is revealed, not painted.** A precise span is the exact
   *   literal and is tinted whole; a declaration can run twenty-two lines —
   *   the starter's fin comb does — and tinting all of it puts a block over a
   *   fifth of the file. So a statement tints its first line only, which is
   *   the line that names the object, and the reveal still lands on it.
   * - **It never fights a cursor.** If the editor has focus the user is
   *   typing in it, and yanking the viewport out from under them is worse
   *   than not moving: the highlight still updates, the scroll does not.
   */
  createEffect(() => {
    const span = props.highlight;
    if (!view) return;
    const limit = view.state.doc.length;
    const valid = span && span.from >= 0 && span.to <= limit && span.from < span.to;
    const mark =
      valid && span
        ? span.precise
          ? span
          : { from: span.from, to: Math.min(span.to, view.state.doc.lineAt(span.from).to) }
        : null;
    view.dispatch({
      effects: [
        setHighlight.of(mark),
        ...(valid && span && !view.hasFocus
          ? [EditorView.scrollIntoView(span.from, { y: "center" })]
          : []),
      ],
    });
  });

  return (
    <section class="pane editor-pane">
      <header class="pane-head">
        <span class="pane-title">{sceneName() ?? "scene.py"}</span>
        <div class="editor-head-actions">
          <span class="pane-hint">Ctrl/⌘ + Enter to run</span>
          {props.onCollapse && (
            <button
              type="button"
              class="editor-collapse"
              onClick={() => props.onCollapse?.()}
              title="Collapse the editor"
              aria-label="Collapse the editor"
              data-testid="editor-collapse"
            >
              ⟨
            </button>
          )}
        </div>
      </header>
      <div class="editor-host" ref={host} data-testid="editor" />
      {consoleText() && <pre class="console">{consoleText()}</pre>}
    </section>
  );
}
