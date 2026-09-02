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
} from "@codemirror/view";
import { createEffect, onCleanup, onMount } from "solid-js";
import { intelligenceExtensions } from "../editor/extensions";
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
 * The mapping is deliberate rather than decorative. There is one accent in
 * this design and its only legal use is a fill behind near-black type, so
 * syntax cannot be painted with it: what is left is the status family, and it
 * is spent on the distinction the reader actually needs while editing a scene
 * — is this the language, or is it a value? Keywords take --info because they
 * are the language's own vocabulary, literals take --ok and --warn because
 * they are the numbers and strings the viewport tools rewrite, names that
 * denote a type take --danger, and comments drop to the muted ink every
 * de-emphasised label in the app uses.
 */
const highlightStyle = HighlightStyle.define([
  { tag: tags.comment, color: "var(--ink-3)", fontStyle: "italic" },
  { tag: tags.keyword, color: "var(--info)" },
  { tag: [tags.controlKeyword, tags.moduleKeyword], color: "var(--info-ink)" },
  { tag: [tags.string, tags.special(tags.string)], color: "var(--ok)" },
  { tag: [tags.number, tags.bool, tags.null], color: "var(--warn)" },
  { tag: [tags.className, tags.typeName, tags.namespace], color: "var(--danger)" },
  { tag: tags.function(tags.variableName), color: "var(--ink)" },
  { tag: tags.definition(tags.variableName), color: "var(--ink)" },
  { tag: tags.propertyName, color: "var(--ink-2)" },
  { tag: [tags.operator, tags.punctuation, tags.separator], color: "var(--ink-2)" },
  { tag: tags.self, color: "var(--danger)", fontStyle: "italic" },
]);

const theme = EditorView.theme(
  {
    "&": {
      height: "100%",
      fontSize: "var(--text-md)",
      backgroundColor: "var(--surface-panel)",
    },
    ".cm-scroller": { fontFamily: "var(--font-mono)", lineHeight: "var(--leading-normal)" },
    ".cm-content": { caretColor: "var(--mode-accent, var(--accent))" },
    ".cm-gutters": {
      backgroundColor: "var(--surface-panel)",
      border: "none",
      color: "var(--ink-3)",
    },
    ".cm-activeLine": { backgroundColor: "var(--surface-inset)" },
    ".cm-activeLineGutter": { backgroundColor: "transparent" },
    ".cm-vertex-highlight": {
      backgroundColor: "rgba(var(--mode-accent-rgb, var(--accent-rgb)), var(--alpha-veil))",
      outline: "1px solid rgba(var(--mode-accent-rgb, var(--accent-rgb)), var(--alpha-mark))",
      borderRadius: "var(--radius)",
    },
    "&.cm-focused": { outline: "none" },
  },
  // The pane is paper, like everything else in this build; CodeMirror uses
  // this only to decide which of its own defaults to reach for.
  { dark: false },
);

export interface EditorPaneProps {
  /** Character span to highlight, or null to clear. */
  highlight: { from: number; to: number } | null;
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

  // Adopt source changes that came from elsewhere (session start, /patch).
  createEffect(() => {
    const text = source();
    if (!view || view.state.doc.toString() === text) return;
    view.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: text },
    });
  });

  createEffect(() => {
    const span = props.highlight;
    if (!view) return;
    const limit = view.state.doc.length;
    const valid = span && span.from >= 0 && span.to <= limit && span.from < span.to;
    view.dispatch({
      effects: [
        setHighlight.of(valid ? span : null),
        ...(valid ? [EditorView.scrollIntoView(span.from, { y: "center" })] : []),
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
