/**
 * The application menu bar: File, Edit, View, Create, Shape, Constrain,
 * Simulate, Program, Window, Help.
 *
 * Nothing here lists a command by hand. The menus are `shell/menuModel.ts`,
 * which is the design page's agreed menu bar (`design/features.ts`) with each
 * leaf bound to a command; each item runs that command through the registry
 * (`shell/commands.ts`) — the very call its key, rail button or panel button
 * makes — and quotes the key `shortcuts.ts` binds to it.
 *
 * An item that does not apply in the current mode is shown disabled with the
 * reason in its title ("Only in sketch mode"); so is one that cannot run on
 * the current selection. Menus follow the native pattern — click (or
 * ArrowDown) opens, arrows move, Escape or a click elsewhere closes.
 *
 * The menu bar owns the dialogs its File and Help items open (Save As…,
 * Export…, Keyboard shortcuts) and registers those commands itself.
 */

import "./shell.css";
import { For, Show, createSignal, onCleanup } from "solid-js";
import * as api from "../api";
import { sanitizeSceneName } from "../scenes";
import { SHORTCUT_GROUPS } from "../shortcuts";
import { dirty, editingMode, sceneName, setSceneName, setStatus, source } from "../state";
import { command, commandBlocked, registerCommands, runCommand } from "../shell/commands";
import {
  MENU_IDS,
  menuModel,
  modeReason,
  windowOfCommand,
  type MenuEntry,
} from "../shell/menuModel";
import { windowManager } from "../windows/manager";
import { ExportDialog } from "./ExportDialog";

type DialogId = "saveAs" | "export" | "help";

const REPO_README_URL = "https://github.com/andrinr/cadjoint#readme";
/** The design language: the second page of this app. */
const DESIGN_URL = "/design.html";

/** Items that are links rather than commands, so they open like links. */
const LINKS: Record<string, string> = {
  "help-readme": REPO_README_URL,
  "help-design": DESIGN_URL,
};

const MENUS = menuModel();

export function MenuBar() {
  const [openMenu, setOpenMenu] = createSignal<string | null>(null);
  const [dialog, setDialog] = createSignal<DialogId | null>(null);
  const [saveAsName, setSaveAsName] = createSignal("");
  const [saveAsError, setSaveAsError] = createSignal("");

  const closeOnOutside = (event: MouseEvent) => {
    if (!(event.target as HTMLElement).closest(".menu-bar")) setOpenMenu(null);
  };
  document.addEventListener("click", closeOnOutside);
  onCleanup(() => document.removeEventListener("click", closeOnOutside));

  // Escape closes an open dialog (or a menu opened without keyboard focus).
  // Capture phase, so the viewer's global Escape handling (clear selection,
  // reset editing mode) does not also fire behind the dialog.
  const closeOnEscape = (event: KeyboardEvent) => {
    if (event.key !== "Escape") return;
    if (dialog() !== null) {
      event.stopPropagation();
      setDialog(null);
    } else if (openMenu() !== null) {
      event.stopPropagation();
      // Keep the keyboard flow: focus returns to the menu's title button.
      document.querySelector<HTMLElement>(".menu-title.active")?.focus();
      setOpenMenu(null);
    }
  };
  document.addEventListener("keydown", closeOnEscape, true);
  onCleanup(() => document.removeEventListener("keydown", closeOnEscape, true));

  const toggleMenu = (id: string) => setOpenMenu(openMenu() === id ? null : id);
  /** An open menubar behaves like hover navigation between its menus. */
  const glideTo = (id: string) => {
    if (openMenu() !== null && openMenu() !== id) setOpenMenu(id);
  };

  /**
   * Open… is the scene browser now, not a list of file names.
   *
   * A modal list of names could only answer "which file", and the question a
   * user actually has in front of a directory of parts is "which *part*".
   * The Scenes window answers that — a picture, the docstring, what each file
   * declares — so Open… raises it rather than opening a smaller, worse
   * version of it in a dialog. It is parked in every desk, so this is
   * usually one click and no layout change.
   */
  const showScenes = () => windowManager()?.open("scenes");

  const saveAs = async (rawName: string): Promise<void> => {
    const name = sanitizeSceneName(rawName);
    if (name === null) {
      setSaveAsError("Use a plain file name such as bracket.py — no folders.");
      return;
    }
    try {
      const result = await api.saveScene(name, source());
      if (!result.ok) {
        const message = result.error ?? `Could not save ${name}.`;
        setSaveAsError(message);
        setStatus({ kind: "error", text: message });
        return;
      }
      setSceneName(name);
      setDialog(null);
      setStatus({ kind: "ready", text: `Saved ${name}` });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSaveAsError(message);
      setStatus({ kind: "error", text: message });
    }
  };

  const showSaveAsDialog = () => {
    setSaveAsName(sceneName() ?? "scene.py");
    setSaveAsError("");
    setDialog("saveAs");
  };

  const save = () => {
    const name = sceneName();
    if (name === null) {
      showSaveAsDialog();
      return;
    }
    void saveAs(name);
  };

  const download = () => {
    const blob = new Blob([source()], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = sceneName() ?? "scene.py";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  /**
   * Export… is a dialog, not a window: it asks four things and produces one
   * file, and a form that is answered and dismissed has no business in the
   * dock. The work it starts is a job, so it is watched — and cancelled —
   * from the chip and the Processes window like a solve.
   */
  onCleanup(
    registerCommands({
      "file-open": { run: showScenes },
      "file-save": { run: save },
      "file-save-as": { run: showSaveAsDialog },
      "file-export": { run: () => setDialog("export") },
      "file-download": { run: download },
      "help-shortcuts": {
        run: () => setDialog(dialog() === "help" ? null : "help"),
      },
      "help-readme": { run: () => void window.open(REPO_README_URL, "_blank", "noreferrer") },
    }),
  );

  /** Why an item is off right now — its mode first, then the command's own. */
  const reasonFor = (entry: MenuEntry): string | null =>
    modeReason(entry.modes, editingMode()) ?? commandBlocked(entry.command);

  /** Arrow-key navigation between the items of an open dropdown. */
  const onMenuKeyDown = (event: KeyboardEvent) => {
    const menu = (event.currentTarget as HTMLElement).closest(".menu");
    if (!menu) return;
    const items = [
      ...menu.querySelectorAll<HTMLElement>(
        ".menu-dropdown button:not([aria-disabled=true]), .menu-dropdown a",
      ),
    ];
    const index = items.indexOf(document.activeElement as HTMLElement);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      (items[(index + 1) % items.length] ?? items[0])?.focus();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      (items[(index - 1 + items.length) % items.length] ?? items.at(-1))?.focus();
    } else if (event.key === "Escape") {
      event.preventDefault();
      setOpenMenu(null);
      menu.querySelector<HTMLElement>(".menu-title")?.focus();
    } else if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      const active = openMenu();
      if (!active) return;
      event.preventDefault();
      const step = event.key === "ArrowRight" ? 1 : -1;
      const next =
        MENU_IDS[(MENU_IDS.indexOf(active) + step + MENU_IDS.length) % MENU_IDS.length];
      setOpenMenu(next);
      document.querySelector<HTMLElement>(`[data-testid=menu-${next}]`)?.focus();
    }
  };

  const menuTitle = (id: string, label: string) => (
    <button
      type="button"
      class="menu-title"
      classList={{ active: openMenu() === id }}
      aria-haspopup="menu"
      aria-expanded={openMenu() === id}
      onClick={() => toggleMenu(id)}
      onPointerEnter={() => glideTo(id)}
      onKeyDown={(event) => {
        if (event.key === "ArrowDown" && openMenu() !== id) {
          event.preventDefault();
          setOpenMenu(id);
        } else {
          onMenuKeyDown(event);
        }
      }}
      data-testid={`menu-${id}`}
    >
      {label}
    </button>
  );

  /** One item: the mark column, the label, the key hung right. */
  const entryView = (entry: MenuEntry) => {
    const impl = () => command(entry.command);
    const checked = () => impl()?.checked?.() ?? false;
    const reason = () => reasonFor(entry);
    const link = LINKS[entry.command];
    if (link) {
      return (
        <a
          href={link}
          target="_blank"
          rel="noreferrer"
          role="menuitem"
          onClick={() => setOpenMenu(null)}
          onKeyDown={onMenuKeyDown}
          data-testid={entry.testid}
        >
          <i class="menu-check" />
          <span>{entry.label}</span>
          <small>↗</small>
        </a>
      );
    }
    // Window visibility leaves the menu open, so the way back from closing
    // one is the same item; everything else is done and dismisses it.
    const staysOpen = windowOfCommand(entry.command) !== null;
    return (
      <button
        type="button"
        role={entry.radio ? "menuitemradio" : impl()?.checked ? "menuitemcheckbox" : "menuitem"}
        aria-checked={impl()?.checked || entry.radio ? checked() : undefined}
        aria-disabled={reason() !== null}
        title={reason() ?? undefined}
        onClick={() => {
          if (reason() !== null) return;
          if (!staysOpen) setOpenMenu(null);
          runCommand(entry.command);
        }}
        onKeyDown={onMenuKeyDown}
        data-testid={entry.testid}
      >
        <i class="menu-check">{checked() ? (entry.radio ? "•" : "✓") : ""}</i>
        <span>{entry.label}</span>
        <Show when={entry.key}>
          <small>{entry.key}</small>
        </Show>
      </button>
    );
  };

  return (
    <>
      <nav class="menu-bar" aria-label="Application menu" data-testid="menu-bar">
        <For each={MENUS}>
          {(menu) => (
            <div class="menu">
              {menuTitle(menu.id, menu.name)}
              <Show when={openMenu() === menu.id}>
                <div class="menu-dropdown" role="menu" aria-label={menu.name}>
                  <For each={menu.groups}>
                    {(group, index) => (
                      <>
                        <Show when={index() > 0}>
                          <hr />
                        </Show>
                        <Show when={group.title}>
                          <div class="menu-section" role="presentation">
                            {group.title}
                          </div>
                        </Show>
                        <For each={group.entries}>{entryView}</For>
                      </>
                    )}
                  </For>
                </div>
              </Show>
            </div>
          )}
        </For>

        <span class="menu-scene" data-testid="menu-scene-name">
          {sceneName() ?? "unsaved scene"}
          {dirty() ? " •" : ""}
        </span>
      </nav>

      <Show when={dialog() === "saveAs"}>
        <div class="dialog-backdrop" onClick={() => setDialog(null)}>
          <div
            class="dialog dialog-compact"
            role="dialog"
            aria-label="Save the scene as"
            onClick={(event) => event.stopPropagation()}
          >
            <header>
              <span>Save scene as</span>
              <button type="button" onClick={() => setDialog(null)}>
                Close
              </button>
            </header>
            <form
              class="dialog-body save-as-form"
              onSubmit={(event) => {
                event.preventDefault();
                void saveAs(saveAsName());
              }}
            >
              <label>
                File name
                <input
                  type="text"
                  value={saveAsName()}
                  onInput={(event) => {
                    setSaveAsName(event.currentTarget.value);
                    setSaveAsError("");
                  }}
                  data-testid="save-as-name"
                />
              </label>
              <Show when={saveAsError()}>
                <p class="dialog-error" data-testid="save-as-error">
                  {saveAsError()}
                </p>
              </Show>
              <p class="dialog-note">
                Written to <code>scenes/</code> in the playground server's
                working directory.
              </p>
              <button type="submit" class="primary" data-testid="save-as-confirm">
                Save
              </button>
            </form>
          </div>
        </div>
      </Show>

      <Show when={dialog() === "export"}>
        <ExportDialog onClose={() => setDialog(null)} />
      </Show>

      <Show when={dialog() === "help"}>
        <div class="dialog-backdrop" onClick={() => setDialog(null)}>
          <div
            class="dialog dialog-compact"
            role="dialog"
            aria-label="Keyboard shortcuts"
            onClick={(event) => event.stopPropagation()}
          >
            <header>
              <span>Keyboard shortcuts</span>
              <button type="button" onClick={() => setDialog(null)}>
                Close
              </button>
            </header>
            <div class="dialog-body shortcut-groups" data-testid="shortcut-dialog">
              <For each={SHORTCUT_GROUPS}>
                {(group) => (
                  <section>
                    <h4>{group.title}</h4>
                    <ul>
                      <For each={group.items}>
                        {(shortcut) => (
                          <li>
                            <span>{shortcut.action}</span>
                            <kbd>{shortcut.keys}</kbd>
                          </li>
                        )}
                      </For>
                    </ul>
                  </section>
                )}
              </For>
            </div>
          </div>
        </div>
      </Show>
    </>
  );
}
