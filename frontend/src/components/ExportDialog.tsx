/**
 * File → Export…: one object of the program, as a file.
 *
 * A small form and nothing else: the format, what to export, how fine, the
 * one option the format has, and Export. The decisions are in `export.ts`
 * (what each format takes, which names to offer, how the body is built),
 * the request is `api.exportFile`, and the download is the browser's own —
 * a blob URL on an anchor, exactly as "Download scene.py" does it.
 *
 * The run is a job like any other, so while it works the chip beside the
 * mode switcher counts the seconds and can cancel it; the dialog only says
 * that it is working, and disables the button so the same file is not
 * requested twice.
 */

import { For, Show, createMemo, createSignal } from "solid-js";
import * as api from "../api";
import {
  EXPORT_FORMATS,
  EXPORT_RESOLUTION,
  clampResolution,
  defaultExportName,
  downloadName,
  errorSummary,
  exportRequest,
  exportTargets,
  formatInfo,
} from "../export";
import { formatBytes } from "../jobs";
import { sceneName, setStatus, source, studies } from "../state";
import type { ExportFormat } from "../types";
import { ToggleSwitch } from "./ui";
import "./export.css";

export interface ExportDialogProps {
  onClose: () => void;
}

/** Hand the browser a file to save, the way the scene download does. */
function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  // Revoked on the next tick: revoking synchronously races the click in
  // some browsers and the download arrives empty.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function ExportDialog(props: ExportDialogProps) {
  const studyNames = createMemo(() => studies().map((study) => study.name));

  const [format, setFormat] = createSignal<ExportFormat>("stl");
  const [name, setName] = createSignal(defaultExportName("stl", studyNames()));
  const [resolution, setResolution] = createSignal<number>(EXPORT_RESOLUTION.default);
  const [binary, setBinary] = createSignal(true);
  const [analytic, setAnalytic] = createSignal(true);
  const [mergePlanar, setMergePlanar] = createSignal(true);
  const [busy, setBusy] = createSignal(false);
  const [error, setError] = createSignal("");

  const info = () => formatInfo(format());
  const targets = createMemo(() => exportTargets(format(), source(), studyNames()));
  /** A study format with nothing to write is the one case the form refuses. */
  const nothingToExport = () => info().takes === "study" && targets().length === 0;

  const pickFormat = (next: ExportFormat) => {
    const before = info().takes;
    setFormat(next);
    setError("");
    // Switching between an object format and a study format changes what
    // the name means; a name that was an object cannot be a study.
    if (formatInfo(next).takes !== before) setName(defaultExportName(next, studyNames()));
  };

  const run = async () => {
    if (busy() || nothingToExport()) return;
    const body = exportRequest(source(), {
      format: format(),
      name: name(),
      resolution: resolution(),
      binary: binary(),
      analytic: analytic(),
      mergePlanar: mergePlanar(),
    });
    if (!body.name) {
      setError("Name the object to export.");
      return;
    }
    setBusy(true);
    setError("");
    setStatus({ kind: "busy", text: `Exporting ${body.name} as ${info().label}…` });
    try {
      const outcome = await api.exportFile(body);
      if (!outcome.ok) {
        const message = errorSummary(outcome.error);
        setError(message);
        setStatus({ kind: "error", text: message });
        return;
      }
      const filename = downloadName(outcome.filename, sceneName());
      saveBlob(outcome.blob, filename);
      setStatus({ kind: "ready", text: `Exported ${filename} · ${formatBytes(outcome.blob.size)}` });
      props.onClose();
    } catch (failure) {
      const message = failure instanceof Error ? failure.message : String(failure);
      setError(message);
      setStatus({ kind: "error", text: message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div class="dialog-backdrop" onClick={() => !busy() && props.onClose()}>
      <div
        class="dialog dialog-compact"
        role="dialog"
        aria-label="Export"
        onClick={(event) => event.stopPropagation()}
        data-testid="export-dialog"
      >
        <header>
          <span>Export</span>
          <button type="button" onClick={props.onClose} disabled={busy()}>
            Close
          </button>
        </header>
        <form
          class="dialog-body export-form"
          onSubmit={(event) => {
            event.preventDefault();
            void run();
          }}
        >
          <div class="export-row">
            <label class="export-field">
              Format
              <select
                value={format()}
                onChange={(event) => pickFormat(event.currentTarget.value as ExportFormat)}
                disabled={busy()}
                data-testid="export-format"
              >
                <For each={EXPORT_FORMATS}>
                  {(entry) => <option value={entry.value}>{entry.label}</option>}
                </For>
              </select>
            </label>
            <div class="export-option">
              <Show when={info().option}>
                {(option) => (
                  <ToggleSwitch
                    checked={
                      option().key === "binary"
                        ? binary()
                        : option().key === "analytic"
                          ? analytic()
                          : mergePlanar()
                    }
                    onChange={(checked) => {
                      if (option().key === "binary") setBinary(checked);
                      else if (option().key === "analytic") setAnalytic(checked);
                      else setMergePlanar(checked);
                    }}
                    disabled={busy()}
                    testId={`export-option-${option().key}`}
                  >
                    {option().label}
                  </ToggleSwitch>
                )}
              </Show>
            </div>
          </div>

          <div class="export-row">
            <label class="export-field">
              {info().takes === "study" ? "Study" : "Object"}
              <Show
                when={info().takes === "object"}
                fallback={
                  <select
                    value={name()}
                    onChange={(event) => setName(event.currentTarget.value)}
                    disabled={busy() || nothingToExport()}
                    data-testid="export-name"
                  >
                    <For each={targets()}>{(entry) => <option value={entry}>{entry}</option>}</For>
                  </select>
                }
              >
                <input
                  type="text"
                  list="export-objects"
                  value={name()}
                  onInput={(event) => {
                    setName(event.currentTarget.value);
                    setError("");
                  }}
                  disabled={busy()}
                  spellcheck={false}
                  autocomplete="off"
                  data-testid="export-name"
                />
                <datalist id="export-objects">
                  <For each={targets()}>{(entry) => <option value={entry} />}</For>
                </datalist>
              </Show>
            </label>
            <Show when={info().takes === "object"}>
              <label class="export-field">
                <span>
                  Resolution <span class="export-unit">cells</span>
                </span>
                <input
                  type="number"
                  min={EXPORT_RESOLUTION.min}
                  max={EXPORT_RESOLUTION.max}
                  step="8"
                  value={resolution()}
                  onChange={(event) => {
                    const value = Number(event.currentTarget.value);
                    if (Number.isFinite(value)) setResolution(clampResolution(value));
                  }}
                  disabled={busy()}
                  data-testid="export-resolution"
                />
              </label>
            </Show>
          </div>

          <p class="dialog-note" data-testid="export-note">
            {nothingToExport()
              ? "The program declares no study to export a result from."
              : info().note}
          </p>

          <Show when={error()}>
            <p class="dialog-error" data-testid="export-error">
              {error()}
            </p>
          </Show>

          <div class="export-actions">
            <button
              type="submit"
              class="primary"
              disabled={busy() || nothingToExport()}
              data-testid="export-confirm"
            >
              {busy() ? "Exporting…" : "Export"}
            </button>
            <Show when={busy()}>
              <p class="export-progress" data-testid="export-progress">
                Extracting {name()} at {resolution()} cells — the chip beside the mode switcher
                can cancel it.
              </p>
            </Show>
          </div>
        </form>
      </div>
    </div>
  );
}
