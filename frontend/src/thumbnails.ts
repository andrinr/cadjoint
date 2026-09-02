/**
 * Thumbnail storage for the scene browser: IndexedDB, keyed by source hash.
 *
 * A thumbnail costs a compile and a frame, which for the shipped gearbox
 * end-cap is several seconds of real work. Doing that on every visit to the
 * browser would be indefensible, and doing it in parallel for a directory of
 * scenes would be worse — so the pictures are cached, and the key is the
 * scene's `source_hash`. That is the same sha256 the server stamps on the
 * listing and on every job, which makes the cache *self-invalidating*: edit a
 * scene and its hash changes, so the old picture is simply never asked for
 * again rather than having to be noticed and thrown away.
 *
 * `localStorage` is the wrong home for this — a 320x200 PNG is tens of
 * kilobytes and the whole origin gets about five megabytes — so it is
 * IndexedDB, and every operation degrades to "no cache" rather than to an
 * error: a private window, a disabled store, or a browser that refuses the
 * upgrade all leave the browser working, just slower.
 */

/** Bumped when the picture itself changes meaning (framing, size, style). */
export const THUMBNAIL_DB = "cadjoint.thumbnails";
export const THUMBNAIL_STORE = "thumbnails";
export const THUMBNAIL_DB_VERSION = 1;

/** The offscreen canvas the browser draws into, in device-independent px. */
export const THUMBNAIL_WIDTH = 320;
export const THUMBNAIL_HEIGHT = 200;

/** One cached picture: a data URL, and when it was made. */
export interface ThumbnailRecord {
  source_hash: string;
  data_url: string;
  made_at: number;
}

let handle: Promise<IDBDatabase | null> | undefined;

/** Open (or reuse) the store; resolves to null wherever IndexedDB is not. */
export function openThumbnailStore(): Promise<IDBDatabase | null> {
  if (handle) return handle;
  handle = new Promise<IDBDatabase | null>((resolve) => {
    if (typeof indexedDB === "undefined") {
      resolve(null);
      return;
    }
    let request: IDBOpenDBRequest;
    try {
      request = indexedDB.open(THUMBNAIL_DB, THUMBNAIL_DB_VERSION);
    } catch {
      resolve(null);
      return;
    }
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(THUMBNAIL_STORE)) {
        db.createObjectStore(THUMBNAIL_STORE, { keyPath: "source_hash" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(null);
    request.onblocked = () => resolve(null);
  });
  return handle;
}

/** The cached picture for a source hash, or null when there is none. */
export async function readThumbnail(hash: string | null): Promise<string | null> {
  if (!hash) return null;
  const db = await openThumbnailStore();
  if (!db) return null;
  return new Promise<string | null>((resolve) => {
    try {
      const request = db
        .transaction(THUMBNAIL_STORE, "readonly")
        .objectStore(THUMBNAIL_STORE)
        .get(hash);
      request.onsuccess = () => {
        const record = request.result as ThumbnailRecord | undefined;
        resolve(record?.data_url ?? null);
      };
      request.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

/** Store one picture. A failure is silent: the cache is an optimisation. */
export async function writeThumbnail(hash: string, dataUrl: string): Promise<void> {
  const db = await openThumbnailStore();
  if (!db) return;
  await new Promise<void>((resolve) => {
    try {
      const transaction = db.transaction(THUMBNAIL_STORE, "readwrite");
      transaction.objectStore(THUMBNAIL_STORE).put({
        source_hash: hash,
        data_url: dataUrl,
        made_at: Date.now(),
      } satisfies ThumbnailRecord);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => resolve();
      transaction.onabort = () => resolve();
    } catch {
      resolve();
    }
  });
}

/**
 * Run jobs strictly one at a time, in the order they were queued.
 *
 * Thumbnails are the reason this exists: each one holds a compile worker on
 * the server and a GPU frame in the browser, so rendering a directory of them
 * at once would make the browser fight the panel the user is trying to read.
 * One at a time is slower in total and far better to sit in front of — the
 * first card fills in immediately and the rest arrive while you read it.
 */
export function createQueue(): {
  push: <T>(job: () => Promise<T>) => Promise<T>;
  readonly pending: () => number;
} {
  let tail: Promise<unknown> = Promise.resolve();
  let waiting = 0;
  return {
    push<T>(job: () => Promise<T>): Promise<T> {
      waiting += 1;
      const queued = tail.then(job, job).finally(() => {
        waiting -= 1;
      });
      tail = queued.catch(() => undefined);
      return queued;
    },
    pending: () => waiting,
  };
}

/** The first line of a compile error, which is the line worth showing. */
export function firstLine(message: string | null | undefined): string {
  if (!message) return "This scene did not compile.";
  const line = message.trim().split("\n").find((entry) => entry.trim().length > 0);
  return line ? line.trim() : "This scene did not compile.";
}

/** "4 Mar 2026" — the listing's ISO stamp, in the reader's locale. */
export function formatModified(iso: string | null | undefined): string {
  if (!iso) return "–";
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return "–";
  return when.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

/** "12 kB", "1.4 MB" — the same scale the process monitor uses. */
export function formatFileSize(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return "–";
  if (bytes < 1_000) return `${Math.round(bytes)} B`;
  const value = bytes / 1_000;
  return value >= 1_000 ? `${(value / 1_000).toFixed(1)} MB` : `${value.toFixed(1)} kB`;
}
