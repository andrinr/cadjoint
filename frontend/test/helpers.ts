/** Fixtures shared by the unit tests. */

import type { KeyValueStorage } from "../src/storage";
import type { Rgb } from "../src/tokens";

/** A Map-backed storage; `data` is the map, for asserting on what was written. */
export function memoryStorage(
  initial: Readonly<Record<string, string>> = {},
): KeyValueStorage & { data: Map<string, string> } {
  const data = new Map(Object.entries(initial));
  return {
    data,
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => void data.set(key, value),
    removeItem: (key) => void data.delete(key),
  };
}

/** A storage that throws on every call, as private-mode and full-quota browsers do. */
export function failingStorage(): KeyValueStorage {
  return {
    getItem: () => {
      throw new Error("private mode");
    },
    setItem: () => {
      throw new Error("quota");
    },
    removeItem: () => {
      throw new Error("quota");
    },
  };
}

/** A promise the test resolves when it wants the stand-in work to answer. */
export function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}

/** Wait `ms` on the timer queue, so every already-resolved callback has run. */
export const settle = (ms = 0): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/** Euclidean distance between two colours in normalized RGB. */
export const distance = (a: Rgb, b: Rgb): number =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
