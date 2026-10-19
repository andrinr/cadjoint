/**
 * Evaluate a scene graph at points on jax-js (Wasm), for the Python tests.
 *
 * Usage: npx tsx eval.mts IN.json [--per-node]
 * IN.json: {"graph": <scene graph>, "points": [x0, y0, z0, x1, ...]}
 * Prints {"values": [...]} as JSON on stdout.
 */
import { init, defaultDevice, numpy as np } from "@jax-js/jax";
import { field, fieldPerNode, sceneRoot } from "./src/runtime.ts";
import { readFileSync } from "node:fs";

const input = JSON.parse(readFileSync(process.argv[2], "utf8"));
await init("wasm");
defaultDevice("wasm");
const scene = sceneRoot(input.graph);
const f = process.argv.includes("--per-node") ? fieldPerNode(scene) : field(scene);
const flat: number[] = input.points;
const n = flat.length / 3;
const col = (k: number) => np.array(Float32Array.from({ length: n }, (_, i) => flat[3 * i + k]));
const values = f({ x: col(0), y: col(1), z: col(2) }).dataSync() as Float32Array;
process.stdout.write(JSON.stringify({ values: Array.from(values) }));
