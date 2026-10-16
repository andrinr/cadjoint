/**
 * Write the generated token files from src/tokens.ts, the one source:
 *
 *   src/tokens.css            the stylesheet's :root custom properties
 *   ../docs/assets/tokens.scss the same values for the Quarto docs theme
 *   src/viewer/tokens.wgsl    the shader constants, for tools that compile the
 *                             .wgsl files outside the app (the Python tests);
 *                             the app itself prepends wgslTokens() at runtime
 *
 * Run with `npm run tokens` (Node strips the types). test/tokens.test.ts
 * fails when either file is out of date.
 */
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { tokensCss, tokensScss, wgslTokens } from "../src/tokens.ts";

const here = dirname(fileURLToPath(import.meta.url));
writeFileSync(join(here, "../src/tokens.css"), tokensCss());
writeFileSync(join(here, "../../docs/assets/tokens.scss"), tokensScss());
writeFileSync(join(here, "../src/viewer/tokens.wgsl"), wgslTokens());
console.log("wrote src/tokens.css, docs/assets/tokens.scss and src/viewer/tokens.wgsl");
