import { chromium } from "/Users/andrinrehnann/code/jaxcad/frontend/node_modules/@playwright/test/index.mjs";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";

const here = path.dirname(fileURLToPath(import.meta.url));
const jobs = JSON.parse(process.argv[2]);

const browser = await chromium.launch({
  args: ["--enable-unsafe-webgpu", "--enable-gpu", "--use-angle=metal", "--ignore-gpu-blocklist",
         "--force-color-profile=srgb", "--font-render-hinting=none"],
});
for (const j of jobs) {
  const page = await browser.newPage({
    viewport: { width: j.w, height: j.h },
    deviceScaleFactor: j.scale ?? 2,
  });
  await page.goto(pathToFileURL(path.join(here, j.file)).href + (j.query || ""));
  await page.waitForLoadState("networkidle");
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(250);
  const out = path.join(here, j.out);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  await page.screenshot({ path: out, clip: { x: 0, y: 0, width: j.w, height: j.h } });
  console.log(out, `${j.w * (j.scale ?? 2)}x${j.h * (j.scale ?? 2)}`);
  await page.close();
}
await browser.close();
