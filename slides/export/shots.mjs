// shots.mjs — step through the deck and save one PNG per fragment state, for review by eye.
//   node export/shots.mjs                 all slides, every fragment state
//   node export/shots.mjs 2 5 --last      slides 2..5 (1-based), first and last state only
// Requires the deck to be served:  python3 tools/serve.py 8001  (from slides/)
import { chromium } from "playwright";
import fs from "node:fs";

const args = process.argv.slice(2);
const lastOnly = args.includes("--last");
const nums = args.filter((a) => /^\d+$/.test(a)).map(Number);
const from = nums[0] || 1, to = nums[1] || nums[0] || 9999;
const out = "export/out/shots";
fs.mkdirSync(out, { recursive: true });

const browser = await chromium.launch({ executablePath: "/usr/bin/chromium" });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push(String(e)));
await page.goto("http://localhost:8001/deck/index.html", { waitUntil: "networkidle" });
await page.waitForFunction(() => document.body.dataset.deckReady === "1", null, { timeout: 30000 });
await page.waitForTimeout(500);

const n = await page.evaluate(() => Reveal.getSlides().length);
const report = [];
for (let h = from - 1; h < Math.min(n, to); h++) {
  const nf = await page.evaluate((h) => Reveal.getSlides()[h].querySelectorAll(".fragment").length, h);
  const states = lastOnly ? [...new Set([-1, nf - 1])] : Array.from({ length: nf + 1 }, (_, i) => i - 1);
  for (const f of states) {
    await page.evaluate(([h, f]) => { const i = Reveal.getIndices(Reveal.getSlides()[h]); Reveal.slide(i.h, i.v, f); }, [h, f]);
    await page.waitForTimeout(250);
    const file = `${out}/s${String(h + 1).padStart(2, "0")}-f${String(f + 1).padStart(2, "0")}.png`;
    await page.screenshot({ path: file });
    report.push(file);
  }
  // overflow check: content taller than the slide or wider than it
  const ov = await page.evaluate((h) => {
    const s = Reveal.getSlides()[h];
    return { h: s.scrollHeight, w: s.scrollWidth, missing: s.querySelectorAll(".bind-missing").length };
  }, h);
  if (ov.h > 900 * 0.95 || ov.w > 1600 || ov.missing) console.log(`slide ${h + 1}: height ${ov.h}, width ${ov.w}, missing binds ${ov.missing}`);
}
console.log(`${report.length} screenshots in ${out}`);
if (errors.length) console.log("console errors:\n" + [...new Set(errors)].join("\n"));
await browser.close();
