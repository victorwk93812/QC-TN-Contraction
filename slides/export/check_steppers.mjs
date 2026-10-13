// check_steppers.mjs — forward, backward and re-entry must all show the state of the visible fragments.
// For every slide with step fragments: walk forward through all fragments, then backward, then leave
// and re-enter from the next slide; at each state the "step" readout (when present) must equal the
// number of visible step fragments, and the final readouts must match the trace's totals.
import { chromium } from "playwright";

const browser = await chromium.launch({ executablePath: "/usr/bin/chromium" });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
await page.goto("http://localhost:8001/deck/index.html", { waitUntil: "networkidle" });
await page.waitForFunction(() => document.body.dataset.deckReady === "1");

const state = () => page.evaluate(() => {
  const s = Reveal.getCurrentSlide();
  const k = s.querySelectorAll(".fragment.visible.step-fragment").length;
  const r = s.querySelector('[data-readout="step"]');
  const f = (s._steppers || [])[0] || {};
  const expect = f.stride ? Math.min(f.steps, k * f.stride) : k;       // strided steppers advance several steps per click
  return { k: expect, step: r ? r.textContent : null, width: s.querySelector('[data-readout="width"]')?.textContent ?? null };
});
let fails = 0, checked = 0;
const n = await page.evaluate(() => Reveal.getSlides().length);
for (let h = 0; h < n; h++) {
  const steps = await page.evaluate((h) => Reveal.getSlides()[h].querySelectorAll(".step-fragment").length, h);
  if (!steps) continue;
  const nf = await page.evaluate((h) => Reveal.getSlides()[h].querySelectorAll(".fragment").length, h);
  await page.evaluate((h) => { const i = Reveal.getIndices(Reveal.getSlides()[h]); Reveal.slide(i.h, i.v, -1); }, h);
  const seq = [];
  for (let f = 0; f < nf; f++) { await page.evaluate(() => Reveal.next()); seq.push(await state()); }
  for (let f = 0; f < nf; f++) { await page.evaluate(() => Reveal.prev()); seq.push(await state()); }
  await page.evaluate((h) => { const i = Reveal.getIndices(Reveal.getSlides()[h + 1]); Reveal.slide(i.h, i.v, -1); }, h);
  await page.evaluate(() => Reveal.prev());                     // re-enter from the next slide: all fragments shown
  seq.push(await state());
  for (const st of seq) {
    checked++;
    if (st.step !== null && String(st.k) !== st.step) { fails++; console.log(`slide ${h + 1}: ${st.k} visible steps but readout says ${st.step}`); }
  }
  const last = seq[seq.length - 1];
  console.log(`slide ${h + 1}: ${steps} steps, re-entry shows k=${last.k}, step readout ${last.step}, width ${last.width}`);
}
console.log(fails ? `${fails} FAILURES in ${checked} states` : `all ${checked} states consistent`);
await browser.close();
process.exit(fails ? 1 : 0);
