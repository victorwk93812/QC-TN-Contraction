// count_states.mjs — expected PDF pages: one per fragment state of every slide (fragments + 1).
import { chromium } from "playwright";
const b = await chromium.launch({ executablePath: "/usr/bin/chromium" });
const p = await b.newPage({ viewport: { width: 1600, height: 900 } });
await p.goto("http://localhost:8001/deck/index.html?export", { waitUntil: "networkidle" });
await p.waitForFunction(() => document.body.dataset.deckReady === "1");
const r = await p.evaluate(() => Reveal.getSlides().map((s) => ({ id: s.id, f: s.querySelectorAll(".fragment").length })));
let page = 1; const firsts = {};
for (const s of r) { firsts[s.id] = page; page += s.f + 1; }
console.log(JSON.stringify({ slides: r.length, pages: page - 1, firsts }));
await b.close();
