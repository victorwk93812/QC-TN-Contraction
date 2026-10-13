/* deck.js — bootstrap: load traces, bind numbers, build figures, wire step fragments, start Reveal.
 *
 * Numbers:  <span data-bind="F10:tree.cost"></span>        value read from data/F10.json
 *           <span data-bind="F03:table.0.cost" data-fmt="int"></span>
 * Figures:  <div data-fig="stepper|tree|graph|chart|…" data-trace="F06" …options…></div>
 *           A figure whose builder returns {length > 0} is a stepping figure: one hidden
 *           .step-fragment per step is inserted into its section (at .steps-here if present,
 *           otherwise before the first .math-anchor, otherwise at the end).
 * render(k) is pure, so forward, backward, re-entry and PDF export all redraw from the trace.
 */
(function () {
  "use strict";
  const DATA = "../data/";
  const cache = {};
  window.Figs = window.Figs || {};

  const get = (obj, path) => path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj);

  async function trace(id) {
    if (!(id in cache)) cache[id] = fetch(`${DATA}${id}.json`).then((r) => {
      if (!r.ok) throw new Error(`missing data/${id}.json`);
      return r.json();
    });
    return cache[id];
  }

  const FMT = {
    raw: (v) => String(v),
    int: (v) => Number(v).toLocaleString("en-US"),
    fixed1: (v) => Number(v).toFixed(1),
    fixed2: (v) => Number(v).toFixed(2),
    pct: (v) => `${Number(v).toFixed(0)}%`,
    pow2: (v) => Math.pow(2, Number(v)).toLocaleString("en-US"),
    list: (v) => (Array.isArray(v) ? v.join(", ") : String(v)),
    set: (v) => `{${(Array.isArray(v) ? v : [v]).join(", ")}}`,
    sci: (v) => {
      const e = Math.floor(Math.log10(Number(v)));
      return `${(Number(v) / Math.pow(10, e)).toFixed(1)}·10^${e}`;
    },
  };
  window.DeckFmt = FMT;

  async function bindAll(root) {
    for (const el of root.querySelectorAll("[data-bind]")) {
      const [id, path] = el.dataset.bind.split(":");
      const v = get(await trace(id), path);
      if (v === undefined) {
        el.textContent = "??";
        el.classList.add("bind-missing");
        console.error(`data-bind ${el.dataset.bind}: no such value`);
        continue;
      }
      el.textContent = (FMT[el.dataset.fmt || "raw"] || FMT.raw)(v);
    }
  }

  function insertSteps(section, n, fig, start = 0) {
    const anchor = section.querySelector(".steps-here") || section.querySelector(".math-anchor.fragment");
    for (let i = start + 1; i <= n; i++) {
      const f = document.createElement("span");
      f.className = "fragment step-fragment";
      f.dataset.step = String(i);
      if (anchor) anchor.parentNode.insertBefore(f, anchor);
      else section.appendChild(f);
    }
    section._steppers = section._steppers || [];
    section._steppers.push(fig);
  }

  async function buildFigs(root) {
    for (const el of root.querySelectorAll("[data-fig]")) {
      const make = window.Figs[el.dataset.fig];
      if (!make) { console.error(`unknown figure type ${el.dataset.fig}`); continue; }
      const t = el.dataset.trace ? await trace(el.dataset.trace) : null;
      const fig = make(el, t, el.dataset);
      el._fig = fig;
      const section = el.closest("section");
      if (!fig || !(fig.length > 0) || el.dataset.static !== undefined) continue;
      if (el.dataset.sharedSteps !== undefined) {
        // steppers that advance together: the section gets max(lengths) fragments, not the sum
        const have = section._sharedSteps || 0;
        insertSteps(section, Math.max(have, fig.length), fig, have);
        section._sharedSteps = Math.max(have, fig.length);
      } else insertSteps(section, fig.length, fig);
    }
  }

  function syncSlide(slide) {
    if (!slide) return;
    const k = slide.querySelectorAll(".fragment.visible.step-fragment").length;
    (slide._steppers || []).forEach((fig) => fig.render(k));
    // elements tied to a step: visible from step `data-show-at` on (e.g. table rows beside a stepper)
    slide.querySelectorAll("[data-show-at]").forEach((e) => e.classList.toggle("shown", k >= +e.dataset.showAt));
  }

  async function main() {
    const root = document.querySelector(".reveal .slides");
    // ?export (PDF export): clips become their poster frame, so each clip page shows the clip's final state
    if (new URLSearchParams(location.search).has("export")) {
      root.querySelectorAll("video.clip").forEach((v) => {
        const img = document.createElement("img");
        img.src = v.getAttribute("poster");
        img.className = "clip";
        v.replaceWith(img);
      });
    }
    await bindAll(root);
    await buildFigs(root);
    await Reveal.initialize({
      hash: true, width: 1600, height: 900, margin: 0.04, center: false,
      transition: "fade", transitionSpeed: "fast", backgroundTransition: "none",
      slideNumber: "h.v", showNotes: false, pdfSeparateFragments: true, pdfMaxPagesPerSlide: 200,
      katex: {
        local: "vendor/katex", trust: true,
        delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
        macros: {
          "\\tw": "\\operatorname{tw}", "\\cw": "\\operatorname{cw}", "\\fillin": "\\operatorname{fill}",
          "\\width": "\\operatorname{width}", "\\cc": "\\operatorname{cc}", "\\poly": "\\operatorname{poly}",
        },
      },
      plugins: [RevealMath.KaTeX, RevealNotes],
    });
    const sync = () => syncSlide(Reveal.getCurrentSlide());
    ["fragmentshown", "fragmenthidden", "slidechanged", "ready", "pdf-ready"].forEach((ev) => Reveal.on(ev, sync));
    document.querySelectorAll(".reveal .slides section").forEach(syncSlide);
    sync();
    document.body.dataset.deckReady = "1";
  }

  main().catch((e) => {
    console.error(e);
    document.body.insertAdjacentHTML("afterbegin", `<pre class="deck-error">${e.stack || e}</pre>`);
  });
})();
