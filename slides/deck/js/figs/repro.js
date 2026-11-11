/* repro.js — renderers for the reproduction section (traces R1–R4, traces/fig_repro.py). */
(function (global) {
  "use strict";
  const { el, text, svgFor, csv } = global.SVG;
  global.Figs = global.Figs || {};

  /* <div data-fig="ranges" data-trace="R3" data-metric="top1" data-title="Top-1">
     One row per model and split: our range over the CV runs (bar), the published value (diamond),
     and per split the exact random expectation (dashed) and MinFill first (solid gray). Axis 0–100 %. */
  global.Figs.ranges = (box, t, d) => {
    const metric = d.metric || "top1";
    const W = 760, rowH = 58, gapH = 40, m = { l: 150, r: 70, t: 86, b: 92 };
    const splits = csv(d.splits || "id,ood");
    const splitLabel = { id: `test circuits (N = ${t.n_id})`, ood: `held-out QFT (N = ${t.n_ood})` };
    const H = m.t + splits.length * (2 * rowH + gapH) + m.b - gapH;
    const svg = svgFor(box, 0, 0, W, H, `${d.title || metric}: our ranges against the published values`);
    const x = (v) => m.l + (W - m.l - m.r) * v / 100;
    const gA = el("g", { class: "c-axis" }, svg);
    const yAxis = H - m.b + 10;
    [0, 25, 50, 75, 100].forEach((v) => {
      el("line", { x1: x(v), y1: m.t - 8, x2: x(v), y2: yAxis, class: "c-grid" }, gA);
      text(gA, x(v), yAxis + 24, `${v}%`, "c-tick", { "text-anchor": "middle" });
    });
    text(svg, 0, 24, d.title || metric, "c-name", { fill: "var(--ink)" });
    splits.forEach((s, si) => {
      const y0 = m.t + si * (2 * rowH + gapH);
      const rows = t.rows.filter((r) => r.split === s);
      text(svg, 0, y0 - 6, splitLabel[s], "c-label", { "font-weight": 600 });
      const ref = t.ref[s];
      const y1 = y0 + 4, y2 = y0 + rows.length * rowH - 4;
      el("line", { x1: x(ref.mf_first[metric]), y1, x2: x(ref.mf_first[metric]), y2, class: "rp-ref-mf" }, svg);
      el("line", { x1: x(ref.random[metric]), y1, x2: x(ref.random[metric]), y2, class: "rp-ref-rand" }, svg);
      rows.forEach((r, ri) => {
        const yc = y0 + ri * rowH + rowH / 2;
        text(svg, 16, yc, r.model, "c-tick", { "dominant-baseline": "central" });
        const [lo, hi] = r[metric];
        const x0 = x(lo), x1 = Math.max(x(hi), x(lo) + 8);
        el("rect", { x: x0 - 4, y: yc - 9, width: x1 - x0 + 8, height: 18, rx: 4, class: "rp-range" }, svg);
        const px = x(r.paper[metric]), dd = 10;
        el("path", { d: `M${px},${yc - dd} L${px + dd},${yc} L${px},${yc + dd} L${px - dd},${yc} Z`, class: "rp-paper" }, svg);
        text(svg, Math.max(x1, px + dd) + 10, yc, r[`${metric}_txt`], "c-tick", { "dominant-baseline": "central", fill: "var(--ink)" });
      });
    });
    const lg = el("g", {}, svg), ly = H - 22;
    el("rect", { x: 16, y: ly - 9, width: 34, height: 18, rx: 4, class: "rp-range" }, lg);
    text(lg, 58, ly, "ours (range)", "c-tick", { "dominant-baseline": "central" });
    el("path", { d: `M220,${ly - 9} L229,${ly} L220,${ly + 9} L211,${ly} Z`, class: "rp-paper" }, lg);
    text(lg, 238, ly, "published", "c-tick", { "dominant-baseline": "central" });
    el("line", { x1: 380, y1: ly - 10, x2: 380, y2: ly + 10, class: "rp-ref-rand" }, lg);
    text(lg, 392, ly, "random", "c-tick", { "dominant-baseline": "central" });
    el("line", { x1: 510, y1: ly - 10, x2: 510, y2: ly + 10, class: "rp-ref-mf" }, lg);
    text(lg, 522, ly, "MinFill first", "c-tick", { "dominant-baseline": "central" });
    return { length: 0, render() {} };
  };
})(window);
