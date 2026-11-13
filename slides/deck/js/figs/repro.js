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

  /* <div data-fig="forest" data-trace="R6"></div>
     One row per measure, grouped (host overhead / compilation / GPU work): mean within-circuit Spearman with the
     labels and its 95% interval; the noise ceiling as a dashed line; median and mean [95%] as text columns. */
  global.Figs.forest = (box, t) => {
    const W = 985, rowH = 34, headH = 38, top = 44, lo = -0.5, hi = 0.8;
    const ax = { l: 365, r: 665 }, cMed = 688, cMean = 768;
    const groups = [...new Set(t.rows.map((r) => r.group))];
    const yEnd = top + groups.length * headH + t.rows.length * rowH + 6;
    const H = yEnd + 66;
    const svg = svgFor(box, 0, 0, W, H, "Spearman of each measure with the labels: means and 95% intervals");
    const x = (v) => ax.l + (ax.r - ax.l) * (v - lo) / (hi - lo);
    const gA = el("g", { class: "c-axis" }, svg);
    [-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8].forEach((v) => {
      el("line", { x1: x(v), y1: top, x2: x(v), y2: yEnd, class: v === 0 ? "c-zero" : "c-grid" }, gA);
      text(gA, x(v), yEnd + 24, v.toFixed(1).replace("-", "−"), "c-tick", { "text-anchor": "middle" });
    });
    const cx = x(t.ceiling.mean);
    el("line", { x1: cx, y1: top - 22, x2: cx, y2: yEnd, class: "rp-ceiling" }, svg);
    text(svg, cx + 6, top - 26, `re-measurement with the labels' noise: ${t.ceiling.mean_txt}`, "c-tick", { "text-anchor": "end", fill: "var(--ink)" });
    text(svg, cMed, top + headH / 2 + 4, "median", "c-tick", { "font-weight": 600, "dominant-baseline": "central" });
    text(svg, cMean, top + headH / 2 + 4, "mean [95%]", "c-tick", { "font-weight": 600, "dominant-baseline": "central" });
    let y = top;
    groups.forEach((g) => {
      text(svg, 0, y + headH / 2 + 4, t.groups[g], "c-label", { "font-weight": 600, "dominant-baseline": "central", class: `c-label rp-g-${g}` });
      y += headH;
      t.rows.filter((r) => r.group === g).forEach((r) => {
        const yc = y + rowH / 2;
        text(svg, 16, yc, r.label, "c-tick", { "dominant-baseline": "central", fill: "var(--ink)" });
        el("line", { x1: x(r.lo), y1: yc, x2: x(r.hi), y2: yc, class: `rp-ci rp-${g}` }, svg);
        el("circle", { cx: x(r.mean), cy: yc, r: 7, class: `rp-dot rp-${g}` }, svg);
        text(svg, cMed, yc, r.median_txt, "c-tick", { "dominant-baseline": "central", fill: "var(--ink)" });
        text(svg, cMean, yc, `${r.mean_txt} ${r.ci_txt}`, "c-tick", { "dominant-baseline": "central", fill: "var(--ink)" });
        y += rowH;
      });
    });
    text(svg, 16, H - 8, "Spearman with the labels, per circuit (positive: the cheaper plan has the larger label)", "c-tick");
    return { length: 0, render() {} };
  };

  /* <div data-fig="phasebars" data-trace="R7"></div>
     One stacked bar per circuit: shares of a step-by-step contract_tn! run (indices, tensor_data, einsum, rest);
     ms per step and CPU / GPU time as text columns. */
  global.Figs.phasebars = (box, t) => {
    const W = 1000, rowH = 32, top = 64, m = { l: 190, r: 760 };
    const parts = [["indices", "find indices"], ["data", "tensor_data"], ["einsum", "einsum"], ["rest", "rest"]];
    const H = top + t.rows.length * rowH + 74;
    const svg = svgFor(box, 0, 0, W, H, "Share of each phase of a contract_tn! run, per circuit");
    const x = (v) => m.l + (m.r - m.l) * v;
    const gA = el("g", { class: "c-axis" }, svg);
    const yEnd = top + t.rows.length * rowH;
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      el("line", { x1: x(v), y1: top - 6, x2: x(v), y2: yEnd + 4, class: "c-grid" }, gA);
      text(gA, x(v), yEnd + 26, `${v * 100}%`, "c-tick", { "text-anchor": "middle" });
    });
    text(svg, 790, top - 18, "ms / step", "c-tick", { "font-weight": 600 });
    text(svg, 905, top - 18, "CPU / GPU", "c-tick", { "font-weight": 600 });
    t.rows.forEach((r, i) => {
      const yc = top + i * rowH + rowH / 2;
      text(svg, 0, yc, r.short, "c-tick", { "dominant-baseline": "central", fill: "var(--ink)" });
      let acc = 0;
      parts.forEach(([k]) => {
        const w = r.shares[k];
        el("rect", { x: x(acc), y: yc - 10, width: Math.max(0, x(acc + w) - x(acc)), height: 20, class: `rp-ph rp-ph-${k}` }, svg);
        acc += w;
      });
      text(svg, 790, yc, r.ms_per_step.toFixed(1), "c-tick", { "dominant-baseline": "central", fill: "var(--ink)" });
      text(svg, 905, yc, `${r.cpu_over_gpu.toFixed(2)}×`, "c-tick", { "dominant-baseline": "central", fill: r.cpu_over_gpu > 1.05 ? "var(--ink)" : "var(--ink-2)" });
    });
    const ly = H - 18;
    let lx = 0;
    parts.forEach(([k, name]) => {
      el("rect", { x: lx, y: ly - 10, width: 26, height: 20, class: `rp-ph rp-ph-${k}` }, svg);
      text(svg, lx + 34, ly, name, "c-tick", { "dominant-baseline": "central" });
      lx += 34 + name.length * 10.5 + 34;
    });
    return { length: 0, render() {} };
  };

  /* <div data-fig="copies" data-trace="R8"></div>
     The release's labels of plans identical to their circuit's MinFill plan, against our timed speedups of the
     same plans (both modes). Log axis; an identical plan should be at 1. */
  global.Figs.copies = (box, t) => {
    const W = 1000, rowH = 92, top = 40, m = { l: 230, r: 975 }, lo = 0.4, hi = 1.3;
    const rows = [
      { name: `release labels (${t.copies})`, vals: t.labels.map((l) => [l.speedup, l.cause]), cls: "rp-lab" },
      { name: `ours, hyperindex (${t.ours.hyper.length})`, vals: t.ours.hyper.map((v) => [v, ""]), cls: "rp-ours" },
      { name: `ours, dense (${t.ours.dense.length})`, vals: t.ours.dense.map((v) => [v, ""]), cls: "rp-ours" },
    ];
    const H = top + rows.length * rowH + 92;
    const svg = svgFor(box, 0, 0, W, H, "Speedups of identical plans: release labels against our timings");
    const x = (v) => m.l + (m.r - m.l) * (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
    const gA = el("g", { class: "c-axis" }, svg);
    const yEnd = top + rows.length * rowH;
    [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3].forEach((v) => {
      el("line", { x1: x(v), y1: top - 6, x2: x(v), y2: yEnd, class: v === 1 ? "c-zero" : "c-grid" }, gA);
      text(gA, x(v), yEnd + 24, String(v), "c-tick", { "text-anchor": "middle" });
    });
    text(svg, x(1), top - 14, "same plan as MinFill: 1", "c-tick", { "text-anchor": "middle", fill: "var(--ink)" });
    rows.forEach((row, ri) => {
      const yc = top + ri * rowH + rowH / 2;
      text(svg, 0, yc, row.name, "c-tick", { "dominant-baseline": "central", fill: "var(--ink)" });
      [...row.vals].sort((a, b) => a[0] - b[0]).forEach(([v, cause], i) => {
        el("circle", { cx: x(v), cy: yc + ((i % 3) - 1) * 16, r: 8, class: `${row.cls}${cause === "unexplained" ? " is-open" : ""}` }, svg);
      });
    });
    const ly = H - 18;
    el("circle", { cx: 8, cy: ly, r: 7, class: "rp-lab" }, svg);
    text(svg, 22, ly, "label: looks like a solver timeout", "c-tick", { "dominant-baseline": "central" });
    el("circle", { cx: 430, cy: ly, r: 7, class: "rp-lab is-open" }, svg);
    text(svg, 444, ly, "label: unexplained copy", "c-tick", { "dominant-baseline": "central" });
    el("circle", { cx: 700, cy: ly, r: 7, class: "rp-ours" }, svg);
    text(svg, 714, ly, "our timing, V100", "c-tick", { "dominant-baseline": "central" });
    text(svg, (m.l + m.r) / 2, yEnd + 52, "speedup against the circuit's MinFill plan (log scale)", "c-tick", { "text-anchor": "middle" });
    return { length: 0, render() {} };
  };
})(window);
