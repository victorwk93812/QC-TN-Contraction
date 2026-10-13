/* arc.js — the talk's narrative arc (F00) drawn from data/F00.json.
 * <div data-fig="arc" data-trace="F00"></div>   step k lights act group k (I, II, III, deep dives, IV, V).
 * data-static: no steps (roadmap shown whole). */
(function (global) {
  "use strict";
  const { el, text, svgFor } = global.SVG;
  global.Figs = global.Figs || {};
  global.Figs.arc = (box, trace, d) => {
    const acts = trace.acts, W = 1500, H = 420, bw = 215, gap = 42;
    const svg = svgFor(box, 0, 0, W, H, "narrative arc of the talk");
    svg.classList.add("arc-svg");
    const defs = el("defs", {}, svg);
    const mid = `arrowhead-${Math.random().toString(36).slice(2, 8)}`;      // unique per instance (slides 6 and 51)
    const mk = el("marker", { id: mid, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: "var(--muted-2)" }, mk);
    const big = acts.filter((a) => !a.id.startsWith("DD")), dds = acts.filter((a) => a.id.startsWith("DD"));
    const cols = [big[0], big[1], big[2], null, big[3], big[4]];
    const groups = [];
    cols.forEach((a, c) => {
      const x = 10 + c * (bw + gap);
      if (a) {
        const g = el("g", { "data-lit": a.group }, svg);
        el("rect", { x, y: 100, width: bw, height: 210, rx: 12, class: "s-box" }, g);
        text(g, x + 16, 136, `ACT ${a.id}`, "act-num");
        wrap(g, a.title, x + 16, 174, bw - 30, "act-title");
        text(g, x + 16, 290, `slides ${a.first}–${a.last} · ${a.minutes} min`, "act-meta");
        groups.push(g);
      } else {
        dds.forEach((dd, i) => {
          const g = el("g", { "data-lit": dd.group }, svg);
          const y = 5 + i * 103;
          el("rect", { x, y, width: bw, height: 94, rx: 10, class: "s-box" }, g);
          text(g, x + 14, y + 32, `DEEP DIVE ${i + 1}`, "act-num");
          wrap(g, dd.title.split(":")[0], x + 14, y + 62, bw - 24, "act-meta");
        });
      }
      if (c < cols.length - 1) el("line", { x1: x + bw + 6, y1: 205, x2: x + bw + gap - 6, y2: 205, class: "arrow", "marker-end": `url(#${mid})` }, svg);
    });
    function wrap(g, str, x, y, w, cls) {
      const words = str.split(" "); let line = "", yy = y;
      words.forEach((wd) => {
        const t = line ? `${line} ${wd}` : wd;
        if (t.length * (cls === "act-title" ? 12.5 : 10.5) > w && line) { text(g, x, yy, line, cls); line = wd; yy += 25; } else line = t;
      });
      if (line) text(g, x, yy, line, cls);
    }
    return global.Figs.lit(box, null, d);
  };
})(window);
