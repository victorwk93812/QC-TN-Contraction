/* chart.js — small line charts drawn from a trace series (no hand-typed values).
 *
 * <div data-fig="chart" data-trace="F07" data-key="chart" data-reveal="series"></div>
 * key → {xlabel, ylabel, ylog: bool, series: [{name, x: [...], y: [...], cls: "accent|derived|side-b|muted"}],
 *        xticks?: [...], yticks?: [...], marks?: [{x, label}]}
 * reveal="series": step k shows the first k series (series 0 always visible if data-first="on").
 */
(function (global) {
  "use strict";
  const { el, text, svgFor, get } = global.SVG;
  const nice = (lo, hi, n) => {
    const span = hi - lo, raw = span / Math.max(1, n), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };

  global.Figs = global.Figs || {};
  global.Figs.chart = (box, trace, d) => {
    const c = get(trace, d.key || "chart");
    const W = +(d.w || 760), H = +(d.h || 430), m = { l: 86, r: +(d.mr || 150), t: 20, b: 66 };
    const svg = svgFor(box, 0, 0, W, H, c.title || "chart");
    const xs = c.series.flatMap((s) => s.x), ys = c.series.flatMap((s) => s.y);
    const xlo = c.xmin ?? Math.min(...xs), xhi = c.xmax ?? Math.max(...xs);
    const ylo = c.ymin ?? Math.min(0, ...ys), yhi = c.ymax ?? Math.max(...ys) * 1.05;
    const X = (v) => m.l + ((v - xlo) / (xhi - xlo || 1)) * (W - m.l - m.r);
    const Y = (v) => H - m.b - ((v - ylo) / (yhi - ylo || 1)) * (H - m.t - m.b);
    const reveal = d.reveal === "series";
    const first = d.first === "on" ? 1 : 0;
    const length = reveal ? c.series.length - first : 0;

    function render(k) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const gA = el("g", { class: "c-axis" }, svg);
      el("line", { x1: m.l, y1: H - m.b, x2: W - m.r, y2: H - m.b }, gA);
      el("line", { x1: m.l, y1: m.t, x2: m.l, y2: H - m.b }, gA);
      (c.xticks || nice(xlo, xhi, 6)).forEach((v, i) => {
        el("line", { x1: X(v), y1: H - m.b, x2: X(v), y2: H - m.b + 6 }, gA);
        text(gA, X(v), H - m.b + 26, c.xticklabels ? c.xticklabels[i] : String(v), "c-tick", { "text-anchor": "middle" });
      });
      (c.yticks || nice(ylo, yhi, 5)).forEach((v) => {
        el("line", { x1: m.l, y1: Y(v), x2: W - m.r, y2: Y(v), class: "c-grid" }, gA);
        text(gA, m.l - 10, Y(v), String(v), "c-tick", { "text-anchor": "end", "dominant-baseline": "central" });
      });
      text(gA, (m.l + W - m.r) / 2, H - 12, c.xlabel || "", "c-label", { "text-anchor": "middle" });
      text(gA, 18, (m.t + H - m.b) / 2, c.ylabel || "", "c-label",
           { "text-anchor": "middle", transform: `rotate(-90 18 ${(m.t + H - m.b) / 2})` });
      if (c.zero) el("line", { x1: m.l, y1: Y(0), x2: W - m.r, y2: Y(0), class: "c-zero" }, gA);
      const shown = reveal ? c.series.slice(0, k + first) : c.series;
      shown.forEach((s) => {
        const g = el("g", { class: `c-series is-${s.cls || "muted"}${s.dashed ? " is-dashed" : ""}` }, svg);
        el("polyline", { points: s.x.map((x, i) => `${X(x)},${Y(s.y[i])}`).join(" ") }, g);
        s.x.forEach((x, i) => el("circle", { cx: X(x), cy: Y(s.y[i]), r: 4.5 }, g));
        const i = s.x.length - 1;
        text(g, X(s.x[i]) + 10, Y(s.y[i]) + (s.dy || 0), s.name, "c-name", { "dominant-baseline": "central" });
      });
      (c.marks || []).forEach((mk) => {
        const g = el("g", { class: "c-mark" }, svg);
        el("line", { x1: X(mk.x), y1: m.t, x2: X(mk.x), y2: H - m.b }, g);
        text(g, X(mk.x) + 6, m.t + 14, mk.label, "c-tick");
      });
    }
    render(0);
    return { length, render };
  };
})(window);
