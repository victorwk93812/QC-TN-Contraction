/* flowcutter.js — F18: FlowCutter-lite on a grid. Source region (derived), target region (side-b),
 * current cut (accent); beside it, every cut found so far as (size, smaller side), with the
 * Pareto front in accent.  <div data-fig="flowcutter" data-trace="F18" data-stride="4"> */
(function (global) {
  "use strict";
  const { el, text, svgFor, extent } = global.SVG;
  global.Figs = global.Figs || {};
  global.Figs.flowcutter = (box, trace, d) => {
    const pos = trace.nodes, steps = trace.steps, N = steps.length, stride = +(d.stride || 1);
    const wrap = document.createElement("div"); wrap.className = "cols"; wrap.style.gridTemplateColumns = "1fr 1.1fr";
    box.appendChild(wrap);
    const gl = document.createElement("div"), gr = document.createElement("div");
    wrap.append(gl, gr);
    const svg = svgFor(gl, ...extent(Object.values(pos), 24), "FlowCutter regions");
    const W = 560, H = 400, m = { l: 70, r: 20, t: 20, b: 64 };
    const plot = svgFor(gr, 0, 0, W, H, "cut size against balance");
    const pareto = new Set(trace.pareto);
    const xmax = Math.max(...steps.map((s) => s.size)) + 1, ymax = Math.max(...steps.map((s) => s.small_side)) + 2;
    const X = (v) => m.l + (v / xmax) * (W - m.l - m.r), Y = (v) => H - m.b - (v / ymax) * (H - m.t - m.b);

    function render(k) {
      const idx = k === 0 ? -1 : Math.min(N - 1, k * stride - 1);
      const st = idx >= 0 ? steps[idx] : null;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const S = new Set(st ? st.S : [trace.meta.source]), T = new Set(st ? st.T : [trace.meta.target]), C = new Set(st ? st.cut : []);
      const A = new Set(st ? st.A : []), B = new Set(st ? st.B : []);
      trace.edges.forEach(([a, b]) => el("line", { x1: pos[a][0], y1: pos[a][1], x2: pos[b][0], y2: pos[b][1], class: "g-edge" }, svg));
      Object.keys(pos).forEach((v) => {
        let c = "g-node";
        if (C.has(v)) c += " is-accent";
        else if (S.has(v)) c += " is-derived is-terminal";
        else if (T.has(v)) c += " is-side-b is-terminal";
        else if (st && A.has(v) && st.grew === "S") c += " is-derived is-light";
        else if (st && B.has(v) && st.grew === "T") c += " is-side-b is-light";
        const g = el("g", { class: c }, svg);
        el("circle", { cx: pos[v][0], cy: pos[v][1], r: 12 }, g);
      });
      // plot
      while (plot.firstChild) plot.removeChild(plot.firstChild);
      const gA = el("g", { class: "c-axis" }, plot);
      el("line", { x1: m.l, y1: H - m.b, x2: W - m.r, y2: H - m.b }, gA);
      el("line", { x1: m.l, y1: m.t, x2: m.l, y2: H - m.b }, gA);
      for (let v = 0; v < xmax; v++) text(gA, X(v), H - m.b + 26, String(v), "c-tick", { "text-anchor": "middle" });
      for (let v = 0; v <= ymax; v += 5) text(gA, m.l - 10, Y(v), String(v), "c-tick", { "text-anchor": "end", "dominant-baseline": "central" });
      text(gA, (m.l + W - m.r) / 2, H - 12, "separator size |S|", "c-label", { "text-anchor": "middle" });
      text(gA, 20, (m.t + H - m.b) / 2, "vertices on the smaller side", "c-label", { "text-anchor": "middle", transform: `rotate(-90 20 ${(m.t + H - m.b) / 2})` });
      const front = [];
      for (let i = 0; i <= idx; i++) {
        const s = steps[i], on = pareto.has(i);
        if (on) front.push(s);
        el("circle", { cx: X(s.size), cy: Y(s.small_side), r: on ? 7 : 4, class: on ? "pt is-accent" : "pt" }, plot);
      }
      if (front.length > 1) el("polyline", { points: front.map((s) => `${X(s.size)},${Y(s.small_side)}`).join(" "), class: "front" }, plot);
      const ro = box.closest("section").querySelector(".fc-readout");
      if (ro) ro.innerHTML = st ? `cut <b>${st.size}</b> · smaller side <b>${st.small_side}</b> of ${trace.n} · grew <b>${st.grew === "S" ? "source" : "target"}</b>` : "source and target corners";
    }
    render(0);
    return { length: Math.ceil(N / stride), render };
  };
})(window);
