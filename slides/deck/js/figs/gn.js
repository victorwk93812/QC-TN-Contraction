/* gn.js — F20: Girvan–Newman step by step. Left: the tensor graph; removed edges fade, the edge just
 * removed is coral; components containing A0 / B0 are purple / teal. Right: the dendrogram grows top-down
 * (an internal node appears at the step its split happens).
 * <div data-fig="gn" data-trace="F20" data-stride="2"> ; readouts in the section: .gn-readout */
(function (global) {
  "use strict";
  const { el, text, svgFor, extent } = global.SVG;
  global.Figs = global.Figs || {};
  global.Figs.gn = (box, trace, d) => {
    const pos = trace.nodes, steps = trace.steps, N = steps.length, stride = +(d.stride || 1);
    const rec = trace.tree;
    const wrap = document.createElement("div"); wrap.className = "cols"; wrap.style.gridTemplateColumns = "1.15fr 1fr";
    box.appendChild(wrap);
    const left = document.createElement("div"), right = document.createElement("div");
    wrap.append(left, right);
    const svg = svgFor(left, ...extent(Object.values(pos), 30), "Girvan–Newman on the tensor graph");
    const tpos = Object.fromEntries(rec.nodes.map((n) => [n.id, [n.x, n.y]]));
    const [x0, y0, w, h] = extent(Object.values(tpos), 26);
    const tsvg = svgFor(right, x0, y0, w, h, "dendrogram");
    const key = (a, b) => [a, b].sort().join("|");

    function render(k) {
      const t = Math.min(N, k * stride);            // number of removals applied
      const removed = new Set(steps.slice(0, t).map((s) => key(...s.removed)));
      const last = t ? key(...steps[t - 1].removed) : null;
      const comps = t ? steps[t - 1].components : [Object.keys(pos)];
      const cls = {};
      comps.forEach((c) => c.forEach((v) => (cls[v] = comps.length < 2 ? "" : c.includes("A0") ? "is-derived" : c.includes("B0") ? "is-side-b" : "")));
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      trace.edges.forEach(([a, b]) => {
        const kk = key(a, b);
        el("line", { x1: pos[a][0], y1: pos[a][1], x2: pos[b][0], y2: pos[b][1],
                     class: kk === last ? "g-edge is-accent is-dashed" : removed.has(kk) ? "g-edge is-faded" : "g-edge" }, svg);
      });
      Object.keys(pos).forEach((v) => {
        const g = el("g", { class: "g-node " + (cls[v] || "") }, svg);
        el("rect", { x: pos[v][0] - 17, y: pos[v][1] - 17, width: 34, height: 34, rx: 4 }, g);
        text(g, pos[v][0], pos[v][1], v, "g-nlabel");
      });
      while (tsvg.firstChild) tsvg.removeChild(tsvg.firstChild);
      const shown = new Set(rec.nodes.filter((n) => n.kind === "leaf" || n.split_step <= t).map((n) => n.id));
      rec.edges.forEach((e) => {
        if (!shown.has(e.parent)) return;
        el("line", { x1: tpos[e.parent][0], y1: tpos[e.parent][1], x2: tpos[e.child][0], y2: tpos[e.child][1], class: "t-edge" }, tsvg);
      });
      rec.nodes.forEach((n) => {
        const [x, y] = tpos[n.id];
        if (n.kind === "leaf") {
          const g = el("g", { class: "t-leaf" }, tsvg);
          el("rect", { x: x - 16, y: y - 16, width: 32, height: 32, rx: 4 }, g);
          text(g, x, y, n.label, "t-nlabel");
        } else if (shown.has(n.id)) {
          el("circle", { cx: x, cy: y, r: 9, class: "t-root" + (t && n.split_step === t ? " is-new" : "") }, tsvg);
        }
      });
      const ro = box.closest("section").querySelector(".gn-readout");
      if (ro) {
        const s = t ? steps[t - 1] : null;
        ro.innerHTML = s ? `removals <b>${t}</b> · last removed <b>${s.removed.join("–")}</b> (betweenness <b>${(+s.betweenness).toFixed(1)}</b>) · components <b>${s.components.length}</b> · Q = <b>${(+s.modularity).toFixed(3)}</b>`
                         : "every edge present · one component";
      }
    }
    render(0);
    return { length: Math.ceil(N / stride), stride: null, render };
  };

  /* F22: tree-decomposition local improvement drawn on the grid: widest bag (coral), region being re-solved (purple). */
  global.Figs.tdimprove = (box, trace, d) => {
    const pos = trace.nodes, states = trace.states;
    const svg = svgFor(box, ...extent(Object.values(pos), 26), "local improvement of a tree decomposition");
    function render(k) {
      const st = states[Math.min(k, states.length - 1)];
      const W = new Set(st.widest), R = new Set(st.region);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      trace.edges.forEach(([a, b]) => el("line", { x1: pos[a][0], y1: pos[a][1], x2: pos[b][0], y2: pos[b][1], class: "g-edge" }, svg));
      Object.keys(pos).forEach((v) => {
        const g = el("g", { class: "g-node " + (R.has(v) && !W.has(v) ? "is-derived" : W.has(v) && !R.size ? "is-accent" : R.has(v) ? "is-derived is-ring" : "") }, svg);
        el("circle", { cx: pos[v][0], cy: pos[v][1], r: 14 }, g);
      });
      const ro = box.closest("section").querySelector(".td-readout");
      if (ro) ro.innerHTML = R.size
        ? `re-solving a region of <b>${R.size}</b> vertices exactly (torso: region + cliques on its separators)`
        : `width <b class="acc">${st.width}</b> · widest bag has <b>${st.widest.length}</b> vertices · ${st.n_bags} bags`;
    }
    render(0);
    return { length: states.length - 1, render };
  };
})(window);
