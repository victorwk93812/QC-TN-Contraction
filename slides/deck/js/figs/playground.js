/* playground.js — F16 separator playground. Every property shown is looked up in the trace's
 * precomputed table (separator_report for all 2^11 subsets), never computed in the browser.
 * <div data-fig="playground" data-trace="F16" data-sequence="c,xy,cd,efg,xyz" data-menger-at="6">
 * Step k shows preset k of the sequence; at data-menger-at the two disjoint a–h paths appear.
 * Clicking a vertex toggles it in S (for Q&A); the next click/step restores the scripted state. */
(function (global) {
  "use strict";
  const { el, text, svgFor, extent, csv } = global.SVG;
  global.Figs = global.Figs || {};
  global.Figs.playground = (box, trace, d) => {
    const pos = trace.nodes, seq = csv(d.sequence), mengerAt = d.mengerAt ? +d.mengerAt : 0;
    const wrap = document.createElement("div"); wrap.className = "pg";
    box.appendChild(wrap);
    const gbox = document.createElement("div"); wrap.appendChild(gbox);
    const panel = document.createElement("div"); panel.className = "pg-panel"; wrap.appendChild(panel);
    const svg = svgFor(gbox, ...extent(Object.values(pos), 34), "separator playground");
    const yn = (b) => `<span class="${b ? "yes" : "no"}">${b ? "yes" : "no"}</span>`;
    const pathEdges = new Set();
    trace.menger_paths.forEach((p) => p.slice(1).forEach((v, i) => { pathEdges.add(p[i] + v); pathEdges.add(v + p[i]); }));
    let manual = null;

    function draw(S, menger) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const key = [...S].sort().join(""), rep = trace.table[key];
      const cls = {};
      rep.comps.forEach((c) => [...c].forEach((v) => (cls[v] = rep.comps.length < 2 ? "" : c.includes("a") ? "is-derived" : c.includes("h") ? "is-side-b" : "")));
      trace.edges.forEach(([a, b]) => {
        const onPath = menger && pathEdges.has(a + b), cut = S.has(a) || S.has(b);
        el("line", { x1: pos[a][0], y1: pos[a][1], x2: pos[b][0], y2: pos[b][1],
                     class: "g-edge" + (onPath ? " is-accent" : "") + (cut && !onPath ? " is-faded" : "") }, svg);
      });
      Object.keys(pos).forEach((v) => {
        const g = el("g", { class: "g-node " + (S.has(v) ? "is-accent" : cls[v] || ""), style: "cursor: pointer" }, svg);
        el("circle", { cx: pos[v][0], cy: pos[v][1], r: 19 }, g);
        text(g, pos[v][0], pos[v][1], v, "g-nlabel");
        g.addEventListener("click", () => {
          const T = new Set(manual || S);
          T.has(v) ? T.delete(v) : T.add(v);
          manual = T;
          draw(T, menger);
        });
      });
      const sizes = rep.comps.map((c) => c.length).sort((a, b) => b - a);
      const n = trace.n;
      panel.innerHTML = `
        <div class="row"><span>S</span><b>${S.size ? "{" + [...S].sort().join(", ") + "}" : "∅"}</b></div>
        <div class="row"><span>separator (G − S disconnected)</span>${yn(rep.sep)}</div>
        <div class="row"><span>component sizes</span><b>${sizes.join(", ") || "–"}</b></div>
        <div class="row"><span>separates a from h</span>${S.has("a") || S.has("h") ? "n/a" : yn(rep.ah)}</div>
        <div class="row"><span>minimal (≥ 2 full components)</span>${yn(rep.min)}</div>
        <div class="row"><span>½-balanced (all ≤ ${(n / 2).toFixed(1)})</span>${yn(rep.b2)}</div>
        <div class="row"><span>⅔-balanced (all ≤ ${(2 * n / 3).toFixed(1)})</span>${yn(rep.b3)}</div>
        <div class="row"><span>minimum a–h separator size</span><b>${trace.min_ah}</b></div>`;
    }
    function render(k) {
      manual = null;
      const S = new Set(k >= 1 && k <= seq.length ? [...seq[k - 1]] : k > seq.length && seq.length ? [...seq[seq.length - 1]] : []);
      draw(S, mengerAt && k >= mengerAt);       // keep the last preset: each path visibly crosses S
    }
    render(0);
    return { length: seq.length + (mengerAt ? 1 : 0), render };
  };
})(window);
