/* graph.js — static drawing of a graph stored in a trace (fixed positions from Python).
 *
 * <div data-fig="graph" data-trace="F09" data-key="tensor_graph" data-shape="square"
 *      data-accent="H" data-derived="" data-side-b="" data-accent-edges="p,q" data-margin="30">
 * key → object with `nodes: {id: [x, y]}` and either `bonds: [[a, b, label]]` (tensor graph:
 * edges labelled by bond) or `edges: [[a, b]]` (default: the trace's own nodes/edges).
 * Edge highlight ids are bond labels, or "a-b" for unlabelled edges.
 * data-steps="p,q;e" makes a stepping figure: step k adds the k-th group to the accent set.
 */
(function (global) {
  "use strict";
  const { el, text, svgFor, extent, csv, get } = global.SVG;

  global.Figs = global.Figs || {};
  global.Figs.graph = (box, trace, d) => {
    const g = get(trace, d.key || "");
    const pos = g.nodes;
    const bonds = g.bonds ? g.bonds : (g.edges || []).map(([a, b]) => [a, b, `${a}-${b}`]);
    const square = d.shape === "square";
    const r = +(d.r || (square ? 16 : 14));
    const ext = extent(Object.values(pos), +(d.margin || r + 18));
    const svg = svgFor(box, ...ext, d.label || trace.meta.title);
    const sets = (s) => new Set(csv(s));
    const accent = sets(d.accent), derived = sets(d.derived), sideB = sets(d.sideB), faded = sets(d.faded);
    if (d.from) {                       // separator panel: {S, A, B} lists stored in the trace
      const o = get(trace, d.from);
      (o.S || []).forEach((v) => accent.add(String(v)));
      (o.A || []).forEach((v) => derived.add(String(v)));
      (o.B || []).forEach((v) => sideB.add(String(v)));
    }
    const accentEdges = sets(d.accentEdges), fadedEdges = sets(d.fadedEdges);
    const groups = d.steps ? d.steps.split(";").map(csv) : [];

    function render(k) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const extra = new Set(groups.slice(0, k).flat());
      const isAcc = (v) => accent.has(v) || extra.has(v);
      const gE = el("g", {}, svg), gN = el("g", {}, svg), gL = el("g", {}, svg);
      bonds.forEach(([a, b, lab]) => {
        const cls = ["g-edge"];
        if (accentEdges.has(lab) || extra.has(lab)) cls.push("is-accent");
        if (fadedEdges.has(lab)) cls.push("is-faded");
        el("line", { x1: pos[a][0], y1: pos[a][1], x2: pos[b][0], y2: pos[b][1], class: cls.join(" ") }, gE);
        if (g.bonds && d.edgeLabels !== "off") {
          const mx = (pos[a][0] + pos[b][0]) / 2, my = (pos[a][1] + pos[b][1]) / 2;
          const dx = pos[b][0] - pos[a][0], dy = pos[b][1] - pos[a][1], L = Math.hypot(dx, dy) || 1;
          const off = +(d.labelOffset || 13);
          text(gL, mx - (dy / L) * off, my + (dx / L) * off, lab,
               "g-elabel" + (accentEdges.has(lab) || extra.has(lab) ? " is-accent" : ""));
        }
      });
      Object.entries(pos).forEach(([v, [x, y]]) => {
        const cls = ["g-node"];
        if (isAcc(v)) cls.push("is-accent");
        else if (derived.has(v)) cls.push("is-derived");
        else if (sideB.has(v)) cls.push("is-side-b");
        if (faded.has(v)) cls.push("is-faded");
        const gg = el("g", { class: cls.join(" ") }, gN);
        if (square) el("rect", { x: x - r, y: y - r, width: 2 * r, height: 2 * r, rx: 4 }, gg);
        else el("circle", { cx: x, cy: y, r }, gg);
        if (d.labels !== "off") text(gg, x, y, v, "g-nlabel");
      });
    }
    render(0);
    return { length: groups.length, render };
  };
})(window);
