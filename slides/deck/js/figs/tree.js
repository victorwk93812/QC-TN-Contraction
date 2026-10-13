/* tree.js — draw a pairwise contraction tree from a trace (carving.tree_record).
 *
 * <div data-fig="tree" data-trace="F04" data-key="trees.good" data-reveal="edges|merges|none"
 *      data-layout="tree|spine" data-highlight="H" data-flag-max="on">
 *
 * Leaves are tensors (squares), internal nodes are intermediates labelled with their rank,
 * the root is a small dot. Edge labels give the open legs of the child = the bonds crossing
 * that tree edge. Nodes whose rank equals the tree's max rank are drawn in accent (the
 * bottleneck); other intermediates in derived purple.
 * reveal="edges": step k shows the labels of the first k edges (merge order).
 * reveal="merges": step k shows the first k merges (internal nodes in merge order).
 * layout="spine": a list schedule drawn as a vertical spine with leaves on the right.
 */
(function (global) {
  "use strict";
  const { el, text, svgFor, extent, csv, get } = global.SVG;
  const setStr = (legs) => `{${legs.join(", ")}}`;

  function spineLayout(rec) {
    // post-order of a caterpillar: leaf0, leaf1, N1, leaf2, N2, ...
    const pos = {}, dy = 46, x = 0, xl = 72;
    const internals = rec.nodes.filter((n) => n.kind !== "leaf");
    const leaves = rec.nodes.filter((n) => n.kind === "leaf");
    const n = internals.length;
    pos[leaves[0].id] = [x, (n + 1) * dy];
    internals.forEach((node, j) => {
      pos[node.id] = [x, (n - j) * dy];
      pos[leaves[j + 1].id] = [xl, (n - j) * dy];
    });
    return pos;
  }

  global.Figs = global.Figs || {};
  global.Figs.tree = (box, trace, d) => {
    const rec = get(trace, d.key);
    const spine = d.layout === "spine";
    const pos = spine ? spineLayout(rec) : Object.fromEntries(rec.nodes.map((n) => [n.id, [n.x, n.y]]));
    const byId = Object.fromEntries(rec.nodes.map((n) => [n.id, n]));
    const hl = new Set(csv(d.highlight));
    const flagMax = d.flagMax !== "off";
    const pad = spine ? [150, 40, 50, 30] : [44, 30, 44, 30];   // left, top, right, bottom
    const [x0, y0, w, h] = extent(Object.values(pos), 0);
    const svg = svgFor(box, x0 - pad[0], y0 - pad[1], w + pad[0] + pad[2], h + pad[1] + pad[3], d.label || "contraction tree");
    const internals = rec.nodes.filter((n) => n.kind !== "leaf").map((n) => n.id);
    const reveal = d.reveal || "none";
    const length = reveal === "edges" ? rec.edges.length : reveal === "merges" ? internals.length : 0;

    const offset = +(d.offset || 0);
    function render(k) {
      k = Math.max(0, Math.min(length, k - offset));
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const shownMerges = reveal === "merges" ? new Set(internals.slice(0, k)) : new Set(internals);
      const shownLabels = reveal === "edges" ? new Set(rec.edges.slice(0, k).map((e) => e.child)) :
                          reveal === "merges" ? new Set(rec.edges.filter((e) => shownMerges.has(e.parent)).map((e) => e.child)) :
                          new Set(rec.edges.map((e) => e.child));
      const gE = el("g", {}, svg), gN = el("g", {}, svg), gL = el("g", {}, svg);
      rec.edges.forEach((e) => {
        const [px, py] = pos[e.parent], [cx, cy] = pos[e.child];
        const on = shownMerges.has(e.parent);
        el("line", { x1: px, y1: py, x2: cx, y2: cy, class: "t-edge" + (on ? "" : " is-ghost") }, gE);
        if (!spine && d.edgeLabels !== "off" && shownLabels.has(e.child)) {
          const left = cx < px;
          const mx = (px + cx) / 2 + (left ? -8 : 8), my = (py + cy) / 2;
          const big = flagMax && e.legs.length === rec.max_rank && rec.max_rank > 0 && byId[e.child].kind !== "leaf";
          text(gL, mx, my, setStr(e.legs), "t-elabel" + (big ? " is-accent" : ""), { "text-anchor": left ? "end" : "start" });
        }
      });
      rec.nodes.forEach((n) => {
        const [x, y] = pos[n.id];
        if (n.kind === "leaf") {
          const g = el("g", { class: "t-leaf" + (hl.has(n.label) ? " is-accent" : "") }, gN);
          el("rect", { x: x - 15, y: y - 15, width: 30, height: 30, rx: 4 }, g);
          text(g, x, y, n.label, "t-nlabel");
          return;
        }
        const on = shownMerges.has(n.id);
        if (n.kind === "root" && n.rank === 0) {
          el("circle", { cx: x, cy: y, r: 6, class: "t-root" + (on ? "" : " is-ghost") }, gN);
        } else {
          const isMax = flagMax && n.rank === rec.max_rank && n.kind === "internal";
          const g = el("g", { class: "t-node " + (isMax ? "is-accent" : "is-derived") + (on ? "" : " is-ghost") }, gN);
          el("circle", { cx: x, cy: y, r: 15 }, g);
          if (on) text(g, x, y, String(n.rank), "t-nlabel");
        }
        if (spine && on && n.kind !== "root") {
          const isMax = flagMax && n.rank === rec.max_rank && n.kind !== "root";
          text(gL, x - 26, y, `${setStr(n.legs)}  ${n.rank}`, "t-elabel" + (isMax ? " is-accent" : ""), { "text-anchor": "end" });
        }
      });
    }
    render(0);
    return { length, render };
  };
})(window);
