/* act4.js — renderers for Act IV: compact plan trees, GEMM step, feature blocks, ranking group. */
(function (global) {
  "use strict";
  const { el, text, svgFor, extent, get } = global.SVG;
  global.Figs = global.Figs || {};
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const mix = (a, b, t) => {
    const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const A = rgb(css(a)), B = rgb(css(b));
    return `rgb(${A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(",")})`;
  };

  /* <div data-fig="plantree" data-trace="F23" data-key="plans.MinFill.tree" data-cmax="14">
     Internal nodes coloured by step exponent c (light → dark purple); bottleneck steps (c = max) in coral. */
  global.Figs.plantree = (box, trace, d) => {
    const rec = get(trace, d.key);
    const pos = Object.fromEntries(rec.nodes.map((n) => [n.id, [n.x, n.y]]));
    const svg = svgFor(box, ...extent(Object.values(pos), 12), "contraction tree");
    const cs = rec.nodes.filter((n) => n.kind !== "leaf").map((n) => n.exp);
    const cmax = Math.max(...cs), lo = +(d.cmin || 0), hi = +(d.cmax || cmax);
    rec.edges.forEach((e) => el("line", { x1: pos[e.parent][0], y1: pos[e.parent][1], x2: pos[e.child][0], y2: pos[e.child][1], class: "pt-edge" }, svg));
    rec.nodes.forEach((n) => {
      const [x, y] = pos[n.id];
      if (n.kind === "leaf") { el("rect", { x: x - 3, y: y - 3, width: 6, height: 6, class: "pt-leaf" }, svg); return; }
      const fill = n.exp === cmax ? css("--accent") : mix("--derived-soft", "--derived-ink", Math.max(0, (n.exp - lo) / (hi - lo || 1)));
      el("circle", { cx: x, cy: y, r: n.exp === cmax ? 7 : 5.5, fill, class: "pt-node" }, svg);
    });
    return { length: 0, render() {} };
  };

  /* <div data-fig="gemm" data-trace="S47"> : (M×K)(K×N) = (M×N) with sides proportional to the exponents. */
  const sup = (n) => String(n).replace(/\d/g, (c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+c]);
  global.Figs.gemm = (box, t) => {
    const u = 34, m = t.m * u, n = t.n * u, k = t.k * u, g = 70;
    const W = k + g + n + g + n + 40, H = Math.max(m, k) + 110;
    const svg = svgFor(box, -20, -40, W + 20, H, "a contraction step as a matrix product");
    const rect = (x, y, w, h, cls, label, sub) => {
      el("rect", { x, y, width: w, height: h, rx: 4, class: "s-box " + cls }, svg);
      text(svg, x + w / 2, y + h / 2 - 8, label, "s-th", { "text-anchor": "middle" });
      text(svg, x + w / 2, y + h / 2 + 18, sub, "s-ts", { "text-anchor": "middle" });
    };
    rect(0, 0, k, m, "is-derived", "A", `2${sup(t.m)} × 2${sup(t.k)}`);
    text(svg, k + g / 2, m / 2, "·", "s-th", { "text-anchor": "middle", "font-size": 40 });
    rect(k + g, 0, n, k, "is-side-b", "B", `2${sup(t.k)} × 2${sup(t.n)}`);
    text(svg, k + g + n + g / 2, m / 2, "=", "s-th", { "text-anchor": "middle" });
    rect(k + g + n + g, 0, n, m, "is-accent", "O", `2${sup(t.m)} × 2${sup(t.n)}`);
    text(svg, 0, -14, `M = ${t.M}, K = ${t.K}, N = ${t.N}`, "s-t");
    return { length: 0, render() {} };
  };

  /* <div data-fig="features" data-trace="P50" data-mode="status|plain"> : the paper's feature blocks as chips. */
  global.Figs.features = (box, t, d) => {
    box.classList.add("feat-grid");
    t.blocks.forEach((b) => {
      const col = document.createElement("div");
      col.className = "feat-col";
      col.innerHTML = `<div class="feat-block">${b.block}</div>` + b.features.map((f) => {
        const cls = d.mode === "status" ? ` st-${f.status.replace(/[^a-z]/g, "")}` : "";
        return `<div class="feat${cls}">${f.name}${d.mode === "status" && f.status !== "kept" ? `<span class="feat-note">${f.status}</span>` : ""}</div>`;
      }).join("");
      box.appendChild(col);
    });
    return { length: 0, render() {} };
  };

  /* <div data-fig="rankgroup" data-trace="S48"> : one group of plans, its labels, and two rankings scored by NDCG@3. */
  global.Figs.rankgroup = (box, t, d) => {
    const P = t.plans, ymax = Math.max(...Object.values(P).map((p) => p.y));
    const list = (title, order, score, hot) => `<div class="rk-col${hot ? " rk-hot" : ""}"><div class="panel-title">${title}</div>` +
      order.map((p, i) => `<div class="rk-row"><span class="rk-pos">${i + 1}</span><span class="rk-name">${p}</span>` +
        `<span class="rk-bar"><span style="width:${(100 * P[p].y / ymax).toFixed(1)}%"></span></span><span class="rk-y">${P[p].y.toFixed(2)}</span></div>`).join("") +
      (score !== null ? `<div class="rk-score">NDCG@3 = <b>${score.toFixed(3)}</b></div>` : "") + "</div>";
    box.classList.add("rk-grid");
    box.innerHTML = list("ideal: sorted by label y", t.ideal, 1.0, false) +
                    list("rank by width (max rank)", t.by_width, t.ndcg3_by_width, false) +
                    list("MinFill first", t.mf_first, t.ndcg3_mf_first, true);
    const cols = box.querySelectorAll(".rk-col");
    const step = d.reveal === "on";
    function render(k) { cols.forEach((c, i) => c.classList.toggle("is-hidden-rk", step && i > k)); }
    render(step ? 0 : 99);
    return { length: step ? 2 : 0, render };
  };
})(window);
