/* extra.js — heat map, bar charts, bag lists. All values come from traces. */
(function (global) {
  "use strict";
  const { el, text, svgFor, get, csv } = global.SVG;
  global.Figs = global.Figs || {};

  // two-stop colour ramp between theme colours (read from CSS variables)
  function ramp(a, b) {
    const css = getComputedStyle(document.documentElement);
    const hex = (v) => css.getPropertyValue(v).trim();
    const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const A = rgb(hex(a)), B = rgb(hex(b));
    return (t) => `rgb(${A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(",")})`;
  }

  /* <div data-fig="heatmap" data-trace="F08"> : cell (r, c) coloured by time[r][c] (early light, late dark). */
  global.Figs.heatmap = (box, trace, d) => {
    const T = trace.time, L = T.length, cs = 40, n = trace.n;
    const svg = svgFor(box, 0, 0, L * cs + 20, L * cs + 64, "elimination time heat map");
    const col = ramp("--derived-soft", "--derived-ink");
    T.forEach((row, r) => row.forEach((t, c) => {
      el("rect", { x: c * cs + 10, y: r * cs + 10, width: cs - 2, height: cs - 2, rx: 3, fill: col(t / (n - 1)) }, svg);
      if (d.labels === "on") text(svg, c * cs + 10 + (cs - 2) / 2, r * cs + 10 + (cs - 2) / 2, String(t), "hm-label" + (t > n / 2 ? " is-dark" : ""));
    }));
    const lg = el("g", {}, svg), y = L * cs + 30;
    for (let i = 0; i < 20; i++) el("rect", { x: 10 + i * (L * cs / 20), y, width: L * cs / 20 + 0.5, height: 12, fill: col(i / 19) }, lg);
    text(lg, 10, y + 32, "eliminated first", "c-tick");
    text(lg, L * cs + 10, y + 32, "last", "c-tick", { "text-anchor": "end" });
    return { length: 0, render() {} };
  };

  /* <div data-fig="bars" data-trace="S25" data-key="tie_hist" data-x="x" data-y="count" data-xlabel=".." data-ylabel="..">
     Simple vertical bars; the tallest (or data-accent-x values) in accent. data-profile="on": thin bars, no labels. */
  global.Figs.bars = (box, trace, d) => {
    const o = get(trace, d.key);
    const ys = Array.isArray(o) ? o : o[d.y || "y"];
    const xs = Array.isArray(o) || !o[d.x || "x"] ? ys.map((_, i) => i + 1) : o[d.x || "x"];
    const profile = d.profile === "on";
    const W = +(d.w || 700), H = +(d.h || 360), m = { l: 70, r: 20, t: 20, b: 62 };
    const svg = svgFor(box, 0, 0, W, H, d.label || "bar chart");
    const mx = d.yscale ? +d.yscale : Math.max(...ys), bw = (W - m.l - m.r) / ys.length;
    const acc = new Set(csv(d.accentX).map(String));
    const accFrom = d.accentFrom !== undefined ? +d.accentFrom : null;
    const gA = el("g", { class: "c-axis" }, svg);
    el("line", { x1: m.l, y1: H - m.b, x2: W - m.r, y2: H - m.b }, gA);
    ys.forEach((y, i) => {
      const h = (H - m.t - m.b) * y / mx, x = m.l + i * bw;
      const hot = acc.has(String(xs[i])) || (accFrom !== null && i >= accFrom) || (!acc.size && accFrom === null && y === Math.max(...ys));
      el("rect", { x: x + (profile ? 0 : bw * 0.15), y: H - m.b - h, width: profile ? Math.max(0.6, bw - 0.4) : bw * 0.7, height: h,
                   class: "bar" + (hot ? " is-accent" : "") }, svg);
      if (!profile) {
        text(svg, x + bw / 2, H - m.b + 26, String(xs[i]), "c-tick", { "text-anchor": "middle" });
        text(svg, x + bw / 2, H - m.b - h - 8, String(y), "c-tick", { "text-anchor": "middle" });
      }
    });
    text(svg, (m.l + W - m.r) / 2, H - 10, d.xlabel || "", "c-label", { "text-anchor": "middle" });
    text(svg, 22, (m.t + H - m.b) / 2, d.ylabel || "", "c-label", { "text-anchor": "middle", transform: `rotate(-90 22 ${(m.t + H - m.b) / 2})` });
    if (profile) text(svg, m.l - 8, m.t + 6, String(mx), "c-tick", { "text-anchor": "end" });
    return { length: 0, render() {} };
  };

  /* <div data-fig="groupbars" data-trace="P26" data-groups="id,ood" data-group-labels="..." data-models="mf_ndcg,random,opt_ndcg"
          data-model-labels="..." data-metric="top1" data-classes="muted,side-b,accent" data-reveal="models">
     Grouped bars: one group per data split, one bar per model; step k reveals the first k models. */
  global.Figs.groupbars = (box, trace, d) => {
    const groups = csv(d.groups), gl = (d.groupLabels || "").split("|"), models = csv(d.models), ml = (d.modelLabels || "").split("|");
    const cls = csv(d.classes), metric = d.metric || "top1";
    const W = 760, H = 420, m = { l: 70, r: 20, t: 30, b: 90 };
    const svg = svgFor(box, 0, 0, W, H, "grouped bar chart");
    const gw = (W - m.l - m.r) / groups.length, bw = gw * 0.8 / models.length;
    const ymax = +(d.ymax || 100);
    const reveal = d.reveal === "models";
    function render(k) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const gA = el("g", { class: "c-axis" }, svg);
      el("line", { x1: m.l, y1: H - m.b, x2: W - m.r, y2: H - m.b }, gA);
      [0, 25, 50, 75, 100].filter((v) => v <= ymax).forEach((v) => {
        const y = H - m.b - (H - m.t - m.b) * v / ymax;
        el("line", { x1: m.l, y1: y, x2: W - m.r, y2: y, class: "c-grid" }, gA);
        text(gA, m.l - 10, y, `${v}%`, "c-tick", { "text-anchor": "end", "dominant-baseline": "central" });
      });
      groups.forEach((g, gi) => {
        const x0 = m.l + gi * gw + gw * 0.1;
        text(svg, x0 + gw * 0.4, H - m.b + 30, gl[gi] || g, "c-label", { "text-anchor": "middle" });
        models.forEach((mo, mi) => {
          if (reveal && mi >= k) return;
          const v = trace[g][mo][metric], h = (H - m.t - m.b) * v / ymax, x = x0 + mi * bw;
          el("rect", { x: x + 2, y: H - m.b - h, width: bw - 4, height: h, class: `bar is-${cls[mi] || "muted"}` }, svg);
          text(svg, x + bw / 2, H - m.b - h - 8, `${v}`, "c-tick", { "text-anchor": "middle" });
          if (gi === 0) text(svg, x + bw / 2, H - m.b + 58, "", "c-tick");
        });
      });
      models.forEach((mo, mi) => {
        if (reveal && mi >= k) return;
        const x = m.l + mi * 230;
        el("rect", { x, y: H - 26, width: 16, height: 16, class: `bar is-${cls[mi] || "muted"}` }, svg);
        text(svg, x + 24, H - 13, ml[mi] || mo, "c-tick");
      });
    }
    render(reveal ? 0 : models.length);
    return { length: reveal ? models.length : 0, render: (k) => render(reveal ? k : models.length) };
  };

  /* <div data-fig="bags" data-trace="S35" data-key="nd.bags"> : the bags of an elimination order as chips. */
  global.Figs.bags = (box, trace, d) => {
    const bags = get(trace, d.key);
    const mx = Math.max(...bags.map((b) => b.length));
    box.classList.add("bag-list");
    bags.forEach((b, i) => {
      const div = document.createElement("div");
      div.className = "bag" + (b.length === mx ? " is-accent" : "");
      div.innerHTML = `<span class="bag-elim">${b[0]}</span>${b.length > 1 ? " · " + b.slice(1).join(" ") : ""}`;
      box.appendChild(div);
    });
    return { length: 0, render() {} };
  };
})(window);
