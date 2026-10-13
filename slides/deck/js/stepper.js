/* stepper.js — draw an elimination trace as SVG, one state per presenter click.
 *
 *   const fig = new TraceStepper(containerEl, traceJson, {readouts: rootEl, spark: svgEl});
 *   fig.render(k);                       // state after k steps; pure: replays the trace
 *   fig.bindReveal(Reveal, sectionEl);   // adds one hidden fragment per step
 *
 * The renderer never computes widths or costs of its own except to display what the
 * trace implies (fill counts for labels are derived from the replayed graph).
 */
(function (global) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const S = (x) => String(x);

  function el(name, attrs, parent) {
    const e = document.createElementNS(NS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  class TraceStepper {
    constructor(container, trace, opts = {}) {
      this.trace = trace;
      this.steps = trace.steps;
      this.ids = Object.keys(trace.nodes);
      this.pos = trace.nodes;
      this.opts = Object.assign(
        { radius: 13, labelMode: (trace.meta && trace.meta.label_mode) || "fill", readouts: null, spark: null, margin: 24 },
        opts
      );
      this.side = {};
      const sides = trace.sides || {};
      (sides.A || []).forEach((v) => (this.side[S(v)] = "side-a"));
      (sides.B || []).forEach((v) => (this.side[S(v)] = "side-b"));
      this.sepDepth = {};
      (trace.separators || []).forEach((s) => s.separator.forEach((v) => (this.sepDepth[S(v)] = s.depth)));
      const xs = this.ids.map((v) => this.pos[v][0]), ys = this.ids.map((v) => this.pos[v][1]);
      const m = this.opts.margin + this.opts.radius;
      const x0 = Math.min(...xs) - m, y0 = Math.min(...ys) - m;
      const w = Math.max(...xs) - Math.min(...xs) + 2 * m, h = Math.max(...ys) - Math.min(...ys) + 2 * m;
      this.svg = el("svg", { class: "tsv", viewBox: `${x0} ${y0} ${w} ${h}`, role: "img",
                             "aria-label": (trace.meta && trace.meta.title) || trace.id || "elimination trace" });
      container.appendChild(this.svg);
      this.k = -1;
      this.render(0);
    }

    get length() { return this.opts.maxSteps ? Math.min(this.opts.maxSteps, this.steps.length) : this.steps.length; }

    state(k) {
      const adj = new Map(this.ids.map((v) => [v, new Set()]));
      this.trace.edges.forEach(([a, b]) => { adj.get(S(a)).add(S(b)); adj.get(S(b)).add(S(a)); });
      const alive = new Set(this.ids), fills = [], order = {};
      for (let i = 0; i < k; i++) {
        const s = this.steps[i], v = S(s.elim);
        s.fill.forEach(([a, b]) => { adj.get(S(a)).add(S(b)); adj.get(S(b)).add(S(a)); fills.push([S(a), S(b)]); });
        adj.get(v).forEach((u) => adj.get(u).delete(v));
        adj.set(v, new Set());
        alive.delete(v);
        order[v] = i;
      }
      return { adj, alive, fills, order };
    }

    fillCount(adj, v) {
      const nb = [...adj.get(v)];
      let f = 0;
      for (let i = 0; i < nb.length; i++) for (let j = i + 1; j < nb.length; j++) if (!adj.get(nb[i]).has(nb[j])) f++;
      return f;
    }

    render(k) {
      k = Math.max(0, Math.min(this.length, k | 0));
      this.k = k;
      const { adj, alive, fills, order } = this.state(k);
      const svg = this.svg, r = this.opts.radius, P = this.pos;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const gE = el("g", { class: "tsv-edges" }, svg), gF = el("g", { class: "tsv-fills" }, svg),
            gB = el("g", { class: "tsv-baglines" }, svg), gN = el("g", { class: "tsv-nodes" }, svg);
      const line = (a, b, cls, g) => el("line", { x1: P[a][0], y1: P[a][1], x2: P[b][0], y2: P[b][1], class: cls }, g);
      // original edges stay as faint ghosts once an endpoint is eliminated, so the graph never disappears
      this.trace.edges.forEach(([a, b]) => line(S(a), S(b), alive.has(S(a)) && alive.has(S(b)) ? "tsv-edge" : "tsv-edge is-ghost", gE));
      fills.forEach(([a, b]) => { if (alive.has(a) && alive.has(b)) line(a, b, "tsv-fill", gF); });
      const cur = k > 0 ? this.steps[k - 1] : null;
      const bag = new Set(cur ? cur.nbrs.map(S) : []);
      if (cur) bag.forEach((u) => line(S(cur.elim), u, "tsv-bagline", gB));
      this.ids.forEach((v) => {
        const cls = ["tsv-node"];
        if (!alive.has(v)) cls.push(cur && S(cur.elim) === v ? "is-current" : "is-gone");
        else {
          if (v in this.sepDepth) cls.push("is-sep");
          else if (this.opts.showSides && this.side[v]) cls.push(this.side[v]);
          if (bag.has(v)) cls.push("is-bag");
        }
        const g = el("g", { class: cls.join(" "), "data-node": v }, gN);
        el("circle", { cx: P[v][0], cy: P[v][1], r }, g);
        let label = "";
        if (this.opts.labelMode === "none") label = "";
        else if (!alive.has(v)) label = this.opts.labelMode === "name" ? v : S(order[v]);
        else if (this.opts.labelMode === "fill") label = S(this.fillCount(adj, v));
        else if (this.opts.labelMode === "degree") label = S(adj.get(v).size);
        else if (this.opts.labelMode === "separator") label = v in this.sepDepth ? S(this.sepDepth[v]) : "";
        else if (this.opts.labelMode === "name") label = v;
        const t = el("text", { x: P[v][0], y: P[v][1] }, g);
        t.textContent = label;
      });
      this.updateReadouts(k, cur);
      this.updateSpark(k);
      if (this.opts.onRender) this.opts.onRender(k, this);
      return this;
    }

    updateReadouts(k, cur) {
      const root = this.opts.readouts;
      if (!root) return;
      const done = this.steps.slice(0, k);
      const width = done.reduce((m, s) => Math.max(m, s.nbrs.length), 0);
      let log2 = "–";
      if (done.length) {
        const mx = Math.max(...done.map((s) => s.bag));
        log2 = (mx + Math.log2(done.reduce((a, s) => a + Math.pow(2, s.bag - mx), 0))).toFixed(2);
      }
      const vals = { step: S(k), bag: cur ? S(cur.bag) : "–", fill: cur ? S(cur.fill.length) : "–",
                     width: S(width), log2cost: log2 };
      root.querySelectorAll("[data-readout]").forEach((e) => { e.textContent = vals[e.dataset.readout] ?? "–"; });
    }

    updateSpark(k) {
      const sp = this.opts.spark;
      if (!sp) return;
      while (sp.firstChild) sp.removeChild(sp.firstChild);
      const n = this.steps.length, W = 300, H = 70, mx = Math.max(...this.steps.map((s) => s.bag));
      const width = this.steps.slice(0, k).reduce((m, s) => Math.max(m, s.bag), 0);
      sp.setAttribute("viewBox", `0 0 ${W} ${H}`);
      this.steps.slice(0, k).forEach((s, i) => {
        const h = (H - 6) * s.bag / mx;
        el("rect", { x: i * W / n + 0.5, y: H - h, width: Math.max(1, W / n - 1), height: h,
                     class: s.bag === width ? "spark-bar is-max" : "spark-bar" }, sp);
      });
      el("line", { x1: 0, y1: H, x2: W, y2: H, class: "spark-axis" }, sp);
    }

    /* One hidden fragment per step; reveal drives render(k) forward and back. */
    bindReveal(Reveal, section) {
      if (!section.querySelector(".fragment[data-step]")) {
        for (let i = 1; i <= this.length; i++) {
          const f = document.createElement("span");
          f.className = "fragment step-fragment";
          f.dataset.step = S(i);
          section.appendChild(f);
        }
      }
      const mine = (e) => e.fragment && section.contains(e.fragment) && e.fragment.dataset.step;
      Reveal.on("fragmentshown", (e) => { if (mine(e)) this.render(+e.fragment.dataset.step); });
      Reveal.on("fragmenthidden", (e) => { if (mine(e)) this.render(+e.fragment.dataset.step - 1); });
      Reveal.on("slidechanged", (e) => {
        if (e.currentSlide === section) this.render(section.querySelectorAll(".fragment.visible[data-step]").length);
      });
      return this;
    }
  }

  global.TraceStepper = TraceStepper;

  /* deck.js figure type: <div data-fig="stepper" data-trace="F06" data-label-mode="fill" data-max-steps="5">
     with a .graph child; readouts and spark are looked up in the enclosing section. */
  global.Figs = global.Figs || {};
  global.Figs.stepper = (el, trace, d) => {
    const sec = el.closest("section");
    const scope = el.closest(".fig") || sec;
    const stride = d.stride ? +d.stride : 1;
    const fig = new TraceStepper(el.querySelector(".graph") || el, trace, {
      readouts: scope.querySelector(".readouts"), spark: scope.querySelector(".spark"),
      labelMode: d.labelMode || (trace.meta && trace.meta.label_mode) || "fill",
      maxSteps: d.maxSteps ? +d.maxSteps : null,
      onRender: d.onRender && global.DeckHooks ? global.DeckHooks[d.onRender] : null,
      radius: d.radius ? +d.radius : 13,
      showSides: d.sides === "on",
    });
    if (stride === 1) return fig;
    const n = fig.length;
    return { length: Math.ceil(n / stride), stride, steps: n, render: (k) => fig.render(Math.min(n, k * stride)) };
  };
})(typeof window !== "undefined" ? window : globalThis);
