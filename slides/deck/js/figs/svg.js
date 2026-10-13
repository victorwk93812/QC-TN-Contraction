/* svg.js — tiny SVG helpers shared by the figure renderers. */
(function (global) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  function el(name, attrs, parent) {
    const e = document.createElementNS(NS, name);
    for (const k in attrs || {}) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, str, cls, attrs) {
    const t = el("text", Object.assign({ x, y, class: cls || "" }, attrs || {}), parent);
    t.textContent = str;
    return t;
  }
  function svgFor(container, x0, y0, w, h, label) {
    const s = el("svg", { viewBox: `${x0} ${y0} ${w} ${h}`, role: "img", "aria-label": label || "figure", class: "figsvg" });
    container.appendChild(s);
    return s;
  }
  function extent(points, margin) {
    const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
    const x0 = Math.min(...xs) - margin, y0 = Math.min(...ys) - margin;
    return [x0, y0, Math.max(...xs) - Math.min(...xs) + 2 * margin, Math.max(...ys) - Math.min(...ys) + 2 * margin];
  }
  const csv = (s) => (s ? String(s).split(",").map((x) => x.trim()).filter(Boolean) : []);
  const get = (obj, path) => (path ? path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj) : obj);
  global.SVG = { NS, el, text, svgFor, extent, csv, get };
})(window);
