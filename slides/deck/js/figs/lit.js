/* lit.js — step through hand-placed SVG groups: <div data-fig="lit"> containing elements with
 * data-lit="1..n". At step k the element(s) with data-lit=k get .is-lit, earlier ones .is-done.
 * data-mode="cumulative" lights 1..k instead. Used for the roadmap (F00) and the technique map (F02). */
(function (global) {
  "use strict";
  global.Figs = global.Figs || {};
  global.Figs.lit = (box, trace, d) => {
    const items = [...box.querySelectorAll("[data-lit]")];
    const n = Math.max(0, ...items.map((e) => +e.dataset.lit));
    const cumulative = d.mode === "cumulative";
    function render(k) {
      items.forEach((e) => {
        const i = +e.dataset.lit;
        e.classList.toggle("is-lit", cumulative ? i <= k : i === k);
        e.classList.toggle("is-done", !cumulative && i < k);
        e.classList.toggle("is-hidden", d.hideFuture === "on" && i > k);
      });
    }
    render(0);
    return { length: d.static !== undefined ? 0 : n, render };
  };
})(window);
