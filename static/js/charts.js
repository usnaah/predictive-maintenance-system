/* Tiny dependency-free SVG bar charts (no CDN needed).
   Usage: <div class="chart" data-chart='{"labels":[...],"values":[...],"format":"pct"|"int"|"dec","horizontal":true}'></div> */
(function () {
  var NS = "http://www.w3.org/2000/svg";
  function el(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function fmt(v, f) {
    if (f === "pct") return (v * 100).toFixed(1) + "%";
    if (f === "dec") return v.toFixed(3);
    return String(Math.round(v));
  }
  function bars(box, cfg) {
    var L = cfg.labels, V = cfg.values, max = Math.max.apply(null, V.concat([1e-9])) * 1.12;
    var colors = cfg.colors || [];
    var svg;
    if (cfg.horizontal) {
      var rowH = 30, W = Math.max(300, box.clientWidth || 560), left = cfg.labelWidth || 170, H = L.length * rowH + 10;
      left = Math.min(left, W * 0.45);
      svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": cfg.title || "bar chart" });
      L.forEach(function (lab, i) {
        var w = Math.max(2, (W - left - 70) * V[i] / max), y = i * rowH + 5;
        svg.appendChild(el("text", { x: left - 8, y: y + 16, "text-anchor": "end" }, lab));
        svg.appendChild(el("rect", { x: left, y: y + 3, width: w, height: 18, rx: 4, fill: colors[i] || "#1f3a5f" }));
        svg.appendChild(el("text", { x: left + w + 6, y: y + 16, "class": "val" }, fmt(V[i], cfg.format)));
      });
    } else {
      var W2 = Math.max(260, box.clientWidth || 560), H2 = 230, pl = 14, pb = 42, pt = 22, n = L.length, slot = (W2 - pl * 2) / n, bw = Math.min(64, slot * 0.6);
      var small = slot < 70;
      svg = el("svg", { viewBox: "0 0 " + W2 + " " + H2, role: "img", "aria-label": cfg.title || "bar chart" });
      svg.appendChild(el("line", { x1: pl, x2: W2 - pl, y1: H2 - pb, y2: H2 - pb, stroke: "#dde4ec" }));
      L.forEach(function (lab, i) {
        var h = Math.max(2, (H2 - pb - pt) * V[i] / max), x = pl + slot * i + (slot - bw) / 2, y = H2 - pb - h;
        svg.appendChild(el("rect", { x: x, y: y, width: bw, height: h, rx: 4, fill: colors[i] || "#1f3a5f" }));
        svg.appendChild(el("text", { x: x + bw / 2, y: y - 6, "text-anchor": "middle", "class": "val" }, fmt(V[i], cfg.format)));
        svg.appendChild(el("text", { x: x + bw / 2, y: H2 - pb + 16, "text-anchor": "middle" }, small && lab.length > 9 ? lab.slice(0, 8) + "…" : lab));
        if (cfg.sublabels) svg.appendChild(el("text", { x: x + bw / 2, y: H2 - pb + 30, "text-anchor": "middle" }, cfg.sublabels[i]));
      });
    }
    box.appendChild(svg);
  }
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".chart[data-chart]").forEach(function (box) {
      try { bars(box, JSON.parse(box.getAttribute("data-chart"))); }
      catch (e) { box.textContent = "Chart could not be drawn."; console.error(e); }
    });
    document.querySelectorAll("form[data-loading]").forEach(function (f) {
      f.addEventListener("submit", function () {
        if (!f.checkValidity || f.checkValidity()) { var b = f.querySelector("button[type=submit]"); if (b) { b.classList.add("loading"); b.disabled = true; } }
      });
    });
    var ex = document.getElementById("fill-example");
    if (ex) ex.addEventListener("click", function () {
      var data = JSON.parse(ex.getAttribute("data-example"));
      Object.keys(data).forEach(function (k) { var i = document.querySelector('[name="' + k + '"]'); if (i) i.value = data[k]; });
    });
  });
})();
