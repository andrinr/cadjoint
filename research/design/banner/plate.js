if (location.search.includes("dark")) document.querySelector(".canvas").classList.add("dark");
// Graticule per design-language.md §10.1, generalised to the banner's frames:
// a constant 48px division (the page module is 80px; the instrument scale is its
// own, finer one), laid out from the plate centre outward so the two centre axes
// are always real axes. 1px #1e1d22, behind the geometry. The centre axes — and
// only they — carry five subdivisions per division at #332f38, arms 5px, 9px on
// every fifth. Corner brackets are 26px / 2px / #413e47, in CSS.
const NS = "http://www.w3.org/2000/svg";
for (const p of document.querySelectorAll(".plate[data-grat]")) {
  const w = p.offsetWidth, h = p.offsetHeight;
  const D = +(p.dataset.div || 48);
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "grat");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const line = (x1, y1, x2, y2, c) => {
    const l = document.createElementNS(NS, "line");
    l.setAttribute("x1", x1); l.setAttribute("y1", y1);
    l.setAttribute("x2", x2); l.setAttribute("y2", y2);
    l.setAttribute("stroke", c); l.setAttribute("stroke-width", 1);
    svg.appendChild(l);
  };
  const cx = Math.round(w / 2) + 0.5, cy = Math.round(h / 2) + 0.5;
  for (let k = -Math.ceil(w / 2 / D); k <= Math.ceil(w / 2 / D); k++) {
    if (k === 0) continue;
    const x = cx + k * D; if (x > 0 && x < w) line(x, 0, x, h, "#1e1d22");
  }
  for (let k = -Math.ceil(h / 2 / D); k <= Math.ceil(h / 2 / D); k++) {
    if (k === 0) continue;
    const y = cy + k * D; if (y > 0 && y < h) line(0, y, w, y, "#1e1d22");
  }
  line(cx, 0, cx, h, "#1e1d22"); line(0, cy, w, cy, "#1e1d22");
  const s = D / 5;
  for (let k = -Math.ceil(w / 2 / s); k <= Math.ceil(w / 2 / s); k++) {
    const x = cx + k * s, a = (k % 5 === 0) ? 9 : 5;
    if (x > 0 && x < w) line(x, cy - a / 2, x, cy + a / 2, "#332f38");
  }
  for (let k = -Math.ceil(h / 2 / s); k <= Math.ceil(h / 2 / s); k++) {
    const y = cy + k * s, a = (k % 5 === 0) ? 9 : 5;
    if (y > 0 && y < h) line(cx - a / 2, y, cx + a / 2, y, "#332f38");
  }
  p.insertBefore(svg, p.firstChild);
}

// The source render carries a probe readout whose value was mocked up. It is
// chrome, not field data, so it is covered by a real one: an opaque void plate
// at the same footprint (57.462% / 83.615% / 42.437% / 5.982% of the geometry
// image), carrying a measured value. Nothing in the field itself is touched.
for (const img of document.querySelectorAll(".plate img.geom[data-probe]")) {
  const s = img.style, box = { l: parseFloat(s.left), t: parseFloat(s.top),
                               w: parseFloat(s.width), h: parseFloat(s.height) };
  const d = document.createElement("div");
  d.className = "probe-plate";
  d.textContent = img.dataset.probe;
  const h = 0.05982 * box.h;
  Object.assign(d.style, {
    position: "absolute",
    left: `${box.l + 0.57462 * box.w}px`,
    top: `${box.t + 0.83615 * box.h}px`,
    width: `${0.42437 * box.w}px`,
    height: `${h}px`,
    fontSize: `${Math.round(h * 0.58)}px`,
  });
  img.after(d);
}
