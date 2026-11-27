"use strict";

// Small native SVG charts: no network libraries, model runtime or recalculated scores.
const motionCharts = (() => {
  const ns = "http://www.w3.org/2000/svg";
  const traces = {
    displacement: [
      {key:"mean_mm", label:"Mean displacement", colour:"#8ddbd1"},
      {key:"p95_mm", label:"95th percentile", colour:"#b4a4ff", dashed:true}
    ],
    rotation: [{key:"rotation_deg", label:"Rotation", colour:"#f4b978"}]
  };
  const element = (tag, text, cls) => {
    const node = document.createElement(tag);
    if(text !== undefined)node.textContent = text;
    if(cls)node.className = cls;
    return node;
  };
  const svgElement = (tag, attributes, text) => {
    const node = document.createElementNS(ns, tag);
    for(const [key, value] of Object.entries(attributes))node.setAttribute(key, value);
    if(text !== undefined)node.textContent = text;
    return node;
  };
  const value = raw => {
    if(raw === null || raw === undefined || raw === "" || typeof raw === "boolean")return null;
    const result = Number(raw);
    return Number.isFinite(result) ? result : null;
  };
  const format = raw => value(raw) === null ? "unavailable" : value(raw).toFixed(2);
  const measured = row => ["mean_mm", "p95_mm", "rotation_deg"].some(key => value(row.pair?.[key]) !== null);

  function phases(series) {
    const pairs = new Map();
    for(const pair of series.pairs || []) {
      const phase = value(pair.phase);
      if(Number.isInteger(phase) && phase > 0)pairs.set(phase, pair);
    }
    const indices = new Set(pairs.keys());
    for(const phase of series.missing_phases || []) {
      const index = value(phase);
      if(Number.isInteger(index) && index > 0)indices.add(index);
    }
    const count = value(series.n_phases);
    if(Number.isInteger(count))for(let phase = 1; phase < count; phase++)indices.add(phase);
    return [...indices].sort((a,b) => a-b).map(phase => ({phase, pair:pairs.get(phase)}));
  }

  function scale(allSeries, definitions) {
    const values = allSeries.flatMap(series => (series.pairs || []).flatMap(pair =>
      definitions.map(trace => value(pair[trace.key])).filter(v => v !== null)));
    const minimum = Math.min(0, ...values), maximum = Math.max(0, ...values);
    const rough = (maximum - minimum || 1) / 4;
    const base = 10 ** Math.floor(Math.log10(rough));
    const step = [1,2,5,10].find(v => v*base >= rough)*base;
    const low = Math.floor(minimum/step)*step;
    const high = Math.ceil(maximum/step)*step || step*4;
    return {low, high:Math.max(high, low+step), step,
            digits:Math.max(0, -Math.floor(Math.log10(step)))};
  }

  function chart(rows, definitions, bounds, title, unit, choose) {
    const figure = element("figure", undefined, "motion-chart");
    figure.append(element("figcaption", title));
    const legend = element("div", undefined, "motion-legend");
    for(const trace of definitions) {
      const label = element("span"), swatch = element("i");
      swatch.style.borderColor = trace.colour;
      if(trace.dashed)swatch.style.borderTopStyle = "dashed";
      label.append(swatch, document.createTextNode(trace.label));
      legend.append(label);
    }
    figure.append(legend);
    const width = 640, height = 240, left = 56, right = 24, top = 16, bottom = 42;
    const x = index => left+(width-left-right)*(rows.length === 1 ? .5 : index/(rows.length-1));
    const y = measurement => height-bottom-(height-top-bottom)*(measurement-bounds.low)/(bounds.high-bounds.low);
    const svg = svgElement("svg", {viewBox:`0 0 ${width} ${height}`, role:"group",
      "aria-label":`${title} by phase, in ${unit}; each phase compared with the reference`});
    for(let tick = bounds.low; tick <= bounds.high+bounds.step/2; tick += bounds.step) {
      svg.append(svgElement("line", {x1:left, x2:width-right, y1:y(tick), y2:y(tick), class:"motion-grid"}));
      svg.append(svgElement("text", {x:left-10, y:y(tick)+4, "text-anchor":"end", class:"motion-axis"},
        tick.toFixed(bounds.digits)));
    }
    svg.append(svgElement("text", {x:left, y:height-5, class:"motion-axis"}, "Phase"));
    for(const trace of definitions) {
      let path = "", connected = false;
      rows.forEach((row, index) => {
        const measurement = value(row.pair?.[trace.key]);
        if(measurement === null) { connected = false; return; }
        path += `${connected ? " L" : " M"}${x(index)} ${y(measurement)}`;
        connected = true;
      });
      svg.append(svgElement("path", {d:path, fill:"none", stroke:trace.colour,
        "stroke-width":2.5, "stroke-dasharray":trace.dashed ? "7 5" : "", class:"motion-line"}));
    }
    rows.forEach((row, index) => {
      const available = definitions.some(trace => value(row.pair?.[trace.key]) !== null);
      const description = definitions.map(trace => `${trace.label}: ${format(row.pair?.[trace.key])} ${unit}`).join(", ");
      const point = svgElement("g", {role:"button", tabindex:0, "data-phase":row.phase,
        "aria-label":`Phase ${row.phase}. ${available ? description : "Measurement unavailable"}`,
        class:"motion-point"});
      point.append(svgElement("title", {}, `Phase ${row.phase} · ${description}`));
      point.append(svgElement("rect", {x:x(index)-14, y:top, width:28,
        height:height-top-bottom+22, fill:"transparent", class:"motion-hit"}));
      point.append(svgElement("line", {x1:x(index), x2:x(index), y1:top, y2:height-bottom,
        class:"motion-cursor"}));
      for(const trace of definitions) {
        const measurement = value(row.pair?.[trace.key]);
        if(measurement !== null)point.append(svgElement("circle", {cx:x(index), cy:y(measurement),
          r:5, fill:trace.colour, stroke:"#15212d", "stroke-width":2}));
      }
      if(!available)point.append(svgElement("text", {x:x(index), y:height-bottom-8,
        "text-anchor":"middle", class:"motion-missing"}, "×"));
      point.append(svgElement("text", {x:x(index), y:height-bottom+20,
        "text-anchor":"middle", class:"motion-axis"}, String(row.phase)));
      point.onclick = () => choose(row.phase);
      point.onkeydown = event => {
        if(event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(row.phase); }
      };
      svg.append(point);
    });
    figure.append(svg);
    return figure;
  }

  function render(container, series, allSeries) {
    const rows = phases(series);
    if(!rows.length) { container.append(element("p", "No phase measurements available.", "hint")); return; }
    const card = element("div", undefined, "motion-chart-card");
    const recorded = rows.filter(row => value(row.pair?.mean_mm) !== null);
    const worst = recorded.reduce((best, row) => !best || value(row.pair.mean_mm) > value(best.pair.mean_mm) ? row : best, null);
    if(worst)card.append(element("p", `Largest mean displacement: ${format(worst.pair.mean_mm)} mm · Phase ${worst.phase}`, "motion-peak"));
    const readout = element("div", undefined, "motion-readout");
    readout.setAttribute("aria-live", "polite");
    function choose(phase) {
      const row = rows.find(row => row.phase === phase);
      readout.replaceChildren(element("strong", `Phase ${phase}`));
      if(!row.pair)readout.append(element("span", "Unavailable · no registration measurement recorded", "warnings"));
      else for(const [label, key, unit] of [["Mean", "mean_mm", "mm"], ["95th percentile", "p95_mm", "mm"], ["Rotation", "rotation_deg", "°"]]) {
        readout.append(element("span", `${label}: ${format(row.pair[key])}${value(row.pair[key]) === null ? "" : " "+unit}`));
      }
      for(const node of card.querySelectorAll("[data-phase]")) {
        const active = Number(node.dataset.phase) === phase;
        node.setAttribute("data-selected", String(active));
        node.setAttribute("aria-pressed", String(active));
      }
    }
    const plots = element("div", undefined, "motion-plots");
    plots.append(chart(rows, traces.displacement, scale(allSeries, traces.displacement), "Displacement (mm)", "mm", choose),
      chart(rows, traces.rotation, scale(allSeries, traces.rotation), "Rotation (°)", "°", choose));
    const choices = element("div", undefined, "motion-phases");
    for(const row of rows) {
      const button = element("button", `Phase ${row.phase}${measured(row) ? "" : " · unavailable"}`);
      button.type = "button";
      button.dataset.phase = row.phase;
      button.onclick = () => choose(row.phase);
      choices.append(button);
    }
    card.append(plots, choices, readout);
    const missing = rows.filter(row => !measured(row)).map(row => row.phase);
    if(missing.length)card.append(element("p", `Unavailable phases: ${missing.join(", ")} · gaps are not zero movement.`, "warnings"));
    card.append(element("p", "Click a point or phase to inspect its values. The 95th percentile describes displacement across the ROI voxels.", "hint"));
    container.append(card);
    choose(worst?.phase || rows[0].phase);
  }
  return {render};
})();
