/* ROMP frontend — fetch metric endpoints and render with Plotly. */

const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#d9e1ec", family: "inherit", size: 12 },
  margin: { l: 50, r: 20, t: 30, b: 40 },
};

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function setStatus(text, cls) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = "status " + (cls || "");
}

function fmtKm2day(v) {
  return (v / 1e6).toFixed(1) + " · 10⁶ km²·day";
}

async function loadSummary() {
  const f = await fetchJSON("/api/fields");
  document.getElementById("init-info").textContent =
    `AIFS init #${f.aifs_init_idx}  ·  NGCM init #${f.ngcm_init_idx} (${f.ens_members} members)`;
  document.getElementById("range-info").textContent =
    `obs ${f.obs_range.map(Math.round).join("–")}  ·  AIFS ${f.aifs_range.map(Math.round).join("–")}  ·  NGCM ${f.ngcm_range.map(Math.round).join("–")} DOY`;
  document.getElementById("iso-info").textContent = f.iso_days.join(", ");
  return f;
}

function plotIsochrones(iso, fields) {
  const traces = [];
  // background: observed-onset DOY heatmap
  traces.push({
    z: fields.obs.values, x: fields.obs.lon, y: fields.obs.lat,
    type: "heatmap",
    colorscale: "Viridis",
    opacity: 0.55,
    colorbar: { title: "Obs DOY", thickness: 10 },
    name: "Obs DOY",
    hoverinfo: "skip",
  });

  const palette = ["#4da3ff", "#7be495", "#ff8a5c", "#c084fc", "#f9d57c"];
  iso.isochrones.forEach((entry, idx) => {
    const color = palette[idx % palette.length];
    entry.forecast.forEach((seg, k) => {
      traces.push({
        x: seg.map(p => p[0]), y: seg.map(p => p[1]),
        mode: "lines", type: "scatter",
        line: { color, width: 2.5 },
        name: `fcst DOY ${entry.day}`,
        legendgroup: `fcst-${entry.day}`,
        showlegend: k === 0,
      });
    });
    entry.observed.forEach((seg, k) => {
      traces.push({
        x: seg.map(p => p[0]), y: seg.map(p => p[1]),
        mode: "lines", type: "scatter",
        line: { color, width: 2.5, dash: "dash" },
        name: `obs DOY ${entry.day}`,
        legendgroup: `obs-${entry.day}`,
        showlegend: k === 0,
      });
    });
  });

  Plotly.newPlot("plot-isochrones", traces, {
    ...PLOT_LAYOUT,
    xaxis: { title: "Longitude", gridcolor: "#223042" },
    yaxis: { title: "Latitude", gridcolor: "#223042", scaleanchor: "x" },
    legend: { bgcolor: "rgba(0,0,0,0)", orientation: "h", y: -0.15 },
  }, { displaylogo: false, responsive: true });

  document.getElementById("iso-distances").innerHTML = iso.days.map((d, i) => {
    const h = iso.hausdorff_km[i], f = iso.frechet_km[i];
    const hs = h == null ? "—" : `${h.toFixed(0)} km`;
    const fs = f == null ? "—" : `${f.toFixed(0)} km`;
    const segs = `${iso.n_segments_fcst[i]}f / ${iso.n_segments_obs[i]}o segs`;
    return `DOY ${d}: Hausdorff ${hs}, Fréchet ${fs} <span style="opacity:.6">(${segs})</span>`;
  }).join(" &nbsp;·&nbsp; ");
}

function plotProgression(p) {
  const toM = xs => xs.map(v => v == null ? null : v / 1e6);
  const traces = [
    { x: p.days, y: toM(p.ioe_km2), name: "IOE (det)", mode: "lines+markers",
      line: { color: "#4da3ff", width: 2 } },
    { x: p.days, y: toM(p.extent_km2), name: "extent", mode: "lines",
      line: { color: "#7be495", width: 1.5, dash: "dash" } },
    { x: p.days, y: toM(p.misplacement_km2), name: "misplacement", mode: "lines",
      line: { color: "#ff8a5c", width: 1.5, dash: "dot" } },
    { x: p.days, y: toM(p.sps_km2), name: "SPS (ens)", mode: "lines+markers",
      line: { color: "#c084fc", width: 2 } },
  ];
  Plotly.newPlot("plot-progression", traces, {
    ...PLOT_LAYOUT,
    xaxis: { title: "DOY", gridcolor: "#223042" },
    yaxis: { title: "Area (10⁶ km²)", gridcolor: "#223042" },
    legend: { bgcolor: "rgba(0,0,0,0)", orientation: "h", y: -0.2 },
    title: {
      text: `season IOE ${fmtKm2day(p.season.ioe_km2_day)} · SPS ${fmtKm2day(p.season.sps_km2_day)}`,
      font: { size: 12, color: "#9fb0c4" }, x: 0, xanchor: "left",
    },
  }, { displaylogo: false, responsive: true });
}

function plotCORP(c) {
  const ideal = { x: [0, 1], y: [0, 1], mode: "lines",
    line: { color: "#7a8699", dash: "dash", width: 1 }, name: "perfect" };
  const curve = { x: c.curve.forecast_prob, y: c.curve.calibrated_prob,
    mode: "lines+markers", line: { color: "#4da3ff", width: 2 },
    marker: { size: 6 }, name: "CORP" };
  // histogram of forecast frequencies behind calibration curve
  const freq = {
    x: c.forecast_prob_histogram, type: "histogram",
    xbins: { start: 0, end: 1, size: 0.05 },
    yaxis: "y2", marker: { color: "#334055", opacity: 0.45 },
    name: "freq", showlegend: false,
  };
  Plotly.newPlot("plot-corp", [freq, ideal, curve], {
    ...PLOT_LAYOUT,
    xaxis: { title: "Forecast probability", range: [0, 1], gridcolor: "#223042" },
    yaxis: { title: "Calibrated", range: [0, 1], gridcolor: "#223042" },
    yaxis2: { overlaying: "y", side: "right", showgrid: false, tickfont: { color: "#6a7686" }, title: "" },
    legend: { bgcolor: "rgba(0,0,0,0)", orientation: "h", y: -0.2 },
  }, { displaylogo: false, responsive: true });

  document.getElementById("corp-numbers").innerHTML =
    `τ = ${c.tau} · BS ${c.mean_score.toFixed(3)} · MCB ${c.mcb.toFixed(3)} ·
     DSC ${c.dsc.toFixed(3)} · UNC ${c.unc.toFixed(3)} ·
     residual ${c.identity_residual.toExponential(1)}`;
}

function plotCRPS(m) {
  Plotly.newPlot("plot-crps", [{
    z: m.field.values, x: m.field.lon, y: m.field.lat,
    type: "heatmap", colorscale: "Magma",
    colorbar: { title: "CRPS (days)", thickness: 10 },
    hovertemplate: "lat %{y:.1f}, lon %{x:.1f}<br>CRPS %{z:.1f} d<extra></extra>",
  }], {
    ...PLOT_LAYOUT,
    xaxis: { title: "Longitude", gridcolor: "#223042" },
    yaxis: { title: "Latitude", gridcolor: "#223042", scaleanchor: "x" },
    title: {
      text: `mean ${m.mean.toFixed(1)} d · max ${m.max.toFixed(1)} d · ${m.n_finite} cells`,
      font: { size: 12, color: "#9fb0c4" }, x: 0, xanchor: "left",
    },
  }, { displaylogo: false, responsive: true });
}

function plotDisplacement(d) {
  Plotly.newPlot("plot-displacement", [
    { x: d.thresholds, y: d.great_circle_km, name: "centroid shift (km)",
      mode: "lines+markers", line: { color: "#4da3ff", width: 2 },
      yaxis: "y1" },
    { x: d.thresholds, y: d.area_bias_fraction.map(v => v == null ? null : 100 * v),
      name: "area bias (%)", mode: "lines+markers",
      line: { color: "#ff8a5c", width: 2, dash: "dot" }, yaxis: "y2" },
  ], {
    ...PLOT_LAYOUT,
    xaxis: { title: "Onset-by-day threshold (DOY)", gridcolor: "#223042" },
    yaxis: { title: "Centroid shift (km)", gridcolor: "#223042" },
    yaxis2: { overlaying: "y", side: "right", title: "Area bias (%)", gridcolor: "#223042" },
    legend: { bgcolor: "rgba(0,0,0,0)", orientation: "h", y: -0.2 },
  }, { displaylogo: false, responsive: true });
}

async function main() {
  setStatus("loading…");
  try {
    const fields = await loadSummary();
    const [iso, prog, corp, crps, disp] = await Promise.all([
      fetchJSON("/api/metrics/isochrones"),
      fetchJSON("/api/metrics/progression"),
      fetchJSON("/api/metrics/corp"),
      fetchJSON("/api/metrics/crps"),
      fetchJSON("/api/metrics/displacement"),
    ]);
    plotIsochrones(iso, fields);
    plotProgression(prog);
    plotCORP(corp);
    plotCRPS(crps);
    plotDisplacement(disp);
    setStatus("ready", "ok");
  } catch (err) {
    console.error(err);
    setStatus("error: " + err.message, "err");
  }
}

main();
