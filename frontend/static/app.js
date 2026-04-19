/* ROMP Onset Observatory — multi-model verification dashboard.
 * Plain vanilla JS; loaded after Plotly CDN.
 */
"use strict";

/* ------------------------------------------------------------------ *
 * Constants, palettes, theming
 * ------------------------------------------------------------------ */

const MODEL_PALETTE = [
  "#f0b264", // amber   (primary)
  "#6eb7ff", // sky
  "#b697e0", // violet
  "#86b97d", // green
  "#e87b85", // rose
  "#e8d06f", // sand
];

const ISO_DAY_PALETTE = ["#6eb7ff", "#86b97d", "#f0b264", "#e87b85"];

const FSS_COLORSCALE = [
  [0.0, "#11171f"],
  [0.4, "#223044"],
  [0.6, "#6eb7ff"],
  [1.0, "#f0b264"],
];

const PARAM_ORDER = [
  "wet_init",
  "wet_spell",
  "wet_threshold",
  "dry_spell",
  "dry_threshold",
  "dry_extent",
];

const PARAM_STEP = {
  wet_init: 1,
  wet_spell: 1,
  wet_threshold: 0.5,
  dry_spell: 1,
  dry_threshold: 0.5,
  dry_extent: 1,
};

const REGION_KEYS = ["lat_min", "lat_max", "lon_min", "lon_max"];

const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(10,16,25,0.4)",
  font: { family: "IBM Plex Sans, system-ui, sans-serif", color: "#e4ddc9", size: 12 },
  margin: { l: 54, r: 20, t: 34, b: 42 },
  hoverlabel: {
    bgcolor: "#131e2a",
    bordercolor: "#f0b264",
    font: { family: "IBM Plex Mono, ui-monospace", color: "#e4ddc9", size: 11 },
  },
  xaxis: {
    gridcolor: "#1a2536",
    zerolinecolor: "#223044",
    linecolor: "#223044",
    tickfont: { family: "IBM Plex Mono, ui-monospace", size: 10, color: "#a8a291" },
  },
  yaxis: {
    gridcolor: "#1a2536",
    zerolinecolor: "#223044",
    linecolor: "#223044",
    tickfont: { family: "IBM Plex Mono, ui-monospace", size: 10, color: "#a8a291" },
  },
  legend: {
    bgcolor: "rgba(0,0,0,0)",
    font: { size: 11 },
    orientation: "h",
    y: -0.22,
  },
};

const PLOT_CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
};

function mergeLayout(...parts) {
  // Shallow-deep merge for Plotly layouts. One level of nesting is enough.
  const out = {};
  for (const p of parts) {
    for (const k of Object.keys(p || {})) {
      const v = p[k];
      if (v && typeof v === "object" && !Array.isArray(v)) {
        out[k] = Object.assign({}, out[k] || {}, v);
      } else {
        out[k] = v;
      }
    }
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * API helpers
 * ------------------------------------------------------------------ */

async function apiGet(path, params) {
  const url = params ? `${path}?${params}` : path;
  const res = await fetch(url);
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body && body.detail ? ` — ${body.detail}` : "";
    } catch (e) {
      /* ignore */
    }
    throw new Error(`${url}: HTTP ${res.status}${detail}`);
  }
  return res.json();
}

function qs(obj) {
  const parts = [];
  for (const [k, v] of Object.entries(obj)) {
    if (v === undefined || v === null || v === "") continue;
    parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(v)}`);
  }
  return parts.join("&");
}

/* ------------------------------------------------------------------ *
 * Global state
 * ------------------------------------------------------------------ */

const state = {
  catalog: null,
  yearFrom: null,
  yearTo: null,
  isoYear: null,
  init: "auto",
  initIdx: null,
  models: [],           // array of {key,label,is_ensemble,n_members,on,primary}
  modelColor: {},       // key -> hex color
  params: {},
  region: { lat_min: "", lat_max: "", lon_min: "", lon_max: "" },
  busy: false,
  progressionShowDecomp: false,
  plotDivs: new Set(),
};

function selectedYears() {
  // Inclusive [from, to] within the primary model's available years.
  const avail = yearsForActiveSelection();
  if (!avail.length) return [];
  const a = state.yearFrom ?? avail[0];
  const b = state.yearTo ?? avail[avail.length - 1];
  const lo = Math.min(a, b), hi = Math.max(a, b);
  return avail.filter(y => y >= lo && y <= hi);
}

/* ------------------------------------------------------------------ *
 * Small UI helpers
 * ------------------------------------------------------------------ */

function $(id) { return document.getElementById(id); }

function setLoading(el, on) {
  if (!el) return;
  if (on) el.classList.add("is-loading");
  else el.classList.remove("is-loading");
}

function setStatus(text, kind /* "ok" | "err" | "" */) {
  const el = $("meta-status");
  if (!el) return;
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
}

function fmt(n, d) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(d);
}

function fmtE6(n, d = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return (n / 1e6).toFixed(d);
}

function fmtKm(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 100) return Math.round(n).toString();
  return n.toFixed(1);
}

function colorForModel(key) {
  if (state.modelColor[key]) return state.modelColor[key];
  const idx = Object.keys(state.modelColor).length % MODEL_PALETTE.length;
  state.modelColor[key] = MODEL_PALETTE[idx];
  return state.modelColor[key];
}

function primaryModelKey() {
  const on = state.models.filter(m => m.on);
  return on.length ? on[0].key : null;
}

function activeModelKeys() {
  return state.models.filter(m => m.on).map(m => m.key);
}

/* ------------------------------------------------------------------ *
 * buildParams — common query-string for every metric call
 * ------------------------------------------------------------------ */

function readParamsFromInputs() {
  const out = {};
  for (const p of PARAM_ORDER) {
    const el = document.querySelector(`#params input[data-key="${p}"]`);
    if (!el) continue;
    const v = el.value === "" ? null : Number(el.value);
    if (v !== null && !Number.isNaN(v)) out[p] = v;
  }
  return out;
}

function readRegionFromInputs() {
  const out = {};
  for (const k of REGION_KEYS) {
    const el = $(k);
    if (!el) continue;
    const raw = el.value;
    if (raw === "" || raw === null) continue;
    const v = Number(raw);
    if (!Number.isNaN(v)) out[k] = v;
  }
  return out;
}

function buildParams(extra) {
  // core: onset params + region (region only if provided)
  const p = readParamsFromInputs();
  const r = readRegionFromInputs();
  state.params = p;
  state.region = Object.assign({ lat_min: "", lat_max: "", lon_min: "", lon_max: "" }, r);
  return qs(Object.assign({}, p, r, extra || {}));
}

function metricParamsCSV(extra) {
  // years range for aggregated metrics (CRPS / FSS / displacement / progression / corp / compare)
  const years = selectedYears();
  const yearsArg = years.length
    ? (years.length === 1 ? { year: years[0] } : { years: `${years[0]}-${years[years.length - 1]}` })
    : {};
  return buildParams(Object.assign({}, yearsArg, extra || {}));
}

function isoParamsCSV(extra) {
  // single year for the isochrone hero
  const yr = state.isoYear ?? selectedYears().slice(-1)[0];
  return buildParams(Object.assign({ year: yr, init: state.init || "auto" }, extra || {}));
}

/* ------------------------------------------------------------------ *
 * Sidebar population
 * ------------------------------------------------------------------ */

function yearsForActiveSelection() {
  // Years available across the union of selected models, intersected with obs.
  // If nothing is selected yet, fall back to catalog.shared_years.
  const obsYears = new Set((state.catalog.obs && state.catalog.obs.years) || []);
  const active = state.models.filter(m => m.on);
  if (!active.length) return state.catalog.shared_years || [];
  const primary = active[0];
  const primaryYears = new Set(primary.years);
  const filtered = (state.catalog.shared_years || [])
    .filter(y => primaryYears.has(y) && obsYears.has(y));
  return filtered;
}

function fillYearOptions(sel, years, currentValue) {
  sel.innerHTML = "";
  if (!years.length) {
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = "—"; opt.disabled = true;
    sel.appendChild(opt);
    return null;
  }
  for (const y of years) {
    const opt = document.createElement("option");
    opt.value = String(y); opt.textContent = String(y);
    sel.appendChild(opt);
  }
  const has = (v) => years.map(String).includes(String(v));
  if (currentValue !== null && currentValue !== undefined && has(currentValue)) {
    sel.value = String(currentValue);
    return Number(currentValue);
  }
  return Number(sel.value);
}

function populateYearSelect() {
  const years = yearsForActiveSelection();
  const fromSel = $("year_from"), toSel = $("year_to"), isoSel = $("iso_year");
  if (!years.length) {
    [fromSel, toSel, isoSel].forEach(s => {
      s.innerHTML = '<option value="" disabled>no overlap</option>';
    });
    state.yearFrom = state.yearTo = state.isoYear = null;
    return;
  }
  // default range: last 5 years (or all if fewer)
  const defFrom = state.yearFrom ?? years[Math.max(0, years.length - 5)];
  const defTo   = state.yearTo   ?? years[years.length - 1];
  state.yearFrom = fillYearOptions(fromSel, years, defFrom);
  state.yearTo   = fillYearOptions(toSel, years, defTo);
  // iso year: middle of currently-selected range
  const sel = selectedYears();
  const defIso = state.isoYear && sel.includes(state.isoYear)
    ? state.isoYear : sel[Math.floor(sel.length / 2)];
  state.isoYear = fillYearOptions(isoSel, sel.length ? sel : years, defIso);
  refreshYearHint();
}

function refreshYearHint() {
  const yrs = selectedYears();
  const el = $("year-hint");
  if (!el) return;
  if (!yrs.length) { el.textContent = "no overlap with primary model"; return; }
  if (yrs.length === 1) el.textContent = "single year — no aggregation";
  else el.textContent = `aggregating ${yrs.length} years (${yrs[0]}–${yrs[yrs.length - 1]}); medians + IQR`;
}

function renderModelChips() {
  const container = $("models");
  container.innerHTML = "";
  state.models.forEach((m, idx) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.dataset.key = m.key;
    b.textContent = m.label + (m.is_ensemble ? ` · ${m.n_members}m` : "");
    b.title = m.is_ensemble
      ? `${m.label}: ${m.n_members}-member ensemble. Click to include/exclude. Leftmost active chip is the primary model (hero panels only use primary).`
      : `${m.label}: deterministic. Click to include/exclude. Leftmost active chip is the primary model.`;
    b.addEventListener("click", () => {
      const model = state.models.find(x => x.key === m.key);
      model.on = !model.on;
      applyChipClasses();
      // Year set depends on the (new) primary model's coverage.
      populateYearSelect();
      refresh();
    });
    container.appendChild(b);
    // pre-assign color by initial order
    colorForModel(m.key);
  });
  applyChipClasses();
}

function applyChipClasses() {
  // leftmost ON becomes primary; others ON become is-on; off = neither.
  const container = $("models");
  let primaryAssigned = false;
  const chips = Array.from(container.children);
  chips.forEach((chip) => {
    const key = chip.dataset.key;
    const model = state.models.find(x => x.key === key);
    chip.classList.remove("is-primary", "is-on");
    if (model && model.on) {
      if (!primaryAssigned) {
        chip.classList.add("is-primary");
        model.primary = true;
        primaryAssigned = true;
      } else {
        chip.classList.add("is-on");
        model.primary = false;
      }
      // tint chip with model color
      chip.style.setProperty("--chip-accent", colorForModel(key));
    } else {
      model.primary = false;
      chip.style.removeProperty("--chip-accent");
    }
  });
}

function renderParamInputs() {
  const container = $("params");
  container.innerHTML = "";
  const defaults = state.catalog.onset_defaults || {};
  const docs = state.catalog.onset_docs || {};
  for (const key of PARAM_ORDER) {
    const label = document.createElement("label");
    label.title = docs[key] || key;
    const span = document.createElement("span");
    span.textContent = key;
    const input = document.createElement("input");
    input.type = "number";
    input.step = String(PARAM_STEP[key] || 0.1);
    input.dataset.key = key;
    if (defaults[key] !== undefined && defaults[key] !== null) {
      input.value = defaults[key];
    }
    input.addEventListener("change", () => {
      const btn = $("apply");
      btn.classList.add("is-dirty");
    });
    label.appendChild(span);
    label.appendChild(input);
    container.appendChild(label);
  }
}

function wireRegionInputs() {
  for (const k of REGION_KEYS) {
    const el = $(k);
    if (!el) continue;
    el.addEventListener("change", () => {
      const btn = $("apply");
      btn.classList.add("is-dirty");
    });
  }
}

/* ------------------------------------------------------------------ *
 * Init dropdown
 * ------------------------------------------------------------------ */

async function refreshInitOptions(primaryKey, year) {
  const sel = $("init");
  const currentVal = sel.value;
  sel.innerHTML = "";
  const auto = document.createElement("option");
  auto.value = "auto";
  auto.textContent = "auto-select overlap";
  sel.appendChild(auto);

  if (!primaryKey || !year) {
    sel.value = "auto";
    $("init-hint").textContent = "pick a primary model to load inits";
    return;
  }

  try {
    const data = await apiGet("/api/inits", qs({ model: primaryKey, year }));
    (data.inits || []).forEach((iso, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      // "2015-04-01T00:00:00" -> "#03 · 2015-04-01"
      const pretty = iso.slice(0, 10);
      opt.textContent = `#${String(i).padStart(2, "0")} · ${pretty}`;
      sel.appendChild(opt);
    });
    // preserve selection if still valid, else reset to auto
    if (currentVal && Array.from(sel.options).some(o => o.value === currentVal)) {
      sel.value = currentVal;
    } else {
      sel.value = "auto";
    }
    state.init = sel.value;
    $("init-hint").textContent = `auto picks whichever of ${data.n} init${data.n === 1 ? "" : "s"} overlaps obs`;
  } catch (e) {
    console.error("inits fetch failed", e);
    $("init-hint").textContent = "init list unavailable";
  }
}

/* ------------------------------------------------------------------ *
 * Summary table
 * ------------------------------------------------------------------ */

function renderSummaryTable(compare) {
  const tbl = $("summary-table");
  tbl.innerHTML = "";
  $("summary-year").textContent = compare && compare.year ? compare.year : "—";

  const heads = ["Model", "yrs / mem", "IOE · 10⁶km²d", "SPS · 10⁶km²d", "CRPS · d", "Brier", "MCB / DSC"];
  for (const h of heads) {
    const d = document.createElement("div");
    d.className = "col-head";
    d.textContent = h;
    tbl.appendChild(d);
  }

  const rows = (compare && compare.rows) || [];
  // preserve the order compare gave us (primary first in compare.rows)
  rows.forEach((row, idx) => {
    const accent = colorForModel(row.model);
    const rowHead = document.createElement("div");
    rowHead.className = "row-head";
    rowHead.style.setProperty("--row-accent", accent);
    const rowHeadText = document.createElement("span");
    rowHeadText.className = "row-head-text";
    rowHeadText.textContent = row.label || row.model;
    rowHead.appendChild(rowHeadText);
    tbl.appendChild(rowHead);

    const members = document.createElement("div");
    members.className = "cell";
    const memTxt = row.is_ensemble ? `${row.n_members}m` : "det";
    members.textContent = (row.n_years && row.n_years > 1)
      ? `${row.n_years}y / ${memTxt}` : memTxt;
    tbl.appendChild(members);

    const season = (row.progression && row.progression.season) || {};
    const ioeCell = document.createElement("div");
    ioeCell.className = "cell" + (idx === 0 ? " primary" : "");
    const ioe = season.ioe_km2_day;
    const ioeQ25 = season.ioe_km2_day_q25, ioeQ75 = season.ioe_km2_day_q75;
    ioeCell.innerHTML = fmtE6(ioe, 1) +
      (ioeQ25 !== undefined && ioeQ25 !== null
        ? ` <span class="cell-iqr">[${fmtE6(ioeQ25,1)}–${fmtE6(ioeQ75,1)}]</span>` : "");
    tbl.appendChild(ioeCell);

    const spsCell = document.createElement("div");
    spsCell.className = "cell";
    const sps = season.sps_km2_day;
    const spsQ25 = season.sps_km2_day_q25, spsQ75 = season.sps_km2_day_q75;
    if (!row.is_ensemble || sps === null || sps === undefined) {
      spsCell.textContent = "—";
    } else {
      spsCell.innerHTML = fmtE6(sps, 1) +
        (spsQ25 !== undefined && spsQ25 !== null
          ? ` <span class="cell-iqr">[${fmtE6(spsQ25,1)}–${fmtE6(spsQ75,1)}]</span>` : "");
    }
    tbl.appendChild(spsCell);

    const crpsCell = document.createElement("div");
    crpsCell.className = "cell";
    crpsCell.textContent = row.crps && row.crps.mean !== null ? fmt(row.crps.mean, 1) : "—";
    tbl.appendChild(crpsCell);

    const bsCell = document.createElement("div");
    bsCell.className = "cell";
    bsCell.textContent = row.corp && row.corp.bs !== null ? fmt(row.corp.bs, 3) : "—";
    tbl.appendChild(bsCell);

    const mdCell = document.createElement("div");
    mdCell.className = "cell";
    if (row.corp && row.corp.mcb !== null && row.corp.dsc !== null) {
      mdCell.textContent = `${fmt(row.corp.mcb, 3)} / ${fmt(row.corp.dsc, 3)}`;
    } else {
      mdCell.textContent = "—";
    }
    tbl.appendChild(mdCell);
  });

  if (!rows.length) {
    const msg = document.createElement("div");
    msg.className = "summary-placeholder";
    msg.textContent = "no rows returned";
    tbl.appendChild(msg);
  }
}

/* ------------------------------------------------------------------ *
 * Plot renderers
 * ------------------------------------------------------------------ */

function rememberPlot(div) { state.plotDivs.add(div); }

/* -- Isochrones (hero) -- */

function renderIsochrones(state_, iso, primaryLabel) {
  const traces = [];

  // Background: observed-onset DOY heatmap
  if (state_ && state_.obs_onset) {
    traces.push({
      type: "heatmap",
      x: state_.obs_onset.lon,
      y: state_.obs_onset.lat,
      z: state_.obs_onset.values,
      colorscale: "Viridis",
      opacity: 0.55,
      colorbar: {
        title: { text: "Obs DOY", font: { size: 10, color: "#a8a291" } },
        thickness: 10,
        len: 0.6,
        x: 1.02,
        tickfont: { size: 9, color: "#a8a291" },
      },
      hovertemplate: "lon %{x:.2f} · lat %{y:.2f}<br>obs DOY %{z:.1f}<extra></extra>",
      showscale: true,
      name: "obs DOY",
    });
  }

  const days = (iso && iso.isochrones) || [];
  days.forEach((entry, i) => {
    const color = ISO_DAY_PALETTE[i % ISO_DAY_PALETTE.length];
    const grp = `day-${entry.day}`;
    const pushSegs = (segs, dash, tag) => {
      (segs || []).forEach((seg, j) => {
        if (!seg || !seg.length) return;
        const xs = seg.map(pt => pt[0]);
        const ys = seg.map(pt => pt[1]);
        traces.push({
          type: "scatter",
          mode: "lines",
          x: xs,
          y: ys,
          line: { color, width: dash === "dash" ? 2.5 : 3.2, dash },
          opacity: dash === "dash" ? 0.95 : 1,
          name: `DOY ${entry.day} · ${tag}`,
          legendgroup: grp,
          showlegend: j === 0,
          hovertemplate: `DOY ${entry.day} ${tag}<br>lon %{x:.2f}, lat %{y:.2f}<extra></extra>`,
        });
      });
    };
    pushSegs(entry.forecast, "solid", "fcst");
    pushSegs(entry.observed, "dash", "obs");
  });

  const layout = mergeLayout(PLOT_LAYOUT, {
    title: {
      text: primaryLabel
        ? `${primaryLabel} · ${(iso && iso.year) || ""} · obs DOY behind, contours = onset front by day`
        : "isochrones",
      font: { family: "Fraunces, serif", color: "#e8e1cf", size: 14 },
      x: 0.01,
      xanchor: "left",
    },
    xaxis: { title: { text: "Longitude", font: { size: 11, color: "#a8a291" } }, scaleanchor: "y" },
    yaxis: { title: { text: "Latitude", font: { size: 11, color: "#a8a291" } } },
    margin: { l: 58, r: 90, t: 46, b: 54 },
    showlegend: true,
  });
  // enforce scaleanchor via xaxis -> y (Plotly uses scaleanchor on x referencing y)
  layout.xaxis.scaleanchor = "y";

  const div = $("plot-isochrones");
  Plotly.react(div, traces, layout, PLOT_CONFIG);
  rememberPlot(div);

  // Distance footer
  const foot = $("iso-distances");
  const parts = [];
  if (iso && iso.days && iso.days.length) {
    iso.days.forEach((d, i) => {
      const h = iso.hausdorff_km ? iso.hausdorff_km[i] : null;
      const f = iso.frechet_km ? iso.frechet_km[i] : null;
      const nf = iso.n_segments_fcst ? iso.n_segments_fcst[i] : 0;
      const no = iso.n_segments_obs ? iso.n_segments_obs[i] : 0;
      parts.push(`DOY ${d}: Hausdorff ${fmtKm(h)} km, Fréchet ${fmtKm(f)} km (${nf}f/${no}o segs)`);
    });
  }
  foot.textContent = parts.join("  ·  ") || "no isochrone days returned";
}

/* -- Progression curves (all models) -- */

function renderProgression(compare) {
  const traces = [];
  const rows = (compare && compare.rows) || [];
  const multiYear = (compare && compare.n_years > 1);

  function rgba(hex, a) {
    const h = hex.replace("#", "");
    const r = parseInt(h.slice(0, 2), 16),
          g = parseInt(h.slice(2, 4), 16),
          b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${a})`;
  }

  rows.forEach((row) => {
    const color = colorForModel(row.model);
    const p = row.progression || {};
    const days = p.days || [];

    if (Array.isArray(p.ioe_km2)) {
      // IQR band first (so the median line draws on top)
      if (multiYear && Array.isArray(p.ioe_km2_q25) && Array.isArray(p.ioe_km2_q75)) {
        traces.push({
          type: "scatter", x: days,
          y: p.ioe_km2_q75.map(v => (v === null ? null : v / 1e6)),
          mode: "lines", line: { width: 0, color },
          showlegend: false, hoverinfo: "skip",
        });
        traces.push({
          type: "scatter", x: days,
          y: p.ioe_km2_q25.map(v => (v === null ? null : v / 1e6)),
          mode: "lines", line: { width: 0, color },
          fill: "tonexty", fillcolor: rgba(color, 0.15),
          name: `${row.label} · IQR`,
          showlegend: false, hoverinfo: "skip",
        });
      }
      traces.push({
        type: "scatter",
        mode: multiYear ? "lines" : "lines+markers",
        x: days,
        y: p.ioe_km2.map(v => (v === null ? null : v / 1e6)),
        name: `${row.label} · IOE${multiYear ? " (median)" : ""}`,
        line: { color, width: 2 },
        marker: { size: 4, color },
        connectgaps: false,
        hovertemplate: `${row.label}<br>DOY %{x}<br>IOE %{y:.2f}·10⁶ km²<extra></extra>`,
      });
    }
    if (Array.isArray(p.sps_km2) && p.sps_km2.some(v => v !== null && v !== undefined)) {
      if (multiYear && Array.isArray(p.sps_km2_q25) && Array.isArray(p.sps_km2_q75)) {
        traces.push({
          type: "scatter", x: days,
          y: p.sps_km2_q75.map(v => (v === null ? null : v / 1e6)),
          mode: "lines", line: { width: 0, color },
          showlegend: false, hoverinfo: "skip",
        });
        traces.push({
          type: "scatter", x: days,
          y: p.sps_km2_q25.map(v => (v === null ? null : v / 1e6)),
          mode: "lines", line: { width: 0, color },
          fill: "tonexty", fillcolor: rgba(color, 0.10),
          showlegend: false, hoverinfo: "skip",
        });
      }
      traces.push({
        type: "scatter",
        mode: "lines",
        x: days,
        y: p.sps_km2.map(v => (v === null ? null : v / 1e6)),
        name: `${row.label} · SPS${multiYear ? " (median)" : ""}`,
        line: { color, width: 1.5, dash: "dot" },
        connectgaps: false,
        hovertemplate: `${row.label} SPS<br>DOY %{x}<br>%{y:.2f}·10⁶ km²<extra></extra>`,
      });
    }
  });

  // Optional decomposition for primary model
  if (state.progressionShowDecomp && rows.length) {
    const primary = rows[0];
    const color = colorForModel(primary.model);
    const p = primary.progression || {};
    const days = p.days || [];
    if (Array.isArray(p.extent_km2)) {
      traces.push({
        type: "scatter",
        mode: "lines",
        x: days,
        y: p.extent_km2.map(v => (v === null ? null : v / 1e6)),
        name: `${primary.label} · extent`,
        line: { color, width: 1.2, dash: "dash" },
        opacity: 0.7,
        connectgaps: false,
      });
    }
    if (Array.isArray(p.misplacement_km2)) {
      traces.push({
        type: "scatter",
        mode: "lines",
        x: days,
        y: p.misplacement_km2.map(v => (v === null ? null : v / 1e6)),
        name: `${primary.label} · misplacement`,
        line: { color: "#e87b85", width: 1.2, dash: "dashdot" },
        opacity: 0.8,
        connectgaps: false,
      });
    }
  }

  const layout = mergeLayout(PLOT_LAYOUT, {
    xaxis: { title: { text: "Day of year", font: { size: 11, color: "#a8a291" } } },
    yaxis: { title: { text: "10⁶ km²", font: { size: 11, color: "#a8a291" } } },
    margin: { l: 56, r: 22, t: 22, b: 60 },
    legend: { orientation: "h", y: -0.26 },
  });

  const div = $("plot-progression");
  Plotly.react(div, traces, layout, PLOT_CONFIG);
  rememberPlot(div);

  // Caption + toggle chip
  const capEl = $("progression-caption");
  capEl.innerHTML = "";

  // Chip toggle for decomposition
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "mini-chip" + (state.progressionShowDecomp ? " is-on" : "");
  toggle.textContent = state.progressionShowDecomp ? "decomp: on" : "decomp";
  toggle.addEventListener("click", () => {
    state.progressionShowDecomp = !state.progressionShowDecomp;
    renderProgression(compare);
  });
  capEl.appendChild(toggle);

  // Text summary
  const seasons = rows
    .map(r => r.progression && r.progression.season ? r.progression.season.ioe_km2_day : null)
    .filter(v => v !== null && v !== undefined && !Number.isNaN(v));
  const txt = document.createElement("span");
  txt.className = "caption-text";
  const yrLabel = multiYear ? ` · ${compare.n_years} yrs (median + IQR shading)` : "";
  if (seasons.length) {
    const lo = Math.min(...seasons) / 1e6;
    const hi = Math.max(...seasons) / 1e6;
    txt.textContent = `${rows.length} model${rows.length === 1 ? "" : "s"}${yrLabel} · season IOE range ${lo.toFixed(1)}–${hi.toFixed(1)} · 10⁶ km²·d`;
  } else {
    txt.textContent = `${rows.length} model${rows.length === 1 ? "" : "s"}${yrLabel}`;
  }
  capEl.appendChild(txt);
}

/* -- CORP reliability -- */

function renderCorp(corp, primaryLabel) {
  const traces = [];

  // Histogram of forecast probabilities (behind)
  const hist = corp.forecast_prob_histogram || [];
  if (hist.length) {
    const xs = hist.map((_, i) => (i + 0.5) / hist.length);
    traces.push({
      type: "bar",
      x: xs,
      y: hist,
      name: "freq",
      marker: { color: "rgba(168,162,145,0.25)", line: { width: 0 } },
      yaxis: "y2",
      hovertemplate: "p %{x:.2f}<br>count %{y}<extra></extra>",
      showlegend: false,
      width: hist.length ? 1 / hist.length * 0.95 : 0.05,
    });
  }

  // 45° perfect reliability
  traces.push({
    type: "scatter",
    mode: "lines",
    x: [0, 1],
    y: [0, 1],
    name: "perfect",
    line: { color: "#6a7689", width: 1, dash: "dash" },
    hoverinfo: "skip",
    showlegend: false,
  });

  // CORP calibration curve
  if (corp.curve) {
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: corp.curve.forecast_prob,
      y: corp.curve.calibrated_prob,
      name: "CORP",
      line: { color: "#f0b264", width: 2.5 },
      marker: { color: "#f0b264", size: 6 },
      hovertemplate: "p_fcst %{x:.2f}<br>p_cal %{y:.2f}<extra></extra>",
    });
  }

  const layout = mergeLayout(PLOT_LAYOUT, {
    xaxis: {
      title: { text: "forecast probability", font: { size: 11, color: "#a8a291" } },
      range: [0, 1],
    },
    yaxis: {
      title: { text: "observed / calibrated", font: { size: 11, color: "#a8a291" } },
      range: [0, 1],
    },
    yaxis2: {
      overlaying: "y",
      side: "right",
      showgrid: false,
      showticklabels: false,
      zeroline: false,
      showline: false,
    },
    margin: { l: 56, r: 30, t: 18, b: 48 },
    showlegend: false,
    bargap: 0.02,
  });

  const div = $("plot-corp");
  Plotly.react(div, traces, layout, PLOT_CONFIG);
  rememberPlot(div);

  // Caption
  const tau = corp.tau;
  const n = corp.n;
  const res = corp.identity_residual;
  const resStr = (res === null || res === undefined || Number.isNaN(res))
    ? "—"
    : Number(res).toExponential(1);
  $("corp-caption").textContent = `τ = ${tau ?? "—"} · N = ${n ?? "—"} · residual = ${resStr}`;

  // Breakdown
  const breakdown = $("corp-breakdown");
  breakdown.innerHTML = "";
  const items = [
    ["bs", corp.mean_score],
    ["mcb", corp.mcb],
    ["dsc", corp.dsc],
    ["unc", corp.unc],
  ];
  for (const [k, v] of items) {
    const it = document.createElement("div");
    it.className = "corp-item";
    const l = document.createElement("span");
    l.className = "corp-label";
    l.textContent = k;
    const val = document.createElement("span");
    val.className = "corp-val";
    val.textContent = fmt(v, 3);
    it.appendChild(l);
    it.appendChild(val);
    breakdown.appendChild(it);
  }
}

/* -- CRPS field -- */

function renderCrps(crps) {
  const f = crps.field || {};
  const traces = [{
    type: "heatmap",
    x: f.lon,
    y: f.lat,
    z: f.values,
    colorscale: "Magma",
    colorbar: {
      title: { text: "CRPS (days)", font: { size: 10, color: "#a8a291" } },
      thickness: 10,
      len: 0.75,
      tickfont: { size: 9, color: "#a8a291" },
    },
    hovertemplate: "lon %{x:.2f} · lat %{y:.2f}<br>CRPS %{z:.2f} d<extra></extra>",
  }];

  const layout = mergeLayout(PLOT_LAYOUT, {
    xaxis: { title: { text: "Longitude", font: { size: 11, color: "#a8a291" } }, scaleanchor: "y" },
    yaxis: { title: { text: "Latitude", font: { size: 11, color: "#a8a291" } } },
    margin: { l: 56, r: 80, t: 18, b: 48 },
  });
  layout.xaxis.scaleanchor = "y";

  const div = $("plot-crps");
  Plotly.react(div, traces, layout, PLOT_CONFIG);
  rememberPlot(div);

  const yrs = (crps.n_years && crps.n_years > 1) ? ` · ${crps.n_years}-yr per-cell mean` : "";
  let extras = "";
  if (crps.median !== undefined && crps.median !== null) {
    extras = ` · median ${fmt(crps.median, 1)}d · IQR ${fmt(crps.q25, 1)}–${fmt(crps.q75, 1)}d`;
  }
  $("crps-caption").textContent =
    `mean ${fmt(crps.mean, 1)}d · max ${fmt(crps.max, 1)}d · ${crps.n_finite ?? "—"} cells${extras}${yrs}`;
}

/* -- Displacement dual-axis -- */

function renderDisplacement(disp) {
  const x = disp.thresholds || [];
  const km = disp.great_circle_km || [];
  const bias = (disp.area_bias_fraction || []).map(v => (v === null ? null : v * 100));

  const traces = [
    {
      type: "scatter",
      mode: "lines+markers",
      x,
      y: km,
      name: "great-circle (km)",
      line: { color: "#6eb7ff", width: 2 },
      marker: { size: 5, color: "#6eb7ff" },
      yaxis: "y",
      hovertemplate: "DOY %{x}<br>shift %{y:.1f} km<extra></extra>",
    },
    {
      type: "scatter",
      mode: "lines+markers",
      x,
      y: bias,
      name: "area bias (%)",
      line: { color: "#f0b264", width: 2, dash: "dot" },
      marker: { size: 5, color: "#f0b264" },
      yaxis: "y2",
      hovertemplate: "DOY %{x}<br>area bias %{y:.1f}%<extra></extra>",
    },
  ];

  const layout = mergeLayout(PLOT_LAYOUT, {
    xaxis: { title: { text: "Threshold (DOY)", font: { size: 11, color: "#a8a291" } } },
    yaxis: {
      title: { text: "km", font: { size: 11, color: "#6eb7ff" } },
      tickfont: { family: "IBM Plex Mono, ui-monospace", size: 10, color: "#6eb7ff" },
    },
    yaxis2: {
      title: { text: "% area bias", font: { size: 11, color: "#f0b264" } },
      overlaying: "y",
      side: "right",
      showgrid: false,
      zeroline: false,
      tickfont: { family: "IBM Plex Mono, ui-monospace", size: 10, color: "#f0b264" },
    },
    legend: { orientation: "h", y: -0.28 },
    margin: { l: 56, r: 60, t: 18, b: 60 },
  });

  const div = $("plot-displacement");
  Plotly.react(div, traces, layout, PLOT_CONFIG);
  rememberPlot(div);
}

/* -- FSS matrix -- */

function renderFss(fss) {
  const thresholds = fss.thresholds || [];
  const nbhds = fss.neighborhoods || [];
  const z = fss.fss || [];
  const baseRate = fss.base_rate || [];
  const noSkill = fss.no_skill_threshold || [];

  // annotations: per-cell value, color-coded by useful (>= 0.5) / no-skill / above no-skill
  const annotations = [];
  for (let i = 0; i < z.length; i++) {
    const ns = noSkill[i];
    for (let j = 0; j < (z[i] || []).length; j++) {
      const v = z[i][j];
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      let color = "#e4ddc9";
      let suffix = "";
      if (v > 0.6) color = "#0a1119";
      if (ns !== null && ns !== undefined) {
        if (v >= 0.5) suffix = "";
        else if (v >= ns) suffix = "·";   // above no-skill but below "useful"
        else suffix = "✕";                 // below no-skill
      }
      annotations.push({
        x: nbhds[j], y: thresholds[i],
        text: v.toFixed(2) + (suffix ? `\n${suffix}` : ""),
        font: { family: "IBM Plex Mono, ui-monospace", size: 10, color },
        showarrow: false,
      });
    }
    // Side annotation: base rate + no-skill value, on the right of the matrix
    if (baseRate[i] !== null && baseRate[i] !== undefined) {
      const txt = `p=${baseRate[i].toFixed(2)} · ns=${(noSkill[i] ?? 0).toFixed(2)}`;
      annotations.push({
        xref: "paper", x: 1.02, y: thresholds[i], yref: "y",
        text: txt, xanchor: "left",
        font: { family: "IBM Plex Mono, ui-monospace", size: 9, color: "#a8a291" },
        showarrow: false,
      });
    }
  }

  const traces = [{
    type: "heatmap",
    x: nbhds, y: thresholds, z,
    colorscale: FSS_COLORSCALE,
    zmin: 0, zmax: 1,
    colorbar: {
      title: { text: "FSS", font: { size: 10, color: "#a8a291" } },
      thickness: 10, len: 0.8,
      tickfont: { size: 9, color: "#a8a291" },
      x: 1.18,
    },
    hovertemplate: (
      "thr DOY %{y} · nbhd %{x}<br>" +
      "FSS %{z:.3f}<extra></extra>"
    ),
  }];

  const layout = mergeLayout(PLOT_LAYOUT, {
    xaxis: { title: { text: "Neighborhood (cells)", font: { size: 11, color: "#a8a291" } },
             type: "category" },
    yaxis: { title: { text: "Threshold (DOY)", font: { size: 11, color: "#a8a291" } },
             type: "category", autorange: "reversed" },
    annotations,
    margin: { l: 64, r: 220, t: 30, b: 52 },
  });

  const div = $("plot-fss");
  Plotly.react(div, traces, layout, PLOT_CONFIG);
  rememberPlot(div);
}

/* ------------------------------------------------------------------ *
 * Orchestration
 * ------------------------------------------------------------------ */

async function loadCatalog() {
  setStatus("loading catalog…");
  const cat = await apiGet("/api/catalog");
  state.catalog = cat;

  // Seed models
  state.models = (cat.models || []).map((m, i) => ({
    key: m.key,
    label: m.label,
    is_ensemble: !!m.is_ensemble,
    n_members: m.n_members || 0,
    years: m.years || [],
    on: i < 2,      // auto-enable first two
    primary: i === 0,
  }));
  state.models.forEach(m => colorForModel(m.key));

  // Meta strings
  const nModels = state.models.length;
  $("meta-catalog").textContent = `${cat.obs?.label ?? "obs"} · ${nModels} model${nModels === 1 ? "" : "s"}`;

  const obsYears = cat.obs?.years || [];
  const yrLo = obsYears.length ? Math.min(...obsYears) : "—";
  const yrHi = obsYears.length ? Math.max(...obsYears) : "—";
  $("meta-obs").textContent = `IMD ${yrLo}–${yrHi} · ${obsYears.length} years`;

  // Colophon root
  const colo = $("colophon-root");
  if (colo) colo.textContent = cat.root || "—";

  populateYearSelect();
  renderModelChips();
  renderParamInputs();
  wireRegionInputs();

  setStatus("ready", "ok");
}

function markPanelsLoading(on) {
  [
    "plot-isochrones",
    "plot-progression",
    "plot-corp",
    "plot-crps",
    "plot-displacement",
    "plot-fss",
  ].forEach(id => setLoading($(id), on));
  setLoading($("summary-table"), on);
}

async function refresh() {
  if (state.busy) return;
  state.busy = true;
  $("apply").disabled = true;
  $("apply").classList.remove("is-dirty");

  try {
    state.year = Number($("year").value);
    state.init = $("init").value || "auto";

    const primary = primaryModelKey();
    const actives = activeModelKeys();

    markPanelsLoading(true);

    if (!primary) {
      setStatus("pick at least one model", "err");
      $("summary-table").innerHTML = '<div class="summary-placeholder">select a model</div>';
      markPanelsLoading(false);
      return;
    }

    // Init options use the iso year (the only place a specific init makes sense)
    const isoYr = state.isoYear ?? selectedYears().slice(-1)[0];
    await refreshInitOptions(primary, isoYr);
    state.init = $("init").value || "auto";

    // Param strings
    const metricArgs   = (extra) => metricParamsCSV(Object.assign({ model: primary }, extra || {}));
    const isoArgs      = (extra) => isoParamsCSV(Object.assign({ model: primary }, extra || {}));
    const compareArgs  = metricParamsCSV({ models: actives.join(",") });
    const statePArgs   = isoParamsCSV({ model: primary });   // for hero state + isochrones

    setStatus("fetching…");

    // 1) compare (summary + progression) — multi-year aggregation
    const comparePromise = apiGet("/api/compare", compareArgs)
      .then(cmp => {
        renderSummaryTable(cmp);
        renderProgression(cmp);
      })
      .catch(err => {
        console.error("compare failed", err);
        $("summary-table").innerHTML =
          `<div class="summary-placeholder">compare failed: ${err.message}</div>`;
      })
      .finally(() => {
        setLoading($("summary-table"), false);
        setLoading($("plot-progression"), false);
      });

    // Hero panel: single-year state + isochrones for the iso year
    const statePromise = apiGet("/api/state", statePArgs)
      .then(s => apiGet("/api/metrics/isochrones", statePArgs).then(iso => ({ s, iso })))
      .then(({ s, iso }) => {
        const pLabel = (state.models.find(m => m.key === primary) || {}).label || primary;
        renderIsochrones(s, iso, pLabel);
      })
      .catch(err => {
        console.error("isochrones/state failed", err);
        $("iso-distances").textContent = `error: ${err.message}`;
      })
      .finally(() => setLoading($("plot-isochrones"), false));

    const crpsPromise = apiGet("/api/metrics/crps", metricArgs())
      .then(renderCrps)
      .catch(err => {
        console.error("crps failed", err);
        $("crps-caption").textContent = `error: ${err.message}`;
      })
      .finally(() => setLoading($("plot-crps"), false));

    const dispPromise = apiGet("/api/metrics/displacement", metricArgs())
      .then(renderDisplacement)
      .catch(err => console.error("displacement failed", err))
      .finally(() => setLoading($("plot-displacement"), false));

    const corpPromise = apiGet("/api/metrics/corp", metricArgs())
      .then(renderCorp)
      .catch(err => {
        console.error("corp failed", err);
        $("corp-caption").textContent = `error: ${err.message}`;
      })
      .finally(() => setLoading($("plot-corp"), false));

    const fssPromise = apiGet("/api/metrics/fss",
        metricArgs({ neighborhoods: "1,3,5,7,9" }))
      .then(renderFss)
      .catch(err => console.error("fss failed", err))
      .finally(() => setLoading($("plot-fss"), false));

    await Promise.allSettled([
      comparePromise, statePromise, crpsPromise, dispPromise, corpPromise, fssPromise,
    ]);

    setStatus("ready", "ok");
  } catch (err) {
    console.error("refresh fatal", err);
    setStatus(`error: ${err.message}`, "err");
  } finally {
    markPanelsLoading(false);
    state.busy = false;
    $("apply").disabled = false;
  }
}

function cleanRegion(r) {
  const out = {};
  for (const k of REGION_KEYS) {
    if (r[k] !== "" && r[k] !== null && r[k] !== undefined) out[k] = r[k];
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * Bindings
 * ------------------------------------------------------------------ */

function bindControls() {
  const onYearChange = () => {
    state.yearFrom = Number($("year_from").value);
    state.yearTo   = Number($("year_to").value);
    populateYearSelect();
    refresh();
  };
  $("year_from").addEventListener("change", onYearChange);
  $("year_to").addEventListener("change", onYearChange);

  document.querySelectorAll(".year-range-actions .mini-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const yrs = yearsForActiveSelection();
      if (!yrs.length) return;
      const tag = btn.dataset.yr;
      const toIdx = yrs.length - 1;
      let fromIdx = 0;
      if (tag === "single") fromIdx = toIdx;
      else if (tag === "all") fromIdx = 0;
      else fromIdx = Math.max(0, toIdx - (Number(tag) - 1));
      state.yearFrom = yrs[fromIdx]; state.yearTo = yrs[toIdx];
      populateYearSelect();
      refresh();
    });
  });

  $("iso_year").addEventListener("change", () => {
    state.isoYear = Number($("iso_year").value);
    refresh();
  });

  $("apply").addEventListener("click", () => { refresh(); });

  $("init").addEventListener("change", () => {
    state.init = $("init").value || "auto";
    refresh();
  });

  window.addEventListener("resize", () => {
    for (const d of state.plotDivs) {
      try { Plotly.Plots.resize(d); } catch (e) { /* ignore */ }
    }
  });
}

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */

async function init() {
  try {
    await loadCatalog();
  } catch (err) {
    console.error("catalog failed", err);
    setStatus(`catalog error: ${err.message}`, "err");
    return;
  }
  bindControls();
  refresh();
}

document.addEventListener("DOMContentLoaded", init);
