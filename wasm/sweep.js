// Multi-dimensional parameter sweep page -- loaded after app.js on
// sweep.html (dengo_wasm.js, then app.js, then this file), reusing
// app.js's physics/helpers (runFreefall, runConstantDensity,
// currentFractions, setIcs, formatTimeAuto, thermodynamicGamma,
// plotableSpecies, csvCell, vlConfig, isDarkMode, TIME_LABEL_EXPR,
// ION_H2_LABELS/COLORS, enforceGroupConservation, currentMode) rather
// than reimplementing any of them -- classic <script> tags on one page
// share a single top-level scope, so this file just refers to those
// functions/`let` bindings directly, no import/export needed.
//
// Every sweep-able parameter (the same sliders as the single-run page,
// plus per-species fractions) is a ".sweep-row": a checkbox that
// toggles it between a single fixed value and a min/max/step range.
// Pick exactly two currently-checked rows as the facet row/column;
// anything else checked gets pinned to one value at a time via its own
// "option slider" instead of appearing in the grid. See NOTES.md.

let sweepConfig = null;

// -- Row bookkeeping ---------------------------------------------------
// Every row (both the static ones generate_site.py wrote from
// PARAM_SPECS and the per-species ones buildSweepSpeciesRows() below
// builds at runtime) is fully self-describing via data-* attributes on
// its wrapping .sweep-row div -- nothing past this point needs to know
// "is this T or a species fraction".
function allSweepRows() {
  return Array.from(document.querySelectorAll(".sweep-row"));
}

function rowElFor(id) {
  return document.getElementById("row-" + id);
}

function rowSpec(rowEl) {
  const input = document.getElementById(rowEl.dataset.id);
  return {
    id: rowEl.dataset.id,
    label: rowEl.dataset.label,
    unit: rowEl.dataset.unit,
    log: rowEl.dataset.log === "1",
    modes: rowEl.dataset.modes.split(","),
    min: parseFloat(input.min), max: parseFloat(input.max), step: parseFloat(input.step),
  };
}

function isChecked(rowEl) {
  return document.getElementById(rowEl.dataset.id + "-sweep").checked;
}

function toPhysical(spec, raw) {
  return spec.log ? Math.pow(10, raw) : raw;
}

function formatPhysical(spec, raw) {
  const phys = toPhysical(spec, raw);
  const s = spec.log ? phys.toExponential(2) : phys.toPrecision(3);
  return spec.unit ? `${s} ${spec.unit}` : s;
}

// Every checked-and-currently-relevant (mode-visible) row -- the pool
// facet-row/facet-col/option-sliders are all drawn from.
function checkedRows() {
  return allSweepRows().filter((r) => r.dataset.modes.includes(currentMode) && isChecked(r));
}

// arange-style sampling in the *slider's own native units* (log10(phys)
// for every log-scale parameter, plain physical units for mach) --
// reuses each dimension's own already-established min/max/step
// convention (PARAM_SPECS in generate_site.py) rather than asking users
// to reason in decades vs. linear units per-parameter. Capped well
// short of a runaway combinatorial blowup from a typo'd tiny step.
const MAX_SWEEP_POINTS = 25;
function sweepValuesFor(rowEl) {
  const id = rowEl.dataset.id;
  const lo = parseFloat(document.getElementById(id + "-min").value);
  const hi = parseFloat(document.getElementById(id + "-max").value);
  const step = Math.abs(parseFloat(document.getElementById(id + "-swstep").value)) || 1;
  const span = Math.abs(hi - lo);
  const n = Math.min(MAX_SWEEP_POINTS, Math.max(1, Math.floor(span / step + 1e-9) + 1));
  const sign = hi >= lo ? 1 : -1;
  const values = [];
  for (let k = 0; k < n; k++) values.push(lo + sign * k * step);
  return values;
}

function updateRangePreview(rowEl) {
  const previewEl = document.getElementById(rowEl.dataset.id + "-preview");
  if (!previewEl) return;
  const spec = rowSpec(rowEl);
  const values = sweepValuesFor(rowEl);
  const lo = formatPhysical(spec, values[0]);
  const hi = formatPhysical(spec, values[values.length - 1]);
  previewEl.textContent = values.length > 1
    ? `→ ${lo} – ${hi}, ${values.length} points`
    : `→ ${lo} (min/max/step give only one point)`;
}

function updateFixedValDisplay(rowEl) {
  const spec = rowSpec(rowEl);
  const valEl = document.getElementById(spec.id + "-val");
  if (!valEl) return;
  const input = document.getElementById(spec.id);
  valEl.textContent = formatPhysical(spec, parseFloat(input.value));
}

// One shared wiring path for every row, static or dynamic: the
// checkbox flips which of .sweep-fixed/.sweep-range is visible, and
// both the single slider and the range trio keep their own live
// readouts current. Checking/unchecking or editing a range also
// refreshes the facet dropdowns and option-slider set, since both
// depend on exactly which rows are currently checked.
function wireRowToggle(rowEl) {
  const id = rowEl.dataset.id;
  const checkbox = document.getElementById(id + "-sweep");
  const fixedWrap = document.getElementById(id + "-fixed-wrap");
  const rangeWrap = document.getElementById(id + "-range-wrap");
  const input = document.getElementById(id);

  const onToggle = () => {
    fixedWrap.hidden = checkbox.checked;
    rangeWrap.hidden = !checkbox.checked;
    updateRangePreview(rowEl);
    refreshFacetOptions();
  };
  checkbox.addEventListener("change", onToggle);

  input.addEventListener("input", () => {
    updateFixedValDisplay(rowEl);
    // Species sliders share the same nuclei-conservation-on-drag
    // mechanism as the main page (see app.js) -- only meaningful while
    // this row is in its ordinary fixed-value state; a checked/swept
    // row's value is being driven by the sweep loop itself, not a drag.
    if (id.startsWith("sp-") && !checkbox.checked && typeof enforceGroupConservation === "function") {
      enforceGroupConservation(id.slice("sp-".length)); // enforceGroupConservation() takes the bare species name (e.g. "H_2"), not the "sp-"-prefixed slider id
    }
  });
  for (const suffix of ["-min", "-max", "-swstep"]) {
    document.getElementById(id + suffix).addEventListener("input", () => {
      updateRangePreview(rowEl);
      refreshOptionSliders();
    });
  }
  updateFixedValDisplay(rowEl);
  updateRangePreview(rowEl);
}

// -- Species rows (dynamic, built once speciesNames is known) ----------
// Mirrors app.js's buildSpeciesSliders() -- same ids/min/max/step/
// default per species, so currentFractions()/setIcs() work completely
// unmodified here -- but wrapped as a .sweep-row like every numeric
// parameter above, so it plugs into the exact same facet/option-slider/
// run machinery with no species-specific branches anywhere else in this
// file.
function buildSweepSpeciesRows(config) {
  computeGroupTargets(config); // shared with app.js -- powers enforceGroupConservation() below
  const container = document.getElementById("species-sliders");
  for (const name of speciesNames) {
    if (name === "ge" || name === "de") continue; // ge has its own T row; de is derived (charge neutrality)
    const frac = config.default_ics[name] !== undefined ? config.default_ics[name] : 1e-12;
    const logFrac = Math.log10(frac);
    const row = document.createElement("div");
    row.className = "row sweep-row";
    row.id = "row-sp-" + name;
    row.dataset.id = "sp-" + name;
    row.dataset.modes = "cool,freefall";
    row.dataset.log = "1";
    row.dataset.unit = "";
    row.dataset.label = name + " fraction";
    row.innerHTML = `
      <label class="checkbox-label">
        <input type="checkbox" id="sp-${name}-sweep" class="sweep-toggle" data-id="sp-${name}">
        sweep <b>${name} fraction</b>
      </label>
      <div class="sweep-fixed" id="sp-${name}-fixed-wrap">
        <input type="range" id="sp-${name}" min="-14" max="0" step="0.1" value="${logFrac}">
        <span class="val" id="sp-${name}-val"></span>
      </div>
      <div class="sweep-range" id="sp-${name}-range-wrap" hidden>
        <label>min <input type="number" id="sp-${name}-min" value="-14" step="any"></label>
        <label>max <input type="number" id="sp-${name}-max" value="0" step="any"></label>
        <label>step <input type="number" id="sp-${name}-swstep" value="0.5" step="any"></label>
        <span class="sweep-range-preview" id="sp-${name}-preview"></span>
      </div>
    `;
    container.appendChild(row);
    wireRowToggle(row);
  }
}

// -- Mode (cool / free-fall) --------------------------------------------
// Reassigns the shared `currentMode` (see app.js) directly -- app.js's
// own setMode() assumes the single-run page's *-row ids, which don't
// exist here, so this page needs its own visibility pass driven by
// each row's data-modes attribute instead.
function setSweepMode(mode) {
  currentMode = mode;
  document.getElementById("mode-cool").classList.toggle("active", mode === "cool");
  document.getElementById("mode-freefall").classList.toggle("active", mode === "freefall");
  for (const row of allSweepRows()) {
    row.style.display = row.dataset.modes.includes(mode) ? "" : "none";
  }
  refreshFacetOptions();
}

// -- Facet pickers -------------------------------------------------------
function refreshFacetOptions() {
  const rows = checkedRows();
  const rowSel = document.getElementById("facet-row");
  const colSel = document.getElementById("facet-col");
  const prevRow = rowSel.value, prevCol = colSel.value;
  for (const sel of [rowSel, colSel]) {
    sel.innerHTML = "";
    const noneOpt = document.createElement("option");
    noneOpt.value = "";
    noneOpt.textContent = "(none)";
    sel.appendChild(noneOpt);
    for (const r of rows) {
      const spec = rowSpec(r);
      const opt = document.createElement("option");
      opt.value = spec.id;
      opt.textContent = spec.label + (spec.unit ? ` (${spec.unit})` : "");
      sel.appendChild(opt);
    }
  }
  const ids = rows.map((r) => r.dataset.id);
  rowSel.value = ids.includes(prevRow) ? prevRow : (ids[0] || "");
  const colCandidate = ids.includes(prevCol) && prevCol !== rowSel.value ? prevCol : ids.find((id) => id !== rowSel.value);
  colSel.value = colCandidate || "";
  refreshOptionSliders();
  updateRunSweepEnabled();
}

// Keeps facet row/column distinct -- if a change makes them collide,
// bump whichever select the user didn't just touch to the next
// available checked dimension (or "(none)" if there isn't one).
function ensureDistinctFacets(justChanged) {
  const rowSel = document.getElementById("facet-row"), colSel = document.getElementById("facet-col");
  if (rowSel.value && rowSel.value === colSel.value) {
    const other = justChanged === "row" ? colSel : rowSel;
    const opts = Array.from(other.options).map((o) => o.value).filter((v) => v && v !== (justChanged === "row" ? rowSel.value : colSel.value));
    other.value = opts[0] || "";
  }
}

function updateRunSweepEnabled() {
  document.getElementById("run-sweep").disabled = !document.getElementById("facet-row").value;
}

// -- Option sliders (checked dims that aren't a facet axis) --------------
// Any dimension checked as "sweep" but not chosen as facet row/column
// is pinned to exactly one value at a time -- shown all at once would
// need a 3rd/4th chart dimension this page deliberately doesn't attempt
// (see NOTES.md); this reduces it back to a plain 2D facet grid at
// whatever cross-section these sliders currently point to.
function refreshOptionSliders() {
  const container = document.getElementById("option-sliders");
  container.innerHTML = "";
  const rowId = document.getElementById("facet-row").value;
  const colId = document.getElementById("facet-col").value;
  const others = checkedRows().filter((r) => r.dataset.id !== rowId && r.dataset.id !== colId);
  if (!others.length) return;
  const note = document.createElement("p");
  note.className = "preset-note";
  note.textContent = "Other swept parameters -- pin each to one cross-section value for the facet grid below:";
  container.appendChild(note);
  for (const rowEl of others) {
    const spec = rowSpec(rowEl);
    const values = sweepValuesFor(rowEl);
    const wrap = document.createElement("div");
    wrap.className = "row";
    const sliderId = "opt-" + spec.id;
    wrap.innerHTML = `
      <label>${spec.label}${spec.unit ? ` (${spec.unit})` : ""} <span class="val" id="${sliderId}-val"></span></label>
      <input type="range" id="${sliderId}" min="0" max="${values.length - 1}" step="1"
             value="${Math.floor((values.length - 1) / 2)}" data-for="${spec.id}">
    `;
    container.appendChild(wrap);
    const slider = wrap.querySelector("input");
    const valEl = wrap.querySelector(".val");
    const update = () => { valEl.textContent = formatPhysical(spec, values[parseInt(slider.value, 10)]); };
    slider.addEventListener("input", update);
    update();
  }
}

// -- Running the sweep ----------------------------------------------------
const MAX_SWEEP_RUNS = 200; // wall-clock guard -- see runMultiSweep()

function setSweepStatus(text) {
  document.getElementById("sweep-status").textContent = text;
}

function runMultiSweep() {
  const rowId = document.getElementById("facet-row").value;
  if (!rowId) {
    setSweepStatus("Check at least one parameter and pick it as the facet row.");
    return;
  }
  const colId = document.getElementById("facet-col").value; // "" is valid -- row-only facet

  const rowEl = rowElFor(rowId);
  const rowSpecVal = rowSpec(rowEl);
  const rowValues = sweepValuesFor(rowEl);

  const colEl = colId ? rowElFor(colId) : null;
  const colSpecVal = colEl ? rowSpec(colEl) : null;
  const colValues = colEl ? sweepValuesFor(colEl) : [null];

  const pinnedRows = checkedRows().filter((r) => r.dataset.id !== rowId && r.dataset.id !== colId);
  const pinned = pinnedRows.map((r) => {
    const spec = rowSpec(r);
    const values = sweepValuesFor(r);
    const idx = parseInt(document.getElementById("opt-" + spec.id).value, 10);
    return { spec, raw: values[idx] };
  });

  const totalRuns = rowValues.length * colValues.length;
  if (totalRuns > MAX_SWEEP_RUNS) {
    setSweepStatus(`${totalRuns} runs would be needed -- narrow a facet range/step first (max ${MAX_SWEEP_RUNS}).`);
    return;
  }

  const button = document.getElementById("run-sweep");
  button.disabled = true;
  setSweepStatus(`running ${totalRuns} run${totalRuns === 1 ? "" : "s"}…`);
  // Yield one frame so the status text actually paints before the
  // (synchronous, possibly multi-second) loop below blocks the main
  // thread -- same reasoning as the old single-parameter sweep.
  requestAnimationFrame(() => setTimeout(
    () => runMultiSweepBody(rowSpecVal, rowValues, colSpecVal, colValues, pinned, totalRuns), 0,
  ));
}

function runOneSweepCell() {
  const nH = Math.pow(10, parseFloat(document.getElementById("nH").value));
  const T = Math.pow(10, parseFloat(document.getElementById("T").value));
  const fractions = currentFractions();
  const tolerance = Math.pow(10, parseFloat(document.getElementById("tolerance").value));
  if (currentMode === "freefall") {
    const logNTarget = parseFloat(document.getElementById("ntarget").value);
    const logNShock = parseFloat(document.getElementById("nshock").value);
    const machSlider = parseFloat(document.getElementById("mach").value);
    const shockEnabled = document.getElementById("shock-enabled").checked;
    const machShock = shockEnabled ? machSlider : 1;
    const collapseFactor = Math.pow(10, parseFloat(document.getElementById("collapse-rate").value));
    const ffStep = Math.pow(10, parseFloat(document.getElementById("ff-step").value));
    return runFreefall(nH, T, fractions, logNTarget, logNShock, machShock, collapseFactor, tolerance, ffStep);
  }
  const logDtf = parseFloat(document.getElementById("dtf").value);
  return runConstantDensity(nH, T, fractions, logDtf, tolerance, undefined, undefined, true);
}

// All PARAM_SPECS ids, whether checked/swept or not -- every run's
// metadata records its *actual* setting for every one of these, not
// just whichever happen to be varying right now, so the CSV/facet
// labels are self-describing on their own (see resultsToCsv()'s own
// per-run `params` bookkeeping in sweepResultsToCsv() below).
function allNumericParamIds() {
  return allSweepRows().filter((r) => !r.dataset.id.startsWith("sp-")).map((r) => r.dataset.id);
}

function runMultiSweepBody(rowSpecVal, rowValues, colSpecVal, colValues, pinned, totalRuns) {
  const t0 = performance.now();

  // Save every input this run touches so it can be restored afterward --
  // a sweep looks at (and briefly drives) other conditions, it doesn't
  // permanently change them.
  const touchedIds = [rowSpecVal.id, ...(colSpecVal ? [colSpecVal.id] : []), ...pinned.map((p) => p.spec.id)];
  const saved = {};
  for (const id of touchedIds) saved[id] = document.getElementById(id).value;
  for (const p of pinned) document.getElementById(p.spec.id).value = p.raw;

  const tRows = [], ihRows = [];
  const runs = []; // for CSV: [{params: {id: physicalValue}, result}]
  const rowLabels = [], colLabels = [];
  let xKey = "time";
  const paramIds = allNumericParamIds();

  for (const rv of rowValues) {
    document.getElementById(rowSpecVal.id).value = rv;
    const rowLabel = `${rowSpecVal.label}=${formatPhysical(rowSpecVal, rv)}`;
    if (!rowLabels.includes(rowLabel)) rowLabels.push(rowLabel);
    for (const cv of colValues) {
      if (colSpecVal) document.getElementById(colSpecVal.id).value = cv;
      const colLabel = colSpecVal ? `${colSpecVal.label}=${formatPhysical(colSpecVal, cv)}` : null;
      if (colLabel && !colLabels.includes(colLabel)) colLabels.push(colLabel);

      const result = runOneSweepCell();
      xKey = result.xKey;

      const params = {};
      for (const id of paramIds) {
        const spec = rowSpec(rowElFor(id));
        params[id] = toPhysical(spec, parseFloat(document.getElementById(id).value));
      }
      // Any *swept* species fraction also gets recorded (unswept ones
      // are already fully captured by the per-step species columns'
      // own i=0 initial-condition row -- see sweepResultsToCsv()).
      for (const r of allSweepRows()) {
        if (r.dataset.id.startsWith("sp-") && isChecked(r)) {
          const spec = rowSpec(r);
          params[spec.id] = toPhysical(spec, parseFloat(document.getElementById(spec.id).value));
        }
      }
      runs.push({ params, result });

      for (let i = 0; i < result.x.length; i++) {
        const base = { facetRowLabel: rowLabel, x: result.x[i], tHuman: formatTimeAuto(result.t[i]) };
        if (colLabel) base.facetColLabel = colLabel;
        tRows.push({ ...base, T: result.T[i] });
        ihRows.push({ ...base, quantity: ION_H2_LABELS.ion, value: result.ion[i] });
        const h2v = result.h2[i];
        if (h2v !== null && h2v !== undefined && h2v > 0) ihRows.push({ ...base, quantity: ION_H2_LABELS.h2, value: h2v });
      }
    }
  }

  for (const id of touchedIds) document.getElementById(id).value = saved[id];

  vegaEmbed(
    "#chart-facet-T",
    facetChartSpec("T", xKey, tRows, rowSpecVal, colSpecVal, rowLabels, colLabels),
    { actions: false, renderer: "svg" },
  );
  vegaEmbed(
    "#chart-facet-ionh2",
    facetChartSpec("ionh2", xKey, ihRows, rowSpecVal, colSpecVal, rowLabels, colLabels),
    { actions: false, renderer: "svg" },
  );

  lastSweepRuns = runs;
  document.getElementById("download-sweep-csv").disabled = false;

  const elapsed = performance.now() - t0;
  setSweepStatus(`${totalRuns} run${totalRuns === 1 ? "" : "s"}, ${elapsed.toFixed(0)} ms`);
  document.getElementById("run-sweep").disabled = false;
}

// -- Faceted chart spec ----------------------------------------------------
function facetChartSpec(kind, xKey, rows, rowSpecVal, colSpecVal, rowDomain, colDomain) {
  const facet = colSpecVal
    ? {
      row: { field: "facetRowLabel", type: "ordinal", sort: rowDomain, header: { title: rowSpecVal.label, labelFontSize: 10 } },
      column: { field: "facetColLabel", type: "ordinal", sort: colDomain, header: { title: colSpecVal.label, labelFontSize: 10 } },
    }
    : { row: { field: "facetRowLabel", type: "ordinal", sort: rowDomain, header: { title: rowSpecVal.label, labelFontSize: 10 } } };

  const xEncoding = {
    field: "x", type: "quantitative",
    scale: { type: xKey === "time" ? "symlog" : "log" },
    axis: {
      title: null, labelOverlap: "greedy", labelFontSize: 8,
      labelAngle: xKey === "time" ? -40 : 0,
      labelExpr: xKey === "time" ? TIME_LABEL_EXPR : undefined,
    },
  };

  let layerSpec;
  if (kind === "T") {
    layerSpec = {
      mark: { type: "line", point: { filled: true, size: 8, opacity: 0.9 } },
      encoding: {
        x: xEncoding,
        y: { field: "T", type: "quantitative", scale: { type: "log" }, axis: { title: "T (K)", titleFontSize: 10 } },
        tooltip: [
          { field: "x", title: xKey === "time" ? "t (s)" : "n (cm⁻³)", type: "quantitative", format: ".3~g" },
          { field: "tHuman", title: "t", type: "nominal" },
          { field: "T", title: "T (K)", type: "quantitative", format: ".4~g" },
        ],
      },
    };
  } else {
    const palette = isDarkMode() ? ION_H2_COLORS.dark : ION_H2_COLORS.light;
    const present = new Set(rows.map((r) => r.quantity));
    const domain = [ION_H2_LABELS.ion, ION_H2_LABELS.h2].filter((l) => present.has(l));
    const range = domain.map((l) => (l === ION_H2_LABELS.ion ? palette.ion : palette.h2));
    layerSpec = {
      mark: { type: "line", point: { filled: true, size: 6, opacity: 0.9 } },
      encoding: {
        x: xEncoding,
        y: { field: "value", type: "quantitative", scale: { type: "log" }, axis: { title: "fraction", titleFontSize: 10 } },
        color: { field: "quantity", type: "nominal", scale: { domain, range }, legend: { title: null, orient: "bottom" } },
        tooltip: [
          { field: "quantity", title: "quantity", type: "nominal" },
          { field: "x", title: xKey === "time" ? "t (s)" : "n (cm⁻³)", type: "quantitative", format: ".3~g" },
          { field: "value", title: "fraction", type: "quantitative", format: ".4~g" },
        ],
      },
    };
  }

  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    background: null,
    config: vlConfig(),
    data: { values: rows },
    facet,
    spec: { width: 150, height: 120, ...layerSpec },
  };
}

// -- CSV export -------------------------------------------------------------
// Long/tidy format, one row per (run, step) -- every swept-or-not
// numeric parameter's actual value for that run as its own column
// (`param_<id>`, physical units), plus every field the single-run
// page's own CSV export has (see resultsToCsv() in app.js): trivial to
// load with `pandas.read_csv(...)` and `groupby()`/`pivot()` on
// whichever param_* columns were actually varied.
let lastSweepRuns = null;

function csvSafeId(id) {
  return id.replace(/[^A-Za-z0-9]+/g, "_");
}

function sweepResultsToCsv(runs) {
  if (!runs || !runs.length) return "";
  const paramIds = Object.keys(runs[0].params);
  const species = plotableSpecies();
  const header = [
    "run_id", ...paramIds.map((id) => "param_" + csvSafeId(id)),
    "step", "x", "x_kind", "t_s", "t_human", "dt_s",
    "T_K", "ge_erg_per_g", "gamma", "ion_frac", "h2_frac",
    ...species,
  ];
  const lines = [header.map(csvCell).join(",")];
  runs.forEach((run, runIdx) => {
    const { result, params } = run;
    for (let i = 0; i < result.x.length; i++) {
      const s = result.s[i];
      const row = [
        runIdx, ...paramIds.map((id) => params[id]),
        i, result.x[i], result.xKey, result.t[i], formatTimeAuto(result.t[i]), result.dt[i],
        result.T[i], s.ge, thermodynamicGamma(s), result.ion[i], result.h2[i],
        ...species.map((name) => s[name]),
      ];
      lines.push(row.map(csvCell).join(","));
    }
  });
  return lines.join("\n");
}

function downloadSweepCsv() {
  if (!lastSweepRuns) return;
  const csv = sweepResultsToCsv(lastSweepRuns);
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${sweepConfig.title.replace(/\s+/g, "_")}_sweep_results.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// -- Page bootstrap -----------------------------------------------------
function initSweepPage(config) {
  sweepConfig = config;
  document.getElementById("page-title").textContent = `${config.title}: parameter sweep`;

  document.getElementById("mode-cool").addEventListener("click", () => setSweepMode("cool"));
  document.getElementById("mode-freefall").addEventListener("click", () => setSweepMode("freefall"));
  document.getElementById("facet-row").addEventListener("change", () => { ensureDistinctFacets("row"); refreshOptionSliders(); updateRunSweepEnabled(); });
  document.getElementById("facet-col").addEventListener("change", () => { ensureDistinctFacets("col"); refreshOptionSliders(); updateRunSweepEnabled(); });
  document.getElementById("run-sweep").addEventListener("click", runMultiSweep);
  document.getElementById("download-sweep-csv").addEventListener("click", downloadSweepCsv);

  for (const rowEl of allSweepRows()) wireRowToggle(rowEl);

  DengoModule().then((m) => {
    mod = m;
    init = mod.cwrap("dengo_wasm_init", null, []);
    step = mod.cwrap("dengo_wasm_step", "number", ["number", "number", "number"]);
    statePtr = mod.cwrap("dengo_wasm_state_ptr", "number", []);
    rhsPtr = mod.cwrap("dengo_wasm_rhs_ptr", "number", []);
    temperature = mod.cwrap("dengo_wasm_temperature", "number", []);
    const namesFn = mod.cwrap("dengo_wasm_species_names", "string", []);
    init();
    speciesNames = namesFn().split(",");
    idx = Object.fromEntries(speciesNames.map((n, i) => [n, i]));
    buildSweepSpeciesRows(config);
    setSweepMode("freefall"); // matches the single-run page's own default
    setSweepStatus("check parameters to sweep, then run");
  });
}
