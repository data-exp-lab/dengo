// Shared driver for every fiducial network's page -- network-specific
// bits (title, default initial fractions/temperature, which compiled
// module to load) come in via `window.DENGO_PAGE_CONFIG`, injected as a
// small inline <script> by generate_site.py. Nothing here is hardcoded
// to a particular species set: sliders are built dynamically from
// dengo_wasm_species_names(), and the free-fall/cooldown physics
// (thermodynamicGamma, ionization fraction) only assumes H_1/H_2 exist
// (true of every fiducial network) plus, optionally, H2_1/H2_2 for the
// bonus H2-fraction readout.
const KB = 1.3806504e-16, MH = 1.67e-24, G_GRAV = 6.674e-8;
const YEAR = 3.1557e7; // Julian year, seconds -- matches primordial_cooling.py's usage elsewhere

let mod, idx, speciesNames;
let init, step, statePtr, rhsPtr, temperature;
let currentMode = "cool";

// -- LaTeX axis labels (rendered via KaTeX, not Vega-Lite's own plain-text
// titles -- see NOTES.md for why: Vega-Lite axis titles are just SVG
// <text>, no math rendering, so proper sub/superscripts need an external
// renderer laid out alongside the chart instead). ----------------------
const AXIS_LATEX = {
  time: "t\\ (\\mathrm{s})",
  density: "n\\ (\\mathrm{cm^{-3}})",
  T: "T\\ (\\mathrm{K})",
  ion: "\\mathrm{H^+} / \\mathrm{H_{tot}}",
  species: "n_i\\ (\\mathrm{cm^{-3}})",
};

// d3's category10, fixed to a stable per-species assignment (see
// plotableSpecies()/speciesColor() below) so a species keeps the same
// color whether or not its neighbors are toggled on -- and so the
// per-species toggle checkboxes can carry a matching swatch and double
// as the chart's legend (Vega-Lite's own legend is turned off for that
// chart to avoid drawing the same information twice).
const SPECIES_COLORS = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
];

// The species actually worth plotting as an abundance: everything except
// `ge` (specific internal energy, not a density -- already has its own
// Temperature chart). `de` (electron density) is included; it's exactly
// as physical as any other species here.
function plotableSpecies() {
  return speciesNames.filter((n) => n !== "ge");
}

function speciesColor(name) {
  const i = plotableSpecies().indexOf(name);
  return SPECIES_COLORS[i % SPECIES_COLORS.length];
}

function renderLatex(elId, key) {
  const el = document.getElementById(elId);
  if (!el || !window.katex) return;
  if (el.dataset.rendered === key) return; // cheap no-op on unrelated redraws
  katex.render(AXIS_LATEX[key], el, { throwOnError: false });
  el.dataset.rendered = key;
}

// Same "pick the least-silly unit" logic in two places by necessity: this
// JS version (for status text and the tooltip's human-readable field) and
// TIME_LABEL_EXPR below (a Vega expression string, for axis tick labels --
// Vega-Lite can't call out to arbitrary JS from a spec). Keep them in sync
// if either changes.
function formatTimeAuto(seconds) {
  const a = Math.abs(seconds);
  if (!(a > 0)) return "0 s";
  if (a >= 1e6 * YEAR) return (seconds / (1e6 * YEAR)).toPrecision(3) + " Myr";
  if (a >= 1e3 * YEAR) return (seconds / (1e3 * YEAR)).toPrecision(3) + " kyr";
  if (a >= YEAR) return (seconds / YEAR).toPrecision(3) + " yr";
  if (a >= 86400) return (seconds / 86400).toPrecision(3) + " d";
  if (a >= 3600) return (seconds / 3600).toPrecision(3) + " hr";
  return seconds.toPrecision(3) + " s";
}

const TIME_LABEL_EXPR =
  "!(abs(datum.value) > 0) ? '0 s' : " +
  "abs(datum.value) >= 1e6*" + YEAR + " ? format(datum.value/1e6/" + YEAR + ",'.3~g') + ' Myr' : " +
  "abs(datum.value) >= 1e3*" + YEAR + " ? format(datum.value/1e3/" + YEAR + ",'.3~g') + ' kyr' : " +
  "abs(datum.value) >= " + YEAR + " ? format(datum.value/" + YEAR + ",'.3~g') + ' yr' : " +
  "abs(datum.value) >= 86400 ? format(datum.value/86400,'.3~g') + ' d' : " +
  "abs(datum.value) >= 3600 ? format(datum.value/3600,'.3~g') + ' hr' : " +
  "format(datum.value,'.3~g') + ' s'";

function isDarkMode() {
  return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
}

// Vega-Lite's default axis/legend colors are tuned for a light page and go
// near-invisible against a dark one (someone I work with actually uses
// dark mode) -- pick chart colors to match the CSS theme instead of
// leaving Vega-Lite's light-mode defaults baked into the spec.
function vlConfig() {
  const dark = isDarkMode();
  return {
    background: null,
    view: { stroke: "transparent" },
    axis: {
      labelColor: dark ? "#b7bcc4" : "#333333",
      titleColor: dark ? "#b7bcc4" : "#333333",
      gridColor: dark ? "#33363d" : "#e6e6e6",
      domainColor: dark ? "#565a63" : "#888888",
      tickColor: dark ? "#565a63" : "#888888",
    },
    line: { color: dark ? "#8ab4ff" : "#3366cc" },
    point: { color: dark ? "#8ab4ff" : "#3366cc" },
  };
}

function getScalar() {
  const ptr = statePtr() >> 3;
  const out = {};
  for (const name of speciesNames) out[name] = mod.HEAPF64[ptr + idx[name]];
  return out;
}

function ionizedFraction(s) {
  return s.H_2 / (s.H_1 + s.H_2);
}

function h2Fraction(s) {
  if (idx.H2_1 === undefined) return null;
  const totalH = s.H_1 + s.H_2 + (s.H_m0 || 0) + 2 * (s.H2_1 + (s.H2_2 || 0));
  return 2 * s.H2_1 / totalH;
}

function thermodynamicGamma(s) {
  let nTotal = 0, invGm1Sum = 0;
  for (const name of speciesNames) {
    if (name === "ge") continue;
    nTotal += s[name];
    const gammaI = (name === "H2_1" || name === "H2_2") ? 7 / 5 : 5 / 3;
    invGm1Sum += s[name] / (gammaI - 1);
  }
  return nTotal / invGm1Sum + 1;
}

function setIcs(nH, T, fractions) {
  const ptr = statePtr() >> 3;
  for (const name of speciesNames) {
    if (name === "ge") {
      mod.HEAPF64[ptr + idx.ge] = 1.5 * KB * T / MH;
    } else {
      const frac = fractions[name] !== undefined ? fractions[name] : 0.0;
      mod.HEAPF64[ptr + idx[name]] = nH * frac;
    }
  }
}

function coolingTime(dtfTotal) {
  const s = getScalar();
  const rhsIdx = rhsPtr() >> 3;
  const dgeDt = mod.HEAPF64[rhsIdx + idx.ge];
  return dgeDt !== 0 ? Math.abs(s.ge / dgeDt) : dtfTotal;
}

function runConstantDensity(nH, T, fractions, logDtf, safetyFactor = 0.1, maxSteps = 2000) {
  setIcs(nH, T, fractions);
  const dtfTotal = Math.pow(10, logDtf);
  let t = 0;
  const tHist = [], THist = [], ionHist = [], h2Hist = [], dtHist = [], sHist = [];
  for (let i = 0; i < maxSteps; i++) {
    const dt = Math.min(safetyFactor * coolingTime(dtfTotal), dtfTotal - t);
    if (dt <= 0) break;
    const converged = step(dt, 200, 1e-5);
    if (!converged) break;
    t += dt;
    const s = getScalar();
    tHist.push(t); THist.push(temperature()); ionHist.push(ionizedFraction(s)); h2Hist.push(h2Fraction(s));
    dtHist.push(dt); sHist.push(s);
    if (t >= dtfTotal) break;
  }
  return { x: tHist, t: tHist, dt: dtHist, T: THist, ion: ionHist, h2: h2Hist, s: sHist, xKey: "time" };
}

function runFreefall(nH, T, fractions, logNTarget, safetyFactor = 0.01, maxSteps = 5000) {
  setIcs(nH, T, fractions);
  const nTarget = Math.pow(10, logNTarget);
  let nCurrent = nH, t = 0;
  const nHist = [], THist = [], ionHist = [], h2Hist = [], tHist = [], dtHist = [], sHist = [];
  for (let i = 0; i < maxSteps; i++) {
    if (nCurrent >= nTarget) break;
    const rho = nCurrent * MH;
    const tFf = Math.sqrt(3 * Math.PI / (32 * G_GRAV * rho));
    const dt = safetyFactor * tFf;
    const rhoNew = Math.pow(Math.pow(rho, -0.5) - Math.sqrt(32 * G_GRAV / (3 * Math.PI)) * dt, -2);
    const densityRatio = rhoNew / rho;

    const ptr = statePtr() >> 3;
    for (const name of speciesNames) {
      if (name !== "ge") mod.HEAPF64[ptr + idx[name]] *= densityRatio;
    }
    const gammaAd = thermodynamicGamma(getScalar());
    mod.HEAPF64[ptr + idx.ge] *= (1 + (gammaAd - 1) * (densityRatio - 1));

    const converged = step(dt, 200, 1e-5);
    if (!converged) break;
    t += dt;
    const s = getScalar();
    nCurrent = 0;
    for (const name of speciesNames) if (name !== "ge" && name !== "de") nCurrent += s[name];
    nHist.push(nCurrent); THist.push(temperature()); ionHist.push(ionizedFraction(s)); h2Hist.push(h2Fraction(s));
    tHist.push(t); dtHist.push(dt); sHist.push(s);
  }
  return { x: nHist, t: tHist, dt: dtHist, T: THist, ion: ionHist, h2: h2Hist, s: sHist, xKey: "density" };
}

const FIELD_TITLE = { T: "T (K)", ion: "H⁺ / H_tot" };

function chartSpec(field, xKey, data, extra) {
  // Point markers double as an annotation of *where* the adaptive
  // stepper actually placed a step -- their spacing on the log x-axis
  // directly shows the step-size ramp (small at first, growing ~2x per
  // step; see BE_chem_solve.C / NOTES.md). Always drawn (not just below
  // some point-count cutoff), sized down as steps pile up so a
  // few-thousand-step free-fall run doesn't turn into a solid smear.
  const n = data.length;
  const pointSize = Math.max(6, Math.min(36, 2500 / Math.max(n, 1)));
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    width: 600, height: 220, background: null,
    config: vlConfig(),
    layer: [
      ...(extra || []),
      {
        data: { values: data },
        mark: { type: "line", point: { filled: true, size: pointSize, opacity: 0.9 } },
        encoding: {
          x: {
            field: "x", type: "quantitative", scale: { type: "log" },
            axis: {
              title: null, labelOverlap: "greedy",
              // human-unit labels ("254 kyr") run wider than the plain
              // numbers this axis used to show, so angle them -- greedy
              // overlap removal alone still let neighbors visually touch
              // on a busy log axis with many same-decade ticks.
              labelAngle: xKey === "time" ? -40 : 0,
              labelExpr: xKey === "time" ? TIME_LABEL_EXPR : undefined,
            },
          },
          y: { field: field, type: "quantitative", scale: { type: "log" }, axis: { title: null } },
          tooltip: [
            { field: "i", title: "step", type: "quantitative" },
            { field: "x", title: xKey === "time" ? "t (s)" : "n (cm⁻³)", type: "quantitative", format: ".3~g" },
            { field: "tHuman", title: "t", type: "nominal" },
            { field: "dt", title: "step Δt (s)", type: "quantitative", format: ".3~g" },
            { field: field, title: FIELD_TITLE[field] || field, type: "quantitative", format: ".4~g" },
          ],
        },
      },
    ],
  };
}

// A multi-series companion to chartSpec(): one line+points per selected
// species (long-format `rows`, one row per (step, species) pair), color
// keyed to the same fixed per-species palette the toggle checkboxes use.
// This is the direct answer to "I can't get an impression of the full
// range of values" -- T and the H+/H_tot ratio were the only things
// plotted before; nothing showed the actual per-species number densities
// or how many decades they span.
function speciesChartSpec(xKey, rows) {
  const n = rows.length;
  const pointSize = Math.max(4, Math.min(30, 2000 / Math.max(n, 1)));
  const domain = plotableSpecies();
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    width: 600, height: 240, background: null,
    config: vlConfig(),
    data: { values: rows },
    mark: { type: "line", point: { filled: true, size: pointSize, opacity: 0.9 } },
    encoding: {
      x: {
        field: "x", type: "quantitative", scale: { type: "log" },
        axis: {
          title: null, labelOverlap: "greedy",
          labelAngle: xKey === "time" ? -40 : 0,
          labelExpr: xKey === "time" ? TIME_LABEL_EXPR : undefined,
        },
      },
      y: { field: "value", type: "quantitative", scale: { type: "log" }, axis: { title: null } },
      color: {
        field: "species", type: "nominal",
        scale: { domain, range: domain.map((n) => speciesColor(n)) },
        legend: null, // the toggle checkboxes (with matching swatches) are the legend
      },
      tooltip: [
        { field: "species", title: "species", type: "nominal" },
        { field: "i", title: "step", type: "quantitative" },
        { field: "x", title: xKey === "time" ? "t (s)" : "n (cm⁻³)", type: "quantitative", format: ".3~g" },
        { field: "tHuman", title: "t", type: "nominal" },
        { field: "dt", title: "step Δt (s)", type: "quantitative", format: ".3~g" },
        { field: "value", title: "n_i (cm⁻³)", type: "quantitative", format: ".4~g" },
      ],
    },
  };
}

let redrawQueued = false;
function scheduleRedraw() {
  if (redrawQueued) return;
  redrawQueued = true;
  requestAnimationFrame(() => { redrawQueued = false; redraw(); });
}

function selectedSpeciesNames() {
  const out = [];
  for (const name of plotableSpecies()) {
    const el = document.getElementById("toggle-" + name);
    if (el && el.checked) out.push(name);
  }
  return out;
}

function currentFractions() {
  const fractions = {};
  for (const name of speciesNames) {
    if (name === "ge") continue;
    const el = document.getElementById("sp-" + name);
    if (el) fractions[name] = Math.pow(10, parseFloat(el.value));
  }
  return fractions;
}

function redraw() {
  const nH = Math.pow(10, parseFloat(document.getElementById("nH").value));
  const T = Math.pow(10, parseFloat(document.getElementById("T").value));
  document.getElementById("nH-val").textContent = nH.toExponential(2);
  document.getElementById("T-val").textContent = T.toFixed(0);
  const fractions = currentFractions();
  for (const name in fractions) {
    document.getElementById("sp-" + name + "-val").textContent = fractions[name].toExponential(1);
  }

  const t0 = performance.now();
  let result;
  if (currentMode === "freefall") {
    const logNTarget = parseFloat(document.getElementById("ntarget").value);
    document.getElementById("ntarget-val").textContent = logNTarget.toFixed(1);
    result = runFreefall(nH, T, fractions, logNTarget);
  } else {
    const logDtf = parseFloat(document.getElementById("dtf").value);
    document.getElementById("dtf-val").textContent = logDtf.toFixed(1);
    result = runConstantDensity(nH, T, fractions, logDtf);
  }
  const elapsed = performance.now() - t0;

  const rows = result.x.map((x, i) => ({
    x, T: result.T[i], ion: result.ion[i],
    i, dt: result.dt[i], tHuman: formatTimeAuto(result.t[i]),
  }));
  const band = {
    data: { values: [{ y0: 1500, y1: 2500 }] },
    mark: { type: "rect", opacity: isDarkMode() ? 0.25 : 0.15, color: "orange" },
    encoding: { y: { field: "y0", type: "quantitative" }, y2: { field: "y1" } },
  };
  vegaEmbed("#chart-T", chartSpec("T", result.xKey, rows, [band]),
            { actions: false, renderer: "svg" });
  vegaEmbed("#chart-ion", chartSpec("ion", result.xKey, rows),
            { actions: false, renderer: "svg" });
  renderLatex("ylabel-T", "T");
  renderLatex("xlabel-T", result.xKey);
  renderLatex("ylabel-ion", "ion");
  renderLatex("xlabel-ion", result.xKey);

  const selectedSpecies = selectedSpeciesNames();
  const chartSpeciesEl = document.getElementById("chart-species");
  if (selectedSpecies.length) {
    const speciesRows = [];
    for (let i = 0; i < result.x.length; i++) {
      const tHuman = formatTimeAuto(result.t[i]);
      for (const name of selectedSpecies) {
        const v = result.s[i][name];
        if (v > 0) speciesRows.push({ x: result.x[i], i, dt: result.dt[i], tHuman, species: name, value: v });
      }
    }
    chartSpeciesEl.innerHTML = "";
    vegaEmbed(chartSpeciesEl, speciesChartSpec(result.xKey, speciesRows), { actions: false, renderer: "svg" });
  } else {
    chartSpeciesEl.innerHTML = '<p class="chart-placeholder">Toggle one or more species above to plot them.</p>';
  }
  renderLatex("ylabel-species", "species");
  renderLatex("xlabel-species", result.xKey);

  const finalT = result.T[result.T.length - 1];
  const finalIon = result.ion[result.ion.length - 1];
  const finalH2 = result.h2[result.h2.length - 1];
  const finalTime = result.t[result.t.length - 1];
  let statusText = `${rows.length} steps, ${elapsed.toFixed(1)} ms -- final T=${finalT ? finalT.toFixed(1) : "?"} K, `
    + `ionized=${finalIon ? finalIon.toExponential(2) : "?"}`;
  if (finalH2 !== null && finalH2 !== undefined) statusText += `, H2/H_tot=${finalH2.toExponential(2)}`;
  if (finalTime !== undefined) statusText += `, elapsed t=${formatTimeAuto(finalTime)} (${finalTime.toExponential(2)} s)`;
  document.getElementById("status").textContent = statusText;
}

function setMode(mode) {
  currentMode = mode;
  document.getElementById("mode-cool").classList.toggle("active", mode === "cool");
  document.getElementById("mode-freefall").classList.toggle("active", mode === "freefall");
  document.getElementById("dtf-row").style.display = mode === "cool" ? "" : "none";
  document.getElementById("ntarget-row").style.display = mode === "freefall" ? "" : "none";
  scheduleRedraw();
}

function buildSpeciesSliders(config) {
  const container = document.getElementById("species-sliders");
  for (const name of speciesNames) {
    if (name === "ge" || name === "de") continue; // ge has its own T slider; de is derived (charge neutrality)
    const frac = config.default_ics[name] !== undefined ? config.default_ics[name] : 1e-12;
    const logFrac = Math.log10(frac);
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `
      <label>${name} fraction <span class="val" id="sp-${name}-val"></span></label>
      <input type="range" id="sp-${name}" min="-14" max="0" step="0.1" value="${logFrac}">
    `;
    container.appendChild(row);
    row.querySelector("input").addEventListener("input", scheduleRedraw);
  }
}

function buildSpeciesToggle() {
  const container = document.getElementById("species-toggle");
  container.innerHTML = "";

  const controls = document.createElement("div");
  controls.className = "species-toggle-controls";
  controls.innerHTML = `<button type="button" id="species-all">all</button><button type="button" id="species-none">none</button>`;
  container.appendChild(controls);

  for (const name of plotableSpecies()) {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" id="toggle-${name}" checked>`
      + `<span class="swatch" style="background:${speciesColor(name)}"></span>${name}`;
    container.appendChild(label);
    label.querySelector("input").addEventListener("change", scheduleRedraw);
  }

  document.getElementById("species-all").addEventListener("click", () => {
    for (const name of plotableSpecies()) document.getElementById("toggle-" + name).checked = true;
    scheduleRedraw();
  });
  document.getElementById("species-none").addEventListener("click", () => {
    for (const name of plotableSpecies()) document.getElementById("toggle-" + name).checked = false;
    scheduleRedraw();
  });
}

function initPage(config) {
  document.getElementById("page-title").textContent = config.title;
  document.getElementById("T").value = Math.log10(config.default_T);

  document.getElementById("mode-cool").addEventListener("click", () => setMode("cool"));
  document.getElementById("mode-freefall").addEventListener("click", () => setMode("freefall"));
  for (const id of ["nH", "T", "dtf", "ntarget"]) {
    document.getElementById(id).addEventListener("input", scheduleRedraw);
  }
  // Charts bake current colors into the Vega-Lite spec at render time
  // (see vlConfig()), so a live OS theme flip needs an explicit redraw --
  // it won't happen on its own the way the CSS-variable-driven rest of
  // the page does.
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", scheduleRedraw);
  }

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
    buildSpeciesSliders(config);
    buildSpeciesToggle();
    document.getElementById("status").textContent = "ready";
    redraw();
  });
}
