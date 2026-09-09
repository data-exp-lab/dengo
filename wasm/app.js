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

let mod, idx, speciesNames;
let init, step, statePtr, rhsPtr, temperature;
let currentMode = "cool";

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
  const tHist = [], THist = [], ionHist = [], h2Hist = [];
  for (let i = 0; i < maxSteps; i++) {
    const dt = Math.min(safetyFactor * coolingTime(dtfTotal), dtfTotal - t);
    if (dt <= 0) break;
    const converged = step(dt, 200, 1e-5);
    if (!converged) break;
    t += dt;
    const s = getScalar();
    tHist.push(t); THist.push(temperature()); ionHist.push(ionizedFraction(s)); h2Hist.push(h2Fraction(s));
    if (t >= dtfTotal) break;
  }
  return { x: tHist, T: THist, ion: ionHist, h2: h2Hist, xlabel: "Time (s)" };
}

function runFreefall(nH, T, fractions, logNTarget, safetyFactor = 0.01, maxSteps = 5000) {
  setIcs(nH, T, fractions);
  const nTarget = Math.pow(10, logNTarget);
  let nCurrent = nH;
  const nHist = [], THist = [], ionHist = [], h2Hist = [];
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
    const s = getScalar();
    nCurrent = 0;
    for (const name of speciesNames) if (name !== "ge" && name !== "de") nCurrent += s[name];
    nHist.push(nCurrent); THist.push(temperature()); ionHist.push(ionizedFraction(s)); h2Hist.push(h2Fraction(s));
  }
  return { x: nHist, T: THist, ion: ionHist, h2: h2Hist, xlabel: "n (cm⁻³)" };
}

function chartSpec(field, xlabel, yLabel, data, extra) {
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    width: 600, height: 220, background: null,
    layer: [
      ...(extra || []),
      {
        data: { values: data },
        mark: { type: "line", point: data.length < 60 },
        encoding: {
          x: { field: "x", type: "quantitative", scale: { type: "log" }, title: xlabel },
          y: { field: field, type: "quantitative", scale: { type: "log" }, title: yLabel },
        },
      },
    ],
  };
}

let redrawQueued = false;
function scheduleRedraw() {
  if (redrawQueued) return;
  redrawQueued = true;
  requestAnimationFrame(() => { redrawQueued = false; redraw(); });
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

  const rows = result.x.map((x, i) => ({ x, T: result.T[i], ion: result.ion[i] }));
  const band = {
    data: { values: [{ y0: 1500, y1: 2500 }] },
    mark: { type: "rect", opacity: 0.15, color: "orange" },
    encoding: { y: { field: "y0", type: "quantitative" }, y2: { field: "y1" } },
  };
  vegaEmbed("#chart-T", chartSpec("T", result.xlabel, "Temperature (K)", rows, [band]),
            { actions: false, renderer: "svg" });
  vegaEmbed("#chart-ion", chartSpec("ion", result.xlabel, "H⁺ / H_tot", rows),
            { actions: false, renderer: "svg" });

  const finalT = result.T[result.T.length - 1];
  const finalIon = result.ion[result.ion.length - 1];
  const finalH2 = result.h2[result.h2.length - 1];
  let statusText = `${rows.length} steps, ${elapsed.toFixed(1)} ms -- final T=${finalT ? finalT.toFixed(1) : "?"} K, `
    + `ionized=${finalIon ? finalIon.toExponential(2) : "?"}`;
  if (finalH2 !== null && finalH2 !== undefined) statusText += `, H2/H_tot=${finalH2.toExponential(2)}`;
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

function initPage(config) {
  document.getElementById("page-title").textContent = config.title;
  document.getElementById("T").value = Math.log10(config.default_T);

  document.getElementById("mode-cool").addEventListener("click", () => setMode("cool"));
  document.getElementById("mode-freefall").addEventListener("click", () => setMode("freefall"));
  for (const id of ["nH", "T", "dtf", "ntarget"]) {
    document.getElementById(id).addEventListener("input", scheduleRedraw);
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
    document.getElementById("status").textContent = "ready";
    redraw();
  });
}
