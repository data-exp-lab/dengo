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
let speciesDisplayMode = "density"; // "density" (cm^-3) or "massfrac" (X_i, dimensionless)
let temperatureDisplayMode = "T"; // "T" (K) or "ge" (specific internal energy, erg/g)

// -- LaTeX axis labels (rendered via KaTeX, not Vega-Lite's own plain-text
// titles -- see NOTES.md for why: Vega-Lite axis titles are just SVG
// <text>, no math rendering, so proper sub/superscripts need an external
// renderer laid out alongside the chart instead). ----------------------
const AXIS_LATEX = {
  time: "t\\ (\\mathrm{s})",
  density: "n\\ (\\mathrm{cm^{-3}})",
  T: "T\\ (\\mathrm{K})",
  ionfrac: "X / \\mathrm{H_{tot}}",
  species: "n_i\\ (\\mathrm{cm^{-3}})",
  massfrac: "X_i",
  ge: "\\varepsilon\\ (\\mathrm{erg\\ g^{-1}})",
  h2frac: "\\mathrm{H_2} / \\mathrm{H_{tot}}",
};

// Ionized fraction is important, but so is molecular fraction -- both are
// "fraction of total hydrogen", so they share one chart/y-axis instead of
// H2 only getting a mention in the status line. Small, fixed 2-series
// legend, so (unlike the species-abundance chart's dynamic set) a plain
// Vega-Lite legend is the right amount of machinery here -- no toggle
// checkboxes needed.
const ION_H2_LABELS = { ion: "H⁺ / H_tot", h2: "H₂ / H_tot" };
const ION_H2_COLORS = {
  light: { ion: "#3366cc", h2: "#e07b00" },
  dark: { ion: "#8ab4ff", h2: "#ffb454" },
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

// Atomic/molecular mass, in amu, keyed by the base element symbol every
// dengo species name starts with (the part before the first "_" --
// "H2_1" -> "H2", "He_3" -> "He", "H_m0" -> "H"; "de" has no underscore
// at all and is its own base). Covers every species the three fiducial
// networks generate; an unrecognized base (some future network) just
// can't be shown in mass-fraction mode -- see speciesMassAmu()/redraw().
const SPECIES_MASS_AMU = { H: 1.00794, He: 4.002602, H2: 2.01588, de: 5.485799e-4 };

function speciesMassAmu(name) {
  return SPECIES_MASS_AMU[name.split("_")[0]];
}

// Total mass (amu * cm^-3, i.e. missing only the shared amu-to-gram
// factor that cancels out of any mass *fraction*) summed over every
// species with a known mass -- the denominator for mass-fraction mode.
// Electrons are included for self-consistency with whatever's actually
// tracked, though their mass contribution is of course negligible.
function totalMassAmu(s) {
  let total = 0;
  for (const name of speciesNames) {
    if (name === "ge") continue;
    const m = speciesMassAmu(name);
    if (m) total += s[name] * m;
  }
  return total;
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

// Sweep charts compare several tracks' *shape* against each other; a
// track whose local cooling/heating timescale is much longer than the
// whole requested run legitimately lets the unforced path below take
// one giant step straight from t=0 to t=dtfTotal (efficient and correct
// for a single run) -- but that renders as a single point with no shape
// at all, which defeats a comparison chart's entire purpose. Sweep runs
// instead sample a fixed, shared, log-spaced checkpoint grid (every
// track gets the same number of points on the same time grid, however
// fast it individually converges), leaning on BE_chem_solve's own
// internal adaptive sub-stepping -- it already halves dt on
// non-convergence -- for whatever a particular gap between checkpoints
// turns out to need, the same robustness the unforced path already
// relies on for its own (usually much larger) single jump.
const SWEEP_CHECKPOINTS = 24;

function runConstantDensity(nH, T, fractions, logDtf, safetyFactor = 0.1, maxSteps = 2000, forSweep = false) {
  setIcs(nH, T, fractions);
  const dtfTotal = Math.pow(10, logDtf);
  let t = 0;
  const tHist = [], THist = [], ionHist = [], h2Hist = [], dtHist = [], sHist = [];
  // The initial condition itself is worth plotting -- without it every
  // chart started mid-story, at whatever state the first accepted step
  // happened to reach, never showing where the run actually began.
  // temperature() reads a *cached* value the solver only refreshes
  // inside calculate_rhs/calculate_jacobian -- setIcs() writes species
  // and ge directly and doesn't itself trigger either, so temperature()
  // here would otherwise still report whatever the *previous* run last
  // left behind. rhsPtr() (already used by coolingTime()) runs
  // calculate_rhs as a side effect, which is what actually refreshes it.
  rhsPtr();
  const s0 = getScalar();
  tHist.push(0); THist.push(temperature()); ionHist.push(ionizedFraction(s0)); h2Hist.push(h2Fraction(s0));
  dtHist.push(0); sHist.push(s0);
  if (forSweep) {
    for (let i = 0; i < SWEEP_CHECKPOINTS; i++) {
      const target = dtfTotal * Math.pow(10, -4 * (1 - (i + 1) / SWEEP_CHECKPOINTS));
      const dt = target - t;
      if (dt <= 0) continue;
      const converged = step(dt, 200, 1e-5);
      if (!converged) break;
      t = target;
      const s = getScalar();
      tHist.push(t); THist.push(temperature()); ionHist.push(ionizedFraction(s)); h2Hist.push(h2Fraction(s));
      dtHist.push(dt); sHist.push(s);
    }
    return { x: tHist, t: tHist, dt: dtHist, T: THist, ion: ionHist, h2: h2Hist, s: sHist, xKey: "time" };
  }
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

// Rankine-Hugoniot jump conditions for a plane, ideal-gas shock at
// upstream Mach number `mach` and (composition-weighted) adiabatic
// index `gammaAd` -- standard closed-form result, not the strong-shock
// limit, so it correctly gives no jump at all at mach=1 (the
// zero-strength/sonic limit) and saturates the density ratio (but not
// the temperature ratio, which keeps growing) as mach -> infinity.
// mach is a free dial here, not derived from an actual radius/enclosed-
// mass infall-speed calculation this zero-dimensional (density-only)
// model has no way to provide -- see the note in generate_site.py.
function shockJumpFactors(gammaAd, mach) {
  const m2 = mach * mach;
  const rhoRatio = ((gammaAd + 1) * m2) / ((gammaAd - 1) * m2 + 2);
  const TRatio = ((2 * gammaAd * m2 - (gammaAd - 1)) * ((gammaAd - 1) * m2 + 2))
    / ((gammaAd + 1) * (gammaAd + 1) * m2);
  return { rhoRatio, TRatio };
}

function runFreefall(nH, T, fractions, logNTarget, logNShock, machShock, safetyFactor = 0.01, maxSteps = 10000) {
  setIcs(nH, T, fractions);
  const nTarget = Math.pow(10, logNTarget);
  const nShock = Math.pow(10, logNShock);
  let nCurrent = nH, t = 0;
  let shocked = nCurrent >= nShock; // already past it at t=0 -- don't fire mid-run
  let shockApplied = false; // only set once the post-shock step actually converges -- see below
  const nHist = [], THist = [], ionHist = [], h2Hist = [], tHist = [], dtHist = [], sHist = [];
  // The initial condition itself is worth plotting, same reasoning as
  // runConstantDensity() -- and free-fall's x-axis is density, always
  // positive, so (unlike time) there's no log-scale concern in placing
  // it as the very first point.
  {
    // temperature()'s cache needs an explicit refresh here too -- see
    // the identical comment in runConstantDensity().
    rhsPtr();
    const s0 = getScalar();
    nHist.push(nCurrent); THist.push(temperature()); ionHist.push(ionizedFraction(s0)); h2Hist.push(h2Fraction(s0));
    tHist.push(0); dtHist.push(0); sHist.push(s0);
  }
  for (let i = 0; i < maxSteps; i++) {
    if (nCurrent >= nTarget) break;
    const rho = nCurrent * MH;
    const tFf = Math.sqrt(3 * Math.PI / (32 * G_GRAV * rho));
    const dt = safetyFactor * tFf;
    const rhoNew = Math.pow(Math.pow(rho, -0.5) - Math.sqrt(32 * G_GRAV / (3 * Math.PI)) * dt, -2);
    let densityRatio = rhoNew / rho;

    const ptr = statePtr() >> 3;
    const gammaAd = thermodynamicGamma(getScalar());
    let tempRatio = 1 + (gammaAd - 1) * (densityRatio - 1);

    // Fires (at most) once per run, the first step whose ordinary
    // free-fall compression would carry the gas across nShock. A shock
    // is a genuine mathematical discontinuity -- that's what the RH jump
    // conditions describe -- so a single-step jump is the physically
    // honest way to represent one here, not a numerical shortcut to
    // smooth over; BE_chem_solve already tolerates state jumps of this
    // kind fine (ordinary free-fall compression already hands it one
    // every step). `shocked` only becomes permanent once this step's
    // step() call below actually converges -- a strong-enough jump can
    // fail to converge on the first attempt, and if it does, this isn't
    // "the shock happened", it's "the run stopped before the shock could
    // be applied" (see shockTriggered below).
    let attemptingShock = false;
    if (!shocked && machShock > 1 && nCurrent * densityRatio >= nShock) {
      const { rhoRatio, TRatio } = shockJumpFactors(gammaAd, machShock);
      densityRatio *= rhoRatio;
      tempRatio *= TRatio;
      attemptingShock = true;
    }

    for (const name of speciesNames) {
      if (name !== "ge") mod.HEAPF64[ptr + idx[name]] *= densityRatio;
    }
    mod.HEAPF64[ptr + idx.ge] *= tempRatio;

    const converged = step(dt, 200, 1e-5);
    if (!converged) break;
    if (attemptingShock) { shocked = true; shockApplied = true; }
    t += dt;
    const s = getScalar();
    nCurrent = 0;
    for (const name of speciesNames) if (name !== "ge" && name !== "de") nCurrent += s[name];
    nHist.push(nCurrent); THist.push(temperature()); ionHist.push(ionizedFraction(s)); h2Hist.push(h2Fraction(s));
    tHist.push(t); dtHist.push(dt); sHist.push(s);
  }
  return {
    x: nHist, t: tHist, dt: dtHist, T: THist, ion: ionHist, h2: h2Hist, s: sHist, xKey: "density",
    shockTriggered: shockApplied, nShock,
  };
}

const FIELD_TITLE = { T: "T (K)", ge: "ε (erg/g)" };

function chartSpec(field, xKey, data, extra, extraTooltip) {
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
            field: "x", type: "quantitative",
            // Time (unlike density) can legitimately be exactly 0 now
            // that the initial condition itself is plotted -- symlog
            // (linear near zero, log further out) shows that point
            // instead of silently dropping it the way a pure log scale
            // would.
            scale: { type: xKey === "time" ? "symlog" : "log" },
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
            ...(extraTooltip || []),
          ],
        },
      },
    ],
  };
}

// Ionized fraction and molecular (H2) fraction together, long-format
// (one row per (step, quantity) pair, `quantity` already holding the
// human-readable legend label) -- see ION_H2_LABELS/ION_H2_COLORS above.
// h2 rows are simply absent for a network with no H2 species (rather
// than plotting a bogus flat line), same graceful-degradation as the
// status line's H2/H_tot readout.
function ionizationChartSpec(xKey, rows) {
  const n = rows.length;
  const pointSize = Math.max(4, Math.min(30, 2000 / Math.max(n, 1)));
  const palette = isDarkMode() ? ION_H2_COLORS.dark : ION_H2_COLORS.light;
  // Built from what's actually *present* in `rows`, not unconditionally
  // both labels -- a hardcoded domain would draw a legend entry for H2
  // even on a network with no H2 species at all (no h2 rows are ever
  // pushed for one; see redraw()), which is misleading on its own.
  const present = new Set(rows.map((r) => r.quantity));
  const domain = [ION_H2_LABELS.ion, ION_H2_LABELS.h2].filter((label) => present.has(label));
  const range = domain.map((label) => (label === ION_H2_LABELS.ion ? palette.ion : palette.h2));
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    width: 600, height: 220, background: null,
    config: vlConfig(),
    data: { values: rows },
    mark: { type: "line", point: { filled: true, size: pointSize, opacity: 0.9 } },
    encoding: {
      x: {
        field: "x", type: "quantitative",
        // Time (unlike density) can legitimately be exactly 0 now that
        // the initial condition itself is plotted -- symlog (linear
        // near zero, log further out) shows that point instead of
        // silently dropping it the way a pure log scale would.
        scale: { type: xKey === "time" ? "symlog" : "log" },
        axis: {
          title: null, labelOverlap: "greedy",
          labelAngle: xKey === "time" ? -40 : 0,
          labelExpr: xKey === "time" ? TIME_LABEL_EXPR : undefined,
        },
      },
      y: { field: "value", type: "quantitative", scale: { type: "log" }, axis: { title: null } },
      color: {
        field: "quantity", type: "nominal",
        scale: { domain, range },
        legend: { title: null, orient: "top-right" },
      },
      tooltip: [
        { field: "quantity", title: "quantity", type: "nominal" },
        { field: "i", title: "step", type: "quantitative" },
        { field: "x", title: xKey === "time" ? "t (s)" : "n (cm⁻³)", type: "quantitative", format: ".3~g" },
        { field: "tHuman", title: "t", type: "nominal" },
        { field: "dt", title: "step Δt (s)", type: "quantitative", format: ".3~g" },
        { field: "value", title: "fraction", type: "quantitative", format: ".4~g" },
      ],
    },
  };
}

// A multi-series companion to chartSpec(): one line+points per selected
// species (long-format `rows`, one row per (step, species) pair), color
// keyed to the same fixed per-species palette the toggle checkboxes use.
// This is the direct answer to "I can't get an impression of the full
// range of values" -- T and the H+/H_tot ratio were the only things
// plotted before; nothing showed the actual per-species number densities
// or how many decades they span.
function speciesChartSpec(xKey, rows, valueTitle) {
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
        field: "x", type: "quantitative",
        // Time (unlike density) can legitimately be exactly 0 now that
        // the initial condition itself is plotted -- symlog (linear
        // near zero, log further out) shows that point instead of
        // silently dropping it the way a pure log scale would.
        scale: { type: xKey === "time" ? "symlog" : "log" },
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
        { field: "value", title: valueTitle, type: "quantitative", format: ".4~g" },
      ],
    },
  };
}

// One line per swept-parameter value (long-format `rows`, each already
// carrying a human-readable `label` -- e.g. "T0=1.0e+03 K" -- as the
// literal color/legend field). Ordinal, not nominal: an explicit
// `domain` in ascending physical order plus a sequential color scheme
// (viridis) reads as "low to high" at a glance, which a fixed set of
// starting conditions warrants and species names (nominal, no natural
// order) didn't. Point markers are small but not optional here (unlike
// the single-run charts, where "where are the steps" is a bonus, not a
// visibility requirement): a run whose adaptive stepper's very first
// step already reaches the requested end time -- legitimate, e.g. a
// starting temperature so far from equilibrium that coolingTime() is
// negligible next to the whole run -- produces exactly one point, and a
// line mark with nothing to connect draws *nothing at all*. Without a
// point, that track would just silently vanish from the chart.
function sweepChartSpec(field, xKey, rows, domainLabels) {
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    width: 600, height: 240, background: null,
    config: vlConfig(),
    data: { values: rows },
    mark: { type: "line", point: { filled: true, size: 12, opacity: 0.9 } },
    encoding: {
      x: {
        field: "x", type: "quantitative",
        // Time (unlike density) can legitimately be exactly 0 now that
        // the initial condition itself is plotted -- symlog (linear
        // near zero, log further out) shows that point instead of
        // silently dropping it the way a pure log scale would.
        scale: { type: xKey === "time" ? "symlog" : "log" },
        axis: {
          title: null, labelOverlap: "greedy",
          labelAngle: xKey === "time" ? -40 : 0,
          labelExpr: xKey === "time" ? TIME_LABEL_EXPR : undefined,
        },
      },
      y: { field: field, type: "quantitative", scale: { type: "log" }, axis: { title: null } },
      color: {
        field: "label", type: "ordinal",
        scale: { domain: domainLabels, scheme: "viridis" },
        legend: { title: null, symbolLimit: domainLabels.length },
      },
      tooltip: [
        { field: "label", title: "run", type: "nominal" },
        { field: "x", title: xKey === "time" ? "t (s)" : "n (cm⁻³)", type: "quantitative", format: ".3~g" },
        { field: "tHuman", title: "t", type: "nominal" },
        { field: field, title: field === "T" ? "T (K)" : "H2/H_tot", type: "quantitative", format: ".4~g" },
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
    const logNShock = parseFloat(document.getElementById("nshock").value);
    const machShock = parseFloat(document.getElementById("mach").value);
    document.getElementById("nshock-val").textContent = logNShock.toFixed(1);
    document.getElementById("mach-val").textContent = machShock.toFixed(1);
    result = runFreefall(nH, T, fractions, logNTarget, logNShock, machShock);
  } else {
    const logDtf = parseFloat(document.getElementById("dtf").value);
    document.getElementById("dtf-val").textContent = logDtf.toFixed(1);
    result = runConstantDensity(nH, T, fractions, logDtf);
  }
  const elapsed = performance.now() - t0;

  // gamma (thermodynamicGamma) is exposed on every point's tooltip
  // regardless of which of T/ge is plotted -- it's the thing that
  // connects them (T = (gamma-1) * ge * mu / k_B), and it moves on its
  // own as composition shifts (e.g. H2 formation dropping gamma from
  // 5/3 toward 7/5), so this is the direct way to "see the impact of
  // gamma": pick thermal energy mode (ge is what the solver actually
  // conserves/evolves) and watch gamma in the tooltip while T bends.
  const rows = result.x.map((x, i) => ({
    x, T: result.T[i], ge: result.s[i].ge, ion: result.ion[i],
    gamma: thermodynamicGamma(result.s[i]),
    i, dt: result.dt[i], tHuman: formatTimeAuto(result.t[i]),
  }));
  const gammaTooltip = [{ field: "gamma", title: "gamma", type: "quantitative", format: ".4f" }];
  const band = {
    data: { values: [{ y0: 1500, y1: 2500 }] },
    mark: { type: "rect", opacity: isDarkMode() ? 0.25 : 0.15, color: "orange" },
    encoding: { y: { field: "y0", type: "quantitative" }, y2: { field: "y1" } },
  };
  const tempField = temperatureDisplayMode === "ge" ? "ge" : "T";
  const tempExtra = tempField === "T" ? [band] : [];
  if (result.shockTriggered) {
    // A real shock is a discontinuity, and this one only ever fires
    // once -- marking exactly where lets the jump in the curve read as
    // "the shock" rather than looking like a numerical glitch.
    tempExtra.push({
      data: { values: [{ x: result.nShock }] },
      mark: { type: "rule", strokeDash: [4, 3], opacity: 0.8, color: isDarkMode() ? "#ff8a80" : "#cc3333" },
      encoding: { x: { field: "x", type: "quantitative" } },
    });
  }
  vegaEmbed("#chart-T", chartSpec(tempField, result.xKey, rows, tempExtra, gammaTooltip),
            { actions: false, renderer: "svg" });
  const ionRows = [];
  for (let i = 0; i < result.x.length; i++) {
    const base = { x: result.x[i], i, dt: result.dt[i], tHuman: formatTimeAuto(result.t[i]) };
    ionRows.push({ ...base, quantity: ION_H2_LABELS.ion, value: result.ion[i] });
    const h2v = result.h2[i];
    if (h2v !== null && h2v !== undefined && h2v > 0) {
      ionRows.push({ ...base, quantity: ION_H2_LABELS.h2, value: h2v });
    }
  }
  vegaEmbed("#chart-ion", ionizationChartSpec(result.xKey, ionRows),
            { actions: false, renderer: "svg" });
  renderLatex("ylabel-T", tempField);
  renderLatex("xlabel-T", result.xKey);
  renderLatex("ylabel-ion", "ionfrac");
  renderLatex("xlabel-ion", result.xKey);

  const massFracMode = speciesDisplayMode === "massfrac";
  const selectedSpecies = massFracMode
    ? selectedSpeciesNames().filter((name) => speciesMassAmu(name) !== undefined)
    : selectedSpeciesNames();
  const chartSpeciesEl = document.getElementById("chart-species");
  if (selectedSpecies.length) {
    const speciesRows = [];
    for (let i = 0; i < result.x.length; i++) {
      const tHuman = formatTimeAuto(result.t[i]);
      const denom = massFracMode ? totalMassAmu(result.s[i]) : 1;
      for (const name of selectedSpecies) {
        let v = result.s[i][name];
        if (massFracMode) v = (v * speciesMassAmu(name)) / denom;
        if (v > 0) speciesRows.push({ x: result.x[i], i, dt: result.dt[i], tHuman, species: name, value: v });
      }
    }
    chartSpeciesEl.innerHTML = "";
    const valueTitle = massFracMode ? "X_i (mass frac.)" : "n_i (cm⁻³)";
    vegaEmbed(chartSpeciesEl, speciesChartSpec(result.xKey, speciesRows, valueTitle), { actions: false, renderer: "svg" });
  } else if (massFracMode) {
    chartSpeciesEl.innerHTML = '<p class="chart-placeholder">No selected species has a known mass -- '
      + 'toggle one on, or switch back to number density.</p>';
  } else {
    chartSpeciesEl.innerHTML = '<p class="chart-placeholder">Toggle one or more species above to plot them.</p>';
  }
  renderLatex("ylabel-species", massFracMode ? "massfrac" : "species");
  renderLatex("xlabel-species", result.xKey);

  const finalT = result.T[result.T.length - 1];
  const finalIon = result.ion[result.ion.length - 1];
  const finalH2 = result.h2[result.h2.length - 1];
  const finalTime = result.t[result.t.length - 1];
  let statusText = `${rows.length} steps, ${elapsed.toFixed(1)} ms -- final T=${finalT ? finalT.toFixed(1) : "?"} K, `
    + `ionized=${finalIon ? finalIon.toExponential(2) : "?"}`;
  if (finalH2 !== null && finalH2 !== undefined) statusText += `, H2/H_tot=${finalH2.toExponential(2)}`;
  if (finalTime !== undefined) statusText += `, elapsed t=${formatTimeAuto(finalTime)} (${finalTime.toExponential(2)} s)`;
  if (result.shockTriggered) statusText += `, shock crossed at n=${result.nShock.toExponential(2)} cm⁻³`;
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

// A few physically-motivated starting points, so exploring the widget
// doesn't have to start from dragging every slider by hand. Each entry
// is just a (nH, T, per-species fraction) bundle -- no new solver-side
// machinery: the "z" in a preset's label/note is baked into the density
// number at authoring time (n_H(z) ~ n_H,0 (1+z)^3, mean cosmic
// baryon/hydrogen density scaling), not a live redshift parameter, so
// there's no UV-background/Compton-coupling physics implied here beyond
// what the network already models (the compton cooling term always runs
// at z=0 in this widget, same as before). Values are order-of-magnitude
// illustrative, not a precision cosmological calculation -- see NOTES.md.
const IC_PRESETS = {
  "bg-z20": {
    nH: 1.8e-3, T: 10,
    fractions: { H_1: 0.9998, H_2: 2e-4, He_1: 0.063, He_2: 1e-7, He_3: 1e-12, H2_1: 2e-6, H2_2: 1e-12, H_m0: 1e-12 },
    note: "Mean IGM density at z≈20 (n_H ∝ (1+z)³) and an adiabatically-cooled gas "
      + "temperature well below T_CMB at that era; residual ionization x_e≈2×10⁻⁴, "
      + "trace gas-phase H2≈2×10⁻⁶.",
  },
  "bg-z1000": {
    nH: 190, T: 3000,
    fractions: { H_1: 0.4, H_2: 0.39, He_1: 0.063, He_2: 1e-7, He_3: 1e-12, H2_1: 1e-12, H2_2: 1e-12, H_m0: 1e-12 },
    note: "Mean IGM density at z≈1000 (n_H ∝ (1+z)³), near the peak of the "
      + "recombination visibility function -- gas tightly Compton-coupled to "
      + "T_CMB, hydrogen roughly half-ionized. Illustrative only: this network "
      + "has no photoionizing background of its own.",
  },
  "virial-shock": {
    nH: 0.3, T: 4000,
    fractions: { H_1: 0.79, H_2: 2e-4, He_1: 0.063, He_2: 1e-7, He_3: 1e-12, H2_1: 1e-6, H2_2: 1e-12, H_m0: 1e-12 },
    note: "Post-shock virial temperature (Barkana & Loeb 2001 fit) and density "
      + "(Δ_vir≈178× the cosmic mean at z≈20) for a 10⁶ M☉ minihalo -- the "
      + "classic first-star-forming halo scale, with a trace H2 seed so H2 "
      + "cooling can trigger collapse.",
  },
  "protostellar-disk": {
    nH: 1e13, T: 800,
    fractions: { H_1: 0.05, H_2: 1e-8, He_1: 0.063, He_2: 1e-8, He_3: 1e-12, H2_1: 0.37, H2_2: 1e-6, H_m0: 1e-12 },
    note: "Disk-forming-region density/temperature, hydrogen mostly molecular -- "
      + "3-body H2 formation saturates near unity by this density, roughly "
      + "where free-fall collapse from primordial conditions ends up.",
  },
};

function applyPreset(key) {
  const noteEl = document.getElementById("preset-note");
  const preset = IC_PRESETS[key];
  if (!preset) { noteEl.textContent = ""; return; }
  document.getElementById("nH").value = Math.log10(preset.nH);
  document.getElementById("T").value = Math.log10(preset.T);
  // A network without H2 chemistry (primordial_atomic, hydrogen_minimal)
  // has no sp-H2_1/sp-H2_2 slider to receive a preset's molecular-hydrogen
  // fraction -- silently dropping it would make that much of the
  // hydrogen budget just vanish (e.g. protostellar-disk's H_1=0.05
  // assumes ~0.74 more is locked up in H2). Fold it back into atomic H
  // instead, so the total hydrogen fraction stays physically sensible on
  // every network rather than only on the one(s) that can represent H2.
  const hasH2 = document.getElementById("sp-H2_1") !== null;
  for (const name of plotableSpecies()) {
    const el = document.getElementById("sp-" + name);
    if (!el) continue; // this network doesn't have that species -- nothing to set
    let frac = (preset.fractions && preset.fractions[name] !== undefined) ? preset.fractions[name] : 1e-12;
    if (name === "H_1" && !hasH2 && preset.fractions) {
      frac += 2 * ((preset.fractions.H2_1 || 0) + (preset.fractions.H2_2 || 0));
    }
    el.value = Math.log10(frac);
  }
  noteEl.textContent = preset.note;
  scheduleRedraw();
}

// Parameter sweep: hold every slider at its current setting except one,
// run the current mode (cool/free-fall) once per sampled value of that
// one, overlay all of them. `sliderId` is read/written directly (same
// elements redraw() itself reads), so this generalizes to any slider
// with no per-parameter special-casing beyond how to convert between
// the slider's own raw units and physical ones (`toRaw`/`toPhysical`):
// T/nH/H2-fraction/shock-density sliders already store log10(physical
// value), so their raw<->physical conversion is log10/10^x; mach's
// slider is linear-in-Mach already, so its conversion is the identity.
// The *sampled* start/stop/count all live in physical units (an actual
// Kelvin range, not a log10 one) -- typing "1000" to "10000" should mean
// what it says regardless of which of these a slider happens to store
// internally -- and `logSpace` controls whether count values are spread
// evenly in physical space or in log-physical space between them.
const SWEEP_PARAMS = {
  T: { sliderId: "T", label: "T₀", unit: "K", logSpace: true, toPhysical: (v) => Math.pow(10, v), toRaw: (v) => Math.log10(v) },
  nH: { sliderId: "nH", label: "n_H,0", unit: "cm⁻³", logSpace: true, toPhysical: (v) => Math.pow(10, v), toRaw: (v) => Math.log10(v) },
  "sp-H2_1": { sliderId: "sp-H2_1", label: "H2 frac₀", unit: "", logSpace: true, toPhysical: (v) => Math.pow(10, v), toRaw: (v) => Math.log10(v) },
  // Mach's *default* range (see updateSweepParamUI) is still log-spaced
  // for the same reason as before (the no-effect -> partial ->
  // saturated transition happens over the first factor of ~10) -- but
  // now that start/stop are explicit and user-editable, that's just the
  // default, not baked into the sampling itself; sample linearly across
  // whatever range is actually entered, same as every other parameter.
  mach: { sliderId: "mach", label: "shock Mach", unit: "", logSpace: false, toPhysical: (v) => v, toRaw: (v) => v },
  nshock: { sliderId: "nshock", label: "shock n", unit: "cm⁻³", logSpace: true, toPhysical: (v) => Math.pow(10, v), toRaw: (v) => Math.log10(v) },
};

function sweepFormat(param, physical) {
  const s = param.unit === "" ? physical.toPrecision(3) : physical.toExponential(2);
  return `${param.label}=${s}${param.unit ? " " + param.unit : ""}`;
}

// Reads the visible from/to/count inputs (physical units) and returns
// raw slider-unit values, ready to assign straight to `slider.value`.
function sweepValues(paramKey) {
  const param = SWEEP_PARAMS[paramKey];
  const startPhys = parseFloat(document.getElementById("sweep-start").value);
  const stopPhys = parseFloat(document.getElementById("sweep-stop").value);
  const count = Math.max(2, Math.min(12, parseInt(document.getElementById("sweep-count").value, 10) || 6));
  if (param.logSpace) {
    const lo = Math.log10(startPhys), hi = Math.log10(stopPhys);
    return Array.from({ length: count }, (_, i) => param.toRaw(Math.pow(10, lo + (hi - lo) * i / (count - 1))));
  }
  return Array.from({ length: count }, (_, i) => param.toRaw(startPhys + (stopPhys - startPhys) * i / (count - 1)));
}

function buildSweepParamOptions() {
  const select = document.getElementById("sweep-param");
  select.innerHTML = "";
  const noneOpt = document.createElement("option");
  noneOpt.value = "";
  noneOpt.textContent = "No sweep";
  select.appendChild(noneOpt);
  for (const [key, param] of Object.entries(SWEEP_PARAMS)) {
    if (!document.getElementById(param.sliderId)) continue; // e.g. no H2 species on this network
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = param.label + (param.unit ? ` (${param.unit})` : "");
    select.appendChild(opt);
  }
}

// Disables whichever slider is currently the sweep target (it's driven
// by the from/to/count range instead while selected) and re-enables the
// previous one, then refills from/to with that slider's own min/max in
// physical units -- a reasonable, visible-and-editable default rather
// than the old behavior of silently always using the full slider range
// no matter what it was actually set to. "No sweep" (paramKey === "")
// is the explicit off state: nothing disabled, range inputs cleared and
// inert, run-sweep button off.
let sweptSliderId = null;
function updateSweepParamUI() {
  const paramKey = document.getElementById("sweep-param").value;
  if (sweptSliderId) {
    const prev = document.getElementById(sweptSliderId);
    if (prev) prev.disabled = false;
    sweptSliderId = null;
  }
  const param = SWEEP_PARAMS[paramKey];
  const runButton = document.getElementById("run-sweep");
  const rangeInputs = [document.getElementById("sweep-start"), document.getElementById("sweep-stop"), document.getElementById("sweep-count")];
  if (!param) {
    for (const el of rangeInputs) { el.value = ""; el.disabled = true; }
    runButton.disabled = true;
    document.getElementById("sweep-status").textContent = "";
    return;
  }
  for (const el of rangeInputs) el.disabled = false;
  runButton.disabled = false;
  sweptSliderId = param.sliderId;
  const slider = document.getElementById(param.sliderId);
  slider.disabled = true;
  const minPhys = param.toPhysical(parseFloat(slider.min));
  const maxPhys = param.toPhysical(parseFloat(slider.max));
  document.getElementById("sweep-start").value = Number(minPhys.toPrecision(4));
  document.getElementById("sweep-stop").value = Number(maxPhys.toPrecision(4));
  document.getElementById("sweep-count").value = 6;
}

function runSweep() {
  if (!document.getElementById("sweep-param").value) return; // "No sweep" -- button is disabled anyway, but guard directly too
  const button = document.getElementById("run-sweep");
  const statusEl = document.getElementById("sweep-status");
  button.disabled = true;
  statusEl.textContent = "running sweep…";
  // Yield one frame so the browser actually paints the line above before
  // the (synchronous, possibly multi-second) sweep loop blocks the main
  // thread -- without this, "running..." would never be visible.
  requestAnimationFrame(() => setTimeout(runSweepBody, 0));
}

function runSweepBody() {
  const button = document.getElementById("run-sweep");
  const statusEl = document.getElementById("sweep-status");
  const paramKey = document.getElementById("sweep-param").value;
  const param = SWEEP_PARAMS[paramKey];
  const slider = document.getElementById(param.sliderId);
  const savedValue = slider.value;
  const values = sweepValues(paramKey);

  const t0 = performance.now();
  const tRows = [], h2Rows = [];
  const domainLabels = [];
  let xKey = "time";
  for (const v of values) {
    slider.value = v;
    const nH = Math.pow(10, parseFloat(document.getElementById("nH").value));
    const T = Math.pow(10, parseFloat(document.getElementById("T").value));
    const fractions = currentFractions();
    let result;
    if (currentMode === "freefall") {
      const logNTarget = parseFloat(document.getElementById("ntarget").value);
      const logNShock = parseFloat(document.getElementById("nshock").value);
      const machShock = parseFloat(document.getElementById("mach").value);
      result = runFreefall(nH, T, fractions, logNTarget, logNShock, machShock);
    } else {
      const logDtf = parseFloat(document.getElementById("dtf").value);
      result = runConstantDensity(nH, T, fractions, logDtf, undefined, undefined, true);
    }
    xKey = result.xKey;
    const label = sweepFormat(param, param.toPhysical(v));
    domainLabels.push(label);
    for (let i = 0; i < result.x.length; i++) {
      const base = { x: result.x[i], label, tHuman: formatTimeAuto(result.t[i]) };
      tRows.push({ ...base, T: result.T[i] });
      const h2v = result.h2[i];
      if (h2v !== null && h2v !== undefined && h2v > 0) h2Rows.push({ ...base, h2: h2v });
    }
  }
  slider.value = savedValue; // restore -- a sweep looks at other conditions, it doesn't change this one

  vegaEmbed("#chart-sweep-T", sweepChartSpec("T", xKey, tRows, domainLabels), { actions: false, renderer: "svg" });
  const chartH2El = document.getElementById("chart-sweep-h2");
  if (h2Rows.length) {
    chartH2El.innerHTML = "";
    vegaEmbed(chartH2El, sweepChartSpec("h2", xKey, h2Rows, domainLabels), { actions: false, renderer: "svg" });
  } else {
    chartH2El.innerHTML = '<p class="chart-placeholder">This network has no H2 species.</p>';
  }
  renderLatex("ylabel-sweep-T", "T");
  renderLatex("xlabel-sweep-T", xKey);
  renderLatex("ylabel-sweep-h2", "h2frac");
  renderLatex("xlabel-sweep-h2", xKey);

  const elapsed = performance.now() - t0;
  statusEl.textContent = `${values.length} runs, ${elapsed.toFixed(0)} ms`;
  button.disabled = false;
  redraw(); // the loop above left nH/T/etc.'s underlying wasm state at the last swept run's -- put the primary charts back to what the sliders actually show
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

function setTemperatureDisplayMode(mode) {
  temperatureDisplayMode = mode;
  document.getElementById("T-mode-T").classList.toggle("active", mode === "T");
  document.getElementById("T-mode-ge").classList.toggle("active", mode === "ge");
  document.getElementById("chart-T-title").textContent = mode === "ge" ? "Thermal energy" : "Temperature";
  scheduleRedraw();
}

function setSpeciesDisplayMode(mode) {
  speciesDisplayMode = mode;
  document.getElementById("species-mode-density").classList.toggle("active", mode === "density");
  document.getElementById("species-mode-massfrac").classList.toggle("active", mode === "massfrac");
  scheduleRedraw();
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
  document.getElementById("species-mode-density").addEventListener("click", () => setSpeciesDisplayMode("density"));
  document.getElementById("species-mode-massfrac").addEventListener("click", () => setSpeciesDisplayMode("massfrac"));
  document.getElementById("T-mode-T").addEventListener("click", () => setTemperatureDisplayMode("T"));
  document.getElementById("T-mode-ge").addEventListener("click", () => setTemperatureDisplayMode("ge"));
  document.getElementById("ic-preset").addEventListener("change", (e) => applyPreset(e.target.value));
  document.getElementById("run-sweep").addEventListener("click", runSweep);
  document.getElementById("sweep-param").addEventListener("change", updateSweepParamUI);
  for (const id of ["nH", "T", "dtf", "ntarget", "nshock", "mach"]) {
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
    buildSweepParamOptions();
    updateSweepParamUI(); // sets up the "No sweep" default state (run-sweep button included) immediately, not just after the dropdown is touched
    document.getElementById("ic-preset").disabled = false;
    document.getElementById("status").textContent = "ready";
    redraw();
  });
}
