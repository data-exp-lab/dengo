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
let temperatureDisplayMode = "T"; // "T" (K) or "ge" (specific internal energy, erg/g)
let lastResult = null; // the current mode's most recent full run (redraw()'s own result), for CSV export
let pageTitle = "dengo"; // network title, for the exported CSV's filename only
let runView = null; // the current run-view's Vega View object, re-set on every redraw()

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

// The species actually worth plotting as an abundance: everything except
// `ge` (specific internal energy, not a density -- already has its own
// Temperature chart). `de` (electron density) is included; it's exactly
// as physical as any other species here. Color assignment is Vega-Lite's
// own "tableau10" scheme (see speciesPanel() below) -- a standard,
// well-vetted 10-color qualitative palette, not a hand-picked array;
// swapped in after the previous custom 10-color array (d3's older
// "category10") turned out to read as barely-distinguishable-from-just-
// blue-and-orange in practice (several of its colors are low-contrast
// grays/browns that don't stand out against the line chart's overlap).
function plotableSpecies() {
  return speciesNames.filter((n) => n !== "ge");
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

// Inverts the solver's own (black-box) ge -> T conversion for whatever
// composition is *currently* in the state buffer, by bisecting on ge
// until temperature() reports targetT -- rather than re-deriving or
// guessing the solver's actual mean-molecular-weight/gamma(T) convention
// in JS (risking a subtly wrong duplicate of logic we don't want to
// maintain twice). ge -> T is monotonic (more thermal energy per unit
// mass means higher temperature, for any fixed composition) even though
// it isn't simply linear -- gamma itself varies with T for H2-bearing
// gas (rovibrational degrees of freedom activating), which is exactly
// why a closed-form inversion isn't a one-liner and bisection is the
// pragmatic choice here. temperature()'s cache is left correctly
// reflecting the returned ge (the last bisection step already wrote and
// evaluated it), so callers don't need a separate refresh after this.
function geForTemperature(targetT, tolerance = 1e-8, maxIter = 50) {
  const ptr = statePtr() >> 3;
  const geIdx = idx.ge;

  function TAtGe(ge) {
    mod.HEAPF64[ptr + geIdx] = ge;
    rhsPtr(); // runs calculate_rhs, which refreshes the cached T-from-ge conversion as a side effect
    return temperature();
  }

  // Bracket around the naive monatomic-hydrogen estimate, wide enough
  // (2 decades either side) to comfortably contain the true root for
  // any of these networks' actual mu (up to ~4x, pure He) and gamma
  // (5/3 down to 7/5) range -- widened further below if that guess
  // somehow isn't enough.
  let lo = 0.01 * 1.5 * KB * targetT / MH;
  let hi = 100 * 1.5 * KB * targetT / MH;
  let TLo = TAtGe(lo), THi = TAtGe(hi);
  for (let guard = 0; TLo > targetT && guard < 20; guard++) { lo *= 0.1; TLo = TAtGe(lo); }
  for (let guard = 0; THi < targetT && guard < 20; guard++) { hi *= 10; THi = TAtGe(hi); }

  let mid = 0.5 * (lo + hi);
  for (let i = 0; i < maxIter; i++) {
    mid = 0.5 * (lo + hi);
    const Tmid = TAtGe(mid);
    if (Math.abs(Tmid - targetT) < tolerance * targetT) break;
    if (Tmid < targetT) lo = mid; else hi = mid;
  }
  return mid;
}

function setIcs(nH, T, fractions) {
  const ptr = statePtr() >> 3;
  // ge depends on every other species (mean molecular weight, and for
  // H2-bearing gas, temperature-dependent gamma too) -- set everything
  // else first, then invert for the ge that actually gives T at *this*
  // composition, rather than assuming pure monatomic hydrogen the way a
  // direct ge = 1.5*k*T/m_H formula would.
  for (const name of speciesNames) {
    if (name === "ge") continue;
    const frac = fractions[name] !== undefined ? fractions[name] : 0.0;
    mod.HEAPF64[ptr + idx[name]] = nH * frac;
  }
  geForTemperature(T);
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

// Right after a strong shock fires, the post-jump gas can sit in a
// regime where radiative cooling and (density-cubed) three-body H2
// formation are many orders of magnitude faster than the free-fall
// timescale that sizes an ordinary step -- BE_chem_solve still
// integrates all of that correctly in one big step (it subcycles
// internally), but only the two endpoints get reported, hiding a real
// spike-then-crash-then-partial-H2-reformation transient entirely
// between two plotted points.
//
// SHOCK_COOLING_SAFETY * coolingTime() (the same instantaneous
// ge/|dge/dt| estimate runConstantDensity's own adaptive dt already
// uses), evaluated once right after the jump, estimates *when the
// refined display should start* -- not a step size to keep re-deriving.
// Re-deriving it every substep was tried first and doesn't work here:
// this cooling curve is so steep near the post-shock temperature (order-
// of-magnitude changes in the derivative itself within a single
// estimated "cooling time") that a locally re-estimated step size keeps
// shrinking out from under itself, chasing a moving target instead of
// converging. Sampling SHOCK_DISPLAY_POINTS times, log-spaced from that
// one starting estimate up to this step's ordinary dt, sidesteps that
// entirely: BE_chem_solve already integrates an arbitrarily large dt
// correctly in one call (that's the whole reason the transient was
// hidden in the first place), so each checkpoint is just as reliable
// however far apart they are -- this only asks for more of them.
const SHOCK_COOLING_SAFETY = 0.05;
const SHOCK_DISPLAY_POINTS = 40;

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

  // Ordinary free-fall compression (density + adiabatic temperature
  // change) for a step of duration `subDt`, then chemistry advanced by
  // that same `subDt` -- factored out so both the outer per-step loop
  // below and the finer post-shock refinement burst apply exactly the
  // same physics per unit time, just at different step sizes. Returns
  // whether step() converged; on success it also pushes one history
  // point (so the refinement burst below shows up as several distinct
  // points, not one).
  function freefallStep(subDt) {
    const rho = nCurrent * MH;
    const rhoNew = Math.pow(Math.pow(rho, -0.5) - Math.sqrt(32 * G_GRAV / (3 * Math.PI)) * subDt, -2);
    const densityRatio = rhoNew / rho;
    const ptr = statePtr() >> 3;
    const gammaAd = thermodynamicGamma(getScalar());
    const tempRatio = 1 + (gammaAd - 1) * (densityRatio - 1);
    for (const name of speciesNames) if (name !== "ge") mod.HEAPF64[ptr + idx[name]] *= densityRatio;
    mod.HEAPF64[ptr + idx.ge] *= tempRatio;
    if (!step(subDt, 200, 1e-5)) return false;
    t += subDt;
    const s = getScalar();
    nCurrent = 0;
    for (const name of speciesNames) if (name !== "ge" && name !== "de") nCurrent += s[name];
    nHist.push(nCurrent); THist.push(temperature()); ionHist.push(ionizedFraction(s)); h2Hist.push(h2Fraction(s));
    tHist.push(t); dtHist.push(subDt); sHist.push(s);
    return true;
  }

  for (let i = 0; i < maxSteps; i++) {
    if (nCurrent >= nTarget) break;
    const rho = nCurrent * MH;
    const tFf = Math.sqrt(3 * Math.PI / (32 * G_GRAV * rho));
    const dt = safetyFactor * tFf;
    const rhoNewOrdinary = Math.pow(Math.pow(rho, -0.5) - Math.sqrt(32 * G_GRAV / (3 * Math.PI)) * dt, -2);

    // Would *this* step's ordinary free-fall compression carry us across
    // nShock? A shock is a genuine mathematical discontinuity -- that's
    // what the RH jump conditions describe -- so applying it as an
    // instantaneous jump (not blended into one ordinary compression
    // step) is the physically honest way to represent one here.
    if (shocked || machShock <= 1 || nCurrent * (rhoNewOrdinary / rho) < nShock) {
      if (!freefallStep(dt)) break;
      continue;
    }

    const ptr = statePtr() >> 3;
    const gammaAd = thermodynamicGamma(getScalar());
    const { rhoRatio, TRatio } = shockJumpFactors(gammaAd, machShock);
    for (const name of speciesNames) if (name !== "ge") mod.HEAPF64[ptr + idx[name]] *= rhoRatio;
    mod.HEAPF64[ptr + idx.ge] *= TRatio;
    nCurrent *= rhoRatio;

    // The jump itself is instantaneous -- record the state right after
    // it, before any chemistry has had time to respond at all, as its
    // own point (same t as just before the shock, genuinely zero
    // elapsed time, but a new density -- so it plots as a distinct point
    // showing the raw post-jump temperature this run would otherwise
    // never display). rhsPtr() refreshes temperature()'s cache, which
    // the jump's direct buffer writes don't trigger themselves.
    rhsPtr();
    const sJump = getScalar();
    nHist.push(nCurrent); THist.push(temperature()); ionHist.push(ionizedFraction(sJump)); h2Hist.push(h2Fraction(sJump));
    tHist.push(t); dtHist.push(0); sHist.push(sJump);

    // Then resolve however it relaxes from there, log-spaced from the
    // estimated start of that relaxation up to this step's ordinary dt
    // (see the constants' comment above for why log-spaced-from-one-
    // estimate beats re-deriving a step size every substep here).
    // `firstOk`/`allOk` mirror the non-shock branch's own convergence
    // handling: the shock only counts as having "happened" if its first
    // post-jump step actually converged, and any later failure still
    // halts the whole run, same as everywhere else in this file.
    const tStart = Math.min(Math.max(SHOCK_COOLING_SAFETY * coolingTime(dt), dt * 1e-8), dt);
    const decades = Math.max(Math.log10(dt / tStart), 0);
    let elapsed = 0, firstOk = null, allOk = true;
    for (let k = 0; k < SHOCK_DISPLAY_POINTS; k++) {
      const frac = (k + 1) / SHOCK_DISPLAY_POINTS;
      const target = tStart * Math.pow(10, decades * frac);
      const subDt = Math.min(target, dt) - elapsed;
      if (subDt <= 0) continue;
      const ok = freefallStep(subDt);
      if (firstOk === null) firstOk = ok;
      if (!ok) { allOk = false; break; }
      elapsed += subDt;
    }
    if (firstOk) { shocked = true; shockApplied = true; }
    if (!allOk) break;
  }
  return {
    x: nHist, t: tHist, dt: dtHist, T: THist, ion: ionHist, h2: h2Hist, s: sHist, xKey: "density",
    shockTriggered: shockApplied, nShock,
  };
}

const FIELD_TITLE = { T: "T (K)", ge: "ε (erg/g)" };

// Elapsed time at an arbitrary density `xVal`, linearly interpolated
// between whichever two adjacent rows bracket it (rows are sorted
// ascending by x -- density strictly increases over a free-fall run).
// Used only for the zoom brush's time-span readout below, where a
// label needs *a* reasonable time for a continuous, dragged density
// value, not the precision runFreefall's own bisection-grade care goes
// into elsewhere -- linear interpolation between adjacent steps is
// plenty for that.
function interpolateT(rows, xVal) {
  if (rows.length === 0) return null;
  if (xVal <= rows[0].x) return rows[0].t;
  if (xVal >= rows[rows.length - 1].x) return rows[rows.length - 1].t;
  let lo = 0, hi = rows.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (rows[mid].x <= xVal) lo = mid; else hi = mid;
  }
  const a = rows[lo], b = rows[hi];
  if (b.x === a.x) return a.t;
  return a.t + ((xVal - a.x) / (b.x - a.x)) * (b.t - a.t);
}

// Reads the temperature-vs-x panel's own "brush" selection (see
// metricPanel()/runViewSpec() above) and shows what elapsed *time* that
// dragged range corresponds to -- in free-fall mode the brush only ever
// operates in density, so without this there'd be no way to tell how
// much real time a given zoomed-in stretch actually spans (in cool mode
// the brush already *is* in time, but this still works: interpolateT()
// against x=t is just the identity).
function updateZoomTimespan(value, rows) {
  const el = document.getElementById("zoom-timespan");
  if (!el) return;
  const xRange = value && value.x;
  if (!xRange || xRange.length < 2) { el.textContent = ""; return; }
  const lo = Math.min(xRange[0], xRange[1]), hi = Math.max(xRange[0], xRange[1]);
  const tLo = interpolateT(rows, lo), tHi = interpolateT(rows, hi);
  el.textContent = `Selected range: ${formatTimeAuto(tLo)} to ${formatTimeAuto(tHi)} `
    + `(Δt ≈ ${formatTimeAuto(tHi - tLo)}).`;
}

// -- One unified spec for the whole current-run view --------------------
// Everything below builds ONE composed Vega-Lite spec (Temperature/
// density panels, the ionization/H2 chart, the species chart), all
// sharing one crosshair and each zoom pair's own brush -- replacing what
// used to be several independently-`vegaEmbed()`-ed charts kept in sync
// by hand-rolled JS (view.addSignalListener() + view.data().runAsync()
// puppeting one view from another's events). That worked, but Vega-Lite
// already has a real, documented feature for exactly this ("Multi-View
// Displays": `resolve: {selection: {name: "global"}}` shares one
// selection param across every sibling view in a composition) -- once
// everything is *one* spec, there's no second view left to bridge by
// hand, and the whole class of "which of two independently-resolving
// vegaEmbed() promises won a race" bugs (see NOTES.md) stops being
// possible, not just handled.

const PANEL_WIDTH = 620, OVERVIEW_HEIGHT = 150, DETAIL_HEIGHT = 170;
// Every panel in this composition is a single, standalone instance now --
// no more paired "overview" (never rescales) + "zoomed" (bound to the
// overview's own brush) copies of the same metric. Zooming instead comes
// from *one* panel (the temperature-vs-x chart, see runViewSpec() below)
// declaring the brush, while every *other* panel binds its own x-domain
// to that one selection -- so dragging a range rescales every other
// panel sharing that axis in place, and there's nothing left to
// duplicate. DETAIL_HEIGHT is kept only for the ion/species panels
// (unrelated to zooming, just their own slightly taller size).

function crosshairColor() {
  return isDarkMode() ? "#ffd54f" : "#c77700";
}

// The shared crosshair: a `rule` mark filtered by the one "hover"
// selection every panel below either declares (the first one built) or
// merely references (via the top-level `resolve` -- see runViewSpec()).
// Filtering by `i` (each row's own step index), not by whatever field
// this panel's x-axis happens to encode, is what lets a density-axis
// panel and a time-axis panel share the exact same selection at all --
// every panel's data already carries `i`, so this works regardless of
// what's actually plotted, and it's immune to the one real edge case
// that ruled out using `x`/`t` directly: runFreefall's shock-refinement
// can give two adjacent rows the *same* elapsed time (the instantaneous
// post-jump point), which would make `t` ambiguous as a join key -- `i`
// never is.
// `xDomainFromBrush`, when given, must be threaded through to *every*
// layer in a panel that has it, not just the one visible line -- found
// directly (an isolated repro first, then confirmed in this file's own
// build) while chasing a real "zooming doesn't work" report: Vega-Lite's
// default *shared* scale across a layered view's own sibling layers
// means that if only the main line layer gets an explicit
// `domain: {param: ..., field: ...}` and this file's crosshair/hover-
// capture layers don't, the merged domain silently becomes the *union*
// of the narrow brushed range and the other layers' own full-data
// range -- i.e. the full range wins and the panel never visibly
// rescales at all, even though the brush param itself fires correctly.
// So every layer-building helper below takes the same
// `xDomainFromBrush` a panel was given and applies the identical
// `domain` to its own x scale, so the shared-scale merge has nothing to
// disagree about.
function domainScale(xDomainFromBrush, xField) {
  return xDomainFromBrush ? { domain: { param: xDomainFromBrush, field: xField } } : {};
}

function crosshairLayer(data, xField, xDomainFromBrush) {
  return {
    data: { values: data },
    transform: [{ filter: { param: "hover", empty: false } }],
    mark: { type: "rule", strokeDash: [4, 3], opacity: 0.85, color: crosshairColor() },
    encoding: { x: { field: xField, type: "quantitative", scale: domainScale(xDomainFromBrush, xField) } },
  };
}

// nearest-by-x-pixel-distance only (`encodings: ["x"]`) so this behaves
// like a vertical crosshair, not a 2D nearest-neighbor search; `line`
// marks don't support `nearest` directly (confirmed directly -- it
// warns and does nothing without a separate point-mark capture layer,
// which is why this is its own tiny invisible layer, not part of the
// visible line).
function hoverCaptureLayer(data, xField, yField, xDomainFromBrush) {
  return {
    data: { values: data },
    mark: { type: "point", opacity: 0 },
    encoding: {
      x: { field: xField, type: "quantitative", scale: domainScale(xDomainFromBrush, xField) },
      y: { field: yField, type: "quantitative" },
    },
    params: [{
      name: "hover",
      select: { type: "point", on: "pointermove", nearest: true, encodings: ["x"], fields: ["i"], clear: "pointerout" },
    }],
  };
}

// One "metric vs x" panel, where x is whichever quantity this mode's
// panels all share (density in free-fall mode, time in cool mode --
// `xKind`, axis scale/formatting only). `brushName`, when given,
// attaches this panel's own zoom-brush (scoped to just this one layer --
// see the note on that in NOTES.md, a selection declared at a layered
// view's outer level gets incorrectly projected onto every layer in it,
// including `extra` layers like the shock-event rule that have no x
// field at all); `xDomainFromBrush`, when given, binds this panel's
// x-domain to *another* panel's brush by name, so it rescales as that
// brush is dragged without ever holding one itself.
function metricPanel({ data, xField, yField, xKind, yTitle, tooltip, extra, brushName, xDomainFromBrush, withHoverParam, title }) {
  const xAxis = {
    // No external KaTeX-rendered label div for this one shared axis
    // title the way single-chart panels used to have -- with several
    // differently-labeled panels now inside one combined spec, a real
    // (plain, not LaTeX) Vega-Lite axis title per panel is simpler than
    // trying to externally position several correctly. Genuinely a
    // deliberate simplification, not just a workaround -- these are all
    // short unit-bearing strings anyway ("T (K)", "n (cm⁻³)"), which
    // read fine without full math typesetting.
    title: xKind === "time" ? "t (s)" : "n (cm⁻³)",
    titleFontSize: 10, labelOverlap: "greedy",
    labelAngle: xKind === "time" ? -40 : 0,
    labelExpr: xKind === "time" ? TIME_LABEL_EXPR : undefined,
  };
  // Time (unlike density) can legitimately be exactly 0 now that the
  // initial condition itself is plotted -- symlog (linear near zero,
  // log further out) shows that point instead of silently dropping it
  // the way a pure log scale would.
  const xScale = {
    type: xKind === "time" ? "symlog" : "log",
    ...(xDomainFromBrush ? { domain: { param: xDomainFromBrush, field: xField } } : {}),
  };
  const mainLayer = {
    data: { values: data },
    mark: "line",
    encoding: {
      x: { field: xField, type: "quantitative", scale: xScale, axis: xAxis },
      y: { field: yField, type: "quantitative", scale: { type: "log" }, axis: { title: yTitle, titleFontSize: 10 } },
      tooltip,
    },
    ...(brushName ? { params: [{ name: brushName, select: { type: "interval", encodings: ["x"] } }] } : {}),
  };
  return {
    title: title ? { text: title, fontSize: 12, fontWeight: "bold", anchor: "start", offset: 6 } : undefined,
    width: PANEL_WIDTH, height: OVERVIEW_HEIGHT,
    layer: [
      ...(extra || []), mainLayer,
      ...(withHoverParam ? [hoverCaptureLayer(data, xField, yField, xDomainFromBrush)] : []),
      crosshairLayer(data, xField, xDomainFromBrush),
    ],
  };
}

// Free-fall mode's density-vs-time panel -- deliberately x=density,
// y=time (the reverse of how this looked before) specifically so its
// x-axis is the exact same field/scale as the temperature-vs-density
// panel below it, letting that panel's brush rescale this one too (see
// runViewSpec()). Density is log (never legitimately zero); time is
// symlog, same reasoning as everywhere else time appears (the initial
// condition's t=0 point is real and worth showing, not silently
// dropped). Never itself holds a brush -- it's purely a *follower* of
// the temperature-vs-density chart's, same as ionPanel/speciesPanel.
function densityTimePanel(rows, xDomainFromBrush) {
  const xScale = {
    type: "log",
    ...(xDomainFromBrush ? { domain: { param: xDomainFromBrush, field: "x" } } : {}),
  };
  const tooltip = [
    { field: "i", title: "step", type: "quantitative" },
    { field: "x", title: "n (cm⁻³)", type: "quantitative", format: ".3~g" },
    { field: "tHuman", title: "t", type: "nominal" },
    { field: "t", title: "t (s)", type: "quantitative", format: ".3~g" },
  ];
  const mainLayer = {
    data: { values: rows },
    mark: "line",
    encoding: {
      x: { field: "x", type: "quantitative", scale: xScale, axis: { title: "n (cm⁻³)", titleFontSize: 10, labelOverlap: "greedy" } },
      y: {
        field: "t", type: "quantitative", scale: { type: "symlog" },
        axis: { title: "t (s)", titleFontSize: 10, labelAngle: 0, labelExpr: TIME_LABEL_EXPR },
      },
      tooltip,
    },
  };
  return {
    title: { text: "density vs. time", fontSize: 12, fontWeight: "bold", anchor: "start", offset: 6 },
    width: PANEL_WIDTH, height: OVERVIEW_HEIGHT,
    layer: [mainLayer, hoverCaptureLayer(rows, "x", "t", xDomainFromBrush), crosshairLayer(rows, "x", xDomainFromBrush)],
  };
}

// Ionized fraction and molecular (H2) fraction together, long-format
// (one row per (step, quantity) pair, `quantity` already holding the
// human-readable legend label) -- see ION_H2_LABELS/ION_H2_COLORS above.
// h2 rows are simply absent for a network with no H2 species (rather
// than plotting a bogus flat line), same graceful-degradation as the
// status line's H2/H_tot readout. Legend click isolates one series
// (Vega-Lite's own "bind: legend" selection, an officially documented
// recipe -- not a bespoke mechanism) instead of a fixed on/off toggle,
// since there are only ever one or two series here anyway.
function ionPanel(xKind, rows, width, xDomainFromBrush) {
  const n = rows.length;
  const pointSize = Math.max(4, Math.min(30, 2000 / Math.max(n, 1)));
  const palette = isDarkMode() ? ION_H2_COLORS.dark : ION_H2_COLORS.light;
  const present = new Set(rows.map((r) => r.quantity));
  const domain = [ION_H2_LABELS.ion, ION_H2_LABELS.h2].filter((label) => present.has(label));
  const range = domain.map((label) => (label === ION_H2_LABELS.ion ? palette.ion : palette.h2));
  return {
    title: { text: "Ionized / H₂ fraction", fontSize: 12, fontWeight: "bold", anchor: "start", offset: 6 },
    width, height: DETAIL_HEIGHT,
    layer: [{
      data: { values: rows },
      mark: { type: "line", point: { filled: true, size: pointSize, opacity: 0.9 } },
      encoding: {
        x: {
          field: "x", type: "quantitative",
          scale: {
            type: xKind === "time" ? "symlog" : "log",
            ...(xDomainFromBrush ? { domain: { param: xDomainFromBrush, field: "x" } } : {}),
          },
          axis: {
            title: xKind === "time" ? "t (s)" : "n (cm⁻³)", titleFontSize: 10, labelOverlap: "greedy",
            labelAngle: xKind === "time" ? -40 : 0,
            labelExpr: xKind === "time" ? TIME_LABEL_EXPR : undefined,
          },
        },
        y: { field: "value", type: "quantitative", scale: { type: "log" }, axis: { title: "fraction", titleFontSize: 10 } },
        color: { field: "quantity", type: "nominal", scale: { domain, range }, legend: { title: null, orient: "bottom", direction: "horizontal" } },
        opacity: { condition: { param: "ionToggle", value: 1 }, value: 0.15 },
        tooltip: [
          { field: "quantity", title: "quantity", type: "nominal" },
          { field: "i", title: "step", type: "quantitative" },
          { field: "x", title: xKind === "time" ? "t (s)" : "n (cm⁻³)", type: "quantitative", format: ".3~g" },
          { field: "tHuman", title: "t", type: "nominal" },
          { field: "dt", title: "step Δt (s)", type: "quantitative", format: ".3~g" },
          { field: "value", title: "fraction", type: "quantitative", format: ".4~g" },
        ],
      },
      params: [{ name: "ionToggle", select: { type: "point", fields: ["quantity"] }, bind: "legend" }],
    }, crosshairLayer(rows, "x", xDomainFromBrush)],
  };
}

// Every plotable species' mass fraction, always all of them (no
// checkbox subset -- "they should all be included in the document").
// Click a legend entry to isolate it (dims every other series via the
// `opacity` condition below); click it again to show all -- Vega-Lite's
// own "Legends as Interactive Filters" recipe (`bind: "legend"` on a
// point selection), the same mechanism the ionization chart's legend
// above uses, just with more than two entries. Always mass fraction
// (not a density/mass-fraction toggle) -- one fewer control, and mass
// fraction is the more physically comparable quantity across species
// spanning very different absolute number densities anyway.
function speciesPanel(xKind, rows, width, xDomainFromBrush) {
  const n = rows.length;
  const pointSize = Math.max(4, Math.min(30, 2000 / Math.max(n, 1)));
  const domain = plotableSpecies().filter((name) => speciesMassAmu(name) !== undefined);
  return {
    title: { text: "Species mass fraction (click a legend entry to isolate it)", fontSize: 12, fontWeight: "bold", anchor: "start", offset: 6 },
    width, height: DETAIL_HEIGHT + 20,
    layer: [{
      data: { values: rows },
      mark: { type: "line", point: { filled: true, size: pointSize, opacity: 0.9 } },
      encoding: {
        x: {
          field: "x", type: "quantitative",
          scale: {
            type: xKind === "time" ? "symlog" : "log",
            ...(xDomainFromBrush ? { domain: { param: xDomainFromBrush, field: "x" } } : {}),
          },
          axis: {
            title: xKind === "time" ? "t (s)" : "n (cm⁻³)", titleFontSize: 10, labelOverlap: "greedy",
            labelAngle: xKind === "time" ? -40 : 0,
            labelExpr: xKind === "time" ? TIME_LABEL_EXPR : undefined,
          },
        },
        y: { field: "value", type: "quantitative", scale: { type: "log" }, axis: { title: "X_i (mass frac.)", titleFontSize: 10 } },
        // "tableau10" -- a standard, well-vetted qualitative palette
        // (not a hand-picked array), chosen specifically because it
        // reads as more genuinely *distinct* per swatch than the old
        // per-species array (d3's older "category10") did in practice.
        // Independent of the ionization chart's own color scale/legend
        // (different field, and belt-and-suspenders explicit
        // resolve.legend/scale in runViewSpec() besides) -- never meant
        // to share one.
        color: { field: "species", type: "nominal", scale: { domain, scheme: "tableau10" }, legend: { title: null, orient: "bottom", direction: "horizontal", columns: 0 } },
        opacity: { condition: { param: "speciesToggle", value: 1 }, value: 0.12 },
        tooltip: [
          { field: "species", title: "species", type: "nominal" },
          { field: "i", title: "step", type: "quantitative" },
          { field: "x", title: xKind === "time" ? "t (s)" : "n (cm⁻³)", type: "quantitative", format: ".3~g" },
          { field: "tHuman", title: "t", type: "nominal" },
          { field: "dt", title: "step Δt (s)", type: "quantitative", format: ".3~g" },
          { field: "value", title: "X_i (mass frac.)", type: "quantitative", format: ".4~g" },
        ],
      },
      params: [{ name: "speciesToggle", select: { type: "point", fields: ["species"] }, bind: "legend" }],
    }, crosshairLayer(rows, "x", xDomainFromBrush)],
  };
}

// The whole current-run view: a single vertical stack (density vs. time,
// only in free-fall mode; temperature/thermal-energy vs. x; ionization/
// H2 fraction; species mass fraction), all sharing one crosshair and one
// zoom-brush. `resolve.selection.hover: "global"` is what makes each
// hover-capturing panel's own "hover" param apply across every sibling
// panel in this whole composition -- confirmed directly in an isolated
// repro before relying on it here (see NOTES.md).
//
// Zooming: exactly one panel (temperature-vs-x, `tempOverview` below)
// declares the actual brush; every *other* panel binds its own x-domain
// to that one selection by name (`xDomainFromBrush`), so dragging a
// range there rescales every other panel sharing that axis in place --
// the brush-holding panel itself is deliberately excluded (it stays at
// the full range, showing the selection box, the standard Vega-Lite
// "overview" role), not a separate "zoomed" copy of each metric the way
// this looked before landing on this design (see NOTES.md).
//
// The composition is a single flat top-level `vconcat` -- every panel a
// direct item, never wrapped in an intermediate row spec -- both because
// that's the vertical single-chart-per-row layout asked for, and because
// a real Vega bug (not this file's own logic; reproduced in isolation
// with plain placeholder data, unrelated to anything in this app) throws
// runtime TypeErrors on hover as soon as an `interval` (brush) selection
// and a globally-resolved `point` (hover) selection coexist anywhere
// beneath *two or more* levels of concat nesting. A flat `vconcat` is
// only one level, so this is safe -- confirmed directly, not assumed
// (see NOTES.md).
function runViewSpec({ mode, rows, tempField, tempExtra, ionRows, speciesRows }) {
  const tempTitle = (FIELD_TITLE[tempField] || tempField).replace(/\s*\(.*\)/, "");
  const xKindOfMode = mode === "freefall" ? "density" : "time";
  const tempTooltip = [
    { field: "i", title: "step", type: "quantitative" },
    { field: "x", title: xKindOfMode === "time" ? "t (s)" : "n (cm⁻³)", type: "quantitative", format: ".3~g" },
    { field: "tHuman", title: "t", type: "nominal" },
    { field: "dt", title: "step Δt (s)", type: "quantitative", format: ".3~g" },
    { field: tempField, title: FIELD_TITLE[tempField] || tempField, type: "quantitative", format: ".4~g" },
  ];

  // The one panel that actually holds the brush -- see the note above.
  const tempOverview = metricPanel({
    data: rows, xField: "x", yField: tempField, xKind: xKindOfMode, yTitle: FIELD_TITLE[tempField] || tempField,
    tooltip: tempTooltip, extra: tempExtra,
    brushName: "brush", withHoverParam: true,
    title: `${tempTitle} vs. ${xKindOfMode}`,
  });

  const items = [];
  // Density-vs-time only makes sense in free-fall mode -- density is
  // held constant in cool mode, so this would just be a flat, useless
  // line there (matches the previous design's same free-fall-only
  // gating for this chart).
  if (mode === "freefall") items.push(densityTimePanel(rows, "brush"));
  items.push(tempOverview);
  if (ionRows.length) items.push(ionPanel(xKindOfMode, ionRows, PANEL_WIDTH, "brush"));
  if (speciesRows.length) items.push(speciesPanel(xKindOfMode, speciesRows, PANEL_WIDTH, "brush"));

  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    background: null,
    config: vlConfig(),
    // legend/scale independence is belt-and-suspenders here (the
    // ionization and species charts already use different color fields
    // -- "quantity" vs "species" -- so Vega-Lite wouldn't merge their
    // legends by default regardless), made explicit because it was
    // asked for directly: the two must never share one legend.
    resolve: { selection: { hover: "global" }, legend: { color: "independent" }, scale: { color: "independent" } },
    vconcat: items,
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

// Two rapid, separate scheduleRedraw() calls can each land in their
// *own* animation frame rather than being coalesced into one
// (coalescing only helps when both happen before the same pending
// frame fires) -- meaning two overlapping redraw() calls, each starting
// its own async vegaEmbed("#chart-run", ...). Whichever call's promise
// happens to resolve *last* would otherwise win, overwriting `runView`
// and wiring the brush-timespan listener onto a view whose SVG has
// already been replaced by the other, newer call -- a real bug this
// surfaced once (see NOTES.md) back when there were two separately-
// embedded charts to keep in sync by hand; folding everything into one
// composed spec removed the *need* for most of that cross-view JS, but
// the same stale-resolution risk still applies to this one remaining
// listener, so the same guard stays. Every redraw() captures its own
// generation number and the callback that touches `runView` checks
// it's still current before doing anything, discarding a stale,
// superseded resolution instead of acting on it.
let redrawGeneration = 0;

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
  const myGeneration = ++redrawGeneration; // see the comment on redrawGeneration above
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
    x, t: result.t[i], T: result.T[i], ge: result.s[i].ge, ion: result.ion[i],
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
  const ionRows = [];
  for (let i = 0; i < result.x.length; i++) {
    const base = { x: result.x[i], i, dt: result.dt[i], tHuman: formatTimeAuto(result.t[i]) };
    ionRows.push({ ...base, quantity: ION_H2_LABELS.ion, value: result.ion[i] });
    const h2v = result.h2[i];
    if (h2v !== null && h2v !== undefined && h2v > 0) {
      ionRows.push({ ...base, quantity: ION_H2_LABELS.h2, value: h2v });
    }
  }
  // Always every plotable species with a known mass, always mass
  // fraction -- no checkbox subset, no density/mass-fraction toggle
  // (removed; see NOTES.md). Species without a known mass (an
  // unrecognized future network) simply can't appear in a *fraction*
  // at all, same reasoning the old mass-fraction mode already had.
  const speciesRows = [];
  for (let i = 0; i < result.x.length; i++) {
    const tHuman = formatTimeAuto(result.t[i]);
    const denom = totalMassAmu(result.s[i]);
    for (const name of plotableSpecies()) {
      const massAmu = speciesMassAmu(name);
      if (massAmu === undefined) continue;
      const v = (result.s[i][name] * massAmu) / denom;
      if (v > 0) speciesRows.push({ x: result.x[i], i, dt: result.dt[i], tHuman, species: name, value: v });
    }
  }

  const spec = runViewSpec({ mode: currentMode, rows, tempField, tempExtra, ionRows, speciesRows });
  vegaEmbed("#chart-run", spec, { actions: false, renderer: "svg" }).then((res) => {
    if (myGeneration !== redrawGeneration) return; // a newer redraw() already superseded this one
    runView = res.view;
    res.view.addSignalListener("brush", (name, value) => updateZoomTimespan(value, rows));
    updateZoomTimespan(null, rows); // nothing selected yet on a fresh embed
  });

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

  lastResult = result;
  document.getElementById("download-csv").disabled = false;
}

function csvCell(v) {
  if (v === null || v === undefined) return "";
  // Plain String(), not one of the display-only .toFixed()/.toExponential()
  // formatters used elsewhere -- this is a data export, so it gets full
  // double precision (JS's own shortest round-trip representation), not
  // whatever a handful of significant figures a tooltip needs.
  const s = String(v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

// The full step-by-step history of the current mode's most recent run
// (redraw()'s own `result`, cached in `lastResult`) -- every species'
// number density plus the derived quantities already shown on the
// charts (T, gamma, ionized/H2 fraction), one row per solver step.
// Deliberately just the one run: a sweep is several of these overlaid,
// not a single result, and exporting all of them together would need a
// different shape entirely -- out of scope here, this is "the results
// on screen right now".
function resultsToCsv(result) {
  const species = plotableSpecies(); // excludes "ge" -- already its own column below
  const header = [
    "step", "x", "x_kind", "t_s", "t_human", "dt_s",
    "T_K", "ge_erg_per_g", "gamma", "ion_frac", "h2_frac",
    ...species,
  ];
  const lines = [header.map(csvCell).join(",")];
  for (let i = 0; i < result.x.length; i++) {
    const s = result.s[i];
    const row = [
      i, result.x[i], result.xKey, result.t[i], formatTimeAuto(result.t[i]), result.dt[i],
      result.T[i], s.ge, thermodynamicGamma(s), result.ion[i], result.h2[i],
      ...species.map((name) => s[name]),
    ];
    lines.push(row.map(csvCell).join(","));
  }
  return lines.join("\n");
}

function downloadResultsCsv() {
  if (!lastResult) return;
  const csv = resultsToCsv(lastResult);
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${pageTitle.replace(/\s+/g, "_")}_${currentMode}_results.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
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

// The actual "Temperature vs. ..." / "Thermal energy vs. ..." panel
// title text is computed fresh from temperatureDisplayMode inside
// runViewSpec() on every redraw -- no separate DOM title element to
// keep in sync here anymore, now that it's baked into the Vega-Lite
// spec itself.
function setTemperatureDisplayMode(mode) {
  temperatureDisplayMode = mode;
  document.getElementById("T-mode-T").classList.toggle("active", mode === "T");
  document.getElementById("T-mode-ge").classList.toggle("active", mode === "ge");
  scheduleRedraw();
}

function initPage(config) {
  document.getElementById("page-title").textContent = config.title;
  document.getElementById("T").value = Math.log10(config.default_T);
  pageTitle = config.title;

  document.getElementById("download-csv").addEventListener("click", downloadResultsCsv);
  document.getElementById("mode-cool").addEventListener("click", () => setMode("cool"));
  document.getElementById("mode-freefall").addEventListener("click", () => setMode("freefall"));
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
    buildSweepParamOptions();
    updateSweepParamUI(); // sets up the "No sweep" default state (run-sweep button included) immediately, not just after the dropdown is touched
    document.getElementById("ic-preset").disabled = false;
    document.getElementById("status").textContent = "ready";
    redraw();
  });
}
