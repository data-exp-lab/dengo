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
let currentMode = "freefall"; // free-fall collapse is the primary case this widget targets -- see NOTES.md
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

function runConstantDensity(nH, T, fractions, logDtf, tolerance = 1e-5, safetyFactor = 0.1, maxSteps = 2000, forSweep = false) {
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
    const converged = step(dt, 200, tolerance);
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

// dρ/dt = FF_RATE_CONST * ρ^(3/2) is the ordinary free-fall compression
// law (its closed-form solution is what freefallStep()'s `rhoNew` line
// below is); `collapseFactor` scales this one rate constant, and only
// this one, everywhere it appears (both the actual compression *and*
// the free-fall *time* used to size the adaptive step) -- so a step
// still represents the same fractional density change regardless of
// collapseFactor (the scaling cancels out of `rate * dt` when `dt`
// itself is sized from the same scaled rate), it just represents more
// or less *real time* for the chemistry to act over, which is the
// entire point: collapsing faster than free-fall (collapseFactor > 1)
// gives chemistry less time per decade of density to respond; slower
// (< 1) gives it more -- a schematic stand-in for whatever isn't
// modeled here (rotation/magnetic/pressure support slowing a real
// collapse below free-fall, or additional infall/turbulence speeding
// one up), not a real dynamical mechanism in its own right.
const FF_RATE_CONST = Math.sqrt(32 * G_GRAV / (3 * Math.PI));

function runFreefall(nH, T, fractions, logNTarget, logNShock, machShock, collapseFactor = 1, tolerance = 1e-5, safetyFactor = 0.01, maxSteps = 10000) {
  setIcs(nH, T, fractions);
  const nTarget = Math.pow(10, logNTarget);
  const nShock = Math.pow(10, logNShock);
  const rate = FF_RATE_CONST * collapseFactor;
  let t = 0;
  const nHist = [], THist = [], ionHist = [], h2Hist = [], tHist = [], dtHist = [], sHist = [];
  // The initial condition itself is worth plotting, same reasoning as
  // runConstantDensity() -- and free-fall's x-axis is density, always
  // positive, so (unlike time) there's no log-scale concern in placing
  // it as the very first point. `nCurrent` here is defined exactly like
  // every later point's (freefallStep() below): the literal sum of
  // tracked species densities, *not* the dialed n_H,0. Those two aren't
  // the same number -- n_H,0 only pins hydrogen's own share; He (and
  // any other species) are independent per-species fractions with no
  // requirement that everything add up to n_H,0 -- so seeding nCurrent
  // from n_H,0 instead made the very first step look like a sudden
  // density *drop* (a bookkeeping level-shift, not a real equilibration
  // transient) before the "n increases monotonically" free-fall
  // physics ever got a chance to run.
  let nCurrent = 0;
  {
    // temperature()'s cache needs an explicit refresh here too -- see
    // the identical comment in runConstantDensity().
    rhsPtr();
    const s0 = getScalar();
    for (const name of speciesNames) if (name !== "ge" && name !== "de") nCurrent += s0[name];
    nHist.push(nCurrent); THist.push(temperature()); ionHist.push(ionizedFraction(s0)); h2Hist.push(h2Fraction(s0));
    tHist.push(0); dtHist.push(0); sHist.push(s0);
  }
  let shocked = nCurrent >= nShock; // already past it at t=0 -- don't fire mid-run
  let shockApplied = false; // only set once the post-shock step actually converges -- see below

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
    const rhoNew = Math.pow(Math.pow(rho, -0.5) - rate * subDt, -2);
    const densityRatio = rhoNew / rho;
    const ptr = statePtr() >> 3;
    const gammaAd = thermodynamicGamma(getScalar());
    const tempRatio = 1 + (gammaAd - 1) * (densityRatio - 1);
    for (const name of speciesNames) if (name !== "ge") mod.HEAPF64[ptr + idx[name]] *= densityRatio;
    mod.HEAPF64[ptr + idx.ge] *= tempRatio;
    if (!step(subDt, 200, tolerance)) return false;
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
    const tFf = 1 / (rate * Math.sqrt(rho));
    const dt = safetyFactor * tFf;
    const rhoNewOrdinary = Math.pow(Math.pow(rho, -0.5) - rate * dt, -2);

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
    // Not just decorative -- a finer step size (a smaller safetyFactor,
    // see the new "step size" slider) takes proportionally more steps
    // to cross the same density range, so it's now genuinely possible
    // to hit `maxSteps` before reaching `nTarget` with an ordinary
    // slider drag, not just a pathological setting. Surfaced in
    // updateRunSummary() rather than silently truncating the run with
    // no visible sign anything was cut short.
    reachedTarget: nCurrent >= nTarget,
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
// density panels, the ionization/H2 chart, the species chart), sharing
// one zoom-brush (a shared *crosshair* used to live here too, across
// every panel -- removed for now, a real measured performance
// regression, see NOTES.md and the note above metricPanel() below) --
// replacing what used to be several independently-`vegaEmbed()`-ed
// charts kept in sync by hand-rolled JS (view.addSignalListener() +
// view.data().runAsync() puppeting one view from another's events).
// That worked, but Vega-Lite already has a real, documented feature for
// exactly this ("Multi-View Displays": `resolve: {selection: {name:
// "global"}}` shares one selection param across every sibling view in a
// composition) -- once everything is *one* spec, there's no second view
// left to bridge by hand, and the whole class of "which of two
// independently-resolving vegaEmbed() promises won a race" bugs (see
// NOTES.md) stops being possible, not just handled.

// 620 (what this used to be) rendered a real chart ~705px wide once
// axis-label margins are added in -- comfortably more than the ~686px
// `.chart-box` actually has to give it at this page's *capped* body
// width (1100px, minus padding/sidebar/gap -- see style.css), on any
// window wide enough to hit that cap (roughly >=1158px, i.e. most
// windows people actually use, confirmed directly on a real 1920px-wide
// one). That's not the "narrow window" case `.chart-box`'s own
// `overflow-x: auto` exists for -- it was overflowing on an ordinary,
// plenty-wide screen. 580 (~700px rendered) fits with a real margin to
// spare, confirmed directly at 1920px width.
const PANEL_WIDTH = 580, OVERVIEW_HEIGHT = 150, DETAIL_HEIGHT = 170;
// One panel (the temperature-vs-x chart, see runViewSpec() below)
// declares the brush; every *other* panel binds its own x-domain to
// that one selection -- so dragging a range rescales every other panel
// sharing that axis in place, no separate "zoomed" copy needed for any
// of them, *except* temperature itself gets one anyway (directly below
// it, showing points): the panel holding the brush can't also be the
// one that rescales to it (it has to stay at the full range to show
// the selection box), so it's the one metric here that still needs an
// actual second, zoomed instance to see rescaled at all.
//
// The shared crosshair this composition used to have -- a globally-
// resolved "hover" point selection, every panel either declaring its
// own invisible nearest-point capture layer or consuming it via a
// filtered rule mark -- is gone (see NOTES.md): confirmed directly, a
// real, large, measured latency regression (~1.6s per mouse move vs.
// ~20ms on a blank page), traced to Vega's own dataflow scheduler not
// cleanly isolating "hover" pulses from the *other* (brush-driven)
// scale-domain recomputation happening in the same resolved
// composition, not something fixable by restructuring this file's own
// spec-building differently. A future, deliberately simpler mechanism
// (a rule that just tracks the pointer's raw x pixel position, no
// nearest-point lookup or cross-view resolution at all) is worth
// trying instead, but that's a different-enough approach that nothing
// here was worth keeping around unused for it.
// `zoomDomain`, when given, is a plain `[lo, hi]` array -- and applied
// by *filtering the data* to that range (see `zoomFilteredData()` below),
// not by overriding the x-scale's own `domain`. Both were tried; the
// scale-domain version is not just slow but confirmed directly to be
// outright broken: giving a `log`-typed x-scale an explicit literal
// `domain` -- with nothing else different about the spec at all, even a
// single fresh-embedded panel outside this file's own composition --
// makes Vega render it at close to *4x* its declared `width` (confirmed
// in an isolated repro: a plain 300px-wide single-panel spec rendered at
// 1357px purely from adding `scale.domain`, `nice: true` or not, same
// panel/data otherwise byte-identical). Root cause not fully chased down
// (the compiled Vega JSON's declared `width` was unaffected; whatever
// inflates it happens at Vega's own runtime layout step), but filtering
// the underlying data instead of touching the scale's domain sidesteps
// it entirely -- confirmed directly, back to the correct width -- and
// happens to also be simpler.
//
// Landing on a *filter*, not a live param binding, also fixes a real,
// separate, measured latency regression: binding a panel's x-domain
// live to another panel's brush *param* (the very first thing tried)
// cost ~700-900ms of extra Vega dataflow work *per signal update*, and
// a drag fires many of those, not just one -- one drag gesture took
// several seconds. Zooming is "commit once the drag settles" now: the
// brush param still lives only on the one anchor panel (for the drag
// rectangle itself, and to read its final value from), but every
// follower panel's data is refiltered and the chart rebuilt only after
// a short pause once dragging actually stops (see `embedChart()`/
// `redraw()`) -- confirmed directly this is dramatically faster than
// live updates during the drag itself.
function zoomFilteredData(data, xField, zoomDomain) {
  if (!zoomDomain) return data;
  const [lo, hi] = zoomDomain;
  return data.filter((d) => d[xField] >= lo && d[xField] <= hi);
}

function metricPanel({ data, xField, yField, xKind, yTitle, tooltip, extra, brushName, initialBrush, zoomDomain, showPoints, pointSize, title }) {
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
  const xScale = { type: xKind === "time" ? "symlog" : "log" };
  // Even the panels with no *visible* points (everything except the
  // zoomed one) still get real, invisible point geometry here -- not
  // just a bare `"line"` mark -- so their tooltip has something to
  // actually hit-test against. A bare line's tooltip only fires when
  // the cursor lands pixel-perfectly on its (1px) rendered stroke;
  // confirmed directly as a real regression once the old "nearest"
  // hover-capture layer (which used to double as a much more forgiving
  // tooltip target) was removed for the crosshair's own sake (see
  // NOTES.md) -- these invisible points restore *some* of that
  // forgiveness back, without needing any selection/param machinery.
  const mainLayer = {
    data: { values: zoomFilteredData(data, xField, zoomDomain) },
    mark: { type: "line", point: { filled: true, size: pointSize || (showPoints ? 50 : 200), opacity: showPoints ? 0.9 : 0 } },
    encoding: {
      x: { field: xField, type: "quantitative", scale: xScale, axis: xAxis },
      y: { field: yField, type: "quantitative", scale: { type: "log" }, axis: { title: yTitle, titleFontSize: 10 } },
      tooltip,
    },
    ...(brushName ? {
      params: [{
        name: brushName,
        select: { type: "interval", encodings: ["x"] },
        // Re-populates the drag rectangle to match the already-committed
        // zoom across a rebuild (redraw() resets zoomDomain to null for
        // a fresh run, but a rebuild triggered by the brush listener
        // itself should still show the box where the user left it).
        ...(initialBrush ? { value: { x: initialBrush } } : {}),
      }],
    } : {}),
  };
  return {
    title: title ? { text: title, fontSize: 12, fontWeight: "bold", anchor: "start", offset: 6 } : undefined,
    width: PANEL_WIDTH, height: showPoints ? DETAIL_HEIGHT : OVERVIEW_HEIGHT,
    layer: [...(extra || []), mainLayer],
  };
}

// Free-fall mode's density-vs-time panel -- x=density, y=*lookback*
// time (time remaining until this run's last step), not elapsed time.
// Plain elapsed time was flagged directly as "not that useful": free-
// fall time is heavily front-loaded (almost all of it elapses while
// density is still low and barely changing; the collapse through the
// remaining many decades of density is comparatively instantaneous),
// so elapsed-time-vs-density rises steeply over just the first couple
// of density decades and then goes flat for the rest of the run --
// which, for this project, means exactly the regime it cares most
// about (the H2-formation-heating/shock physics, all happening at high
// density) falls in the boring flat part. Counting backward from the
// end instead flips which part is flat: lookback time is large and
// slowly-varying at *low* density (almost the entire run is still
// ahead) and shrinks rapidly toward zero at *high* density (as the
// solver approaches its last step) -- moving the detail to line up
// with the high-density regime that's actually interesting, exactly
// the "lookback time" idea suggested directly, and confirmed by
// screenshot to read as a real improvement, not just a relabeling.
// Drops the very last row (lookback = 0 there, by definition) rather
// than plotting it, the same reasoning the old elapsed-time version
// dropped its t=0 row for: once that one zero is gone, this can stay
// plain `log` instead of `symlog`.
function densityTimePanel(rows, zoomDomain) {
  const tEnd = rows[rows.length - 1].t;
  const plotRows = zoomFilteredData(
    rows.map((r) => ({ ...r, lookback: tEnd - r.t, lookbackHuman: formatTimeAuto(tEnd - r.t) })).filter((r) => r.lookback > 0),
    "x", zoomDomain,
  );
  const xScale = { type: "log" };
  const tooltip = [
    { field: "i", title: "step", type: "quantitative" },
    { field: "x", title: "n (cm⁻³)", type: "quantitative", format: ".3~g" },
    { field: "lookbackHuman", title: "time to end", type: "nominal" },
    { field: "lookback", title: "time to end (s)", type: "quantitative", format: ".3~g" },
    { field: "tHuman", title: "elapsed t", type: "nominal" },
  ];
  const mainLayer = {
    data: { values: plotRows },
    // Invisible point geometry, not a bare line -- same reasoning as
    // metricPanel()'s own mainLayer: gives the tooltip something real
    // to hit-test against instead of a 1px stroke.
    mark: { type: "line", point: { filled: true, size: 200, opacity: 0 } },
    encoding: {
      x: { field: "x", type: "quantitative", scale: xScale, axis: { title: "n (cm⁻³)", titleFontSize: 10, labelOverlap: "greedy" } },
      y: {
        field: "lookback", type: "quantitative", scale: { type: "log" },
        axis: { title: "time to end (s)", titleFontSize: 10, labelAngle: 0, labelExpr: TIME_LABEL_EXPR },
      },
      tooltip,
    },
  };
  return {
    title: { text: "density vs. time-to-end", fontSize: 12, fontWeight: "bold", anchor: "start", offset: 6 },
    width: PANEL_WIDTH, height: OVERVIEW_HEIGHT,
    layer: [mainLayer],
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
function ionPanel(xKind, rows, width, zoomDomain) {
  const n = rows.length;
  const pointSize = Math.max(4, Math.min(30, 2000 / Math.max(n, 1)));
  const palette = isDarkMode() ? ION_H2_COLORS.dark : ION_H2_COLORS.light;
  const present = new Set(rows.map((r) => r.quantity));
  const domain = [ION_H2_LABELS.ion, ION_H2_LABELS.h2].filter((label) => present.has(label));
  const range = domain.map((label) => (label === ION_H2_LABELS.ion ? palette.ion : palette.h2));
  const ionTooltip = [
    { field: "quantity", title: "quantity", type: "nominal" },
    { field: "i", title: "step", type: "quantitative" },
    { field: "x", title: xKind === "time" ? "t (s)" : "n (cm⁻³)", type: "quantitative", format: ".3~g" },
    { field: "tHuman", title: "t", type: "nominal" },
    { field: "dt", title: "step Δt (s)", type: "quantitative", format: ".3~g" },
    { field: "value", title: "fraction", type: "quantitative", format: ".4~g" },
  ];
  return {
    title: { text: "Ionized / H₂ fraction", fontSize: 12, fontWeight: "bold", anchor: "start", offset: 6 },
    width, height: DETAIL_HEIGHT,
    layer: [{
      data: { values: zoomFilteredData(rows, "x", zoomDomain) },
      mark: { type: "line", point: { filled: true, size: pointSize, opacity: 0.9 } },
      encoding: {
        x: {
          field: "x", type: "quantitative",
          scale: { type: xKind === "time" ? "symlog" : "log" },
          axis: {
            title: xKind === "time" ? "t (s)" : "n (cm⁻³)", titleFontSize: 10, labelOverlap: "greedy",
            labelAngle: xKind === "time" ? -40 : 0,
            labelExpr: xKind === "time" ? TIME_LABEL_EXPR : undefined,
          },
        },
        y: { field: "value", type: "quantitative", scale: { type: "log" }, axis: { title: "fraction", titleFontSize: 10 } },
        color: { field: "quantity", type: "nominal", scale: { domain, range }, legend: { title: null, orient: "bottom", direction: "horizontal" } },
        opacity: { condition: { param: "ionToggle", value: 1 }, value: 0.15 },
        tooltip: ionTooltip,
      },
      params: [{ name: "ionToggle", select: { type: "point", fields: ["quantity"] }, bind: "legend" }],
    }],
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
function speciesPanel(xKind, rows, width, zoomDomain) {
  const n = rows.length;
  const pointSize = Math.max(4, Math.min(30, 2000 / Math.max(n, 1)));
  const domain = plotableSpecies().filter((name) => speciesMassAmu(name) !== undefined);
  const speciesTooltip = [
    { field: "species", title: "species", type: "nominal" },
    { field: "i", title: "step", type: "quantitative" },
    { field: "x", title: xKind === "time" ? "t (s)" : "n (cm⁻³)", type: "quantitative", format: ".3~g" },
    { field: "tHuman", title: "t", type: "nominal" },
    { field: "dt", title: "step Δt (s)", type: "quantitative", format: ".3~g" },
    { field: "value", title: "X_i (mass frac.)", type: "quantitative", format: ".4~g" },
  ];
  return {
    title: { text: "Species mass fraction (click a legend entry to isolate it)", fontSize: 12, fontWeight: "bold", anchor: "start", offset: 6 },
    width, height: DETAIL_HEIGHT + 20,
    layer: [{
      data: { values: zoomFilteredData(rows, "x", zoomDomain) },
      mark: { type: "line", point: { filled: true, size: pointSize, opacity: 0.9 } },
      encoding: {
        x: {
          field: "x", type: "quantitative",
          scale: { type: xKind === "time" ? "symlog" : "log" },
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
        tooltip: speciesTooltip,
      },
      params: [{ name: "speciesToggle", select: { type: "point", fields: ["species"] }, bind: "legend" }],
    }],
  };
}

// The whole current-run view: a single vertical stack -- density vs.
// time (free-fall only), temperature/thermal-energy vs. x, *zoomed*
// temperature vs. x, ionization/H2 fraction, species mass fraction --
// sharing one zoom (no shared crosshair any more -- see NOTES.md).
//
// Zooming: exactly one panel (temperature-vs-x, `tempOverview` below)
// declares the actual brush; every *other* panel -- including the
// dedicated zoomed copy of temperature itself, `tempZoomed` -- gets the
// same `zoomDomain` baked in as a literal `[lo, hi]` array (see
// metricPanel()'s own note on why this isn't a live param binding any
// more). The brush-holding panel itself is deliberately excluded from
// rescaling (it stays at the full range, showing the selection box,
// the standard Vega-Lite "overview" role) -- which is exactly why
// temperature alone still needs a real second, zoomed instance: it's
// the one metric whose own overview can't double as its own zoomed
// view.
//
// The composition is a single flat top-level `vconcat` -- every panel a
// direct item, never wrapped in an intermediate row spec -- simply the
// vertical single-chart-per-row layout asked for.
// `tempExtra`'s shock-event rule has its own single-row dataset (just
// `{x: nShock}`), entirely separate from `rows` -- so it was never
// touched by `zoomFilteredData()` at all, and Vega-Lite's default
// shared-scale-across-layers behavior pulled that one raw, unfiltered
// point straight into the "zoomed" panel's x-domain regardless of the
// actual zoom range, silently widening its right edge out to wherever
// the shock happened to be (found directly: reported as "the left edge
// is the same but not the right" against the ionization panel, which
// has no such extra layer to do this). The temperature *overview*
// panel is unaffected on purpose -- it always shows the full range
// anyway, so an unfiltered shock point there was never wrong. Only the
// zoomed copy needs its shock rule (not the shaded 1500-2500K band,
// which has no `x` field to begin with) dropped once it falls outside
// the current zoom.
function extraForZoom(tempExtra, zoomDomain) {
  if (!zoomDomain) return tempExtra;
  return tempExtra.filter((layer) => {
    const row = layer.data && layer.data.values && layer.data.values[0];
    if (!row || row.x === undefined) return true; // no x field (the band) -- always keep
    return row.x >= zoomDomain[0] && row.x <= zoomDomain[1];
  });
}

function runViewSpec({ mode, rows, tempField, tempExtra, ionRows, speciesRows, zoomDomain }) {
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
  // `initialBrush` re-populates its drag rectangle to match an
  // already-committed `zoomDomain` across a rebuild (see embedChart()) --
  // without it, every zoom commit would visibly reset the box to empty
  // even though the other panels stayed zoomed.
  const tempOverview = metricPanel({
    data: rows, xField: "x", yField: tempField, xKind: xKindOfMode, yTitle: FIELD_TITLE[tempField] || tempField,
    tooltip: tempTooltip, extra: tempExtra,
    brushName: "brush", initialBrush: zoomDomain,
    title: `${tempTitle} vs. ${xKindOfMode}`,
  });
  // The one *zoomed* panel this composition has -- directly below the
  // temperature-vs-x panel above. Circles/points on this one
  // specifically (not on any other panel here) -- an explicit, standing
  // preference from earlier in this project.
  const tempZoomed = metricPanel({
    data: rows, xField: "x", yField: tempField, xKind: xKindOfMode, yTitle: FIELD_TITLE[tempField] || tempField,
    tooltip: tempTooltip, extra: extraForZoom(tempExtra, zoomDomain),
    zoomDomain, showPoints: true, pointSize: 50,
    title: "zoomed",
  });

  const items = [];
  // Density-vs-time only makes sense in free-fall mode -- density is
  // held constant in cool mode, so this would just be a flat, useless
  // line there (matches the previous design's same free-fall-only
  // gating for this chart).
  if (mode === "freefall") items.push(densityTimePanel(rows, zoomDomain));
  items.push(tempOverview, tempZoomed);
  if (ionRows.length) items.push(ionPanel(xKindOfMode, ionRows, PANEL_WIDTH, zoomDomain));
  if (speciesRows.length) items.push(speciesPanel(xKindOfMode, speciesRows, PANEL_WIDTH, zoomDomain));

  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    background: null,
    config: vlConfig(),
    // legend/scale independence is belt-and-suspenders here (the
    // ionization and species charts already use different color fields
    // -- "quantity" vs "species" -- so Vega-Lite wouldn't merge their
    // legends by default regardless), made explicit because it was
    // asked for directly: the two must never share one legend.
    resolve: { legend: { color: "independent" }, scale: { color: "independent" } },
    vconcat: items,
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

// The inputs runViewSpec() needs to rebuild the chart *without*
// re-running the solver -- set once per redraw() (a real new run), then
// reused every time embedChart() rebuilds just for a zoom commit (see
// the note on `zoomDomain` in metricPanel()). `zoomDomain` is `null`
// (full range) or a committed `[lo, hi]` array; `zoomCommitTimer`
// debounces the brush's own live signal (which fires continuously while
// dragging) down to a single rebuild once dragging actually pauses.
let lastSpecInputs = null;
let zoomDomain = null;
let zoomCommitTimer = null;

// (Re-)embeds `#chart-run` from `lastSpecInputs`/`zoomDomain` -- called
// once per real redraw() (a fresh solve) and again, independently, every
// time a brush drag settles on a new range. `generation`, when given,
// is the calling redraw()'s own guard token; a bare call (from the
// brush listener itself) mints a fresh one, since that's a legitimate
// new "most recent" request in its own right, not a stale leftover.
function embedChart(generation) {
  const myGeneration = generation === undefined ? ++redrawGeneration : generation;
  const spec = runViewSpec({ ...lastSpecInputs, zoomDomain });
  // The editor action is enabled (and the other three left off) purely
  // so vega-embed builds its own "Open in Vega Editor" link/postMessage
  // handshake into the DOM -- the sidebar's "Open in Vega editor" button
  // (see initPage()) just finds and clicks that link itself rather than
  // reimplementing it; vega-embed's own floating action menu is hidden
  // via CSS (style.css) so it doesn't show up twice.
  vegaEmbed("#chart-run", spec, {
    actions: { export: false, source: false, compiled: false, editor: true },
    renderer: "svg",
  }).then((res) => {
    if (myGeneration !== redrawGeneration) return; // a newer redraw()/embedChart() already superseded this one
    runView = res.view;
    res.view.addSignalListener("brush", (name, value) => {
      updateZoomTimespan(value, lastSpecInputs.rows);
      // Debounced, not live: binding every follower panel's domain
      // straight to this signal (so they rescale continuously while
      // dragging) was tried first and is a real, measured performance
      // problem of its own -- see the note on `zoomDomain` in
      // metricPanel() -- so this only *reads* the live value here, and
      // commits a rebuilt chart with it baked in as a plain array once
      // dragging actually pauses for a moment, not on every intermediate
      // drag position.
      if (zoomCommitTimer) clearTimeout(zoomCommitTimer);
      zoomCommitTimer = setTimeout(() => {
        const xRange = value && value.x;
        const next = (xRange && xRange.length === 2)
          ? [Math.min(xRange[0], xRange[1]), Math.max(xRange[0], xRange[1])]
          : null;
        if (JSON.stringify(next) === JSON.stringify(zoomDomain)) return; // nothing actually changed
        zoomDomain = next;
        embedChart();
      }, 250);
    });
    updateZoomTimespan(zoomDomain ? { x: zoomDomain } : null, lastSpecInputs.rows);
  });
}

// Electron density has no slider of its own -- a user shouldn't be
// able to set it independently of the ionization state that actually
// implies it -- so it's derived via charge neutrality from whatever
// ionized species this network happens to track, each contributing
// its own ionic charge (H2_2 is H2+; He_3 is He++, hence the 2).
// Missing species (a network without H2 or without He) just don't
// contribute, so this generalizes across all three fiducial networks
// without needing to know which ones exist.
const IONIC_CHARGE = { H_2: 1, H2_2: 1, He_2: 1, He_3: 2 };

function currentFractions() {
  const fractions = {};
  for (const name of speciesNames) {
    if (name === "ge" || name === "de") continue;
    const el = document.getElementById("sp-" + name);
    if (el) fractions[name] = Math.pow(10, parseFloat(el.value));
  }
  // Previously fell through setIcs()'s own missing-species fallback
  // straight to a hardcoded 0 -- confirmed directly as the real cause
  // of a reported "density isn't monotonic just after t=0" artifact:
  // every run started genuinely charge-*non*-neutral, and the first
  // real step was mostly a very fast, physically-real charge-
  // neutralization transient, not a free-fall/chemistry effect worth
  // seeing at all.
  let de = 0;
  for (const [name, charge] of Object.entries(IONIC_CHARGE)) {
    if (fractions[name] !== undefined) de += charge * fractions[name];
  }
  fractions.de = de;
  return fractions;
}

function redraw() {
  const myGeneration = ++redrawGeneration; // see the comment on redrawGeneration above
  const logNH = parseFloat(document.getElementById("nH").value);
  const nH = Math.pow(10, logNH);
  const T = Math.pow(10, parseFloat(document.getElementById("T").value));
  // Same "10^x" framing as target n (and every other log-scale slider
  // here -- tolerance, collapse rate, step size, shock density): the
  // readout is the exponent the slider itself is dialing, not the
  // physical value, so initial and target n read the same way instead
  // of one showing "4.00e+04" and the other showing a bare "15.0".
  document.getElementById("nH-val").textContent = logNH.toFixed(1);
  document.getElementById("T-val").textContent = T.toFixed(0);
  const fractions = currentFractions();
  for (const name in fractions) {
    // "de" (electron density) is derived, not slider-controlled -- see
    // currentFractions() -- so it has no "-val" display element to
    // update, unlike every other key this loop sees.
    const valEl = document.getElementById("sp-" + name + "-val");
    if (valEl) valEl.textContent = fractions[name].toExponential(1);
  }

  const t0 = performance.now();
  let result;
  // Solver tolerance and (free-fall only) collapse-rate multiplier are
  // both plain, already-exposed runtime arguments to the compiled
  // solver/this file's own free-fall math -- neither needed any change
  // to the generated C++ itself, just threading an existing knob (or,
  // for collapse rate, one existing constant) through from a slider.
  const logTolerance = parseFloat(document.getElementById("tolerance").value);
  const tolerance = Math.pow(10, logTolerance);
  document.getElementById("tolerance-val").textContent = logTolerance.toFixed(1);
  if (currentMode === "freefall") {
    const logNTarget = parseFloat(document.getElementById("ntarget").value);
    document.getElementById("ntarget-val").textContent = logNTarget.toFixed(1);
    const logNShock = parseFloat(document.getElementById("nshock").value);
    const machSlider = parseFloat(document.getElementById("mach").value);
    // Forced to 1 (the zero-strength/no-op limit -- see shockJumpFactors())
    // when unchecked, regardless of the slider's own value, so toggling
    // the checkbox back on restores exactly the Mach number it was left
    // at instead of having reset it to 1.
    const shockEnabled = document.getElementById("shock-enabled").checked;
    const machShock = shockEnabled ? machSlider : 1;
    document.getElementById("nshock-val").textContent = logNShock.toFixed(1);
    document.getElementById("mach-val").textContent = machSlider.toFixed(1);
    const logCollapse = parseFloat(document.getElementById("collapse-rate").value);
    const collapseFactor = Math.pow(10, logCollapse);
    document.getElementById("collapse-rate-val").textContent = logCollapse.toFixed(1);
    const logFfStep = parseFloat(document.getElementById("ff-step").value);
    const ffStep = Math.pow(10, logFfStep);
    document.getElementById("ff-step-val").textContent = logFfStep.toFixed(1);
    result = runFreefall(nH, T, fractions, logNTarget, logNShock, machShock, collapseFactor, tolerance, ffStep);
  } else {
    const logDtf = parseFloat(document.getElementById("dtf").value);
    document.getElementById("dtf-val").textContent = logDtf.toFixed(1);
    result = runConstantDensity(nH, T, fractions, logDtf, tolerance);
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

  // A fresh run always starts unzoomed -- cached here, not passed as an
  // argument, so the brush-commit path below (embedChart(), fired later
  // from a signal listener, long after this particular redraw() call
  // has returned) can rebuild the exact same chart with a new zoom
  // without re-running the solver.
  lastSpecInputs = { mode: currentMode, rows, tempField, tempExtra, ionRows, speciesRows };
  zoomDomain = null;
  embedChart(myGeneration);

  updateRunSummary(result, rows, elapsed);

  lastResult = result;
  document.getElementById("download-csv").disabled = false;
  document.getElementById("export-editor").disabled = false;
}

function summaryStat(label, value, extra) {
  return `<span class="stat"><span class="label">${label}</span><strong>${value}</strong>`
    + (extra ? ` <span class="stat-extra">${extra}</span>` : "") + `</span>`;
}

// A prominent, always-visible run-metadata readout up near the top of
// the page -- replaces what used to be one small muted line at the
// bottom of the sidebar (easy to miss, and easy to lose track of once
// the species-fraction sliders pushed it further down): how many steps
// the adaptive stepper actually took, wall-clock solve time, and the
// same final-state numbers the old line had (final T, ionization, H2
// fraction, shock crossing), each labeled instead of packed into one
// run-on sentence.
function updateRunSummary(result, rows, elapsed) {
  const el = document.getElementById("run-summary");
  if (!el) return;
  const finalT = result.T[result.T.length - 1];
  const finalIon = result.ion[result.ion.length - 1];
  const finalH2 = result.h2[result.h2.length - 1];
  const finalTime = result.t[result.t.length - 1];
  const chips = [
    summaryStat("steps", rows.length),
    summaryStat("solve time", `${elapsed.toFixed(1)} ms`),
  ];
  if (finalTime !== undefined) {
    chips.push(summaryStat("elapsed sim. time", formatTimeAuto(finalTime), `(${finalTime.toExponential(2)} s)`));
  }
  chips.push(summaryStat("final T", finalT !== undefined ? `${finalT.toFixed(1)} K` : "?"));
  chips.push(summaryStat("final ionized", finalIon !== undefined ? finalIon.toExponential(2) : "?"));
  if (finalH2 !== null && finalH2 !== undefined) chips.push(summaryStat("final H₂/H_tot", finalH2.toExponential(2)));
  if (result.shockTriggered) chips.push(summaryStat("shock crossed at", `${result.nShock.toExponential(2)} cm⁻³`));
  // `reachedTarget` is only ever `false` (not just falsy/undefined --
  // cool mode's result has no such field at all) when free-fall's outer
  // loop hit its step cap before reaching the requested target density
  // -- genuinely possible now that step size is a user-exposed slider,
  // not just a pathological setting. Surfaced here, not silently.
  if (result.reachedTarget === false) {
    chips.push(`<span class="stat stat-warn">stopped before reaching target n (step cap) -- try a coarser step size</span>`);
  }
  el.innerHTML = chips.join("");
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
  for (const id of ["ntarget-row", "collapse-rate-row", "ff-step-row", "shock-row"]) {
    document.getElementById(id).style.display = mode === "freefall" ? "" : "none";
  }
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

// Species-fraction sliders are otherwise fully independent (see
// currentFractions()/setIcs(), which just does nH * frac per species) --
// nothing stops setting e.g. both an ionized-H and an H2 fraction so
// high that, together, they'd claim more H nuclei than actually exist.
// A genuine "these sliders must sum to a fixed budget" control needs
// real N-dimensional UI, which is more than this is worth building (see
// NOTES.md) -- this is the cheap approximation that's actually worth
// having: species that share the same nucleus are grouped (H_1/H_2/
// H_m0/H2_1/H2_2 all draw on the same H budget -- H2 counts double,
// two H nuclei per molecule; He_1/He_2/He_3 on the same He budget), and
// dragging one slider proportionally rescales the *other* sliders in
// its group to hold the group's total nuclei fraction at whatever the
// network's own default initial conditions implied -- never touching
// the slider actually being dragged. Only if the others are already
// all the way down at the floor and the dragged slider *alone* still
// exceeds the group's budget does this clamp the dragged slider itself,
// as a last resort.
const ELEMENT_GROUPS = {
  H: ["H_1", "H_2", "H_m0", "H2_1", "H2_2"],
  He: ["He_1", "He_2", "He_3"],
};
const NUCLEI_WEIGHT = { H2_1: 2, H2_2: 2 }; // everything else is one nucleus per formula unit
const SPECIES_FRAC_FLOOR = 1e-14, SPECIES_FRAC_CEIL = 1; // matches every species slider's min/max (see below)
let groupTargets = {}; // element -> nuclei-weighted fraction total, fixed once at page load from config.default_ics

function nucleiWeight(name) {
  return NUCLEI_WEIGHT[name] || 1;
}

function computeGroupTargets(config) {
  groupTargets = {};
  for (const [el, names] of Object.entries(ELEMENT_GROUPS)) {
    let total = 0;
    for (const name of names) {
      const frac = config.default_ics[name];
      if (frac !== undefined) total += nucleiWeight(name) * frac;
    }
    groupTargets[el] = total;
  }
}

function enforceGroupConservation(draggedName) {
  const el = Object.keys(ELEMENT_GROUPS).find((k) => ELEMENT_GROUPS[k].includes(draggedName));
  if (!el) return; // not a grouped species (shouldn't happen for a species slider)
  const target = groupTargets[el];
  if (!(target > 0)) return; // this network's own defaults never populated the group -- nothing to conserve against

  const sliders = {};
  for (const name of ELEMENT_GROUPS[el]) {
    const s = document.getElementById("sp-" + name);
    if (s) sliders[name] = s;
  }
  const draggedSlider = sliders[draggedName];
  if (!draggedSlider) return;
  const draggedWeighted = nucleiWeight(draggedName) * Math.pow(10, parseFloat(draggedSlider.value));

  const others = Object.keys(sliders).filter((n) => n !== draggedName);
  const othersFrac = {};
  let othersWeightedSum = 0;
  for (const name of others) {
    othersFrac[name] = Math.pow(10, parseFloat(sliders[name].value));
    othersWeightedSum += nucleiWeight(name) * othersFrac[name];
  }

  const budget = target - draggedWeighted; // what's left in the group's budget for everyone else
  if (budget <= 0) {
    // The dragged slider alone already claims the whole group's budget
    // (or more) -- push everyone else to the floor, then, only as a
    // last resort, clamp the dragged slider itself back down to fit.
    for (const name of others) sliders[name].value = Math.log10(SPECIES_FRAC_FLOOR);
    const maxDraggedFrac = target / nucleiWeight(draggedName);
    if (maxDraggedFrac >= SPECIES_FRAC_FLOOR) draggedSlider.value = Math.log10(maxDraggedFrac);
    return;
  }
  if (othersWeightedSum <= 0) return; // nothing to redistribute proportionally from

  const scale = budget / othersWeightedSum;
  for (const name of others) {
    const newFrac = Math.min(SPECIES_FRAC_CEIL, Math.max(SPECIES_FRAC_FLOOR, othersFrac[name] * scale));
    sliders[name].value = Math.log10(newFrac);
  }
}

function buildSpeciesSliders(config) {
  computeGroupTargets(config);
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
    row.querySelector("input").addEventListener("input", () => {
      enforceGroupConservation(name);
      scheduleRedraw();
    });
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
  // Reuses vega-embed's own "Open in Vega Editor" action (built into
  // #chart-run's spec -- see embedChart()) rather than re-implementing
  // its postMessage handshake with the online editor: that action is
  // still a real link in the DOM (just hidden, see style.css), so
  // clicking it programmatically does exactly what vega-embed's own
  // hover menu would have done.
  document.getElementById("export-editor").addEventListener("click", () => {
    const link = document.querySelector("#chart-run .vega-actions a");
    if (link) link.click();
  });
  document.getElementById("mode-cool").addEventListener("click", () => setMode("cool"));
  document.getElementById("mode-freefall").addEventListener("click", () => setMode("freefall"));
  document.getElementById("T-mode-T").addEventListener("click", () => setTemperatureDisplayMode("T"));
  document.getElementById("T-mode-ge").addEventListener("click", () => setTemperatureDisplayMode("ge"));
  document.getElementById("ic-preset").addEventListener("change", (e) => applyPreset(e.target.value));
  for (const id of ["nH", "T", "dtf", "ntarget", "nshock", "mach", "collapse-rate", "tolerance", "ff-step"]) {
    document.getElementById(id).addEventListener("input", scheduleRedraw);
  }
  const shockEnabledEl = document.getElementById("shock-enabled");
  const updateShockEnabledUI = () => {
    document.getElementById("nshock").disabled = !shockEnabledEl.checked;
    document.getElementById("mach").disabled = !shockEnabledEl.checked;
  };
  shockEnabledEl.addEventListener("change", () => { updateShockEnabledUI(); scheduleRedraw(); });
  updateShockEnabledUI();
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
    document.getElementById("ic-preset").disabled = false;
    document.getElementById("status").textContent = "ready";
    redraw();
  });
}

// Long, genuinely-useful-but-not-always-needed explanatory paragraphs
// (target n, collapse rate, step size, tolerance, shock) are gated
// behind a small "?" button next to their slider's label rather than
// always on screen -- one delegated listener here covers all of them
// (present or future) instead of wiring each one up individually.
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".info-btn");
  if (!btn) return;
  const note = document.getElementById(btn.getAttribute("aria-controls"));
  if (!note) return;
  const wasExpanded = btn.getAttribute("aria-expanded") === "true";
  btn.setAttribute("aria-expanded", String(!wasExpanded));
  note.hidden = wasExpanded;
});
