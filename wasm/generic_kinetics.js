// Prototype: a hand-written, reaction-agnostic mass-action-kinetics
// RHS/Jacobian assembler, driven entirely by data (a reaction database
// exported once by generate_reaction_db.py, plus whatever subset of it
// is currently checked) -- see wasm/README-generic.md. Unlike the
// production path (dengo's sympy codegen -> per-network C++ ->
// Emscripten), nothing here is regenerated or recompiled when the
// selection changes: ordinary mass-action kinetics
//   rate_r(T) * product over reactants of [n_i]^(stoichiometric power)
// is the same formula for every reaction, so it only needs writing
// once. Cooling actions aren't one universal formula the way reactions
// are, so each one's *equation* is lowered once, in Python at export
// time (sympy's jscode printer, see generate_reaction_db.py) -- still
// no per-*selection* codegen, and no compile step either way; `new
// Function()` below is just JS's own built-in "make a callable from a
// source string" primitive, not a build/toolchain step.

// -- Reaction database ---------------------------------------------------
function loadReactionDb(db) {
  // Per-species net stoichiometric change and per-species reactant
  // *power* (summed across every occurrence of that species on that
  // side -- e.g. k22's left side lists H_1 twice, [[2,"H_1"],[1,"H_1"]],
  // meaning n_H^3 in the rate law and a net change of -2 once the
  // 1*H_1 on the right side is subtracted back out; see reaction_
  // classes.py's Reaction.lhs_equation()/net_change(), which this
  // mirrors) are computed once here, not re-derived on every RHS call.
  const reactions = db.reactions.map((r) => {
    const leftPower = {};
    for (const [n, name] of r.left) leftPower[name] = (leftPower[name] || 0) + n;
    const netChange = {};
    for (const [n, name] of r.left) netChange[name] = (netChange[name] || 0) - n;
    for (const [n, name] of r.right) netChange[name] = (netChange[name] || 0) + n;
    // left/right themselves are kept too (not just leftPower/netChange)
    // -- purely for display (generic_ui.js's reactionLabel()/
    // reactionSpeciesNames()), never read by the RHS/Jacobian math below.
    return { name: r.name, left: r.left, right: r.right, leftPower, netChange, rate: r.rate };
  });
  const cooling = (db.cooling || []).map(compileCoolingAction);
  return { T_grid: db.T_grid, constants: db.constants, species: db.species, reactions, cooling };
}

// One real JS function per cooling action, compiled once when the
// database loads (not once per RHS/Jacobian call, and not once per
// selection -- every action in the catalog is compiled regardless of
// what's later checked). Every compiled function takes the same
// argument shape -- (this action's own species names, then its own
// rate-table names, then T) -- unused parameters (most actions don't
// use T; only compton's exported `js` actually references it) are
// harmless, so there's no need to special-case which actions do.
function compileCoolingAction(entry) {
  const argNames = [...entry.species, ...Object.keys(entry.tables), "T"];
  const fn = new Function(argNames, "return (" + entry.js + ");");
  return { name: entry.name, species: entry.species, tables: entry.tables, argNames, fn };
}

// Linear interpolation of a reaction's tabulated rate(T) onto an
// arbitrary T -- same idea as the compiled solver's own rate-table
// interpolation (see {name}_interpolate_rates() in cython_solver.C.
// template), just against this prototype's own (coarser, JSON-exported)
// table and a single scalar T rather than a per-cell array, since this
// prototype only ever runs at one fixed, dialed T per run.
function interpolateRate(db, rateTable, T) {
  const grid = db.T_grid;
  if (T <= grid[0]) return rateTable[0];
  if (T >= grid[grid.length - 1]) return rateTable[rateTable.length - 1];
  let lo = 0, hi = grid.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (grid[mid] <= T) lo = mid; else hi = mid;
  }
  const frac = (T - grid[lo]) / (grid[hi] - grid[lo]);
  return rateTable[lo] + frac * (rateTable[hi] - rateTable[lo]);
}

// Rate coefficients don't depend on the species state, only on T --
// with no thermal coupling (T fixed for the whole run), this is
// evaluated once per run and reused; with cooling active, T itself
// evolves (see TFromGe() below) so this is called fresh every RHS/
// Jacobian evaluation instead -- same interpolateRate() either way.
function activeRatesAtT(db, activeNames, T) {
  const rates = {};
  for (const r of db.reactions) {
    if (activeNames.has(r.name)) rates[r.name] = interpolateRate(db, r.rate, T);
  }
  return rates;
}

// -- Thermal coupling: ge <-> T, and the cooling-action sum --------------
// A single constant (monatomic-ideal-gas) gamma for every species,
// rather than the compiled solver's own T-dependent interpolated gamma
// for H2-bearing gas (roto-vibrational degrees of freedom activating)
// -- a real simplification (H2-heavy gas's heat capacity reads a bit
// off), not just a labeling one; see generate_reaction_db.py's own
// comment on `constants.gamma` and NOTES.md/README-generic.md.
//
// Ideal gas: internal energy density u = n_total*kB*T/(gamma-1)
// [erg/cm^3]; ge (specific internal energy, erg/g) = u / mass_density.
function computeMdensityAmu(db, state) {
  // mass_density in amu/cm^3 -- caller multiplies by constants.mh for
  // grams/cm^3, same split app.js's own totalMassAmu() uses and for the
  // same reason (the amu->gram factor is one guaranteed-shared constant
  // factor, not worth threading through every intermediate value).
  let m = 0;
  for (const sp of db.species) {
    const n = state[sp.name];
    if (n !== undefined) m += n * sp.weight;
  }
  return m;
}

function totalNumberDensity(db, state) {
  let n = 0;
  for (const sp of db.species) {
    if (state[sp.name] !== undefined) n += state[sp.name];
  }
  return n;
}

function TFromGe(db, ge, mdensityAmu, nTotal) {
  const { kboltz, mh, gamma } = db.constants;
  return (ge * (gamma - 1) * mdensityAmu * mh) / (nTotal * kboltz);
}

function geFromT(db, T, mdensityAmu, nTotal) {
  const { kboltz, mh, gamma } = db.constants;
  return (nTotal * kboltz * T) / ((gamma - 1) * mdensityAmu * mh);
}

// Sum of every *active* cooling action's contribution at the given
// state/T, in erg/s/cm^3 (matching the compiled solver's own
// convention -- print_cooling()'s ccode is likewise volumetric,
// divided by mass density only afterward, in calculate_rhs's
// `rhs[ge_index] /= mdensity`; see genericRhsFull() below for the
// matching division here).
function coolingSumAtT(db, activeCoolingNames, state, T) {
  let sum = 0;
  for (const c of db.cooling) {
    if (!activeCoolingNames.has(c.name)) continue;
    const args = c.argNames.map((name) => {
      if (name === "T") return T;
      if (name in c.tables) return interpolateRate(db, c.tables[name], T);
      return state[name];
    });
    sum += c.fn(...args);
  }
  return sum;
}

// d(species)/dt at the given state (a plain object: name -> number
// density), for whichever reactions are in `rates` (activeRatesAtT()'s
// output -- its keys *are* the active set). Standard mass-action
// kinetics: this is the same formula for every reaction, so nothing
// about which ones are active needs special-casing.
function genericRhs(db, rates, state) {
  const rhs = {};
  for (const sp of db.species) rhs[sp.name] = 0;
  for (const r of db.reactions) {
    const rate = rates[r.name];
    if (rate === undefined) continue; // not in the active set
    let term = rate;
    for (const name in r.leftPower) term *= Math.pow(state[name], r.leftPower[name]);
    for (const name in r.netChange) rhs[name] += r.netChange[name] * term;
  }
  return rhs;
}

// d(rhs_i)/d(state_j) -- the standard partial derivative of the mass-
// action monomial above: d/dn_j [n_j^p * (other factors)] = p * n_j^(p-1)
// * (other factors) = p/n_j * (the whole term), so this reuses the same
// per-reaction `term` the RHS pass computes rather than re-deriving a
// separate expression per (species, species) pair the way sympy's
// print_jacobian_component() has to.
function genericJacobian(db, rates, state, speciesOrder) {
  const n = speciesOrder.length;
  const idx = {};
  speciesOrder.forEach((name, i) => { idx[name] = i; });
  const J = new Float64Array(n * n); // column-major, J[i + j*n] = d(rhs_i)/d(state_j), matching BE_chem_solve.C's Gauss_Elim layout
  for (const r of db.reactions) {
    const rate = rates[r.name];
    if (rate === undefined) continue;
    const leftNames = Object.keys(r.leftPower);
    for (const jName of leftNames) {
      const p = r.leftPower[jName];
      // d/dn_j [n_j^p * (other reactants' factors)] = p * n_j^(p-1) *
      // (other factors) -- computed directly (not as term/n_j, which
      // is 0/0 -- NaN, not 0 -- the moment any reactant's density is
      // exactly zero, a real case here (an intermediate/product
      // species genuinely starting at 0, not just astrophysically
      // "small"), not just a theoretical edge case; see NOTES.md).
      let dterm_dj = p * Math.pow(state[jName], p - 1) * rate;
      for (const otherName of leftNames) {
        if (otherName !== jName) dterm_dj *= Math.pow(state[otherName], r.leftPower[otherName]);
      }
      const j = idx[jName];
      if (j === undefined) continue; // a reactant not in this run's species subset -- shouldn't happen if the UI keeps selections consistent, but skip rather than throw
      for (const iName in r.netChange) {
        const i = idx[iName];
        if (i === undefined) continue;
        J[i + j * n] += r.netChange[iName] * dterm_dj;
      }
    }
  }
  return J;
}

// -- Thermal coupling wrappers: species RHS/Jacobian above, plus ge -----
// When no cooling action is active, T is passed straight through (the
// original fixed-T behavior, unchanged) and the returned RHS has no
// `ge` entry at all -- ge simply isn't part of the ODE system in that
// mode. When at least one cooling action is active, ge is: T is instead
// *derived* from the current ge (TFromGe()), used to interpolate both
// reaction and cooling rates, and ge's own RHS is the cooling sum
// (converted from erg/s/cm^3 to erg/s/g, matching the compiled solver's
// own `rhs[ge_index] /= mdensity` -- see coolingSumAtT()'s comment).
function genericRhsFull(db, state, activeReactionNames, activeCoolingNames, fixedT) {
  const hasCooling = activeCoolingNames && activeCoolingNames.size > 0;
  const mdensityAmu = computeMdensityAmu(db, state);
  const nTotal = totalNumberDensity(db, state);
  const T = hasCooling ? TFromGe(db, state.ge, mdensityAmu, nTotal) : fixedT;
  const rates = activeRatesAtT(db, activeReactionNames, T);
  const rhs = genericRhs(db, rates, state);
  if (hasCooling) {
    const coolSum = coolingSumAtT(db, activeCoolingNames, state, T); // erg/s/cm^3
    rhs.ge = coolSum / (mdensityAmu * db.constants.mh); // -> erg/s/g
  }
  return { rhs, T };
}

// The species-species block is exact/analytic (genericJacobian(), same
// as the fixed-T case, evaluated at the state's current T). The ge row
// (d(ge_rhs)/d(species)) and column (d(everything)/d(ge), including
// d(ge_rhs)/d(ge) itself) are finite-differenced instead: the compiled
// solver gets these analytically (see reaction_classes.py's
// ReactionCoefficient._eval_derivative(), which differentiates every
// rate/cooling table symbol w.r.t. ge via a precomputed `dr<name>`
// table) -- reproducing that here would mean symbolically
// differentiating every jscode-lowered cooling expression and every
// reaction's rate table w.r.t. T too, which is real additional work for
// what's ultimately still just a Newton-iteration convergence aid (an
// approximate Jacobian slows convergence, typically costing a few extra
// sweeps or step retries; it doesn't change what a *converged* step
// means, since BE_chem_solve.C's convergence check is on the actual
// residual/update norm, not on Jacobian fidelity) -- a deliberate scope
// cut for this prototype, not an oversight.
function genericJacobianFull(db, state, activeReactionNames, activeCoolingNames, speciesOrder, fixedT) {
  const hasCooling = activeCoolingNames && activeCoolingNames.size > 0;
  const n = speciesOrder.length;
  if (!hasCooling) {
    const T = fixedT;
    const rates = activeRatesAtT(db, activeReactionNames, T);
    return genericJacobian(db, rates, state, speciesOrder);
  }

  const total = n + 1, geIdx = n;
  const { T } = genericRhsFull(db, state, activeReactionNames, activeCoolingNames, fixedT);
  const rates = activeRatesAtT(db, activeReactionNames, T);
  const Jspecies = genericJacobian(db, rates, state, speciesOrder);
  const J = new Float64Array(total * total);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) J[i + j * total] = Jspecies[i + j * n];
  }

  const base = genericRhsFull(db, state, activeReactionNames, activeCoolingNames, fixedT);
  const RELSTEP = 1e-6, FLOOR = 1e-30;

  // ge column: bump ge, re-evaluate the *entire* RHS (T shifts too),
  // finite-difference every row against the baseline.
  {
    const dge = Math.max(Math.abs(state.ge) * RELSTEP, FLOOR);
    const bumped = Object.assign({}, state, { ge: state.ge + dge });
    const perturbed = genericRhsFull(db, bumped, activeReactionNames, activeCoolingNames, fixedT);
    for (let i = 0; i < n; i++) {
      J[i + geIdx * total] = (perturbed.rhs[speciesOrder[i]] - base.rhs[speciesOrder[i]]) / dge;
    }
    J[geIdx + geIdx * total] = (perturbed.rhs.ge - base.rhs.ge) / dge;
  }
  // ge row: bump each species in turn, finite-difference ge's own RHS.
  for (let j = 0; j < n; j++) {
    const name = speciesOrder[j];
    const dn = Math.max(Math.abs(state[name]) * RELSTEP, FLOOR);
    const bumped = Object.assign({}, state, { [name]: state[name] + dn });
    const perturbed = genericRhsFull(db, bumped, activeReactionNames, activeCoolingNames, fixedT);
    J[geIdx + j * total] = (perturbed.rhs.ge - base.rhs.ge) / dn;
  }
  return J;
}

// -- Wiring the generic (compiled-once) wasm integrator to this JS ------
// -- RHS/Jacobian via Emscripten's addFunction() ------------------------
// Builds the two C-callable function pointers dengo_generic_step()
// expects (rhs_f/jac_f, see dengo_generic.cpp), each just a thin
// marshaling layer between wasm linear memory (a Float64Array view,
// same idiom app.js's getScalar()/setIcs() already use for the
// production per-network modules) and the plain-JS-object state the
// generic*Full() functions above operate on. `stateOrder` is
// speciesOrder with "ge" appended when cooling is active, [] otherwise
// -- the actual layout of the wasm state buffer this run uses.
function makeCallbacks(mod, db, activeReactionNames, activeCoolingNames, speciesOrder, stateOrder, fixedT) {
  const rhsFn = mod.addFunction((uPtr, fuPtr, nstrip, nchem, sdata) => {
    const u = mod.HEAPF64.subarray(uPtr >> 3, (uPtr >> 3) + nchem);
    const state = {};
    stateOrder.forEach((name, i) => { state[name] = u[i]; });
    const { rhs } = genericRhsFull(db, state, activeReactionNames, activeCoolingNames, fixedT);
    const fu = mod.HEAPF64.subarray(fuPtr >> 3, (fuPtr >> 3) + nchem);
    stateOrder.forEach((name, i) => { fu[i] = rhs[name]; });
    return 0;
  }, "iiiiii"); // rhs_f = int(double*, double*, int, int, void*) -- 5 params + 1 return = 6 slots

  const jacFn = mod.addFunction((uPtr, JuPtr, nstrip, nchem, sdata) => {
    const u = mod.HEAPF64.subarray(uPtr >> 3, (uPtr >> 3) + nchem);
    const state = {};
    stateOrder.forEach((name, i) => { state[name] = u[i]; });
    const J = genericJacobianFull(db, state, activeReactionNames, activeCoolingNames, speciesOrder, fixedT);
    const Ju = mod.HEAPF64.subarray(JuPtr >> 3, (JuPtr >> 3) + nchem * nchem);
    Ju.set(J);
    return 0;
  }, "iiiiii"); // jac_f -- same 6-slot signature as rhs_f above

  return { rhsFn, jacFn };
}

// -- Full run: constant density, whatever species/reaction/cooling -----
// -- subset is currently active -------------------------------------------
// Log-spaced checkpoints approaching dtfTotal, same idea as the
// production runConstantDensity()'s forSweep branch (app.js) -- a
// fixed, display-oriented checkpoint schedule rather than an adaptive-
// step-size-driven one (there's no cooling-time-based step sizing here
// -- see runOneSweepCell()/coolingTime() in app.js for the production
// equivalent this doesn't attempt). dengo_generic_step() itself still
// does its own internal adaptive substepping (see dengo_generic.cpp) to
// actually hit each checkpoint; this just picks *which* times are worth
// recording.
const GENERIC_CHECKPOINTS = 40;

// `activeCoolingNames`: pass an empty Set (or omit) for the original
// fixed-T behavior -- `T0` is then used throughout and never evolves.
// With at least one entry, ge is added to the tracked state (seeded via
// geFromT() from T0) and T instead evolves from it every step; `T0` is
// then just the starting temperature.
function runGenericIntegration(mod, db, activeSpeciesNames, activeReactionNames, activeCoolingNames, initialState, T0, dtfTotal, tolerance) {
  tolerance = tolerance || 1e-5;
  activeCoolingNames = activeCoolingNames || new Set();
  const hasCooling = activeCoolingNames.size > 0;
  const speciesOrder = Array.from(activeSpeciesNames);
  const stateOrder = hasCooling ? [...speciesOrder, "ge"] : speciesOrder;
  const nchem = stateOrder.length;

  const statePtr = mod.ccall("dengo_generic_alloc", "number", ["number"], [nchem]);
  const sview = mod.HEAPF64.subarray(statePtr >> 3, (statePtr >> 3) + nchem);
  speciesOrder.forEach((name, i) => { sview[i] = initialState[name]; });
  if (hasCooling) {
    const mdensityAmu = computeMdensityAmu(db, initialState);
    const nTotal = totalNumberDensity(db, initialState);
    sview[speciesOrder.length] = geFromT(db, T0, mdensityAmu, nTotal);
  }

  const { rhsFn, jacFn } = makeCallbacks(mod, db, activeReactionNames, activeCoolingNames, speciesOrder, stateOrder, T0);

  const snapshot = () => {
    const s = {};
    stateOrder.forEach((name, i) => { s[name] = sview[i]; });
    if (hasCooling) {
      s.T = TFromGe(db, s.ge, computeMdensityAmu(db, s), totalNumberDensity(db, s));
    } else {
      s.T = T0;
    }
    return s;
  };

  const tHist = [0], sHist = [snapshot()];
  let t = 0;
  for (let k = 0; k < GENERIC_CHECKPOINTS; k++) {
    const target = dtfTotal * Math.pow(10, -4 * (1 - (k + 1) / GENERIC_CHECKPOINTS));
    const dt = target - t;
    if (dt <= 0) continue;
    const converged = mod.ccall(
      "dengo_generic_step", "number",
      ["number", "number", "number", "number", "number", "number", "number"],
      [rhsFn, jacFn, statePtr, dt, 200, tolerance, nchem],
    );
    if (!converged) break;
    t = target;
    tHist.push(t);
    sHist.push(snapshot());
  }

  mod.removeFunction(rhsFn);
  mod.removeFunction(jacFn);
  mod.ccall("dengo_generic_free", null, ["number"], [statePtr]);

  return { t: tHist, s: sHist, speciesOrder, hasCooling };
}
