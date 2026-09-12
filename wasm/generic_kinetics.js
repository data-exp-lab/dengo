// Prototype: a hand-written, reaction-agnostic mass-action-kinetics
// RHS/Jacobian assembler, driven entirely by data (a reaction database
// exported once by generate_reaction_db.py, plus whatever subset of it
// is currently checked) -- see wasm/README-generic.md. Unlike the
// production path (dengo's sympy codegen -> per-network C++ ->
// Emscripten), nothing here is regenerated or recompiled when the
// selection changes: ordinary mass-action kinetics
//   rate_r(T) * product over reactants of [n_i]^(stoichiometric power)
// is the same formula for every reaction, so it only needs writing
// once. Fixed-T only (no thermal/ge coupling) -- see the reaction
// database's own doc comment for why that's this prototype's scope.

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
  return { T_grid: db.T_grid, species: db.species, reactions };
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

// Rate coefficients don't depend on the species state, only on T, which
// is fixed for the whole run here -- evaluated once per run (not once
// per RHS/Jacobian call, which happens many times per Newton sweep) and
// reused, same reasoning as the compiled solver caching its own table
// lookups per accepted step rather than recomputing every call.
function activeRatesAtT(db, activeNames, T) {
  const rates = {};
  for (const r of db.reactions) {
    if (activeNames.has(r.name)) rates[r.name] = interpolateRate(db, r.rate, T);
  }
  return rates;
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
    let term = rate;
    for (const name in r.leftPower) term *= Math.pow(state[name], r.leftPower[name]);
    for (const jName in r.leftPower) {
      const p = r.leftPower[jName];
      const dterm_dj = (p * term) / state[jName];
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

// -- Wiring the generic (compiled-once) wasm integrator to this JS ------
// -- RHS/Jacobian via Emscripten's addFunction() ------------------------
// Builds the two C-callable function pointers dengo_generic_step()
// expects (rhs_f/jac_f, see dengo_generic.cpp), each just a thin
// marshaling layer between wasm linear memory (a Float64Array view,
// same idiom app.js's getScalar()/setIcs() already use for the
// production per-network modules) and the plain-JS-object state the
// generic*() functions above operate on.
function makeCallbacks(mod, db, rates, speciesOrder) {
  const n = speciesOrder.length;

  const rhsFn = mod.addFunction((uPtr, fuPtr, nstrip, nchem, sdata) => {
    const u = mod.HEAPF64.subarray(uPtr >> 3, (uPtr >> 3) + nchem);
    const state = {};
    speciesOrder.forEach((name, i) => { state[name] = u[i]; });
    const rhs = genericRhs(db, rates, state);
    const fu = mod.HEAPF64.subarray(fuPtr >> 3, (fuPtr >> 3) + nchem);
    speciesOrder.forEach((name, i) => { fu[i] = rhs[name]; });
    return 0;
  }, "iiiiii"); // rhs_f = int(double*, double*, int, int, void*) -- 5 params + 1 return = 6 slots

  const jacFn = mod.addFunction((uPtr, JuPtr, nstrip, nchem, sdata) => {
    const u = mod.HEAPF64.subarray(uPtr >> 3, (uPtr >> 3) + nchem);
    const state = {};
    speciesOrder.forEach((name, i) => { state[name] = u[i]; });
    const J = genericJacobian(db, rates, state, speciesOrder);
    const Ju = mod.HEAPF64.subarray(JuPtr >> 3, (JuPtr >> 3) + nchem * nchem);
    Ju.set(J);
    return 0;
  }, "iiiiii"); // jac_f -- same 6-slot signature as rhs_f above

  return { rhsFn, jacFn };
}

// -- Full run: constant T, constant density, whatever species/reaction --
// -- subset is currently active -------------------------------------------
// Log-spaced checkpoints approaching dtfTotal, same idea as the
// production runConstantDensity()'s forSweep branch (app.js) -- a fixed,
// display-oriented checkpoint schedule rather than an adaptive-step-
// size-driven one, since there's no cooling time to size steps off of
// here (T is constant by construction -- see this file's own header
// comment on scope). dengo_generic_step() itself still does its own
// internal adaptive substepping (see dengo_generic.cpp) to actually hit
// each checkpoint; this just picks *which* times are worth recording.
const GENERIC_CHECKPOINTS = 40;

function runGenericIntegration(mod, db, activeSpeciesNames, activeReactionNames, initialState, T, dtfTotal, tolerance) {
  tolerance = tolerance || 1e-5;
  const speciesOrder = Array.from(activeSpeciesNames);
  const n = speciesOrder.length;
  const rates = activeRatesAtT(db, activeReactionNames, T);

  const statePtr = mod.ccall("dengo_generic_alloc", "number", ["number"], [n]);
  const sview = mod.HEAPF64.subarray(statePtr >> 3, (statePtr >> 3) + n);
  speciesOrder.forEach((name, i) => { sview[i] = initialState[name]; });

  const { rhsFn, jacFn } = makeCallbacks(mod, db, rates, speciesOrder);

  const snapshot = () => {
    const s = {};
    speciesOrder.forEach((name, i) => { s[name] = sview[i]; });
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
      [rhsFn, jacFn, statePtr, dt, 200, tolerance, n],
    );
    if (!converged) break;
    t = target;
    tHist.push(t);
    sHist.push(snapshot());
  }

  mod.removeFunction(rhsFn);
  mod.removeFunction(jacFn);
  mod.ccall("dengo_generic_free", null, ["number"], [statePtr]);

  return { t: tHist, s: sHist, speciesOrder };
}
