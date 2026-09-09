// A small, dependency-free wrapper around any dengo-generated WASM
// solver module (ChemicalNetwork.write_wasm_solver() + the Emscripten
// build in generate_site.py) -- reusable by any consumer, not just this
// project's own widget. app.js deliberately keeps its own from-scratch
// plumbing for now rather than being migrated onto this (see NOTES.md
// for why) -- this file has no dependency on app.js or vice versa.
//
// The C ABI here (dengo_wasm.cpp.template) is identical across every
// generated network: the same 7 functions always, at the same names.
// Only the species *names* differ between networks, and those are read
// at runtime via dengo_wasm_species_names(), not baked into any
// per-network build -- so one class covers every network dengo can
// generate a wasm solver for, not just the three this project ships.
//
// Two real gotchas in the underlying C API are preserved here exactly
// as they behave, not silently papered over -- this is meant to be a
// thin, predictable wrapper, not a "smart" layer with surprising
// implicit behavior:
//   1. temperature() reads a value that's only refreshed as a side
//      effect of calculate_rhs/calculate_jacobian -- i.e. by rhs() or
//      step(), not by writeState()/set() alone. Call rhs() first if
//      you need a fresh reading right after changing state without
//      stepping.
//   2. Emscripten's HEAPF64 typed-array view can be invalidated by WASM
//      memory growth (this build uses ALLOW_MEMORY_GROWTH=1) -- never
//      hold onto a reference to it longer than one synchronous access.
//      Every method here re-reads mod.HEAPF64 fresh on each call for
//      exactly that reason; bulkUpdate() hands your callback a freshly
//      read one too, rather than exposing a property you could
//      accidentally cache yourself.

export class DengoSolver {
  /**
   * @param {any} mod an already-resolved dengo-generated Emscripten
   *   module (the awaited return value of its ModuleFactory). Prefer
   *   `DengoSolver.create()` unless you have your own reason to resolve
   *   the module yourself first.
   */
  constructor(mod) {
    this._mod = mod;
    this._init = mod.cwrap("dengo_wasm_init", null, []);
    this._step = mod.cwrap("dengo_wasm_step", "number", ["number", "number", "number"]);
    this._statePtr = mod.cwrap("dengo_wasm_state_ptr", "number", []);
    this._rhsPtr = mod.cwrap("dengo_wasm_rhs_ptr", "number", []);
    this._temperatureFn = mod.cwrap("dengo_wasm_temperature", "number", []);
    const namesFn = mod.cwrap("dengo_wasm_species_names", "string", []);

    /** Every species this network's state vector holds, in the exact
     * order the state buffer uses -- read once at construction time,
     * frozen so nothing downstream can mutate it by accident. */
    this.speciesNames = Object.freeze(namesFn().split(","));
    this._index = Object.fromEntries(this.speciesNames.map((n, i) => [n, i]));

    this._init();
  }

  /**
   * @param {() => Promise<any>} moduleFactory a dengo-generated
   *   Emscripten module factory (e.g. the page-global `DengoModule`
   *   from a `<script src="dengo_wasm.js">`, or `require("./dengo_wasm.js")`
   *   in Node -- the same compiled file works either way).
   * @returns {Promise<DengoSolver>}
   */
  static async create(moduleFactory) {
    const mod = await moduleFactory();
    return new DengoSolver(mod);
  }

  /** Number of species in this network's state vector (including `ge`). */
  get nspecies() {
    // Derived from speciesNames.length rather than calling
    // dengo_wasm_nspecies() again -- the two are guaranteed identical
    // by construction (same species_names() string this was parsed
    // from), so a second FFI round-trip would just be redundant.
    return this.speciesNames.length;
  }

  /** Every species' current value, keyed by name (a fresh plain object
   * each call -- see bulkUpdate() for a way to avoid that allocation
   * when updating many species at once). */
  readState() {
    const heap = this._mod.HEAPF64;
    const ptr = this._statePtr() >> 3;
    const out = {};
    for (const name of this.speciesNames) out[name] = heap[ptr + this._index[name]];
    return out;
  }

  /** Overwrite one or more species by name; any species not present in
   * `values` is left untouched. */
  writeState(values) {
    const heap = this._mod.HEAPF64;
    const ptr = this._statePtr() >> 3;
    for (const name of Object.keys(values)) {
      if (!(name in this._index)) throw new Error(`DengoSolver: unknown species "${name}"`);
      heap[ptr + this._index[name]] = values[name];
    }
  }

  /** One species' current value. */
  get(name) {
    if (!(name in this._index)) throw new Error(`DengoSolver: unknown species "${name}"`);
    return this._mod.HEAPF64[(this._statePtr() >> 3) + this._index[name]];
  }

  /** Set one species' value. */
  set(name, value) {
    if (!(name in this._index)) throw new Error(`DengoSolver: unknown species "${name}"`);
    this._mod.HEAPF64[(this._statePtr() >> 3) + this._index[name]] = value;
  }

  /**
   * Escape hatch for bulk/vectorized state edits (e.g. scaling every
   * species by a density ratio in one pass, the way a free-fall step
   * does) without readState()/writeState()'s per-call object
   * allocation. `fn` gets a freshly read heap/pointer/index every call
   * -- never a cached one (see the module doc comment on why that
   * matters with ALLOW_MEMORY_GROWTH).
   * @param {(heap: Float64Array, ptr: number, index: Record<string, number>) => void} fn
   */
  bulkUpdate(fn) {
    fn(this._mod.HEAPF64, this._statePtr() >> 3, this._index);
  }

  /**
   * d(state)/dt at the *current* state, without advancing anything --
   * keyed by species name, same shape as readState(). Also refreshes
   * temperature()'s cache as a side effect (see the module doc comment).
   */
  rhs() {
    const heap = this._mod.HEAPF64;
    const ptr = this._rhsPtr() >> 3;
    const out = {};
    for (const name of this.speciesNames) out[name] = heap[ptr + this._index[name]];
    return out;
  }

  /**
   * Cached temperature reading, in Kelvin -- only current as of the
   * last rhs() or step() call (see the module doc comment); reading it
   * right after writeState()/set() alone returns a stale value.
   */
  temperature() {
    return this._temperatureFn();
  }

  /**
   * Evolve the state forward by `dtf` seconds, in place. Returns
   * whether it converged (reached dtf) -- on failure the state buffer
   * is left as the underlying BE_chem_solve last left it (its own
   * internal retry/bisection, not this wrapper's), not necessarily
   * unchanged from before the call.
   */
  step(dtf, niter = 200, reltol = 1e-5) {
    return !!this._step(dtf, niter, reltol);
  }
}
