# Grackle API compatibility: current state and what a C-level drop-in would need

Written 2026-09-09, in response to "are we Grackle API compatible, so
I can drop in the generated solver where it needs to be for simulation
codes." Pure investigation -- read the real Grackle C headers (a clone
at `.grackle/`, gitignored, see below), `src/dengo/grackle_compat.py`,
and dengo's own generated C template (`src/dengo/templates/cython_solver/
cython_solver.C.template`, `src/dengo/BE_chem_solve.C`) directly rather
than going from memory. No code changed to produce this document.

## Short answer

**Python-level: yes, for a real but narrower physics scope, and it's
validated.** `dengo.grackle_compat` (`src/dengo/grackle_compat.py`)
gives `chemistry_data`/`FluidContainer`/`solve_chemistry()`/
`calculate_temperature()`/etc. matching gracklepy's (pygrackle's) own
object model closely enough that code written against gracklepy can run
against dengo instead, for Grackle's `primordial_chemistry=2`-equivalent
physics (H/H+/He/He+/He++/H-/H2/H2+/e-, no metal cooling, no UV
background, no dust, no radiative transfer, no D/D+/HD). Anything
outside that raises `GrackleCompatError` at `initialize()` rather than
silently computing wrong physics. Validated directly against real
`gracklepy` (`.grackle_compare/validate_grackle_compat.py`): temperature
agrees to ~1%, gamma to 4+ significant figures. Re-confirmed working
today: `uv run pytest tests/test_grackle_compat.py` -- 15/15 pass.

**C-level: no.** If "simulation codes" means a compiled C/C++/Fortran
HPC/AMR code (Enzo, Enzo-E, etc.) linking `libgrackle.so`/`.a` directly
-- there is currently no adapter that would let dengo's generated
solver be called through Grackle's actual C API. Not a missing config
flag; three independent structural mismatches, each confirmed directly
against the real Grackle source and dengo's real generated template:

1. **No shared function names/signatures.** Grackle's public entry
   points (`.grackle/src/include/grackle.h`): `solve_chemistry(code_units*,
   grackle_field_data*, double dt)`, `local_solve_chemistry(...)`,
   `calculate_cooling_time/temperature/gamma/pressure(...)`,
   `initialize_chemistry_data(...)`. Dengo generates
   `{{solver_name}}_setup_data(...)`, `calculate_rhs_{{solver_name}}(...)`,
   `calculate_jacobian_{{solver_name}}(...)`, `BE_chem_solve(...)` --
   entirely different names and signatures, zero overlap.
2. **Incompatible memory layout** -- see the dedicated section below,
   this is the one asked about specifically.
3. **No grid geometry / ghost-zone support.** Grackle's
   `grackle_field_data` (`.grackle/src/include/grackle_types.h`) carries
   `grid_rank`/`grid_dimension`/`grid_start`/`grid_end`, so a hydro/AMR
   code hands over a padded array (with ghost zones) and Grackle only
   touches the interior `[grid_start, grid_end]` range. Dengo's
   generated loops are a flat `for (int i = 0; i < nstrip; i++)` over an
   unpadded array (confirmed in every hot function --
   `{{solver_name}}_interpolate_rates`, `calculate_rhs_{{solver_name}}`,
   `calculate_jacobian_{{solver_name}}`, `ensure_electron_consistency`,
   and `BE_chem_solve` itself) -- no concept of a sub-range at all.
   Every real AMR code passes exactly this kind of padded/ghost-zoned
   array, so this isn't an edge case, it's the normal calling pattern.

There's also a large parameter-surface gap (real `chemistry_data` has
~40+ fields in `.grackle/src/include/grackle_chemistry_data.h` --
comoving coordinates, UV background, dust, self-shielding, radiative
transfer coupling) that the existing Python shim already correctly
rejects (`GrackleCompatError`) rather than pretending to support. A
C-level layer would inherit the same physics-scope limit, which is
correct, just worth stating plainly.

## The memory-layout question, specifically: difficulty estimate

Asked directly: how hard would it be to address Enzo (and Grackle
generally) using separate storage per species, rather than dengo's
current per-zone-contiguous storage?

**What each side actually does, confirmed by reading the source, not
assumed:**

- Grackle/Enzo: `grackle_field_data` is struct-of-arrays with one
  *separate pointer per species* -- `HI_density`, `HII_density`,
  `H2I_density`, ... each its own flat array spanning every cell. Enzo's
  own `BaryonField[num_fields]` is the same shape (Grackle was
  co-developed against Enzo's field layout).
- Dengo: `cython_solver.C.template` packs *all species for one cell
  contiguously* -- `int j = i * nchem;` then each species read/written
  at `input[j]`, `j++` per species, in `calculate_rhs`/
  `calculate_jacobian`/`{{solver_name}}_calculate_temperature`. The
  header comment in `BE_chem_solve.C` states this convention
  explicitly: "`u[nstrip*nchem]`, with the `nchem` variables in a given
  cell stored contiguously."

**The one fact that matters most for scoping this: the actual chemistry
solve has zero cross-cell coupling already, regardless of layout.**
Confirmed directly in `BE_chem_solve.C`: the Newton linear solve is
`Gauss_Elim(&(Ju[ix*nchem*nchem]), s, &(gu[ioff]), nchem)` inside a
`for (ix = 0; ix < nstrip; ix++)` loop -- a *separate, independent*
`nchem x nchem` linear system solved per cell, every single sweep.
There is no term anywhere that couples cell `ix`'s chemistry to cell
`ix+1`'s (physically correct -- chemistry has no spatial coupling).
That means the per-cell-contiguous memory layout is a pure bookkeeping/
indexing convention sitting on top of an algorithm that would run
identically no matter how the same numbers were arranged in memory.
This is the load-bearing fact behind the estimate below: it turns "does
the layout mismatch require touching the algorithm" into a clean no.

### Two ways to actually close this, at very different costs

**(A) Boundary-marshaling adapter -- recommended.** Add a thin C
(Jinja-templated, like everything else dengo generates) function that:
takes Grackle-shaped separate per-species pointers plus
`grid_start`/`grid_end`, copies the interior sub-range of each species'
array into a scratch AoS buffer (dengo's existing packed layout), calls
the *existing, already-optimized, already-tested*
`calculate_rhs`/`BE_chem_solve` exactly as today with zero changes to
either, then copies the results back out into the separate per-species
arrays. Ghost-zone handling (mismatch #3 above) falls out of the same
copy loop for free -- it's just copying a sub-range instead of the
whole array, not a separate problem.

- *Why this is low-risk*: doesn't touch the numerically-verified,
  already-cache-stride-audited internals at all (see NOTES.md's
  2026-09-09 code-level audit entry, which found and fixed a real
  wrong-loop-order Jacobian bug in exactly this hot path -- that's
  the risk category a native rewrite would reopen; a boundary adapter
  can't, because it never touches that code).
- *Why the performance cost is negligible*: a transpose of `nstrip *
  nchem` doubles is a handful of nanoseconds per cell (memory-bandwidth-
  bound, moving maybe 80 bytes/cell); the chemistry solve itself
  measured 28.9-588 us/cell in this project's own C-level benchmarking
  (`.grackle_compare/c_bench/`, see NOTES.md) -- four to five orders of
  magnitude more expensive. The marshaling step would not be
  measurable against that, at any realistic grid size.
- *Why it's cheap to build*: this is more of the same kind of
  species-list-driven Jinja templating dengo's codegen already does
  everywhere (`network.required_species | sort`) -- a copy-in/copy-out
  loop generated the same way `dengo_wasm.cpp.template`'s own
  marshaling already is, not a new kind of code for this project.
- **Estimate: multi-day, not multi-week** -- comparable in scope to the
  existing Python-level `grackle_compat` build (NOTES.md called that
  "moderate, well-scoped... mostly glue + unit-conversion code"; this is
  the same shape of problem one level down, in C instead of Python).

**(B) Native SoA rewrite of the internal templates.** Change
`calculate_rhs`/`calculate_jacobian`/`{{solver_name}}_calculate_temperature`/
`{{solver_name}}_interpolate_rates`/`ensure_electron_consistency`/
`BE_chem_solve` themselves to index per-species-separate-array style,
so there's no transpose anywhere, ever.

- *Why this is higher-risk*: touches the exact hot loops a recent audit
  already found one real cache-stride bug in -- re-deriving every
  loop's indexing here reopens that entire risk category, this time
  across every one of those functions rather than the two the audit
  found. `BE_chem_solve.C` is also hand-written and shared across every
  generated network (not templated), so committing to one layout there
  means touching one file everyone depends on, not something scoped per
  network.
- *Why it's a bigger decision than it sounds*: dengo's Cython bindings,
  the wasm shim (`dengo-solver.js`/`dengo_wasm.cpp.template`), the
  example scripts, and the test suite all currently assume the packed
  layout -- committing the *internal* representation to SoA means
  either updating all of those too, or generating and maintaining two
  parallel code paths forever (against this project's own "minimize
  code reuse/duplication" stance, applied elsewhere this session to
  consolidating near-duplicate species lists).
- *Why the performance case for doing it is weak*: (A) already makes
  the marshaling cost immeasurable against the actual solve cost, so
  there's no real performance win being left on the table by choosing
  the adapter instead.
- **Estimate: weeks, not days**, and probably not worth it unless a real
  profiling result someday shows the marshaling step actually mattering
  -- which, given the numbers above, is unlikely at any realistic grid
  size.

### Recommendation

(A). It closes the actual gap (a real HPC code can hand over its own
separate-per-species field arrays, ghost zones included) without
touching the parts of this codebase that are already validated,
already performance-audited, and already used by every other consumer
(Cython, wasm, examples, tests). (B) is a real option only if (A) is
someday shown to be a measured bottleneck, which the numbers already in
hand make unlikely.

## What this document is not

This is a scoping/estimation document, not a design doc for (A) or a
commitment to build it -- no code was written to produce it, per the
explicit ask ("put it in a new file" was about the investigation, not
an instruction to build the adapter). See `NOTES.md`'s 2026-09-09
entries for the fuller history this was drawn from (the original
head-to-head physics comparison, the Python-level `grackle_compat`
build and its validation against real `gracklepy`, and the pure-C++
performance benchmarking against `libgrackle.so` that this document's
"negligible marshaling cost" claim is grounded in).
