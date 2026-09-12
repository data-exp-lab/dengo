# "Build your own network" -- generic kinetics prototype

**Status: prototype, on branch `wasm-generic-kinetics-prototype`, not
merged.** See NOTES.md's 2026-09-11 entry for the full writeup
(motivation, design, the two real bugs hit building it, and what was
verified).

A second, deliberately different path alongside the three fixed
fiducial networks (`fiducial_networks.py`) `generate_site.py` already
builds: instead of a Python-authored network compiled once via
Emscripten, `generic/index.html` lets you check which species and
reactions to include from a full catalog (today: everything
`build_primordial()` knows), with **no compile step for whatever you
pick**.

## Why this needed no new compiler at all

Ordinary mass-action kinetics -- `rate(T) * product of reactant
densities`, accumulated per species by its net stoichiometric change --
is the same formula for every reaction. It doesn't need per-reaction
generated code (sympy's role in the production path is mostly a
compile-time fusion/CSE optimization, not something semantically
required); it needs one hand-written assembler plus a data table. So:

- `wasm/generic_solver/dengo_generic.cpp` -- a network-agnostic build of
  the stiff Newton solver, vendoring `src/dengo/solvers/BE_chem_solve.C`
  **unmodified**. Compiled once, ever; never rebuilt when the selection
  changes. `dengo_generic_step()`'s adaptive-substep policy mirrors the
  production `dengo_wasm_step()` (`dengo_wasm.cpp.template`) exactly,
  generalized from a compile-time `NSPECIES` to a runtime `nchem`.
- `wasm/generate_reaction_db.py` -- exports every species/reaction in
  `build_primordial()` (name, stoichiometry, a `rate(T)` table) as one
  JSON file. No new chemistry authored -- reuses the network's own
  already-registered reactions and rate functions directly.
- `wasm/generic_kinetics.js` -- the generic RHS/Jacobian assembler,
  driven by that JSON plus whatever subset is currently checked.
  Registers its RHS/Jacobian as real C function pointers via
  Emscripten's `Module.addFunction()` -- `BE_chem_solve.C` always took
  `rhs_f`/`jac_f` as plain function pointers and never cared how they
  were implemented, so a JS closure is just as valid a callback as
  compiled C.
- `wasm/generic_ui.js` / `generic/index.html` -- the checkbox UI. A
  reaction or cooling action is only selectable once every species it
  touches is checked (mirrors `ChemicalNetwork.add_reaction`/
  `add_cooling(auto_add=False)`'s own validation).

Cooling *isn't* one universal formula the way reactions are -- each
action is its own bespoke sympy expression -- so it needed one more
piece: `export_cooling_action()` (generate_reaction_db.py) lowers each
action's equation *once*, via sympy's own `jscode` printer, into a JS
expression string embedded in the exported JSON; `generic_kinetics.js`
turns each into a real callable via `new Function()` at load time.
Still no per-*selection* codegen (every action in the catalog is
lowered regardless of what's later checked) and no compile step either
way. Checking at least one cooling action switches the run from fixed-T
to tracking `ge` (specific internal energy) as a real ODE variable,
with T derived from it every step -- see NOTES.md's 2026-09-11
"continued" entry for the full design (ge<->T conversion, why 2 of the
17 primordial cooling actions can't be exported this way, and the
Jacobian's finite-difference `ge` row/column).

## Explicitly out of scope right now

- **Cooling uses a single constant gamma (5/3, monatomic ideal gas)**
  for the ge<->T conversion, not the compiled solver's own T-dependent
  interpolated gamma for H2-bearing gas. A real simplification (H2-heavy
  gas's heat capacity reads a bit off), not just a labeling one.
- **Compton cooling always runs at z=0** (matches this project's
  existing compiled widget's own convention) -- no live redshift.
- **No CHIANTI/UMIST/photoionization reactions yet** -- deliberately
  deferred to a separate project. The reaction-database format already
  accommodates CHIANTI's ion-by-ion rates with zero changes (they're
  T-indexed, like every `primordial_rates.py` rate function); UMIST/
  photoionization rates are z-/redshift-indexed instead, which is the
  one real format gap that project would need to solve (a `T_index`
  field already sits in the exported JSON as a hook for this, doing
  nothing yet).

## Trying it locally

Same build command as the rest of the site (`generate_site.py`'s
`main()` calls `build_generic_page()` automatically, after the three
fiducial networks):

```sh
uv run python wasm/generate_site.py /some/output/dir
python3 -m http.server -d /some/output/dir 8080
# then open http://localhost:8080/generic/
```

## Correctness check

`wasm/generic_kinetics.js`'s RHS assembler was cross-checked directly
against the existing, already-validated compiled `hydrogen_minimal`
module (same initial conditions, same T) -- see NOTES.md for the exact
numbers. It matches to within an interpolation-scheme difference
(this prototype's rate tables are linearly interpolated against a
log-spaced T grid; the compiled solver interpolates in log-T-uniform-
bin space), not a stoichiometry or rate-law bug -- confirmed by
re-running at full (non-downsampled) table resolution, where the
residual shrank by the expected amount rather than staying fixed.

Cooling was checked with a physically unambiguous sign test rather than
a numeric cross-check (the compiled widget doesn't expose a fixed-
composition, no-chemistry cooling-only mode to compare against
directly): primordial gas at T=1e5 K with H2/H- species unchecked (no
formation-heating channel available at all) and only atomic cooling
actions checked (collisional excitation/ionization, radiative
recombination, bremsstrahlung, Compton) cools from 1.000e5 K to
6.209e3 K over the run -- a large, correctly-signed net cooling. See
NOTES.md for the full result including the (much smaller, also
consistent) default-conditions case.
