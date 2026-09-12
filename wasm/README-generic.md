# "Build your own network" -- generic kinetics prototype

**Status: prototype, on branch `wasm-generic-kinetics-prototype`, not
merged.** See NOTES.md's 2026-09-11 entry for the full writeup
(motivation, design, the two real bugs hit building it, and what was
verified).

Two more paths alongside the three fixed fiducial networks
(`fiducial_networks.py`) `generate_site.py` already builds and compiles
once via Emscripten:

- `generic/index.html` -- check which species/reactions/cooling actions
  to include from a full catalog (today: everything `build_primordial()`
  knows), with **no compile step for whatever you pick**.
- `generic/construct.html` -- write your *own* dengo `Species`/
  `Reaction` definitions, live, one reaction per editor box, not
  limited to the catalog at all. See "A third tool" below.

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

## A third tool: construct.html

`generic/index.html` still only ever picks a subset of one fixed,
pre-baked catalog. `construct.html` is for networks that aren't in any
catalog at all: one CodeMirror-edited Python box per reaction (`+ Add
reaction` for more), each defining real dengo `Species`/`Reaction`
objects, run through the *exact same* engine (`generic_kinetics.js`,
`dengo_generic.js`/`.wasm`) the checkbox tool uses -- the engine never
changes, only where its `{species, reactions}` data comes from: a
static JSON there, whatever a user's own Python just constructed here.

- **The real dengo package**, not a reimplementation and not a hand-
  picked subset of files: `build_dengo_wheel()` (generate_site.py) runs
  `uv build --wheel` (a pure filesystem operation -- dengo isn't on
  PyPI, and this project isn't ready for that) and serves the resulting
  wheel as a static asset; `construct_ui.js`'s `bootstrapDengo()`
  installs it into Pyodide via `micropip.install(url, {deps: false})`
  (dengo's declared dependencies -- h5py/cython/setuptools -- are
  neither available nor needed just to construct Species/Reaction
  objects) and stubs a bare `h5py` module first (reaction_classes.py
  imports it unconditionally at module level but never actually calls
  it for anything this page exercises).
- Each reaction box is self-contained: define `Species`, define a
  `rate(state)` function (`state.T` is available, same convention as
  every real dengo rate function), call `Reaction(name, rate, left,
  right)`. After "Build network", every registered reaction's rate is
  tabulated over a generic T grid and handed to the same
  `loadReactionDb()`/`runGenericIntegration()` the checkbox tool calls.
- No cooling/thermal coupling in this tool (yet) -- see the checkbox
  tool for that.

See NOTES.md's second 2026-09-11 entry for the full writeup, including
a real, general bug this tool's own "boring" test case surfaced in the
shared Jacobian (a `0/0` when a reactant's density is exactly zero --
latent in the checkbox tool too, just never triggered there).

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
# checkbox tool: http://localhost:8080/generic/
# construct-with-Python tool: http://localhost:8080/generic/construct.html
```

(`build_generic_page()` invokes `uv build --wheel` as part of this --
requires `uv` on `PATH`, same as everything else in this repo's own
tooling; no network/PyPI contact happens during the build itself, only
later, in the browser, when Pyodide's own CDN assets and the wheel this
build just produced are fetched.)

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
