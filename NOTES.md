# Dengo modernization notes

Append-only running log of what's been done, why, and what's still open.
Newest entries at the bottom. See `.claude`-style plan for the full phased
plan this follows (Phase 0-4 as agreed with the user).

Ground rules for this effort: no git commands/PRs, everything driven by
`uv`, stay inside this repo directory, `ion_by_ion.py`/`umist_rates.py`/
`get_rates.py`/RATE12.txt/CHIANTI tables are intentionally left untouched
for now (out of scope, not because they're unimportant).

---

## 2026-09-08 — Kickoff

Explored the existing repo. Key findings that shape the plan:

- Packaging is `numpy.distutils` + a bare `setup.py`, Python2/3-transitional,
  vendors `distribute-0.6.14`/`cvode-3.1.0.tar.gz` tarballs, has `.hg*`
  Mercurial leftovers and a `.travis.yml`. None of it installs cleanly
  today.
- 7 solver "backends" under `dengo/templates/`: `cv`, `cv_omp`,
  `cvode_omp`, `cvode_cuda`, `cuda-accelerInt`, `be_chem_solve`, `jax`.
  All but `be_chem_solve` require SUNDIALS/CVODE, which isn't installed
  here (or trivially installable via `uv`, which is a Python package
  manager, not a system one). `be_chem_solve` is the one backend that is
  actually self-contained: `dengo/solvers/BE_chem_solve.C` is a solid,
  generic, strip-based (many cells at once) implicit backward-Euler
  Newton solver with dense Gaussian elimination and adaptive sub-stepping
  on Newton failure — no external numerical library needed. This becomes
  the basis for the new default backend (see plan: "shared codegen,
  pluggable backends").
- `dengo/chemical_network.py`'s codegen (`species_total`,
  `print_jacobian_component`, `temperature_calculation`,
  `get_sparse_matrix_component`, HDF5 rate-table writing) is solid and
  reused as-is; it's the templates and the build glue around them that
  need replacing.
- `write_solver()` currently hard-*requires* `CVODE_PATH` even when asked
  for the CVODE-free `be_chem_solve` backend (a real bug, not just
  legacy cruft) — one of several reasons the "working" backend doesn't
  actually run today either.
- The live `be_chem_solve` `.pyx` template targets an Enzo-grid-coupled
  interface (`dengo_field_data`, `code_units`) and still carries a ~230
  line dead Python-2 (`print "..."`) code block as a stray triple-quoted
  string literal — exactly the kind of duplication/rot the cleanup phase
  targets.
- Confirmed on this machine: gcc 13.3, cmake, HDF5 headers via apt, `uv`
  0.12.6, no system Cython/numba/jax; numpy/scipy/sympy/jinja2 present.
  No SUNDIALS installed; `apt` has `libsundials-dev` candidate and
  conda/mamba is on PATH, but neither is being touched per the user's
  explicit "ask me first" instruction.

Plan approved by user (see decisions recorded at the top of this file's
intro). Proceeding with Phase 0.

## 2026-09-08 — Phase 0 done; Phase 1 backend design

Phase 0 complete:
- Package moved to `src/dengo/` (and `input/` -> `src/input/` alongside it,
  since `ion_by_ion.py`/`reaction_classes.py` locate it via
  `os.path.dirname(__file__)/../input` — moving both together means zero
  code changes needed to keep that untouched code working).
- Removed: `setup.py`, `dengo/setup.py`, `requirements.txt`,
  `distribute_setup.py`/`distribute-0.6.14*`, `cvode-3.1.0.tar.gz`,
  `dist/`, `build/`, `dengo.egg-info/`, `.hg*`, `.travis.yml`, the
  `install-*.sh` scripts (obsolete external-toolchain build infra, made
  obsolete by the no-HDF5/no-CVODE design below), stray
  `.ipynb_checkpoints/`.
- New `pyproject.toml` (hatchling backend), new `.gitignore`, new
  `README.md`. `uv sync` succeeds; Cython 3.3.0 resolved.
- Fixed two real bugs hit while just importing the package: a dead
  `import docutils.utils.roman` (unused, `docutils` isn't even a
  dependency) and an invalid regex escape `"\d+|\D+[a-z]|\D"` (now a raw
  string) in `reaction_classes.py`. `import dengo` now works clean.

Phase 1 backend design (`dengo/templates/cython_solver/`), replacing the
`be_chem_solve` backend rather than the CVODE ones (that one was the only
SUNDIALS-free backend to begin with):

- Read through `be_chem_solve`'s existing C template + `BE_chem_solve.C`.
  The numerical core (strip-based implicit backward-Euler/Newton with
  dense Gaussian elimination and adaptive sub-stepping in
  `dengo/solvers/BE_chem_solve.C`) is solid and kept verbatim. The rate
  table interpolation / temperature Newton-iteration / RHS / Jacobian
  codegen in the old `.C.template` is also correct, and reuses
  `chemical_network.py` methods (`print_ccode`, `print_jacobian_component`,
  `temperature_calculation`, `print_mass_density`,
  `cie_optical_depth_approx`) that need zero changes.
- What's being dropped as unnecessary complexity for a standalone
  package: the `NTHREADS`-indexed OpenMP arrays (single-threaded only —
  OpenMP batching can come back later behind the same interface if
  needed), the Enzo-coupled structs/functions (`dengo_field_data`,
  `code_units`, `*_solve_chemistry_enzo`, `flatten/reshape_*`), and
  redshift/photoionization table support (`z_bounds`, `pi`/`ph` reaction
  branches) — none of which the primordial network uses.
- **Bigger change: no HDF5 dependency in the generated solver at all.**
  This machine has no `libhdf5-dev` (no `hdf5.h`, confirmed via
  `pkg-config`/filesystem search), and requiring it would just reproduce
  the "can't build" problem this whole effort is meant to fix, for anyone
  without HDF5 dev headers. Rate/cooling/gamma tables are now written as
  one flat binary file (`{name}_tables.bin`, plain concatenated
  `double`s in a fixed, documented order) and read with plain `fread` —
  zero external C library dependency for the generated solver. `h5py`
  stays a dengo *Python*-side dependency (used for other bookkeeping),
  just not for the compiled solver.
- Per-cell cache arrays (`Ts`, `rs_*`, etc.) are now heap-allocated sized
  to the actual `nstrip` requested at setup time, instead of a hardcoded
  `MAX_NCELLS` silently truncating larger batches; a matching
  `{name}_free_data()` teardown is added (there wasn't one before).
- Build glue: no Makefile/pyxbld/pyximport magic. A small
  `dengo.solver_build.build_solver()` helper drives `Cython.Build.cythonize`
  + an in-process `setuptools` `build_ext` to produce an importable `.so`
  directly in the output directory — one Python call, works under `uv run`.

Writing the templates + build helper next, then the example script.

## 2026-09-08 — Phase 1: new backend working end-to-end, three real bugs found

Wrote the new backend (`dengo/templates/cython_solver/{.h,.C,_run.pyx}.template`,
`dengo/solver_build.py`, `ChemicalNetwork.write_cython_solver` +
`_write_solver_tables_bin`) as designed. Got it compiling and running for a
2-species (H/H+/e-) smoke network, then the full primordial H/He/H2
network with all cooling terms. Along the way, found and fixed three real
correctness bugs -- two inherited from the legacy code, one introduced by
me and caught immediately by testing:

1. **`calculate_rhs`'s `mdensity` was missing the `*weight` factor**
   (present) -- the legacy `be_chem_solve` template computed
   `mdensity` for the `ge` (gas energy) equation's normalization as a bare
   sum of number densities (no atomic-weight factor), while the
   *Jacobian* block for the same quantity correctly did
   `species*weight`. Since He (weight 4) and H2 (weight 2) are major
   species in this network, this under-weighted the mass density used to
   normalize the temperature-energy equation -- i.e. the gas-energy RHS
   itself was wrong by an abundance-dependent factor whenever He/H2 were
   present. Fixed by using `ChemicalNetwork.print_mass_density()`
   (the same helper `calculate_temperature` already used) uniformly in
   both `calculate_rhs` and `calculate_jacobian`, instead of three
   separate hand-rolled sums that had drifted out of sync -- and then
   actually multiplying by `mh` (grams/amu), which none of my first pass
   did either, since `print_mass_density()`'s formula is in amu/cm^3, not
   g/cm^3.
2. **`print_cooling`'s CIE optical-depth suppression only excluded the
   literal key `"h2formation"`, not `"h2formation_extra"`** -- both are
   direct chemical-heating terms (not radiative), so both need to be
   exempted from the continuum optical-depth factor that legitimately
   applies to radiative cooling lines. The code's own `# TODO: make it a
   more general check case?` comment shows this was a known gap. Fixed in
   `chemical_network.py`'s `print_cooling` (`term not in ["h2formation"]`
   -> `"h2formation" not in term`).
3. **`BE_chem_solve`'s Newton iteration normalizes by `1/scale[i]`, and
   the Python/Cython driver set `scale[i] = input[i]` verbatim.** Any
   species starting at exactly zero abundance -- which is the *normal*
   case for primordial initial conditions (no He+/He++/H2 yet) -- made
   `scale[i] = 0`, so `1/scale[i] = inf`, and the very first internal
   rescale (`u[i] *= inv_scaling[i]`) turned `0 * inf` into NaN, which
   then poisoned `data->Ts` permanently (it's a persistent cache, never
   reset). This existed in the legacy `be_chem_solve` driver too
   (`scale[j] = input[j]` in the same place) -- it just never got
   exercised because nothing exercised that code path. Fixed by flooring:
   `scale[i] = max(abs(input[i]), floor_value)` everywhere `scale` is
   set. This is exactly the kind of robustness the "solve collapse
   problems over many orders of magnitude" goal needs -- species
   legitimately spend a lot of that range at or near zero.

Also found while debugging (not yet fixed, noting for later): initial
attempts used the legacy `cdef np.ndarray[dtype, ndim=N]` buffer-typing
Cython syntax, which was suspected as a possible culprit on this
Python 3.14 + numpy 2.5 combination before the real bug (#3 above) was
isolated. It wasn't the actual cause here, but it's legacy/deprecated
style regardless, so the wrapper was rewritten to use plain typed
memoryviews (`double[::1]`) and ordinary Python-level numpy objects
instead -- simpler, more modern, and it removes a whole class of
numpy-C-API version-skew risk from the one part of this pipeline most
likely to be run against whatever Python/numpy a user already has.

Validated two ways:
- 2-species H/H+/e- network (isothermal, no cooling) vs. an independent
  `scipy.integrate.solve_ivp(method="BDF")` reference on the exact same
  rate functions: agreement to ~0.08% relative on the equilibrium
  ionization ratio after 1e6 s -- consistent with the expected first-order
  truncation error of the backward-Euler scheme against a high-order
  reference, not a red flag.
- Full primordial network (H, He, H2, all cooling channels) integrated
  from typical diffuse-gas initial abundances for 1e6 yr: converges,
  final T ~9828 K is physically sane for the input gas energy, H2
  fraction decreases (correctly dissociating at ~10^4 K), ionization
  fraction rises toward collisional equilibrium.

Next: turn these validated scripts into `tests/`, write the `examples/`
scripts, then Phase 3 cleanup + pluggable rate coefficients, then Phase 4
(physics review of the H2 formation-heating term + a real free-fall
collapse run in the 1500-2500 K / ~1e15 amu/cc target regime).

## 2026-09-08 — A serious silent-corruption bug, and a working free-fall validation

While chasing down why the free-fall collapse example ran but produced
unphysical monotonic heating (T shooting straight to the 1e8 K table
ceiling instead of cooling), found and fixed the most consequential bug
of this pass:

**Jinja's `dictsort`/`sort` filters default to `case_sensitive=False`,
but the Python-side table writer uses plain `sorted()`, which is
case-sensitive.** For any two table/reaction names whose case-sensitive
and case-insensitive alphabetical order disagree -- e.g. `"ciHI"` sorts
before `"ciHeI"` case-sensitively (`'I' < 'e'` in ASCII) but *after* it
case-insensitively (`'i' > 'e'`), same for `"gaHI"` vs `"gaHe"` -- the
generated C's `fread()` sequence in `_read_tables` silently read each
such table from the *wrong slot* in the binary file relative to what
`ChemicalNetwork._write_solver_tables_bin` wrote. No error, no crash --
just a rate coefficient quietly swapped for a different one, off by up
to tens of orders of magnitude, exactly the kind of bug that's invisible
in a smoke test and only shows up as "the physics looks wrong" in a real
run. Diagnosed by: byte-verifying the written file was correct (it was)
and only then comparing the generated C source's actual `fread` order,
name by name, against Python's `sorted()` order -- they diverged exactly
at `ciHI` and every case-sensitivity-dependent table after it.

Fixed by adding `dictsort(case_sensitive=true)` /
`sort(case_sensitive=true)` to every such filter in the `cython_solver`
templates, and added a regression test
(`tests/test_codegen.py::test_tables_bin_contents_match_write_order`)
that reads the table file back with the writer's own ordering and
independently re-evaluates every reaction/cooling function to check
against it, *and* textually checks the generated C's `fread` order
against Python's `sorted()` order -- so this can't silently regress.

With that fixed, `examples/free_fall_collapse.py` (plain free-fall, see
that file's docstring for why it's not the Omukai et al. 2005
pressure-retarded scheme) now produces a genuinely textbook-shaped
primordial collapse curve, run from n~1 to n~3e15 cm^-3:

- T dips from ~370 K to a minimum of ~250 K around n~1e4-1e5 cm^-3 (H2
  line cooling), matching the well-known Omukai/Yoshida-type minimum.
- T rises back up along the H2-cooling-saturated track through
  n~1e6-1e9, with a visible plateau/kink around n~1e9-1e10 cm^-3 where
  three-body H2 formation heating switches on (H2 fraction visibly jumps
  from ~1e-3 to ~0.5 in that same range in the companion plot).
- T continues rising to ~1500-2000 K by n~1e12-1e15 cm^-3, landing
  squarely in the 1500-2500 K target band at the ~1e15 amu/cc target
  density, then turns over slightly -- consistent with H2-line/CIE
  optical-depth suppression (`h2_optical_depth_approx`,
  `cie_optical_depth_approx`) starting to regulate cooling right where
  it should.

This is the single strongest piece of evidence so far that the
generated solver's physics -- including the H2 formation-heating term
this whole effort is centered on -- is quantitatively sane across the
full ~15-decade density range, not just correct at one test point.

Two bugs were fixed and then found to be non-issues / already-inherited,
worth recording so they aren't re-investigated:
- An earlier version of the free-fall driver used the exact Omukai et
  al. (2005) force-factor/effective-gamma scheme from the legacy
  reference. That reference had `include_pressure = False` hardcoded, so
  the scheme was never actually exercised in the original code either;
  reimplementing it faithfully surfaced that it's internally
  inconsistent (the retarded collapse-rate formula picks up a
  `sqrt(1 - force_factor)` term that the compressional-heating formula
  never did). Rather than debug/validate a scheme the original authors
  never ran, switched to plain (unimpeded) free-fall -- itself a
  standard test problem, and sufficient for what this pass cares about.
- Conflating that scheme's history-based `gamma_eff` (dlnP/dlnrho, only
  meaningful for the force factor) with the gas's own thermodynamic
  Gamma (needed for compressional heating) was a real bug along the way;
  fixed by computing them as clearly separate quantities
  (`thermodynamic_gamma()` in the example).

Existing-but-hidden feature worth documenting rather than re-building:
`ChemicalNetwork.threebody` (an int 0-5, default 4) already selects
between six different published fits for the three-body H2
formation/collisional-dissociation rates (k13, k22), matching the
convention Grackle uses for its own `three_body_rate` parameter -- see
`primordial_rates.py`'s `k13`/`k22` functions. This is effectively
already "choose from multiple rate-coefficient sources," just for one
specific physical process and via a bare integer with no naming. Did not
extend this to a fully general per-reaction mechanism in this pass
(would need literature citations for each of the 6 branches that aren't
reliably recoverable from the code comments alone, and a broader
mechanism is a bigger design task than remaining time allows) --
flagging as the concrete next step for the "pick a rate-coefficient
source" feature, rather than fabricating citations or a half-general
mechanism now.

## 2026-09-08 — Housekeeping, and where this stands

Found and fixed one more thing while doing a final clean-repo check:
`solver_build.build_solver()` was passing `--inplace` to `build_ext`
alongside `--build-lib`. For a flat (non-package) extension name,
distutils' in-place copy step resolves relative to the *caller's current
working directory*, not `build_lib`, regardless of where the sources
live -- so every build was correctly landing in `output_dir` (found via
the glob search right after) *and* silently dropping a duplicate `.so`
at whatever directory `uv run` was invoked from. Dropped `--inplace`;
`--build-lib` alone is sufficient. Cleaned up the stray `.so` files this
had left at the repo root.

Also did a light cleanup pass: `print()` calls in the hot path
(`add_reaction`, `add_cooling`, `species_gamma`) now go through
`logging` / raise proper exception messages instead.

### Where this stands against the original ask

- **Modernized, `uv`-based, examples added, latest Cython**: done.
  `pyproject.toml` + `uv sync` work cleanly; Cython 3.3.0 resolved;
  `examples/primordial_network.py` and `examples/free_fall_collapse.py`
  run end-to-end and are the two actively-maintained examples.
- **Simple unit tests**: done -- 62 tests in `tests/`, covering registry
  bookkeeping, rate-coefficient sanity, codegen (including the table
  write/read-order regression test), solver correctness against an
  independent `scipy` reference, mass/element conservation, and the
  analytic-vs-finite-difference Jacobian.
- **Define chemistry, pick from multiple rate-coefficient sources,
  generate a fast solver**: the definition + codegen + fast
  (SUNDIALS/HDF5-free, compiled) solver pipeline is done and validated.
  The "pick a source" part exists today only for the three-body H2 rates
  (`ChemicalNetwork.threebody`); generalizing it to an arbitrary
  per-reaction mechanism is the clearest remaining gap.
- **Clean, modern, minimized code reuse**: the new backend replaced 7
  overlapping solver-template backends with one (the other 6 archived,
  not deleted, under `dengo/templates/legacy/`); `chemical_network.py`'s
  cooling-normalization logic was consolidated onto one shared
  `print_mass_density()` call instead of three hand-rolled, drifted-apart
  copies. Not exhaustively done across the whole codebase (e.g.
  `chemical_network.py`'s older `write_solver`/`write_cuda_solver`
  methods, tied to the archived CVODE backends, weren't touched -- no
  reason to risk them since ion-by-ion/UMIST-adjacent legacy examples may
  still reference that path, and it was explicitly out of scope).
- **Optimization pass, 1500-2500 K / ~1e15 amu/cc, H2 formation heating
  (4.48 eV/molecule), many decades of density**: `examples/
  free_fall_collapse.py` runs a real free-fall collapse from n~1 to
  n~3e15 cm^-3 (>15 decades) without failing, and produces a physically
  correct-shaped temperature curve that lands in the 1500-2500 K target
  band at the target density -- the H2 formation-heating term
  (`h2formation`/`h2formation_extra` in `primordial_cooling.py`) and its
  high-density optical-depth suppression (`h2_optical_depth_approx`,
  `cie_optical_depth_approx`) are both exercised and doing something
  physically sane. What this pass did *not* do: a systematic,
  literature-referenced validation of the H2 formation-heating rate
  fits themselves against Omukai (2000) / Glover & Abel (2008) beyond
  what the existing rate fits already encode, or a partial-equilibrium/
  QSS treatment for fast species (H-, H2+, e-) to further reduce
  stiffness -- both flagged as good next steps, not done here.

Three serious, previously-silent bugs were found and fixed along the
way (see the dated entries above for each): the missing `*weight`/`*mh`
factors in the gas-energy mass-density normalization, the CIE
optical-depth suppression incorrectly applying to `h2formation_extra`,
the `1/scale` blow-up for exactly-zero initial abundances, and -- the
most consequential one -- the case-sensitive/case-insensitive sort
mismatch between the Python table writer and the generated C reader,
which silently corrupted cooling-rate lookups by tens of orders of
magnitude with no error at all. All four now have regression coverage
in `tests/`.

## 2026-09-08 — OpenMP parallelization, and a real benchmark

Added OpenMP back to the new backend (the legacy `be_chem_solve`/`cv_omp`
templates had per-cell `#pragma omp` before this whole rewrite; the new
backend didn't, until now). Every cell in a batch is independent -- no
cross-cell terms anywhere in this network -- so this is a straightforward
parallel-for over cells in: `calculate_rhs`, `calculate_jacobian`,
`calculate_temperature`, `interpolate_rates`, `ensure_electron_consistency`
(all in `cython_solver.C.template`), and `BE_chem_solve`'s per-cell
Newton-update loop (`dengo/solvers/BE_chem_solve.C`).

Two things had to change to make that safe, not just add `#pragma omp`:

1. **Every per-cell local variable (species values, `T`, `mdensity`, the
   optical-depth terms, the whole H2-gamma Newton-iteration scratch space)
   had to move from function-scope to inside the loop body.** They were
   all declared once above the loop and reassigned each iteration -- fine
   serially, an immediate, silent data race the moment the loop runs on
   multiple threads at once, since every thread would read/write the same
   memory. Cython/C++ locals declared inside a parallel-for loop body are
   automatically per-thread (own stack frame), which is the standard,
   structurally-enforced way to get this right, rather than enumerating
   every variable in a `private()` clause and hoping none get missed.
2. **A plain `return 1` isn't legal from inside a `#pragma omp parallel
   for` body** (the region has to run to completion across all threads).
   Every such early-return-on-failure (a negative species, a
   non-converging temperature Newton iteration, a singular Jacobian, a
   NaN) became "set a shared flag (a `+` reduction), skip the rest of
   this cell, check the flag once the loop finishes" instead. Also had to
   resize `BE_chem_solve`'s `s` (Newton update) scratch buffer from
   `nchem` to `nstrip*nchem` -- it used to be one shared buffer reused
   cell-by-cell in sequence, which is exactly the kind of thing that's
   invisible in serial code and wrong the instant it's parallel.

`solver_build.build_solver()` now passes `-fopenmp`, but tries that first
and *automatically falls back to a serial build* if it fails to compile
(stock Xcode clang on macOS, for instance, doesn't ship OpenMP) -- so
"just works with a C++ compiler and Cython, nothing exotic" still holds.

**The overhead of spawning a thread team is not free**, and for a single
cell (or a few) it swamped the actual work by ~300x when I first measured
it uniformly enabled -- exactly the case `examples/free_fall_collapse.py`
is in (one cell, thousands of sequential steps). Fixed with OpenMP's
`if()` clause: every parallel region is guarded with
`if (nstrip > DENGO_OMP_MIN_CELLS)` (2048, an empirical round number from
this machine, documented as such in the generated header), so small
batches automatically stay serial and large ones automatically go
parallel -- no user-facing knob needed, though `DENGO_OMP_MIN_CELLS` is
one `#define` to tune if 2048 is wrong for someone else's hardware/access
pattern.

**Benchmark, same 128^3-cell / one-Courant-step measurement as before**,
same machine (24-thread Xeon X5650, 2.67GHz, 2010-era, no AVX):

| | cells/sec | 128^3 wall time |
|---|---|---|
| before (serial) | ~4.8e4 | 44.1 s |
| after (OpenMP, default thread count) | ~1.5e5 | ~13.7-14.3 s |

**~3.1-3.2x speedup**, not the ~24x a naive "N threads -> N times faster"
expectation would suggest. Didn't chase this further (`OMP_PROC_BIND=close
OMP_PLACES=cores` made no measurable difference, so it's not obviously a
NUMA placement issue) -- plausible remaining causes, not verified: memory
bandwidth on this old 2-socket system, and/or the five separate
`#pragma omp parallel for` regions per Newton sweep each paying their own
(amortized-but-not-free) fork/join cost rather than one region doing all
five phases per cell before rejoining. That consolidation -- restructure
so a cell's full Newton-sweep work happens inside one parallel region
instead of five -- is the clear next step if more speedup is wanted;
didn't do it here. Confirmed small-batch calls (`nstrip` below the 2048
threshold) are bit-for-bit unaffected and `examples/free_fall_collapse.py`
reproduces the identical physics curve after this change.

## 2026-09-08 — First head-to-head against Grackle

Set up a real comparison against Grackle rather than just reading its
source. Constraint from the user: no system-wide HDF5 install. Two
different paths ended up being used for two different purposes:

- **For the comparison run itself**: `gracklepy` (Grackle's Python
  bindings, renamed from `pygrackle`) ships prebuilt manylinux wheels on
  PyPI with HDF5 *and* the compiled Grackle C library bundled inside the
  wheel (same trick h5py uses) -- no HDF5 needed at all. Only wrinkle:
  the wheels only go up to cp313, and this project's main env is on
  Python 3.14. Fixed by giving it its own isolated environment
  (`uv venv --python 3.12 .grackle_compare/.venv` + `uv pip install
  --python ...`) instead of touching the main project's env/dependencies
  -- `uv init` from inside a subdirectory auto-registers as a workspace
  member of the parent project's `pyproject.toml`, which is *not* what
  we want here (this is comparison tooling, not a dengo dependency), so
  that got created via plain `uv venv`/`uv pip install` instead, and the
  accidental workspace registration from a first, wrong attempt was
  reverted.
- **Investigated separately, not needed for the wheel path, but kept for
  reference**: `apt-get download libhdf5-dev` (+ the runtime packages
  `libhdf5-103-1t64`, `libhdf5-hl-100t64`) followed by `dpkg -x` into a
  local prefix (`.grackle_deps/`, gitignored) gets real, working
  HDF5 headers and libraries with zero system-wide installation --
  `apt-get download` and `dpkg -x` don't touch the system package
  database or write outside the target directory. Confirmed this
  actually works (compiled and ran a trivial HDF5 program against it).
  Would matter if building Grackle from source turns out to be needed
  later (e.g. for a Python 3.14-native build).

`grackle-project/grackle` was cloned to `.grackle/` (gitignored) for its
`grackle_data_files` submodule (Cloudy tables Grackle wants a path to
even when metal cooling is off) and to read its Python utilities'
source directly.

**The comparison** (`.grackle_compare/run_dengo.py` /
`run_grackle.py`, not yet committed -- see note below): the same
two-phase test problem through both codes -- start hot & ionized
(T=50000 K, n=0.1 cm^-3), cool at constant density down to 300 K, then
free-fall collapse up to n=3e15 cm^-3 -- using each code's own
Omukai et al. (2005) free-fall utility (dengo's own, and gracklepy's
built-in `evolve_freefall`/`evolve_constant_density`), same
`primordial_chemistry=2`-equivalent 9-species set, same `three_body_rate`
index (4) on both sides.

Results (single-cell, single-threaded, same old Xeon X5650):

|  | steps (cooldown+freefall) | wall time | final T | final H2 frac |
|---|---|---|---|---|
| dengo | 48 + 1929 | 1.68 s | 1492 K | 0.50 |
| grackle | 742 + 7273 | 6.39 s | 2083 K | 0.73 |

**The T(n) curves track each other closely across all ~15 decades of
density** -- same shape (initial heating bump, H2-cooling minimum at the
same density, three-body-formation reheating at the same density,
landing within ~500 K of each other right at the target regime). That's
a meaningful independent validation: two codes with different rate-fit
sources and different implementations agree well on the thing this whole
effort is centered on. dengo lands at the bottom edge of the 1500-2500 K
target band, Grackle mid-band -- a real, modest difference, not
alarming on its own.

**The H2 fraction curves do not agree well at low density**: dengo's H2
fraction is many orders of magnitude below Grackle's from n~0.1 up to
~1e9-1e10 cm^-3 (both converge to the same three-body-dominated ~0.5-0.7
by n~1e12+, where the T curves also converge most closely). This is a
real, unresolved discrepancy -- not yet root-caused. Candidates, none
confirmed: dengo's cooldown phase took 48 steps to Grackle's 742 (using
a cruder cooling-time estimate for the adaptive step, `ge/|dge/dt|`
instead of Grackle's own `calculate_cooling_time()`), which may be
under-resolving the H-/H2 formation channel during cooldown; a
difference in which literature fit each code uses for H- radiative
association or H2 formation despite matching `three_body_rate`; or an
initial-condition mismatch in `run_dengo.py`'s `ionized_ics()` (chosen
without cross-checking gracklepy's own `setup_fluid_container(state=
"ionized")` values). Also worth being honest about the performance
number: with a 15x-fewer-steps cooldown phase, dengo's ~3.8x speedup
here is not a clean apples-to-apples "faster at the same resolution"
result -- some of it may just be coarser stepping. Both of these are the
natural next things to dig into if this comparison continues.

Not yet done: `.grackle_compare/`'s comparison scripts (small, no heavy
deps) haven't been committed -- everything heavy (`.grackle/` clone,
`.grackle_compare/.venv/`, `.grackle_deps/`, per-run JSON output) is
gitignored, but the two `run_*.py` scripts themselves are real, reusable
comparison tooling worth keeping; left for the user to decide whether to
commit them as-is or reorganize first (e.g. into a proper top-level
`comparisons/` directory rather than a dot-prefixed one).

## 2026-09-08 — The H2-fraction "discrepancy" was a bug in the comparison script, not dengo

Followed up on the apparent low-density H2-fraction gap from the
previous entry. Tried tightening `run_primordial`'s `reltol` (1e-5 down
to 1e-11) for the cooldown phase first, since that was the most direct
hypothesis: **no effect at all**, to 6+ significant figures, at any
tolerance tested. This makes sense in hindsight and is itself worth
recording -- `BE_chem_solve`'s Newton iteration converges to the same
fixed point regardless of the requested tolerance (tightening it just
costs more sweeps to converge to that same point, it doesn't change
*what* it converges to), so tolerance was never going to explain a
16-order-of-magnitude gap, and didn't.

Investigating why tolerance didn't move anything led to the real bug:
`run_dengo.py`'s H2-fraction normalization computed `total_H` **once,
from the state at the very end of the free-fall phase** (after
collapsing to n=3e15 cm^-3), and used that single number to normalize
*every* point in the history, including the low-density ones from the
start of the run. Total-H *number density* isn't a conserved constant
you can compute once and reuse -- it scales up with compression just
like every other species' number density does (that's the whole point
of the free-fall test). Dividing an early, low-density H2 number density
by the *final*, ~10^16x-larger total-H number density made every early
point look ~10^16x smaller than it actually was. Grackle's own H2
fraction was already computed as a per-point NumPy array the whole
time, so this bug only existed on the dengo side.

Fixed by computing the H2 nuclei fraction from each state's own
densities at the point it's recorded, never carrying a value over from
another time (`h2_nuclei_fraction()` in `run_dengo.py`). With that
fixed, dengo and Grackle's H2 nuclei fraction curves now agree well
across the **entire ~15-decade density range**: both flat around
3-5e-3 through the low-density plateau, both turning up sharply at the
same density (~1e10 cm^-3, three-body formation onset), both landing
within ~4% of each other (0.9998 vs 0.96) at full collapse. There is no
remaining H2-fraction discrepancy to explain -- what looked like one was
entirely this bug.

Net result: two independent, previously-reported findings from the last
entry are now retracted/corrected (the ~1e22-year cooldown time was a
separate unit-conversion bug in the *time* diagnostic, already fixed;
this H2-fraction gap was this normalization bug) and nothing about
dengo's own chemistry/solver needed to change. Updated
`.grackle_compare/dengo_vs_grackle.png` shows both temperature and H2
fraction tracking closely across all ~15 decades of density now.

## 2026-09-08 — High-density timing breakdown, and why Grackle's curve looks noisier

Instrumented both scripts with actual per-step wall-clock timestamps
(`time.perf_counter()` around just the chemistry-solve call, not the
surrounding Python bookkeeping) to answer two questions honestly instead
of by eyeballing the earlier plot.

**The earlier "1.7s dengo vs 6.4s Grackle" total-time comparison was
measuring different things.** Summing Grackle's own per-step
`fc.solve_chemistry(dt)` calls comes to only **0.35s** total -- the
other ~5.5s of Grackle's reported free-fall time is Python-level
bookkeeping in its `evolve_freefall` *example utility* (`add_to_data`
copies every one of ~20-30 FluidContainer fields into a growing list,
7273 times), not the C library's actual solve cost. Dengo's own driver
loop has far less per-step bookkeeping, so its reported time (1.60s) and
its summed per-step solve time (1.49s) are close to each other already.
Comparing solve-only cost: Grackle's C library is *much* faster per call
than dengo's generated solver (4.8e-5s/step vs 7.7e-4s/step, ~16x).

**Why**: profiled a fixed, trivial dengo workload (single cell, niter=1,
even a bare `evaluate_rhs` with no time-stepping at all) and got
~1.4-2.0e-4s per call -- i.e. **most of that 16x gap is fixed per-call
overhead, not the actual chemistry math**. Every single
`run_primordial`/`evaluate_rhs`/`evaluate_temperature` call does a fresh
`{name}_setup_data()` (malloc every per-cell array) + `{name}_read_tables()`
(fread the whole rate-table file back off disk) + `{name}_free_data()`,
even when called with a single cell and one substep. That's the right
tradeoff for the one-shot "hand me `dtf` and let it run to completion
internally" use this API was designed for, but it's exactly the wrong
one for a driver that needs to interleave dengo with external state
updates (rescale densities for compression, then take one more step,
repeat -- thousands of times) the way both this free-fall script and any
real hydro coupling would. Grackle's `chemistry_data`/`FluidContainer`
are set up once and `solve_chemistry(dt)` is cheap precisely because it
skips all of that on every call. **This is a real, actionable gap**: a
persistent-handle API (`setup()` once, `step(handle, state, dt)` many
times, explicit teardown) would let dengo's per-step cost drop close to
its actual compute cost instead of being swamped by setup/tables I/O --
not done here, flagging as the clearest concrete next step if
performance at this calling pattern matters.

**Segmenting by density** (pure solve time, not the inflated Grackle
total): from n=1e10 up to the 3e15 target, dengo takes 659 (larger)
steps in 0.60s; Grackle takes 2434 (smaller) steps in 0.18s. Dengo needs
~3.7x fewer steps to cover the same range, but its ~16x per-step
overhead more than eats that advantage in this specific high-density
segment.

**Is Grackle's visible noise a resolution artifact -- i.e. does dengo's
smooth curve cost something?** No, and the evidence points the other
way: Grackle actually takes *more* steps than dengo through the noisy
region (871 vs 229 steps in the 1e13-1e15 range used for this check),
and its per-step dt there is monotonically decreasing, not itself
jittery. Quantified the noise directly (RMS deviation from a rolling
local median): Grackle ~0.85%, dengo ~0.35% -- both small in absolute
terms, Grackle about 2.4x noisier. The likely explanation is a modeling
difference, not a numerics one: Grackle's `evolve_freefall` uses the
pressure-retarded Omukai et al. (2005) scheme by default
(`include_pressure=True`), which estimates the local effective adiabatic
index from a finite difference of the last few (pressure, density)
points -- an estimator that's inherently sensitive to small step-to-step
fluctuations. Dengo's free-fall script deliberately does *not* use that
scheme (see the earlier NOTES.md entry: the reference implementation it
was adapted from never actually exercised it, and reimplementing it
faithfully surfaced an internal inconsistency) -- it follows the exact
analytic unimpeded free-fall solution, which is smooth by construction
because there's no finite-difference estimator in the loop at all. So
dengo's smoothness isn't bought with a coarser or less accurate solve --
it's a simpler (and, not incidentally, unvalidated-in-Grackle-either)
collapse-dynamics model choice, not a resolution/accuracy tradeoff.

## 2026-09-08 — Persistent Solver handle (the calling-pattern fix), and confirming the free-fall noise source

**Persistent handle.** Added a `Solver` class to the generated `_run.pyx`
(`dengo/templates/cython_solver/cython_solver_run.pyx.template`):
`{name}_setup_data()`/table read happens once in `__cinit__`, all of
BE_chem_solve's scratch buffers are allocated once, `step()` reuses all
of it, `close()`/`__dealloc__`/context-manager protocol tear it down.
`run_{name}()`/`evaluate_rhs()`/`evaluate_temperature()`/
`evaluate_jacobian()` are unchanged in signature and behavior -- they're
now thin wrappers that create a temporary `Solver`, call the matching
method, and close it, so nothing that already used the one-shot API
needed to change (all 62 tests still pass unmodified).

Clean before/after (same trivial single-cell workload, isolated from any
free-fall bookkeeping):

| | one-shot API | persistent `Solver` | speedup |
|---|---|---|---|
| `step`/`run_primordial`-equivalent | 2.05e-4 s/call | 6.0e-5 s/call | ~3.4x |
| `evaluate_rhs` | 1.46e-4 s/call | 4.7e-6 s/call | ~31x |

`evaluate_rhs`'s ~31x is the cleanest evidence for what was actually
going on: that call does *no* time-stepping at all, so in the one-shot
version essentially its entire cost *was* setup+table-read+teardown.

At grid scale (100,000 cells, 20 repeated calls to the same handle --
the actual "call chemistry once per hydro step" pattern): 0.634s/step
(one-shot) vs 0.514s/step (persistent), **~19%** faster. Smaller than the
single-cell case because the fixed per-call overhead doesn't scale with
`nstrip` -- at 100k cells it's already a small fraction of a call
dominated by real O(nstrip) chemistry work. The free-fall comparison
script (`run_dengo.py`, now using `mod.Solver(1)` for both phases)
improved similarly modestly (~10-15%) for the same reason: BE_chem_solve
often needs several Newton sweeps per step there, so the now-eliminated
setup/table-read cost was a smaller fraction of an already-real-work-heavy
call than in the trivial micro-benchmark.

**Confirmed, empirically, why Grackle's curve is noisier and dengo's
isn't** (the user asked whether this means dengo would give a "better,
smoother" result if coupled into a real hydro code -- it wouldn't, and
here's the direct evidence): took dengo's own T(n)/n(t) trajectory from
the free-fall run (independently measured at only ~0.35% RMS noise) and
ran it through Grackle's *exact* finite-difference
gamma_eff-then-force_factor estimator (`calculate_collapse_factor`,
copied verbatim). The resulting force_factor has a **13% standard
deviation** and ~4.6% RMS deviation from a local median -- genuinely
jittery, computed from a smooth input. So the jitter is a property of
that specific *test-script* estimator (finite-differencing a short,
inherently slightly-noisy history of (P, rho) pairs to guess a local
effective adiabatic index), not of which chemistry solver is driving it.

That estimator only exists because both free-fall scripts are a 0-D
stand-in for real hydrodynamics: neither dengo's plain free-fall formula
nor Grackle's pressure-retarded one is what a real hydro code would use
to get density(t) for a fluid element -- a real code computes that from
actual pressure gradients and gravity on a grid/mesh, and *that*
determines how smooth the simulation's own trajectory is, not which
chemistry module is plugged in to respond to it. Swap which chemistry
solver drives Grackle's own force-factor-based free-fall script and the
jitter would very likely persist (it's upstream of the chemistry call);
conversely, adding that same scheme to dengo's free-fall script would
very likely reproduce it too. So: dengo's smoother-looking curve here
isn't evidence it would give a smoother/better hydro-coupled simulation
than Grackle would -- the two free-fall scripts are testing the
chemistry solvers under two different *prescribed* compression models,
and the visible noise is a feature of one of those prescriptions, not of
either solver's numerics.

**2026-09-08, continued: does identifying the noise source answer "which
solver produces better solutions"? No -- correctly pushed back on by the
user.** That question needs two different experiments, both now done:

1. *Convergence check, each code against itself* (`grackle_convergence.py`
   in the scratchpad, mirroring the dengo convergence check from earlier
   today): Grackle's own pressure-retarded free-fall at
   safety_factor = 1e-2, 1e-3, 3e-4 gives final T = 2083.2, 2116.1,
   2165.5 K and H2 nuclei fraction = 0.9597, 0.9501, 0.9375 -- **not
   converged** at these step sizes; still drifting by several percent
   between the two finest step sizes, an order of magnitude apart. This
   contrasts with dengo's own convergence check (recorded earlier): T
   changes only ~0.6% over the same safety_factor range. So at their
   respective *default* step sizes, dengo's answer is the better-trusted
   one on its own terms -- Grackle's free-fall driver would need a
   noticeably smaller safety_factor than its 0.01 default to reach a
   comparable level of convergence.

2. *Decoupled comparison* (`run_decoupled.py`): drives dengo's `Solver`
   and Grackle's `FluidContainer` through the *exact same* prescribed
   plain free-fall compression (identical density_ratio, identical dt at
   every step, computed once and applied to both) -- removing the
   collapse-model confound entirely, so any remaining difference is
   attributable only to the chemistry/cooling solve itself. Two real
   bugs surfaced getting this running: (a) `fc["temperature"]` is never
   auto-computed by `solve_chemistry()` -- it has to be refreshed with an
   explicit `fc.calculate_temperature()` after every step, or it just
   reads back whatever was last written (zero, initially); (b)
   `fc.solve_chemistry(dt)` expects `dt` in Grackle's *code* time units
   (`time_units`, here `sec_per_Myr`), not raw CGS seconds -- passing
   dengo's cgs-second dt directly silently evolved Grackle by
   ~1/3.16e13 of the intended physical interval every step (visible as
   Grackle's temperature staying pinned near its 1 K floor). Fixed with
   `dt / my_chemistry.time_units`.

   Result (`decoupled_comparison.png`): the two solvers track within
   ~2-10% of each other across most of the ~15 decades of density, but
   **diverge to ~34% in final temperature** at the target density
   (n=3e15 cm^-3: dengo T=1492 K, Grackle T=2004 K), with a non-monotonic
   pattern -- a ~23% bump around n~1e13, narrowing to ~2-5% around
   n~1e14, then widening again toward the end. H2 nuclei fraction agrees
   much more closely throughout (both saturate to ~0.98-1.0 by the target
   density). This is a genuine, chemistry/cooling-attributable
   difference -- not a collapse-model artifact, since both codes were
   driven by literally the same compression history here. Also notable:
   small jaggedness is still visible in Grackle's curve even in this
   decoupled test (where the noisy force-factor estimator is never
   invoked), confirming the earlier finding's caveat that some jitter
   also comes from Grackle's own internal chemistry substep/step-doubling
   logic, not solely from the force-factor estimator.

   This still isn't an absolute ground truth (neither curve is a known-
   correct reference) -- pinning down *which* solver's ~1500-2500 K
   answer at n~1e15 is more accurate would need comparing both against a
   solve at much tighter tolerance/step size than either uses by default,
   or against a third, independently-implemented reference (e.g. a raw
   `scipy.integrate.solve_ivp(method="BDF")` run on dengo's own generated
   RHS at very tight tolerance, extended to also drive Grackle's rate
   tables through the same ODE for a true apples-to-apples check) --
   noted here as an open question, not resolved in this session.

**2026-09-08, continued: user's hunch was right -- a real gamma-treatment
bug found and fixed, and it explains most of the 34% gap above.** The
user asked to verify everything really is the same between the two
codes, suspecting Grackle's H2 gamma might include "more accurate H2
moments." Traced this by reading Grackle's actual C source
(`calculate_gamma.c`/`calculate_pressure.c`): Grackle applies a
composition- and temperature-dependent correction to its mixture
adiabatic index whenever H2 is a non-negligible fraction of the gas,
using the Omukai & Nishi (1998, ApJ 508, 141) partition-function formula
for H2's roto-vibrational internal degrees of freedom (gamma_H2 -> 7/5
at low T, -> 9/7 at high T as vibrational modes activate) --
self-consistently, every time it converts internal_energy <-> T.

Checking dengo's side turned up two real, independent bugs:

1. **dengo's own generated solver was never applying an H2-specific
   gamma at all**, even though the machinery for it (`interpolate_gamma_
   species`, `species_gamma()`, a whole gamma-interpolation-table code
   path in `cython_solver.C.template`) is fully implemented. The cause:
   `ChemicalNetwork.add_collection()` (used by every example/comparison
   script in this repo) calls `add_reaction(r, auto_add=False)`, and the
   bookkeeping that populates `interpolate_gamma_species` lived only
   inside `add_reaction`'s `auto_add=True` branch -- never reached via
   `add_collection`. So `interpolate_gamma_species` was silently always
   empty, the generated C template's `{%- if network.interpolate_gamma_
   species | length > 0 %}` block never fired, and H2 fell back to the
   flat `double gamma = 5.0/3.0` used for every other species -- a much
   cruder approximation than Grackle's. **Fix**: moved the bookkeeping
   into `add_species()` (the one method every code path funnels through
   -- add_collection, add_reaction, and add_cooling's auto_add branches
   were all changed to call it instead of touching `required_species`
   directly), so it's populated regardless of how a species enters the
   network. Verified directly: `network.interpolate_gamma_species` now
   contains H2_1/H2_2, and the interpolated table gives gamma_H2(1500K) =
   1.358, matching Grackle's own Omukai & Nishi value at that T exactly.

2. Once (1) exposed the *other* fit dengo actually had switched on,
   `species_gamma()`'s active formula for H2 turned out to be a
   different (unattributed, no docstring/citation found anywhere in the
   repo or git history back through the original pre-modernization
   commits) 10-parameter empirical fit -- while the *same* Omukai & Nishi
   formula Grackle uses was sitting right next to it, commented out,
   unused, apparently since before this project's history begins.
   Checked numerically: the two fits agree at T<~500K but diverge by
   10-15% in (gamma-1) at T=1500-3000K -- squarely dead center of this
   project's target regime. **Fix**: swapped which formula is active,
   commenting out the unattributed 10-parameter fit instead (kept for
   reference) and enabling the Omukai & Nishi formula, so dengo's and
   Grackle's H2 gamma physics now agree by construction.

   Fixing both (species_gamma() in `chemical_network.py`, plus a fresh
   `sort(attribute="name", case_sensitive=true)` on the newly-reachable
   `interpolate_gamma_species | sort` in `cython_solver.C.template` --
   the exact same case-sensitivity footgun as the earlier dictsort bug,
   dormant until now because the set was always empty -- and updating
   `tests/test_codegen.py::test_tables_bin_contents_match_write_order`,
   which had (correctly) started failing once the gamma tables actually
   got written) all 62 tests still pass.

   Rebuilt and reran the decoupled comparison
   (`.grackle_compare/run_decoupled.py`) with the fix in place: final
   temperature gap at n=3e15 dropped from **34% to 2.9%** (dengo 1998 K
   vs. Grackle 1941 K, was 1492 K vs. 2004 K), and H2 nuclei fraction now
   matches almost exactly (0.983 vs. 0.989, was 0.9998 vs. 0.9844). RMS
   relative T difference across the whole ~15-decade run dropped from
   double digits to 8.5%, with the *worst* remaining disagreement now at
   very low density/H2 fraction (the initial cooldown phase, where this
   gamma physics doesn't apply at all -- so likely a separate, smaller
   effect, e.g. He ionization-state cooling curve differences, not
   investigated further here). This is a good example of the decoupled-
   comparison harness (built specifically in response to "which produces
   better solutions") actually doing its job: it isolated a genuine,
   fixable chemistry-solver bug, not a collapse-model artifact.

**2026-09-08, continued: pinned down exactly why Grackle's curve is still
noisier even with the gamma bug fixed and the force-factor estimator
completely out of the picture.** User noticed the decoupled-test plot
still shows visible jitter in Grackle's curve and asked about it directly.
Quantified it first (n > 1e12 cm^-3, the visually-smooth-looking part of
the plot): Grackle's T has ~15x higher local jitter than dengo's (1.10%
RMS vs. 0.073%, relative to a local median-filtered baseline) and ~7x
higher step-to-step variation (1.37% vs. 0.19% RMS).

Read Grackle's actual solver algorithm (`solve_rate_cool_g.F`,
`do iter = 1, itmax` subcycle loop) to find the real mechanism, rather
than re-asserting the earlier (too narrow) force-factor-estimator
explanation, which cannot apply here since the decoupled test never
calls it at all. Grackle's Fortran core does its own *explicit*
sub-cycling to advance through a single externally-given `dt`:
`dtit(i) = min(0.1*de/dedot, 0.1*HI/HIdot, dt-ttot(i), 0.5*dt)`, grown by
up to 1.5x per iteration after iter>10, accumulated into `ttot` until it
closes `dt` (final sub-step truncated to fit exactly). This means the
*number* of sub-cycles needed to cover a given external `dt` is an
integer that can shift by +-1 as `dt` or the local state drifts even
slightly step to step, and the size of the final truncated remainder
sub-step varies non-smoothly too -- a structural, few-percent-level
jitter floor baked into this closure rule itself, unrelated to the
chemistry rates, the gamma physics, or the collapse model. Dengo's
generated solver, by contrast, is a single implicit backward-Euler step
per external `dt` with its own internal adaptive sub-stepping controlled
against a real relative-error tolerance (verified converged earlier: T
changes only ~0.6% across a 30x range in step size) -- no analogous
"integer subcycle count to close a fixed target" discontinuity.

Revised conclusion (supersedes the narrower one two entries up): dengo's
smoother curve in these free-fall tests isn't solely because the
force-factor estimator wasn't exercised -- dengo's adaptive-tolerance
implicit integrator is inherently smoother per call than Grackle's fixed-
fractional-timestep explicit subcycling, independent of the collapse
model. Both effects are real and additive: the force-factor estimator
adds jitter when a pressure-retarded free-fall model is used on top of
this; this per-call subcycling jitter is present regardless of which
free-fall model drives either solver.

**2026-09-08, continued: speed tests in the high-density regime, both
confounded (own free-fall driver) and controlled (decoupled, identical
dt/step-count).** Reran `run_dengo.py`/`run_grackle.py` fresh (post gamma
fix) and added per-step wall-clock instrumentation to
`run_decoupled.py`'s free-fall phase (both `solver.step()` and
`fc.solve_chemistry()` timed individually with `time.perf_counter()`).

*Own-driver comparison* (each code's own free-fall model, safety_factor
0.01 both): to cover n=[1e13, 3e15] cm^-3, dengo took 283 steps/0.57s,
Grackle 1102 steps/0.11s -- Grackle ~5.3x less *total* wall time despite
needing ~4x more calls, because it uses a pressure-retarded collapse
model that takes smaller density steps (not a chemistry-solver
difference -- see the two entries above on collapse-model confounds).

*Decoupled comparison* (identical dt and step count for both --
isolates pure per-call solver cost): dengo costs **~14x more per
`solver.step()`/`solve_chemistry()` call than Grackle, consistently
across all ~15 decades of density** (609us vs 43us/step at low density,
1981us vs 139us/step at n=[1e13,3e15] -- the ratio holds essentially flat
even as both sides' absolute cost grows ~3x with density). See
`speed_comparison.png`.

Traced roughly why: reran the decoupled test with dengo's `reltol`
loosened from its default 1e-5 to 1e-2 (Grackle's own subcycling has no
comparably strict convergence criterion, so this isn't a fully apples-
to-apples control, just a diagnostic). Dengo's per-step cost dropped
from 909us to 360us avg (final T barely moved, 1997.99 -> 1999.38 K) --
tolerance explains roughly half of the gap in log terms, but a ~5.7x
per-step disadvantage remains even at that loose tolerance. Most likely
structural: dengo's vendored `BE_chem_solve.C` does an implicit backward-
Euler solve with a dense Newton-iteration linear solve over the full
~9-species+energy coupled system every internal sub-step, while
Grackle's Fortran subcycling (see the entry above on its jitter) does
semi-implicit, closed-form, per-species scalar updates with no matrix
solve at all -- a cheaper but less rigorously-controlled update. This
wasn't verified by direct instrumentation of Newton-iteration counts
inside BE_chem_solve.C (would need adding a counter and rebuilding), so
treat the "which piece of the ~5.7x floor is Newton-solve cost vs. other
per-call overhead" attribution as reasoned-but-unconfirmed.

Practical implication: dengo's OpenMP parallelism (per-cell, embarrassingly
parallel) is the more relevant lever for a real grid-scale hydro
coupling, not single-cell serial speed -- and there may be room to loosen
dengo's default reltol somewhat (it's already known to be converged well
past what 1e-5 requires, per the earlier convergence check) to recover
some of this gap without sacrificing accuracy, not yet done here.

**2026-09-08, continued: user asked to think carefully about cache misses
in the table interpolation -- checked with a real profiler rather than
reasoning about it, and the hypothesis doesn't hold up; something else
does.** `perf` doesn't work in this sandbox (`perf_event_paranoid=4`
blocks it even for own-process use), so used `valgrind --tool=cachegrind
--cache-sim=yes` instead (a software cache simulator, no special kernel
perms needed) on a minimal, isolated benchmark: 30 repeated
`solver.step()` calls at a fixed high-density/high-H2 state (n=3e14,
T~1800K), run directly against the venv's python (not through the `uv
run` wrapper, which forks a child valgrind doesn't trace by default).

Result: D1 (L1 data) miss rate over the whole run was 3.63% (144M/3.98B
reads), with only ~14% of those D1 misses also missing the last-level
cache -- an unremarkable, not-thrashing profile overall. But the
per-function breakdown was the real finding: **the entire compiled
solver core -- `BE_chem_solve.C`, which houses the Newton iteration,
`Gauss_Elim`, and (via function pointers) every call into
`interpolate_rates`/`calculate_rhs`/`calculate_jacobian` -- accounts for
only ~0.1% of total instructions and 0.1% of D1 misses** across the
whole benchmark. `primordial_interpolate_rates`/`calculate_rhs_
primordial`/`calculate_jacobian_primordial` are confirmed present as
named symbols in the built .so (`nm -C`), but don't appear as separate
lines in cg_annotate's output at all -- most likely fully inlined into
BE_chem_solve's call sites at `-O3`, which is consistent with (not
contradicting) their combined cost being part of that same ~0.1%.

Instead, >23% of total instructions (a conservative partial sum over
just ~19 named functions; the true total including bytecode-dispatch
stubs is considerably higher) is CPython object machinery: `PyDict_
SetItem`, `PyDict_GetItemRef`, `PyObject_GenericGetAttr`, `dict_
traverse`/`dictresize`, `_PyObject_Malloc`/`_Free`, and -- tellingly --
`gc_collect_main` actually running periodically inside what should be a
tight numerical loop. Reading `cython_solver_run.pyx.template`'s
`Solver.step()` explains why: every single call does `np.
ascontiguousarray(state[name], dtype=np.float64).reshape(-1)` for each
of the ~10 species (fresh NumPy C-API calls + a fresh view object each,
every call), then at the end builds a brand-new Python dict plus 10
fresh `np.array(...)` objects for the return value -- all of this
Python/NumPy-level object churn is fixed per-call overhead that doesn't
shrink with `dims`, so at `dims=1` (a single cell, called in a tight
Python loop -- exactly this project's free-fall driving scripts'
pattern) there's no large numerical payload to amortize it against, and
it dominates.

Practical implication, revised from the earlier (reasoned-but-
unconfirmed) Newton-solve-cost guess two entries up: the ~14x per-call
cost gap vs. Grackle is much more likely dominated by this Python/Cython
calling-convention overhead than by C-level table-interpolation cache
behavior or even the Newton solve itself. The fix this points to is
narrower and cheaper than a table-layout rework: give `Solver.step()` a
lower-overhead path for the repeated-single-cell case (skip re-wrapping
already-contiguous float64 arrays, avoid rebuilding a dict + 10 arrays
every call), rather than restructuring how rate tables are stored. Not
yet implemented -- flagged as the concrete next step if pursued.

**2026-09-08, continued: implemented reuse/avoidance of the per-call
Python/NumPy marshaling identified above -- then measured it honestly,
which corrects that earlier finding rather than confirming it.**

Added a low-overhead calling convention to `cython_solver_run.pyx.
template`'s `Solver`: `self.state` (shape `(dims, NSPECIES)`) and
`self.T` (shape `(dims,)`) are now zero-copy numpy views straight onto
the persistent `_input`/`Ts` buffers, built once in `__cinit__` via
Cython's pointer-to-memoryview cast (`<double[:dims, :n]> self._input`)
and cached as the same ndarray object for the handle's whole lifetime --
not recomputed/rewrapped on each access. A new `step_inplace(dtf, niter,
reltol, ...)` reads/writes those directly and returns just `(converged,
t)` -- no dict, no `np.array()`/`ascontiguousarray()` call, ever. The
existing dict-based `step()` still works exactly as before (same tests
pass unchanged) but is now a thin wrapper around the same shared core
(factored into `_advance()`, a `cdef` method with a C-tuple return, used
by both). `SPECIES_INDEX` (name -> column index) added at module level
so callers don't have to hardcode column order. Two new tests
(`test_step_inplace_matches_step`, `test_solver_state_view_is_persistent
_and_writable`) confirm `step_inplace()` reaches bit-identical state to
`step()` and that `solver.state`/`solver.T` really are the same,
directly-writable ndarray object across calls, not a fresh wrapper each
time. All 64 tests (62 previous + 2 new) pass.

Caveat documented in the code: `solver.state`/`solver.T` are unsafe to
use after `close()` -- they wrap the same raw buffers `close()` frees,
and nothing revokes a numpy array's memory out from under it. Same
general caveat as any zero-copy buffer view; not engineered around
further since the intended caller (a driver holding the `Solver` open
for its whole run) naturally avoids it.

**Then measured the actual effect, and it does NOT match the earlier
cachegrind-based conclusion two entries up -- that conclusion is hereby
corrected, not just supplemented.** At `dims=1` (the free-fall scripts'
actual usage), `step_inplace()` vs. the old `step()`: 574us vs. 612us
per call -- only a ~6% reduction, not the order-of-magnitude the "96%+
CPython object overhead" reading implied. At `dims=100000` (a
grid-scale chunk, the actual "HPC case" this was meant to help): 13.84s
vs. 13.95s per call -- again only ~1%. A second diagnostic (loosening
the inner temperature-Newton convergence criterion in `calculate_
temperature` 10000x, from 1e-8 to 1e-4, then reverted -- see
`cython_solver.C.template`'s `Tdiff/Tnew` check) also only bought ~7%.

Both hypotheses this session raised for "why is dengo's per-call cost
so much higher than Grackle's" -- table-interpolation cache misses, and
then Python/dict marshaling overhead -- turn out not to be it, once
tested by actually removing each one rather than by profiling alone.
The cachegrind run's "96%+ unresolved/CPython" attribution was most
likely simply wrong: `primordial_interpolate_rates`/`calculate_rhs_
primordial`/`calculate_jacobian_primordial` are confirmed present as
named symbols in the built `.so` (`nm -C`) but never appeared anywhere
in `cg_annotate`'s output under any name search -- a real profiler
attribution failure for this build (cause not identified; not resolved
by adding `-g`), not evidence those functions cost nothing. Retracting
the specific "it's CPython overhead, not the C solve" claim; the honest
current answer is that the per-call cost is real, legitimate numerical
work: `BE_chem_solve`'s own outer adaptive-dt loop (~39-45 attempts to
cover one external `dtf` at this density/tolerance) runs an inner Newton
iteration of up to `sweeps=10` sub-iterations *each* (see `BE_chem_
solve.C`), and every one of those needs a fresh `interpolate_rates` pass
(~90 separate reaction/cooling/gamma tables) plus RHS and Jacobian
assembly plus (now that H2's gamma is properly T-dependent) `calculate_
temperature`'s own nested Newton solve -- on the order of a few hundred
such evaluations per external `step()`/`step_inplace()` call. That is
inherent to solving this stiff a system this tightly with a dense
Newton method, not an artifact of how the table data is laid out or how
Python hands state to the solver.

Kept the fix anyway -- `step_inplace()`/`solver.state`/`solver.T` are
still the architecturally right calling convention for any repeated-call
use (no Python-object churn added on top of whatever the real solve
costs, at any `dims`), and existing free-fall/comparison scripts should
migrate to it -- but it should not be oversold as a major speed win on
its own. The productive next lever for that, not attempted here, is
reducing how much of that per-call numerical work is needed in the
first place -- e.g. the original project plan's still-unimplemented
partial-equilibrium/QSS treatment for fast species (H-, H2+, e-), which
would cut the size/stiffness of the system BE_chem_solve has to Newton-
iterate on, rather than making each iteration itself cheaper.

   Remaining known inconsistency, not yet fixed: the *driving scripts'*
   own compressional-heating formula (the ad hoc `thermodynamic_gamma()`
   Python helper used to bump `ge`/`internal_energy` after each
   compression step, before handing off to either solver) still uses a
   simplified placeholder gamma on both sides (dengo driver: fixed 7/5
   for H2 regardless of T; Grackle driver: fixed `my_chemistry.Gamma` =
   5/3, matching gracklepy's own reference `evolve_freefall` utility) --
   neither matches either solver's own (now-consistent) internal EOS.
   This driving-script simplification is shared by both sides of the
   comparison, so it's a smaller and more symmetric effect than the bug
   just fixed, but replacing it with each solver's actual gamma (`fc.
   calculate_gamma()` for Grackle; a call into dengo's own gamma table)
   would be the natural next refinement.

**2026-09-09: evaluated, then built, a Grackle-API-compatible shim
(`dengo.grackle_compat`).** User asked how hard it would be to expose
Grackle's API for a drop-in replacement. Split this into two very
different projects and scoped both:

- **(A) API-shape compatibility for what dengo already implements**
  (primordial H/He/H2 chemistry, Grackle's `primordial_chemistry=2`, no
  metals/UV background/dust/radiative transfer) -- moderate,
  well-scoped, multi-day-not-multi-week effort, mostly glue + unit-
  conversion code given `.grackle_compare/`'s prior work already solved
  the hard parts (species/field mapping, units conversion, confirming
  the two codes' physics agrees once the gamma bug was fixed).
- **(B) Full Grackle feature parity** (metal-line Cloudy cooling tables,
  UV background + self-shielding, dust physics, D/D+/HD tracking,
  radiative transfer coupling) -- a much larger, separate physics-
  content project (new rate/cooling tables and species dengo doesn't
  have at all), not a shim; explicitly out of scope here.

Built (A) as `src/dengo/grackle_compat.py`:

- `chemistry_data`: instantiate-then-set-attributes ergonomics matching
  gracklepy's own, with `set_velocity_units()` reproducing Grackle's
  exact formula (`grackle_units.c`: `velocity_units =
  length_units/time_units`, `/= a_value` if comoving). `initialize()`
  validates every parameter in `_UNSUPPORTED_UNLESS_DEFAULT` (metal_
  cooling, dust_chemistry, UVbackground, primordial_chemistry != 2,
  self-shielding, radiative transfer, ...) and raises
  `GrackleCompatError` -- loudly, not a silent no-op -- for anything
  outside what this shim implements.
- `FluidContainer`: a dict-like field container backed by a dengo
  `Solver`, with Grackle's exact field names/units convention (density
  fields as *mass* density in `chemistry_data.density_units`,
  `internal_energy` in `velocity_units**2`) -- `_push()`/`_pull()`
  convert to/from dengo's native cgs number densities and specific
  energy at the boundary, the same conversions
  `.grackle_compare/run_grackle.py`/`run_decoupled.py` already had to
  get right by hand.
- `solve_chemistry(dt)`, `calculate_temperature()`, `calculate_pressure()`
  -> trivial ideal-gas-law from already-known n_tot/T,
  `calculate_gamma()` -> reimplemented the Omukai & Nishi H2-gamma
  mixture formula directly in Python/numpy from `solver.state`/`solver.T`
  (no new C/Cython needed -- it's a simple closed form, and dengo's own
  internal copy already exists for the *solver's* internal use, just
  wasn't queryable from outside), `calculate_cooling_time()` -> `ge /
  |d(ge)/dt|` via the new bulk RHS evaluator (below). `calculate_dust_
  temperature()` raises `GrackleCompatError` (dust not implemented).
- `setup_fluid_container()`: a restricted version of gracklepy's own
  (ionized/neutral single-cell setup); `converge=True` (iterate to a
  self-consistent starting T) and nonzero `metal_mass_fraction` both
  raise rather than silently doing the wrong thing.

Two small, genuinely general (not Grackle-specific) additions to the
Solver template made this possible without new C/Cython plumbing beyond
what already existed: `evaluate_temperature_bulk()`/`evaluate_rhs_bulk()`
-- the multi-cell (`dims`-wide), marshaling-free counterparts to the
existing single-cell `evaluate_temperature()`/`evaluate_rhs()`, reading/
writing `self.state`/`self.T`/the new `self.rhs` directly. New tests
confirm they agree with the single-cell dict-based API (to ~1e-4
relative -- both converge their own Newton iteration to 1e-8 internally
from potentially different starting guesses, so exact bit-agreement
isn't expected) and give a genuinely independent answer per cell, not a
broadcast from cell 0.

Also consolidated three near-duplicate copies of the primordial
species/cooling/reaction lists (`examples/primordial_network.py`,
`.grackle_compare/primordial_network_helpers.py`, `tests/conftest.py`)
into one canonical `src/dengo/primordial_network.py`, used by all three
plus the new `grackle_compat` module -- the "minimize duplication"
constraint applies to this project's own internals just as much as to
solver templates.

**Validated against the real gracklepy directly**
(`.grackle_compare/validate_grackle_compat.py`, not a repo test
dependency -- needs the `.grackle_compare/.venv` gracklepy install):
built the identical ionized-cooldown test problem through both real
`gracklepy.chemistry_data`/`FluidContainer` and
`dengo.grackle_compat`'s, stepped both forward with `solve_chemistry()`,
compared `calculate_temperature()`/`calculate_gamma()` at every step.
After an initial few steps of larger transient disagreement (both sides
start from the same approximate `1.5*kB*T/mh` internal-energy guess,
which is only approximately consistent with either solver's real EOS),
temperature converges to agree within ~1% and gamma matches to 4+
significant figures (both ~5/3, negligible H2 at these conditions) --
the same level of agreement established throughout this investigation
between dengo's and Grackle's underlying chemistry, now confirmed to
survive going through the compat layer's unit conversions too, not just
the native APIs. 80/80 tests pass (65 previous + 15 new
`test_grackle_compat.py` tests).

Known limitation worth flagging: `solve_chemistry()`'s convergence
check uses a fixed `reltol=1.0e-5` rather than exposing Grackle's own
tolerance-equivalent controls, and `max_iterations` is the only
Grackle-side solver-tuning parameter actually honored (mapped to
`niter`). Not a correctness issue, but a caller relying on Grackle's
other iteration-control parameters won't find them respected.

**2026-09-09, continued: user correctly pushed back on comparing any
Python code at all -- "the thing we care about is what gets called by
the HPC code."** Right: a real simulation calling dengo or Grackle
never touches Python at runtime -- it links the compiled library and
calls its C API directly. Built `.grackle_compare/c_bench/`: two
standalone benchmarks with *no Python or Cython anywhere in the timed
path*, on either side.

- `grackle_bench.c`: adapted directly from Grackle's own reference
  example (`.grackle/src/example/c_local_example.c`) -- calls
  `local_initialize_chemistry_data()`/`local_solve_chemistry()` from
  `<grackle.h>`, the exact entry points a real HPC code (Enzo, etc.)
  uses. Linked directly against the prebuilt `libgrackle-3.4.1.so`
  bundled in the gracklepy wheel already used throughout this
  investigation (a `libgrackle.so` symlink, since `-lgrackle` needs the
  unversioned name; the wheel only ships the versioned SONAME). Needed
  one missing generated header, `grackle_float.h` (normally emitted by
  Grackle's own build system from `grackle_float.h.in`, absent from a
  bare source checkout) -- created by hand, `#define GRACKLE_FLOAT_8`,
  confirmed correct by checking gracklepy's own Cython wrapper assumes
  `gr_float` == `double` (`fluid_container.py`'s default `dtype=
  "float64"`, `grackle_wrapper.pyx`'s `gr_float[::1] view = arr` cast).
- `dengo_bench.cpp`: calls `primordial_setup_data()`/`BE_chem_solve()`
  directly, compiled straight against the already-generated
  `primordial_solver.C`/`BE_chem_solve.C` in `.grackle_compare/
  _dengo_build/` -- no `.pyx`, no Cython, no `setuptools` build step at
  all. The adaptive-dt loop is a line-for-line C++ translation of
  `_advance()` in `cython_solver_run.pyx.template`.
- `run_sweep.sh`: builds both and runs the same grid-size sweep as
  yesterday's Python-level benchmark (dims = 1, 100, 2048, 100000),
  same synthetic state (n=1e13 cm^-3, T=1500K, 10% molecular) and same
  dt (1e-3 of the local free-fall time) as `benchmark_amortized.py`
  used (kept for reference, superseded by this as the number that
  actually matters).

**Result, at the C level, no Python anywhere:**

| dims    | dengo (us/cell) | grackle (us/cell) | ratio |
|---------|------------------|--------------------|-------|
| 1       | 588.2            | 110.6              | 5.3x  |
| 100     | 322.3            | 45.9               | 7.0x  |
| 2048    | 307.6            | 47.9               | 6.4x  |
| 100000  | 110.1            | 66.3               | 1.7x  |

These essentially match yesterday's Python-level numbers at every
scale (e.g. dims=100000: 110.1 vs. yesterday's 101.1 us/cell for dengo,
66.3 vs. 64.2 for grackle) -- confirming the Python/Cython layer's own
overhead was *already* negligible for the calling conventions used
(`Solver.step_inplace()`/`solver.state`, `FluidContainer.
solve_chemistry()`), at every grid size tested, not just at scale. That
also means `dengo.grackle_compat`'s own overhead (measured yesterday at
~2x native dengo, from its Python-level unit-conversion `_push()`/
`_pull()`) is real but *irrelevant to this question* -- no real HPC/AMR
code embeds dengo through that shim's Python layer in a performance-
critical inner loop; it exists for Python-level drop-in convenience
(an analysis script, a Python-driven test), not for compiled
simulation coupling, which is exactly why this second benchmark
bypasses it entirely.

**One-time setup, corrected**: yesterday's ~20s "one-time cost" for
dengo was codegen + compilation -- a *build-time* cost in any real
deployment (you compile the generated C++ into your simulation once,
same as Grackle's own Fortran/C is already compiled into
`libgrackle.so` once), not something paid at every simulation launch.
The actual per-run cost -- reading the rate tables into memory once at
startup -- is small for both and, notably, *smaller for dengo*:
`primordial_setup_data()`'s flat-binary `fread()` takes 0.5-3ms;
Grackle's `local_initialize_chemistry_data()` (parsing the Cloudy HDF5
file) takes ~33ms, consistently, regardless of grid size.

Per-cell cost still drops sharply with grid size for dengo (588 -> 111
us/cell, OpenMP crossing DENGO_OMP_MIN_CELLS=2048) while Grackle (this
wheel has no OpenMP compiled in -- confirmed via `ldd`/`nm -D` on
`libgrackle-3.4.1.so`, no `libgomp`/`omp_get_*` symbols at all) stays
roughly flat -- consistent with yesterday's explicit OMP_NUM_THREADS
scaling check (1/4/24 threads: 526/192/118 us/cell, ~4.5x speedup on
24x the cores, i.e. ~19% parallel efficiency). That scaling quality,
not marshaling or setup cost, is the real remaining lever if closing
the gap further at grid scale is wanted -- not attempted here.

Caveats: single synthetic uniform-composition state at one density/
temperature, not a validation across the full ~15-decade range this
project otherwise targets; this specific prebuilt Grackle wheel has no
OpenMP, a source build configured with `--enable-openmp` could behave
differently at scale; dengo's `DENGO_OMP_MIN_CELLS=2048` threshold
means grid patches smaller than that see none of the scaling benefit
visible in the table above.
