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

**2026-09-09, continued: code-level (not algorithmic) audit of the hot
path, per the user's request -- "cache issues, double loops that
should be reversed, work done multiple times unnecessarily."** Read
`calculate_rhs`/`calculate_jacobian`/`calculate_temperature` (generated
from `cython_solver.C.template`) and `BE_chem_solve.C`'s Newton/linear-
solve loop line by line. Found six concrete issues, matching all three
categories:

1. **Redundant table-interpolation pass** (not fixed, see below):
   `calculate_rhs` calls `calculate_temperature` -- whose own per-cell
   Newton loop already computes each cell's final `bin_id`/`Tdef`/`dT`
   via `interpolate_gamma()`'s last iteration, once T converges -- then
   immediately runs `interpolate_rates` as a second, wholly separate
   full pass over all cells that recomputes the *identical*
   `bin_id`/`Tdef`/`dT` from the same (now frozen) `logTs[i]`, just to
   interpolate the other ~54 reaction/cooling tables.
2. **Same rescale round-trip done twice per Newton sweep** (fixed):
   the old `BE_Resid_Fun`/`BE_Resid_Jac` each independently did `u *=
   scaling; call function; u *= inv_scaling` on the *same* u (nothing
   changes it between the two calls within one sweep) -- the second
   round-trip repeated the first's identical work and threw away two
   floating-point roundings for nothing.
3. **A real wrong-loop-order cache-stride bug** (fixed): `BE_Resid_Jac`'s
   Jacobian row-rescale loop correctly nested `ivar` innermost (matching
   `Ju`'s contiguous fastest-varying index); the column-rescale loop
   right below it had `ivar`/`jvar` swapped, striding by `nchem` (80
   bytes for nchem=10) on every access instead of walking contiguous
   memory.
4. **Two full-array passes that should be one** (fixed): those same two
   loops each swept the entire `nstrip*nchem*nchem` Jacobian separately;
   fused into `Ju[k] *= inv_scaling[row] * scaling[col]` in one pass,
   with `ivar` innermost -- fixes #3 for free.
5. **Missed parallelization + redundant serial pass** (fixed):
   `calculate_rhs`'s NaN check ran in a second loop, *after and outside*
   the per-cell `#pragma omp parallel for` block, with no OpenMP pragma
   of its own -- a fully serial full-array re-scan on every single RHS
   evaluation at any grid size. Moved inline, checked per-species right
   after each `rhs[j]` is computed, inside the existing parallel loop.
6. **~112 separate small heap allocations for per-cell interpolated
   rates** (not fixed): `data->rs_k01`, `data->drs_k01`, etc. are each
   their own `malloc(nstrip*sizeof(double))` rather than one combined
   block -- fragmented, especially at small `nstrip`. (Correction to an
   earlier session's speculation: the *read-only base* tables,
   `r_k01[1024]` etc., are fixed inline struct members and already
   contiguous -- only the per-cell *interpolated* result arrays are
   fragmented like this.)

Implemented #2/#3/#4/#5 (all mechanically-provable, equivalence-checkable
transformations, safe to change without touching Newton-convergence
internals other call sites depend on): combined `BE_Resid_Fun`/
`BE_Resid_Jac` into `BE_Resid_FunJac` in `BE_chem_solve.C`; folded the
NaN check into `calculate_rhs`'s existing parallel loop in the C
template. Deliberately held off #1 and #6: #1 requires restructuring
`calculate_temperature`'s internals in a way that would also add rate-
table interpolation cost to every *standalone* T-only caller (`Solver.
evaluate_temperature()`/`evaluate_temperature_bulk()`, used by the
free-fall scripts to re-derive T after a compression step without
advancing chemistry) unless carefully split into two variants -- real,
but a bigger, riskier change than the others, better done as its own
follow-up with its own before/after measurement. #6 is a genuine
structural change (touches every generated network's data layout, not
just the vendored solver), same reasoning.

All 80 tests pass; `examples/free_fall_collapse.py` reproduces the
*exact* same 1814-step trajectory as before the change (same n, T at
every checkpoint) -- strong evidence the transformations are truly
equivalent, not just "probably fine."

**Measured, pure-C++ harness (`.grackle_compare/c_bench/dengo_bench`,
no Python/Cython in the timed path -- see the entry above), same grid
sweep, multiple repeated trials for a clean signal:**

| dims    | before (us/cell) | after (us/cell) | improvement |
|---------|-------------------|-------------------|-------------|
| 1       | 588.2             | ~520.0            | ~11.6%      |
| 100     | 322.3             | ~292.1            | ~9.4%       |
| 2048    | 307.6             | ~278.8            | ~9.4%       |
| 100000  | 110.1             | ~96.4             | ~12.4%      |

A real, reproducible ~9-12% per-call speedup at every grid size tested,
from three small, safe, equivalence-preserving changes -- no algorithm
change, no new physics, nothing touched outside `BE_chem_solve.C`'s
internal residual/Jacobian bookkeeping and one inlined check.
Correspondingly narrows the dengo-vs-Grackle gap from the previous
entry's 5.3x/7.0x/6.4x/1.7x to roughly 4.7x/6.4x/5.8x/1.45x across the
same grid sizes.

Remaining known opportunities, not attempted here: #1 and #6 above
(each would need its own careful, isolated before/after measurement
given their larger blast radius), and the OpenMP parallel-scaling
efficiency question already on record two entries up (~19% efficiency
at 24 threads) -- still the largest single lever if further narrowing
the gap at grid scale is wanted, and unrelated to any of today's fixes.

**2026-09-09, continued: user correctly refused to accept the residual
gap -- "iterating to convergence should indeed take longer, but a 5x
difference is incredible, especially when we should expect the values
to already be near converged equilibrium."** Right, and it led straight
to the real, dominant bottleneck -- a step-size *policy* bug, not
genuine physics-driven work.

Checked directly: drove the same near-equilibrium state (n=1e13 cm^-3,
T=1500K, 10% molecular) through 15 repeated `solver.step()` calls,
printing the internal sub-step count each time. Result: **exactly 32
internal iterations, every single call, completely flat**, even as T
settled to a change of ~0.03K/call (i.e. visibly at quasi-equilibrium
by the later calls). That's not a physics signal -- solving
`(dtf/200)*(1.1^N-1)/0.1 = dtf` for N gives N=32 exactly, for *any*
state at all. `_advance()`'s adaptive-step schedule always starts each
external call at `dtf/niter` (tiny) and grows by a fixed 1.1x per
success; the sub-step count is a pure function of `niter`/growth
factor, never informed by how easily the Newton solve actually
converges or how close to equilibrium the state already is. The
solver never even tries a large step first to see if one would work.

Tested growth factors 1.5 and 2.0 directly against this same
near-equilibrium probe and, more importantly, against the *full*
free-fall collapse trajectory (the hardest test available, spanning
~15 decades of density including the rapid H2-formation transition,
not just a near-equilibrium snapshot):

| growth factor | internal iterations (near-eq. probe) |
|---|---|
| 1.1 (original) | 32 |
| 1.5 | 12 |
| 2.0 | 8 |

Full collapse trajectory at growth=2.0: identical 1814 steps, T at
every checkpoint matching the 1.1x baseline to ~0.05-0.12% (e.g. step
1600: 1831.7 K baseline vs. 1832.4 K at growth=2.0) -- a legitimate,
tiny truncation-level difference from taking fewer/larger backward-
Euler sub-steps at the same `reltol=1e-5` Newton-convergence tolerance,
nowhere near large enough to explain a 5x wall-clock gap. Changed the
default growth factor from 1.1x to 2.0x in `cython_solver_run.pyx.
template`'s `_advance()` (docstring/comments updated to match); all 80
tests still pass.

**Combined effect, measured with the pure-C++ harness (both this fix
and the four code-optimizations from the previous entry stacked
together), same grid sweep:**

| dims | dengo, original | dengo, now | grackle | ratio, original | ratio, now |
|------|-----------------|------------|---------|------------------|------------|
| 1      | 588.2 us/cell | ~149 us/cell  | 109.3 us/cell | 5.3x  | **1.4x**  |
| 100    | 322.3 us/cell | 90.4 us/cell  | 44.7 us/cell  | 7.0x  | **2.0x**  |
| 2048   | 307.6 us/cell | 95.3 us/cell  | 47.6 us/cell  | 6.4x  | **2.0x**  |
| 100000 | 110.1 us/cell | 28.9 us/cell  | 65.5 us/cell  | 1.7x  | **0.44x (dengo faster)** |

At grid scale (dims=100000, the regime that matters for an actual
simulation coupling), dengo is now measurably *faster* than this
Grackle build, not ~1.7x slower -- purely from fixing how the internal
step size ramps up, with no change to the chemistry, the tolerance, or
any physics at all. This is by far the largest single change in this
whole investigation's speed story, and it was found by taking the
user's skepticism seriously rather than accepting "iterating to
convergence takes longer" as a sufficient explanation for a gap this
large.

Open follow-up worth flagging: growth=2.0 was chosen because it's a
clean, standard doubling policy that tested well here, not because it
was tuned as an optimum -- a genuinely adaptive scheme (grow more
aggressively after an *easy* convergence, more conservatively after a
*hard* one, rather than a fixed multiplier regardless of how the
previous sub-step went) would likely do even better and is a natural
next step, but wasn't attempted here.

**2026-09-09, continued: quality-of-life -- an interactive Jupyter
widget explorer.** User asked to shift from performance work to
quality-of-life, leading with a concrete want: "a fun little web
interface that let me set ICs and then immediately see the results...
implemented in Jupyter Widgets in a single notebook."

Added `examples/interactive_explorer.ipynb`: `ipywidgets` sliders for
initial density, temperature, ionized fraction, and H2 fraction, plus a
mode toggle between the two existing example scripts' physics --
"cool at constant density" (`primordial_network.py`'s test problem,
dt set each step from the current cooling time) and "free-fall
collapse" (`free_fall_collapse.py`'s plain free-fall prescription, up
to a user-chosen target density). Every slider release re-solves and
redraws a two-panel T(x)/H2-fraction(x) log-log plot in place via an
`ipywidgets.Output()`.

Deliberately built on today's own optimization work rather than the
dict-based `step()` API: one persistent `Solver` handle held for the
notebook's whole lifetime, `solver.state`/`SPECIES_INDEX` written to
directly and `step_inplace()` called in the hot loop -- zero Python
marshaling per step, and (per this session's step-size-policy fix)
~4x fewer internal Newton sub-steps than before that fix -- meaning a
few-thousand-step trajectory now genuinely redraws fast enough to feel
live, which it would not have before today's fixes. A nice direct
payoff of the performance work feeding straight into the UX one.

Added a `notebook` dependency group (`ipywidgets`, `jupyterlab`,
`matplotlib` -- kept separate from `dev` since nothing else needs it)
via `uv add --group notebook`. Verified the whole notebook executes
cleanly end to end with `uv run --group notebook jupyter execute
examples/interactive_explorer.ipynb` (solver builds, widget UI
constructs, default-mode plot renders, no errors) -- this can't
simulate an actual slider drag, so also directly re-ran the free-fall
branch's exact logic standalone to confirm it independently (1298
steps, final n~1.02e15 cm^-3, final T=2198K -- consistent with this
project's known free-fall physics). No baked-in cell outputs/execution
counts committed (kept the notebook clean, matching normal practice).

Also fixed a stale claim caught while touching this area:
`examples/README.md` still described `free_fall_collapse.py` as using
the Omukai et al. (2005) force-factor scheme, which an earlier entry in
this file already replaced with plain free-fall -- the docstring was
updated at the time but this README wasn't. Fixed, and cross-linked the
new notebook from both `examples/README.md` and the top-level README.

Run with: `uv run --group notebook jupyter lab
examples/interactive_explorer.ipynb`.

**2026-09-09, continued: two more quality-of-life items -- a one-line
convenience entry point, and non-convergence diagnostics that actually
say why.**

**`dengo.quick_solve(nH=..., T=..., dtf=...)`** (`src/dengo/quick_solve.py`,
wired up via a newly non-empty `src/dengo/__init__.py`): builds and
compiles the primordial network's solver once per process (module-level
cache, `~/tempfile.mkdtemp()` build dir, same pattern `grackle_compat`
already uses), then reuses it across calls -- first call ~23s (the
codegen+compile cost, same one-time cost noted for the C-level
benchmarks two entries up), every call after that sub-millisecond.
`full_output=True` returns `(final, solver)` so a failed
(`final["converged"] is False`) call can be followed by
`solver.last_error` (see below) to find out why. Note `import dengo`
itself is no longer a total no-op (previously an empty `__init__.py`):
it now eagerly imports `dengo.primordial_network`'s chain (chemical_
network/reaction_classes/sympy etc.), adding real but modest import
time (~1s) -- nothing is *compiled* until `quick_solve()` is actually
called, so the README's "zero compiled extensions required to import
dengo" claim still holds, just not "instant to import" anymore.

**`Solver.last_error`**: a new property surfacing exactly what its name
says -- which species (and, for `dims>1`, which cell) was hardest to
satisfy on a step that returned `converged=False`, instead of just
"didn't converge". The raw numbers (which species violated tolerance by
how much) were already computed by BE_chem_solve.C's existing
convergence check; they used to just get discarded after an
occasionally-firing raw `fprintf(stderr, ...)` debug print gated behind
`if (dt < 1.0)`. Added a small `BE_chem_solve_diag` struct + `static`
tracking + a `BE_chem_solve_last_failure()` getter (new
`cython_solver.h.template`/`.pyx.template` declarations, `BE_chem_
solve.C` capture logic) -- reason 1 (tolerance not met, with
species/cell/value/change/atol/rtol/ratio), 2 (NaN), 3 (singular
Jacobian, Gauss_Elim), or 4 (a generic fallback for when calculate_rhs/
calculate_jacobian themselves reject the state, e.g. a negative species
density -- doesn't currently carry per-species detail, since that
isn't threaded back through f()'s/J()'s own return code; noted as a
scoped gap, not attempted here).

Found and fixed a real bug while validating this against an actual
successful step: capturing unconditionally at every tolerance-violation
check (the first version) left `last_error` showing a *stale* failure
from an early Newton sweep even after the call went on to converge on a
later sweep within the same call -- normal Newton iteration routinely
"fails" tolerance on early sweeps before succeeding, that's not an
error. Fixed by tracking in a call-scoped (not `static`) `local_diag`,
reset every sweep, only copied into the exposed `static` diagnostic at
the two points the function is actually about to return failure --
verified directly: `last_error` is `None` before any step, stays `None`
after a step that converges (even though intermediate sweeps within
that same call had failed the tolerance check), and correctly reports a
real species/ratio/message when a step is forced to fail (tested with
an unsatisfiable `reltol=1e-300`).

Both new tests/test_quick_solve.py and the two new tests in
tests/test_solver.py pass; 85/85 tests total. `examples/free_fall_
collapse.py` reproduces the identical 1814-step trajectory (BE_chem_
solve.C's diagnostic capture is purely additive bookkeeping, changes no
control flow or numbers).

**2026-09-09, continued: WASM feasibility -- evaluated, then proven,
then built into a real working browser widget. New branch `wasm-
solver` (branched from `modernize-uv-cython-openmp`), per the user's
request.**

User asked about the feasibility of a WASM target for the solver, to
enable an in-browser widget. Rather than answer from first principles,
installed the Emscripten SDK locally (`emsdk`, no root needed) and
actually compiled dengo's generated primordial solver to WebAssembly.

**Result: it just works, no changes needed to the generated solver.**
`primordial_solver.C`/`BE_chem_solve.C` are already portable C/C++
(`math.h`/`stdio.h`/`stdlib.h`/`string.h` only, no POSIX-specific
calls), and OpenMP is already optional with a working serial fallback
-- irrelevant anyway for a single-cell widget (`nstrip=1` never
approaches `DENGO_OMP_MIN_CELLS=2048`). Wrote a small (~100-line) hand-
written C API wrapper mirroring `_advance()` exactly (including the
growth=2.0 step-size fix), embedded the 475KB rate-table binary
directly into the module via `--embed-file` (no separate network
fetch), and compiled with `em++`.

**Verified, not assumed:** ran the identical initial conditions from
the README's quickstart through both the wasm build (in Node) and the
native Python/Cython solver -- bit-for-bit identical results (same T,
H2_1, H_1 to every printed digit). Per-call cost in Node: ~112 us/cell,
matching the already-optimized native per-cell cost from two sessions
ago -- no wasm performance penalty. Total footprint: 527KB wasm + 62KB
JS glue, ~393KB gzipped -- a small, instant-loading download.

Presented two paths: (A) this custom minimal wasm+JS approach, small
and fast, vs (B) Pyodide/JupyterLite (reuse the existing ipywidgets
notebook almost unchanged, but tens of MB and several seconds to
initialize a full wasm-compiled CPython+numpy stack). Recommended (A)
for "a fun little widget"; user agreed and asked to build it, then
generalize it into a proper codegen target, emphasizing interactivity
and OK with Vega-Lite for visualization. Also asked this work go on its
own branch.

**Built `wasm/`:**
- `dengo_wasm.cpp`: the C API (`init`, `step`, `state_ptr`, `rhs_ptr`
  -- the last added specifically so the JS driver could compute a real
  cooling-time-limited dt for the constant-density mode, `|ge/d(ge)/
  dt)|`, the same recipe `run_dengo.py`/the notebook use, rather than
  an ad hoc fixed-fraction placeholder; `calculate_rhs_primordial` is
  self-contained, doesn't need a prior `step()` call to have run,
  matching how `Solver.evaluate_rhs()` already works) and
  `species_names()` (one source of truth for column order, rather than
  the JS side independently hardcoding the same sorted-name order).
- `build.sh`: regenerates the network's C++ source via
  `dengo.primordial_network` (the same shared module every other
  example/tool uses) and compiles to wasm with `em++`.
- `index.html`: sliders for density/temperature/ionized/H2 fraction, a
  mode toggle (mirroring the notebook's two modes), Vega-Lite charts
  (loaded from CDN) that redraw on every slider `input` event,
  coalesced to once per animation frame via `requestAnimationFrame` so
  a fast drag doesn't queue up redundant recomputes -- each recompute
  is a few hundred us to ~250ms (free-fall's ~1300 steps) at the actual
  wasm per-step cost, easily fast enough to feel live.
- `dengo_wasm.js`/`dengo_wasm.wasm` themselves are **committed** (a
  deliberate exception to this repo's usual "don't commit generated
  artifacts" convention -- see `wasm/README.md` for the reasoning:
  unlike a `.so`, this is small, portable, and *is* the deliverable for
  a statically-served page; asking every visitor to install Emscripten
  to view a widget would defeat the point).

**Found and fixed one real bug building the actual page**, caught by
testing in a real headless browser (Playwright driving the system's
installed Chrome, not just Node) rather than assuming the same
correctness proof from the command line transferred automatically: the
Vega-Lite chart panels rendered as blank/zero-width. `width:
"container"` needs the container element to have a measurable width at
embed time, which wasn't reliably true inside the CSS grid layout at
first render; switched to a plain fixed `width: 600` and the charts
render correctly. Verified visually (screenshots) and numerically (the
page-driven free-fall run reproduces the exact same 1298-step, T=2198.0K
trajectory as every earlier from-the-command-line check) for both
modes.

Not yet done (explicitly deferred to the next phase, per the user's
"then move on to the second" -- generalizing this into a real Jinja-
templated wasm codegen path, not hand-written per network): see
`wasm/README.md`'s "Generalizing" section.

**2026-09-09, continued: GitHub Pages CI, the generalized wasm codegen
path, and three fiducial networks -- all three parts of the user's
next request landed together.**

**CI decision, made explicitly as asked ("you can decide... at what
level"):** full regeneration on every push (codegen + Emscripten
compile for every fiducial network), not just republishing whatever
`.wasm`/`.js` happen to be committed. Reasoning: this project's own
history has repeatedly found real, previously-silent bugs in the
generated solver (the H2-gamma bug, the case-sensitive-sort bug, the
step-size-policy bug, ...) -- a demo site that could silently drift
from source because nobody remembered to rebuild and commit new
binaries is a worse failure mode than a few extra minutes of CI time.
GitHub-hosted runners easily afford it: three small single-cell
networks, each a few seconds of codegen + a few seconds of `em++`. A
consequence of this choice: the wasm build artifacts are **no longer
committed to the repo at all** (a reversal of the previous entry's
"deliberate exception" -- CI regenerating them on every push removes
the staleness risk that exception was accepting, so there's no reason
left to keep it).

**Generalized `dengo_wasm.cpp` into a real codegen path**, per the
user's earlier-stated next step: `src/dengo/templates/wasm_solver/
dengo_wasm.cpp.template` (Jinja, mirrors `cython_solver_run.pyx.
template`'s conventions) + `ChemicalNetwork.write_wasm_solver()`.
Refactored `write_cython_solver()`/`write_wasm_solver()` to share a new
`_write_solver_core()` helper (renders `{solver_name}_solver.h`/`.C` +
copies `BE_chem_solve.C` -- identical either way, since that part is
plain, target-agnostic C++; only the thin wrapper around it differs).
Verified the generated `dengo_wasm.cpp` differs from the previous
hand-written version only in comments/wording (one genuine cleanup: two
`#define`s that were dead code, correctly dropped) -- recompiled and
reran the exact same quickstart initial conditions, bit-for-bit
identical result to every earlier check (T=5971.048503743692,
H2_1=0.00012072995205076718, H_1=7600.935053098028). New
`tests/test_codegen.py::test_write_wasm_solver_creates_expected_files`
pins the codegen output shape (file existence/content, not a full
Emscripten compile -- `em++` isn't a repo/CI test dependency).

**Three fiducial networks** (`wasm/fiducial_networks.py`), deliberately
spanning a wide complexity range to prove the codegen path isn't
hand-fitted to one network:
- `primordial` -- the flagship, full H/He/H2 network (reuses
  `dengo.primordial_network.build_network()` directly).
- `primordial_atomic` -- H/He ionization/recombination only (reactions
  k01-k06, cooling terms that don't need H2: recombination, collisional
  excitation/ionization, bremsstrahlung, Compton), no H2 chemistry at
  all -- a genuine contrast case, not just a smaller version of the
  same thing. Confirmed idempotent to call `primordial_rates.
  setup_primordial()` from a second, independent module-level guard
  (plain dict registration, `reaction_registry[name] = self` -- checked
  directly rather than assumed) before relying on that to build this
  network without going through `dengo.primordial_network`'s own guard.
- `hydrogen_minimal` -- H/H+/e- with just k01/k02, no cooling terms at
  all (matches `tests/conftest.py`'s `make_hydrogen_network`) -- the
  simplest network dengo can generate a solver for.

All three generate, compile with `em++`, and run correctly -- checked
directly (not assumed): each converges and gives a physically sensible
trajectory in both modes (constant-density cooldown, free-fall
collapse). `primordial_atomic`'s free-fall run reaches a much higher
final temperature than `primordial`'s at the same target density
(5817K vs. 2198K) -- expected and correct: without H2's cooling/
formation-heating channel, adiabatic compressional heating has nothing
to radiate away. `hydrogen_minimal`'s free-fall run stops early (~4e4
cm^-3 of a 1e15 target, 68 steps) since with zero cooling terms
temperature only ever rises, eventually outrunning what the fixed
Newton tolerance can track at that state -- a real, expected limitation
of this deliberately-minimal network, not a bug, and left as-is (it's
included specifically to show the fast/minimal case, not to reach
astrophysical densities).

**Generalized the site itself**, not just the codegen: `wasm/app.js` is
now shared across all three networks' pages -- builds per-species
initial-fraction sliders dynamically from `dengo_wasm_species_names()`
(nothing hardcoded to a particular species set), uses "ionized fraction"
(`H_2/(H_1+H_2)`) as the universal second chart panel (every fiducial
network tracks H_1/H_2), with H2 fraction as a bonus status-line metric
only when `H2_1` exists. `thermodynamicGamma()`'s existing "H2 gets
7/5, everything else gets 5/3" logic already generalizes correctly to
species sets without H2 at all (the check simply never matches).
`wasm/generate_site.py` orchestrates codegen + compile + per-network
page + landing page for the whole set; `wasm/index.html`/`dengo_wasm.
cpp`/`.js`/`.wasm`/`build.sh` (the single-network, hand-written
precursor from the previous entry) are removed, fully superseded.

Verified the whole site end-to-end with a real headless browser
(Playwright + the system's Chrome) serving the actual generated output,
not just checking each piece in isolation: landing page links to all
three networks; each network's page loads, builds its sliders, solves,
and redraws with no console errors, in both modes. Screenshots
confirm correct rendering (dynamic species sliders matching each
network's actual species list, sensible-looking T/ionization curves).
86/86 tests pass (85 previous + 1 new).

Manual step still required, not automatable from here: enabling GitHub
Pages for this repo (Settings -> Pages -> Source: GitHub Actions) --
the workflow deploys once that's on, not before. Also note the
workflow currently triggers on pushes to `wasm-solver` specifically
(where this work lives right now); update the branch name once this
merges elsewhere.

**2026-09-09, continued: merged to main.** User called this session's
work done and asked to land it: opened a PR (`wasm-solver` -> `master`,
data-exp-lab/dengo#1 -- summarizing the whole arc: packaging/build,
solver correctness fixes, Grackle comparison + compat shim, performance
work, quality-of-life additions, and the wasm codegen path), merged it
(a real merge commit, not squashed, to keep the detailed per-change
history NOTES.md cross-references), then renamed the repo's default
branch from `master` to `main` via GitHub's native rename endpoint
(`POST /repos/{owner}/{repo}/branches/{branch}/rename` -- updates the
default-branch pointer and fully removes the old ref, rather than
leaving a stale `master` alongside a new `main`; confirmed via
`git ls-remote` that `refs/heads/master` is genuinely gone, not just a
redirect). One follow-up fixed in the same motion: `.github/workflows/
gh-pages.yml`'s trigger branch, previously `wasm-solver` with a comment
flagging it as temporary, now points at `main`.

One real snag along the way, worth recording: the git remote is
configured over SSH (`git@github.com:...`), and this sandbox has no SSH
key set up for it -- `git push`/`fetch` failed outright until the
remote was switched to HTTPS and `gh auth setup-git` wired up its
token-based credential helper. Separately, the first push attempt was
rejected because the authenticated token lacked the `workflow` OAuth
scope (needed specifically to push changes under `.github/workflows/`)
-- required the user to run `gh auth refresh -s workflow` interactively
(a browser-based re-authorization, not something scriptable from here)
before the push could succeed.

**2026-09-09, new branch `web-interface-cleanup`: LaTeX axis labels,
step annotation, human-readable time units, dark-mode-aware charts.**
Three specific asks about `wasm/`'s widget, plus a mid-flow one about
dark mode ("I don't use Dark Themes but somebody else I work with
does"):

- **Proper math typesetting.** Vega-Lite axis titles are plain SVG
  `<text>` -- no subscript/superscript/LaTeX rendering at all, which is
  why the old labels were hand-approximated Unicode (`n (cm⁻³)`,
  `H⁺ / H_tot`). Rather than keep extending that approximation, pulled
  in [KaTeX](https://katex.org/) (CDN, `wasm/generate_site.py`'s
  `PAGE_TEMPLATE`) and render each axis's title as real LaTeX
  (`AXIS_LATEX` in `app.js`) into small HTML labels laid out alongside
  each chart (`.axis-y`/`.axis-x` in `style.css`, `.chart-row` markup in
  `PAGE_TEMPLATE`) instead of asking Vega-Lite to draw a title at all
  (`axis: {title: null}`). `.katex { color: inherit }` so the labels
  follow the page's light/dark text color rather than KaTeX's own
  light-mode-only default.
- **Annotate solver steps.** Each entry in the plotted series already
  *is* one accepted adaptive step (one call to the generated `step()`);
  the old spec only drew point markers below a 60-point cutoff, so nearly
  every free-fall run (hundreds to thousands of steps) rendered as a bare
  line with the step-size ramp invisible. Now always draws point markers,
  sized down as the step count grows (`pointSize` in `chartSpec()`,
  `app.js`) so density stays legible instead of turning into a smear, and
  added a tooltip (step index, `x`, `Δt`, the y-field) so hovering a point
  shows exactly what step it is and how big a jump was taken.
- **Auto time units.** The old x-axis for the constant-density mode was
  always raw seconds, unreadable at the kyr-Myr scales this project
  actually runs at. Axis ticks now auto-pick a unit (s/hr/d/yr/kyr/Myr)
  via a Vega expression (`TIME_LABEL_EXPR`) evaluated per tick -- Vega-Lite
  can't call out to arbitrary JS from a spec, so there's a second,
  same-logic JS version (`formatTimeAuto()`) for the tooltip and status
  line, which show the auto-unit value *and* raw seconds side by side
  ("in addition to", not instead of, per the ask). Also threaded actual
  elapsed simulation time through the free-fall path (previously only
  tracked density, not time at all), so even though free-fall's x-axis is
  density, its tooltip/status still report elapsed time.
  One snag: converted tick labels ("254 kyr") are wider than the plain
  numbers they replaced, and on a busy same-decade log axis started
  visibly overlapping ("222 kyr254 kyr") despite `labelOverlap: "greedy"`
  being set -- fixed by angling time-axis labels (`labelAngle: -40`,
  density axis left alone) rather than relying on overlap removal alone.
- **Dark mode.** Chart colors were never actually theme-aware -- Vega-Lite's
  defaults (near-black axis/label colors) were baked into every spec
  regardless of the page's already-dark-mode-aware CSS. Added `vlConfig()`
  (axis/line/point colors picked from `isDarkMode()`) to every chart spec,
  bumped the orange target-temperature-band opacity in dark mode (0.15
  read as nearly invisible against the dark panel background), and added
  a `matchMedia` `change` listener so flipping the OS theme live redraws
  the charts -- everything else on the page already follows the OS
  preference via CSS custom properties and needed no JS at all.

Verified the same way as the original wasm work: a real Emscripten build
of all three fiducial networks plus a real headless-browser pass
(Playwright + system Chrome) over every page in both `colorScheme: light`
and `colorScheme: dark`, checking (a) zero console errors, (b) the exact
same physics results as every prior check in this file (primordial
free-fall T=2198.5K, atomic T=5817.6K, hydrogen_minimal stopping at 68
steps) -- confirming the tooltip/dt/t plumbing didn't touch the actual
solver path, (c) KaTeX genuinely rendered (not left as raw LaTeX source
or a red KaTeX error span), and (d) the angled tick-label fix by
rendering and reading back actual tick text from the DOM.

**2026-09-09, same branch: a third "species abundance" chart.** Follow-up
ask -- T and the H+/H_tot ratio were the only plotted quantities, so
there was no way to see actual per-species number densities or how many
decades they span ("I can't really get an impression of the full range
of values"). Added:

- A per-step full-species snapshot (`getScalar()`'s result) is now kept
  alongside the existing T/ion/h2/dt/t histories in both
  `runConstantDensity()`/`runFreefall()` (`sHist`), not just the couple
  of derived scalars those already computed.
- A row of species-toggle checkboxes (`buildSpeciesToggle()`), one per
  species except `ge` (which already has its own Temperature chart) --
  "all"/"none" buttons plus individual toggles, defaulting to all on so
  the initial view is the full dynamic range across every tracked
  species, per the ask.
- A third chart (`speciesChartSpec()`), long-format (one row per
  step x selected-species) multi-series line/point plot, log-scale y so
  the ~20+ decade spread between trace and dominant species is legible
  in one view.
- Each species gets a fixed color (`SPECIES_COLORS`, d3's category10,
  indexed by a stable position in `plotableSpecies()` so a species keeps
  its color regardless of which others are toggled on) that the toggle
  checkboxes carry as a swatch -- the checkboxes double as the chart's
  legend, so Vega-Lite's own legend is turned off (`legend: null`) rather
  than showing the same mapping twice.

Verified the same way as the rest of this branch: real Emscripten build
of all three fiducial networks, headless-browser pass (light + dark)
confirming (a) checkboxes are built from the actual compiled solver's
species list (not hardcoded), (b) toggling "none" shows a placeholder and
"all" restores the full multi-series plot, (c) toggling one species off
and back on doesn't disturb the others or throw, (d) zero console errors,
(e) same physics numbers as ever. `hydrogen_minimal`'s `H_2` and `de`
lines visibly coincide in the screenshot -- expected, not a bug: in a
pure H/H+/e- network, charge neutrality makes electron density exactly
equal to H+ density.

**2026-09-09, same branch: mass fraction toggle + temperature/thermal
energy toggle, both with gamma exposed.** Two more asks about the same
widget:

- **Species abundance: number density vs. mass fraction.** Added a
  `SPECIES_MASS_AMU` table (H=1.00794, He=4.002602, H2=2.01588,
  de=5.485799e-4) keyed by each species name's base element symbol (the
  part before the first `_` -- `H2_1` -> `H2`, `He_3` -> `He`, `de` has
  no underscore and is its own base; this parses cleanly for every
  species the three fiducial networks produce). `X_i = n_i m_i / sum_j
  n_j m_j`, denominator recomputed per step over every species with a
  known mass (electrons included, for self-consistency, though
  negligible). A species with no recognized mass (a hypothetical future
  network) is silently excluded from mass-fraction mode rather than
  breaking it. **Verified the definition, not just the code path**: summed
  every plotable species' mass fraction at a fixed state across all three
  networks and got 1.0 (to float precision) in every case.
- **Temperature vs. thermal energy, with gamma drawn out.** Added a
  second toggle switching the first chart's y-field between `T` (K) and
  `ge`, the specific internal energy (erg/g) the solver actually
  evolves -- `T` is a *derived* quantity via `T = (gamma-1) ge mu / k_B`,
  so this is the direct way to see gamma's effect: `ge` alone doesn't
  show composition changes bending the T trajectory (e.g. gamma easing
  from 5/3 toward 7/5 as H2 forms), because ge's own trajectory doesn't
  encode that ratio. To make gamma itself visible rather than just
  implied, `thermodynamicGamma()` (already used internally to evolve
  `ge` through a free-fall compression step) is now also computed per
  plotted step and exposed as an extra tooltip field on every point,
  regardless of which of T/ge is displayed. The target-temperature band
  (1500-2500K) is correctly display-only for `T` mode -- it has no
  meaningful equivalent in `ge` without redoing the gamma-dependent
  conversion, so it's just omitted rather than shown misleadingly.
- Both toggles reuse one small button-pair component (renamed
  `.species-mode` -> `.mini-modebar` in `style.css` since it's no longer
  species-specific) for a consistent look with the existing cool/free-fall
  mode buttons.

Verified the usual way: real Emscripten build of all three fiducial
networks, headless-browser pass confirming (a) the mass-fraction identity
above, (b) both toggles' button active-states and axis-label swaps land
correctly, (c) the temperature-band layer is actually absent in `ge`
mode (not just hidden), (d) switching modes and back, plus running a full
free-fall in the "thermal energy" + "mass fraction" combination, changes
nothing about the underlying physics (same step counts/final T/ionized
values as every prior check in this file), (e) zero console errors.

**2026-09-09, feasibility only, not implemented: per-cooling-process
breakdown of the energy budget.** User asked how hard it would be to see
each named cooling/heating process's (ceHI, brem, compton, gloverabel08
H2-line cooling, h2formation, cie_cooling, etc.) fractional contribution
to the total `dge/dt`, rather than just the summed total. Investigated
before writing any code (two background agents read the actual codegen
pipeline and a freshly-generated `.C` file rather than guessing):

- Every cooling process is already a separately-named, self-contained
  sympy expression (`primordial_cooling.py`'s `@cooling_action`
  decorator), registered individually in `ChemicalNetwork.cooling_actions`
  with enough metadata (name, equation, species deps, rate-table names)
  to regenerate per-term output cleanly.
- The *only* place separability is destroyed is
  `ChemicalNetwork.print_cooling()` (`chemical_network.py`), which sums
  every term into one sympy expression *before* calling `ccode()` once --
  confirmed by generating the real primordial `.C` file and reading the
  emitted `ge` RHS: one fused `rhs[j] = -brem*(...) - ceHI*(...) - ... +
  h2formation*(...);` statement, no per-term C variable survives codegen.
  No CSE/algebraic fusion is involved (plain `sympy.sympify("0")`
  accumulation then one `ccode()` call), so nothing structurally blocks
  un-fusing it.
- Assessed as **moderate effort**: (1) a sibling to `print_cooling()`
  that calls `ccode()` per-term instead of summing first (small,
  mechanical -- same conditional `cie_optical_depth_approx` logic just
  applied per term); (2) the shared C template emits the per-term values
  into a new output array alongside the existing summed `rhs[j]`, plus
  one new exported wasm getter (boilerplate, mirrors the existing
  `dengo_wasm_rhs_ptr()`); (3) JS/UI reuses the species-abundance chart's
  existing pattern almost exactly (per-step snapshot, toggle checkboxes,
  multi-series chart).
- One real open design question, not a blocker: several terms have mixed
  signs (`h2formation` is usually heating; `compton` flips sign depending
  on gas T vs. T_CMB), so "fractional contribution to the cooling budget"
  needs a choice -- split cooling/heating into two separately-normalized
  stacked-area budgets (recommended -- matches the literal "cooling
  budget" framing), or one signed chart of each term's share of net
  `|dge/dt|`.

User: "I'll hold off on it." Not implemented. Recorded here so the
investigation (and the concrete plan) doesn't need to be redone if this
comes back up later.

**2026-09-09, same branch: a few sets of initial conditions to select
from.**

For this one: added an "Initial conditions" dropdown (`#ic-preset`,
`applyPreset()` in `app.js`) with four presets -- IGM background at
z≈20 and z≈1000 (mean cosmic density `n_H ∝ (1+z)³`, an adiabatic-cooling
estimate for gas temperature at z=20 vs. the tightly Compton-coupled
~T_CMB at z=1000), a virial shock (Barkana & Loeb 2001 T_vir fit and
Δ_vir≈178× the cosmic mean, for a 10⁶ M☉ minihalo at z≈20 -- the classic
first-star-forming halo scale), and a primordial protostellar disk
(disk-forming density/temperature, mostly-molecular hydrogen). Each
preset just sets the nH/T/species-fraction sliders to specific numbers
and calls the existing redraw path -- no new solver-side machinery.

**Redshift was intentionally *not* wired up as a live parameter.** The
first design considered actually exposing `current_z` from the wasm
wrapper (confirmed feasible -- the field already exists end-to-end,
read by the `compton` cooling term, and the Cython path already exposes
it via a `redshift` property/constructor arg; the wasm wrapper just never
got a setter). User caught this before it was built: "I'm not sure I want
to have it actually expose redshift. I just meant the background
density. We don't need to add UV background etc." So a preset's "z" is
just a label/note for where the density number came from, not a live
knob -- the compton term still always runs at z=0, same as before this
change; no new wasm export, no template change, no redshift slider.

Two real things found and fixed along the way:
- **Unphysical hydrogen budget on H2-less networks.** The protostellar
  disk preset's `H_1=0.05` assumes ~0.74 more is locked up in `H2_1` --
  fine for `primordial`, but `primordial_atomic`/`hydrogen_minimal` have
  no H2 species to hold that mass, so applying it as-is would make ~74%
  of the hydrogen budget just vanish. Fixed by folding molecular-hydrogen
  fractions back into `H_1` whenever the current network has no
  `sp-H2_1` slider (`applyPreset()`), landing on the same ~0.79 baseline
  every other preset already uses.
- **Solver diagnostic noise on stderr.** Presets that start further from
  equilibrium than any default IC before them (expected -- that's the
  point) pushed `BE_chem_solve.C`'s adaptive stepper into more
  step-halving retries than this project had exercised before, each
  logging a `dt < 1.0` "Unsolved[...]" progress line -- pre-existing
  code (not introduced by this change), present in every build, not a
  correctness issue (final results were sane throughout), but it was on
  `fprintf(stderr, ...)`, which Emscripten routes to `console.error` --
  breaking this project's own "zero console errors" verification bar for
  the first time not because of a real bug, but because nothing had
  stressed the solver into this path before. This is a genuine mid-sweep
  progress message (the file's real hard-failure path, "unsolved case in
  Gauss_Elim", correctly stays on stderr), and the file already uses
  plain `printf`/stdout for its other debug output a few lines above, so
  `stderr` here looks like a copy-paste inconsistency -- changed both
  occurrences to `stdout`. This touches a file shared by every build
  (Cython and wasm alike); reran the full `pytest` suite (86/86) after
  the change to confirm nothing depends on it.

Verified: real Emscripten build of all three fiducial networks, every
preset applied and run in both modes on every network (24 combinations),
confirming (a) sliders/notes update correctly and the dropdown starts
disabled until the wasm module is ready, (b) zero console errors after
the two fixes above (was noisy before them), (c) sane, cross-checkable
physics -- e.g. `bg-z20`, `bg-z1000`, and the default IC all independently
converge to the same ~2198K H2-cooling equilibrium at n≈10¹⁵ cm⁻³ during
free-fall despite starting from very different states, and
`hydrogen_minimal` (a network with zero cooling terms by construction)
shows essentially no temperature change from the protostellar-disk
preset in constant-density mode, as it should.

**2026-09-09, new branch `wasm-panel-layout`: molecular fraction gets
equal billing, sliders panel stays reachable while charts stay visible.**
Two asks: molecular (H2) fraction was only ever a status-line footnote,
not plotted; and a long sliders panel (IC presets, mode, nH/T, 6-9
per-species sliders, dtf/target, status) meant dragging a lower slider
scrolled the charts above it out of view. Explicitly asked to evaluate
switching to a reactive UI framework/toolkit (SvelteKit/Skeleton/etc)
before doing anything -- recommended against it and said why: the
layout/scroll problem isn't a reactivity problem (redraw-on-every-drag
already works exactly as wanted), it's a CSS layout problem; adopting a
framework would mean a Node/npm build step added to a currently pure
Python+Emscripten CI job, a bundler, and converting ~500 lines of plain
JS into components, for zero benefit to the actual complaint. User agreed
("Yeah, do this") to the proposed cheaper plan instead:

- **Ionization & molecular fraction, one chart.** Both are "fraction of
  total hydrogen", so instead of adding a *fourth* chart panel (which
  would make the scrolling problem worse, working against the rest of
  this), folded H2 fraction into the existing Ionization chart as a
  second line (`ionizationChartSpec()`, long-format rows with a
  `quantity` field holding the actual legend label string). Real
  Vega-Lite legend used here (not the species chart's checkbox-legend
  pattern) since the set is small and fixed -- two known series, not a
  dynamic per-network list.
  **Bug caught by testing, not assumed away**: the color scale's
  `domain` was originally hardcoded to both labels unconditionally,
  which drew an "H₂ / H_tot" legend entry even on `primordial_atomic`/
  `hydrogen_minimal` (no H2 species, no h2 rows ever pushed) -- a legend
  entry for a line that's never drawn. Fixed by building `domain` from
  the labels actually present in that call's `rows`.
- **Collapsible species-fraction sliders.** Wrapped in a native
  `<details>`/`<summary>` (`.species-details` in `generate_site.py`) --
  zero JS, built into every browser, no new dependency. Open by default
  so nothing changes for anyone who never notices it; the single biggest
  contributor to panel height, so collapsing it does the most to shrink
  the panel when you don't need it.
- **Sliders panel is now an independently-scrolling sticky sidebar**
  (`position: sticky` + `max-height: calc(100vh - 32px)` +
  `overflow-y: auto` on `.panel`), reset to normal static flow on the
  existing narrow-screen media query. First attempt was `position:
  sticky` on the *charts* column instead (the originally-stated ask,
  "still see some of the charts") -- verified with real height
  measurements in a headless browser that this did nothing at all: sticky
  only has room to act while its grid row is taller than the sticky item,
  and the charts column turns out to *be* the taller column here (three
  chart boxes vs. a sliders panel), so it was already at its natural
  position with no slack to hold onto. Making the *shorter* column
  (the panel) the sticky+capped+internally-scrolling one instead gives a
  strictly better result than the original ask: not only do the charts
  stay put while scrolling to a lower slider, the *sliders panel* also
  stays put while scrolling down to see a lower chart -- both panels
  visible/reachable simultaneously regardless of which one you're
  scrolling.

Verified: real Emscripten build of all three fiducial networks,
headless-browser pass (light + dark) confirming (a) real height/position
measurements (not just visual impression) proving the sticky-panel
approach actually holds its position while the page scrolls and the
naive sticky-charts attempt didn't, (b) the bottom-most slider (`dtf`)
is reachable via the panel's own internal scroll while the page is
scrolled to show the last chart, and the panel stays visible throughout,
(c) the ion/H2 chart legend matches what's actually plotted on every
network (fixed after being wrong), (d) `<details>` collapse verified by
screenshot after `getComputedStyle().display`/`offsetParent` checks gave
false negatives (modern Chrome doesn't hide `<details>` content via a
plain `display: none` on the child -- an internal UA mechanism handles
it, so those specific DOM checks aren't reliable evidence either way;
the rendered pixels are), (e) zero console errors, (f) same physics
results as every prior check in this file. Full `pytest` suite (86/86)
also passes.

**2026-09-09, same branch: `generate_site.py` fail-fast fix (found while
walking the user through testing locally).** User followed the
"build locally" instructions in `wasm/README.md` and got a bare
directory listing (`app.js` and friends, no page) when serving
`wasm/_site/`. Root cause: `em++` wasn't on `PATH` in the shell they ran
`generate_site.py` from (hadn't re-sourced `emsdk_env.sh`), and
`find_emxx()` calls `sys.exit(1)` when that happens -- `SystemExit`
isn't an `Exception`, so it wasn't caught by `main()`'s per-network
`try/except Exception`, and the whole process died immediately after
finishing `primordial`'s *codegen* but before compiling it (confirmed:
`wasm/_site/primordial/` had the generated `.C`/`.h`/`.bin` files but no
`dengo_wasm.js`/`.wasm`, and no `index.html` anywhere -- `primordial_atomic`/
`hydrogen_minimal` were never attempted). The script did print a clear
"install/activate the Emscripten SDK" message, but to stderr, easy to
miss, and by then a half-built, confusing `_site/` already existed.

Fixed by calling `find_emxx()` once at the very top of `main()`, before
`out_dir` is even created -- verified this now fails fast with nothing
written at all (rather than a partial build) when `em++` is missing, and
still builds all three networks correctly (confirmed `index.html`
present at every level, including the landing page) when it's present.
Full `pytest` suite (86/86) still passes.

**2026-09-09, new branch `wasm-freefall-target-density`: configurable
free-fall target density, up to 10²⁰ cm⁻³.** Ask: widen the free-fall
mode's target-density slider (previously capped at 10¹⁸) up to 10²⁰,
explicitly past the point where the simple single-zone free-fall model
is a realistic dynamical model, to try to resolve H2's collisional-
dissociation phase. Delivered: slider `max` raised 18→20,
`runFreefall()`'s `maxSteps` bumped 5000→10000 (headroom -- reaching
10²⁰ takes ~1870 steps from the default IC, comfortably under either
cap but not the original one by a huge margin), and a note under the
slider explaining the ~10¹⁶ cm⁻³ physics-relevance caveat and naming the
actual dissociation reactions (`H2 + H -> 3H`, `H2 + H2 -> 2H + H2`,
i.e. `k13`/`k23` in `primordial_rates.py`) so it's concrete, not vague.

**Important finding, reported rather than papered over**: a diagnostic
sweep (`runFreefall()` called directly at logNTarget = 15 through 23,
bypassing the slider) shows this simplified model does *not* actually
resolve H2 dissociation by 10²⁰ cm⁻³ -- H2/H_tot is still ~99.6%
molecular there (T≈2808K), and the trend from 10¹⁵ up through 10²¹ is H2
fraction *climbing* toward ~99.9%, not collapsing toward dissociation.
This is very plausibly because the free-fall loop's energetics are a
bare adiabatic-index heating estimate (`thermodynamicGamma()`-weighted
PdV work) with no shock/radiative-transfer treatment -- real "second
collapse" H2-dissociation physics (Palla, Salpeter & Stahler 1983;
Omukai 2000; Yoshida et al. 2006) relies on the dissociation reaction
itself acting as a thermostat that absorbs compressional heating over a
huge density range, which this simple energy-update doesn't capture.
Separately, the same sweep found the integration stops advancing at all
beyond logNTarget≈21.17 (n≈1.49×10²¹, identical final state whether
logNTarget=22 or 23 was requested) -- `!converged` from `step()` is
presumably firing and breaking the loop permanently; not investigated
further, since it's past what was actually asked for (10²⁰), but worth
knowing this model has a real ceiling near there regardless of slider
range.

Delivered exactly what was asked (10²⁰ now reachable, converges cleanly,
zero console errors, same-as-ever physics below 10¹⁶, full `pytest`
86/86) -- but flagged this to the user rather than claiming the stated
physics goal (seeing dissociation) is met, since the data doesn't show
it yet at 10²⁰. Fixing that would mean improving the free-fall energy
update itself, a separate, bigger piece of work not undertaken here.

**2026-09-09, new branch `wasm-shock-heating`: parameterized accretion-
shock heating in free-fall mode -- H2 dissociation is now actually
visible.** Followed a speculative discussion (not logged per the user's
"stop appending to NOTES for a bit, we're going to speculate" -- covered
here in one entry now that it moved from speculation to implementation).

**Diagnosis first**: checked whether the chemistry network itself
correctly costs dissociation energy before assuming a driver-level fix
was even the right lever. It does -- `primordial_cooling.py`'s
`h2formation`/`h2formation_extra` cooling actions already have a
correctly-signed `h2mcool`/`h2mcool_extra` term (energy sink, matching
reactions k13/k23) alongside the formation-heating term. The gap is
purely in the one-zone free-fall *driver*: its per-step compressional
heating uses a caloric `thermodynamicGamma()` (just the H2-vs-atomic
degrees-of-freedom mix) with no knowledge that dissociation is actively
consuming that same compressional work as latent heat -- the classic
"generalized adiabatic exponent Γ₁ dips during an ionization/dissociation
zone" effect, well known in stellar structure, that a bare caloric gamma
doesn't capture. That's why raising the free-fall target density alone
(previous entry) never showed dissociation: the driver was always
over-heating relative to a properly-coupled solve.

**Feasibility confirmed before implementing**: this needn't touch the
compiled solver at all. `calculate_rhs_*`/`BE_chem_solve` have no concept
of free-fall, density, or heating laws -- they just integrate whatever
state they're handed for a given `dt`. All of the free-fall-specific
physics (today's compression law, and now the shock) lives entirely in
`runFreefall()`, manipulating the exposed `statePtr()` buffer directly in
JS before calling the existing `step()`.

**Design, per user's own framing** (energy injection vs. a parameterized
shock -- explicitly asked to evaluate both): recommended the shock. Pure
energy injection would only make the over-heating worse, not fix it; the
"correct" fix (folding latent heat into an effective gamma every step) is
real physics work, not a quick exposed dial. A parameterized accretion
shock at a chosen density is well-precedented for exactly this kind of
one-zone exploration (Omukai & Nishi 1998; Ripamonti & Abel 2004 -- "free-
fall until some density, then switch regime" instead of real radiation-
hydrodynamics), and naturally has the two free parameters the user
anticipated.

**Implementation** (`runFreefall()`, `app.js`): standard (not strong-limit)
Rankine-Hugoniot jump conditions for an ideal-gas shock, in terms of
upstream Mach number and the gas's own composition-weighted gamma at that
moment:
```
rhoRatio = (gamma+1)*M^2 / ((gamma-1)*M^2 + 2)
TRatio   = (2*gamma*M^2 - (gamma-1)) * ((gamma-1)*M^2 + 2) / ((gamma+1)^2 * M^2)
```
correctly giving no jump at all at M=1 (the zero-strength/sonic limit --
used as the "disable" state, no separate toggle needed) and a density
ratio that saturates with Mach number while temperature keeps climbing
(the real strong-shock behavior). Applied as a single-step multiplicative
jump the first time ordinary free-fall compression would carry the gas
across the shock density -- deliberately *not* smoothed out: a shock is a
genuine mathematical discontinuity (that's what RH conditions describe),
so a sharp jump is the physically honest choice, and `BE_chem_solve`
already tolerates state jumps of this kind fine (ordinary free-fall
compression hands it one every step already). Two new sliders: shock
density (10¹⁰-10²⁰ cm⁻³, default 10¹⁴) and Mach number (1-100, default 5).
Mach number is a free dial here, not derived from an actual radius/mass-
dependent infall speed (this zero-dimensional density-only model has no
way to compute one) -- the UI note says so explicitly, framed as "how
strong a shock would it take", not a prediction of where/how strong a
real one occurs.

**Bug caught by a diagnostic sweep before calling it done**: the
`shocked` flag was being set the moment a jump was *attempted*, before
knowing whether the subsequent `step()` call would actually converge. A
strong-enough jump can fail to converge on the first attempt (found this
directly: `nshock=14, mach=60` and several `nshock=16` cases), and the
loop then breaks immediately -- meaning the run stops *before* the
density the flag claimed had been crossed. Fixed by only committing
`shocked = true` (and the reported `shockTriggered`) after that step
actually converges; verified the flag now correctly reads `false` for
every case where the final recorded density is still below the
requested shock density.

**The physics goal is now demonstrably met**: a sweep over
(shock density, Mach) found `nshock=1e14, mach=5` gives a clean, fully-
converged run where H2/H_tot crashes from ~0.9 down to **1.3×10⁻²** right
at the shock (visible directly in the default view, no parameter hunting
needed) before re-forming as compression continues past it -- set as the
new default. Also found the effect is genuinely non-monotonic in Mach
number for this coupled nonlinear system (mach=5 gives a dramatic dip,
mach=10 a partial one, mach=30 almost none, mach=60+ fails to converge
at all at this shock density) -- not investigated further, but worth
knowing before assuming "stronger shock = more dissociation" here.

A vertical dashed marker at the shock density is drawn on the Temperature
chart only (not the other two, to keep this well-scoped) when the shock
actually fired, and the status line reports the density it crossed.

Verified: real Emscripten build of all three fiducial networks,
headless-browser pass (light + dark) confirming (a) `mach=1` reproduces
the pre-existing free-fall behavior exactly (bit-for-bit same step count/
final T/ionized/H2 as every prior check in this file) -- the "disabled"
state is a genuine no-op, not an approximation, (b) the shock fires and
is visible (screenshot-verified: the Temperature chart's jump and
marker, and the Ionization & molecular fraction chart's H2 crash-and-
recover, both line up at the shock density), (c) `shockTriggered`/the
status line's "shock crossed" message only appear when the jump actually
converged, (d) zero console errors, (e) full `pytest` suite (86/86)
passes. One real process note: a rebuild step was believed complete
(files present, `succeeded: [...]` printed) but had actually run *before*
a source edit landed, serving stale slider defaults for a while --
caught by comparing file mtimes against the edit timestamp rather than
trusting `ls`/exit codes alone; worth remembering that `generate_site.py`
must actually be re-invoked (and its exit code/output checked) after
every source change, not just checked for output *existing*.

**2026-09-09, new branch `wasm-parameter-sweep`: overlay several full
runs at once, to compare starting conditions.** Discussed as a
speculative "what if" first (not logged, per the same "we're going to
speculate" pause as the shock-heating discussion) -- landed on comparing
several *starting conditions* (the user's own motivating case: several
initial temperatures, "to see differential evolution from initial
conditions" -- does the gas forget where it started, or remember it)
rather than my own first suggestion (sweeping the new shock Mach
number), with lines-overlay recommended over a heatmap (a heatmap needs
every track resampled onto a shared x-grid our adaptive, run-length-
varying steppers don't naturally produce, and is better suited to a true
2-parameter sweep anyway).

**Design**: a new, collapsed-by-default `<details>` section below the
existing three charts (`.sweep-section` in `generate_site.py`) -- holds
every slider at its current value except one, samples `SWEEP_N=6` values
of that one across its existing range, runs the current mode
(cool/free-fall) once per value, overlays all of them on two new charts
(Temperature, H2 fraction). Deliberately *not* live: a sweep is several
full runs, not one (up to ~800ms for 6 free-fall runs to 10²⁰ in testing
-- see below), so it's a "Run sweep" button, not another slider.
Sweepable parameters (`SWEEP_PARAMS`): T₀, n_H,0, initial H2_1 fraction
(only offered when the current network actually has that species),
shock Mach, shock density -- adding a parameter to the picker means
adding one entry to this table, no other code changes, since the sweep
loop reads/writes whatever slider element the config points at exactly
the way `redraw()` itself does. Sampling is log-spaced physical values
for every log-scale slider (T, nH, H2 fraction, shock density already
*store* log10(value), so linear interpolation of their own min/max
already *is* log-spaced sampling) except Mach, whose slider is linear-
in-Mach -- that one needs an explicit log-space construction, so the
interesting no-effect-to-saturated transition (see the shock-heating
entry above) gets more than one sample point instead of being spread
evenly across 1-100.

Colors: an *ordinal* (not nominal) scale with an explicit ascending-value
domain and Vega-Lite's `viridis` scheme -- unlike species names, a swept
parameter has a natural order, and a sequential palette shows "low to
high" at a glance the way the species chart's arbitrary categorical
palette shouldn't/doesn't need to.

**Bug caught by looking at the actual per-track data, not just eyeballing
the chart**: the first version had no point markers (species-chart-style
reasoning: "steps are already shown on the single-run charts, this is
about comparing shapes"), and several tracks in the default T-sweep
(T₀=10K, 55K, 302K) rendered as *nothing at all* -- not a thin line, no
data. Checked directly (calling `runConstantDensity()` for those exact
values outside the UI) rather than assuming a rendering bug: each
produces exactly **one** point. `coolingTime()`-based adaptive stepping
legitimately jumps straight to the requested end time in a single step
when the starting temperature is far enough from equilibrium that the
estimated cooling/heating timescale dwarfs the whole run -- a correct,
converged answer, just not a multi-point curve. A line mark with one
point draws nothing, so that track silently vanished. Fixed by adding
small point markers back (unlike the single-run charts, where they're a
bonus annotation, here they're load-bearing: without one, a legitimately-
one-point track has no visual representation at all).

Verified: real Emscripten build of all three fiducial networks,
headless-browser pass (light + dark) confirming (a) the sweep-parameter
list correctly omits H2 fraction on the two H2-less networks, (b) every
slider's value (and the main charts' displayed state) is correctly
restored to its pre-sweep setting afterward -- confirmed by comparing
full status-line text before/after (identical aside from the timing
substring, i.e. bit-for-bit same physics), not just spot-checking one
field, (c) legend entries render in ascending physical-value order, (d)
the single-point-track visibility fix (screenshot-verified: those tracks
now show as small dots), (e) zero console errors across all three
networks/both themes/both modes, (f) full `pytest` suite (86/86) passes.

**2026-09-09, same branch: made the sweep's slider-to-range mapping
explicit instead of implicit.** User pushback, and a good catch: the
first version silently sampled the swept slider's *entire* min/max range
regardless of what it was actually set to -- correct once you know that,
but nothing in the UI said so, and it threw away whatever value you'd
carefully set. Fixed exactly as proposed: picking a sweep parameter now
disables its regular slider (visibly, via a plain `input:disabled`
dimming rule -- it's genuinely not readable from that slider while the
sweep drives it) and reveals an explicit from/to/count range, pre-filled
with that slider's own min/max in physical units (a visible, editable
default, not a hidden one) rather than requiring the user to already
know what range they wanted before touching anything.

Reworked `SWEEP_PARAMS` so every entry carries both `toPhysical` and
`toRaw` (physical <-> slider-raw-unit conversion) plus a `logSpace`
flag; `sweepValues()` now reads the visible from/to/count inputs
(always physical units, e.g. actual Kelvin, not log10(Kelvin)) and
produces evenly-log-spaced or evenly-linear-spaced raw values between
them, converting only at the boundary. This also *simplified* the Mach
special-case from the previous entry: since the user now sets Mach's
own from/to explicitly, the sweep no longer needs to hardcode "always
sample 1-100" -- log-spacing within whatever range is actually entered
covers the same "resolve the low-Mach transition" need without a
hardcoded range.

Verified: real Emscripten build, headless-browser pass across all three
networks/both themes confirming (a) the initially-selected parameter's
slider is disabled and its range pre-filled immediately on page load
(not only after touching the dropdown once), (b) switching the sweep
parameter re-enables the previous slider and disables/re-fills the new
one, (c) a narrowed custom range (500-2000 K, count=4) produces exactly
the 4 expected log-spaced legend values -- not the old full-range
behavior -- confirming the fix actually changes behavior, not just the
label, (d) a full 21-decade n_H sweep (10⁻⁴ to 10¹⁷ cm⁻³) still runs
cleanly, (e) zero console errors, (f) full `pytest` suite (86/86)
passes.

**2026-09-09, same branch, user-reported bugs: "No sweep" default state,
sweeping 10-2000K gave one visible track instead of six, and (found
along the way) charts never plotted the initial condition at all.**
Three related fixes, landed together since the second two turned out to
be entangled:

- **Explicit "No sweep" option.** `#sweep-param`'s first, default-
  selected option is now `"No sweep"` -- nothing disabled, from/to/count
  inputs cleared and inert, "Run sweep" off, rather than always having
  some parameter pre-armed (and a slider silently pre-disabled) from the
  moment the page loads.
- **The "only one result" bug, reproduced and root-caused, not
  guessed at**: called `runConstantDensity()` directly for the exact
  reported values (T0=10,55,302,...,2000K) and found 5 of 6 converge in
  **exactly one step** -- `coolingTime()`-based adaptive stepping
  legitimately jumps straight from t=0 to t=dtfTotal in a single step
  when the local cooling/heating timescale is much longer than the
  whole run (correct and efficient for a *single* run), which renders as
  a single point with zero visible shape -- exactly wrong for a
  *comparison* chart, whose entire purpose is showing several tracks'
  shapes against each other. Fixed by giving sweep runs a dedicated
  `forSweep` path in `runConstantDensity()`: a fixed, shared, log-spaced
  24-point checkpoint grid, capping `dt` at each checkpoint regardless of
  what the adaptive estimate would prefer, leaning on `BE_chem_solve`'s
  own internal sub-stepping (already robust to exactly this kind of
  externally-imposed jump -- the unforced path already relies on the
  same robustness for its own, usually much larger, single jump).
- **Include the initial condition as the first plotted point**
  (suggested mid-fix, and directly related: it's part of why the sweep
  bug read as confusing rather than obviously "these tracks are just
  fast") -- `runConstantDensity()`/`runFreefall()` now both push the
  state immediately after `setIcs()`, before any `step()` call. Free-
  fall's x-axis (density) is always positive, no complication; but
  time can now legitimately be exactly 0, which a `type: "log"` x-scale
  cannot render at all -- switched every time-axis chart (not
  density-axis ones, which don't need it) to `type: "symlog"`
  (Vega-Lite/Vega's linear-near-zero, log-further-out scale), verified
  by screenshot that t=0 renders sensibly at the left edge rather than
  being silently dropped.

**A second, deeper, pre-existing bug found while verifying the above,
not introduced by it**: adding the initial point required calling
`temperature()` right after `setIcs()`, which surfaced that
`temperature()` reads a *cached* value `calculate_rhs`/
`calculate_jacobian` only refresh as a side effect -- `setIcs()` writes
species/`ge` directly and triggers neither, so the "initial" temperature
was actually reporting whatever the *previous* run last left behind.
Fixed narrowly by calling `rhsPtr()` (already used by `coolingTime()`,
runs `calculate_rhs` as a side effect) once right after `setIcs()` in
both run functions, before reading `temperature()`.

That fix then exposed a **third, separate, genuinely pre-existing bug,
flagged rather than silently fixed or ignored**: with the cache
correctly refreshed, `primordial`'s initial point reads **1220K** for a
*requested* 1000K -- a 22% discrepancy. Root cause, confirmed by
reading `setIcs()`: `ge = 1.5*KB*T/MH` assumes pure monatomic hydrogen
(mean molecular weight mu=1) unconditionally, completely ignoring the
actual He/H2 composition passed to it -- for any mixture with real He
content (`primordial`, `primordial_atomic`; NOT `hydrogen_minimal`,
which has no He and mu=1 exactly, matching), the solver's true starting
temperature has silently differed from whatever the T slider displayed
for this entire project's history, invisible until now purely because
nothing ever plotted the instant-after-`setIcs()` state before. Confirmed
directly: `hydrogen_minimal` (mu=1, no He) reads 8017.8K for a requested
8000K (0.2%, consistent with minor residual effects like electron
counting) vs. `primordial`'s 22% mu-driven miss. This is a real,
separate correctness issue, not scope-crept into this fix -- noted here
and flagged to the user rather than either quietly patched (risk of
guessing the wrong mu convention under time pressure, since it needs to
match whatever the compiled solver's own T-from-ge code actually does)
or swept under the rug.

Verified: real Emscripten build of all three fiducial networks,
headless-browser pass (light + dark) confirming (a) "No sweep" is the
genuine default (button/inputs disabled, no slider pre-disabled), (b)
selecting/deselecting a sweep parameter correctly enables/disables the
right slider and range inputs each time, (c) the exact reported bug
scenario (T, 10 to 2000, count 6) now shows all six tracks as real,
distinct, multi-point curves (screenshot-verified -- previously only one
of six was visible), (d) the initial-condition point now appears
correctly on every chart, at the correct value once the cache-refresh
fix landed, (e) zero console errors across all three networks/both
themes, (f) full `pytest` suite (86/86) passes.

**2026-09-09, new branch `wasm-invert-ge-from-temperature`: fixed the
mean-molecular-weight bug flagged in the previous entry, per the user's
own proposed approach.** Asked to gauge whether inverting the solver's
own T-from-ge conversion needed a proper symbolic derivation (sympy
producing a real f(T) -> ge for the actual state vector) versus
something cheaper; user's own suggestion -- "a reasonably fast binary
search to invert it... I don't know that we need to invert the
equation" -- is exactly what got built, and for exactly the reason it's
the right call here: `ge -> T` is monotonic (more thermal energy per
unit mass means higher temperature, for any fixed composition) but not
simply linear -- gamma itself varies with T for H2-bearing gas
(rovibrational degrees of freedom activating), so a closed-form
inversion isn't a one-liner, and re-deriving/guessing the solver's own
mu/gamma(T) convention in JS risked exactly the "subtly wrong duplicate
logic" concern raised when this bug was first flagged. Bisection
sidesteps that entirely: it never needs to know the formula, only to
evaluate it (already possible via `rhsPtr()` + `temperature()`).

`geForTemperature(targetT)` (`app.js`) brackets around the old naive
monatomic estimate (2 decades either side -- generous enough for these
networks' actual mu range, up to ~4x for pure He, and gamma 5/3 down to
7/5), widens further if that guess isn't enough (guarded, capped at 20
expansions), then bisects (50 iterations, 1e-8 relative tolerance) by
writing a candidate `ge` directly into the state buffer, calling
`rhsPtr()` (runs `calculate_rhs`, refreshing the cached T-from-ge
conversion as a side effect -- already relied on for the previous
entry's cache-refresh fix), and reading `temperature()`. `setIcs()` now
sets every other species first (needed for the composition the
bisection evaluates against), then calls this instead of the old
`ge = 1.5*KB*T/MH` one-liner.

Verified: for the exact composition that read 1220K for a requested
1000K before this fix (`primordial`'s default IC), initial T across
10-50,000K now matches the requested value to ~1e-7% (bisection
tolerance, effectively exact) -- down from the prior 22% miss. Real
Emscripten build of all three fiducial networks, headless-browser pass
(light + dark) across cool/free-fall modes, an IC preset, and a
parameter sweep: zero console errors, full `pytest` suite (86/86)
passes. Downstream "final T" values across the widget have shifted from
every prior NOTES.md entry's reference numbers (e.g. primordial's
default cool-mode run now ends at 273.4K, not the old 250.3K) --
expected and correct, not a regression: the actual initial conditions
were wrong before, so anything evolved from them legitimately changes
once they're fixed. Any future entry citing a specific "final T"-style
number should be re-verified against the current build, not assumed to
still match an older entry in this file.

**2026-09-09, new branch `wasm-rate-viewer`: added a reaction-rate
viewer/editor page, browser-only scope by explicit decision.** This
returns to the "reaction rate viewer" idea flagged earlier in the
session, now scoped down per an explicit instruction: "let's see what
we can do just for the web interface for now and worry about the rest
later." So this entry is the whole feature -- there is deliberately no
attempt here to feed edits back into the real Python/dengo codegen;
that's a separate, later phase.

Architecture decision, made after evaluating it directly rather than
guessing: use Vega-Lite's own `data: {sequence: {...}}` generator plus
a `transform: [{calculate: ...}]` chain as the entire formula-evaluation
mechanism, instead of writing a custom JS expression parser. Vega's
`calculate` transform already is a small sandboxed expression language
(supports `pow`/`exp`/`log`/`sqrt`/`min`/`max`/ternary), so a rate
"formula" is just one more `calculate` step computing `T -> tev ->
logtev/logT -> rate` in a chain, using the same natural-log convention
`primordial_rates.py` itself uses (confirmed via k09's own explicit
`np.log10` recovery: `state.logT / np.log(10.0)`). This means editing a
formula string and re-embedding the same spec shape *is* the entire
live-update story -- no evaluator to maintain at all beyond the
formula-string transcription itself.

New `wasm/reaction_rates.py`: hand-transcribed, hand-verified formulas
for all 22 primordial reactions (k01-k19, k21-k23), including both
three-body reactions' (k13, k22) 6 alternate `threebody=0..5` literature
fits as named presets (matching dengo/grackle's own `state.threebody`
convention) rather than folding them into one T-formula -- they're not
actually T-branches, they're a selection among entirely different
fits for the same reaction. New `wasm/rates.js` (viewer/editor logic)
and a `rates.html` page per network (built by `generate_site.py`,
filtered to whatever reactions that network actually uses): toggle
which reactions are plotted, an editable formula textarea per card that
redraws live on input, a preset dropdown for k13/k22, revert-to-default,
and JSON export/import of the whole edit session (which reactions are
selected, current formula, current preset) so a browser session's
exploration isn't lost on reload.

**Two real bugs found during verification, both in the transcription,
not the architecture:**
1. Every formula string initially used bare `T`/`tev`/`logtev`/`logT`.
   Vega's `calculate` expression language interprets a bare identifier
   as a reactive *signal* reference, not a field lookup -- so the very
   first real-browser embed crashed with "Unrecognized signal name."
   Fixed by systematically prefixing every reference with `datum.`
   (word-boundary regex is safe here since these are pure-alphabetic,
   non-overlapping identifiers -- `\btev\b` never matches inside
   `logtev`). Re-verified numerically afterward, not just "no longer
   crashes."
2. A regeneration pass (needed to apply fix #1 file-wide) had a bug of
   its own: k13/k22 preset keys came out as bare `"0"`-`"5"` instead of
   `"threebody=0"`-`"threebody=5"`, silently breaking the preset-label
   convention the UI and JSON export depend on. Caught by reading the
   regenerated file's diff carefully rather than trusting the
   regeneration script; fixed in the generation script itself and
   rebuilt.

Verification methodology (two independent checks against real ground
truth, not just "looks right"): (a) evaluated every transcribed formula
directly (a small JS `Function` sandbox, same `datum.*` inputs) against
the real dengo `coeff_fn` output for the same T values, called from
actual Python (`ChemicalNetwork`'s real primordial network, T = 10 K to
1e8 K, all 22 reactions plus all 12 threebody presets) -- all matched to
better than 1e-6 relative error. (b) went one step further and verified
the values Vega itself computes at runtime, not just the formula string
in isolation: built real `rateChartSpec()` specs, embedded them via the
real `vegaEmbed()` used on the page, pulled the actual computed dataset
back out (`result.view.data(<compiled dataset name>)`), and compared
those against the same Python reference via log-linear interpolation --
matched to a few percent (attributable to the interpolation itself, not
transcription error, since check (a) already confirmed the underlying
formulas exactly). Both checks were re-run after fixes #1 and #2 landed,
against the corrected build, not just the first draft.

Full UI regression pass on the corrected build (real Emscripten-built
site, headless Chrome, light + dark, all three fiducial networks):
card counts correct (22/6/2), k13's preset dropdown lists all six
`threebody=N` options with `threebody=4` as the pre-selected default,
toggle-all/toggle-none/reset-edits all work, live-editing a formula
redraws only that card's chart and revert restores the original, 
switching k13's preset updates both the formula textarea and the chart
together, an intentionally-malformed formula degrades to a
`.chart-placeholder` error message instead of crashing the page (Vega
returns a deliberately generic "Disabled." message for some malformed
inputs and a specific one, e.g. "Unrecognized function: ...", for
others -- either way it's caught and shown per-card, the rest of the
page is unaffected), and a full export -> reload -> import round-trip
correctly restores which reactions were selected, edited formula text,
and the chosen preset. Zero console errors throughout. Full `pytest`
suite (86/86) still passes (unaffected, as expected -- this feature
touches no Python code path the tests exercise).

**Deliberately not done here, per the scoping decision above:**
- No feedback path from an edited/preset-selected formula back into the
  compiled solver on the main widget page -- this page is read/explore/
  export only.
- Cooling rates and the CIE cooling table are out of scope; only
  reaction rate coefficients are covered.
- No attempt at a general Python-formula -> Vega-expression translator.
  Investigated this directly (prompted by "how hard would a single
  source of truth be"): `primordial_rates.py`'s rate functions are
  short imperative numpy, not sympy expressions, so this isn't a
  transpile-an-existing-symbolic-form problem. A mechanical pass over
  all 22 reactions found the *branching* vocabulary is actually very
  narrow -- 8 are a single expression with no branching at all, 12 are
  exactly one two-way boolean-mask split (`cond ? A : B`), 1 (k09) is a
  clamp (`min`) feeding into a formula followed by one low-T override,
  and the only apparent "multi-way" case (k13/k22's threebody 0-5) isn't
  T-branching at all -- it's a switch on a separate discrete parameter,
  already correctly modeled here as presets rather than a formula
  branch. A small closed schema (`{op: "expr"|"piecewise"|"clamp_then",
  ...}` using `datum.field` leaf expressions throughout) could plausibly
  cover the whole file with two small renderers -- one producing a
  numpy `np.where(...)`-based Python `coeff_fn` for the real solver,
  one producing this page's Vega `calculate` string -- eliminating the
  hand-duplicated-formula-drift risk that produced bug #1 above
  entirely, since both would come from one source dict. Worth
  revisiting as a real follow-up, not attempted this pass; a
  regex-only (non-AST) version of the branch-restructuring step was
  considered and rejected as too fragile for that follow-up if it's
  ever built -- the risk is a plausible-looking wrong formula compiling
  silently, which is strictly worse than the loud crash bug #1 above
  actually was.

**2026-09-09, new branch `wasm-rate-formula-drift-check`: made the
transcription-drift check from the previous entry permanent instead of
a one-off scratch script.** Asked to go explore the single-source-of-
truth idea further (edge-case count, how invasive a real fix would be)
before deciding what to build; the answer landed on three options ranging
from "just detect drift" (cheap, no invasiveness) up to "rewrite
primordial_rates.py into a shared declarative schema" (a real
single source of truth, but touches the one file the actual solver
depends on). Decision: do the cheap one now, nothing else yet.

New `tests/test_wasm_rate_formulas.py`, wired into the normal `pytest`
run (no separate script to remember to invoke): for every reaction (and
every k13/k22 threebody preset) in `wasm/reaction_rates.py`, evaluates
its Vega formula string directly in Python -- a small evaluator handles
the one real syntax gap (Vega's `cond ? a : b` ternary has no Python
equivalent, so it's rewritten to `a if cond else b`; everything else in
these formulas, arithmetic/comparisons/`pow`/`exp`/`log`/`sqrt`/`min`/
`max`/dotted attribute access, is valid Python already) -- and compares
it against the real `dengo.reaction_classes.reaction_registry[name]
.coeff_fn(state)` output across T = 10 to 1e8 K. Fails loudly, by
design, on anything more than 1e-6 relative error (with a floor for
both-effectively-zero comparisons, matching the tolerance already used
in the one-off verification scripts from the previous entry). Confirmed
it actually catches drift, not just passes trivially: temporarily
mutated one formula's exponent by hand and reran -- correctly failed
with the mismatching values named in the assertion message; restored
and reran clean.

Building this surfaced one genuine (if practically harmless) latent bug
of its own: k13 and k22's top-level `"formula"` field -- meant as the
fallback for reactions with no presets -- had been transcribed as
whichever threebody branch happened to come first in the original
if/elif chain (`threebody=0`) rather than the network's actual default
(`threebody=4`, `default_preset` in the same dict). `rates.js` never
actually hits that fallback for these two reactions (it sets
`SELECTED_PRESET` to `default_preset` immediately on card build, before
ever reading `activeFormula()`), so this was dead-in-practice, not a
UI-visible bug -- but it's exactly the kind of quiet inconsistency this
whole check exists to prevent, so fixed both to literally equal
`presets["threebody=4"]`, removing the need for the test to special-
case them.

Verified: `uv run pytest` -- 120/120 (86 original + 34 new: 22 plain
formulas + 12 threebody presets). Real Emscripten rebuild of all three
fiducial networks + headless-browser pass (light/dark): zero console
errors, k13's preset dropdown/default unchanged, and confirmed directly
that the default-shown k13 formula in the browser is now actually the
threebody=4 expression (previously it would have shown threebody=0's
text on first load before any preset interaction, momentarily
inconsistent with the dropdown already reading "threebody=4").

Deliberately not done: this is option "C" from the exploration -- two
copies of each formula still exist (real `coeff_fn` and
`reaction_rates.py`), just with drift now impossible to miss. The AST-
transpiler idea (mechanically deriving the Vega formula from the real
Python source, the actual single-source-of-truth option, with its edge
cases already catalogued: non-`vals`-named accumulators, sequential
reassignment/dead-code overwrites, local-variable clamps, mask-variable
name reuse, the threebody-switch-vs-T-branch classification) and the
declarative-schema rewrite option remain unbuilt, per explicit
instruction to stop here for now.

**2026-09-09, new branch `wasm-temp-band-note`: labeled the shaded band
on the Temperature chart.** Flagged as "the yellow bar" -- it had no
on-page explanation at all; it's the `band` rect in `redraw()`
(`app.js`, y0=1500/y1=2500, orange at low opacity, which is exactly why
it reads as yellow rather than orange, especially in dark mode where it
looks distinctly gold), only ever shown in temperature view (`tempField
=== "T"`; it disappears in thermal-energy view since 1500-2500 is a
temperature range specifically). It marks this project's target regime
(T ~ 1500-2500 K, from the H2-formation-heating-driven fragmentation
physics this whole widget exists to explore) -- context, not a solver
threshold or a claim that any given run passes through it.

Added a `.preset-note` caption directly under the Temperature chart in
`generate_site.py`'s page template saying exactly that, matching the
existing note style used elsewhere on the page (e.g. the shock-event
note). No app.js/rates.js logic changed -- purely a label for something
that already existed silently.

Verified: real Emscripten rebuild of all three fiducial networks,
headless-browser check (light + dark) confirming the note renders under
the chart on every network, reads correctly against both themes, and
zero console errors. Full `pytest` unaffected (120/120) -- this only
touches the HTML template, no path any Python test exercises.

**2026-09-09, new branch `wasm-shock-refinement`: free-fall's shock
event now resolves and displays the actual post-shock relaxation,
instead of hiding it inside one step.** A strong accretion shock (e.g.
Mach 5 fired from a mostly-atomic, H2-poor pre-shock state -- the
"virial-shock" preset with the shock density lowered to 1e11 was the
case that surfaced this) can heat the gas by 3-4 orders of magnitude
instantly, into a regime where radiative cooling and (density-cubed)
three-body H2 formation are many orders of magnitude faster than the
free-fall timescale that sizes an ordinary step. `BE_chem_solve`
already integrates all of that correctly in a single big step (that's
the whole point of an implicit, stiffness-tolerant solver) -- but only
the step's two endpoints were ever recorded, so a real, fast spike-then-
crash-then-partial-H2-reformation transient was completely invisible on
the chart, reading instead as "the shock causes an inexplicable drop in
T and rise in H2 fraction," backwards from what the Rankine-Hugoniot
jump itself actually does (always heats, for any Mach>1).

Confirmed the mechanism directly before changing anything: manually
sub-stepping through the same total elapsed time the single big step
would have covered showed the real trajectory -- instant post-jump
T≈56,300 K (matching the RH jump prediction exactly for that state),
crashing to ≈1,400 K within a small fraction of the step, then a slow
climb back to ≈2,040 K (the well-known H2-cooling-floor regime) as
three-body H2 formation proceeds, finally continuing the ordinary
free-fall track. This confirmed it's a real, correctly-integrated
physical result being hidden by output resolution, not a solver bug.

Implementation (`app.js`, `runFreefall`): the shock is still applied as
an instantaneous jump (a real discontinuity, not blended into one
compression step, as before), but the state right after the jump is now
recorded as its own point (capturing the true instant post-jump value,
which the old code never displayed), and the remainder of that step's
elapsed time is covered by `SHOCK_DISPLAY_POINTS` (40) checkpoints,
log-spaced from an estimated relaxation start time up to the full
step's ordinary dt. That starting estimate is `SHOCK_COOLING_SAFETY`
(0.05) times `coolingTime()` -- the same instantaneous ge/|dge/dt|
estimate `runConstantDensity`'s own adaptive stepping already uses --
evaluated once, right after the jump.

That "once" is deliberate, and cost a false start: the first attempt
re-derived a cooling-time-based step size *every* substep (the more
obviously "adaptive" design), and it doesn't work here -- this cooling
curve is so steep near the post-shock temperature that the derivative
itself changes by orders of magnitude within a single estimated
"cooling time," so each re-estimate immediately invalidated the last,
chasing a moving target instead of converging (observed directly:
substep sizes collapsing from ~1s to ~3e-4s in a single subsequent
step, never completing even at 3000 allowed substeps). Log-spacing from
one initial estimate up to the known total sidesteps this entirely --
it doesn't need the local derivative to stay still, only asks
BE_chem_solve for values at more, arbitrary-sized times than before,
which it was already proven able to give correctly (that's the reason
the transient was hidden in the first place).

Verified: real Emscripten rebuild, headless-browser pass across all
three fiducial networks, both themes, both display modes (T/thermal
energy), the parameter sweep (which also calls `runFreefall`), and
several IC presets -- zero console errors throughout. Confirmed the
extreme end of the Mach slider (100) still fails to converge on this
same preset+density, exactly as the pre-existing single-big-step code
already did on the identical state (checked directly, side by side) --
not a regression, a pre-existing, already-documented limit of how
strong a shock the solver can resolve at all, from a state this
extreme. Full `pytest` suite unaffected (120/120) -- pure `app.js`
change, no Python path touched.

**2026-09-09, new branch `wasm-chart-zoom`: drag-to-zoom detail view for
the Temperature chart, opt-in.** Direct follow-up to the shock-
refinement entry above -- once the shock's actual spike-then-crash-
then-recovery became visible as real points on the chart, the next
thing wanted was a way to look closely at that (or any) region without
permanently altering the main chart's own axes. Explicitly not
scroll/drag pan-and-zoom on the one chart (continuously rescaling axes
the way that always risks losing your place or fat-fingering the wrong
zoom level) -- instead the standard Vega-Lite "overview + detail"
recipe: drag a rectangle on the (unchanged) chart to select a range,
and a second, smaller chart underneath shows that exact range enlarged,
with its own axes actually zoomed. Gated behind a new "zoom view"
checkbox next to the existing temperature/thermal-energy mode buttons
(off by default, always visible so the capability is discoverable) --
added at the user's own suggestion, once the two-chart layout's
appearance was in question, rather than forcing every user of the plain
chart into a taller page.

Implementation (`app.js`'s `chartSpec()`): builds a `vconcat` of two
views sharing the same underlying data and the existing shaded-band/
shock-line "extra" layers, the top one carrying an `interval` selection
param (`brush`) on its x-encoding, the bottom one's x-scale domain bound
to that selection (`scale: {domain: {param: "brush", field: "x"}}`) --
no custom pan/zoom event handling anywhere, this is Vega-Lite's own
declarative binding. The detail view always draws its point markers at
a fixed, comfortably visible size, regardless of how many points the
whole run has (unlike the overview, which still shrinks its points as
step count grows) -- since the entire point of zooming in is to see
individual steps clearly.

One real bug surfaced building this, caught immediately by the existing
zero-console-errors check rather than needing to be searched for: a
first attempt attached the `brush` selection `param` at the *view*
level of the (multi-layer) overview chart, which crashed on load with
"Duplicate signal name: brush_tuple" -- Vega-Lite tries to project a
view-level selection onto every layer in that view, including the
shaded-band layer that has no x-field at all, and its compiler doesn't
handle that cleanly (confirmed via a minimal, dependency-free repro
outside the actual app before touching real code again -- isolated it
to exactly one layer vs. more than one layer in the same trick, not
object-identity/data reuse, which was the first, wrong guess). Fixed by
attaching `params` to the one specific layer that needs it instead of
the outer view -- scoped selections don't have this problem.

Verified: real Emscripten rebuild, headless-browser pass across all
three networks, both themes, both temperature/thermal-energy modes,
zoom on and off (including toggling it back off after being on) -- zero
console errors throughout. Confirmed directly, not just assumed: the
checkbox itself is always visible and unchecked by default (single
plain chart, byte-for-byte the old layout); checking it swaps in the
overview+detail pair; a real mouse drag on the overview produces a
visible selection rectangle and a correctly zoomed, correctly-scoped
detail view underneath (confirmed via DOM inspection that both views
render actual point-mark elements, one per data point, not just a
line); clicking an empty area clears the selection and both views
revert to the same full range. Full `pytest` unaffected (120/120) --
pure `app.js`/HTML-template/CSS change.

**2026-09-09, new branch `wasm-csv-export`: "Download CSV" button for
the current run's full step-by-step history.** A "Download CSV" button
next to the status line, enabled once the first run completes (matches
the existing disabled-until-ready pattern already used for the IC
preset dropdown and the sweep button), exports every step of whichever
run (cool or free-fall) is currently on screen -- explicitly the single
run, not a sweep's several overlaid ones, which has a different shape
entirely and would need its own export design; the button's tooltip
says so directly.

One column per plotted species (whatever that network actually has --
`plotableSpecies()`, so hydrogen_minimal's 3-column file and
primordial's 9-column one both fall out of the same code with no
per-network special-casing), plus T, ge, gamma, ionized fraction, H2
fraction (blank, not a bogus zero, on networks without H2 chemistry --
same graceful-degradation convention as the status line and charts
already use), the step index, x (whatever's plotted -- density or
time), elapsed time in both seconds and human units, and the step size.
Values are plain `String()`, not any of the `.toFixed()`/
`.toExponential()` display formatters used elsewhere on this page --
this is a data export, so it keeps full double precision rather than
the handful of significant figures a tooltip needs.

Confirmed directly (not just assumed from reading the code) that this
covers what was actually asked: the full time series, not a final-state
snapshot -- row count matches step count exactly on every network/mode
tested (e.g. 1230 data rows for a 1230-step free-fall run) -- and that
row 0 is genuinely the initial condition (`dt_s=0`, T matching the
requested value exactly), not the first *solved* step, inheriting that
for free from the same "IC is history entry 0" convention `runFreefall`/
`runConstantDensity` already established earlier in this file.

Verified: real Emscripten rebuild, headless-browser pass across all
three networks, both themes, both modes, after running a sweep (button
still exports the single run underneath it, unaffected) -- zero console
errors. Downloaded and parsed the actual files with Python's `csv`
module (not just eyeballing them): correct, consistent column count on
every row, every numeric field parses as a float. Full `pytest`
unaffected (120/120) -- pure `app.js`/HTML-template/CSS change.

**2026-09-09, new branch `wasm-reusable-solver-shim`: `dengo-solver.js`,
a reusable wrapper for using a compiled dengo wasm module outside this
widget.** Asked to explore making the compiled wasm usable "in other
modules" -- two options on the table: Emscripten's own `--emit-tsd`
flag (auto-generates a `.d.ts` at compile time) plus switching the
build's output to `.mjs` for real ES-module semantics, or a small
hand-written reusable wrapper. Verified the first option directly
before ruling anything out (real finding, not a guess): the currently-
installed Emscripten (6.0.9) genuinely supports `--emit-tsd` for this
project's plain `EXPORTED_FUNCTIONS` style (no embind needed), and
`.mjs` output does produce a real `export default` module -- both
confirmed by actually compiling `hydrogen_minimal` with each flag and
inspecting the result. But it only types the raw `_dengo_wasm_step(...)`
calling convention, not whatever `cwrap("dengo_wasm_step", ...)`
produces (permanently `any`, since it's built from a runtime string),
and needs `tsc` on the build machine -- a genuinely new toolchain
dependency, low-cost (GH-hosted runners already have Node; `npx
typescript` fetches `tsc` with no explicit setup step, verified
locally) but new all the same. Asked which piece(s) to actually build;
answer was the reusable shim only, explicitly not the compile-step
changes -- this entry is just that.

New `wasm/dengo-solver.js` (a real ES module, `export class
DengoSolver`) + hand-written `wasm/dengo-solver.d.ts` (this project
isn't running a TypeScript build step for `wasm/`, so there's no
compiler to generate one, and the class is small enough that hand-
maintaining the two in sync is the cheaper trade). One class covers
every network dengo can generate a wasm solver for, not just this
project's three fiducial ones -- the C API in `dengo_wasm.cpp.template`
is identical across all of them (same 7 functions, same names always);
only the species *names* differ, and those are already read at runtime
via `dengo_wasm_species_names()`, not baked into any particular build.
Wraps state access by species name (`readState()`/`writeState()`/`get()`/
`set()`) plus a `bulkUpdate()` escape hatch for vectorized edits (e.g.
scaling every species by a density ratio in one pass, the way
`app.js`'s free-fall stepping already does) without forcing that
through slower per-call object allocation. Two real gotchas in the
underlying C API are preserved exactly as they behave, not silently
smoothed over: `temperature()` still reads a cache only refreshed by
`rhs()`/`step()` (documented prominently, the same gotcha this project
hit and fixed earlier in `app.js` itself -- see the mean-molecular-
weight entry above), and every method re-reads `mod.HEAPF64` fresh on
each call rather than caching a reference, since Emscripten's typed-
array view can be invalidated by WASM memory growth (`ALLOW_MEMORY_
GROWTH=1`, needed for the larger networks) -- a real footgun a new
consumer unfamiliar with Emscripten specifically wouldn't know to avoid.

Copied once at the top level in `generate_site.py` (alongside `app.js`/
`rates.js`/`style.css`), not duplicated per network, since nothing about
it is network-specific.

Deliberately not done, per the explicit answer to "which piece(s)":
`app.js` itself was NOT migrated onto this class -- it keeps its own
from-scratch `mod`/`idx`/`speciesNames` plumbing for now, so this file
is currently unused by the widget itself. That does mean the class
isn't exercised by this project's own production code path, which is
exactly why verification here leaned unusually hard on independent
cross-checks rather than "the existing app still works":

Verified: (1) in Node (no browser at all -- a direct, real test of
"usable in other modules"), built via the shim and, side by side, via
hand-written raw `cwrap` calls mirroring `app.js`'s own approach exactly
-- state after an identical sequence of set/rhs/step/bulkUpdate calls
matched *bit-for-bit* between the two paths, including the `ge`/
temperature cache-staleness behavior. (2) In a real headless-browser
pass against the actual generated `primordial`/`primordial_atomic`/
`hydrogen_minimal` pages, dynamically `import()`-ing `dengo-solver.js`
alongside the page's already-loaded classic-script `dengo_wasm.js` and
driving a full set/rhs/step/bulkUpdate/read sequence against each
network's real compiled module -- zero console errors, correct
species lists and counts for each. (3) Confirmed the existing widget
pages themselves are completely unaffected (all three networks, both
themes, zero console errors) -- adding two new unreferenced files
changes nothing about what `app.js` does. Full `pytest` suite
unaffected (120/120) -- no Python path touched.

**2026-09-10, new branch `wasm-tn-chart-crosshair`: an opt-in density-
vs-time chart, a crosshair synced between it and the Temperature chart,
and a time-span readout for the zoom feature's brush.** Free-fall's own
charts all plot density on x, leaving elapsed time visible only in
tooltips -- asked whether putting time on a secondary top axis was
feasible; it wasn't a clean fit (density and time are related by
free-fall's own inverse-square-root law, not a simple linear/log
transform, so a naive secondary axis's gridlines wouldn't align with
the data at all). Asked instead for a genuinely new chart (time vs.
density, gated behind a checkbox like the zoom feature) with a hover
crosshair mirrored onto the Temperature chart, plus (mid-build) two
smaller asks: points only on the zoomed chart, not the overview, and a
time-span readout when brushing to zoom.

**Points-on-overview removed** (`mainLayer()`'s new `showPoints` param,
default `true`, `false` only for the overview layer when zoom is on) --
now only the detail/zoomed chart shows dots, the overview stays a plain
line. Small, mechanical, verified via DOM inspection (one symbol group
instead of two) and a screenshot.

**New `tnChartSpec()`** (`app.js`): a second, independently-embedded
chart (`#chart-tn`), x=time/y=density, built from the exact same `rows`
array `chart-T` already has -- no new computation, just a different
pair of encodings. Gated behind a new "show" checkbox next to it (off
by default); shows a placeholder in cool mode (density is held fixed
there, so there's nothing to plot) and before being toggled on, same
graceful-degradation convention as the species chart's "toggle one to
plot it."

**The hover crosshair was the harder piece, and worth explaining why:**
Vega-Lite has no built-in mechanism for linking a selection across two
*independently embedded* views (its own linked-selection recipes are
all within one composed spec, like the zoom feature's brush). Given the
brush-selection bug earlier in this project (relying on an assumption
about Vega-Lite's internal compiled signal names, which turned out
wrong), this was deliberately built and verified as an isolated,
minimal standalone repro *before* touching the real page -- same
methodology, since this had even more surface for exactly that kind of
surprise. Landed on splitting it into two independent, well-documented
touchpoints rather than one clever mechanism: (1) a `point` selection
with `nearest: true` on an *invisible* point layer (`nearest` isn't
supported on a `line` mark directly -- confirmed directly, it warns and
does nothing without this separate capture layer), read back via the
one stable, documented API this needs, `view.addSignalListener(name,
...)`; (2) the actual crosshair is a `rule` mark bound to a small,
explicitly-managed named dataset (`cursor`), pushed to via
`view.data("cursor", [...]).runAsync()` -- ordinary first-class Vega
API, not a selection-driven conditional encoding. Both this chart's own
crosshair and the *other* chart's mirrored one are driven the same way
from one JS callback (`wireHoverLink()`), keyed off the row's own
step index so no cross-field translation/interpolation is needed --
both charts already have every field for the same row.

**Brush time-span readout**: reads the zoom feature's own existing
`"brush"` param (the same one that already drives the detail view's
domain) via `addSignalListener`, linearly interpolates the selected
density range's two endpoints against the run's own `(x, t)` pairs
(`interpolateT()`), and writes "Selected range: 513 kyr to 556 kyr
(Δt ≈ 42.7 kyr)" into a note under the chart -- clears when the
selection does. The brush only ever operates in density; without this
there'd be no way to tell from the zoomed-in view alone how much real
time a given zoomed stretch actually covers.

Verified: full regression across all three networks, both themes, and
every combination of the zoom/tn-chart toggles together (each alone,
both together, plus a sweep run and a CSV download on top) -- zero
console errors throughout. Confirmed directly, not assumed: the
crosshair's underlying `cursor` datasets update correctly and
bit-exactly on real mouse movement in both directions (hovering either
chart moves both), confirmed visually via screenshots showing the
marker on both charts at once; the brush text appears/updates/clears
correctly using the zoom feature's own already-established clear
gesture (a drag, not a plain click -- confirmed by replaying the
original zoom-clear test unchanged against this build, ruling out a
regression before concluding the behavior was already like this).
Full `pytest` suite unaffected (120/120) -- pure `app.js`/HTML-template
change.

**2026-09-10: unify the free-fall/cool-mode charts, ionization/H2 chart,
and species chart into one Vega-Lite spec (sweep charts untouched).**

Previously four separate charts (temperature, optionally a density-vs-time
companion behind a checkbox, ionization/H2, species with a checkbox
subset and a density/mass-fraction toggle) are now one composed Vega-Lite
spec per run, `runViewSpec()`, driving a single `#chart-run` div. Three
requirement changes came with it, all from the user directly:
- Species: always all of them, always mass fraction (removes both the
  species checkbox row and the density/mass-fraction toggle -- one fewer
  decision for the user, and mass fraction is the physically comparable
  quantity across species of very different absolute abundance anyway).
- Species/ionization selection no longer checkboxes -- both now use
  Vega-Lite's own "Legends as Interactive Filters" recipe (a `point`
  selection with `bind: "legend"`, `opacity` conditioned on it): click a
  legend entry to isolate it, click again to show all. First-class
  Vega-Lite, not a bespoke mechanism.
- Every panel (temperature, density, zoomed views, ionization, species)
  shares one crosshair via Vega-Lite's own `resolve: {selection: {hover:
  "global"}}`, replacing last week's entirely hand-rolled
  `wireHoverLink()`/`view.data("cursor",...)` cross-chart JS bridge for
  this same feature -- confirmed directly in an isolated repro before
  relying on it (see below), the declarative version is strictly less
  code and no longer needs a manually-managed shared dataset at all.
  Each overview panel's crosshair line is now just a `rule` mark with
  `transform: [{filter: {param: "hover", empty: false}}]`, reading the
  resolved selection directly -- no signal listener needed for
  rendering, only for the (still JS-side) brush timespan readout.

Sweep charts explicitly excluded from this ("skip the sweep charts for
this") -- they still use the old per-chart `chartSpec()`-style path,
untouched, and still show the old checkbox-driven species subset.

**A real, reproduced-in-isolation Vega bug shaped the final
composition.** The natural structure -- `vconcat` of `hconcat` rows
(overview row, zoomed row) plus ion/species panels as further `vconcat`
siblings -- throws real Vega runtime `TypeError`s on hover
("Cannot read properties of undefined (reading '0')" / "...('datum')")
as soon as *any* panel with an `interval` (brush) selection and *any*
panel with a globally-resolved `point` (hover) selection coexist
anywhere beneath two or more levels of concat nesting (`vconcat: [{hconcat:
[...]}]` alone is enough, even with only one row and only one panel
declaring the brush). It does not reproduce with only one level of
nesting (a flat top-level `hconcat` or `vconcat`, brush and hover
included), confirmed with a battery of minimal repros using placeholder
data completely unrelated to this app (see below) before touching
`app.js` at all, and confirmed again against a real build under a
stress test that swept the mouse across empty regions, multiple panels,
and both directions repeatedly -- the flat version threw nothing; every
nested variant, several restructurings of it (`hconcat`-of-`vconcat`
columns instead of `vconcat`-of-`hconcat` rows, explicit
`resolve: {selection: {brushX: "independent"}}`, per-selection `resolve:
"global"` on the select objects themselves, reordering layers, dropping
`nearest`/`fields` from the hover selection), threw the same pair of
errors. The fix: the whole composition is one single flat top-level
`concat` (with `columns: 2` doing the row-wrapping) with every panel --
overview, zoomed, ion, species alike -- as a direct sibling in that one
array, never wrapped in an intermediate row/column spec. The one real
cost: ion/species panels have no natural row partner, so they end up
paired with each other at `PANEL_WIDTH` (620px) instead of each
spanning the full combined width the way they used to when built as
`vconcat` siblings -- a genuinely fine trade in practice, confirmed by
screenshot (two legend-driven charts side by side, each still plenty
readable with its horizontal wrapped legend underneath).

**Legend position**: also fixed along the way, found via a screenshot
(the user caught it too, independently: "Legend seems to bump into the
species. Shouldn't the legend be down at the bottom, too?"). Both the
ionization and species legends now use `orient: "bottom", direction:
"horizontal"` (species additionally `columns: 0` to auto-wrap many
entries) instead of the earlier default/`top-right` placement, which
had been overlapping the first panel of the (then still nested)
composition rather than sitting near its own panel.

**Captions**: kept the previous explanatory prose (one consolidated
`.preset-note` paragraph below the whole combined chart, in the HTML
template, not the spec -- Vega-Lite has no rich-text/paragraph mark, so
this is the practical way to "include that as captions inside the
spec"), plus a short bold `title` on every individual panel inside the
spec itself (e.g. "density vs. time", "Species mass fraction (click a
legend entry to isolate it)") as the more literal in-spec captioning
the user also asked for.

**Axis titles**: plain Vega-Lite `axis: {title: "...", titleFontSize:
10}` strings now (e.g. "T (K)", "n (cm⁻³)", "X_i (mass frac.)") instead
of the external-KaTeX-div-next-to-chart trick (`renderLatex`/
`AXIS_LATEX`) the old one-chart-per-box layout used -- a deliberate
simplification, not just a workaround: one combined spec with several
differently-labeled panels can't cleanly share one external label div
per axis the way separate single-chart boxes could. `renderLatex`/
`AXIS_LATEX` themselves are unchanged and still used by the untouched
sweep charts.

**Page width**: `body` max-width 1100px -> 1700px, `.layout` sidebar
column 320px -> 340px ("there's a lot of remaining real estate on my
desktop monitor... If we need to make the charts wider we should do
that, too").

Removed entirely: `hoverLinkLayers()`, `wireHoverLink()`, the old
`chartSpec()`/`tnChartSpec()`/`ionizationChartSpec()`/
`speciesChartSpec()`/`updateTnChart()`/`selectedSpeciesNames()`/
`setSpeciesDisplayMode()`/`buildSpeciesToggle()`, and the
`temperatureZoomEnabled`/`tnChartEnabled`/`tChartView`/`tnChartView`/
`speciesDisplayMode` state vars, plus their HTML (species checkboxes,
species mode buttons, T-zoom-toggle/tn-chart-toggle checkboxes) and CSS
(`.zoom-toggle`, `.species-toggle-controls`, `.species-toggle .swatch`).
`.species-toggle`'s base rules are kept -- still used by `rates.js`'s
own `#rate-toggle`.

Verified: full rebuild (real Emscripten build, all three fiducial
networks) + headless-Chrome/Playwright against the real generated site,
not just the isolated repros -- zero console errors across cool mode,
freefall mode, both themes, a network with H2 species and one without
(`hydrogen_minimal`), a stress-tested mouse sweep across every panel and
both overview columns (including deliberately crossing empty regions to
exercise the selection-clear path), a brush-zoom drag with correct
timespan text ("Selected range: 0 s to 553 kyr (Δt ≈ 553 kyr)."), and a
legend click. Full `pytest` suite unaffected (120/120) -- pure JS/HTML/
CSS change, confirmed by rerunning it after this work, not assumed.

Not yet shipped -- staying on local `main`, no branch/commit/push yet,
per the same "don't commit until I say so" instruction this feature
started under; a fresh go-ahead is needed for this specific piece of
work before it ships (the last one covered the now-superseded
density-vs-time-chart-with-checkbox feature, already shipped separately
as PR #19).

**2026-09-10, continued: page width -- responsive instead of a fixed
1700px, plus a real horizontal-scroll fix underneath it.**

The fixed `max-width: 1700px` from the same day's earlier change could
exceed an actual (unmaximized, or genuinely "HD"-class) browser window,
forcing the whole page to scroll horizontally just to reach the right
edge -- reported directly ("the left margin pushes it to a horizontal
scroll"). Changed to `max-width: min(1700px, 95vw)`: a fraction of the
viewport, capped so it doesn't keep growing past a comfortable reading
width on a 4K display.

That alone wasn't the whole fix, and testing (headless Chrome at five
viewport widths, 1366px up through 3840px) caught why: `#charts` is a
direct `.layout` grid item, and a grid item's default `min-width: auto`
means it won't shrink below its content's own intrinsic width no matter
what track size the grid assigns it -- so the fixed-pixel-width Vega
charts inside were forcing the whole page wider regardless of the vw
change. Confirmed directly: at 1366/1600/1920px-wide viewports the page
still had `scrollWidth > clientWidth` after the vw change alone. Fix:
`#charts { min-width: 0; }`, plus `overflow-x: auto` on `.chart-box`
(shared with the sweep charts, so both benefit) as the actual container
that scrolls locally once the grid item is allowed to shrink past its
content. Re-tested the same five widths after this second change:
zero page-level horizontal scroll at every one, confirmed the local
chart-box scrollbar genuinely reaches hidden content (scrolled it
programmatically and screenshotted the result), and 120/120 pytest
still passing (a pure CSS change, but re-run anyway rather than
assumed).

**2026-09-10, continued: redesign the run chart -- vertical single-chart
stack, and a real fix for a genuinely-broken zoom.**

The multi-column grid from earlier today ("clunky", per direct
feedback) is replaced with a single vertical stack, one panel per
row, reordered and reworked per explicit direction:

1. **density vs. time** (free-fall mode only -- density is held
   constant in cool mode, so this would just be a flat line there).
   Deliberately **x = density, y = time** -- the reverse of how this
   looked when first added -- specifically so its x-axis is the same
   field as panel 2's, letting one brush there zoom both. Density is
   log (never legitimately zero); time is symlog on the *y* axis now
   (still can legitimately be exactly 0 -- same reasoning as always,
   just on the other axis this time).
2. **temperature/thermal-energy vs. x** (x = density in free-fall
   mode, time in cool mode). The one panel that actually holds the
   zoom brush.
3. **ionization/H2 fraction**.
4. **species mass fraction**.

**Zoom model changed entirely.** The previous "overview never
rescales, a separate zoomed *copy* of the same metric does" pattern
(the multi-column grid's `tempDetail`/`tnDetail`) is gone. Instead,
exactly one panel (temperature-vs-x) declares the brush; every *other*
panel in the stack binds its own x-domain to that one selection by
name, so dragging a range there rescales every other panel sharing
that axis *in place* -- no separate zoomed duplicates left to draw.
The brush-holding panel itself is excluded on purpose (stays at the
full range, showing the selection box) -- explicitly asked for
("zoom on all *other* charts"), and it's also the only technically
sound option: a single view can't simultaneously show the full-range
selection box *and* be rescaled by its own selection.

**A real "zooming doesn't work" bug got fixed along the way -- and it
turns out to predate today's redesign entirely.** Confirmed directly
in an isolated repro before touching `app.js`: a panel built as a
*layered* spec (main line + an invisible hover-capture layer + a
crosshair rule layer, `layer: [...]`, i.e. every panel in this file)
only rescales correctly if **every** layer's x-scale gets the *same*
explicit `domain: {param: "brush", field: "x"}`, not just the visible
line's. Vega-Lite's default is to *share* one merged scale across
sibling layers within one view, and the merge is a **union** -- if
only one layer has an explicit narrow domain and the others don't,
the union with their own full-auto-range domain is just the full
range again, so the panel visibly never rescales at all even though
the brush param itself fires correctly (confirmed the param value was
right the whole time; only the rendered domain was wrong). This
exact shape (`xDomainFromBrush` applied only to the main line, not to
that panel's own hover-capture/crosshair layers) was already present
in the *previous* design's zoomed detail panels, meaning that "zoomed"
row was probably already silently broken before today, not something
this redesign introduced. Fixed by threading `xDomainFromBrush`
through `crosshairLayer()`/`hoverCaptureLayer()` too, applying the
identical domain object everywhere a panel needs it -- confirmed
directly: before the fix, an isolated repro with a layered follower
panel reproduced the exact "stays at full range" bug; after applying
the domain to every layer, the same repro rescaled correctly, and the
real app followed the same pattern.

**Species colors**: switched from a hand-picked 10-color array (d3's
older "category10") to Vega-Lite's own `scheme: "tableau10"` --
flagged directly ("the species all seem to have only orange and blue
colors... we should use a standard set of colors that are all
different"). `category10` includes several low-saturation
grays/browns that read as much less distinct than tableau10's set in
practice against this chart's many overlapping log-scale lines, even
though both are nominally "10 different colors." `SPECIES_COLORS`/
`speciesColor()` are gone entirely -- nothing else referenced them.

**Legend independence**: explicit `resolve: {legend: {color:
"independent"}, scale: {color: "independent"}}` at the composition's
top level, so the ionization and species charts' legends can never
merge -- belt-and-suspenders (they already use different color fields,
"quantity" vs. "species", so Vega-Lite wouldn't have merged them by
default regardless), made explicit because it was asked for directly.

**Zoom-timespan text**: moved from below the whole combined chart to
just above it (directly under the mode buttons), per "put it up...
closer to where the zooming occurs" -- as close as this architecture
allows without reintroducing multiple `vegaEmbed()` calls (one
combined spec renders as one continuous SVG; HTML can't be interleaved
mid-panel), landing one panel away from the brush-holding chart
instead of three. Also corrected the explanatory text's claim about
how to clear a zoom: confirmed directly it's a plain click (no drag),
not "drag an empty area" as worded before.

Verified: real Emscripten rebuild, headless-Chrome regression --
zero console errors across cool mode, free-fall mode, both themes, a
network with H2 species and one without. Specifically confirmed (not
assumed): dragging a narrow range on the temperature-vs-density panel
visibly rescales the density-vs-time, ionization, and species panels
to the same range while the temperature-vs-density panel itself stays
at the full range showing the selection box; dragging on the
density-vs-time panel does nothing (no brush there, by design);
clicking (no drag) on the temperature-vs-density panel clears the
zoom and restores every panel to the full range; the brush-timespan
text tracks correctly in both modes; species now render in visibly
distinct colors (confirmed by screenshot, not just by scheme name);
isolating one species via its legend dims every other species'
*own* color at reduced opacity (not a flat gray) and leaves the
ionization chart's own legend/lines completely unaffected. Full
`pytest` suite unaffected (120/120), rerun after this change.

**2026-09-10, continued: density-vs-time data fix, a zoomed temperature
panel back, real tooltip/crosshair fixes, and a lag investigation.**

Several follow-up requests on the vertical-stack redesign, all from
direct feedback after using it:

**Density-vs-time chart was hard to read.** Free-fall time is heavily
front-loaded (almost all elapsed time passes while density is still low
and barely changing; the actual collapse through the remaining many
decades of density is comparatively instantaneous), so a y-axis (time)
that has to include an exact t=0 point wastes most of its vertical
range on a near-flat early rise and squeezes everything else into a
sliver. Fixed exactly as suggested: this one panel's data now starts
from the first real *result* row, not the initial condition (`rows.
filter((r) => r.t > 0)`) -- once t=0 is gone, y never needs to include
zero either, so it's plain `log` now instead of `symlog`. Confirmed by
screenshot: the y-axis now spans "3.17 kyr" to "2.22 Myr" using the
full log range, instead of collapsing to a flat line near the top.

**A zoomed temperature-vs-density panel is back, directly below the
overview** (third panel now: density-vs-time, temperature-vs-density,
*zoomed* temperature-vs-density, ionization/H2, species). The brush-
holding panel can't also show its own rescaled view (it has to stay at
the full range to show the selection box), so temperature is the one
metric that still needs a real second, zoomed instance -- unlike
density-vs-time/ionization/species, which just rescale their one
existing panel in place. Circles/points only on this one panel, per a
standing preference from earlier in this project.

**Real bug hit and fixed while wiring "hover" onto more panels.** Tried
attaching `hover` directly to ionPanel's/speciesPanel's/the new zoomed
panel's already-visible `line`-with-`point` mark, instead of a separate
invisible capture layer, on the theory that a mark with real point
geometry wouldn't need one. Confirmed directly this doesn't work:
Vega-Lite warns "nearest transform is not supported for line marks" for
that composite mark exactly like it does for a bare line, silently
breaking the selection there -- and because `hover` is globally
resolved, it broke the crosshair on *every* panel, not just the ones
with the bad wiring (confirmed: caught by inspecting the DOM directly
for the crosshair's own dashed rule elements after hovering, since nothing
appeared anywhere, then bisected which panels' wiring was actually at
fault by checking console warnings). Reverted to the established
pattern -- `nearest` always gets its own dedicated invisible `point`
capture layer, even on panels that also show real points.

**Tooltips had stopped working, including on temperature-vs-density.**
Root cause: the invisible hover-capture layer, being the topmost mark
under the cursor, is what actually receives every pointer event --
being invisible doesn't stop it from doing that -- so it was silently
starving the visible line's own `tooltip` encoding of ever seeing one.
Fixed by giving `hoverCaptureLayer()` its own matching `tooltip`
encoding, so *it* serves the tooltip instead. Confirmed directly:
hovering a real data point now shows the expected tooltip content on
every panel, including temperature-vs-density.

**Crosshair now genuinely active and synchronized on every panel**
(previously only two of them could initiate it). All five panels
declare their own `hover` capture layer; `resolve.selection.hover:
"global"` still merges them into one shared value, same mechanism as
before, just applied more broadly. Confirmed directly via a DOM check
for the crosshair's own dashed line elements (not just screenshots,
which are too subtle to fully trust for a thin dashed line) --
hovering any one panel produces a matching dashed line at the same x
position in all five. (One test-methodology trap along the way, worth
recording: Playwright's multi-step interpolated `mouse.move()` can
pass through the narrow gaps *between* panels and end up dispatching a
final position that reads as "pointer left the capture layer," so a
naive before/after screenshot comparison read as "nothing happens" when
the real, single-jump mouse position worked correctly the whole time --
switching the test to a plain single-position move resolved the false
alarm.)

**Crosshair color changed to light gray** (was yellow/orange before;
flagged as reading too similarly to the shock event's own red dashed
line in practice) -- the shock line's own color is untouched.

**Lag investigation -- a real, measured regression, only partially
fixed.** Confirmed directly (not just "it feels slow"): a single
`mouse.move()` on this chart took ~1.6-1.7 seconds end-to-end, versus
~20ms on a blank page -- a real, large regression, not a subjective
impression. Traced (via a series of isolated repros matching this
app's actual panel/data-size shape) to the combination of `hover` being
globally resolved across many views *and* several of those views having
an x-scale `domain` bound to the *other* ("brush") selection: merely
having such a domain-from-param binding present anywhere in the
composition cost roughly 700-900ms extra per hover tick in the repro,
regardless of how many layers repeated it (removing it from 2 of 3
layers per panel, or all but one, made no measurable difference) --
this points to Vega's own dataflow scheduler not cleanly isolating
"hover changed" pulses from nodes that depend on the unrelated "brush"
selection, rather than anything fixable by restructuring the JSON spec
differently. Switching the renderer from `"svg"` to `"canvas"` was
tested and made no difference either (ruling out DOM/paint cost as the
bottleneck). One real, verified win was found -- feeding the
ionization/species hover-capture layers the smaller per-step dataset
instead of their own long-format one cut the isolated repro's lag by
about 20% -- but implementing it would mean that layer's own tooltip
encoding could no longer show the actual quantity/species value (that
data isn't in the smaller dataset), which is a worse regression than
the lag it would fix, so it was not applied. Documented here rather
than shipped silently: this is a real, only-partially-addressed
limitation, not something resolved -- consistent with the standing
"this may not be possible to address" allowance.

**Species legend "annoying to turn everything on/off"**: no code
change needed here -- confirmed directly that Vega-Lite's `bind:
"legend"` point selection already supports shift-click to select
several entries at once (a documented, built-in part of the mechanism,
not something added), which is exactly the "easy way" asked for.
Updated the explanatory caption to actually say so, since nothing in
the UI itself hinted at it before.

**Page width shrunk back down**: the vertical single-chart-per-row
layout doesn't need the two-column grid's ~1700px width any more, and
at that width the explanatory paragraph below the chart -- constrained
only by the page's own width, not the chart's -- wrapped at a line
length far wider than the now-single-column chart sitting above it, an
imbalance flagged directly ("the text is much larger than the
charts"). `max-width` dropped from `min(1700px, 95vw)` to `min(1100px,
95vw)` -- sized to the single ~620px-wide chart column plus the
sidebar, not the old two-column figure. Confirmed by screenshot: text
and chart now read as proportionate, and confirmed via the same
five-viewport-width sweep as the previous responsive-width fix (1366px
through 4K) that this still causes zero page-level horizontal scroll
at every one of them.

Verified: real Emscripten rebuild, headless-Chrome regression across
cool/free-fall modes, both themes, a network with H2 species and one
without, brush-zoom (confirmed the new zoomed panel and every other
follower panel rescale together, the anchor panel stays full-range),
tooltip content on every panel, crosshair sync via direct DOM
inspection, shift-click multi-select on the species legend, and zero
page-level horizontal scroll at five viewport widths. Full `pytest`
suite unaffected (120/120).

**2026-09-10, continued: run-metadata readout, a genuinely more useful
density-vs-time chart, and disabling the crosshair -- plus a second,
independent, real Vega bug found while trying to fix zoom speed.**

**Run metadata**: a new, prominent `#run-summary` block right below the
page's own subtitle -- steps, wall-clock solve time, elapsed simulated
time, final T, final ionized/H2 fraction, shock-crossing density, each
labeled -- replacing what used to be one small, easy-to-miss, unlabeled
line at the very bottom of the sidebar (below every slider, where a
species-heavy network could push it well out of view). `#status` in the
sidebar keeps its own narrower original job (module-loading state only:
"loading solver..." / "ready").

**Density-vs-time panel now plots *lookback* time (time remaining until
this run's last step), not elapsed time** -- flagged directly as "not
that useful" the way it was: free-fall time is heavily front-loaded
(almost all of it elapses while density is still low; the collapse
through the remaining many decades of density is comparatively
instantaneous), so elapsed-time-vs-density rises steeply for a couple of
density decades and goes flat for the rest of the run -- which, for this
project, means exactly the regime it cares most about (H2-formation-
heating/shock physics, all at high density) falls in the boring flat
part. Counting backward from the run's own last step instead flips
which part is flat, moving the detail to line up with the high-density
regime that's actually interesting -- exactly the "lookback time" idea
suggested directly. Confirmed by screenshot: what used to be a rapid
rise into a long flat plateau is now a single, smoothly decaying curve
using the *entire* vertical range, decade after decade, no flat part at
all.

**The shared crosshair is gone, for now** -- disabled outright, not
just tuned, per direct instruction once a real, measured performance
regression was confirmed (a single mouse move took ~1.6s on this chart
vs. ~20ms on a blank page). `crosshairColor()`/`crosshairLayer()`/
`hoverCaptureLayer()`/`withHoverParam` and the `hover`-selection
`resolve` entry are removed outright (not just unused) -- a future,
deliberately *simpler* mechanism (a bar that just tracks the pointer's
raw x pixel position, no per-panel nearest-point lookup or cross-view
selection resolution at all -- explicitly suggested as worth trying:
"we would not necessarily need it to do a 'nearest finder'") is different
enough in kind that nothing here was worth keeping around unused for it.
Removing it also surfaced a real side-effect regression of its own: the
invisible hover-capture layer had also been serving, incidentally, as a
much more forgiving tooltip target than a bare `line` mark's 1px stroke
-- confirmed directly (tooltips on the two line-only panels stopped
working reliably once that layer was gone). Fixed by giving even the
non-`showPoints` panels real (just invisible, `opacity: 0`) point
geometry instead of a bare line, so tooltips have *something* to
hit-test against -- less forgiving than the old full-panel "nearest"
search was, but confirmed directly to genuinely work again, just not as
buttery smooth.

**Zooming was independently, severely slow too -- confirmed directly
this was NOT just the crosshair's fault, and root-caused separately.**
After removing the crosshair, a single brush-drag gesture still took
~6.6-7 *seconds*. Traced to binding every follower panel's x-domain
*live* to the anchor panel's own brush `param` (`domain: {param:
"brush", field: ...}`, continuously rescaling while dragging) -- the
same ~700-900ms-per-signal-update Vega dataflow tax an earlier
investigation had already found for the crosshair, just triggered by
the brush's own many intermediate drag-position updates instead of
hover ticks. Fixed by switching zooming from *live* (rescale
continuously while dragging) to *debounced-commit* (read the brush's
final value once dragging actually pauses for ~250ms, then rebuild the
whole chart with that range baked in as a literal value, via a new
`embedChart()` that reuses the last solve's cached inputs -- no
re-running the solver). Confirmed directly: brush-drag time dropped
from ~6.6-7s to ~850ms-1.05s.

**That fix immediately hit a second real, independent Vega bug of its
own.** The natural way to "bake in" a committed zoom range is an
explicit `scale.domain: [lo, hi]` literal array -- and that turned out
to be badly broken: giving a `log`-typed x-scale an explicit literal
domain, with *nothing else different* about the spec, made Vega render
the chart at close to *4x* its declared width. Confirmed in an isolated
repro completely unrelated to this app's own composition (a single,
fresh-embedded, non-composed 300px-wide panel, one mark, no vconcat, no
re-embedding): adding `scale.domain` alone took it from 360px to 1357px;
`nice: true` alongside it helped some (1114px) but nowhere near enough;
an explicit `autosize: {type: "pad"}` made no difference at all. The
compiled Vega JSON's own declared `width` was unaffected in both cases
(confirmed by diffing `vegaLite.compile(...).spec` directly -- the
*only* structural difference was the domain itself), so whatever
inflates the rendered width happens at Vega's own runtime layout step,
not something visible in -- or fixable from -- the spec Vega-Lite hands
it. Fixed by not touching `scale.domain` for the zoom at all: instead,
each follower panel's own *data* is filtered to the committed `[lo,
hi]` range in plain JS before being embedded (`zoomFilteredData()`),
letting Vega-Lite compute the (now naturally narrower) domain from data
the normal way -- confirmed directly, back to the correct width, and
arguably simpler than the domain-override version besides. The one
real trade-off: an extremely tight zoom that happens to fall entirely
between two adjacent data points would show nothing (no interpolated
line, since there's no data left to draw) rather than a clipped
segment the old domain-override approach would still have shown -- a
rare edge case, accepted rather than adding padding logic for it.

Verified: real Emscripten rebuild, headless-Chrome regression across
cool/free-fall modes, both themes, a network with H2 species and one
without, run-summary content in both modes, the lookback-time chart's
shape (confirmed by screenshot, not just by the underlying math),
zoom/re-zoom/clear (including a stress sequence: zoom, zoom again to a
different range without clearing first, then clear), zero page-width
regression from the zoom fix (confirmed the SVG's own `width` attribute
stays correct through repeated re-embeds), and tooltip content restored
on every panel. Full `pytest` suite unaffected (120/120).

Not done in this round, deliberately: splitting the parameter sweep
out to its own page (a separate, larger piece of work, tackled next).

**2026-09-10, continued: zoomed panel's axis mismatch, and a real
chart-column scrollbar on an ordinary, plenty-wide screen.**

Two more real bugs, both caught directly by the user rather than found
proactively.

**"Why don't the zoomed and ionization panels have the same x-axis
right edge?"** -- the "zoomed" temperature panel's shock-event marker
rule has its own tiny, separate one-row dataset (just `{x: nShock}`),
which `zoomFilteredData()` never touched (it only filters the *main*
data). Vega-Lite's default shared-scale-across-layers behavior then
pulled that one raw, unfiltered point straight into the panel's
x-domain regardless of the actual zoom range, silently stretching its
right edge out to wherever the shock happened to be -- while the
ionization panel, with no such extra layer, correctly showed just the
zoomed range. New `extraForZoom()` filters the shock rule the same way
as everything else when building the zoomed panel specifically (the
overview panel keeps it unfiltered on purpose -- it always shows the
full range anyway). Confirmed by screenshot in both directions: axes
now match exactly when the shock falls outside the zoom, and the shock
line correctly reappears when zooming in on it.

**"I'm getting a horizontal scroll bar... it's the chart column."** --
reported on an ordinary 1920px-wide window, not a narrow one. Measured
directly: at this page's own capped body width (1100px, the same for
any window past ~1158px wide), `.chart-box` has ~686px to give the
chart, but the actual rendered chart (`PANEL_WIDTH` 620px plus
axis-label margins) came out to ~705px -- overflowing by a real margin,
not a rounding error, and not the "narrow window" case `.chart-box`'s
own `overflow-x: auto` fallback exists for. `PANEL_WIDTH` 620 -> 580
(confirmed by direct measurement to render at ~665px, comfortably under
the ~686px budget) fixes it with real margin to spare -- confirmed at
every width from 1158px up through 4K, zero local scrollbar anywhere in
that range.

Verified: real Emscripten rebuild, headless-Chrome regression across
cool/free-fall modes, both themes, a network with H2 species and one
without, zoom/re-zoom/clear, and the two specific bugs above (axis
match with the shock both in and out of the zoomed range; chart-box
scroll absence at 1158px through 3840px). Full `pytest` suite
unaffected (120/120).

**2026-09-10, continued: a free-fall collapse-rate multiplier, an
exposed solver tolerance, and a shock on/off checkbox.**

Three new free-fall controls, all implementable without touching the
compiled dengo-generated solver at all -- confirmed directly before
building anything: the solver's own C++ already treats convergence
tolerance as a genuine runtime argument (`dengo_wasm_step(dt, maxIter,
tolerance)`, already called with a literal `1e-5` at every call site),
and the collapse-rate question is purely about a rate constant in this
file's own free-fall math, nothing the solver ever sees.

**Collapse rate (× free-fall)** -- a new slider multiplying the
ordinary free-fall compression rate by a user-chosen factor (0.01x to
100x, log-scale), answering a direct thought-experiment question ("is
there value in... collapsing faster/slower than free-fall by some
factor"). Judged yes: real collapsing gas doesn't necessarily contract
at exactly the free-fall rate (rotation/magnetic fields/pressure can
slow it below free-fall; additional infall/turbulence can speed it up
past it), and how much *real time* the chemistry gets per decade of
density to react is exactly the physically meaningful thing this
project's whole 1500-2500K fragmentation question already cares about.
Implemented as a single new constant, `FF_RATE_CONST`, with a
`collapseFactor` multiplier applied consistently everywhere it's used
-- both the actual density-compression formula *and* the free-fall time
used to size the adaptive step -- so a step still represents the same
fractional density change regardless of the factor (the two scalings
cancel out of the step-sizing math), it just represents proportionally
more or less real elapsed time. Confirmed directly: at 10x, elapsed
simulated time for the same density range dropped to almost exactly
1/10th of the baseline; at 0.1x, it rose to almost exactly 10x --
matching the intended physics exactly, not just running without
erroring.

**Solver tolerance** -- a new slider (10^-8 to 10^-3, default 10^-5,
matching this page's previous hardcoded value) exposing the same
tolerance argument every `step()` call already accepted, threaded
through both `runFreefall()` and `runConstantDensity()`'s single-run
path (not the sweep path -- sweep is out of scope for now, deliberately
left on its own unexposed `1e-5`, matching how it already worked).
Confirmed directly: loosening it measurably speeds up the solve (206ms
-> 106ms in one test run) with only a tiny change in the converged
answer; tightening it measurably slows it down (206ms -> 505ms) with
an equally tiny change the other way -- exactly the accuracy/speed
trade a tolerance control should show, not a no-op.

**Shock on/off checkbox** -- explicit "Enable shock" checkbox next to
the existing shock density/Mach sliders, requested directly ("I know
setting to Mach 1 does it, but I want it manually specifiable"). When
unchecked, the *sliders themselves* also get visually disabled, and the
Mach number actually passed into `runFreefall()` is forced to 1 (the
same zero-strength/no-op limit `shockJumpFactors()` already treated as
"no shock") regardless of the slider's own value -- but the slider's
value itself is left untouched, so re-checking the box restores exactly
the Mach number it was left at, instead of the old "set Mach to 1"
approach silently discarding whatever it was set to. Confirmed
directly: unchecking removes the "shock crossed at" chip from the run
summary entirely (a real behavior change, not just a UI state); the
Mach slider's own value is unchanged the whole time; re-checking
reproduces the exact original with-shock result.

One real regression caught and fixed while wiring these through:
inserting the new parameters into `runFreefall()`'s and
`runConstantDensity()`'s signatures shifted the positional arguments at
their sweep call sites (`runSweep()`), which still call them with
explicit trailing `undefined`s for the older parameters -- confirmed
directly (by reading the call sites, not just assuming) and fixed by
adding one more `undefined` at the affected call site so `forSweep`
still lands in the right position; the other sweep call site needed no
change (it never explicitly overrode anything past the first new
parameter's position).

Verified: real Emscripten rebuild, headless-Chrome regression across
cool/free-fall modes, both themes, a network with H2 species and one
without, collapse-rate at 1x/10x/0.1x (confirmed the resulting elapsed
time scales correctly), tolerance at loose/tight/default (confirmed
solve time and converged answer both move the expected direction), the
shock checkbox's full on/off/on-again cycle, zoom/re-zoom/clear
unaffected, and the CSV download button still enabling correctly. Full
`pytest` suite unaffected (120/120).

**2026-09-10, continued: free-fall step size exposed too, plus a
step-cap safety net.**

Asked directly ("what about freefall increment -- is that worth
exposing?"): yes, judged genuinely worthwhile, and not just as a
performance knob. `safetyFactor` (the fraction of the local free-fall
time each outer step advances) was already a named default parameter
on `runFreefall()`, never overridden by any caller -- exposing it
needed no solver change, same as the other two sliders added earlier
today. New "step size (× t_ff)" slider, log-scale from 10^-2.5 to
10^-1 (roughly 0.3x to 10x the previous fixed default of 0.01).

Confirmed directly this has a *real* accuracy effect, not just a
speed one: coarsening by 10x cut the step count from 1231 to 154 and
the solve time from ~217ms to ~62ms, but also visibly shifted the
answer (final T 3628.6 K -> 2864.0 K) -- each step applies its
adiabatic compression as one instantaneous jump *before* letting
chemistry integrate over that interval, so a bigger step is a cruder
operator-splitting approximation, not just a blockier plot. Finer by
~3x (1231 -> 3820 steps) moved the answer the other way, converging
back toward the same value (3628.6 -> 3642.1 K) -- consistent with
approaching a continuum limit as steps shrink, not just noise.

**A genuinely new failure mode came with it**: a small enough step
size, combined with a wide enough target-density range, can now hit
`runFreefall()`'s `maxSteps` cap before actually reaching the
requested target -- previously untestable in practice (nothing let a
user push the step count that high), and previously silent (the run
would just stop, with nothing in the UI saying so). Checked directly
that the chosen slider range's full combination with every other
slider's own full range (including the widest target-n and the
lowest starting density) stays comfortably under the existing 10000-
step cap -- confirmed at 8686 steps at the most extreme combination
the exposed sliders can reach together, so this can't actually happen
through ordinary use of the new slider. Added a visible warning chip
in the run summary anyway (a new `reachedTarget` field on
`runFreefall()`'s own return value, checked in `updateRunSummary()`),
confirmed to work by deliberately forcing a slider value outside its
declared range in a test (not reachable through the UI itself) --
worth having regardless, since some future change to this page could
plausibly make the combination reachable again.

Verified: real Emscripten rebuild, headless-Chrome regression across
cool/free-fall modes, both themes, a network with H2 species and one
without, step size at coarser/finer/default (confirmed step count and
the real accuracy shift described above), the step-cap warning
(confirmed both that it stays silent across the full legitimate slider
range, and that it correctly appears when the cap is actually hit).
Full `pytest` suite unaffected (120/120).

**2026-09-11: "build your own network" prototype -- generic (data-
driven, no-recompile) mass-action kinetics engine, on a branch, not
merged.**

Prompted by a discussion of which parts of dengo's pipeline (network
definition -> codegen -> Emscripten compile -> browser widget) could
run inside e.g. a JupyterLite/Pyodide environment. Short version of
that discussion: definition/rates/codegen (sympy + Jinja2, pure Python)
already would, unmodified; the compile-to-wasm step is the genuine
bottleneck (no mature, shippable "C++ compiler in the browser"), but
there's more hope than expected there too -- a real, working
Numba-in-JupyterLite pipeline already does the moral equivalent
(llvmlite emits a wasm object, LLD links it in-process, Emscripten
loads it as a side module, all client-side, no server round-trip).
Redirected mid-discussion, twice, to a narrower and more useful target
than "define arbitrary networks via live Python": (1) pick species/
reactions from a preexisting, already-vetted catalog via checkboxes,
not by writing Python; (2) build the engine assuming a richer catalog
(CHIANTI ion-by-ion, UMIST) arrives later as a separate project --
don't block on it, don't design it away.

That reframing turned out to simplify the actual engineering a lot.
Ordinary mass-action kinetics -- `rate(T) * product of reactant
densities`, `d[X]/dt = net_stoichiometric_change * that same term` --
is *generic math*, identical for every reaction regardless of which
network it's in (see `Reaction.lhs_equation()`/`net_change()` in
reaction_classes.py, which this mirrors). It doesn't need sympy or
per-reaction code generation at all; it needs one hand-written
assembler plus a data table (species, stoichiometry, a rate(T) table)
of whatever reactions exist. So instead of "compile a solver for
whatever's checked", the split is: a network-agnostic Newton solver
(`wasm/generic_solver/dengo_generic.cpp`, vendoring `BE_chem_solve.C`
unmodified, generalized from the production wasm build's compile-time
`NSPECIES` to a runtime `nchem`) compiled *once*, ever, regardless of
what gets checked -- and a generic RHS/Jacobian assembler
(`wasm/generic_kinetics.js`) driven entirely by a reaction database
JSON (`wasm/generate_reaction_db.py`, reusing `build_primordial()`'s
already-registered species/reactions/rate functions directly, no new
chemistry authored) and whatever subset of it a user has checked
(`wasm/generic_ui.js`, `generic/index.html`). No em++ invocation
happens for any selection, ever, after the one-time build.

The JS callback boundary is Emscripten's `addFunction()`: the generic
`dengo_generic_step()` takes the exact same `rhs_f`/`jac_f` C function-
pointer types `BE_chem_solve.C` always took (it never knew or cared how
`calculate_rhs_<name>` was implemented) -- so a JS closure registered
via `addFunction()`, doing the mass-action sum directly against wasm
linear memory, is just as valid a function pointer to it as compiled C
was. A reaction is only selectable once every species it touches is
checked (mirrors `ChemicalNetwork.add_reaction(auto_add=False)`'s own
validation, not a new rule invented here).

Two real bugs on the way to a working version, both straightforward
once found: (1) `BE_chem_solve.C`'s own definition isn't `extern "C"`
(plain, name-mangled C++, same as the production build already links
against it), so declaring it `extern "C"` in the new generic wrapper
was a linker-symbol mismatch, not a real language boundary -- fixed by
matching its actual (mangled) linkage instead. (2) `Module.addFunction`
signature strings are `(1 return) + (every parameter)` letters, one
per 32-bit slot -- `rhs_f`/`jac_f` take 5 parameters
(`double*, double*, int, int, void*`, all i32-sized on wasm32) plus an
`int` return, i.e. 6 slots (`"iiiiii"`); using the 5-slot `"iiiii"`
produced `"function signature mismatch"` at call time, not at
compile/link time -- this class of bug won't be caught by anything
short of actually invoking the call, so it's worth remembering as a
specific, easy-to-get-wrong spot the next time a new C callback gets
wired up this way.

Verified: standalone Playwright cross-check of the generic JS RHS
assembler against the *existing, previously-validated* compiled
`hydrogen_minimal` wasm module, same initial conditions, same T (read
back from the compiled solver's own ge->T conversion rather than
independently inverting it) -- matched to 2e-5 relative error at full
(1024-point) rate-table resolution, worse (3e-4) at this prototype's
default 8x-downsampled (128-point) table; the residual is consistent
with the compiled solver interpolating its rate tables in log-T-
uniform-bin space while this prototype's `interpolateRate()` does a
plain linear search+interpolate against the same (log-spaced, not
downsampled-differently) T grid -- an interpolation-*scheme* mismatch,
not a stoichiometry/rate-law bug, and expected to shrink toward zero as
either table gets finer (confirmed: full-resolution error was ~14x
smaller than the default-downsampling error). Full end-to-end UI
regression: all 9 primordial species/22 reactions checked by default,
full-network run converges to the requested end time with zero console
errors; unchecking every He/H2/H- species correctly disables/unchecks
every reaction that needs one (leaving exactly hydrogen_minimal's own
k01/k02 subset selectable), and that restricted run also converges
cleanly.

**Explicitly out of scope for this prototype** (by design, not by
oversight -- see the discussion this came from): no thermal/cooling
coupling at all (fixed, user-dialed T only -- a cooling action's rate
of energy exchange is *not* generic mass-action math the way a
chemical reaction's rate is, so it doesn't fit this engine's one
formula the reactions do; would need its own, separate generalization
if pursued); no CHIANTI/UMIST/photoionization reactions in the catalog
yet (deliberately deferred to "a separate project" per the discussion
that prompted this -- the reaction-database JSON format should already
accommodate CHIANTI's ion-by-ion rates with zero changes, since those
are T-indexed exactly like every primordial_rates.py rate function;
UMIST's/reaction_classes.py's photoionization rates are z-/redshift-
indexed instead, which is the one real format gap a future project
extending the catalog would need to actually solve, not just a
labeling nicety -- flagged, not solved, here).

Not merged to main -- lives on `wasm-generic-kinetics-prototype`,
explicitly exploratory.

**2026-09-11, continued: cooling implemented.**

Asked directly to implement the "explicitly out of scope" cooling gap
from the entry above. Cooling actions are genuinely *not* one universal
formula the way reactions are (each is its own bespoke sympy
expression), so unlike reactions -- generic math, needing no per-
reaction code at all -- this needed lowering each action's *equation*
once via sympy's own `jscode` printer (`export_cooling_action()` in
generate_reaction_db.py), embedded as a JS expression string in the
exported JSON and turned into a real callable via `new Function()` at
load time. Still no per-*selection* codegen (every action in the
catalog is lowered regardless of what's later checked) and no em++/
compile step either way -- `new Function()` is JS's own built-in
"make a callable from a source string" primitive, not a build step.

Checking which cooling actions are actually *exportable* this way
turned out better than expected: of the primordial network's 17
cooling actions, only 2 (`gloverabel08`, `cie_cooling`) reference
symbols unresolvable from their own equation tree (dengo's C codegen
resolves them from hand-written surrounding C -- a critical-density/
optical-depth-approximation formula each -- not from the symbolic
equation alone); the other 15, including the non-trivial `h2formation`/
`h2formation_extra` (temporaries nested a level deep: `h2heatfrac`,
itself built from `ncrn`/`ncrd1`/`ncrd2` table lookups), lower cleanly.
Detected generically by checking `eq.free_symbols` against an
"accounted for" set (species + T + this action's own renamed table
symbols + `ge`, the last needed only because `ReactionCoefficient.
free_symbols` -- reaction_classes.py -- always forces `ge` into the set
regardless of whether an equation is actually ge-dependent, dengo's own
mechanism for symbolically differentiating a coefficient w.r.t. energy)
-- not hardcoded by action name, so this keeps working correctly if
primordial_cooling.py's own set of actions ever changes.

Design choices, each a real (documented, not hidden) simplification
versus the compiled solver:
- **Single constant gamma=5/3** (monatomic ideal gas) for the ge<->T
  conversion, instead of the compiled solver's T-dependent interpolated
  gamma for H2-bearing gas (roto-vibrational degrees of freedom
  activating). ge<->T is then closed-form both directions (no
  bisection needed, unlike app.js's own geForTemperature()) --
  `ge = n_total*kB*T / ((gamma-1)*mdensity_amu*mh)`, generic from
  whichever species are active via their already-exported `weight`.
- **z=0 always** for Compton cooling (the only cooling action using
  redshift) -- matches this project's existing compiled widget's own
  established convention (see the "why is z always 0" note on
  app.js's IC_PRESETS).
- **Approximate (not exact) Jacobian** for the `ge` row/column: the
  species-species block stays exact/analytic (unchanged from the
  chemistry-only prototype); `d(ge_rhs)/d(species)` and
  `d(everything)/d(ge)` (including the T-dependence of reaction rates
  now that T isn't fixed) are finite-differenced instead of derived
  analytically. The compiled solver gets these exactly, via
  `ReactionCoefficient._eval_derivative()`'s precomputed `dr<name>`
  tables -- reproducing that here would mean symbolically
  differentiating every jscode-lowered cooling expression *and* every
  reaction's own rate table w.r.t. T, real additional work for what's
  ultimately a Newton-convergence aid, not something that changes what
  a *converged* answer means (BE_chem_solve.C's convergence check is on
  the actual residual/update norm, not Jacobian fidelity). A deliberate
  scope cut, not an oversight.

Verified: a physically unambiguous sanity check, not just "it runs
without errors" -- primordial gas at T=1e5 K (H2/H- species unchecked,
so no formation-heating channel exists at all) with only the atomic
cooling actions checked (collisional excitation/ionization, radiative
recombination, bremsstrahlung, Compton) cools from 1.000e5 K to 6.209e3
K over the run -- a large, correctly-signed net *cooling*, confirming
the mdensity normalization, the jscode-lowered expressions' signs, and
the ge<->T conversion all agree with each other rather than merely
"not crashing". Separately, the full default-conditions run (all 9
species, all 15 exportable cooling actions, T0=1000K, a cool/mostly-
neutral starting point where net cooling power should genuinely be
small) showed only a modest T change (1000K -> 1001K) over the same
span -- also consistent, not a sign of a sign error, since collisional
cooling scales with ionization fraction and this starting point is
only trace-ionized. Full existing-page regression (all three fiducial
networks, sweep, rates) and the chemistry-only subset-selection check
(hydrogen_minimal's own k01/k02 reproduced from the full catalog) both
re-run clean after this change, zero console errors. CSV export
confirmed to carry a new `T_K` column with the evolving values.
