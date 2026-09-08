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
