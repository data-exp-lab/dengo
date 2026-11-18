# Running unmodified Enzo on dengo-grackle

How dengo-grackle (`src/dengo/grackle_shim/`) was checked inside a real,
unmodified Enzo, and what came out. The comparison builds the same Enzo
checkout twice: once against dengo-grackle and once against the real
libgrackle 3.4.1.

## Build

```bash
# 1. dengo-grackle
uv run python -m dengo.grackle_shim --prefix $PWD/dengo-grackle

# 2. Enzo (github.com/enzo-project/enzo-dev; tested at 7409c9a), no source changes
cmake -S enzo-dev -B build-dengo -DCMAKE_BUILD_TYPE=Release \
      -DENZO_USE_GRACKLE=ON -DGRACKLE_ROOT=$PWD/dengo-grackle \
      -DENZO_USE_MPI=OFF -DENZO_USE_UUID=OFF
cmake --build build-dengo -j
```

For the reference build, point `GRACKLE_ROOT` at a real Grackle 3.4.1
install. Without one, use a directory with dengo-grackle's `include/` (the
headers are identical) and a `lib/` holding the gracklepy wheel's
`libgrackle-3.4.1.so`. Run that binary with `LD_LIBRARY_PATH` set to the
wheel's `gracklepy.libs/`.

Things that bit along the way, none of them dengo-specific:

- **Enzo writes `bin/enzo` into the source tree**, so two builds from one
  checkout overwrite each other. Copy the binary out after each build.
- **Enzo's Fortran `ERROR_MESSAGE` macro expands `__FILE__`** into a
  132-column fixed-form line. A deep source path (≳100 characters)
  overflows it and `acml_st1.F` fails to compile. Build from a short path,
  such as a symlink.
- **Enzo needs HDF5.** Here it came from `apt-get download` + `dpkg -x` into
  `.grackle_deps/`, passed via `CMAKE_INCLUDE_PATH`/`CMAKE_LIBRARY_PATH`;
  see NOTES.md.

## Test problems

- **`CoolingTest_Primordial.enzo`**: Enzo's `run/Cooling/CoolingTest_Grackle`
  switched to Grackle's 9-species network (`MultiSpecies = 2`, no metals). A
  5×5×71 grid spans n_H = 0.1–1000 cm⁻³ and T = 10–10⁸ K, with energies
  reset after every step. Compare the runs with `enzo_compare.py RUN_A RUN_B`.
- **`OneZoneFreefall_Primordial.enzo`**: Enzo's one-zone free-fall test
  (problem 63) made primordial.
  - With real Grackle 3.4.1 this problem **segfaults** inside Grackle's
    `ceiling_species_g`. Grackle reads three entries of
    `grid_dimension`/`grid_start`/`grid_end` whatever `grid_rank` is, while
    Enzo allocates only `GridRank` entries, and this problem only allows rank
    1 or 2. That's a latent Enzo + Grackle problem for any rank < 3 grid;
    dengo-grackle reads only `grid_rank` entries.
  - So the real-Grackle reference comes from `onezone_driver.py` instead. It
    reimplements Enzo's one-zone step (`Grid_SolveOneZoneFreefall.C`,
    `Grid_ComputeOneZoneCollapseFactor.C`) on top of gracklepy. Run it as-is
    for real Grackle, or with `LD_PRELOAD=dengo-grackle/lib/libgrackle.so`
    to drive dengo-grackle through Grackle's own, unmodified Python API.
  - Get the initial state from the Enzo run with
    `freefall_tracks.py init RUN INIT.json`, and tabulate the tracks with
    `freefall_tracks.py extract` and `freefall_tracks.py compare`.

## Results (2026-11-18)

### What matches

- **No Enzo changes.** Unmodified Enzo builds, links and runs both
  problems against dengo-grackle, including with radiative transfer
  compiled in, which exercises `grackle_rates.k1`/`k2`.
- **CoolingTest time-stepping is identical.** Both libraries take the same
  1289 cycles at the same dump times. Temperatures agree to 0.1% (median).
- **The free-fall track through Enzo matches the standalone driver.** Enzo
  + dengo-grackle and the driver + dengo-grackle agree to <1% in T and
  f_H2 from n_H = 10 to 10¹¹ cm⁻³. The shim behaves the same inside Enzo
  as through Grackle's Python API.
- **dengo agrees with real Grackle up to n_H ≈ 10¹³ cm⁻³**: T within a
  few percent, f_H2 within ~10%.

### Physics differences (in dengo, not the shim)

- **Above n_H ≈ 10¹³ cm⁻³, dengo dissociates its H2 under this driving.**
  It heats to about 5800 K and re-forms H2 by ~10¹⁶ cm⁻³; Grackle stays
  molecular at ~2000 K. This is the protostellar regime, so it matters for
  Pop III collapse.
  - **Not the shim.** dengo's pre-existing Python compat layer (Cython
    `Solver`, no C library) gives the same track under the same driver.
  - **Different from September.** NOTES.md's decoupled comparison found
    dengo staying molecular (f_H2 ≈ 0.98–1.0) through 10¹³–3×10¹⁵ cm⁻³
    within ~2–34% of Grackle in T. That run used a different driver and
    initial state. Enzo's one-zone test heats with a fixed Gamma = 5/3
    and starts with helium fully ionized.
  - **Open question:** which part of this driving tips dengo, and not
    Grackle, into dissociation. dengo already runs ~3–10% hotter than
    Grackle at 10¹²–10¹³ cm⁻³, just before the runaway.

  | n_H [cm⁻³] | Enzo + dengo: T, f_H2 | driver + Grackle: T, f_H2 |
  |---|---|---|
  | 10¹² | 1457 K, 0.65 | 1496 K, 0.64 |
  | 10¹³ | 2039 K, 0.41 | 2025 K, 0.44 |
  | 10¹⁴ | 5774 K, 2×10⁻⁶ | 2066 K, 0.63 |
  | 10¹⁶ | 2774 K, 0.44 | 1943 K, 0.75 |

- **Neutral gas heated to 6000–16,000 K stays neutral in dengo.** dengo's
  network has no collisional ionization by neutral atoms (Grackle's k57,
  H + H, and k58, H + He; Lenzuni, Chernoff & Salpeter 1991). Starting
  from Enzo's fully neutral CoolingTest state, dengo cannot seed
  ionization, and its electron fraction stays ~10⁻⁸–10⁻¹³ of Grackle's.
  Gas with the usual residual ionization (~10⁻⁴) isn't affected.
- **Species after 10⁴ yr otherwise differ by 20–35% (median)** in the
  CoolingTest, from different rate fits. The largest gaps are H2 line
  cooling at 600–2000 K.

### Performance

Measured on the CoolingTest: 1775 cells × 1289 cycles, about 2.29 M
cell-solves. `DENGO_GRACKLE_STATS=1` makes the library print its own
counters at exit.

| Setup | Wall time | Per cell-solve |
|---|---|---|
| Real Grackle 3.4.1 (gracklepy wheel build, serial) | 14 s | ~5 µs (including Enzo) |
| dengo-grackle, 1 thread | 170 s | 70.9 µs |
| dengo-grackle, 24 threads | 24 s | 6.5 µs |

How these numbers were checked:

- **Sub-steps match standalone.** Inside Enzo, each cell-solve takes
  8.10 accepted + 0.13 rejected sub-steps on average. dengo's standalone
  Cython `Solver`, run on the same states with the same `dt`, takes 8.10.
- **The shim adds only ~4%.** Before the OpenMP fix below, standalone
  cost 146 µs per cell-solve against 153 µs inside Enzo.
- **Setup is not repeated in any way that matters.** Per-call setup (the
  table copy) is 0.1% of instructions (valgrind callgrind).
- **Caches are not a problem.** Cachegrind shows an L1-data miss rate of
  1.0% and a last-level miss rate of 0.0%. The 2.4% instruction-cache miss
  rate comes mostly from the long, generated Jacobian.

**Fixed, 2026-11-18:** about half the cost was nested-OpenMP overhead.
The C API parallelizes over cells, and inside that region every per-cell
RHS, Jacobian and rate call entered its own `omp parallel` region
(serialized by its `if` clause). libgomp still sets up, and allocates, a
team for each of those, 5.8 M times in a 30-cycle run. The library now
compiles the solver core without OpenMP; only the loop over cells is
parallel. That cut the cost from 153 to 71 µs per cell-solve (2.15×),
with identical results. Earlier fixes in the same session: the first
sub-step now starts at `dt/200`, as dengo was tuned (it had been
`dt/max_iterations`); before both fixes the single-thread run took 577 s.

**Remaining gap: ~12× per core.** Most cells in this test are easy, low
density and close to equilibrium. Grackle crosses those in about one
sub-cycle, while dengo's policy always takes about 8 implicit sub-steps
(doubling from `dt/200`), at ~8.7 µs each. Letting easy cells try larger
first steps would close much of that. It changes the solver's
robustness, so it hasn't been done here. The September C benchmark, where
dengo was faster at grid scale, used a stiff n = 10¹³ cm⁻³ state, where
Grackle needs many sub-cycles too.
