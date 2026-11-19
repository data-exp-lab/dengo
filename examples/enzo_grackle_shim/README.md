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

| Setup | Wall time | Per cell-solve | Sub-steps per cell-solve |
|---|---|---|---|
| Real Grackle 3.4.1 (gracklepy wheel build, serial) | 14 s | ~5 µs (including Enzo) | – |
| dengo-grackle, 1 thread | 35 s | 11.8 µs | 1.20 + 0.22 rejected |
| dengo-grackle, 24 threads | 12.6 s | 1.9 µs | same |

How dengo-grackle got there, single-threaded, all on 2026-11-18:

| Change | Per cell-solve | Effect on results |
|---|---|---|
| First version | ~250 µs | – |
| First sub-step `dt/200` instead of `dt/max_iterations` | 153 µs | small (median 0.06%) |
| Solver core compiled without OpenMP (nested teams were ~half the cost) | 71 µs | bit-identical |
| `-fno-math-errno` | 61 µs | bit-identical |
| Integer powers generated as products (`pow()` calls: 22 → 4) | 54 µs | ~1e-15 |
| First sub-step = min(dt, 0.1 × the cell's shortest timescale) | **11.8 µs** | see below |

How these numbers were checked:

- **Sub-steps match standalone.** Inside Enzo, the sub-step counts match
  dengo's standalone solver on the same states with the same `dt`.
- **The shim adds only ~4%** on top of the standalone solver.
- **Setup is not repeated in any way that matters.** Per-call setup (the
  table copy) is 0.1% of instructions (valgrind callgrind).
- **Caches are not a problem.** Cachegrind shows an L1-data miss rate of
  1.0% and a last-level miss rate of 0.0%.

The timescale-based first step was chosen by checking all candidate
policies against a tight BDF reference (NOTES.md). On easy cells it is
3–7× cheaper, for about twice the solver's existing truncation error
there. In this test, the outputs move by a median of ≲1% in species,
except for electron fractions near the 10⁻²⁰ floor in cold neutral gas,
which can't ionize at all (no k57/k58). The free-fall collapse track
moves by <0.5% in T.

What's left per sub-step: the dense 10×10 linear solve is about a third
of the instructions (NOTES.md has the options), and dengo's sub-steps
carry no truncation-error control, which is the next item.
