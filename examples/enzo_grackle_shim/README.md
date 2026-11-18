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

- **Above n_H ≈ 10¹³ cm⁻³, dengo dissociates its H2.** It heats to about
  5800 K and re-forms H2 by ~10¹⁶ cm⁻³; Grackle stays molecular at
  ~2000 K. The cause hasn't been found yet. This is the protostellar
  regime, so it matters for Pop III collapse.

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

dengo-grackle is far slower than real Grackle on the CoolingTest
(1775 cells × 1289 cycles, wall time for the whole run):

| Library | Wall time |
|---|---|
| Real Grackle 3.4.1 (gracklepy wheel build) | 14 s |
| dengo-grackle, `OMP_NUM_THREADS=1` | 577 s (~250 µs per cell per step) |
| dengo-grackle, 4 threads | 169 s |
| dengo-grackle, 24 threads | 56 s |

That's roughly 40× slower per core. It hasn't been profiled. The likely
cause is the sub-stepping policy (`_advance`). Each call starts at
`dt / max_iterations`, which with Grackle's default `max_iterations =
10000` is `dt/10⁴`, and only doubles from there. So every cell takes at
least ~14 implicit sub-steps per hydro step, even when a single step
would converge. Starting at the full `dt` and halving on failure is the
obvious thing to try. That changes how robust the solver is, so it's
left for a separate change with its own validation.

