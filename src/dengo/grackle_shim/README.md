# dengo-grackle

A drop-in replacement for Grackle's C library (`libgrackle` + `grackle.h`,
Grackle 3.4.1 API) backed by dengo's primordial H/He/H2 network. A code
written against Grackle, such as Enzo, links it without source changes.

```bash
uv run python -m dengo.grackle_shim --prefix /path/to/dengo-grackle
```

This writes:

- `include/`: Grackle 3.4.1's public headers, copied unchanged (Enzo Public
  License; see `share/dengo-grackle/GRACKLE_LICENSE`), with `gr_float` set
  to `double`.
- `lib/libgrackle.a` and `lib/libgrackle.so`.
- `share/dengo-grackle/`: the generated sources it was built from, plus
  `BUILD_INFO` (the dengo revision, and whether OpenMP was used).

Point a Grackle build at the prefix. For Enzo with CMake:
`-DENZO_USE_GRACKLE=ON -DGRACKLE_ROOT=/path/to/dengo-grackle`. With the
classic Makefile build, set `MACH_INCLUDES_GRACKLE`/`MACH_LIBS_GRACKLE` to
the prefix's `include/` and `lib/`. Nothing at run time needs Python, HDF5
or a data file: the rate tables are compiled into the library.
`grackle_data_file` is accepted and ignored.

## Scope

This covers Grackle's `primordial_chemistry = 2` network: HI, HII, HeI,
HeII, HeIII, H-, H2, H2+ and e-, with radiative cooling. If any setting
outside that is requested, `initialize_chemistry_data()` prints which one
and returns FAIL, rather than silently computing different physics.
Settings that are refused:

- `metal_cooling`, `UVbackground`, any dust setting
- self-shielding, radiative transfer, a Lyman-Werner background
- volumetric or specific heating fields, `use_temperature_floor`
- `CaseBRecombination`, and `Gamma` other than 5/3
- turning off any of Grackle's individual rate switches

Settings that are honoured:

| Grackle setting | What it does here |
|---|---|
| `three_body_rate` 0–5 | Same fits and numbering as Grackle. |
| `cie_cooling` | Switches the CIE cooling term and its continuum optical-depth suppression on or off. |
| `h2_optical_depth_approximation` | Switches the H2 line-cooling optical-depth suppression on or off. |
| `max_iterations` | Maximum accepted sub-steps per cell per call. |
| `exit_after_iterations_exceeded` | Return FAIL when a cell does not reach `dt`. |
| `omp_nthreads` | Number of OpenMP threads. |
| `HydrogenFractionByMass` | Used when making the species consistent with the gas density, as in Grackle. |
| `cmb_temperature_floor` | Accepted. In Grackle it only affects metal and HD cooling, so it has no effect on this network in either code. |

Functions:

- `solve_chemistry`, `calculate_cooling_time`, `calculate_temperature`, and
  their `local_*` forms.
- Parameter defaults and the dynamic parameter API.
- `grackle_rates.k1` and `k2`, filled on Grackle's own temperature grid and
  in its code units. Enzo's photon timestep reads them directly.

`calculate_gamma`, `calculate_pressure`, `calculate_dust_temperature` and
`gr_query_units` return FAIL. Enzo calls none of them.

## How it maps onto dengo

Units follow Grackle's conventions (see the header of `grackle_shim.C`):

- species densities are converted with `density_units`, using masses of 1,
  4 and 2 m_H for H, He and H2 species (`e_density` is n_e·m_H);
- `internal_energy` is multiplied by `velocity_units²`;
- the redshift is z = 1/(a_value·a_units) − 1.

Only the interior cells (`grid_start` to `grid_end`) are touched. Every
cell is integrated independently, with its own adaptive sub-steps, in
parallel with OpenMP. The results do not depend on the thread count.
After the solve, species are rescaled to the gas density exactly as
Grackle's `make_consistent_g` does.

`calculate_cooling_time` returns e/(de/dt) in code time units, signed
(negative when the gas is cooling), from radiative processes only. Grackle
also leaves H2 formation heating and dissociation cooling out of its
cooling time.

If a cell does not reach `dt`, the default is the same as Grackle's: print
a warning, keep that cell's last converged state, and return SUCCESS.

## Validation

See `tests/test_grackle_shim.py`. `tests/grackle_shim/compare_grackle.c`
makes Enzo's sequence of calls on a ghost-zone-padded 3D grid that spans
n = 10⁻² to 10¹³ cm⁻³ and T = 30 to 2×10⁴ K.

- **Against dengo itself:** for every cell, the result matches the Cython
  `Solver` bit for bit (proper and comoving units), once Grackle's unit
  conventions and `make_consistent` are applied. The cooling time matches
  to 1e-12. Ghost zones are left untouched.
- **Against the real libgrackle 3.4.1** (the copy bundled in the gracklepy
  wheel; set `DENGO_REAL_LIBGRACKLE`):
  - the initial-state temperatures agree to under 1%, from the slightly
    different m_H constants;
  - the `k1`/`k2` tables agree to 10⁻³;
  - proper and comoving runs give identical results in both codes;
  - cooling times agree in sign for every cell, with a 0.6% median
    difference.

**Inside Enzo:** unmodified enzo-dev builds and runs against this library.
`examples/enzo_grackle_shim/` has the setup, scripts and results: a
primordial CoolingTest and a one-zone free-fall collapse, compared with
the real libgrackle.

Chemistry after a step is not asserted against Grackle, because the two
codes use different rate fits. Known differences in dengo's physics, from
the comparison grid and the Enzo runs:

- **No collisional ionization by neutral atoms.** dengo lacks Grackle's
  k57 (H + H) and k58 (H + He). Fully neutral gas heated to ~10⁴ K
  therefore stays neutral, because there are no electrons to seed
  electron-impact ionization.
- **Free-fall collapse above n_H ≈ 10¹³ cm⁻³:** dengo dissociates its H2
  and heats to ~5800 K, where Grackle stays molecular at ~2000 K. Below
  that density the two tracks agree to a few percent.

- **Tables cover 10 K to 10⁸ K** in dengo, against 1 K to 10⁹ K in
  Grackle. Rates outside that range are held at the edge values.
- **Dense gas held at fixed density for about 1 Myr** (n ≳ 10¹⁰ cm⁻³):
  dengo cools to its 10 K table floor and reports those cells as not
  converged, while Grackle levels off at 60–80 K. Over 1000-year steps
  both codes agree closely. In a collapsing cloud, compressional heating
  keeps such gas far from this regime, but it has not been checked inside
  an actual Enzo run.
- **Three-body H2 formation is different at n ≈ 10⁹–10¹⁰ cm⁻³ and
  T ≈ 4000–10,000 K**: Grackle forms H2 much faster there. This is the same
  rate-fit difference seen in the earlier free-fall comparisons
  (`NOTES.md`).
- **Electron fractions at 10²–10⁷ cm⁻³ and low T** differ by up to ~2×
  after 1000 years. The recombination rate `k2` matches, so this comes
  from the other channels that create and destroy electrons.
