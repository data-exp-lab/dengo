"""Efficiency comparison structured like an actual simulation coupling:
build each solver's persistent, dims-independent state ONCE (network
codegen + compile for dengo, chemistry_data.initialize() -- which reads
the Cloudy HDF5 tables -- for Grackle) and report that one-time cost
separately, since a real simulation pays it once at program start, not
per grid patch or per timestep. Then, at each grid size, build only the
dims-dependent handle (Solver(dims) / FluidContainer(dims)) and report
THAT setup cost separately from steady-state per-timestep cost -- the
pattern a real hydro/AMR code's chemistry coupling actually uses (one
handle per patch, reused for every timestep that patch takes), not the
single-cell free-fall driver pattern used everywhere else in
.grackle_compare/.

Three calling conventions compared at each grid size:
  - dengo native (Solver.step_inplace(), zero-copy self.state/self.T --
    how a hydro code integrating dengo directly, not through the
    compat shim, would call it)
  - dengo via grackle_compat (FluidContainer.solve_chemistry() -- the
    cost of drop-in API compatibility, unit conversions included)
  - real gracklepy (FluidContainer.solve_chemistry())

Run under .grackle_compare/.venv (needs gracklepy):
    .grackle_compare/.venv/bin/python .grackle_compare/benchmark_amortized.py [dims ...]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import gracklepy
from gracklepy.utilities.physical_constants import mass_hydrogen_cgs, sec_per_Myr, cm_per_mpc

import dengo.grackle_compat as compat
from dengo.primordial_network import build_network, build_solver

KB = 1.3806504e-16
MH = 1.67e-24
G_GRAV = 6.674e-8

BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dengo_build")

# Representative target-regime state (n=1e13 cm^-3, T=1500K, moderately
# molecular -- partway through the free-fall collapse this project
# targets). dt is 1e-3 of the local free-fall time at this density -- a
# physically motivated "one hydro timestep" at this density, not an
# arbitrary placeholder. Both this density/temperature and this dt
# fraction were chosen by checking that both solvers actually converge
# cleanly step after step (see NOTES.md) -- a more extreme,
# further-along snapshot (n=1e15, T=2000K, 70% molecular) hit Grackle's
# max_iterations cap immediately, which would have measured the cost of
# failing to converge, not of a normal solve.
NH = 1.0e13
T0 = 1500.0
RHO = NH * MH
DT_SECONDS = 1.0e-3 * np.sqrt(3.0 * np.pi / (32.0 * G_GRAV * RHO))

NAME_TO_FIELD = {
    "H_1": "HI_density", "H_2": "HII_density", "He_1": "HeI_density",
    "He_2": "HeII_density", "He_3": "HeIII_density", "H_m0": "HM_density",
    "H2_1": "H2I_density", "H2_2": "H2II_density", "de": "e_density",
}
AMU = {"H_1": 1.00794, "H_2": 1.00794, "He_1": 4.002602, "He_2": 4.002602,
       "He_3": 4.002602, "H_m0": 1.00794, "H2_1": 2.01588, "H2_2": 2.01588, "de": 1.00794}


def build_grackle_chemistry():
    my_chemistry = gracklepy.chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 1
    my_chemistry.primordial_chemistry = 2
    my_chemistry.metal_cooling = 0
    my_chemistry.CaseBRecombination = 1
    my_chemistry.cie_cooling = 1
    my_chemistry.h2_optical_depth_approximation = 1
    my_chemistry.three_body_rate = 4
    grackle_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".grackle", "grackle_data_files", "input",
    )
    my_chemistry.grackle_data_file = os.path.join(grackle_data_dir, "CloudyData_noUVB.h5")
    my_chemistry.comoving_coordinates = 0
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0
    my_chemistry.density_units = mass_hydrogen_cgs
    my_chemistry.length_units = cm_per_mpc
    my_chemistry.time_units = sec_per_Myr
    my_chemistry.set_velocity_units()
    return my_chemistry


def build_compat_chemistry():
    my_chemistry = compat.chemistry_data()
    my_chemistry.CaseBRecombination = 1
    my_chemistry.cie_cooling = 1
    my_chemistry.h2_optical_depth_approximation = 1
    my_chemistry.three_body_rate = 4
    my_chemistry.density_units = mass_hydrogen_cgs
    my_chemistry.length_units = cm_per_mpc
    my_chemistry.time_units = sec_per_Myr
    my_chemistry.set_velocity_units()
    return my_chemistry


def h2_ics(dims):
    """Uniform across all `dims` cells -- a throughput benchmark, not a
    physics comparison, so identical cells are fine (a real grid would
    have cell-to-cell variation, but that doesn't change the per-cell
    solve cost in any systematic way this benchmark needs to capture)."""
    return {
        "H_1": np.full(dims, NH * 0.4), "H_2": np.full(dims, NH * 1e-6),
        "He_1": np.full(dims, NH * 0.24 / 4), "He_2": np.full(dims, NH * 1e-8),
        "He_3": np.full(dims, NH * 1e-10), "H_m0": np.full(dims, NH * 1e-10),
        "H2_1": np.full(dims, NH * 0.1), "H2_2": np.full(dims, NH * 1e-9),
        "de": np.full(dims, NH * 1e-6), "ge": np.full(dims, 1.5 * KB * T0 / MH),
    }


def fill_fc_from_ics(fc, cd, ics):
    for name, field in NAME_TO_FIELD.items():
        fc[field][:] = ics[name] * AMU[name] * MH / cd.density_units
    fc["internal_energy"][:] = ics["ge"] / cd.velocity_units ** 2
    # total mass density -- with h2_optical_depth_approximation=1,
    # Grackle uses this internally for an H2-column-density-based
    # cooling suppression estimate; leaving it unset (zero) produced a
    # nonsensical suppression factor and made the solve diverge to the
    # temperature floor within one call (see NOTES.md).
    fc["density"][:] = sum(ics[name] * AMU[name] * MH for name in NAME_TO_FIELD) / cd.density_units


def time_calls(fn, n_warmup=1, n_timed=20):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_timed):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / n_timed


def bench_dengo_native(dims, mod, n_timed):
    t0 = time.perf_counter()
    solver = mod.Solver(dims)
    t1 = time.perf_counter()
    ics = h2_ics(dims)
    for name, val in ics.items():
        solver.state[:, mod.SPECIES_INDEX[name]] = val
    per_step = time_calls(lambda: solver.step_inplace(DT_SECONDS, niter=200, reltol=1e-5),
                           n_timed=n_timed)
    solver.close()
    return (t1 - t0), per_step


def bench_dengo_compat(dims, cd, n_timed):
    t0 = time.perf_counter()
    fc = compat.FluidContainer(cd, dims)
    t1 = time.perf_counter()
    ics = h2_ics(dims)
    fill_fc_from_ics(fc, cd, ics)
    dt_code = DT_SECONDS / cd.time_units
    per_step = time_calls(lambda: fc.solve_chemistry(dt_code), n_timed=n_timed)
    fc.close()
    return (t1 - t0), per_step


def bench_grackle(dims, cd, n_timed):
    t0 = time.perf_counter()
    fc = gracklepy.FluidContainer(cd, dims)
    t1 = time.perf_counter()
    ics = h2_ics(dims)
    fill_fc_from_ics(fc, cd, ics)
    dt_code = DT_SECONDS / cd.time_units
    per_step = time_calls(lambda: fc.solve_chemistry(dt_code), n_timed=n_timed)
    return (t1 - t0), per_step


def main():
    dims_list = [int(a) for a in sys.argv[1:]] or [1, 2048, 100_000]
    print("dt = %.4e s (1e-3 of free-fall time at n=%.1e cm^-3, T=%.0fK)" % (DT_SECONDS, NH, T0))

    # One-time costs: paid once at program start in a real simulation,
    # not per grid patch/timestep -- measured here ONCE, outside the
    # per-dims loop, so the per-dims "setup" column below only reflects
    # what actually scales with grid size (handle/FluidContainer
    # construction), not first-time codegen/compile/table-read.
    t0 = time.perf_counter()
    network = build_network()
    mod = build_solver(network, BUILD_DIR)
    t1 = time.perf_counter()
    print("one-time: dengo network codegen + compile:        %8.3f s" % (t1 - t0))

    t0 = time.perf_counter()
    compat.use_existing_solver_module(mod)  # share the build above -- same network,
    # no reason to pay for a second compile of the identical solver.
    t1 = time.perf_counter()
    print("one-time: dengo (via grackle_compat), sharing the build above: %8.3f s" % (t1 - t0))

    t0 = time.perf_counter()
    cd_grackle = build_grackle_chemistry()
    cd_grackle.initialize()
    t1 = time.perf_counter()
    print("one-time: grackle chemistry_data.initialize() (Cloudy HDF5 read): %8.3f s" % (t1 - t0))

    cd_compat = build_compat_chemistry()
    cd_compat.initialize()

    print()
    print("%10s  %-12s  %14s  %12s  %14s" %
          ("dims", "backend", "handle setup", "us/call", "us/call/cell"))

    for dims in dims_list:
        n_timed = 20 if dims < 10_000 else 5
        for label, fn in [
            ("dengo-native", lambda: bench_dengo_native(dims, mod, n_timed)),
            ("dengo-compat", lambda: bench_dengo_compat(dims, cd_compat, n_timed)),
            ("grackle", lambda: bench_grackle(dims, cd_grackle, n_timed)),
        ]:
            setup_s, per_call_s = fn()
            print("%10d  %-12s  %14.5f  %12.2f  %14.4f" %
                  (dims, label, setup_s, 1e6 * per_call_s, 1e6 * per_call_s / dims))


if __name__ == "__main__":
    main()
