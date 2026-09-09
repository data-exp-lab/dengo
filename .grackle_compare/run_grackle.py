"""
Run Grackle's own primordial (9-species, primordial_chemistry=2, no
metals/UVB) free-fall collapse test, matching dengo's
examples/free_fall_collapse.py as closely as possible: same species set
(H, H+, He, He+, He++, H-, H2, H2+, e-), same Omukai et al. (2005)
free-fall scheme, same safety_factor and target density.
"""
import json
import os
import time

import numpy as np

from gracklepy import (
    chemistry_data,
    evolve_constant_density,
    setup_fluid_container,
)
from gracklepy.utilities.evolve import add_to_data, calculate_collapse_factor
from gracklepy.utilities.physical_constants import (
    gravitational_constant_cgs,
    mass_hydrogen_cgs,
    sec_per_Myr,
    cm_per_mpc,
)

HERE = os.path.dirname(os.path.abspath(__file__))
GRACKLE_DATA_DIR = os.path.join(
    os.path.dirname(HERE), ".grackle", "grackle_data_files", "input"
)


def evolve_freefall_timed(fc, final_density, safety_factor=0.01, include_pressure=True):
    """Same as gracklepy.utilities.evolve.evolve_freefall, verbatim,
    except it also records a perf_counter() wall-clock timestamp and the
    per-step dt every iteration, to see where the wall-clock cost and any
    step-size instability actually live across the run."""
    from collections import defaultdict

    my_chemistry = fc.chemistry_data
    gravitational_constant = (
        4.0 * np.pi * gravitational_constant_cgs *
        my_chemistry.density_units * my_chemistry.time_units**2)
    freefall_time_constant = np.power(((32. * gravitational_constant) /
                                       (3. * np.pi)), 0.5)

    data = defaultdict(list)
    wall = []
    dts = []
    current_time = 0.0
    while fc["density"][0] * my_chemistry.density_units < final_density:
        dt = safety_factor * \
          np.power(((3. * np.pi) /
                    (32. * gravitational_constant *
                     fc["density"][0])), 0.5)
        add_to_data(fc, data, extra={"time": current_time})
        if include_pressure:
            force_factor = calculate_collapse_factor(data["pressure"], data["density"])
        else:
            force_factor = 0.
        data["force_factor"].append(force_factor)

        new_density = np.power((np.power(fc["density"][0], -0.5) -
                                (0.5 * freefall_time_constant * dt *
                                 np.power((1 - force_factor), 0.5))), -2.)
        density_ratio = new_density / fc["density"][0]
        for field in fc.density_fields:
            fc[field] *= density_ratio
        fc["internal_energy"][0] += (my_chemistry.Gamma - 1.) * \
          fc["internal_energy"][0] * freefall_time_constant * \
          np.power(fc["density"][0], 0.5) * dt

        t_before = time.perf_counter()
        fc.solve_chemistry(dt)
        wall.append(time.perf_counter() - t_before)
        dts.append(float(dt))

        current_time += dt

    for field in data:
        data[field] = np.squeeze(np.array(data[field]))
    result = fc.finalize_data(data=data)
    result["wall_per_step"] = np.array(wall)
    result["dt_per_step"] = np.array(dts)
    return result


def build_chemistry():
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 1
    my_chemistry.primordial_chemistry = 2  # H,H+,He,He+,He++,H-,H2,H2+,e- -- matches dengo's set
    my_chemistry.metal_cooling = 0
    my_chemistry.dust_chemistry = 0
    my_chemistry.photoelectric_heating = 0
    my_chemistry.self_shielding_method = 0
    my_chemistry.H2_self_shielding = 0
    my_chemistry.CaseBRecombination = 1
    my_chemistry.cie_cooling = 1
    my_chemistry.h2_optical_depth_approximation = 1
    my_chemistry.three_body_rate = 4  # matches dengo's ChemicalNetwork.threebody default
    my_chemistry.grackle_data_file = os.path.join(GRACKLE_DATA_DIR, "CloudyData_noUVB.h5")

    my_chemistry.comoving_coordinates = 0
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0
    my_chemistry.density_units = mass_hydrogen_cgs
    my_chemistry.length_units = cm_per_mpc
    my_chemistry.time_units = sec_per_Myr
    my_chemistry.set_velocity_units()
    return my_chemistry


def main():
    my_chemistry = build_chemistry()

    initial_temperature = 50000.0
    initial_density = 1e-1 * mass_hydrogen_cgs
    final_density = 3e15 * mass_hydrogen_cgs  # match dengo's n_target

    fc = setup_fluid_container(
        my_chemistry,
        density=initial_density,
        temperature=initial_temperature,
        state="ionized",
        converge=False,
    )

    t0 = time.perf_counter()
    # cool at constant density to get sensible starting abundances, same
    # recipe as grackle's own examples/freefall.py
    data0 = evolve_constant_density(fc, final_temperature=300.0, safety_factor=0.1)
    t1 = time.perf_counter()
    data = evolve_freefall_timed(fc, final_density, safety_factor=0.01, include_pressure=True)
    t2 = time.perf_counter()

    print("cool-down phase: %d steps, %.3f s" % (len(data0["time"]), t1 - t0))
    print("free-fall phase: %d steps, %.3f s" % (len(data["time"]), t2 - t1))
    print("total wall time: %.3f s" % (t2 - t0))

    # NOTE: data["time"] from finalize_data() is already in physical
    # seconds (gracklepy converts it internally) -- multiplying by
    # my_chemistry.time_units again here was a real bug in an earlier
    # version of this script (inflated the printed elapsed time by
    # ~3e13x, i.e. exactly time_units/sec_per_Myr's own magnitude).
    sec_per_year = sec_per_Myr / 1.0e6
    print("cooldown elapsed physical time: %.6e yr" % (data0["time"][-1] / sec_per_year))

    # "fraction of H nuclei locked in H2" -- 0 to 1, comparable across
    # codes. NOT H2I_density / density (total_gas_density): that's H2
    # mass as a fraction of *all* mass including He, which tops out
    # around the H mass fraction (~0.76), not 1 -- not comparable to
    # dengo's nuclei-based fraction. HI/HII/HM/H2I/H2II densities here
    # are each already "mass of hydrogen nuclei in this state" (an H2
    # molecule's mass very nearly equals 2*m_H, so H2I_density directly
    # equals the mass of the H nuclei within it, no factor of 2 needed).
    total_H_density = (
        data["HI_density"] + data["HII_density"] + data["HM_density"]
        + data["H2I_density"] + data["H2II_density"]
    )
    total_H_density0 = (
        data0["HI_density"] + data0["HII_density"] + data0["HM_density"]
        + data0["H2I_density"] + data0["H2II_density"]
    )

    out = {
        "n": (data["density"] / mass_hydrogen_cgs).tolist(),
        "T": data["temperature"].tolist(),
        "H2_frac": (data["H2I_density"] / total_H_density).tolist(),
        "cooldown_n": (data0["density"] / mass_hydrogen_cgs).tolist(),
        "cooldown_T": data0["temperature"].tolist(),
        "cooldown_H2_frac": (data0["H2I_density"] / total_H_density0).tolist(),
        "cooldown_time_yr": (data0["time"] / sec_per_year).tolist(),
        "cooldown_steps": len(data0["time"]),
        "freefall_steps": len(data["time"]),
        "t_cooldown": t1 - t0,
        "t_freefall": t2 - t1,
        "t_total": t2 - t0,
        "wall_per_step": data["wall_per_step"].tolist(),
        "dt_per_step": data["dt_per_step"].tolist(),
    }
    with open(os.path.join(HERE, "grackle_result.json"), "w") as f:
        json.dump(out, f)
    print("Wrote", os.path.join(HERE, "grackle_result.json"))


if __name__ == "__main__":
    main()
