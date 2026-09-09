"""Validate dengo.grackle_compat against the REAL gracklepy, side by
side, on the identical test problem: does code written against the real
API produce physically-consistent results when only the import is
swapped? This is the actual "is it a drop-in" test -- everything else
in dengo.grackle_compat's own docstring/design is necessary but not
sufficient without this.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import gracklepy
from gracklepy.utilities.physical_constants import mass_hydrogen_cgs, sec_per_Myr, cm_per_mpc

import dengo.grackle_compat as compat

GRACKLE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".grackle",
    "grackle_data_files", "input",
)


def build_real():
    my_chemistry = gracklepy.chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 1
    my_chemistry.primordial_chemistry = 2
    my_chemistry.metal_cooling = 0
    my_chemistry.dust_chemistry = 0
    my_chemistry.UVbackground = 0
    my_chemistry.self_shielding_method = 0
    my_chemistry.H2_self_shielding = 0
    my_chemistry.CaseBRecombination = 1
    my_chemistry.cie_cooling = 1
    my_chemistry.h2_optical_depth_approximation = 1
    my_chemistry.three_body_rate = 4
    my_chemistry.grackle_data_file = os.path.join(GRACKLE_DATA_DIR, "CloudyData_noUVB.h5")
    my_chemistry.comoving_coordinates = 0
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0
    my_chemistry.density_units = mass_hydrogen_cgs
    my_chemistry.length_units = cm_per_mpc
    my_chemistry.time_units = sec_per_Myr
    my_chemistry.set_velocity_units()
    my_chemistry.initialize()
    return my_chemistry


def build_shim():
    my_chemistry = compat.chemistry_data()
    my_chemistry.CaseBRecombination = 1
    my_chemistry.cie_cooling = 1
    my_chemistry.h2_optical_depth_approximation = 1
    my_chemistry.three_body_rate = 4
    my_chemistry.comoving_coordinates = 0
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0
    my_chemistry.density_units = mass_hydrogen_cgs
    my_chemistry.length_units = cm_per_mpc
    my_chemistry.time_units = sec_per_Myr
    my_chemistry.set_velocity_units()
    my_chemistry.initialize()
    return my_chemistry


def ionized_fc(mod, my_chemistry, nH=0.1, T=50000.0):
    fc = mod.FluidContainer(my_chemistry, 1)
    fc["HI_density"][:] = nH * 0.76 * 1e-10
    fc["HII_density"][:] = nH * 0.76
    fc["HeI_density"][:] = nH * 0.24 / 4.0 * 1e-10
    fc["HeII_density"][:] = nH * 0.24 / 4.0 * 1e-10
    fc["HeIII_density"][:] = nH * 0.24 / 4.0
    fc["HM_density"][:] = nH * 1e-20
    fc["H2I_density"][:] = nH * 1e-20
    fc["H2II_density"][:] = nH * 1e-20
    fc["e_density"][:] = nH * 0.76 + 2 * nH * 0.24 / 4.0
    fc["density"][:] = nH
    ge_cgs = 1.5 * 1.3806504e-16 * T / mass_hydrogen_cgs
    fc["internal_energy"][:] = ge_cgs / my_chemistry.velocity_units ** 2
    return fc


def main():
    cd_real = build_real()
    cd_shim = build_shim()
    fc_real = ionized_fc(gracklepy, cd_real)
    fc_shim = ionized_fc(compat, cd_shim)

    print("%6s  %12s %12s %10s  %12s %12s %10s" %
          ("step", "T_real", "T_shim", "reldiff%", "gamma_real", "gamma_shim", "reldiff%"))
    dt_code = 0.1 * sec_per_Myr / cd_real.time_units  # ~0.1 Myr in code units, both sides
    for step in range(30):
        fc_real.calculate_temperature()
        fc_shim.calculate_temperature()
        fc_real.calculate_gamma()
        fc_shim.calculate_gamma()
        Tr, Ts = float(fc_real["temperature"][0]), float(fc_shim["temperature"][0])
        gr, gs = float(fc_real["gamma"][0]), float(fc_shim["gamma"][0])
        print("%6d  %12.3f %12.3f %10.3f  %12.6f %12.6f %10.3f" %
              (step, Tr, Ts, 100 * (Ts - Tr) / Tr, gr, gs, 100 * (gs - gr) / gr))
        fc_real.solve_chemistry(dt_code)
        fc_shim.solve_chemistry(dt_code)

    fc_real.calculate_temperature()
    fc_shim.calculate_temperature()
    print("\nfinal: T_real=%.2f T_shim=%.2f" % (fc_real["temperature"][0], fc_shim["temperature"][0]))
    h2_real = fc_real["H2I_density"][0] / (fc_real["HI_density"][0] + fc_real["HII_density"][0]
                                            + fc_real["HM_density"][0] + fc_real["H2I_density"][0]
                                            + fc_real["H2II_density"][0])
    h2_shim = fc_shim["H2I_density"][0] / (fc_shim["HI_density"][0] + fc_shim["HII_density"][0]
                                            + fc_shim["HM_density"][0] + fc_shim["H2I_density"][0]
                                            + fc_shim["H2II_density"][0])
    print("final: H2_real=%.6e H2_shim=%.6e" % (h2_real, h2_shim))


if __name__ == "__main__":
    main()
