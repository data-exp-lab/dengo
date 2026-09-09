"""Tests for dengo.grackle_compat -- the Grackle-API-compatible shim
over dengo's primordial solver (see NOTES.md for the scoping/difficulty
assessment, .grackle_compare/validate_grackle_compat.py for validation
against the real gracklepy, which isn't a dependency of this repo's own
test suite).
"""
import numpy as np
import pytest

from dengo.grackle_compat import (
    FluidContainer,
    GrackleCompatError,
    calculate_cooling_time,
    calculate_gamma,
    calculate_pressure,
    calculate_temperature,
    chemistry_data,
)

DENSITY_UNITS = 1.67e-24  # mass_hydrogen_cgs
LENGTH_UNITS = 3.0857e24  # ~1 Mpc
TIME_UNITS = 3.15e13      # ~1 Myr


def _make_chemistry():
    cd = chemistry_data()
    cd.three_body_rate = 4
    cd.cie_cooling = 1
    cd.h2_optical_depth_approximation = 1
    cd.density_units = DENSITY_UNITS
    cd.length_units = LENGTH_UNITS
    cd.time_units = TIME_UNITS
    cd.set_velocity_units()
    cd.initialize()
    return cd


def _ionized_fc(cd, nH=0.1, T=50000.0):
    fc = FluidContainer(cd, 1)
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
    ge_cgs = 1.5 * 1.3806504e-16 * T / DENSITY_UNITS
    fc["internal_energy"][:] = ge_cgs / cd.velocity_units ** 2
    return fc


def test_set_velocity_units_matches_grackle_formula():
    """velocity_units = length_units/time_units (/a_value if comoving) --
    see grackle_units.c's get_velocity_units(), copied in the module
    docstring/set_velocity_units()."""
    cd = chemistry_data()
    cd.length_units = LENGTH_UNITS
    cd.time_units = TIME_UNITS
    cd.set_velocity_units()
    assert cd.velocity_units == pytest.approx(LENGTH_UNITS / TIME_UNITS)

    cd.comoving_coordinates = 1
    cd.a_value = 2.0
    cd.set_velocity_units()
    assert cd.velocity_units == pytest.approx(LENGTH_UNITS / TIME_UNITS / 2.0)


@pytest.mark.parametrize("field,value", [
    ("metal_cooling", 1),
    ("dust_chemistry", 1),
    ("UVbackground", 1),
    ("primordial_chemistry", 1),
    ("primordial_chemistry", 3),
    ("self_shielding_method", 1),
    ("H2_self_shielding", 1),
    ("use_radiative_transfer", 1),
])
def test_unsupported_configs_rejected_loudly(field, value):
    """Anything outside primordial_chemistry=2/no-metals/no-UV/no-dust
    has to fail at initialize() time, not silently compute the wrong
    physics -- see NOTES.md's difficulty assessment for why this
    boundary is where it is."""
    cd = chemistry_data()
    setattr(cd, field, value)
    with pytest.raises(GrackleCompatError):
        cd.initialize()


def test_fluid_container_requires_initialized_chemistry():
    cd = chemistry_data()
    with pytest.raises(RuntimeError):
        FluidContainer(cd, 1)


def test_calculate_temperature_and_gamma_are_sane():
    cd = _make_chemistry()
    fc = _ionized_fc(cd)
    calculate_temperature(fc)
    calculate_gamma(fc)
    assert np.isfinite(fc["temperature"][0])
    assert fc["temperature"][0] > 0
    # negligible H2 at these conditions -> mixture gamma ~= monatomic 5/3
    assert fc["gamma"][0] == pytest.approx(5.0 / 3.0, rel=1e-3)


def test_calculate_pressure_matches_ideal_gas_law():
    cd = _make_chemistry()
    fc = _ionized_fc(cd)
    calculate_temperature(fc)
    calculate_pressure(fc)

    kboltz = 1.380e-16
    mh = 1.67e-24
    n_tot = sum(fc[f][0] for f in (
        "HI_density", "HII_density", "HeI_density", "HeII_density",
        "HeIII_density", "HM_density", "H2I_density", "H2II_density",
        "e_density")) * cd.density_units / mh
    # (approximately -- density fields are stored per-species with
    # slightly different masses in dengo's own units; this is a
    # sanity-level check, not an exact one)
    P_expected_cgs = n_tot * kboltz * fc["temperature"][0]
    P_actual_cgs = fc["pressure"][0] * cd.density_units * cd.velocity_units ** 2
    assert P_actual_cgs == pytest.approx(P_expected_cgs, rel=0.1)


def test_calculate_cooling_time_is_positive_and_finite():
    cd = _make_chemistry()
    fc = _ionized_fc(cd)
    calculate_cooling_time(fc)
    assert np.isfinite(fc["cooling_time"][0])
    assert fc["cooling_time"][0] > 0


def test_solve_chemistry_cools_a_hot_ionized_gas():
    cd = _make_chemistry()
    fc = _ionized_fc(cd, T=50000.0)
    calculate_temperature(fc)
    T0 = fc["temperature"][0]

    dt_code = 0.1 * TIME_UNITS / cd.time_units  # ~0.1 Myr, in code units
    for _ in range(20):
        fc.solve_chemistry(dt_code)
    calculate_temperature(fc)
    T1 = fc["temperature"][0]

    assert T1 < T0  # hot ionized gas should cool, not heat, at constant density
    assert np.isfinite(T1)
    # total H nuclei conserved through the solve (mass-density fields,
    # so a straight sum, not the number-density-weighted one dengo uses
    # internally)
    total_H = (fc["HI_density"][0] + fc["HII_density"][0] + fc["HM_density"][0]
               + fc["H2I_density"][0] + fc["H2II_density"][0])
    nH0 = 0.1 * 0.76
    assert total_H == pytest.approx(nH0, rel=1e-3)


def test_calculate_dust_temperature_raises():
    cd = _make_chemistry()
    fc = _ionized_fc(cd)
    with pytest.raises(GrackleCompatError):
        fc.calculate_dust_temperature()
