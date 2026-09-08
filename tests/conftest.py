"""Shared fixtures for the dengo test suite.

Building a solver means generating C++ and compiling it, which is slow
(several seconds), so the networks below are built once per test session
(`scope="session"`) and reused across every test that needs them.
"""
import numpy as np
import pytest

from dengo.chemical_network import ChemicalNetwork
import dengo.primordial_rates as primordial_rates
import dengo.primordial_cooling as primordial_cooling  # noqa: F401 -- registers cooling actions
import dengo.solver_build as solver_build

primordial_rates.setup_primordial()


PRIMORDIAL_SPECIES = [
    "H_1", "H_2", "He_1", "He_2", "He_3", "H_m0", "H2_1", "H2_2", "de", "ge",
]
PRIMORDIAL_COOLING = [
    "cie_cooling", "gloverabel08", "h2formation", "h2formation_extra",
    "reHII", "reHeII1", "reHeII2", "reHeIII", "brem", "compton",
    "ceHI", "ceHeI", "ceHeII", "ciHI", "ciHeI", "ciHeII", "ciHeIS",
]
PRIMORDIAL_REACTIONS = [
    "k01", "k02", "k03", "k04", "k05", "k06", "k07", "k08", "k09", "k10",
    "k11", "k12", "k13", "k14", "k15", "k16", "k17", "k18", "k19",
    "k21", "k22", "k23",
]


def make_primordial_network(T_bounds=(1e1, 1e8)):
    """A fresh ChemicalNetwork for the full H/He/H2 primordial chemistry."""
    network = ChemicalNetwork()
    network.add_collection(
        species_names=PRIMORDIAL_SPECIES,
        cooling_names=PRIMORDIAL_COOLING,
        reaction_names=PRIMORDIAL_REACTIONS,
    )
    network.init_temperature(T_bounds)
    return network


def make_hydrogen_network(T_bounds=(1e2, 1e5)):
    """A minimal 2-reaction H/H+/e- network (no cooling, no H2) -- the
    simplest possible case, used for fast, low-machinery tests."""
    network = ChemicalNetwork()
    network.add_collection(
        species_names=["H_1", "H_2", "de", "ge"],
        cooling_names=[],
        reaction_names=["k01", "k02"],
    )
    network.init_temperature(T_bounds)
    return network


@pytest.fixture(scope="session")
def hydrogen_solver(tmp_path_factory):
    network = make_hydrogen_network()
    outdir = tmp_path_factory.mktemp("hydrogen_solver")
    network.write_cython_solver("test_hydrogen", output_dir=str(outdir))
    mod = solver_build.build_solver(str(outdir), "test_hydrogen")
    return network, mod


@pytest.fixture(scope="session")
def primordial_solver(tmp_path_factory):
    network = make_primordial_network()
    outdir = tmp_path_factory.mktemp("primordial_solver")
    network.write_cython_solver("test_primordial", output_dir=str(outdir))
    mod = solver_build.build_solver(str(outdir), "test_primordial")
    return network, mod


def primordial_ics(nH=1e4, x_ion=1e-4, x_H2=1e-6, T_guess=1.0e12):
    """A typical low-density, mostly-neutral set of initial abundances
    (number densities, cm^-3) for the primordial network."""
    return {
        "H_1": np.array([nH * 0.76]),
        "H_2": np.array([nH * x_ion]),
        "He_1": np.array([nH * 0.24 / 4.0]),
        "He_2": np.array([0.0]),
        "He_3": np.array([0.0]),
        "H_m0": np.array([1.0e-12]),
        "H2_1": np.array([nH * x_H2]),
        "H2_2": np.array([1.0e-12]),
        "de": np.array([nH * x_ion]),
        "ge": np.array([T_guess]),
    }
