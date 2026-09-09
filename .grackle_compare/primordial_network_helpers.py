"""Build the primordial network/solver for use inside .grackle_compare's
own venv (Python 3.12, to match gracklepy's wheel) -- a separate build
directory from examples/_primordial_network_build, since a compiled
Cython extension is tied to the interpreter ABI it was built for and
that directory's .so is built for whatever Python the main project env
uses."""
import os

import dengo.primordial_cooling  # noqa: F401 -- registers cooling actions
import dengo.primordial_rates as primordial_rates
import dengo.solver_build as solver_build
from dengo.chemical_network import ChemicalNetwork

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dengo_build")

SPECIES = ["H_1", "H_2", "He_1", "He_2", "He_3", "H_m0",
           "H2_1", "H2_2", "de", "ge"]
COOLING = ["cie_cooling", "gloverabel08", "h2formation", "h2formation_extra",
           "reHII", "reHeII1", "reHeII2", "reHeIII", "brem", "compton",
           "ceHI", "ceHeI", "ceHeII", "ciHI", "ciHeI", "ciHeII", "ciHeIS"]
REACTIONS = ["k01", "k02", "k03", "k04", "k05", "k06", "k07", "k08", "k09",
             "k10", "k11", "k12", "k13", "k14", "k15", "k16", "k17", "k18",
             "k19", "k21", "k22", "k23"]

_setup_done = False


def build_network():
    global _setup_done
    if not _setup_done:
        primordial_rates.setup_primordial()
        _setup_done = True
    network = ChemicalNetwork()
    network.add_collection(
        species_names=SPECIES, cooling_names=COOLING, reaction_names=REACTIONS,
    )
    network.init_temperature((1e1, 1e8))
    return network


def build_solver(network):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    network.write_cython_solver("primordial", output_dir=OUTPUT_DIR)
    return solver_build.build_solver(OUTPUT_DIR, "primordial")
