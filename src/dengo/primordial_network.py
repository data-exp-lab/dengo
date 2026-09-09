"""Canonical definition of dengo's primordial (metal-free) H/He/H2
chemistry network: the 9 species H, H+, He, He+, He++, H-, H2, H2+, e-
plus specific internal energy (ge) -- the same chemistry Grackle tracks
under primordial_chemistry=2, three_body_rate=4 (this network's default,
see ChemicalNetwork.threebody).

This is the one place the species/cooling/reaction list lives;
examples/primordial_network.py, .grackle_compare/, and
dengo.grackle_compat all build their solver from build_network()/
build_solver() here instead of each keeping their own copy.
"""
import os

import dengo.primordial_cooling  # noqa: F401 -- registers cooling actions
import dengo.primordial_rates as primordial_rates
import dengo.solver_build as solver_build
from dengo.chemical_network import ChemicalNetwork

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
    """A fresh ChemicalNetwork for the species/cooling/reactions above.
    Cheap (pure Python/sympy bookkeeping) -- the expensive part is
    build_solver(), below."""
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


def build_solver(network, output_dir):
    """Generate and compile the Cython/C++ solver for `network` into
    `output_dir` (created if missing), returning the compiled module."""
    os.makedirs(output_dir, exist_ok=True)
    network.write_cython_solver("primordial", output_dir=output_dir)
    return solver_build.build_solver(output_dir, "primordial")
