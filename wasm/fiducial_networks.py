"""Three fiducial networks for the WASM demo site, deliberately spanning
a wide range of complexity dengo can generate a solver for -- from a
2-reaction toy up to the full primordial H/He/H2 network this project
targets -- to show the wasm codegen path (ChemicalNetwork.
write_wasm_solver(), src/dengo/templates/wasm_solver/) genuinely works
for more than just the one hand-picked network it was built against.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import dengo.primordial_cooling  # noqa: F401 -- registers cooling actions
import dengo.primordial_rates as primordial_rates
from dengo.chemical_network import ChemicalNetwork

_setup_done = False


def _ensure_setup():
    global _setup_done
    if not _setup_done:
        primordial_rates.setup_primordial()
        _setup_done = True


def build_primordial():
    """The flagship network: full H/H+/He/He+/He++/H-/H2/H2+/e-
    chemistry (dengo.primordial_network) -- the ~1e15 amu/cc,
    1500-2500K H2-formation-heating regime this project targets."""
    from dengo.primordial_network import build_network
    return build_network()


def build_primordial_atomic():
    """H/He ionization/recombination only -- no H2 at all (reactions
    k01-k06, none of the H2-chain k07+), a smaller contrast case: the
    same atomic cooling physics (recombination, collisional excitation/
    ionization, bremsstrahlung, Compton) without any H2 formation/
    dissociation chemistry."""
    _ensure_setup()
    network = ChemicalNetwork()
    network.add_collection(
        species_names=["H_1", "H_2", "He_1", "He_2", "He_3", "de", "ge"],
        cooling_names=["reHII", "reHeII1", "reHeII2", "reHeIII", "brem", "compton",
                       "ceHI", "ceHeI", "ceHeII", "ciHI", "ciHeI", "ciHeII", "ciHeIS"],
        reaction_names=["k01", "k02", "k03", "k04", "k05", "k06"],
    )
    network.init_temperature((1e1, 1e8))
    return network


def build_hydrogen_minimal():
    """The simplest possible network: H/H+/e- with just ionization/
    recombination (k01/k02), no cooling terms at all -- a fast, minimal
    demo (matches tests/conftest.py's make_hydrogen_network)."""
    _ensure_setup()
    network = ChemicalNetwork()
    network.add_collection(
        species_names=["H_1", "H_2", "de", "ge"],
        cooling_names=[],
        reaction_names=["k01", "k02"],
    )
    network.init_temperature((1e2, 1e5))
    return network


# name -> (display title, default_ics, build_fn). default_ics gives each
# species a starting number density (cm^-3) at nH=1 -- scaled by the
# widget's density slider -- plus a starting temperature (K); see
# index.html/app.js.
FIDUCIAL_NETWORKS = {
    "primordial": {
        "title": "Primordial H/He/H2",
        "build": build_primordial,
        "default_ics": {
            "H_1": 0.76, "H_2": 1e-4, "He_1": 0.06, "He_2": 0.0, "He_3": 0.0,
            "H_m0": 1e-12, "H2_1": 1e-6, "H2_2": 1e-12, "de": 1e-4,
        },
        "default_T": 1000.0,
    },
    "primordial_atomic": {
        "title": "Primordial H/He (no H2)",
        "build": build_primordial_atomic,
        "default_ics": {
            "H_1": 0.76, "H_2": 1e-4, "He_1": 0.06, "He_2": 0.0, "He_3": 0.0, "de": 1e-4,
        },
        "default_T": 1000.0,
    },
    "hydrogen_minimal": {
        "title": "Minimal H/H+/e-",
        "build": build_hydrogen_minimal,
        "default_ics": {"H_1": 0.999, "H_2": 0.001, "de": 0.001},
        "default_T": 8000.0,
    },
}
