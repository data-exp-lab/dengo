"""
Define the primordial (metal-free) H/He/H2 chemistry network, generate and
compile its solver, and integrate a constant-density cooling box.

Run with:
    uv run python examples/primordial_network.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import dengo.primordial_cooling  # noqa: F401 -- registers cooling actions
import dengo.primordial_rates as primordial_rates
import dengo.solver_build as solver_build
from dengo.chemical_network import ChemicalNetwork

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "_primordial_network_build")

SPECIES = ["H_1", "H_2", "He_1", "He_2", "He_3", "H_m0",
           "H2_1", "H2_2", "de", "ge"]
COOLING = ["cie_cooling", "gloverabel08", "h2formation", "h2formation_extra",
           "reHII", "reHeII1", "reHeII2", "reHeIII", "brem", "compton",
           "ceHI", "ceHeI", "ceHeII", "ciHI", "ciHeI", "ciHeII", "ciHeIS"]
REACTIONS = ["k01", "k02", "k03", "k04", "k05", "k06", "k07", "k08", "k09",
             "k10", "k11", "k12", "k13", "k14", "k15", "k16", "k17", "k18",
             "k19", "k21", "k22", "k23"]


def build_network():
    primordial_rates.setup_primordial()
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


def initial_conditions(nH=1.0e4, T=8000.0):
    """A diffuse, mostly-neutral gas at density nH (cm^-3) with a gas
    energy corresponding roughly to temperature T (before the solver's own
    Newton iteration refines it self-consistently)."""
    kb = 1.3806504e-16
    mh = 1.67e-24
    ge_guess = 1.5 * kb * T / mh
    return {
        "H_1": np.array([nH * 0.76]),
        "H_2": np.array([nH * 1e-4]),
        "He_1": np.array([nH * 0.24 / 4.0]),
        "He_2": np.array([0.0]),
        "He_3": np.array([0.0]),
        "H_m0": np.array([1.0e-12]),
        "H2_1": np.array([nH * 1e-6]),
        "H2_2": np.array([1.0e-12]),
        "de": np.array([nH * 1e-4]),
        "ge": np.array([ge_guess]),
    }


def main():
    network = build_network()
    mod = build_solver(network)

    ics = initial_conditions()
    dtf = 3.15e15  # ~100 Myr
    final, traj = mod.run_primordial(ics, dtf=dtf, niter=20000)

    print("Converged:", final["converged"])
    print("Final T = %0.1f K" % final["T"][0])
    for name in ("H_1", "H_2", "H2_1", "He_1", "He_2", "de"):
        print("  %-6s = %0.4e cm^-3" % (name, final[name][0]))

    fig, (ax_T, ax_frac) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    t = traj["t"]
    ax_T.loglog(t, traj["T"][0])
    ax_T.set_ylabel("Temperature (K)")

    total_H = ics["H_1"][0] + ics["H_2"][0] + ics["H_m0"][0] + 2 * ics["H2_1"][0] + 2 * ics["H2_2"][0]
    ax_frac.loglog(t, traj["H2_1"][0] / total_H, label="H2 / H_tot")
    ax_frac.loglog(t, traj["de"][0] / total_H, label="e- / H_tot")
    ax_frac.set_ylabel("Fraction")
    ax_frac.set_xlabel("Time (s)")
    ax_frac.legend()

    fig.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "primordial_network.png")
    fig.savefig(out_png, dpi=150)
    print("Wrote", out_png)


if __name__ == "__main__":
    main()
