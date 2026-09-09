"""
Define the primordial (metal-free) H/He/H2 chemistry network, generate and
compile its solver, and integrate a constant-density cooling box.

Run with:
    uv run python examples/primordial_network.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from dengo.primordial_network import build_network as _build_network
from dengo.primordial_network import build_solver as _build_solver

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "_primordial_network_build")


def build_network():
    return _build_network()


def build_solver(network):
    return _build_solver(network, OUTPUT_DIR)


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
