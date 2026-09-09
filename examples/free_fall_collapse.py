"""
A free-fall collapse of primordial (metal-free) gas, from a diffuse cloud
(n ~ 1 cm^-3) up through the target regime for this project: n ~ 1e15
amu/cc, T ~ 1500-2500 K -- the density/temperature range where H2
formation heating (4.48 eV/molecule) and its later optical-depth
suppression regulate fragmentation in the first stars. This is a 0-D
"single zone" model, not a hydro simulation -- it's the standard test
problem for validating a chemistry/cooling solver across many decades of
density (see NOTES.md).

This is *unimpeded* free-fall (gas in gravitational free-fall the whole
time, never pressure-supported) rather than the pressure-retarded
"modified free-fall" scheme of Omukai et al. (2005) that an earlier
version of this script used: that scheme estimates a "force factor" from
the effective adiabatic index dlnP/dlnrho and uses it to slow the
collapse as the gas heats. The reference implementation this was adapted
from (examples/test/evolve_free_fall.py, removed in this modernization
pass) computed that force factor but then hardcoded `include_pressure =
False`, so it was never actually exercised or validated, and it turned
out to be internally inconsistent (the collapse-rate formula picks up a
sqrt(1 - force_factor) suppression that the compressional-heating formula
does not, so density and temperature evolve on inconsistent tracks the
moment force_factor departs from 0). Plain free-fall avoids relying on
that unvalidated piece while still being the right test for what this
pass cares about: solving the chemistry/cooling correctly across ~15
decades of density.

Run with:
    uv run python examples/free_fall_collapse.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from primordial_network import build_network, build_solver

KB = 1.3806504e-16       # erg/K
MH = 1.67e-24            # g (matches dengo.chemistry_constants.mh)
G_GRAV = 6.674e-8        # cm^3 g^-1 s^-2

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "_free_fall_collapse_build")


def total_number_density(state, species_names):
    return sum(state[name][0] for name in species_names if name not in ("ge", "de"))


def thermodynamic_gamma(state, species_names):
    """The gas's adiabatic index Gamma = 1 + n/sum(n_i/(gamma_i-1)), from
    composition alone. H2 is taken at a fixed gamma=7/5 (its low-T
    rotational value); this is the same simplification the reference
    implementation this is adapted from used (it computed the fully
    T-interpolated gammaH2(T) and then discarded it in favor of the
    constant 7/5 anyway)."""
    n_total = sum(state[name][0] for name in species_names if name != "ge")
    inv_gm1_sum = 0.0
    for name in species_names:
        if name == "ge":
            continue
        gamma_i = 7.0 / 5.0 if name in ("H2_1", "H2_2") else 5.0 / 3.0
        inv_gm1_sum += state[name][0] / (gamma_i - 1.0)
    return n_total / inv_gm1_sum + 1.0


def initial_conditions(nH=1.0):
    ge_guess = 1.5 * KB * 300.0 / MH  # ~300 K seed
    # Relic ionization fraction left over from cosmological recombination
    # (e.g. Anninos & Norman 1996; Abel et al. 1997): ~2e-4, not
    # arbitrarily tiny -- these free electrons are the catalyst for H2
    # formation via H + e- -> H- -> H2 (k07/k08), so setting this too low
    # starves both that channel and the electron-collision cooling terms.
    x_e = 2.0e-4
    return {
        "H_1": np.array([nH * 0.76]),
        "H_2": np.array([nH * x_e]),
        "He_1": np.array([nH * 0.24 / 4.0]),
        "He_2": np.array([0.0]),
        "He_3": np.array([0.0]),
        "H_m0": np.array([1.0e-11 * nH]),
        "H2_1": np.array([nH * 2.0e-6]),
        "H2_2": np.array([1.0e-11 * nH]),
        "de": np.array([nH * x_e]),
        "ge": np.array([ge_guess]),
    }


def main(n_target=3.0e15, safety_factor=1.0e-2, max_steps=20000):
    network = build_network()
    species_names = sorted(s.name for s in network.required_species)
    mod = build_solver(network)

    state = initial_conditions()
    t_total = 0.0

    history = {name: [] for name in species_names}
    history.update(t=[], n=[], T=[])

    n_current = total_number_density(state, species_names)
    step = 0
    # One persistent handle for the whole collapse, instead of a fresh
    # run_primordial() (setup_data + table-read + free_data) every one of
    # these ~2000 steps -- see NOTES.md, that fixed per-call overhead can
    # dominate a driver that interleaves many small solver calls with
    # externally-updated state, which is exactly this loop's pattern (and
    # a real hydro code's per-timestep chemistry coupling too).
    with mod.Solver(1) as solver:
        while n_current < n_target and step < max_steps:
            # standard free-fall time, t_ff = sqrt(3*pi / (32*G*rho))
            rho = n_current * MH
            t_ff = np.sqrt(3.0 * np.pi / (32.0 * G_GRAV * rho))
            dt = safety_factor * t_ff

            # analytic free-fall solution advanced by dt: rho^-1/2
            # decreases linearly in time (Larson 1969 / standard free-fall
            # collapse), at the rate that makes rho diverge at exactly
            # rho's own t_ff.
            rho_new = (rho ** -0.5 - np.sqrt(32.0 * G_GRAV / (3.0 * np.pi)) * dt) ** -2.0
            density_ratio = rho_new / rho

            for name in species_names:
                if name != "ge":
                    state[name] = state[name] * density_ratio
            # adiabatic compressional heating: d(ge)/ge = (Gamma-1) d(rho)/rho
            gamma_ad = thermodynamic_gamma(state, species_names)
            state["ge"] = state["ge"] * (1.0 + (gamma_ad - 1.0) * (density_ratio - 1.0))

            final, _ = solver.step(state, dtf=dt, niter=200, intermediate=False)
            if not final["converged"]:
                print("Step %d: chemistry solve did not converge, stopping." % step)
                break
            for name in species_names:
                state[name] = final[name]

            t_total += dt
            n_current = total_number_density(state, species_names)

            history["t"].append(t_total)
            history["n"].append(n_current)
            history["T"].append(final["T"][0])
            for name in species_names:
                history[name].append(state[name][0])

            if step % 200 == 0:
                print("step %5d: n = %10.3e cm^-3, T = %8.1f K"
                      % (step, n_current, final["T"][0]))
            step += 1

    for key in history:
        history[key] = np.asarray(history[key])

    print("Stopped after %d steps at n = %0.3e cm^-3, t = %0.3e s" % (step, n_current, t_total))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, (ax_T, ax_h2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax_T.loglog(history["n"], history["T"])
    ax_T.axhspan(1500, 2500, color="orange", alpha=0.2, label="target 1500-2500 K")
    ax_T.set_ylabel("Temperature (K)")
    ax_T.legend()

    total_H = history["H_1"] + history["H_2"] + history["H_m0"] + 2 * history["H2_1"] + 2 * history["H2_2"]
    ax_h2.loglog(history["n"], history["H2_1"] / total_H)
    ax_h2.set_ylabel("H2 / H_tot")
    ax_h2.set_xlabel(r"total number density (cm$^{-3}$)")
    fig.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "free_fall_collapse.png")
    fig.savefig(out_png, dpi=150)
    print("Wrote", out_png)


if __name__ == "__main__":
    main()
