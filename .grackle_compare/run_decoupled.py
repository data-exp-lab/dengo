"""
Remove the collapse-model confound entirely: drive BOTH dengo's and
Grackle's chemistry solvers through the *exact same* prescribed
compression (plain, deterministic free-fall -- no history-dependent
force-factor estimator on either side), computed once and applied
identically to both. Whatever difference remains between the two
resulting T(n)/H2(n) curves is then attributable to the chemistry/
cooling solve itself (rate coefficients, integration accuracy), not to
different collapse dynamics -- unlike the run_dengo.py/run_grackle.py
comparison, where each code follows its own (dengo: plain, Grackle:
pressure-retarded) free-fall model.

This still is not an absolute ground truth (neither curve is a known-
correct answer) -- it isolates *which* differences are chemistry-
attributable, and the two convergence checks (dengo_convergence.py-style
sweep over safety_factor for each code, done separately) tell us how much
to trust each code's own answer on its own terms.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from primordial_network_helpers import build_network, build_solver  # noqa: E402
from run_grackle import build_chemistry  # noqa: E402

from gracklepy import setup_fluid_container  # noqa: E402
from gracklepy.utilities.physical_constants import mass_hydrogen_cgs  # noqa: E402

KB = 1.3806504e-16
MH = 1.67e-24
G_GRAV = 6.674e-8
SEC_PER_YEAR = 3.15e7

HERE = os.path.dirname(os.path.abspath(__file__))


def thermodynamic_gamma(state_scalar, species_names):
    n_total = sum(state_scalar[name] for name in species_names if name != "ge")
    inv_gm1_sum = 0.0
    for name in species_names:
        if name == "ge":
            continue
        gamma_i = 7.0 / 5.0 if name in ("H2_1", "H2_2") else 5.0 / 3.0
        inv_gm1_sum += state_scalar[name] / (gamma_i - 1.0)
    return n_total / inv_gm1_sum + 1.0


def h2_nuclei_fraction(state_scalar):
    total_H = (
        state_scalar["H_1"] + state_scalar["H_2"] + state_scalar["H_m0"]
        + 2.0 * (state_scalar["H2_1"] + state_scalar["H2_2"])
    )
    return 2.0 * state_scalar["H2_1"] / total_H


def ionized_ics(nH):
    T = 50000.0
    ge = 1.5 * KB * T / MH
    return {
        "H_1": nH * 0.76 * 1e-10,
        "H_2": nH * 0.76,
        "He_1": nH * 0.24 / 4.0 * 1e-10,
        "He_2": nH * 0.24 / 4.0 * 1e-10,
        "He_3": nH * 0.24 / 4.0,
        "H_m0": nH * 1e-20,
        "H2_1": nH * 1e-20,
        "H2_2": nH * 1e-20,
        "de": nH * 0.76 + 2 * nH * 0.24 / 4.0,
        "ge": ge,
    }


def main(n_target=3.0e15, safety_cool=0.1, safety_ff=0.01, max_steps=200000, reltol=1.0e-5):
    network = build_network()
    species_names = sorted(s.name for s in network.required_species)
    mod = build_solver(network)
    solver = mod.Solver(1)

    my_chemistry = build_chemistry()
    fc = setup_fluid_container(
        my_chemistry, density=1e-1 * mass_hydrogen_cgs, temperature=50000.0,
        state="ionized", converge=False,
    )

    dengo_state = {k: np.array([v]) for k, v in ionized_ics(0.1).items()}

    t0 = time.perf_counter()

    # --- phase 1: cool at constant density, dengo's own adaptive dt,
    # applied to BOTH codes (same dt sequence, since it only depends on
    # dengo's cooling-time estimate -- this phase doesn't involve
    # density changes, so there's no collapse-model confound here at all,
    # just establishing comparable starting abundances for phase 2). ---
    step = 0
    T = 50000.0
    while T > 300.0 and step < max_steps:
        state_scalar = {name: dengo_state[name][0] for name in species_names}
        rhs = solver.evaluate_rhs(state_scalar)
        cooling_time = abs(state_scalar["ge"] / rhs["ge"]) if rhs["ge"] != 0 else 3.15e16
        dt = safety_cool * cooling_time

        final, _ = solver.step(dengo_state, dtf=dt, niter=200, intermediate=False, reltol=reltol)
        if not final["converged"]:
            print("dengo cooldown: step %d did not converge" % step)
            break
        for name in species_names:
            dengo_state[name] = final[name]
        T = final["T"][0]

        # dt above is in raw CGS seconds (dengo works in cgs throughout);
        # solve_chemistry() expects dt in grackle's *code* time units
        # (my_chemistry.time_units == sec_per_Myr here) -- passing raw
        # seconds silently evolves grackle for ~1/3e13 of the intended
        # physical interval (this was a real bug caught by grackle's
        # temperature staying pinned near its 1 K floor throughout).
        fc.solve_chemistry(dt / my_chemistry.time_units)
        fc.calculate_temperature()
        step += 1
    t1 = time.perf_counter()
    print("cooldown: %d steps, %.3fs -- dengo T=%.2f  grackle T=%.2f" %
          (step, t1 - t0, T, float(fc["temperature"][0])))

    # --- phase 2: SAME plain free-fall prescription drives both codes ---
    n_current = sum(dengo_state[name][0] for name in species_names if name not in ("ge", "de"))
    history = {"n": [], "dengo_T": [], "grackle_T": [], "dengo_H2_frac": [], "grackle_H2_frac": [],
               "dengo_wall": [], "grackle_wall": []}
    step = 0
    while n_current < n_target and step < max_steps:
        rho = n_current * MH
        t_ff = np.sqrt(3.0 * np.pi / (32.0 * G_GRAV * rho))
        dt = safety_ff * t_ff
        rho_new = (rho ** -0.5 - np.sqrt(32.0 * G_GRAV / (3.0 * np.pi)) * dt) ** -2.0
        density_ratio = rho_new / rho

        # -- dengo side --
        for name in species_names:
            if name != "ge":
                dengo_state[name] = dengo_state[name] * density_ratio
        state_scalar = {name: dengo_state[name][0] for name in species_names}
        gamma_ad = thermodynamic_gamma(state_scalar, species_names)
        dengo_state["ge"] = dengo_state["ge"] * (1.0 + (gamma_ad - 1.0) * (density_ratio - 1.0))
        t_before = time.perf_counter()
        final, _ = solver.step(dengo_state, dtf=dt, niter=200, intermediate=False, reltol=reltol)
        dengo_wall_this_step = time.perf_counter() - t_before
        if not final["converged"]:
            print("dengo freefall: step %d did not converge" % step)
            break
        for name in species_names:
            dengo_state[name] = final[name]

        # -- grackle side: same density_ratio, same dt, same adiabatic
        # heating formula (Gamma from grackle's own chemistry_data, not
        # dengo's -- the one place the two codes are allowed to differ,
        # since Gamma is itself a chemistry-composition-dependent
        # quantity each code computes its own way) --
        for field in fc.density_fields:
            fc[field] *= density_ratio
        fc["internal_energy"][0] += (my_chemistry.Gamma - 1.0) * fc["internal_energy"][0] * (density_ratio - 1.0)
        t_before = time.perf_counter()
        fc.solve_chemistry(dt / my_chemistry.time_units)  # dt: cgs seconds -> code units
        grackle_wall_this_step = time.perf_counter() - t_before
        fc.calculate_temperature()

        n_current = sum(dengo_state[name][0] for name in species_names if name not in ("ge", "de"))
        total_H_g = (
            fc["HI_density"][0] + fc["HII_density"][0] + fc["HM_density"][0]
            + fc["H2I_density"][0] + fc["H2II_density"][0]
        )
        history["n"].append(n_current)
        history["dengo_T"].append(final["T"][0])
        history["grackle_T"].append(float(fc["temperature"][0]))
        state_scalar = {name: dengo_state[name][0] for name in species_names}
        history["dengo_H2_frac"].append(h2_nuclei_fraction(state_scalar))
        history["grackle_H2_frac"].append(float(fc["H2I_density"][0] / total_H_g))
        history["dengo_wall"].append(dengo_wall_this_step)
        history["grackle_wall"].append(grackle_wall_this_step)
        if step % 500 == 0:
            print("step %5d: n=%10.3e  dengo_T=%8.1f  grackle_T=%8.1f" %
                  (step, n_current, final["T"][0], float(fc["temperature"][0])))
        step += 1
    t2 = time.perf_counter()
    solver.close()

    print("free-fall: %d steps, %.3fs" % (step, t2 - t1))
    print("final: dengo T=%.2f H2=%.5f | grackle T=%.2f H2=%.5f" % (
        history["dengo_T"][-1], history["dengo_H2_frac"][-1],
        history["grackle_T"][-1], history["grackle_H2_frac"][-1],
    ))

    with open(os.path.join(HERE, "decoupled_result.json"), "w") as f:
        json.dump(history, f)
    print("Wrote", os.path.join(HERE, "decoupled_result.json"))


if __name__ == "__main__":
    main()
