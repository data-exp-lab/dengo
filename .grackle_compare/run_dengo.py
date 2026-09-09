"""
Run the same two-phase test problem through dengo that run_grackle.py runs
through Grackle: start hot & ionized at n=0.1 cm^-3, T=50000 K, cool at
constant density down to 300 K, then free-fall collapse up to the same
target density, same safety factors.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))

from primordial_network import build_network, build_solver  # noqa: E402

KB = 1.3806504e-16
MH = 1.67e-24
G_GRAV = 6.674e-8

HERE = os.path.dirname(os.path.abspath(__file__))


def thermodynamic_gamma(state, species_names):
    n_total = sum(state[name] for name in species_names if name != "ge")
    inv_gm1_sum = 0.0
    for name in species_names:
        if name == "ge":
            continue
        gamma_i = 7.0 / 5.0 if name in ("H2_1", "H2_2") else 5.0 / 3.0
        inv_gm1_sum += state[name] / (gamma_i - 1.0)
    return n_total / inv_gm1_sum + 1.0


def h2_nuclei_fraction(state_scalar):
    """Fraction of H nuclei locked in H2 (0-1) -- computed from this
    state's OWN species densities, never from a value taken at any other
    time: total-H number density scales up with compression just like
    everything else, so normalizing every point in a history by one
    (e.g. the final) state's total-H is wrong by orders of magnitude at
    low density -- exactly the bug an earlier version of this script
    had (it computed total_H once, after the free-fall finished, and
    used it to normalize every earlier point too)."""
    total_H = (
        state_scalar["H_1"] + state_scalar["H_2"] + state_scalar["H_m0"]
        + 2.0 * (state_scalar["H2_1"] + state_scalar["H2_2"])
    )
    return 2.0 * state_scalar["H2_1"] / total_H


def ionized_ics(nH):
    # fully ionized H/He, matching gracklepy's setup_fluid_container(state="ionized")
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

    nH0 = 0.1
    state = {k: np.array([v]) for k, v in ionized_ics(nH0).items()}

    t0 = time.perf_counter()

    # One persistent handle for the whole run (both phases: dims=1
    # throughout) -- setup_data/read_tables happens once here, not on
    # every one of the ~2000 steps below. See NOTES.md: with the old
    # one-shot run_primordial()/evaluate_rhs() API, most of the reported
    # per-step cost was this fixed setup/teardown overhead, not the
    # actual chemistry solve.
    solver = mod.Solver(1)
    try:
        # --- phase 1: cool at constant density from 50000 K to 300 K ---
        # matches gracklepy.utilities.evolve.evolve_constant_density: dt is
        # set each step from the current cooling time (ge/|dge/dt|), not fixed.
        step = 0
        T = 50000.0
        cooldown_steps = 0
        cooldown_time = 0.0
        cooldown_history = {"n": [], "T": [], "H2_frac": [], "t": []}
        while T > 300.0 and step < max_steps:
            state_scalar = {name: state[name][0] for name in species_names}
            rhs = solver.evaluate_rhs(state_scalar)
            cooling_time = abs(state_scalar["ge"] / rhs["ge"]) if rhs["ge"] != 0 else 3.15e16
            dt = safety_cool * cooling_time

            final, _ = solver.step(state, dtf=dt, niter=200, intermediate=False, reltol=reltol)
            if not final["converged"]:
                print("cooldown: step %d did not converge" % step)
                break
            for name in species_names:
                state[name] = final[name]
            T = final["T"][0]
            cooldown_time += dt

            state_scalar = {name: state[name][0] for name in species_names}
            n_now = sum(state_scalar[name] for name in species_names if name not in ("ge", "de"))
            cooldown_history["n"].append(n_now)
            cooldown_history["T"].append(T)
            cooldown_history["H2_frac"].append(h2_nuclei_fraction(state_scalar))
            cooldown_history["t"].append(cooldown_time)
            step += 1
            cooldown_steps += 1
            if step % 200 == 0:
                print("cooldown step %5d: T = %8.2f K" % (step, T))
        t1 = time.perf_counter()
        print("cooldown phase: %d steps, %.3f s, final T=%.2f" % (cooldown_steps, t1 - t0, T))
        print("cooldown elapsed physical time: %.6e yr" % (cooldown_time / 3.15e7))
        print("H2 nuclei fraction right after cooldown: %.6e" % cooldown_history["H2_frac"][-1])

        # --- phase 2: free-fall collapse ---
        n_current = sum(state[name][0] for name in species_names if name not in ("ge", "de"))
        history = {"n": [], "T": [], "H2_frac": [], "wall_per_step": [], "dt_per_step": []}
        step = 0
        while n_current < n_target and step < max_steps:
            rho = n_current * MH
            t_ff = np.sqrt(3.0 * np.pi / (32.0 * G_GRAV * rho))
            dt = safety_ff * t_ff
            rho_new = (rho ** -0.5 - np.sqrt(32.0 * G_GRAV / (3.0 * np.pi)) * dt) ** -2.0
            density_ratio = rho_new / rho

            for name in species_names:
                if name != "ge":
                    state[name] = state[name] * density_ratio
            gamma_ad = thermodynamic_gamma({n: state[n][0] for n in species_names}, species_names)
            state["ge"] = state["ge"] * (1.0 + (gamma_ad - 1.0) * (density_ratio - 1.0))

            t_before = time.perf_counter()
            final, _ = solver.step(state, dtf=dt, niter=200, intermediate=False, reltol=reltol)
            wall_this_step = time.perf_counter() - t_before
            if not final["converged"]:
                print("freefall: step %d did not converge" % step)
                break
            for name in species_names:
                state[name] = final[name]

            state_scalar = {name: state[name][0] for name in species_names}
            n_current = sum(state_scalar[name] for name in species_names if name not in ("ge", "de"))
            history["n"].append(n_current)
            history["T"].append(final["T"][0])
            history["H2_frac"].append(h2_nuclei_fraction(state_scalar))
            history["wall_per_step"].append(wall_this_step)
            history["dt_per_step"].append(dt)
            if step % 500 == 0:
                print("freefall step %5d: n=%10.3e cm^-3, T=%8.1f K" % (step, n_current, final["T"][0]))
            step += 1
        t2 = time.perf_counter()
    finally:
        solver.close()

    print("free-fall phase: %d steps, %.3f s" % (step, t2 - t1))
    print("total wall time (compute only): %.3f s" % (t2 - t0))

    out = {
        "n": history["n"],
        "T": history["T"],
        "H2_frac": history["H2_frac"],
        "cooldown_n": cooldown_history["n"],
        "cooldown_T": cooldown_history["T"],
        "cooldown_H2_frac": cooldown_history["H2_frac"],
        "cooldown_time_yr": [t / 3.15e7 for t in cooldown_history["t"]],
        "cooldown_steps": cooldown_steps,
        "freefall_steps": step,
        "t_cooldown": t1 - t0,
        "t_freefall": t2 - t1,
        "t_total": t2 - t0,
        "reltol": reltol,
        "wall_per_step": history["wall_per_step"],
        "dt_per_step": history["dt_per_step"],
    }
    with open(os.path.join(HERE, "dengo_result.json"), "w") as f:
        json.dump(out, f)
    print("Wrote", os.path.join(HERE, "dengo_result.json"))


if __name__ == "__main__":
    reltol = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0e-5
    main(reltol=reltol)
