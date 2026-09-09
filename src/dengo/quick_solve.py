"""A one-line convenience entry point for a quick single-zone chemistry
check, without building a `ChemicalNetwork`/`Solver` yourself.

    >>> import dengo
    >>> final = dengo.quick_solve(nH=1e4, T=8000.0, dtf=3.15e13)
    >>> final["T"], final["H2_1"] / final["H_1"]

Not for production/performance-sensitive use -- see `Solver`/
`step_inplace()` (`dengo.primordial_network.build_network`/
`build_solver`) for that, and `examples/primordial_network.py`/
`examples/free_fall_collapse.py` for fully worked drivers. This exists
for a fast one-off script or REPL check: one call, one dict back.
"""
import tempfile

from dengo.chemistry_constants import kboltz, mh
from dengo.primordial_network import build_network, build_solver

__all__ = ["quick_solve"]

_cache = None  # (mod, solver) -- built once per process, shared across calls


def _get_solver():
    global _cache
    if _cache is None:
        network = build_network()
        build_dir = tempfile.mkdtemp(prefix="dengo_quick_solve_")
        mod = build_solver(network, build_dir)
        solver = mod.Solver(1)
        _cache = (mod, solver)
    return _cache


def _make_ics(nH, T, x_ion, x_H2):
    """A mostly-neutral primordial gas at density nH (cm^-3), x_ion
    ionized and x_H2 molecular by number, with an initial gas energy
    corresponding to (roughly) temperature T -- the solver's own Newton
    iteration refines T self-consistently as it evolves."""
    return {
        "H_1": nH * 0.76 * (1.0 - x_ion),
        "H_2": nH * 0.76 * x_ion,
        "He_1": nH * 0.24 / 4.0 * (1.0 - x_ion),
        "He_2": 0.0,
        "He_3": nH * 0.24 / 4.0 * x_ion,
        "H_m0": nH * 1e-12,
        "H2_1": nH * x_H2,
        "H2_2": nH * 1e-12,
        "de": nH * 0.76 * x_ion + 2.0 * nH * 0.24 / 4.0 * x_ion,
        "ge": 1.5 * kboltz * T / mh,
    }


def quick_solve(nH=1e4, T=1000.0, dtf=3.15e13, x_ion=1e-4, x_H2=1e-6,
                niter=10000, reltol=1.0e-5, redshift=0.0, full_output=False):
    """Evolve a single zone of primordial (H/He/H2) gas at density `nH`
    for `dtf` seconds starting from temperature `T`, and return the
    final state as a plain dict.

    Builds and compiles the primordial network's solver once per
    process (cached, ~1-3s the first call, negligible after), so
    repeated calls in the same script/session are fast; each call still
    starts fresh from `nH`/`T`/`x_ion`/`x_H2`, not from any previous
    call's result.

    Parameters
    ----------
    nH : float
        Initial number density (cm^-3) of the gas (H+He+... combined,
        split into species assuming primordial abundances -- see
        `_make_ics`).
    T : float
        Initial temperature (K) -- just a starting guess; the solver's
        own Newton iteration re-derives T self-consistently as it
        evolves.
    dtf : float
        Total time to evolve, in seconds.
    x_ion : float
        Initial ionized fraction (0-1).
    x_H2 : float
        Initial H2/H fraction by number (0-1).
    niter : int
        Maximum number of adaptive internal sub-steps.
    reltol : float
        Relative tolerance for the Newton solver.
    redshift : float
        Only used for the CMB temperature floor in Compton cooling.
    full_output : bool
        If True, return `(final, solver)` instead of just `final` --
        `solver` is the shared, cached `Solver` handle, useful to check
        `solver.last_error` when `final["converged"]` is False.

    Returns
    -------
    final : dict[str, float]
        Every species' number density (cm^-3), plus "T" (K), "t" (the
        physical time actually reached, seconds), and "converged"
        (bool) -- check this before trusting the result; if False, see
        `full_output=True`'s `solver.last_error` for why.
    """
    mod, solver = _get_solver()
    solver.redshift = redshift
    ics = _make_ics(nH, T, x_ion, x_H2)
    for name, value in ics.items():
        solver.state[:, mod.SPECIES_INDEX[name]] = value

    converged, t = solver.step_inplace(dtf, niter=niter, reltol=reltol)

    final = {name: float(solver.state[0, mod.SPECIES_INDEX[name]]) for name in mod.SPECIES_NAMES}
    final["T"] = float(solver.T[0])
    final["t"] = t
    final["converged"] = converged

    if full_output:
        return final, solver
    return final
