"""Check the sympy-differentiated analytic Jacobian against a
finite-difference approximation of the generated RHS."""
import numpy as np
import pytest

from conftest import primordial_ics


def _finite_difference_jacobian(evaluate_rhs, state, species_names, eps=1e-6):
    # Central differences: much less sensitive than one-sided differences
    # to the curvature of the (steeply T-dependent) rate coefficients.
    # Species sitting exactly at zero (e.g. He++ / H2 before any has
    # formed) get a forward difference instead -- calculate_rhs rejects
    # any negative abundance outright, so a centered step would perturb
    # them below zero and raise.
    base = evaluate_rhs(state)
    fd = {}
    for col in species_names:
        h = eps * max(abs(state[col]), 1.0)
        plus_state = dict(state)
        plus_state[col] = state[col] + h
        plus = evaluate_rhs(plus_state)
        if state[col] - h > 0:
            minus_state = dict(state)
            minus_state[col] = state[col] - h
            minus = evaluate_rhs(minus_state)
            for row in species_names:
                fd[(row, col)] = (plus[row] - minus[row]) / (2 * h)
        else:
            for row in species_names:
                fd[(row, col)] = (plus[row] - base[row]) / h
    return fd, base


def _assert_jacobians_agree(analytic, fd, rtol=1e-3, atol_fraction=1e-4):
    """Compare two {(row, col): value} Jacobian dicts with a combined
    absolute+relative tolerance (`numpy.isclose`'s convention), the same
    way `numpy.allclose` reconciles two different failure modes here:

    - Most (row, col) pairs in a multi-species network have no real
      coupling at a given state, so both values are ~0 -- down at the
      level of floating-point roundoff, not a meaningful *relative*
      comparison. `atol` (scaled to the size of the biggest entry in the
      problem) absorbs that.
    - The analytic Jacobian is not exact: reaction rates k(T) are only
      differentiated through "ge" (`ReactionCoefficient._eval_derivative`
      special-cases exactly the energy symbol), so the indirect
      dT/d(species_i) contribution through density/gamma_factor for
      species_i != "ge" is dropped. That's an intentional (inherited)
      quasi-Newton-style approximation -- BE_chem_solve's Newton
      iteration only needs an approximate Jacobian to converge -- and it
      shows up as a small, roughly state-independent *absolute* error,
      which is again what `atol` is for. `rtol` is what actually catches
      a real bug in the dominant, well-resolved entries.
    """
    max_abs = max(max(abs(v) for v in analytic.values()),
                  max(abs(v) for v in fd.values()))
    atol = atol_fraction * max_abs
    mismatches = [
        (key, a_val, fd[key])
        for key, a_val in analytic.items()
        if not np.isclose(a_val, fd[key], rtol=rtol, atol=atol)
    ]
    assert not mismatches, mismatches


def test_hydrogen_jacobian_matches_finite_difference(hydrogen_solver):
    network, mod = hydrogen_solver
    n = 1.0e10
    state = {
        "H_1": n * 0.999,
        "H_2": n * 0.001,
        "de": n * 0.001,
        "ge": 1.5e12,
    }
    analytic = mod.evaluate_jacobian(state)
    fd, base = _finite_difference_jacobian(mod.evaluate_rhs, state, mod.SPECIES_NAMES)
    # With only 3 non-"ge" species, "de" is ~1/8000th... no -- here it's a
    # much bigger fractional share of the density/gamma_factor sum than in
    # the 10-species primordial network, so the dropped dT/d(de) term (see
    # _assert_jacobians_agree's docstring) is proportionally bigger too:
    # ~1.3% on the dominant d(rhs)/d(de) entries, not the usual <0.1%.
    _assert_jacobians_agree(analytic, fd, rtol=2e-2)


def test_primordial_jacobian_matches_finite_difference(primordial_solver):
    network, mod = primordial_solver
    ics = primordial_ics()
    state = {name: ics[name][0] for name in mod.SPECIES_NAMES}
    analytic = mod.evaluate_jacobian(state)
    fd, base = _finite_difference_jacobian(mod.evaluate_rhs, state, mod.SPECIES_NAMES)
    _assert_jacobians_agree(analytic, fd)
