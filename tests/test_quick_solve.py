"""Tests for dengo.quick_solve -- the one-line convenience entry point
(see NOTES.md). Uses the real, module-level cached solver (like a real
caller would), not the tests/conftest.py fixtures, since the whole
point of this module is to avoid needing any of that.
"""
import pytest

import dengo


def test_quick_solve_returns_a_converged_physical_result():
    final = dengo.quick_solve(nH=1e4, T=8000.0, dtf=3.15e13)
    assert final["converged"]
    assert final["T"] > 0
    assert final["t"] == pytest.approx(3.15e13, rel=1e-6)
    for name in ("H_1", "H_2", "He_1", "He_2", "He_3", "H_m0", "H2_1", "H2_2", "de"):
        assert final[name] >= 0, name


def test_quick_solve_caches_the_solver_across_calls():
    """Second call onward should reuse the same compiled solver -- not
    a strict behavioral requirement a caller can observe directly, but
    the whole point of the module is that repeated calls are fast, so
    pin the caching mechanism itself."""
    from dengo.quick_solve import _get_solver

    mod1, solver1 = _get_solver()
    dengo.quick_solve(nH=1e3, T=500.0, dtf=3.15e10)
    mod2, solver2 = _get_solver()
    assert solver1 is solver2
    assert mod1 is mod2


def test_quick_solve_full_output_exposes_solver_and_last_error():
    from dengo.quick_solve import _get_solver

    mod, _ = _get_solver()
    final, solver = dengo.quick_solve(
        nH=1e4, T=1000.0, dtf=3.15e13, reltol=1.0e-300, full_output=True,
    )
    assert not final["converged"]
    err = solver.last_error
    assert err is not None
    assert err["reason"] == "tolerance"
    assert err["species"] in mod.SPECIES_NAMES
    assert err["species"] in err["message"]
