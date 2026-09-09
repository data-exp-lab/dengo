"""End-to-end tests of the compiled Cython/C++ solver: build once (via the
`hydrogen_solver`/`primordial_solver` fixtures), then check it against an
independent reference and basic physical invariants."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dengo.reaction_classes import reaction_registry
from conftest import primordial_ics


class _FakeState:
    def __init__(self, T):
        self.T = np.asarray(T, dtype="float64")
        self.tev = self.T / 1.1605e4
        self.logtev = np.log(self.tev)


def test_hydrogen_network_matches_scipy_reference(hydrogen_solver):
    network, mod = hydrogen_solver
    n = 1.0e10
    ics = {
        "H_1": np.array([n * 0.999]),
        "H_2": np.array([n * 0.001]),
        "de": np.array([n * 0.001]),
        "ge": np.array([1.5e12]),
    }
    dtf = 3.15e13
    final, _ = mod.run_test_hydrogen(ics, dtf=dtf, niter=2000, intermediate=False)
    assert final["converged"]

    T = final["T"][0]
    state = _FakeState([T])
    k01 = reaction_registry["k01"].coeff_fn(state)[0]
    k02 = reaction_registry["k02"].coeff_fn(state)[0]

    def rhs(t, y):
        H, Hp, e = y
        r = k01 * H * e - k02 * Hp * e
        return [-r, r, r]

    sol = solve_ivp(
        rhs, [0, dtf],
        [n * 0.999, n * 0.001, n * 0.001],
        method="BDF", rtol=1e-10, atol=1e-6 * n,
    )
    scipy_ratio = sol.y[0, -1] / sol.y[1, -1]
    dengo_ratio = final["H_1"][0] / final["H_2"][0]
    assert dengo_ratio == pytest.approx(scipy_ratio, rel=5e-3)


def test_hydrogen_network_conserves_total_hydrogen(hydrogen_solver):
    network, mod = hydrogen_solver
    n = 1.0e10
    ics = {
        "H_1": np.array([n * 0.999]),
        "H_2": np.array([n * 0.001]),
        "de": np.array([n * 0.001]),
        "ge": np.array([1.5e12]),
    }
    final, _ = mod.run_test_hydrogen(ics, dtf=3.15e13, niter=2000, intermediate=False)
    total = final["H_1"][0] + final["H_2"][0]
    assert total == pytest.approx(n, rel=1e-6)


def test_primordial_network_converges_and_is_physical(primordial_solver):
    network, mod = primordial_solver
    ics = primordial_ics()
    final, traj = mod.run_test_primordial(ics, dtf=3.15e13, niter=10000)
    assert final["converged"]
    assert np.isfinite(final["T"][0])
    assert final["T"][0] > 0
    for name in mod.SPECIES_NAMES:
        assert final[name][0] >= 0, name


def test_primordial_network_conserves_hydrogen_nuclei(primordial_solver):
    """Total H nuclei locked in H, H+, H-, 2*H2, 2*H2+ should not change."""
    network, mod = primordial_solver
    ics = primordial_ics()
    total_H_initial = (
        ics["H_1"][0] + ics["H_2"][0] + ics["H_m0"][0]
        + 2 * ics["H2_1"][0] + 2 * ics["H2_2"][0]
    )
    final, _ = mod.run_test_primordial(ics, dtf=3.15e13, niter=10000, intermediate=False)
    total_H_final = (
        final["H_1"][0] + final["H_2"][0] + final["H_m0"][0]
        + 2 * final["H2_1"][0] + 2 * final["H2_2"][0]
    )
    assert total_H_final == pytest.approx(total_H_initial, rel=1e-3)


def test_primordial_network_conserves_helium_nuclei(primordial_solver):
    network, mod = primordial_solver
    ics = primordial_ics()
    total_He_initial = ics["He_1"][0] + ics["He_2"][0] + ics["He_3"][0]
    final, _ = mod.run_test_primordial(ics, dtf=3.15e13, niter=10000, intermediate=False)
    total_He_final = final["He_1"][0] + final["He_2"][0] + final["He_3"][0]
    assert total_He_final == pytest.approx(total_He_initial, rel=1e-3)


def test_step_inplace_matches_step(primordial_solver):
    """`step_inplace()` (zero dict/array marshaling per call, writes/reads
    `solver.state`/`solver.T` directly -- see NOTES.md) has to reach
    exactly the same state as the dict-based `step()`, since both funnel
    through the same `_advance()` core; run the same evolution both ways
    from the same initial condition and compare."""
    network, mod = primordial_solver
    ics = primordial_ics()
    dtf = 3.15e13

    with mod.Solver(1) as solver:
        final, _ = solver.step(ics, dtf=dtf, niter=10000, intermediate=False)

    with mod.Solver(1) as solver:
        for name, arr in ics.items():
            solver.state[:, mod.SPECIES_INDEX[name]] = arr
        converged, t = solver.step_inplace(dtf=dtf, niter=10000)
        assert converged == final["converged"]
        assert t == pytest.approx(final["t"])
        assert float(solver.T[0]) == pytest.approx(final["T"][0], rel=1e-10)
        for name in mod.SPECIES_NAMES:
            assert solver.state[0, mod.SPECIES_INDEX[name]] == pytest.approx(
                final[name][0], rel=1e-10
            ), name


def test_solver_state_view_is_persistent_and_writable(primordial_solver):
    """`solver.state`/`solver.T` should be the same ndarray object across
    calls (a real zero-copy view onto the persistent buffer, not
    something recomputed/reallocated per access) and directly writable,
    e.g. to apply a compression step in place between calls."""
    network, mod = primordial_solver
    ics = primordial_ics()
    with mod.Solver(1) as solver:
        state_ref = solver.state
        T_ref = solver.T
        for name, arr in ics.items():
            solver.state[:, mod.SPECIES_INDEX[name]] = arr
        assert solver.state is state_ref
        assert solver.T is T_ref
        # a plain in-place rescale, as a free-fall compression step would do
        h2_idx = mod.SPECIES_INDEX["H2_1"]
        before = solver.state[0, h2_idx]
        solver.state[:, h2_idx] *= 2.0
        assert solver.state[0, h2_idx] == pytest.approx(2.0 * before)


def test_evaluate_bulk_methods_match_single_cell_and_multi_cell(primordial_solver):
    """evaluate_temperature_bulk()/evaluate_rhs_bulk() (all `dims` cells,
    via self.state, no marshaling) have to agree with the existing
    single-cell evaluate_temperature()/evaluate_rhs() dict-based API for
    dims=1, and give an independent answer per cell for dims>1 (not just
    broadcasting cell 0)."""
    network, mod = primordial_solver
    ics = primordial_ics()

    with mod.Solver(1) as solver:
        for name, arr in ics.items():
            solver.state[:, mod.SPECIES_INDEX[name]] = arr
        T_bulk = float(solver.evaluate_temperature_bulk()[0])
        rhs_bulk = {name: solver.evaluate_rhs_bulk()[0, mod.SPECIES_INDEX[name]]
                    for name in mod.SPECIES_NAMES}
        T_single = solver.evaluate_temperature({name: ics[name][0] for name in mod.SPECIES_NAMES})
        rhs_single = solver.evaluate_rhs({name: ics[name][0] for name in mod.SPECIES_NAMES})

    # Both converge their own Newton iteration to 1e-8 relative internally
    # (see calculate_temperature's Tdiff/Tnew check) from potentially
    # different starting guesses, so they agree closely but not to
    # floating-point precision -- rel=1e-10 is tighter than that solve
    # itself guarantees.
    assert T_bulk == pytest.approx(T_single, rel=1e-4)
    for name in mod.SPECIES_NAMES:
        assert rhs_bulk[name] == pytest.approx(rhs_single[name], rel=1e-4), name

    # primordial_ics() takes T_guess as ge's literal value and holds
    # abundance *fractions* fixed as nH scales, so density/gamma_factor
    # (and hence T) is actually nH-independent unless T_guess also
    # differs -- vary both to get a genuinely different T for cell 1.
    ics2 = primordial_ics(nH=1e6, T_guess=5.0e4)
    with mod.Solver(2) as solver:
        for name in mod.SPECIES_NAMES:
            solver.state[0, mod.SPECIES_INDEX[name]] = ics[name][0]
            solver.state[1, mod.SPECIES_INDEX[name]] = ics2[name][0]
        T2 = solver.evaluate_temperature_bulk()
        rhs2 = solver.evaluate_rhs_bulk()
        assert T2[0] == pytest.approx(T_single, rel=1e-4)
        assert T2[1] != pytest.approx(T2[0])  # genuinely different, not broadcast
        ge_idx = mod.SPECIES_INDEX["ge"]
        assert rhs2[0, ge_idx] == pytest.approx(rhs_single["ge"], rel=1e-4)


def test_zero_abundance_species_does_not_produce_nan(primordial_solver):
    """Regression test: He_2 = He_3 = 0 (fully neutral helium) used to
    poison the whole solver with NaN via BE_chem_solve's 1/scale
    normalization -- see NOTES.md, 2026-09-08."""
    network, mod = primordial_solver
    ics = primordial_ics()
    assert ics["He_2"][0] == 0.0
    assert ics["He_3"][0] == 0.0
    final, _ = mod.run_test_primordial(ics, dtf=3.15e13, niter=10000, intermediate=False)
    assert final["converged"]
    assert np.isfinite(final["T"][0])
