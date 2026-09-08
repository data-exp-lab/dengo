"""Sanity checks on the primordial reaction-rate and cooling-rate fits."""
import numpy as np
import pytest

from dengo.chemical_network import ChemicalNetwork
from dengo.reaction_classes import reaction_registry, cooling_registry
import dengo.primordial_rates as primordial_rates
import dengo.primordial_cooling as primordial_cooling  # noqa: F401

primordial_rates.setup_primordial()


class _FakeState:
    """A bare ChemicalNetwork's T-grid interface for a rate function to
    evaluate against, without needing any species/reactions added -- rate
    functions only read state.T/logT/tev/logtev and, for the three-body
    H2 formation channels, state.threebody."""

    def __init__(self, T):
        network = ChemicalNetwork()
        network.T = np.asarray(T, dtype="float64")
        network.logT = np.log(network.T)
        network.tev = network.T / 1.1605e4
        network.logtev = np.log(network.tev)
        self._network = network

    def __getattr__(self, name):
        return getattr(self._network, name)


@pytest.fixture(scope="module")
def state():
    return _FakeState(np.logspace(1, 8, 64))


@pytest.mark.parametrize("name", sorted(reaction_registry))
def test_reaction_rate_is_finite_and_nonnegative(name, state):
    vals = reaction_registry[name].coeff_fn(state)
    assert np.all(np.isfinite(vals)), name
    assert np.all(vals >= 0), name


@pytest.mark.parametrize("name", sorted(cooling_registry))
def test_cooling_tables_are_finite(name, state):
    action = cooling_registry[name]
    for tab_name, tab_fn in action.tables.items():
        vals = tab_fn(state)
        assert np.all(np.isfinite(vals)), (name, tab_name)


def test_k01_matches_reference_value():
    # H + e- -> H+ + 2e- (Abel et al. 1997 fit). Spot-check against a
    # value computed independently from the same fit coefficients.
    state = _FakeState([1.0e4])
    val = reaction_registry["k01"].coeff_fn(state)[0]
    tev = 1.0e4 / 1.1605e4
    logtev = np.log(tev)
    expected = np.exp(
        -32.71396786375
        + 13.53655609057 * logtev
        - 5.739328757388 * logtev**2
        + 1.563154982022 * logtev**3
        - 0.2877056004391 * logtev**4
        + 0.03482559773736999 * logtev**5
        - 0.00263197617559 * logtev**6
        + 0.0001119543953861 * logtev**7
        - 2.039149852002e-6 * logtev**8
    )
    assert val == pytest.approx(expected, rel=1e-10)


def test_ionization_rate_increases_with_temperature():
    # Collisional ionization should be a strongly increasing function of T
    # over this range (it is exponentially suppressed at low T).
    state = _FakeState([1.0e3, 1.0e4, 1.0e5])
    vals = reaction_registry["k01"].coeff_fn(state)
    assert vals[0] < vals[1] < vals[2]


def test_recombination_rate_decreases_with_temperature():
    # Case-B-like recombination (k02) falls off with increasing T over
    # this range.
    state = _FakeState([1.0e3, 1.0e4, 1.0e5])
    vals = reaction_registry["k02"].coeff_fn(state)
    assert vals[0] > vals[1] > vals[2]
