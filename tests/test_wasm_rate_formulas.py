"""Permanent drift check between the real dengo rate coefficients
(src/dengo/primordial_rates.py, actually executed by the solver via
Reaction.coeff_fn) and their hand-transcribed Vega-Lite formula strings
in wasm/reaction_rates.py (used only by the browser rate viewer,
wasm/rates.js -- see NOTES.md's 2026-09-09 entry).

That viewer's formulas are a second, independently-maintained copy of
the same physics, evaluated by a different language (Vega's own
sandboxed expression language, not Python/numpy) for a different
purpose (an editable in-browser chart, not the compiled solver). A
hand-transcription bug there is real and already happened once this
project (missing `datum.` prefixes) -- caught only by a one-off
verification script. This test productionizes that check: it runs on
every `pytest` invocation and fails loudly if the two ever disagree,
rather than relying on someone remembering to re-run a scratch script
after the next edit to either file.

This is deliberately NOT a single source of truth (see NOTES.md for why
that's a bigger, separate project) -- it's the cheap, permanent
alternative: two copies still exist, but silent drift between them is
no longer possible.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dengo.chemical_network import ChemicalNetwork
from dengo.reaction_classes import reaction_registry
import dengo.primordial_rates as primordial_rates

primordial_rates.setup_primordial()

# wasm/reaction_rates.py is a plain data module (no wasm/emscripten
# dependency), so it's safe to import directly -- just needs its
# directory on sys.path, same trick generate_site.py already uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wasm"))
from reaction_rates import REACTION_RATES  # noqa: E402


class _FakeState:
    """Same bare T-grid interface as test_rates.py's _FakeState -- rate
    functions only read state.T/logT/tev/logtev and (for the two
    three-body channels) state.threebody. Duplicated rather than
    imported: it's ~10 lines and this file should be readable/runnable
    on its own, matching this repo's existing preference for small
    independent copies over cross-test-file coupling (e.g. rates.js's
    own independent copy of vlConfig()).
    """

    def __init__(self, T):
        network = ChemicalNetwork()
        network.T = np.asarray(T, dtype="float64")
        network.logT = np.log(network.T)
        network.tev = network.T / 1.1605e4
        network.logtev = np.log(network.tev)
        self._network = network

    def __getattr__(self, name):
        return getattr(self._network, name)


# Vega's `calculate` expression language is close enough to Python
# syntax (arithmetic, comparisons, dotted attribute access, function
# calls) that the only real translation needed is its JS-style ternary
# (`cond ? a : b`), which Python has no operator for. Confirmed (see
# the generation script) that no formula in reaction_rates.py has more
# than one top-level ternary and none nest a ternary inside another,
# so a single, non-recursive split is enough -- a second "?" at depth 0
# raises rather than silently mis-splitting.
def _split_top_level_ternary(expr):
    depth = 0
    qpos = None
    for i, ch in enumerate(expr):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "?" and depth == 0:
            qpos = i
            break
    if qpos is None:
        return None
    depth = 0
    cpos = None
    for i in range(qpos + 1, len(expr)):
        ch = expr[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "?" and depth == 0:
            raise ValueError(f"nested/chained ternary not supported by this check: {expr!r}")
        elif ch == ":" and depth == 0:
            cpos = i
            break
    if cpos is None:
        raise ValueError(f"malformed ternary (no matching ':'): {expr!r}")
    return expr[:qpos].strip(), expr[qpos + 1 : cpos].strip(), expr[cpos + 1 :].strip()


def _vega_to_python(expr):
    split = _split_top_level_ternary(expr)
    if split is None:
        return expr
    cond, then, else_ = split
    # Python's conditional expression, like Vega's ternary (and unlike
    # numpy boolean-mask assignment), only evaluates the taken branch --
    # so this can't spuriously hit e.g. an exp() overflow in the branch
    # that isn't selected.
    return f"({then}) if ({cond}) else ({_vega_to_python(else_)})"


class _Datum:
    """Stand-in for Vega's per-row `datum` in a `calculate` transform."""

    __slots__ = ("T", "tev", "logtev", "logT")


def _eval_vega_formula(formula, T):
    datum = _Datum()
    datum.T = T
    datum.tev = T / 1.1605e4
    datum.logtev = math.log(datum.tev)
    datum.logT = math.log(T)
    namespace = {
        "datum": datum,
        "pow": pow,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "min": min,
        "max": max,
    }
    return eval(_vega_to_python(formula), {"__builtins__": {}}, namespace)  # noqa: S307


_T_GRID = np.logspace(1, 8, 24)


@pytest.mark.parametrize("name", sorted(REACTION_RATES))
def test_wasm_rate_formula_matches_real_coeff_fn(name):
    if name not in reaction_registry:
        pytest.skip(f"{name} is in reaction_rates.py but not registered (setup didn't run?)")
    entry = REACTION_RATES[name]
    state = _FakeState(_T_GRID)
    real_vals = reaction_registry[name].coeff_fn(state)

    for i, T in enumerate(_T_GRID):
        want = real_vals[i]
        got = _eval_vega_formula(entry["formula"], T)
        _assert_close(name, T, got, want)


@pytest.mark.parametrize(
    "name,preset_key",
    [
        (name, preset_key)
        for name, entry in REACTION_RATES.items()
        if "presets" in entry
        for preset_key in entry["presets"]
    ],
)
def test_wasm_rate_preset_matches_real_coeff_fn(name, preset_key):
    if name not in reaction_registry:
        pytest.skip(f"{name} is in reaction_rates.py but not registered (setup didn't run?)")
    threebody = int(preset_key.split("=")[1])
    formula = REACTION_RATES[name]["presets"][preset_key]
    state = _FakeState(_T_GRID)
    state.threebody = threebody  # shadows _FakeState.__getattr__'s delegation directly
    real_vals = reaction_registry[name].coeff_fn(state)

    for i, T in enumerate(_T_GRID):
        want = real_vals[i]
        got = _eval_vega_formula(formula, T)
        _assert_close(f"{name} ({preset_key})", T, got, want)


def _assert_close(label, T, got, want, rel_tol=1e-6, floor=1e-25):
    if abs(got) < floor and abs(want) < floor:
        return  # both effectively zero/tiny -- a ratio here is meaningless noise, not a real mismatch
    rel_err = abs(got - want) / abs(want) if want != 0 else abs(got)
    assert rel_err < rel_tol, (
        f"{label} at T={T}: wasm/reaction_rates.py formula gives {got!r}, "
        f"real coeff_fn gives {want!r} (rel err {rel_err:.3e}) -- these two "
        f"copies of the same rate have drifted apart, see NOTES.md"
    )
