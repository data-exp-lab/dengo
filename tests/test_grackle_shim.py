"""dengo-grackle (`dengo.grackle_shim`): a libgrackle replacement, driven
through Grackle's C API by tests/grackle_shim/compare_grackle.c -- the
same calls, in the same order, Enzo makes.

Two kinds of check:

* The shim reproduces dengo itself: for every cell, converting Grackle's
  fields to cgs with Grackle's own conventions, stepping with the Cython
  `Solver`, converting back and applying Grackle's make_consistent gives
  the shim's answer. This pins down the unit conventions, ghost-zone
  handling and field mapping independent of any physics differences
  between dengo and Grackle.
* If the real libgrackle 3.4.1 is available (gracklepy's wheel bundles
  one; set DENGO_REAL_LIBGRACKLE to its path, or it's looked for in
  .grackle_compare/.venv), the same program is run against it too and
  the parts that should agree exactly -- unit conventions (temperature
  of the initial state), the code-unit k1/k2 tables, ghost zones,
  proper vs comoving -- are compared. Chemistry after a step is not
  asserted against Grackle: the two codes use different rate fits.
"""
import glob
import json
import os
import subprocess

import numpy as np
import pytest

import dengo.grackle_shim as grackle_shim
from conftest import timescale_first_step

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "grackle_shim", "compare_grackle.c")
MH = 1.67262171e-24
FIELDS = ["HI", "HII", "HeI", "HeII", "HeIII", "HM", "H2I", "H2II", "e"]
DENGO = {"HI": "H_1", "HII": "H_2", "HeI": "He_1", "HeII": "He_2", "HeIII": "He_3",
         "HM": "H_m0", "H2I": "H2_1", "H2II": "H2_2", "e": "de"}
MASS = {"HI": 1, "HII": 1, "HeI": 4, "HeII": 4, "HeIII": 4, "HM": 1, "H2I": 2, "H2II": 2, "e": 1}


def _real_libgrackle():
    path = os.environ.get("DENGO_REAL_LIBGRACKLE")
    if path:
        return path
    hits = glob.glob(os.path.join(HERE, "..", ".grackle_compare", ".venv", "lib", "python*",
                                  "site-packages", "gracklepy", "libgrackle-3.4*.so"))
    return hits[0] if hits else None


@pytest.fixture(scope="session")
def shim(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("dengo_grackle")
    built = grackle_shim.build(str(prefix))
    exe = str(prefix / "compare_ours")
    subprocess.run(["gcc", "-O2", "-I" + built["include"], HARNESS, "-o", exe,
                    "-L" + os.path.dirname(built["shared"]), "-lgrackle",
                    "-Wl,-rpath," + os.path.dirname(built["shared"]), "-lm"], check=True)
    real_exe = None
    real = _real_libgrackle()
    if real:
        real_exe = str(prefix / "compare_real")
        libdir = os.path.dirname(real)
        subprocess.run(["gcc", "-O2", "-I" + built["include"], HARNESS, "-o", real_exe, real,
                        "-Wl,-rpath," + libdir,
                        "-Wl,-rpath-link," + os.path.join(libdir, "..", "gracklepy.libs"),
                        "-lm"], check=True)
    return dict(prefix=prefix, built=built, exe=exe, real_exe=real_exe)


def run(exe, out, comoving, three_body, cie, h2opt, dt, *extra):
    proc = subprocess.run([exe, str(out), str(comoving), str(three_body), str(cie), str(h2opt),
                           repr(dt)] + [str(e) for e in extra],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    with open(out) as f:
        return json.load(f), proc.stderr + proc.stdout


def initial_fields(res):
    """The harness's initial interior state (code units), per cell."""
    names = ["density"] + FIELDS + ["internal_energy"]
    cols = {g: res["%s0" % ("%s_density" % g if g in FIELDS else g)] for g in names}
    return [{g: cols[g][c] for g in names} for c in range(res["ncell"])]


def to_cgs_factors(res):
    """Exactly the factors (and operation order) grackle_shim.C uses."""
    n_unit = res["density_units"] / MH
    v = res["velocity_units"]
    fac = {g: (n_unit / MASS[g] if MASS[g] != 1 else n_unit) for g in FIELDS}
    fac["internal_energy"] = v * v
    return fac


def dengo_reference(mod, res, dt, z):
    """Each cell through the Cython Solver, with Grackle's conventions."""
    fac = to_cgs_factors(res)
    out = {k: [] for k in FIELDS + ["internal_energy"]}
    for f in initial_fields(res):
        s = mod.Solver(1, redshift=z)
        for g in FIELDS:
            s.state[0, mod.SPECIES_INDEX[DENGO[g]]] = f[g] * fac[g]
        s.state[0, mod.SPECIES_INDEX["ge"]] = f["internal_energy"] * fac["internal_energy"]
        x = np.array(s.state[0])
        s.step_inplace(dt, niter=10000, first_step=timescale_first_step(mod, x, dt, redshift=z))
        new = {g: s.state[0, mod.SPECIES_INDEX[DENGO[g]]] / fac[g] for g in FIELDS}
        new["internal_energy"] = s.state[0, mod.SPECIES_INDEX["ge"]] / fac["internal_energy"]
        s.close()
        # Grackle's make_consistent_g, in grackle_shim.C's operation order
        fh = 0.76
        d = f["density"]
        totH = new["HI"] + new["HII"] + new["HM"] + new["H2I"] + new["H2II"]
        totHe = new["HeI"] + new["HeII"] + new["HeIII"]
        cH = fh * d / totH
        cHe = (1.0 - fh) * d / totHe
        for g in ("HI", "HII", "HM", "H2I", "H2II"):
            new[g] *= cH
        for g in ("HeI", "HeII", "HeIII"):
            new[g] *= cHe
        new["e"] = (new["HII"] + new["HeII"] / 4.0 + new["HeIII"] / 2.0
                    - new["HM"] + new["H2II"] / 2.0)
        for k in out:
            out[k].append(new[k])
    return {k: np.array(v) for k, v in out.items()}


@pytest.mark.parametrize("comoving", [0, 1])
def test_shim_reproduces_dengo(shim, primordial_solver, comoving, tmp_path):
    _, mod = primordial_solver
    dt = 3.15e10
    res, _ = run(shim["exe"], tmp_path / "o.json", comoving, 4, 1, 1, dt)
    assert res["initialized"] and res["ghosts_untouched"]
    z = 20.0 if comoving else 0.0
    ref = dengo_reference(mod, res, dt, z)
    for g in FIELDS:
        np.testing.assert_array_equal(res["%s_density1" % g], ref[g], err_msg=g)
    np.testing.assert_array_equal(res["internal_energy1"], ref["internal_energy"])


def test_cooling_time_units(shim, primordial_solver, tmp_path):
    """calculate_cooling_time returns dengo's ge/(dge/dt) in code time
    units, with dge/dt from radiative processes only -- the H2 formation
    heating / dissociation cooling tables zeroed, as in Grackle, whose
    cooling time leaves chemical heating out."""
    network, mod = primordial_solver
    res, _ = run(shim["exe"], tmp_path / "o.json", 0, 4, 1, 1, 1.0)
    _, names = grackle_shim._table_arrays(network)
    tables = np.fromfile(mod._DEFAULT_TABLE_FILE, dtype="float64").reshape(len(names), -1)
    for i, nm in enumerate(names):
        if nm.startswith("c_h2formation") and ("_h2mheat" in nm or "_h2mcool" in nm):
            tables[i] = 0.0
    table_file = str(tmp_path / "radiative_only.bin")
    tables.tofile(table_file)
    fac = to_cgs_factors(res)
    for c, f in enumerate(initial_fields(res)):
        state = {DENGO[g]: f[g] * fac[g] for g in FIELDS}
        state["ge"] = f["internal_energy"] * fac["internal_energy"]
        s = mod.Solver(1, table_file=table_file)
        rhs = s.evaluate_rhs(state)
        s.close()
        expected = state["ge"] / rhs["ge"] / res["time_units"]
        assert res["tcool0"][c] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("param,value", [
    ("primordial_chemistry", 3), ("metal_cooling", 1), ("UVbackground", 1),
    ("dust_chemistry", 1), ("self_shielding_method", 3), ("H2_self_shielding", 1),
    ("use_radiative_transfer", 1), ("CaseBRecombination", 1), ("use_temperature_floor", 1),
    ("three_body_rate", 6), ("LWbackground_intensity", 1.0), ("Gamma", 1.4),
])
def test_unsupported_configuration_refused(shim, param, value, tmp_path):
    res, log = run(shim["exe"], tmp_path / "o.json", 0, 4, 1, 1, 1.0, "reject", param, value)
    assert res["initialized"] is False
    assert param in log


def test_against_real_grackle(shim, tmp_path):
    if shim["real_exe"] is None:
        pytest.skip("real libgrackle 3.4.x not found (set DENGO_REAL_LIBGRACKLE)")
    for comoving in (0, 1):
        ours, _ = run(shim["exe"], tmp_path / "o.json", comoving, 4, 1, 1, 3.15e10)
        real, _ = run(shim["real_exe"], tmp_path / "r.json", comoving, 4, 1, 1, 3.15e10)
        assert ours["ghosts_untouched"] and real["ghosts_untouched"]
        assert ours["velocity_units"] == real["velocity_units"]
        assert ours["HydrogenFractionByMass"] == real["HydrogenFractionByMass"]
        # Same initial state -> same temperature: this is the unit-convention
        # check (density, energy, species masses). The residual is the two
        # codes' different m_H constants and H2 gamma(T) treatment.
        np.testing.assert_allclose(ours["T0"], real["T0"], rtol=1e-2)
        # code-unit k1/k2 (read directly by Enzo's photon timestep)
        assert ours["k1_mid"] == pytest.approx(real["k1_mid"], rel=2e-3)
        assert ours["k2_mid"] == pytest.approx(real["k2_mid"], rel=2e-3)
        # gas density is conserved by both
        np.testing.assert_allclose(ours["density1"], real["density1"], rtol=1e-14)
        # Cooling time: same definition (radiative only, signed, code
        # units), so same sign everywhere; magnitudes differ only through
        # the rate fits (largest at ~600-2000 K, H2 line cooling).
        a, b = np.array(ours["tcool0"]), np.array(real["tcool0"])
        assert np.all(np.sign(a) == np.sign(b))
        assert np.median(np.abs(a - b) / np.abs(b)) < 0.05


def test_rates_struct_copied_by_value(shim, tmp_path):
    """A by-value copy of chemistry_data_storage must work like the
    original (gracklepy copies it before every call)."""
    res, _ = run(shim["exe"], tmp_path / "o.json", 0, 4, 1, 1, 1.0)
    assert res["copied_rates_ok"] is True


def test_slightly_negative_species(shim, tmp_path):
    """Enzo's own initializers can leave species slightly negative; like
    Grackle (ceiling_species_g) the solve floors them instead of failing."""
    res, log = run(shim["exe"], tmp_path / "o.json", 0, 4, 1, 1, 3.15e10, "negative")
    assert res["solve_rv"] == 1
    assert "WARNING" not in log
    assert all(np.isfinite(res["T1"])) and min(res["HII_density1"]) > 0
    if shim["real_exe"]:
        real, _ = run(shim["real_exe"], tmp_path / "r.json", 0, 4, 1, 1, 3.15e10, "negative")
        np.testing.assert_allclose(res["T0"], real["T0"], rtol=1e-2)

