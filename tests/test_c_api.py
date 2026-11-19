"""The grid-level C API (`ChemicalNetwork.write_c_api`), called through
ctypes the way a simulation code would call it from C: one array per
species over a ghost-zone-padded grid, in arbitrary units via `to_cgs`.

The reference everywhere is the Cython `Solver` stepping each cell on its
own -- the C API steps every cell independently with the same
`_advance()` loop, so the results should be bit-for-bit identical.
"""
import ctypes

import numpy as np
import pytest

import dengo.c_build as c_build
from conftest import make_primordial_network, timescale_first_step

NAME = "capi"


@pytest.fixture(scope="session")
def capi(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("c_api")
    network = make_primordial_network()
    network.write_c_api(NAME, output_dir=str(outdir))
    sources = [str(outdir / ("%s_solver.C" % NAME)), str(outdir / ("%s_c_api.C" % NAME)),
               str(outdir / "BE_chem_solve.C")]
    built = c_build.build_library(NAME, sources, str(outdir), include_dirs=[str(outdir)],
                                  defines=["DENGO_QUIET"], static=False)
    lib = ctypes.CDLL(built["shared"])
    names = sorted(s.name for s in network.required_species)
    n = len(names)

    class Grid(ctypes.Structure):
        _fields_ = [("grid_rank", ctypes.c_int),
                    ("grid_dimension", ctypes.POINTER(ctypes.c_int)),
                    ("grid_start", ctypes.POINTER(ctypes.c_int)),
                    ("grid_end", ctypes.POINTER(ctypes.c_int)),
                    ("fields", ctypes.POINTER(ctypes.c_double) * n),
                    ("to_cgs", ctypes.c_double * n)]

    class Options(ctypes.Structure):
        _fields_ = [("niter", ctypes.c_int), ("first_step_timescale", ctypes.c_double),
                    ("first_step_fraction", ctypes.c_double),
                    ("error_tolerance", ctypes.c_double),
                    ("quiet_path", ctypes.c_int),
                    ("reltol", ctypes.c_double),
                    ("floor_value", ctypes.c_double),
                    ("use_cie_optical_depth", ctypes.c_int),
                    ("use_h2_optical_depth", ctypes.c_int),
                    ("coupled_error_filter", ctypes.c_int)]

    lib.capi_create_from_file.restype = ctypes.c_void_p
    lib.capi_create_from_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(Options)]
    lib.capi_destroy.argtypes = [ctypes.c_void_p]
    lib.capi_default_options.argtypes = [ctypes.POINTER(Options)]
    lib.capi_solve_grid.argtypes = [ctypes.c_void_p, ctypes.POINTER(Grid), ctypes.c_double,
                                    ctypes.c_double, ctypes.POINTER(ctypes.c_long)]
    lib.capi_cooling_time_grid.argtypes = [ctypes.c_void_p, ctypes.POINTER(Grid),
                                           ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
    lib.capi_temperature_grid.argtypes = [ctypes.c_void_p, ctypes.POINTER(Grid),
                                          ctypes.POINTER(ctypes.c_double)]

    def make_context(**overrides):
        opts = Options()
        lib.capi_default_options(ctypes.byref(opts))
        for k, v in overrides.items():
            setattr(opts, k, v)
        ctx = lib.capi_create_from_file(str(outdir / ("%s_tables.bin" % NAME)).encode(),
                                        ctypes.byref(opts))
        assert ctx
        return ctx

    return dict(lib=lib, Grid=Grid, names=names, make_context=make_context)


DIMS = (6, 5, 4)      # padded extent, x fastest

GHOST = 1


def random_states(names, ncell, seed=1):
    """Per-cell cgs states across ~16 decades of density and 10^1.5-10^4.5 K."""
    rng = np.random.default_rng(seed)
    nH = 10 ** rng.uniform(-2, 14, ncell)
    T = 10 ** rng.uniform(1.5, 4.5, ncell)
    st = {"H_1": 0.76 * nH * (1 - 1e-3), "H_2": 0.76 * nH * 1e-3, "He_1": 0.24 / 4 * nH,
          "He_2": 1e-20 * nH, "He_3": 1e-20 * nH, "H_m0": 1e-12 * nH, "H2_1": 1e-4 * nH,
          "H2_2": 1e-12 * nH, "de": 0.76 * nH * 1e-3,
          "ge": 1.5 * 1.38e-16 * T / (1.22 * 1.67e-24)}
    return np.array([st[name] for name in names]).T  # (ncell, nfield)


class PaddedGrid:
    """Fields in "code units" (cgs / to_cgs), with sentinel ghost zones."""

    def __init__(self, env, states):
        self.env = env
        names = env["names"]
        n = len(names)
        nx, ny, nz = DIMS
        self.to_cgs = np.array([1e-3 if name != "ge" else 1e10 for name in names])
        self.fields = np.full((n, nz, ny, nx), -12345.0)  # C order: x fastest
        interior = (slice(GHOST, nz - GHOST), slice(GHOST, ny - GHOST), slice(GHOST, nx - GHOST))
        self.interior = interior
        shape = self.fields[0][interior].shape
        for f in range(n):
            self.fields[f][interior] = (states[:, f] / self.to_cgs[f]).reshape(shape)
        self.dims = (ctypes.c_int * 3)(nx, ny, nz)
        self.start = (ctypes.c_int * 3)(GHOST, GHOST, GHOST)
        self.end = (ctypes.c_int * 3)(nx - 1 - GHOST, ny - 1 - GHOST, nz - 1 - GHOST)
        g = env["Grid"]()
        g.grid_rank = 3
        g.grid_dimension, g.grid_start, g.grid_end = self.dims, self.start, self.end
        for f in range(n):
            g.fields[f] = self.fields[f].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            g.to_cgs[f] = self.to_cgs[f]
        self.grid = g

    def interior_cgs(self):
        n = len(self.env["names"])
        return np.array([self.fields[f][self.interior].ravel() * self.to_cgs[f] for f in range(n)]).T

    def scalar(self):
        out = np.full(DIMS[::-1], -999.0)
        return out, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def ncell_interior():
    return int(np.prod([d - 2 * GHOST for d in DIMS]))


def test_solve_grid_matches_cython_per_cell(capi, primordial_solver):
    _, mod = primordial_solver
    assert list(mod.SPECIES_NAMES) == capi["names"]
    states = random_states(capi["names"], ncell_interior())
    pg = PaddedGrid(capi, states)
    ctx = capi["make_context"]()
    dt = 3.15e13
    z = 15.0
    nfail = ctypes.c_long(-1)
    rv = capi["lib"].capi_solve_grid(ctx, ctypes.byref(pg.grid), dt, z, ctypes.byref(nfail))
    capi["lib"].capi_destroy(ctx)
    assert rv in (0, 1) and nfail.value >= 0

    # ghost zones untouched
    ghost = np.ones(DIMS[::-1], bool)
    ghost[pg.interior] = False
    assert np.all(pg.fields[:, ghost] == -12345.0)

    got = pg.interior_cgs()
    expected_failures = 0
    for c in range(states.shape[0]):
        x = states[c] / pg.to_cgs * pg.to_cgs  # same rounding as the C gather
        first = timescale_first_step(mod, x, dt, redshift=z)
        s = mod.Solver(1, redshift=z)
        s.state[0, :] = x
        ok, _ = s.step_inplace(dt, first_step=first, error_tolerance=0.1)
        expected_failures += (not ok)
        # the C side divides by to_cgs on the way out; undo identically
        np.testing.assert_array_equal(got[c], (s.state[0] / pg.to_cgs) * pg.to_cgs)
        s.close()
    assert nfail.value == expected_failures


def test_cooling_time_and_temperature(capi, primordial_solver):
    _, mod = primordial_solver
    states = random_states(capi["names"], ncell_interior(), seed=7)
    pg = PaddedGrid(capi, states)
    ctx = capi["make_context"]()
    tcool, tcool_p = pg.scalar()
    temp, temp_p = pg.scalar()
    assert capi["lib"].capi_cooling_time_grid(ctx, ctypes.byref(pg.grid), 10.0, tcool_p) == 0
    assert capi["lib"].capi_temperature_grid(ctx, ctypes.byref(pg.grid), temp_p) == 0
    capi["lib"].capi_destroy(ctx)

    ghost = np.ones(DIMS[::-1], bool)
    ghost[pg.interior] = False
    assert np.all(tcool[ghost] == -999.0) and np.all(temp[ghost] == -999.0)

    cgs = pg.interior_cgs()
    ige = capi["names"].index("ge")
    for c, (tc, T) in enumerate(zip(tcool[pg.interior].ravel(), temp[pg.interior].ravel())):
        s = mod.Solver(1, redshift=10.0)
        state = dict(zip(capi["names"], cgs[c]))
        assert T == s.evaluate_temperature(state)
        s2 = mod.Solver(1, redshift=10.0)
        rhs = s2.evaluate_rhs(state)
        assert tc == cgs[c, ige] / rhs["ge"]
        s.close(); s2.close()


def test_h2_optical_depth_switch(capi):
    """At n ~ 1e12 cm^-3 the H2 line-cooling optical-depth factor is well
    below 1, so turning it off must make the gas cool faster. (Fully
    molecular gas, so three-body H2-formation heating doesn't dominate.)"""
    names = capi["names"]
    nH, T = 1e12, 1500.0
    st = {"H_1": 1e-4 * nH, "H_2": 1e-8 * nH, "He_1": 0.24 / 4 * nH, "He_2": 1e-20,
          "He_3": 1e-20, "H_m0": 1e-12, "H2_1": 0.38 * nH, "H2_2": 1e-12,
          "de": 1e-8 * nH, "ge": 1.5 * 1.38e-16 * T / (1.22 * 1.67e-24)}
    states = np.tile([st[name] for name in names], (ncell_interior(), 1))
    times = []
    for flag in (1, 0):
        pg = PaddedGrid(capi, states)
        ctx = capi["make_context"](use_h2_optical_depth=flag)
        tcool, p = pg.scalar()
        assert capi["lib"].capi_cooling_time_grid(ctx, ctypes.byref(pg.grid), 0.0, p) == 0
        capi["lib"].capi_destroy(ctx)
        times.append(tcool[pg.interior].ravel()[0])
    assert times[0] < 0 and times[1] < 0
    assert abs(times[1]) < 0.5 * abs(times[0])


def test_invalid_arguments(capi):
    states = random_states(capi["names"], ncell_interior())
    pg = PaddedGrid(capi, states)
    ctx = capi["make_context"]()
    lib = capi["lib"]
    pg.grid.grid_rank = 4
    assert lib.capi_solve_grid(ctx, ctypes.byref(pg.grid), 1.0, 0.0, None) == -1
    pg.grid.grid_rank = 3
    pg.end[0] = DIMS[0]  # past the end of the array
    assert lib.capi_solve_grid(ctx, ctypes.byref(pg.grid), 1.0, 0.0, None) == -1
    pg.end[0] = DIMS[0] - 1 - GHOST
    pg.grid.to_cgs[0] = 0.0
    assert lib.capi_solve_grid(ctx, ctypes.byref(pg.grid), 1.0, 0.0, None) == -1
    lib.capi_destroy(ctx)
