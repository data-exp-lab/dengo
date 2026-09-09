"""A Grackle-API-compatible shim over dengo's own primordial H/He/H2
solver -- lets code written against gracklepy's `chemistry_data`/
`FluidContainer` API run against dengo instead, *for the physics dengo
actually implements*: primordial (metal-free) H/H+/He/He+/He++/H-/H2/
H2+/e- chemistry, matching Grackle's `primordial_chemistry=2` with no
metal cooling, no UV background, no dust, no radiative transfer, and no
D/D+/HD tracking.

This is deliberately narrow rather than a best-effort partial
translation: `chemistry_data.initialize()` raises `GrackleCompatError`
for any configuration outside that (`metal_cooling=1`,
`UVbackground=1`, `dust_chemistry=1`, `primordial_chemistry` != 2,
self-shielding, radiative transfer, ...) instead of silently ignoring
the setting and giving wrong physics. See NOTES.md for the difficulty
assessment this was scoped from, and .grackle_compare/ for the harness
that validates dengo's chemistry against Grackle's own in the first
place.

Usage mirrors gracklepy:

    from dengo.grackle_compat import chemistry_data, FluidContainer

    my_chemistry = chemistry_data()
    my_chemistry.density_units = mass_hydrogen_cgs
    my_chemistry.length_units = cm_per_mpc
    my_chemistry.time_units = sec_per_Myr
    my_chemistry.set_velocity_units()
    my_chemistry.three_body_rate = 4
    my_chemistry.cie_cooling = 1
    my_chemistry.h2_optical_depth_approximation = 1
    my_chemistry.initialize()

    fc = FluidContainer(my_chemistry, 1)
    fc["HI_density"][:] = ...
    fc["internal_energy"][:] = ...
    fc.solve_chemistry(dt)
    print(fc["temperature"])

Field values are stored in the same units convention real Grackle uses
(density fields in `chemistry_data.density_units`, `internal_energy` in
`velocity_units**2`) -- code that only touches fields by name and never
assumes a specific rate-table/solver implementation underneath shouldn't
need to know this isn't gracklepy.
"""
import tempfile

import numpy as np

from dengo.chemistry_constants import kboltz, mh
from dengo.primordial_network import build_network, build_solver

__all__ = [
    "GrackleCompatError",
    "chemistry_data",
    "FluidContainer",
    "solve_chemistry",
    "calculate_temperature",
    "calculate_pressure",
    "calculate_gamma",
    "calculate_cooling_time",
    "calculate_dust_temperature",
    "setup_fluid_container",
]


class GrackleCompatError(NotImplementedError):
    """Raised when a `chemistry_data` configuration requests physics
    this shim doesn't implement (metal cooling, UV background, dust,
    D/D+/HD chemistry, radiative transfer, self-shielding, ...) --
    instead of silently ignoring it and computing the wrong thing. Use
    the real gracklepy for any of those."""


# Grackle FluidContainer field name -> (dengo species name, species mass
# in amu) -- used to convert between Grackle's mass-density fields (code
# units) and dengo's number densities (cm^-3). Masses match gracklepy's
# own `_species_masses` in fluid_container.py.
_SPECIES_MASS = {
    "HI_density": ("H_1", 1.00794),
    "HII_density": ("H_2", 1.00794),
    "HeI_density": ("He_1", 4.002602),
    "HeII_density": ("He_2", 4.002602),
    "HeIII_density": ("He_3", 4.002602),
    "HM_density": ("H_m0", 1.00794),
    "H2I_density": ("H2_1", 2.01588),
    "H2II_density": ("H2_2", 2.01588),
    "e_density": ("de", 1.00794),
}
DENSITY_FIELDS = list(_SPECIES_MASS.keys())
_EXTRA_FIELDS = ["internal_energy", "x_velocity", "y_velocity", "z_velocity"]
_CALCULATED_FIELDS = ["temperature", "pressure", "gamma", "cooling_time"]

# gamma_i for every non-H2 species is the fixed monatomic/ionic value
# 5/3; H2's gamma(T) uses the Omukai & Nishi (1998) roto-vibrational
# formula -- same one dengo's own solver uses internally (see
# chemical_network.py's species_gamma()) -- reimplemented here in plain
# Python/numpy since it's a simple closed form and this way
# calculate_gamma()/calculate_pressure() need no new C/Cython plumbing.
_H2_NAMES = ("H2_1", "H2_2")


def _gamma_h2(T):
    x = 6100.0 / T
    with np.errstate(over="ignore", invalid="ignore"):
        expx = np.exp(x)
        g = 2.0 / (5.0 + 2.0 * x * x * expx / (expx - 1.0) ** 2.0) + 1.0
    return np.where(np.isfinite(g), g, 7.0 / 5.0)


def _mixture_gamma(state, species_names, T):
    """1 + n_tot / sum(n_i/(gamma_i(T)-1)) -- same combining rule as
    dengo's own ChemicalNetwork.gamma_factor()/Grackle's
    local_calculate_gamma(), computed here directly from `state`
    (shape (dims, NSPECIES)) and `T` (shape (dims,))."""
    n_tot = np.zeros(T.shape)
    inv_gm1_sum = np.zeros(T.shape)
    gamma_h2 = _gamma_h2(T)
    for name, idx in species_names.items():
        if name == "ge":
            continue
        n = state[:, idx]
        n_tot += n
        gamma_i = gamma_h2 if name in _H2_NAMES else (5.0 / 3.0)
        inv_gm1_sum += n / (gamma_i - 1.0)
    return 1.0 + n_tot / inv_gm1_sum


# Parameters this shim honors; anything else is accepted (so callers
# following gracklepy's own "instantiate then set attributes" idiom
# don't get an AttributeError on an unrelated field) but validated at
# initialize() time -- see _UNSUPPORTED_UNLESS_DEFAULT.
_DEFAULTS = dict(
    use_grackle=1,
    with_radiative_cooling=1,
    primordial_chemistry=2,
    dust_chemistry=0,
    metal_cooling=0,
    UVbackground=0,
    grackle_data_file="",
    cmb_temperature_floor=1,
    Gamma=5.0 / 3.0,
    three_body_rate=0,
    cie_cooling=0,
    h2_optical_depth_approximation=0,
    CaseBRecombination=0,
    self_shielding_method=0,
    H2_self_shielding=0,
    H2_custom_shielding=0,
    use_dust_density_field=0,
    use_radiative_transfer=0,
    use_volumetric_heating_rate=0,
    use_specific_heating_rate=0,
    use_temperature_floor=0,
    use_isrf_field=0,
    comoving_coordinates=0,
    density_units=1.0,
    length_units=1.0,
    time_units=1.0,
    a_units=1.0,
    a_value=1.0,
    velocity_units=1.0,
    max_iterations=10000,
    omp_nthreads=0,
)

# name -> the only value this shim supports (its "off"/no-op default).
_UNSUPPORTED_UNLESS_DEFAULT = {
    "primordial_chemistry": 2,
    "metal_cooling": 0,
    "dust_chemistry": 0,
    "UVbackground": 0,
    "self_shielding_method": 0,
    "H2_self_shielding": 0,
    "H2_custom_shielding": 0,
    "use_dust_density_field": 0,
    "use_radiative_transfer": 0,
    "use_volumetric_heating_rate": 0,
    "use_specific_heating_rate": 0,
    "use_isrf_field": 0,
}

_SOLVER_MODULE = None  # lazily built once, shared by every chemistry_data/FluidContainer


def _get_solver_module():
    global _SOLVER_MODULE
    if _SOLVER_MODULE is None:
        network = build_network()
        build_dir = tempfile.mkdtemp(prefix="dengo_grackle_compat_")
        _SOLVER_MODULE = build_solver(network, build_dir)
    return _SOLVER_MODULE


class chemistry_data:
    """Grackle-API-compatible configuration object: instantiate with no
    args, set attributes, call `initialize()` -- same ergonomics as
    gracklepy's `chemistry_data`, restricted to what dengo's primordial
    H/He/H2 network implements. Setting any of
    `_UNSUPPORTED_UNLESS_DEFAULT` away from its listed value raises
    `GrackleCompatError` at `initialize()` time.
    """

    def __init__(self):
        for name, val in _DEFAULTS.items():
            setattr(self, name, val)
        self._initialized = False

    def set_velocity_units(self):
        """velocity_units = length_units/time_units (divided by a_value
        if comoving) -- matches Grackle's grackle_units.c exactly."""
        self.velocity_units = self.length_units / self.time_units
        if self.comoving_coordinates:
            self.velocity_units /= self.a_value

    def get_velocity_units(self):
        return self.velocity_units

    def initialize(self):
        for name, required in _UNSUPPORTED_UNLESS_DEFAULT.items():
            value = getattr(self, name, required)
            if value != required:
                raise GrackleCompatError(
                    "dengo.grackle_compat only implements primordial H/He/H2 "
                    "chemistry (Grackle's primordial_chemistry=2, no metal "
                    "cooling/UV background/dust/radiative transfer/self-"
                    "shielding) -- %s=%r was requested, but this shim requires "
                    "%s==%r. Use the real gracklepy for this configuration." %
                    (name, value, name, required)
                )
        _get_solver_module()  # build/compile now, so a bad build fails here
        self._initialized = True
        return 1


class FluidContainer(dict):
    """Grackle-API-compatible field container backed by a dengo
    `Solver`. Field names/units match gracklepy's `FluidContainer` for
    the primordial_chemistry=2 field set -- see module docstring.
    """

    def __init__(self, my_chemistry, n_vals, dtype="float64"):
        super().__init__()
        self._closed = True  # until fully constructed, so a failed
                              # __init__ (e.g. uninitialized chemistry_data)
                              # doesn't touch self._solver in __del__
        if not my_chemistry._initialized:
            raise RuntimeError("chemistry_data.initialize() must be called first")
        self.chemistry_data = my_chemistry
        self.n_vals = n_vals
        self.dtype = dtype
        self._mod = _get_solver_module()
        self._solver = self._mod.Solver(n_vals)
        self._closed = False

        for field in self.all_fields:
            self[field] = np.zeros(n_vals, dtype=dtype)

    def close(self):
        if not self._closed:
            self._solver.close()
            self._closed = True

    def __del__(self):
        self.close()

    @property
    def density_fields(self):
        return DENSITY_FIELDS + ["density"]

    @property
    def input_fields(self):
        return _EXTRA_FIELDS + self.density_fields

    @property
    def all_fields(self):
        return self.input_fields + _CALCULATED_FIELDS + ["mean_molecular_weight"]

    def _push(self):
        """Write this FluidContainer's current fields (Grackle code
        units) into the backing Solver's persistent state buffer
        (dengo's native cgs number densities / specific energy),
        without advancing anything."""
        cd = self.chemistry_data
        state = self._solver.state
        idx = self._mod.SPECIES_INDEX
        for field, (name, amu) in _SPECIES_MASS.items():
            mass_density_cgs = self[field] * cd.density_units
            state[:, idx[name]] = mass_density_cgs / (amu * mh)
        state[:, idx["ge"]] = self["internal_energy"] * cd.velocity_units ** 2

    def _pull(self):
        """Inverse of _push(): copy the Solver's current state back into
        this FluidContainer's fields (Grackle code units)."""
        cd = self.chemistry_data
        state = self._solver.state
        idx = self._mod.SPECIES_INDEX
        for field, (name, amu) in _SPECIES_MASS.items():
            n = state[:, idx[name]]
            self[field][:] = n * amu * mh / cd.density_units
        self["internal_energy"][:] = state[:, idx["ge"]] / cd.velocity_units ** 2

    def solve_chemistry(self, dt):
        """Evolve this FluidContainer's fields forward by `dt` (in
        chemistry_data's code time units, matching Grackle) -- pushes
        the current fields into the solver, steps, pulls the result
        back out. Raises RuntimeError if the solve doesn't converge
        within the internal step budget, matching Grackle's own
        behavior on GRACKLE_FAIL_VALUE."""
        cd = self.chemistry_data
        self._push()
        dtf = dt * cd.time_units
        converged, t = self._solver.step_inplace(
            dtf, niter=cd.max_iterations, reltol=1.0e-5,
        )
        if not converged:
            raise RuntimeError(
                "solve_chemistry did not converge (reached t=%.6e of dt=%.6e)"
                % (t, dtf)
            )
        self._pull()

    def calculate_temperature(self):
        self._push()
        T = self._solver.evaluate_temperature_bulk()
        self["temperature"][:] = T

    def calculate_gamma(self):
        self._push()
        T = self._solver.evaluate_temperature_bulk()
        idx = self._mod.SPECIES_INDEX
        self["gamma"][:] = _mixture_gamma(self._solver.state, idx, T)

    def calculate_pressure(self):
        cd = self.chemistry_data
        self._push()
        T = self._solver.evaluate_temperature_bulk()
        idx = self._mod.SPECIES_INDEX
        n_tot = sum(self._solver.state[:, idx[name]]
                    for name in self._mod.SPECIES_NAMES if name != "ge")
        # pressure in code units: P_cgs / (density_units * velocity_units^2)
        P_cgs = n_tot * kboltz * T
        self["pressure"][:] = P_cgs / (cd.density_units * cd.velocity_units ** 2)

    def calculate_cooling_time(self):
        cd = self.chemistry_data
        self._push()
        rhs = self._solver.evaluate_rhs_bulk()
        idx = self._mod.SPECIES_INDEX
        ge = self._solver.state[:, idx["ge"]]
        dge_dt = rhs[:, idx["ge"]]
        cooling_time_cgs = np.where(dge_dt != 0, np.abs(ge / dge_dt), np.inf)
        self["cooling_time"][:] = cooling_time_cgs / cd.time_units

    def calculate_dust_temperature(self):
        raise GrackleCompatError(
            "dust physics (dust_chemistry) is not implemented by dengo's "
            "primordial network -- use the real gracklepy for dust runs."
        )

    def calculate_mean_molecular_weight(self):
        cd = self.chemistry_data
        self.calculate_temperature()
        self.calculate_gamma()
        self["mean_molecular_weight"][:] = self["temperature"] / (
            self["internal_energy"] * (self["gamma"] - 1.0)
            * kboltz / (mh * cd.velocity_units ** 2)
        )

    def finalize_data(self, data=None):
        """Matches gracklepy's FluidContainer.finalize_data() closely
        enough for evolve_constant_density()/evolve_freefall() (see
        setup_fluid_container() below) -- returns `data` with each
        entry stacked into an array, `time` left as-is (already in
        physical seconds by convention in those utilities)."""
        if data is None:
            return self
        out = {}
        for key, val in data.items():
            out[key] = np.squeeze(np.array(val))
        return out


def solve_chemistry(fc, dt):
    fc.solve_chemistry(dt)


def calculate_temperature(fc):
    fc.calculate_temperature()


def calculate_pressure(fc):
    fc.calculate_pressure()


def calculate_gamma(fc):
    fc.calculate_gamma()


def calculate_cooling_time(fc):
    fc.calculate_cooling_time()


def calculate_dust_temperature(fc):
    fc.calculate_dust_temperature()


def setup_fluid_container(my_chemistry, density=1.0, temperature=None,
                           converge=True, tolerance=0.01, max_iterations=10000,
                           metal_mass_fraction=0.0, state="ionized"):
    """A restricted version of gracklepy's own convenience function of
    the same name: builds a single-cell FluidContainer at the given
    (code-unit) density and temperature, in either a fully "ionized" or
    "neutral" starting composition. `converge` (iteratively solving to a
    fixed cooling-time-limited dt until temperature stabilizes) and
    metal fields are not supported here -- metal_mass_fraction must be
    0, matching this shim's no-metal-cooling restriction.
    """
    if metal_mass_fraction != 0.0:
        raise GrackleCompatError("metal_mass_fraction != 0 requires metal_cooling, "
                                  "not supported by this shim.")
    if converge:
        raise GrackleCompatError("setup_fluid_container(converge=True) (iterate to a "
                                  "self-consistent starting temperature) is not "
                                  "implemented by this shim -- call with converge=False "
                                  "and drive convergence yourself, e.g. via "
                                  "evolve_constant_density().")

    fc = FluidContainer(my_chemistry, 1)
    fc["density"][:] = density
    for field in DENSITY_FIELDS:
        fc[field][:] = 0.0

    x_ion = 1.0e-10 if state == "neutral" else (1.0 - 1.0e-10)
    hydrogen_fraction = 0.76
    fc["HII_density"][:] = density * hydrogen_fraction * x_ion
    fc["HI_density"][:] = density * hydrogen_fraction * (1.0 - x_ion)
    fc["HeIII_density"][:] = density * (1.0 - hydrogen_fraction) * x_ion
    fc["HeII_density"][:] = 0.0
    fc["HeI_density"][:] = density * (1.0 - hydrogen_fraction) * (1.0 - x_ion)
    fc["e_density"][:] = fc["HII_density"] + fc["HeII_density"] / 4.0 + fc["HeIII_density"] / 2.0
    fc["H2I_density"][:] = density * 1.0e-20
    fc["H2II_density"][:] = density * 1.0e-20
    fc["HM_density"][:] = density * 1.0e-20

    if temperature is not None:
        cd = my_chemistry
        mu = 1.0 if state == "neutral" else 0.6
        internal_energy_cgs = 1.5 * kboltz * temperature / (mu * mh)
        fc["internal_energy"][:] = internal_energy_cgs / cd.velocity_units ** 2

    return fc
