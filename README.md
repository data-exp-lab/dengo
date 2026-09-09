# Dengo

Dengo is a meta-solver for chemical and radiative-cooling reaction
networks in astrophysical fluids. You declare a network — species, rate
coefficients, cooling terms — in Python, and Dengo uses `sympy` and
`Jinja2` to generate a compiled, fast ODE solver for exactly that
network.

## Status

This is a from-the-ground-up modernization of a much older codebase (see
`NOTES.md` for the full, dated log of what changed and why — including a
couple of serious, previously-silent physics/correctness bugs found and
fixed along the way). What's actively supported and tested right now:

- The Python-level network-definition API (`Species`, `Reaction`,
  `CoolingAction`, `ChemicalNetwork` in `dengo/reaction_classes.py` and
  `dengo/chemical_network.py`).
- The primordial (metal-free) H/He/H2 chemistry network
  (`dengo/primordial_rates.py`, `dengo/primordial_cooling.py`), which is
  the priority use case: early-universe, pre-metal-enrichment collapse
  chemistry, including H2 formation/destruction heating (4.48 eV per
  molecule) and its later optical-depth suppression at high density.
- A single, self-contained solver backend
  (`dengo/templates/cython_solver/`): a generated C++ reaction-rate RHS
  and analytic Jacobian, driven by a vendored implicit
  backward-Euler/Newton integrator (`dengo/solvers/BE_chem_solve.C`) with
  adaptive sub-stepping. No external ODE library (e.g. SUNDIALS/CVODE)
  and no HDF5 are required — everything builds with a C++ compiler and
  Cython. Validated against an independent `scipy` BDF reference, and
  (see `examples/free_fall_collapse.py`) against the known qualitative
  shape of the primordial free-fall collapse temperature curve across
  ~15 decades of density.
- Choosing between literature rate-coefficient fits: right now this
  exists for the three-body H2 formation/collisional-dissociation rates
  specifically, via `ChemicalNetwork.threebody` (an int 0-5, default 4,
  matching the convention Grackle uses for its own `three_body_rate`
  parameter — see `k13`/`k22` in `primordial_rates.py`). Extending this
  to a general, cleanly-named per-reaction mechanism is the natural next
  step, not yet done (see `NOTES.md`).
- `dengo.grackle_compat`: a Grackle-API-compatible shim (`chemistry_data`/
  `FluidContainer`, matching Grackle's own field names/units convention)
  over this same primordial network, so code written against gracklepy
  can run against dengo instead for the physics dengo implements
  (Grackle's `primordial_chemistry=2`, no metal cooling/UV background/
  dust/radiative transfer -- anything else raises `GrackleCompatError`
  at `initialize()` rather than silently giving wrong physics).
  Validated directly against the real gracklepy in
  `.grackle_compare/validate_grackle_compat.py` (not a repo dependency,
  needs its own venv -- see `NOTES.md`).

What's present in the repo but **not** part of the modernized, tested
path (left alone intentionally, not because it's unimportant):

- Ion-by-ion metal-line cooling (`dengo/ion_by_ion.py`), which depends on
  ChiantiPy + the CHIANTI atomic database.
- The UMIST diffuse-cloud astrochemistry network
  (`dengo/umist_rates.py`, `dengo/get_rates.py`, `dengo/RATE12.txt`).
- The older CVODE/CVODE+OpenMP/CVODE+CUDA/JAX solver templates under
  `dengo/templates/legacy/` — kept for reference for anyone with SUNDIALS
  installed, unmaintained.
- `cookbook/`, `doc/`, `viz_network_examples/`, and most of `examples/`
  — older notebooks/docs/scripts, not refreshed as part of this pass
  (see `examples/README.md` for the two that are).

## Getting started

This project uses [uv](https://docs.astral.sh/uv/) for all Python
environment/dependency management. A C++ compiler (g++/clang++) is
required to build generated solvers.

```bash
uv sync                 # create the environment, install dengo + deps
uv run pytest            # run the unit test suite
uv run python examples/primordial_network.py
uv run python examples/free_fall_collapse.py
```

## Quickstart

```python
from dengo.chemical_network import ChemicalNetwork
import dengo.primordial_rates as primordial_rates
import dengo.primordial_cooling as primordial_cooling
import dengo.solver_build as solver_build
import numpy as np

primordial_rates.setup_primordial()

network = ChemicalNetwork()
network.add_collection(
    species_names=["H_1", "H_2", "He_1", "He_2", "He_3", "H_m0",
                   "H2_1", "H2_2", "de", "ge"],
    cooling_names=["cie_cooling", "gloverabel08", "h2formation",
                   "h2formation_extra", "reHII", "reHeII1", "reHeII2",
                   "reHeIII", "brem", "compton", "ceHI", "ceHeI",
                   "ceHeII", "ciHI", "ciHeI", "ciHeII", "ciHeIS"],
    reaction_names=["k01", "k02", "k03", "k04", "k05", "k06", "k07",
                     "k08", "k09", "k10", "k11", "k12", "k13", "k14",
                     "k15", "k16", "k17", "k18", "k19", "k21", "k22",
                     "k23"],
)
network.init_temperature((1e1, 1e8))

network.write_cython_solver("primordial", output_dir="/tmp/dengo_primordial")
mod = solver_build.build_solver("/tmp/dengo_primordial", "primordial")

ics = {
    "H_1": np.array([7.6e3]), "H_2": np.array([1.0]),
    "He_1": np.array([6.0e2]), "He_2": np.array([0.0]), "He_3": np.array([0.0]),
    "H_m0": np.array([1e-12]), "H2_1": np.array([1e-2]), "H2_2": np.array([1e-12]),
    "de": np.array([1.0]), "ge": np.array([1.5e12]),
}
final, trajectory = mod.run_primordial(ics, dtf=3.15e13, niter=10000)
print(final["T"])  # converged temperature
```

See `examples/primordial_network.py` for the same thing organized into
reusable functions plus a T(t) plot, `examples/free_fall_collapse.py`
for the free-fall collapse test problem across the density range this
project prioritizes (n ~ 1e15 amu/cc, T ~ 1500-2500 K), and
`examples/interactive_explorer.ipynb` for a Jupyter-widget version of
both you can poke at live with sliders.
