# Examples

Actively-maintained, curated examples, built on the current
`dengo.chemical_network.ChemicalNetwork.write_cython_solver` +
`dengo.solver_build.build_solver` pipeline (no HDF5, no SUNDIALS):

- `primordial_network.py` -- define the primordial H/He/H2 network,
  generate + compile its solver, integrate a constant-density cooling box,
  and plot T(t) and the species fractions.
- `free_fall_collapse.py` -- plain (unimpeded) free-fall collapse from
  diffuse-gas density up through the 1500-2500 K / ~1e15 amu cm^-3 regime
  that's the priority target for the H2-formation-heating physics (see
  `NOTES.md` for why this uses plain free-fall rather than the Omukai et
  al. 2005 pressure-retarded/force-factor scheme an earlier version did).
- `interactive_explorer.ipynb` -- a Jupyter-widget notebook: set initial
  density/temperature/ionization/H2 fraction with sliders and see the
  constant-density cooldown or free-fall collapse solved and plotted
  immediately (uses the persistent `Solver`/`step_inplace()` API, so it's
  fast enough to feel live). Run with `uv run --group notebook jupyter
  lab examples/interactive_explorer.ipynb`.

Everything else in this directory predates this modernization pass
(notebooks, the ion-by-ion/UMIST/CHIANTI/grackle-comparison scripts, the
old CVODE-based solver generation calls) and is not wired up to the
current backend -- left in place for reference, not maintained or tested.
