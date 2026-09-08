# Examples

Two actively-maintained, curated examples, built on the current
`dengo.chemical_network.ChemicalNetwork.write_cython_solver` +
`dengo.solver_build.build_solver` pipeline (no HDF5, no SUNDIALS):

- `primordial_network.py` -- define the primordial H/He/H2 network,
  generate + compile its solver, integrate a constant-density cooling box,
  and plot T(t) and the species fractions.
- `free_fall_collapse.py` -- a modified free-fall collapse (Omukai et al.
  2005 force-factor scheme) from diffuse-gas density up through the
  1500-2500 K / ~1e15 amu cm^-3 regime that's the priority target for the
  H2-formation-heating physics (see `NOTES.md`).

Everything else in this directory predates this modernization pass
(notebooks, the ion-by-ion/UMIST/CHIANTI/grackle-comparison scripts, the
old CVODE-based solver generation calls) and is not wired up to the
current backend -- left in place for reference, not maintained or tested.
