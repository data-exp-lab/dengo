# Legacy solver backends

These are the pre-modernization solver templates, kept for reference and
not part of the tested/maintained path:

- `cv`, `cv_omp`, `cvode_omp`, `cvode_cuda` — require SUNDIALS/CVODE
  (`CVODE_PATH`) and, for the `_omp`/`cuda` variants, OpenMP or CUDA
  toolchains. Useful if you already have SUNDIALS installed and want
  CVODE's adaptive BDF/Newton-Krylov machinery instead of the vendored
  solver.
- `cuda-accelerInt` — CUDA port targeting the accelerInt integrators.
- `be_chem_solve` — the predecessor of the current default
  `cython_solver` backend. Superseded because it hard-required
  `CVODE_PATH` even when it didn't need CVODE (a bug), required HDF5 at
  build+run time, and coupled its `.pyx`/C templates tightly to an
  Enzo-grid interface (`dengo_field_data`, `code_units`) not needed for
  standalone use. See `NOTES.md` at the repo root for the full writeup.
- `jax` — an incomplete JAX-based template stub.

None of these are exercised by the test suite. The actively maintained
backend is `dengo/templates/cython_solver/`.
