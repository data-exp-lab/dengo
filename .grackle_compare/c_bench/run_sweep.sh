#!/usr/bin/env bash
# Build and run both pure-C/C++ benchmarks (no Python, no Cython, in the
# timed path on either side -- see NOTES.md for why) across a range of
# grid sizes, and print a clean side-by-side table.
#
# Prerequisites (both one-time, not re-done by this script):
#   - .grackle_compare/_dengo_build/{primordial_solver.C,.h,BE_chem_solve.C,
#     primordial_tables.bin} already generated (dengo_bench.cpp compiles
#     directly against these -- run
#     `.grackle_compare/.venv/bin/python -c "..."` or any script that calls
#     ChemicalNetwork.write_cython_solver()/build_solver() into that dir
#     first, e.g. .grackle_compare/primordial_network_helpers.py)
#   - .grackle/src/include/grackle_float.h generated from
#     grackle_float.h.in (defining GRACKLE_FLOAT_8, matching the
#     installed gracklepy wheel's build -- see NOTES.md)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GP="$(pwd)/../.venv/lib/python3.12/site-packages/gracklepy"

# -lgrackle needs a "libgrackle.so" symlink; the installed wheel only
# ships the versioned "libgrackle-3.4.1.so" (its real SONAME).
mkdir -p _link
ln -sf "$GP/libgrackle-3.4.1.so" _link/libgrackle.so

echo "--- compiling ---"
gcc -O2 -o grackle_bench grackle_bench.c \
  -I../../.grackle/src/include \
  -L_link -Wl,-rpath,"$GP" -Wl,-rpath,"$GP/../gracklepy.libs" \
  -lgrackle -lm

g++ -O3 -fopenmp -I../_dengo_build -o dengo_bench dengo_bench.cpp \
  ../_dengo_build/primordial_solver.C ../_dengo_build/BE_chem_solve.C -lm

echo
echo "--- running (cd into _dengo_build so dengo_bench finds primordial_tables.bin;"
echo "    grackle_bench's data-file path is relative to c_bench/) ---"
echo

DIMS_LIST=(1 100 2048 100000)

echo "== dengo (pure C++, primordial_setup_data + BE_chem_solve directly) =="
(cd ../_dengo_build && for d in "${DIMS_LIST[@]}"; do ../c_bench/dengo_bench "$d" 10; done)

echo
echo "== grackle (pure C, local_initialize_chemistry_data + local_solve_chemistry directly) =="
for d in "${DIMS_LIST[@]}"; do ./grackle_bench "$d" 10; done
