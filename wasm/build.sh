#!/usr/bin/env bash
# Build dengo_wasm.{js,wasm} from the current primordial network
# definition. Requires the Emscripten toolchain (em++) on PATH -- see
# https://emscripten.org/docs/getting_started/downloads.html (emsdk
# install/activate; no root needed) -- and `uv` for generating the C++
# solver source via the same dengo.primordial_network module every
# other example/tool in this repo uses.
#
# Regenerate + recompile whenever the primordial network definition
# changes (species/reactions/cooling, rate fits, ...) -- the committed
# dengo_wasm.js/.wasm are a build artifact, kept in the repo so
# index.html works for anyone who just clones the repo and serves this
# directory, without needing Emscripten themselves.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v em++ >/dev/null 2>&1; then
    echo "em++ not found on PATH -- install/activate the Emscripten SDK first" >&2
    echo "(https://emscripten.org/docs/getting_started/downloads.html)" >&2
    exit 1
fi

echo "--- generating primordial network C++ source ---"
rm -rf _build
uv run --project .. python -c "
from dengo.primordial_network import build_network
network = build_network()
network.write_cython_solver('primordial', output_dir='_build')
"

echo "--- compiling to WASM ---"
em++ -O3 -o dengo_wasm.js dengo_wasm.cpp \
    _build/primordial_solver.C _build/BE_chem_solve.C \
    -I_build \
    --embed-file _build/primordial_tables.bin@primordial_tables.bin \
    -sEXPORTED_FUNCTIONS=_dengo_wasm_init,_dengo_wasm_step,_dengo_wasm_state_ptr,_dengo_wasm_rhs_ptr,_dengo_wasm_temperature,_dengo_wasm_nspecies,_dengo_wasm_species_names,_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=cwrap,ccall,HEAPF64 \
    -sMODULARIZE=1 -sEXPORT_NAME=DengoModule \
    -sALLOW_MEMORY_GROWTH=1

echo "--- done: dengo_wasm.js / dengo_wasm.wasm ---"
ls -la dengo_wasm.js dengo_wasm.wasm
