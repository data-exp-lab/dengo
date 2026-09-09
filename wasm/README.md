# Dengo in the browser (WebAssembly)

`index.html` is a self-contained, client-side version of
`examples/interactive_explorer.ipynb`: the same primordial H/He/H2
solver, compiled to WebAssembly, running entirely in the browser -- no
Python, no server, no Jupyter kernel. Open `index.html` (served over
http(s), not `file://` -- browsers block loading wasm/CDN scripts from
`file://`) and drag the sliders.

```
cd wasm && python3 -m http.server 8000
# then open http://localhost:8000/
```

## What's here

- `dengo_wasm.cpp` -- a small hand-written C API (`init()`,
  `step(dtf, niter, reltol)`, direct pointers to the state and RHS
  buffers) around dengo's generated solver, compiled with Emscripten.
  Hardcoded to the primordial network specifically for now (species
  order matches Python's `SPECIES_NAMES`) -- see "Generalizing" below.
- `build.sh` -- regenerates the primordial network's C++ source (via
  `dengo.primordial_network`, same as every other example) and compiles
  it + `dengo_wasm.cpp` to `dengo_wasm.js`/`dengo_wasm.wasm` with
  `em++`, embedding the ~475KB rate-table binary directly into the
  module (no separate fetch). Requires the
  [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html)
  (`emsdk install latest && emsdk activate latest`, no root needed) on
  `PATH`.
- `dengo_wasm.js`/`dengo_wasm.wasm` -- the compiled output, **committed**
  (unlike other generated artifacts in this repo) so `index.html` works
  for anyone who clones the repo and serves this directory, without
  needing Emscripten themselves. Re-run `build.sh` and commit the result
  whenever the primordial network definition changes.
- `index.html` -- the widget: sliders (density, temperature, ionized/H2
  fraction), a mode toggle (constant-density cooldown vs. free-fall
  collapse, matching the notebook's two modes), and
  [Vega-Lite](https://vega.github.io/vega-lite/) charts that redraw on
  every slider input.

## Verified, not assumed

Built and tested with a real Emscripten toolchain and a real headless
browser (Playwright + the system's Chrome) before being called done --
see NOTES.md for the full account:

- The compiled wasm module gives **bit-for-bit identical** results to
  the native Python/Cython solver on the same initial conditions.
- Per-call cost in Node/the browser (~110 us/cell) matches the
  already-optimized native per-cell cost -- no wasm performance penalty.
- The actual page, driven by a headless browser, reproduces the same
  free-fall trajectory (1298 steps, final T=2198.0K) as the native
  solver and the standalone wasm check.

## Generalizing beyond the primordial network

`dengo_wasm.cpp` is hand-written and specific to the primordial network
(hardcoded `NSPECIES`, species name order, `calculate_rhs_primordial`/
`calculate_jacobian_primordial` symbol names). Turning this into a
proper wasm codegen target -- so *any* dengo network can produce a
build like this one, the way `cython_solver_run.pyx.template` already
does for the Cython wrapper -- would mean a new
`dengo/templates/wasm_solver/dengo_wasm.cpp.template`, Jinja-rendered
the same way, plus wiring `build.sh`'s logic into
`ChemicalNetwork`/`solver_build`-style tooling rather than a one-off
shell script. Not done yet -- this directory is the validated proof of
concept for the primordial network specifically.
