# Dengo in the browser (WebAssembly)

A demo site: dengo's generated chemistry/cooling solver, compiled to
WebAssembly and running entirely client-side -- no Python, no server,
no Jupyter kernel -- with an interactive widget (sliders + live
[Vega-Lite](https://vega.github.io/vega-lite/) charts) for each of a
few fiducial networks. Deployed automatically to GitHub Pages by
[`.github/workflows/gh-pages.yml`](../.github/workflows/gh-pages.yml)
on every push (see that workflow's comments for why it fully
regenerates every network from source on every push, rather than just
republishing whatever's already built).

## What's here

- `../src/dengo/templates/wasm_solver/dengo_wasm.cpp.template` -- the
  Jinja template (rendered by `ChemicalNetwork.write_wasm_solver()`,
  alongside `../src/dengo/chemical_network.py`) for a small C API
  (`init`, `step(dtf, niter, reltol)`, direct pointers to the state and
  RHS buffers, `species_names()`) around the generated solver, meant to
  be compiled with Emscripten instead of Cython -- the wasm-equivalent
  of `cython_solver_run.pyx.template`. Always a single cell (`nstrip=1`
  -- this targets an interactive one-zone widget, not a grid-scale
  simulation).
- `fiducial_networks.py` -- three networks spanning a wide range of
  complexity (see "Fiducial networks" below), to show the codegen path
  genuinely works for more than the one network it was built against.
- `generate_site.py` -- for each fiducial network: generates the wasm
  solver source, compiles it with `em++`, and writes its page; plus one
  landing page linking to all of them. Requires the
  [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html)
  (`emsdk install <version> && emsdk activate <version>`, no root
  needed) on `PATH` -- see the pinned version in
  `.github/workflows/gh-pages.yml`.
- `app.js`/`style.css` -- the shared JS driver and styling every
  network's page loads: builds initial-fraction sliders dynamically
  from `dengo_wasm_species_names()` (nothing here is hardcoded to a
  particular species set), runs the same two modes as
  `examples/interactive_explorer.ipynb` (constant-density cooldown,
  free-fall collapse), and redraws on every slider `input` event,
  coalesced to once per animation frame.

Build locally with Emscripten on `PATH`:

```
uv run python wasm/generate_site.py wasm/_site
cd wasm/_site && python3 -m http.server 8000
# open http://localhost:8000/ (not file:// -- browsers block wasm/CDN
# script loading from file://)
```

## Fiducial networks

| name | species | reactions | cooling |
|---|---|---|---|
| `primordial` | full H/H+/He/He+/He++/H-/H2/H2+/e- | 22 (incl. the full H2 formation/dissociation chain) | all 17, incl. H2 cooling/formation heating |
| `primordial_atomic` | H/H+/He/He+/He++/e- (no H2 at all) | 6 (ionization/recombination only) | 13 (recombination, collisional excitation/ionization, bremsstrahlung, Compton -- no H2-specific terms) |
| `hydrogen_minimal` | H/H+/e- | 2 (`k01`/`k02`) | none |

`primordial` is the flagship: the ~1e15 amu/cc, 1500-2500K
H2-formation-heating regime this whole project targets. The other two
are deliberately smaller contrasts -- `primordial_atomic` shows the same
free-fall collapse without H2's cooling channel (compare its much
higher final temperature to `primordial`'s), and `hydrogen_minimal` is
the simplest network dengo can generate a solver for at all (matches
`tests/conftest.py`'s `make_hydrogen_network`).

## Verified, not assumed

Built and tested with a real Emscripten toolchain and a real headless
browser (Playwright + the system's Chrome) before being called done --
see NOTES.md for the full account:

- The compiled wasm module gives **bit-for-bit identical** results to
  the native Python/Cython solver on the same initial conditions (both
  the original hand-written wrapper and the current Jinja-templated
  codegen path, checked independently).
- Per-call cost in Node/the browser (~110 us/cell) matches the
  already-optimized native per-cell cost -- no wasm performance penalty.
- Each fiducial network's page, driven by a headless browser, converges
  and produces physically sensible trajectories in both modes;
  `primordial`'s free-fall run reproduces the same final temperature
  (~2198K) as every native/command-line check elsewhere in this repo.

## Generalizing further

Done: a proper Jinja-templated codegen path
(`ChemicalNetwork.write_wasm_solver()`), used identically by all three
fiducial networks above -- not hand-written per network anymore. Not
done: exposing multi-cell (`dims>1`) or redshift-dependent behavior
through the wasm API (irrelevant for a single-zone interactive widget,
so not attempted), and the per-species slider UI is generic but simple
(log-fraction sliders for every non-`ge`/non-`de` species) rather than
network-specific curated defaults beyond the flat `default_ics`/
`default_T` each fiducial network specifies in `fiducial_networks.py`.
