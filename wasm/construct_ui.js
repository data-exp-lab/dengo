// Prototype "construct a network" page: one CodeMirror-edited Python
// snippet per reaction, each defining real dengo Species/Reaction
// objects -- the actual dengo package (not a reimplementation of its
// API), installed into Pyodide from a wheel built locally at site-
// generation time (uv build --wheel, see generate_site.py's
// build_generic_page()) and served as a plain static asset -- then run
// through the *exact same* generic mass-action engine
// (generic_kinetics.js) and compiled-once wasm integrator
// (dengo_generic.js/.wasm) the checkbox-driven catalog tool
// (generic/index.html) already uses. The engine doesn't change at all
// between the two tools, only where its input data comes from: a
// static, pre-baked reaction_db.json there, versus whatever a user's
// own Python constructs here. See NOTES.md and wasm/README-generic.md
// for the full design writeup.
//
// Reuses app.js's vlConfig()/formatTimeAuto()/csvCell()/TIME_LABEL_EXPR
// and generic_kinetics.js's loadReactionDb()/runGenericIntegration()
// (all loaded before this file, same page, shared top-level scope).

let pyodide = null;
let constructMod = null;
let constructDb = null;
let lastConstructResult = null;
let cardCounter = 0;
const editors = {}; // card element id -> CodeMirror instance

const DEFAULT_CARD_SOURCE = `from dengo.reaction_classes import Species, Reaction

A = Species("A", 1.0)
B = Species("B", 1.0)

def rate(state):
    return 0.1 + 0 * state.T  # constant -- a "really boring" rate law

Reaction("decay_A", rate, [(1, A)], [(1, B)])
`;

function setStatus(text) {
  document.getElementById("ck-status").textContent = text;
}

function addReactionCard(source) {
  const id = "ck-card-" + (cardCounter++);
  const card = document.createElement("div");
  card.className = "ck-card";
  card.id = id;
  card.innerHTML = `
    <div class="ck-card-header">
      <span>Reaction</span>
      <button type="button" class="btn-secondary ck-remove">Remove</button>
    </div>
    <textarea></textarea>
  `;
  document.getElementById("ck-cards").appendChild(card);
  const textarea = card.querySelector("textarea");
  textarea.value = source || DEFAULT_CARD_SOURCE;
  const cm = CodeMirror.fromTextArea(textarea, {
    mode: "python", lineNumbers: true, indentUnit: 4, viewportMargin: Infinity,
  });
  editors[id] = cm;
  card.querySelector(".ck-remove").addEventListener("click", () => {
    delete editors[id];
    card.remove();
  });
  return id;
}

// Installs the *real* dengo package -- not a reimplementation, not a
// hand-picked subset of files -- from a wheel built locally (uv build
// --wheel, see generate_site.py's build_generic_page()) and served as
// a plain static asset alongside this page, entirely offline: dengo
// isn't on PyPI (not ready for that yet), but micropip can install a
// wheel from *any* URL, including a same-origin relative one, which is
// all a locally-built wheel needs to be. `deps: false` skips trying to
// also resolve/install dengo's own declared dependencies (h5py, cython,
// setuptools -- pyproject.toml) -- none are actually needed just to
// construct Species/Reaction objects, and h5py in particular isn't a
// reliably available Pyodide package (see NOTES.md).
//
// `import dengo.reaction_classes` still hits a bare `import h5py`
// statement at that module's own top level -- never actually *called*
// by anything this page exercises (that only happens for CHIANTI/
// ion-by-ion rate functions, out of scope here and in the checkbox
// tool alike) -- so a bare stub module in sys.modules satisfies the
// import without needing the real package.
async function bootstrapDengo(py) {
  py.runPython("import sys, types\nsys.modules.setdefault('h5py', types.ModuleType('h5py'))\n");
  await py.loadPackage("micropip");
  const micropip = py.pyimport("micropip");
  await micropip.install(window.DENGO_WHEEL_FILENAME, { deps: false });
  py.runPython("import dengo.reaction_classes\n");
}

async function ensurePyodideLoaded() {
  if (pyodide) return pyodide;
  setStatus("loading Pyodide (first time only, several MB)…");
  pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/" });
  setStatus("loading dengo source…");
  await bootstrapDengo(pyodide);
  return pyodide;
}

// Reactions/species are dengo module-level registries (reaction_
// classes.py's reaction_registry/species_registry) -- global, mutable
// state that persists across Pyodide calls, so a rebuild has to clear
// whatever a *previous* build left behind (e.g. a card the user just
// deleted) rather than only ever adding to it.
function clearRegistries() {
  pyodide.runPython(
    "from dengo.reaction_classes import reaction_registry, species_registry\n" +
    "reaction_registry.clear()\n" +
    "species_registry.clear()\n",
  );
}

// Tabulates every registered reaction's rate over a fixed, generic T
// grid (this sketch's rate functions need not depend on T at all --
// "really boring" constant rates work fine, same code path either way)
// and hands back the exact {T_grid, species, reactions, cooling: []}
// shape generate_reaction_db.py's own export produces -- so
// loadReactionDb()/runGenericIntegration() downstream don't need to
// know or care that this one came from a user's own Python instead of
// build_primordial().
function collectReactionDb() {
  const jsonText = pyodide.runPython(`
import json
import numpy as np
from dengo.reaction_classes import reaction_registry, species_registry

T_grid = np.logspace(0, 4, 64)
class _State:
    pass
_state = _State()
_state.T = T_grid

reactions_out = []
touched = set()
for name, rxn in sorted(reaction_registry.items()):
    vals = np.asarray(rxn.coeff_fn(_state), dtype=float)
    if vals.shape == ():
        vals = np.full_like(T_grid, float(vals))
    reactions_out.append({
        "name": name,
        "left": [[n, s.name] for n, s in rxn.left_side],
        "right": [[n, s.name] for n, s in rxn.right_side],
        "rate": vals.tolist(),
    })
    for n, s in rxn.left_side:
        touched.add(s.name)
    for n, s in rxn.right_side:
        touched.add(s.name)

species_out = [
    {"name": nm, "weight": species_registry[nm].weight}
    for nm in sorted(touched)
]
json.dumps({"T_grid": T_grid.tolist(), "species": species_out, "reactions": reactions_out, "cooling": []})
`);
  return JSON.parse(jsonText);
}

// A species that's never *produced* by any reaction (only ever
// consumed -- a starting material, not an intermediate or end product)
// defaults to a nonzero starting value; anything a reaction produces
// defaults to 0, so a chain like A -> B -> C starts entirely in A and
// visibly fills B then C, rather than every species starting equally
// full and the intermediate's own rise-then-fall being invisible. Just
// a default -- each species still gets its own input either way.
function defaultInitialValue(dbRaw, name) {
  const isProducedSomewhere = dbRaw.reactions.some((r) => r.right.some(([, n]) => n === name));
  return isProducedSomewhere ? 0.0 : 1.0;
}

function buildSpeciesInputs(dbRaw) {
  const container = document.getElementById("ck-species-inputs");
  container.innerHTML = "";
  for (const sp of dbRaw.species) {
    const row = document.createElement("div");
    row.className = "row-inline";
    row.innerHTML = `
      <label>${sp.name} initial value</label>
      <input type="number" step="any" id="ck-init-${sp.name}" value="${defaultInitialValue(dbRaw, sp.name)}">
    `;
    container.appendChild(row);
  }
}

async function buildNetwork() {
  await ensurePyodideLoaded();
  if (!constructMod) constructMod = await DengoGenericModule(); // the same compiled-once integrator generic/index.html uses, loaded lazily here too
  clearRegistries();

  for (const id in editors) {
    try {
      pyodide.runPython(editors[id].getValue());
    } catch (e) {
      setStatus(`Error while running ${id}: ${e.message.split("\n").pop()}`);
      return;
    }
  }

  const dbRaw = collectReactionDb();
  if (!dbRaw.reactions.length) {
    setStatus("No reactions defined (every card must call Reaction(...)).");
    return;
  }
  constructDb = loadReactionDb(dbRaw);
  buildSpeciesInputs(dbRaw);
  document.getElementById("ck-run").disabled = false;
  setStatus(`built: ${dbRaw.species.length} species, ${dbRaw.reactions.length} reaction(s)`);
}

function runConstruct() {
  const activeSpecies = new Set(constructDb.species.map((sp) => sp.name));
  const activeReactions = new Set(constructDb.reactions.map((r) => r.name));

  const T0 = parseFloat(document.getElementById("ck-T").value);
  const dtfTotal = parseFloat(document.getElementById("ck-dtf").value);
  const initialState = {};
  for (const sp of constructDb.species) {
    initialState[sp.name] = parseFloat(document.getElementById("ck-init-" + sp.name).value);
  }

  const t0 = performance.now();
  const result = runGenericIntegration(
    constructMod, constructDb, activeSpecies, activeReactions, new Set(), initialState, T0, dtfTotal,
  );
  const elapsed = performance.now() - t0;
  lastConstructResult = result;

  const rows = [];
  for (let i = 0; i < result.t.length; i++) {
    for (const name of result.speciesOrder) {
      rows.push({ t: result.t[i], tHuman: formatTimeAuto(result.t[i]), species: name, value: result.s[i][name] });
    }
  }
  const spec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    width: 600, height: 320, background: null,
    config: vlConfig(),
    data: { values: rows },
    mark: { type: "line", point: { filled: true, size: 10, opacity: 0.9 } },
    encoding: {
      x: {
        field: "t", type: "quantitative", scale: { type: "symlog" },
        axis: { title: "t (s)", labelExpr: TIME_LABEL_EXPR, labelAngle: -40 },
      },
      y: { field: "value", type: "quantitative", axis: { title: "value" } },
      color: { field: "species", type: "nominal", scale: { scheme: "tableau10" } },
      tooltip: [
        { field: "species", type: "nominal" },
        { field: "tHuman", title: "t", type: "nominal" },
        { field: "value", title: "value", type: "quantitative", format: ".4~g" },
      ],
    },
  };
  vegaEmbed("#ck-chart", spec, { actions: false, renderer: "svg" });

  setStatus(`${result.t.length} points, ${elapsed.toFixed(1)} ms, reached t=${result.t[result.t.length - 1].toExponential(3)} s of ${dtfTotal.toExponential(3)} s requested`);
  document.getElementById("ck-download-csv").disabled = false;
}

function downloadConstructCsv() {
  if (!lastConstructResult) return;
  const { t, s, speciesOrder } = lastConstructResult;
  const header = ["step", "t_s", "t_human", ...speciesOrder];
  const lines = [header.map(csvCell).join(",")];
  for (let i = 0; i < t.length; i++) {
    lines.push([i, t[i], formatTimeAuto(t[i]), ...speciesOrder.map((name) => s[i][name])].map(csvCell).join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "construct_results.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Starter cards (if any) are added by the page's own inline script,
// via addReactionCard(source), *before* this runs -- so a fresh page
// load can seed specific example reactions (see CONSTRUCT_PAGE_
// TEMPLATE in generate_site.py) rather than always starting from one
// generic default.
function initConstructPage() {
  document.getElementById("ck-add-card").addEventListener("click", () => addReactionCard());
  document.getElementById("ck-build").addEventListener("click", buildNetwork);
  document.getElementById("ck-run").addEventListener("click", runConstruct);
  document.getElementById("ck-download-csv").addEventListener("click", downloadConstructCsv);
  setStatus("click “Build network” to load Pyodide and run your reactions");
}
