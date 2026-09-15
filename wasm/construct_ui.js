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
# Every real dengo rate function (primordial_rates.py) references some
# subset of these -- not just T -- via the same state.tev/logtev/logT
# convention documented in reaction_rates.py; state.threebody (gating
# k13/k22's three-body H2 formation/dissociation channel choice) has
# no UI here, so it's fixed at ChemicalNetwork's own default (see
# chemical_network.py: self.threebody = 4) -- a hand-written card need
# not touch any of this (the two "really boring" starter cards don't),
# but one built from a real network's own reaction source (see
# generate_construct_examples.py) does.
_state.tev = T_grid / 11605.0
_state.logtev = np.log(_state.tev)
_state.logT = np.log(T_grid)
_state.threebody = 4

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
  // Disabled up front, not just left alone -- a *previous* build's
  // success shouldn't linger as a false "still runnable" signal if this
  // rebuild (e.g. after editing a card) fails partway through. Also
  // what importProject() checks afterward to tell a real rebuild
  // success apart from a failure, rather than assuming its own instance
  // of that same "not yet cleared" ambiguity away.
  document.getElementById("ck-run").disabled = true;
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

// -- Save/load in-progress work ------------------------------------------
// Everything a user has actually typed here (one or more real, possibly
// nontrivial Python reactions) lives only in this page's DOM/CodeMirror
// state -- a reload loses it outright, with no warning. Same Blob-
// download / FileReader-import idiom already used three times elsewhere
// in this codebase (rates.js, app.js's CSV export, sweep.js) -- no new
// mechanism, just applied to a project file instead of results.
function removeAllCards() {
  for (const id of Object.keys(editors)) delete editors[id];
  document.getElementById("ck-cards").innerHTML = "";
}

function exportProject() {
  // DOM order (not Object.keys(editors) insertion order, though those
  // should always agree) -- ties the exported card order directly to
  // what's visibly on screen, robust even if that ever changes.
  const cards = Array.from(document.querySelectorAll("#ck-cards .ck-card"))
    .map((card) => editors[card.id].getValue());
  const out = {
    tool: "generic-construct",
    T: parseFloat(document.getElementById("ck-T").value),
    dtf: parseFloat(document.getElementById("ck-dtf").value),
    cards,
  };
  // Initial-value overrides only exist once "Build network" has run at
  // least once (buildSpeciesInputs() is what creates the ck-init-*
  // fields) -- included when available so a deliberately-tweaked-away-
  // from-default starting point round-trips too, but their absence
  // (a project saved before ever building) isn't an error either.
  if (constructDb) {
    out.species_initial = Object.fromEntries(
      constructDb.species.map((sp) => [sp.name, parseFloat(document.getElementById("ck-init-" + sp.name).value)]),
    );
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "construct_project.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// The actual load logic, shared by two entry points: a user's own file
// (importProject(), below) and a build-time-generated fiducial-network
// example (loadExampleNetwork(), see initConstructPage()) -- same
// {tool, T, dtf, cards, species_initial} shape either way, so there's
// nothing entry-point-specific in here at all.
async function applyProjectData(data) {
  if (!Array.isArray(data.cards)) {
    alert("That file doesn't look like a construct.html project (no \"cards\" array).");
    return;
  }
  removeAllCards();
  for (const source of data.cards) addReactionCard(source);
  if (data.T !== undefined) document.getElementById("ck-T").value = data.T;
  if (data.dtf !== undefined) document.getElementById("ck-dtf").value = data.dtf;
  document.getElementById("ck-run").disabled = true;
  document.getElementById("ck-download-csv").disabled = true;
  setStatus(`loaded ${data.cards.length} reaction(s) -- click "Build network" to continue`);

  // Restoring species_initial needs the ck-init-* inputs to exist,
  // which only happens after a real build (same Pyodide/dengo path
  // "Build network" itself takes, not skipped or faked here) --
  // done automatically so a saved project's starting point comes
  // back exactly as it was, not just its reaction source.
  if (data.species_initial) {
    await buildNetwork();
    // buildNetwork() only clears ck-run's disabled flag once it
    // actually succeeds (every earlier failure path returns before
    // that line) -- checked rather than assumed, so a build failure
    // here (e.g. a since-broken card) reports *that* status instead
    // of silently claiming initial values were restored when the
    // ck-init-* inputs to restore them into were never created.
    if (!document.getElementById("ck-run").disabled) {
      for (const [name, value] of Object.entries(data.species_initial)) {
        const el = document.getElementById("ck-init-" + name);
        if (el) el.value = value;
      }
      setStatus(`loaded ${data.cards.length} reaction(s), rebuilt, restored initial values`);
    }
  }
}

function importProject(file) {
  const reader = new FileReader();
  reader.onload = () => {
    let data;
    try {
      data = JSON.parse(reader.result);
    } catch (e) {
      alert("Could not parse that file as JSON: " + e.message);
      return;
    }
    applyProjectData(data);
  };
  reader.readAsText(file);
}

// Starter cards (if any) are added by the page's own inline script,
// via addReactionCard(source), *before* this runs -- so a fresh page
// load can seed specific example reactions (see CONSTRUCT_PAGE_
// TEMPLATE in generate_site.py) rather than always starting from one
// generic default.
// Fetches one build-time-generated fiducial-network example
// (generate_construct_examples.py -- real dengo reaction source,
// inspect.getsource()'d out of primordial_rates.py, not reimplemented,
// see that file's own module docstring) and loads it exactly like an
// imported project file -- same applyProjectData(), just fetched from
// a same-origin static asset instead of read from a File.
async function loadExampleNetwork(key) {
  if (!key) return;
  setStatus(`loading example "${key}"…`);
  let data;
  try {
    const resp = await fetch(`examples/${key}.json`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    data = await resp.json();
  } catch (e) {
    setStatus(`could not load example "${key}": ${e.message}`);
    return;
  }
  await applyProjectData(data);
}

// Reuses app.js's own IC_PRESETS/applyPreset() data directly (app.js
// is loaded on this page before construct_ui.js, same top-level scope
// -- see CONSTRUCT_PAGE_TEMPLATE) rather than re-authoring the same
// four physically-motivated (nH, T, per-species fraction) bundles here
// -- one set of numbers, one place they could go stale. Mirrors
// applyPreset()'s own behavior as closely as this tool's different
// input shape allows: that one sets log-fraction *sliders*
// (Math.log10(frac)); construct.html's own species-initial inputs are
// plain number fields holding an absolute density, so this sets
// `frac * preset.nH` directly instead -- same source data, same "every
// species gets set, a preset-less one still gets the 1e-12 trace
// floor rather than being left alone" rule, just adapted to this
// tool's own units.
function applyIcPreset(key) {
  const preset = IC_PRESETS[key];
  if (!preset) return;
  if (!constructDb) {
    setStatus("Build a network first -- initial-value inputs don't exist until then.");
    return;
  }
  document.getElementById("ck-T").value = preset.T;
  // Same H2-folding rule as applyPreset(): a network with no H2_1
  // species can't represent the preset's molecular-hydrogen fraction
  // at all -- folded back into atomic H so the total hydrogen budget
  // stays physically sensible rather than partly vanishing.
  const hasH2 = constructDb.species.some((sp) => sp.name === "H2_1");
  let applied = 0;
  for (const sp of constructDb.species) {
    const el = document.getElementById("ck-init-" + sp.name);
    if (!el) continue; // this network doesn't have that species -- nothing to set
    let frac = (preset.fractions && preset.fractions[sp.name] !== undefined) ? preset.fractions[sp.name] : 1e-12;
    if (sp.name === "H_1" && !hasH2 && preset.fractions) {
      frac += 2 * ((preset.fractions.H2_1 || 0) + (preset.fractions.H2_2 || 0));
    }
    el.value = frac * preset.nH;
    applied++;
  }
  setStatus(`applied "${key}" initial conditions (T=${preset.T} K, n_H=${preset.nH} cm⁻³, ${applied} species set) -- click "Run" to see it`);
}

function initConstructPage() {
  document.getElementById("ck-add-card").addEventListener("click", () => addReactionCard());
  document.getElementById("ck-build").addEventListener("click", buildNetwork);
  document.getElementById("ck-run").addEventListener("click", runConstruct);
  document.getElementById("ck-download-csv").addEventListener("click", downloadConstructCsv);
  document.getElementById("ck-export-project").addEventListener("click", exportProject);
  document.getElementById("ck-import-project").addEventListener("change", (e) => {
    if (e.target.files[0]) importProject(e.target.files[0]);
    e.target.value = ""; // allow re-importing the same filename twice in a row
  });
  const exampleSelect = document.getElementById("ck-load-example");
  if (exampleSelect) {
    exampleSelect.addEventListener("change", (e) => {
      const key = e.target.value;
      e.target.value = ""; // back to the placeholder -- this is a one-shot action, not a persistent mode
      if (key) loadExampleNetwork(key);
    });
  }
  const icSelect = document.getElementById("ck-ic-preset");
  if (icSelect) {
    icSelect.addEventListener("change", (e) => {
      const key = e.target.value;
      e.target.value = ""; // one-shot action, same convention as ck-load-example above
      if (key) applyIcPreset(key);
    });
  }
  setStatus("click “Build network” to load Pyodide and run your reactions");
}
