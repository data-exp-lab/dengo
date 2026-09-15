// Prototype "build your own network" page: checkboxes over every
// species/reaction/cooling-action generate_reaction_db.py exported
// (today: everything in dengo's primordial chemistry -- see that
// file's own doc comment for why the catalog itself is a separate,
// later concern from the engine built here), driving the generic
// (compiled-once, reused unmodified) wasm integrator via
// generic_kinetics.js's mass-action assembler and cooling-action
// evaluator. No em++ invocation happens for any selection a user picks
// -- see wasm/README-generic.md.
//
// Reuses app.js's vlConfig()/isDarkMode()/formatTimeAuto()/csvCell()
// (loaded before this file, same page, shared top-level scope -- see
// sweep.js's identical reuse of app.js) rather than reimplementing
// them.

let genericMod = null;
let genericDb = null;
let lastGenericResult = null; // {t, s, speciesOrder, hasCooling} -- CSV export source
let originalFormulas = {}; // reaction name -> catalog-default formula string, snapshotted once at load (see initGenericPage()) -- exportSelection()'s only way to tell a rate-explorer-imported override apart from an untouched catalog default

function reactionLabel(r) {
  const side = (arr) => arr.map(([n, name]) => (n === 1 ? name : `${n}${name}`)).join(" + ");
  return `${side(r.left)} → ${side(r.right)}`;
}

// A reaction/cooling action is only selectable once every species it
// touches is itself checked -- mirrors ChemicalNetwork.add_reaction/
// add_cooling(auto_add=False)'s own validation (raises if either
// references a species not already in required_species) rather than
// inventing a different rule here.
function reactionSpeciesNames(r) {
  const names = new Set();
  for (const [, name] of r.left) names.add(name);
  for (const [, name] of r.right) names.add(name);
  return names;
}

function checkedSpeciesNames() {
  const names = new Set();
  for (const sp of genericDb.species) {
    if (document.getElementById("gk-sp-" + sp.name).checked) names.add(sp.name);
  }
  return names;
}

function refreshAvailability() {
  const checkedSpecies = checkedSpeciesNames();
  for (const r of genericDb.reactions) {
    const checkbox = document.getElementById("gk-rxn-" + r.name);
    const available = Array.from(reactionSpeciesNames(r)).every((n) => checkedSpecies.has(n));
    checkbox.disabled = !available;
    if (!available) checkbox.checked = false;
  }
  for (const c of genericDb.cooling) {
    const checkbox = document.getElementById("gk-cool-" + c.name);
    const available = c.species.every((n) => checkedSpecies.has(n));
    checkbox.disabled = !available;
    if (!available) checkbox.checked = false;
  }
  updateRunEnabled();
}

function refreshSpeciesRowVisibility() {
  for (const sp of genericDb.species) {
    const row = document.getElementById("gk-sp-row-" + sp.name);
    row.classList.toggle("gk-species-active", document.getElementById("gk-sp-" + sp.name).checked);
  }
}

function updateRunEnabled() {
  const anyReaction = genericDb.reactions.some((r) => document.getElementById("gk-rxn-" + r.name).checked);
  const anyCooling = genericDb.cooling.some((c) => document.getElementById("gk-cool-" + c.name).checked);
  document.getElementById("gk-run").disabled = !anyReaction && !anyCooling;
}

function buildSpeciesRows() {
  const container = document.getElementById("gk-species-list");
  for (const sp of genericDb.species) {
    const row = document.createElement("div");
    row.className = "row gk-species-row";
    row.id = "gk-sp-row-" + sp.name;
    const logFrac = Math.log10(sp.default_fraction);
    row.innerHTML = `
      <label class="checkbox-label">
        <input type="checkbox" id="gk-sp-${sp.name}" checked>
        <b>${sp.name}</b> <span class="gk-weight">(${sp.weight} amu)</span>
      </label>
      <div class="gk-species-fraction">
        <label>initial fraction <span class="val" id="gk-sp-${sp.name}-val"></span></label>
        <input type="range" id="gk-sp-${sp.name}-frac" min="-14" max="0" step="0.1" value="${logFrac}">
      </div>
    `;
    container.appendChild(row);
    const checkbox = row.querySelector("input[type=checkbox]");
    const fracSlider = row.querySelector("input[type=range]");
    const valEl = document.getElementById(`gk-sp-${sp.name}-val`);
    const updateVal = () => { valEl.textContent = Math.pow(10, parseFloat(fracSlider.value)).toExponential(2); };
    fracSlider.addEventListener("input", updateVal);
    checkbox.addEventListener("change", () => { refreshAvailability(); refreshSpeciesRowVisibility(); });
    updateVal();
  }
  refreshSpeciesRowVisibility();
}

function buildReactionRows() {
  const container = document.getElementById("gk-reaction-list");
  for (const r of genericDb.reactions) {
    const row = document.createElement("div");
    row.className = "row-inline gk-reaction-row";
    row.innerHTML = `
      <label class="checkbox-label">
        <input type="checkbox" id="gk-rxn-${r.name}" checked>
        <b>${r.name}</b> <span class="gk-eq">${reactionLabel(r)}</span>
      </label>
    `;
    container.appendChild(row);
    row.querySelector("input").addEventListener("change", updateRunEnabled);
  }
}

// Cooling actions have no left/right equation the way reactions do
// (they're a sink/source term on ge, not a species-to-species
// transformation) -- shown instead as the action name plus which
// species it depends on, unchecked/disabled by refreshAvailability()
// the same as a reaction whenever one of those species isn't checked.
function buildCoolingRows() {
  const container = document.getElementById("gk-cooling-list");
  for (const c of genericDb.cooling) {
    const row = document.createElement("div");
    row.className = "row-inline gk-reaction-row";
    row.innerHTML = `
      <label class="checkbox-label">
        <input type="checkbox" id="gk-cool-${c.name}">
        <b>${c.name}</b> <span class="gk-eq">(${c.species.join(", ")})</span>
      </label>
    `;
    container.appendChild(row);
    row.querySelector("input").addEventListener("change", updateRunEnabled);
  }
}

function runGeneric() {
  const activeSpecies = checkedSpeciesNames();
  const activeReactions = new Set(
    genericDb.reactions.filter((r) => document.getElementById("gk-rxn-" + r.name).checked).map((r) => r.name),
  );
  const activeCooling = new Set(
    genericDb.cooling.filter((c) => document.getElementById("gk-cool-" + c.name).checked).map((c) => c.name),
  );

  const T0 = Math.pow(10, parseFloat(document.getElementById("gk-T").value));
  const nH = Math.pow(10, parseFloat(document.getElementById("gk-nH").value));
  const logDtf = parseFloat(document.getElementById("gk-dtf").value);
  const dtfTotal = Math.pow(10, logDtf);

  const initialState = {};
  for (const sp of genericDb.species) {
    if (!activeSpecies.has(sp.name)) continue;
    const fracSlider = document.getElementById(`gk-sp-${sp.name}-frac`);
    initialState[sp.name] = nH * Math.pow(10, parseFloat(fracSlider.value));
  }

  const t0 = performance.now();
  const result = runGenericIntegration(
    genericMod, genericDb, activeSpecies, activeReactions, activeCooling, initialState, T0, dtfTotal,
  );
  const elapsed = performance.now() - t0;
  lastGenericResult = result;

  const rows = [];
  for (let i = 0; i < result.t.length; i++) {
    for (const name of result.speciesOrder) {
      rows.push({ t: result.t[i], tHuman: formatTimeAuto(result.t[i]), species: name, value: result.s[i][name] });
    }
  }
  const speciesSpec = {
    width: 600, height: 260,
    data: { values: rows },
    mark: { type: "line", point: { filled: true, size: 10, opacity: 0.9 } },
    encoding: {
      x: {
        field: "t", type: "quantitative", scale: { type: "symlog" },
        axis: { title: "t (s)", labelExpr: TIME_LABEL_EXPR, labelAngle: -40 },
      },
      y: { field: "value", type: "quantitative", scale: { type: "log" }, axis: { title: "n (cm⁻³)" } },
      color: { field: "species", type: "nominal", scale: { scheme: "tableau10" } },
      tooltip: [
        { field: "species", type: "nominal" },
        { field: "tHuman", title: "t", type: "nominal" },
        { field: "value", title: "n (cm⁻³)", type: "quantitative", format: ".4~g" },
      ],
    },
  };
  vegaEmbed("#gk-chart", { $schema: "https://vega.github.io/schema/vega-lite/v5.json", background: null, config: vlConfig(), ...speciesSpec }, { actions: false, renderer: "svg" });

  const chartTBoxEl = document.getElementById("gk-chart-T-box");
  if (result.hasCooling) {
    const tRows = result.t.map((t, i) => ({ t, tHuman: formatTimeAuto(t), T: result.s[i].T }));
    const tSpec = {
      width: 600, height: 180,
      data: { values: tRows },
      mark: { type: "line", point: { filled: true, size: 10, opacity: 0.9 } },
      encoding: {
        x: {
          field: "t", type: "quantitative", scale: { type: "symlog" },
          axis: { title: "t (s)", labelExpr: TIME_LABEL_EXPR, labelAngle: -40 },
        },
        y: { field: "T", type: "quantitative", scale: { type: "log" }, axis: { title: "T (K)" } },
        tooltip: [
          { field: "tHuman", title: "t", type: "nominal" },
          { field: "T", title: "T (K)", type: "quantitative", format: ".4~g" },
        ],
      },
    };
    chartTBoxEl.hidden = false;
    vegaEmbed("#gk-chart-T", { $schema: "https://vega.github.io/schema/vega-lite/v5.json", background: null, config: vlConfig(), ...tSpec }, { actions: false, renderer: "svg" });
  } else {
    chartTBoxEl.hidden = true;
  }

  const finalT = result.s[result.s.length - 1].T;
  const tDescr = result.hasCooling
    ? `T: ${T0.toExponential(3)} K → ${finalT.toExponential(3)} K`
    : `T=${T0.toExponential(3)} K (fixed)`;
  document.getElementById("gk-status").textContent =
    `${result.t.length} points, ${elapsed.toFixed(0)} ms, ${tDescr}, reached t=${result.t[result.t.length - 1].toExponential(3)} s of ${dtfTotal.toExponential(3)} s requested`;
  document.getElementById("gk-download-csv").disabled = false;
}

// Imports the *exact* JSON rates.html's own "Download JSON" produces
// (rates.js's exportJson(): {network, reactions: {name: {selected,
// formula, preset}}}) -- no new export format needed on that side at
// all. This is what actually lets the rate explorer's edits feed
// something real: rates.js's own doc comment calls that page
// "exploratory only... does not feed back into the compiled solver",
// but the generic engine here never needs a recompile, so there's
// nothing to feed back *into* -- a reaction's live-compiled formula
// (setReactionFormula(), generic_kinetics.js) is just swapped for the
// new one and the very next run uses it.
//
// A reaction rates.html doesn't know about (this tool's catalog and a
// given network's rates.html page should always agree today, since
// both come from build_primordial(), but nothing enforces that at
// import time) is skipped rather than failing the whole import, same
// defensive convention rates.js's own importJson() already uses for
// the reverse direction.
function importRateOverrides(file) {
  const reader = new FileReader();
  reader.onload = () => {
    let data;
    try {
      data = JSON.parse(reader.result);
    } catch (e) {
      alert("Could not parse that file as JSON: " + e.message);
      return;
    }
    const byName = Object.fromEntries(genericDb.reactions.map((r) => [r.name, r]));
    let matched = 0, skipped = 0;
    for (const [name, override] of Object.entries(data.reactions || {})) {
      const entry = byName[name];
      if (!entry || override.formula === undefined) { skipped++; continue; }
      setReactionFormula(entry, override.formula);
      const checkbox = document.getElementById("gk-rxn-" + name);
      // refreshAvailability() (called below) still has the final say --
      // it re-disables/unchecks this if the reaction's own species
      // aren't all checked, same rule every other checkbox here follows.
      if (checkbox) checkbox.checked = !!override.selected;
      matched++;
    }
    refreshAvailability();
    document.getElementById("gk-status").textContent =
      `imported rate overrides: ${matched} reaction(s) matched, ${skipped} skipped`;
  };
  reader.readAsText(file);
}

function downloadGenericCsv() {
  if (!lastGenericResult) return;
  const { t, s, speciesOrder } = lastGenericResult;
  const header = ["step", "t_s", "t_human", "T_K", ...speciesOrder];
  const lines = [header.map(csvCell).join(",")];
  for (let i = 0; i < t.length; i++) {
    const row = [i, t[i], formatTimeAuto(t[i]), s[i].T, ...speciesOrder.map((name) => s[i][name])];
    lines.push(row.map(csvCell).join(","));
  }
  const csv = lines.join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "generic_kinetics_results.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// -- Save/load in-progress work ------------------------------------------
// Every checkbox, slider, and imported rate override here lives only in
// this page's DOM/JS state -- same Blob-download / FileReader-import
// idiom as everything else in this codebase (rates.js, downloadGenericCsv()
// above, construct_ui.js's exportProject()).
function exportSelection() {
  const species = {};
  for (const sp of genericDb.species) {
    species[sp.name] = {
      checked: document.getElementById("gk-sp-" + sp.name).checked,
      log_fraction: parseFloat(document.getElementById(`gk-sp-${sp.name}-frac`).value),
    };
  }
  const reactions = {};
  for (const r of genericDb.reactions) {
    reactions[r.name] = { checked: document.getElementById("gk-rxn-" + r.name).checked };
  }
  const cooling = {};
  for (const c of genericDb.cooling) {
    cooling[c.name] = { checked: document.getElementById("gk-cool-" + c.name).checked };
  }
  // Only a reaction whose *current* formula differs from the catalog's
  // own default is worth round-tripping here -- everything else is
  // already implied by the exported network/reaction name and would
  // just bloat the file with 22 reactions' worth of formula text that
  // reaction_db.json already provides on its own.
  const rate_overrides = {};
  for (const r of genericDb.reactions) {
    if (r.formula !== originalFormulas[r.name]) rate_overrides[r.name] = { formula: r.formula };
  }
  const out = {
    tool: "generic-checkbox",
    T: parseFloat(document.getElementById("gk-T").value),
    nH: parseFloat(document.getElementById("gk-nH").value),
    dtf: parseFloat(document.getElementById("gk-dtf").value),
    species, reactions, cooling,
  };
  if (Object.keys(rate_overrides).length) out.rate_overrides = rate_overrides;
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "generic_selection.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// A reaction/species/cooling name this page's current catalog doesn't
// have is skipped rather than failing the whole import -- same
// defensive convention importRateOverrides()/rates.js's importJson()
// already use, since nothing here enforces the imported file actually
// came from *this* catalog version.
function importSelection(file) {
  const reader = new FileReader();
  reader.onload = () => {
    let data;
    try {
      data = JSON.parse(reader.result);
    } catch (e) {
      alert("Could not parse that file as JSON: " + e.message);
      return;
    }
    for (const [name, override] of Object.entries(data.rate_overrides || {})) {
      const entry = genericDb.reactions.find((r) => r.name === name);
      if (entry && override.formula !== undefined) setReactionFormula(entry, override.formula);
    }
    for (const [name, s] of Object.entries(data.species || {})) {
      const cb = document.getElementById("gk-sp-" + name);
      const frac = document.getElementById(`gk-sp-${name}-frac`);
      if (cb) cb.checked = !!s.checked;
      if (frac && s.log_fraction !== undefined) {
        frac.value = s.log_fraction;
        frac.dispatchEvent(new Event("input")); // updates its own val readout
      }
    }
    refreshSpeciesRowVisibility();
    for (const [name, r] of Object.entries(data.reactions || {})) {
      const cb = document.getElementById("gk-rxn-" + name);
      if (cb) cb.checked = !!r.checked;
    }
    for (const [name, c] of Object.entries(data.cooling || {})) {
      const cb = document.getElementById("gk-cool-" + name);
      if (cb) cb.checked = !!c.checked;
    }
    for (const id of ["T", "nH", "dtf"]) {
      if (data[id] === undefined) continue;
      const el = document.getElementById("gk-" + id);
      el.value = data[id];
      el.dispatchEvent(new Event("input")); // updates its own val readout (see GENERIC_PAGE_TEMPLATE's inline script)
    }
    // Has the final say on every reaction/cooling checkbox above -- a
    // species the import left unchecked still disables/unchecks
    // anything that depends on it, the same rule every other checkbox
    // change on this page already goes through.
    refreshAvailability();
    document.getElementById("gk-status").textContent = "loaded selection from file";
  };
  reader.readAsText(file);
}

async function initGenericPage() {
  const dbResp = await fetch("reaction_db.json");
  genericDb = loadReactionDb(await dbResp.json());
  originalFormulas = Object.fromEntries(genericDb.reactions.map((r) => [r.name, r.formula]));

  buildSpeciesRows();
  buildReactionRows();
  buildCoolingRows();
  refreshAvailability();

  document.getElementById("gk-run").addEventListener("click", runGeneric);
  document.getElementById("gk-download-csv").addEventListener("click", downloadGenericCsv);
  document.getElementById("gk-select-all-rxn").addEventListener("click", () => {
    for (const r of genericDb.reactions) {
      const cb = document.getElementById("gk-rxn-" + r.name);
      if (!cb.disabled) cb.checked = true;
    }
    updateRunEnabled();
  });
  document.getElementById("gk-select-none-rxn").addEventListener("click", () => {
    for (const r of genericDb.reactions) document.getElementById("gk-rxn-" + r.name).checked = false;
    updateRunEnabled();
  });
  document.getElementById("gk-select-all-cool").addEventListener("click", () => {
    for (const c of genericDb.cooling) {
      const cb = document.getElementById("gk-cool-" + c.name);
      if (!cb.disabled) cb.checked = true;
    }
    updateRunEnabled();
  });
  document.getElementById("gk-select-none-cool").addEventListener("click", () => {
    for (const c of genericDb.cooling) document.getElementById("gk-cool-" + c.name).checked = false;
    updateRunEnabled();
  });
  document.getElementById("gk-rates-import").addEventListener("change", (e) => {
    if (e.target.files[0]) importRateOverrides(e.target.files[0]);
    e.target.value = "";
  });
  document.getElementById("gk-export-selection").addEventListener("click", exportSelection);
  document.getElementById("gk-import-selection").addEventListener("change", (e) => {
    if (e.target.files[0]) importSelection(e.target.files[0]);
    e.target.value = ""; // allow re-importing the same filename twice in a row
  });

  genericMod = await DengoGenericModule();
  document.getElementById("gk-status").textContent = "ready";
}
