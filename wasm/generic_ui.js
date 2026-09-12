// Prototype "build your own network" page: checkboxes over every
// species/reaction generate_reaction_db.py exported (today: everything
// in dengo's primordial chemistry -- see that file's own doc comment
// for why the catalog itself is a separate, later concern from the
// engine built here), driving the generic (compiled-once, reused
// unmodified) wasm integrator via generic_kinetics.js's mass-action
// assembler. No em++ invocation happens for any selection a user picks
// -- see wasm/README-generic.md.
//
// Reuses app.js's vlConfig()/isDarkMode()/formatTimeAuto()/csvCell()
// (loaded before this file, same page, shared top-level scope -- see
// sweep.js's identical reuse of app.js) rather than reimplementing
// them.

let genericMod = null;
let genericDb = null;
let lastGenericResult = null; // {t, s, speciesOrder} -- CSV export source

function reactionLabel(r) {
  const side = (arr) => arr.map(([n, name]) => (n === 1 ? name : `${n}${name}`)).join(" + ");
  return `${side(r.left)} → ${side(r.right)}`;
}

// A reaction is only selectable once every species it touches is itself
// checked -- mirrors ChemicalNetwork.add_reaction(auto_add=False)'s own
// validation (raises if a reaction references a species not already in
// required_species) rather than inventing a different rule here.
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

function refreshReactionAvailability() {
  const checkedSpecies = checkedSpeciesNames();
  for (const r of genericDb.reactions) {
    const checkbox = document.getElementById("gk-rxn-" + r.name);
    const needed = reactionSpeciesNames(r);
    const available = Array.from(needed).every((n) => checkedSpecies.has(n));
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
  document.getElementById("gk-run").disabled = !anyReaction;
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
    checkbox.addEventListener("change", () => { refreshReactionAvailability(); refreshSpeciesRowVisibility(); });
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

function runGeneric() {
  const activeSpecies = checkedSpeciesNames();
  const activeReactions = new Set(
    genericDb.reactions.filter((r) => document.getElementById("gk-rxn-" + r.name).checked).map((r) => r.name),
  );

  const T = Math.pow(10, parseFloat(document.getElementById("gk-T").value));
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
  const result = runGenericIntegration(genericMod, genericDb, activeSpecies, activeReactions, initialState, T, dtfTotal);
  const elapsed = performance.now() - t0;
  lastGenericResult = result;

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
      y: { field: "value", type: "quantitative", scale: { type: "log" }, axis: { title: "n (cm⁻³)" } },
      color: { field: "species", type: "nominal", scale: { scheme: "tableau10" } },
      tooltip: [
        { field: "species", type: "nominal" },
        { field: "tHuman", title: "t", type: "nominal" },
        { field: "value", title: "n (cm⁻³)", type: "quantitative", format: ".4~g" },
      ],
    },
  };
  vegaEmbed("#gk-chart", spec, { actions: false, renderer: "svg" });

  document.getElementById("gk-status").textContent =
    `${result.t.length} points, ${elapsed.toFixed(0)} ms, T=${T.toExponential(3)} K (fixed), reached t=${result.t[result.t.length - 1].toExponential(3)} s of ${dtfTotal.toExponential(3)} s requested`;
  document.getElementById("gk-download-csv").disabled = false;
}

function downloadGenericCsv() {
  if (!lastGenericResult) return;
  const { t, s, speciesOrder } = lastGenericResult;
  const header = ["step", "t_s", "t_human", ...speciesOrder];
  const lines = [header.map(csvCell).join(",")];
  for (let i = 0; i < t.length; i++) {
    const row = [i, t[i], formatTimeAuto(t[i]), ...speciesOrder.map((name) => s[i][name])];
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

async function initGenericPage() {
  const dbResp = await fetch("reaction_db.json");
  genericDb = loadReactionDb(await dbResp.json());

  buildSpeciesRows();
  buildReactionRows();
  refreshReactionAvailability();

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

  genericMod = await DengoGenericModule();
  document.getElementById("gk-status").textContent = "ready";
}
