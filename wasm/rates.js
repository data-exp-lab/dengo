// Reaction-rate viewer/editor. Deliberately has no WASM/Emscripten
// dependency at all -- every rate is evaluated entirely by Vega-Lite's
// own data-generation (`data: {sequence: ...}`) and transform
// (`calculate`, using Vega's own sandboxed expression language) pipeline.
// That's the same mechanism app.js already uses for axis-label
// formatting (TIME_LABEL_EXPR); here it's the actual rate evaluator, not
// just formatting -- so editing a formula and re-embedding the same
// spec shape is the entire "live update" story, no custom parser/
// evaluator needed anywhere in this file.
//
// Exploratory only, by design (see the note on the page itself): edits
// here update that reaction's own plot immediately, and can be exported
// to/imported from JSON, but do not feed back into the compiled solver
// on the widget page. That's a deliberate, separate follow-up, not an
// oversight -- see NOTES.md.

let RATE_CONFIG = null;
const EDITED = {}; // reaction name -> current formula string, only once it differs from the active default/preset
const SELECTED_PRESET = {}; // reaction name -> preset key, for reactions that have presets

function isDarkMode() {
  return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
}

// Small, deliberately independent copy of app.js's vlConfig() -- this
// page doesn't load app.js at all (it has no use for the solver-driving
// code in it), so a shared helper isn't worth a new cross-page module
// for two colors' worth of config.
function vlConfig() {
  const dark = isDarkMode();
  return {
    background: null,
    view: { stroke: "transparent" },
    axis: {
      labelColor: dark ? "#b7bcc4" : "#333333",
      titleColor: dark ? "#b7bcc4" : "#333333",
      gridColor: dark ? "#33363d" : "#e6e6e6",
      domainColor: dark ? "#565a63" : "#888888",
      tickColor: dark ? "#565a63" : "#888888",
    },
    line: { color: dark ? "#8ab4ff" : "#3366cc" },
  };
}

function activeFormula(name) {
  if (EDITED[name] !== undefined) return EDITED[name];
  const rate = RATE_CONFIG.rates[name];
  const preset = SELECTED_PRESET[name];
  if (preset && rate.presets && rate.presets[preset] !== undefined) return rate.presets[preset];
  return rate.formula;
}

function rateChartSpec(formula) {
  const Tmin = RATE_CONFIG.Tmin, Tmax = RATE_CONFIG.Tmax;
  const logTmin = Math.log10(Tmin), logTmax = Math.log10(Tmax);
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    width: 260, height: 160, background: null,
    config: vlConfig(),
    // Sampled log-uniformly in T (a linear sequence in log10(T), then
    // exponentiated) so a log-x axis gets even coverage across
    // however many decades this network's temperature range spans.
    data: { sequence: { start: logTmin, stop: logTmax, step: (logTmax - logTmin) / 200, as: "logTax" } },
    transform: [
      { calculate: "pow(10, datum.logTax)", as: "T" },
      // tev/logtev/logT match primordial_rates.py's own state.tev/
      // state.logtev/state.logT exactly -- natural logs, not log10
      // (confirmed by primordial_rates.py's k09, which explicitly
      // divides state.logT by ln(10) to recover log10(T) -- see
      // reaction_rates.py's module docstring).
      { calculate: "datum.T / 11605.0", as: "tev" },
      { calculate: "log(datum.tev)", as: "logtev" },
      { calculate: "log(datum.T)", as: "logT" },
      { calculate: formula, as: "rate" },
    ],
    mark: { type: "line" },
    encoding: {
      x: {
        field: "T", type: "quantitative", scale: { type: "log" },
        axis: { title: "T (K)", labelFontSize: 9, titleFontSize: 10, labelOverlap: "greedy" },
      },
      y: {
        field: "rate", type: "quantitative", scale: { type: "log" },
        axis: { title: null, labelFontSize: 9 },
      },
      tooltip: [
        { field: "T", title: "T (K)", type: "quantitative", format: ".3~g" },
        { field: "rate", title: "rate", type: "quantitative", format: ".4~g" },
      ],
    },
  };
}

function renderCardChart(name) {
  const el = document.getElementById("chart-" + name);
  if (!el) return;
  let spec;
  try {
    spec = rateChartSpec(activeFormula(name));
  } catch (e) {
    el.innerHTML = '<p class="chart-placeholder">Invalid formula: ' + e.message + "</p>";
    return;
  }
  el.innerHTML = "";
  vegaEmbed(el, spec, { actions: false, renderer: "svg" }).catch((e) => {
    el.innerHTML = '<p class="chart-placeholder">Invalid formula: ' + e.message + "</p>";
  });
}

function buildCard(name) {
  const rate = RATE_CONFIG.rates[name];
  const card = document.createElement("div");
  card.className = "rate-card";
  card.id = "rate-card-" + name;

  const presetKeys = rate.presets ? Object.keys(rate.presets) : ["default"];
  const presetOptions = presetKeys
    .map((k) => `<option value="${k}">${k}</option>`)
    .join("");
  const currentPreset = SELECTED_PRESET[name] || rate.default_preset || presetKeys[0];

  card.innerHTML = `
    <div class="rate-card-header">
      <b>${name}</b>: ${rate.equation}
      ${rate.source ? `<span class="rate-source">${rate.source}</span>` : ""}
    </div>
    <div class="rate-card-controls">
      <label>preset <select id="preset-${name}">${presetOptions}</select></label>
      <button type="button" id="revert-${name}" title="discard edits, back to the selected preset/default">revert</button>
    </div>
    <textarea id="formula-${name}" spellcheck="false" rows="3"></textarea>
    <div id="chart-${name}"></div>
  `;
  card.querySelector(`#preset-${name}`).value = currentPreset;
  if (rate.presets) SELECTED_PRESET[name] = currentPreset;
  card.querySelector(`#formula-${name}`).value = activeFormula(name);

  card.querySelector(`#preset-${name}`).addEventListener("change", (e) => {
    SELECTED_PRESET[name] = e.target.value;
    delete EDITED[name]; // switching presets discards a from-scratch edit -- the box below is about to be repopulated from the preset itself, so keep it in sync rather than silently keeping a stale hand-edit under a new preset's name
    card.querySelector(`#formula-${name}`).value = activeFormula(name);
    renderCardChart(name);
  });
  card.querySelector(`#formula-${name}`).addEventListener("input", (e) => {
    EDITED[name] = e.target.value;
    renderCardChart(name);
  });
  card.querySelector(`#revert-${name}`).addEventListener("click", () => {
    delete EDITED[name];
    card.querySelector(`#formula-${name}`).value = activeFormula(name);
    renderCardChart(name);
  });

  return card;
}

function renderCards() {
  const container = document.getElementById("rate-cards");
  container.innerHTML = "";
  for (const name of Object.keys(RATE_CONFIG.rates)) {
    const toggle = document.getElementById("toggle-" + name);
    if (!toggle || !toggle.checked) continue;
    const card = buildCard(name);
    container.appendChild(card);
    renderCardChart(name);
  }
}

function buildToggleList() {
  const container = document.getElementById("rate-toggle");
  container.innerHTML = "";
  for (const name of Object.keys(RATE_CONFIG.rates)) {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" id="toggle-${name}" checked> ${name}`;
    container.appendChild(label);
    label.querySelector("input").addEventListener("change", renderCards);
  }
  document.getElementById("rates-all").addEventListener("click", () => {
    for (const name of Object.keys(RATE_CONFIG.rates)) document.getElementById("toggle-" + name).checked = true;
    renderCards();
  });
  document.getElementById("rates-none").addEventListener("click", () => {
    for (const name of Object.keys(RATE_CONFIG.rates)) document.getElementById("toggle-" + name).checked = false;
    renderCards();
  });
  document.getElementById("rates-reset").addEventListener("click", () => {
    for (const name of Object.keys(RATE_CONFIG.rates)) delete EDITED[name];
    renderCards();
  });
}

function exportJson() {
  const out = { network: RATE_CONFIG.title, reactions: {} };
  for (const name of Object.keys(RATE_CONFIG.rates)) {
    const toggle = document.getElementById("toggle-" + name);
    out.reactions[name] = {
      selected: !!(toggle && toggle.checked),
      formula: activeFormula(name),
      preset: SELECTED_PRESET[name] || null,
    };
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = (RATE_CONFIG.title || "reaction-rates").replace(/\s+/g, "_") + "_rates.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function importJson(file) {
  const reader = new FileReader();
  reader.onload = () => {
    let data;
    try {
      data = JSON.parse(reader.result);
    } catch (e) {
      alert("Could not parse that file as JSON: " + e.message);
      return;
    }
    for (const [name, entry] of Object.entries(data.reactions || {})) {
      if (!RATE_CONFIG.rates[name]) continue; // this network doesn't have that reaction -- skip rather than fail the whole import
      const toggle = document.getElementById("toggle-" + name);
      if (toggle) toggle.checked = !!entry.selected;
      if (entry.preset) SELECTED_PRESET[name] = entry.preset;
      const rate = RATE_CONFIG.rates[name];
      const defaultNow = (entry.preset && rate.presets && rate.presets[entry.preset]) || rate.formula;
      if (entry.formula !== undefined && entry.formula !== defaultNow) EDITED[name] = entry.formula;
      else delete EDITED[name];
    }
    renderCards();
  };
  reader.readAsText(file);
}

function initRatesPage(config) {
  RATE_CONFIG = config;
  buildToggleList();
  renderCards();
  document.getElementById("rates-export").addEventListener("click", exportJson);
  document.getElementById("rates-import").addEventListener("change", (e) => {
    if (e.target.files[0]) importJson(e.target.files[0]);
  });
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderCards);
  }
}
