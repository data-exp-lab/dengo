"""Hand-transcribed Vega-expression-syntax versions of every reaction rate
coefficient in src/dengo/primordial_rates.py, for the browser reaction-rate
viewer (see rates.js/generate_rates_page.py).

Why transcribed rather than derived automatically: dengo's rate functions
are raw numpy code (branches via boolean-array masking), not symbolic
expressions -- there's no automatic path from "numpy function" to "Vega
expression string" to walk. Each formula below was copied from
primordial_rates.py by hand and then verified numerically against the
*actual* dengo rate functions (not just re-read for a second opinion) --
evaluated both the real Python coeff_fn and this file's transcription at
T = 10, 50, 300, 1000, 5000, 3.2e4, 1e5, 1e6, 1e7, 1e8 (and, for k13/k22,
at all 6 threebody options each) and confirmed relative error < 1e-6
everywhere. See NOTES.md for the verification script.

Variables available to every formula string (matching primordial_rates.py's
`state.T`/`state.tev`/`state.logtev`/`state.logT` exactly): T (Kelvin),
tev (= T/11605), logtev (= ln(tev)), logT (= ln(T)) -- note these are
natural logs, not log10, matching dengo's own convention (confirmed by
primordial_rates.py's k09, which explicitly divides state.logT by ln(10)
to recover log10(T)).

`presets`, when present, are alternative formulas for the same reaction
selectable in the UI (a dropdown with only one entry -- "default" -- for
every other reaction, per the intended consistent UI). k13/k22 are the
only two reactions with real alternatives in the current codebase: both
are gated by a single network-wide `state.threebody` integer (0-5) in
primordial_rates.py, a convention borrowed from grackle with no
per-option literature citation recorded anywhere in dengo -- so these are
labeled by that same index ("threebody=0" etc.) rather than inventing
citations that aren't actually in the source. `default_preset` matches
ChemicalNetwork's own default (self.threebody = 4, chemical_network.py).
"""

REACTION_RATES = {
    "k01": {
        "equation": "H + e⁻ → H⁺ + 2e⁻",
        "source": "Abel 1999",
        "formula": "max(exp(-32.71396786375 + 13.53655609057*datum.logtev - 5.739328757388*pow(datum.logtev,2) + 1.563154982022*pow(datum.logtev,3) - 0.2877056004391*pow(datum.logtev,4) + 0.03482559773736999*pow(datum.logtev,5) - 0.00263197617559*pow(datum.logtev,6) + 0.0001119543953861*pow(datum.logtev,7) - 2.039149852002e-6*pow(datum.logtev,8)), 1e-20)",
    },
    "k02": {
        "equation": "H⁺ + e⁻ → H",
        "source": None,
        "formula": "datum.T > 5500 ? exp(-28.61303380689232 - 0.7241125657826851*datum.logtev - 0.02026044731984691*pow(datum.logtev,2) - 0.002380861877349834*pow(datum.logtev,3) - 0.0003212605213188796*pow(datum.logtev,4) - 0.00001421502914054107*pow(datum.logtev,5) + 4.989108920299513e-6*pow(datum.logtev,6) + 5.755614137575758e-7*pow(datum.logtev,7) - 1.856767039775261e-8*pow(datum.logtev,8) - 3.071135243196595e-9*pow(datum.logtev,9)) : 3.92e-13*pow(datum.tev,-0.6353)",
    },
    "k03": {
        "equation": "He + e⁻ → He⁺ + 2e⁻",
        "source": None,
        "formula": "datum.tev > 0.8 ? exp(-44.09864886561001 + 23.91596563469*datum.logtev - 10.75323019821*pow(datum.logtev,2) + 3.058038757198*pow(datum.logtev,3) - 0.5685118909884001*pow(datum.logtev,4) + 0.06795391233790001*pow(datum.logtev,5) - 0.005009056101857001*pow(datum.logtev,6) + 0.0002067236157507*pow(datum.logtev,7) - 3.649161410833e-6*pow(datum.logtev,8)) : 1e-20",
    },
    "k04": {
        "equation": "He⁺ + e⁻ → He",
        "source": None,
        "formula": "datum.tev > 0.8 ? (1.54e-9*(1+0.3/exp(8.099328789667/datum.tev))/(exp(40.49664394833662/datum.tev)*pow(datum.tev,1.5)) + 3.92e-13*pow(datum.tev,-0.6353)) : 3.92e-13*pow(datum.tev,-0.6353)",
    },
    "k05": {
        "equation": "He⁺ + e⁻ → He²⁺ + 2e⁻",
        "source": None,
        "formula": "datum.tev > 0.8 ? exp(-68.71040990212001 + 43.93347632635*datum.logtev - 18.48066993568*pow(datum.logtev,2) + 4.701626486759002*pow(datum.logtev,3) - 0.7692466334492*pow(datum.logtev,4) + 0.08113042097303*pow(datum.logtev,5) - 0.005324020628287001*pow(datum.logtev,6) + 0.0001975705312221*pow(datum.logtev,7) - 3.165581065665e-6*pow(datum.logtev,8)) : 1e-20",
    },
    "k06": {
        "equation": "He²⁺ + e⁻ → He⁺",
        "source": None,
        "formula": "3.36e-10/sqrt(datum.T)/pow(datum.T/1000,0.2)/(1+pow(datum.T/1e6,0.7))",
    },
    "k07": {
        "equation": "H + e⁻ → H⁻",
        "source": "Stancil, Lepp & Dalgarno 1998",
        "formula": "3.0e-16*pow(datum.T/300,0.95)*exp(-datum.T/9320)",
    },
    "k08": {
        "equation": "H⁻ + H → H2 + e⁻",
        "source": "Kreckel et al. 2010",
        "formula": "1.35e-9*(pow(datum.T,9.8493e-2) + 3.2852e-1*pow(datum.T,5.5610e-1) + 2.771e-7*pow(datum.T,2.1826)) / (1 + 6.191e-3*pow(datum.T,1.0461) + 8.9712e-11*pow(datum.T,3.0424) + 3.2576e-14*pow(datum.T,3.7741))",
    },
    "k09": {
        "equation": "H + H⁺ → H2⁺",
        "source": None,
        "formula": "datum.T < 30 ? 2.0e-20*pow(datum.T/30,-0.15) : pow(10, -18.20 - 3.194*min(log(datum.T)/log(10), 4.505149978319906) + 1.786*pow(min(log(datum.T)/log(10), 4.505149978319906),2) - 0.2072*pow(min(log(datum.T)/log(10), 4.505149978319906),3))",
    },
    "k10": {
        "equation": "H2⁺ + H → H2 + H⁺",
        "source": None,
        "formula": "6.0e-10",
    },
    "k11": {
        "equation": "H2 + H⁺ → H2⁺ + H",
        "source": None,
        "formula": "datum.tev > 0.3 ? (exp(-21237.15/datum.T) * (-3.3232183e-07 + 3.3735382e-07*datum.logT - 1.4491368e-07*pow(datum.logT,2) + 3.4172805e-08*pow(datum.logT,3) - 4.7813720e-09*pow(datum.logT,4) + 3.9731542e-10*pow(datum.logT,5) - 1.8171411e-11*pow(datum.logT,6) + 3.5311932e-13*pow(datum.logT,7))) : 1e-20",
    },
    "k12": {
        "equation": "H2 + e⁻ → 2H + e⁻",
        "source": "Trevisan & Tennyson 2002",
        "formula": "datum.tev > 0.3 ? 4.4886e-9*pow(datum.T,0.109127)*exp(-101858.0/datum.T) : 1e-20",
    },
    "k13": {
        "equation": "H2 + H → 3H",
        "source": "Glover 2008 (three-body dissociation channel)",
        "formula": "datum.tev > 0.3 ? 1.0670825e-10*pow(datum.tev,2.012)/(exp(4.463/datum.tev)*pow(1+0.2472*datum.tev,3.512)) : 1e-20",
        "default_preset": "threebody=4",
        "presets": {
            "threebody=0": "datum.tev > 0.3 ? 1.0670825e-10*pow(datum.tev,2.012)/(exp(4.463/datum.tev)*pow(1+0.2472*datum.tev,3.512)) : 1e-20",
            "threebody=1": "1.3e-22*pow(datum.T/300.0,-1.0)",
            "threebody=2": "8.4e-11*pow(datum.T,0.515)*exp(-5.2e4/datum.T)",
            "threebody=3": "(1.38e-4/pow(datum.T,1.025))*exp(-5.2e4/datum.T)",
            "threebody=4": "pow(10, -178.4239 - 68.42243*(log(datum.T)/log(10)) + 43.20243*pow(log(datum.T)/log(10),2) - 4.633167*pow(log(datum.T)/log(10),3) + 69.70086*(log(1+40870.38/datum.T)/log(10)) - (23705.7/datum.T))",
            "threebody=5": "datum.T > 3000.0 ? 2.4e-8*exp(-5.2e4/datum.T) : 2.2e-6*pow(datum.T,-0.565)*exp(-5.2e4/datum.T)",
        },
    },
    "k14": {
        "equation": "H⁻ + e⁻ → H + 2e⁻",
        "source": None,
        "formula": "datum.tev > 0.04 ? exp(-18.01849334273 + 2.360852208681*datum.logtev - 0.2827443061704*pow(datum.logtev,2) + 0.01623316639567*pow(datum.logtev,3) - 0.03365012031362999*pow(datum.logtev,4) + 0.01178329782711*pow(datum.logtev,5) - 0.001656194699504*pow(datum.logtev,6) + 0.0001068275202678*pow(datum.logtev,7) - 2.631285809207e-6*pow(datum.logtev,8)) : 1e-20",
    },
    "k15": {
        "equation": "H⁻ + H → 2H + e⁻",
        "source": None,
        "formula": "datum.tev > 0.1 ? exp(-20.37260896533324 + 1.139449335841631*datum.logtev - 0.1421013521554148*pow(datum.logtev,2) + 0.00846445538663*pow(datum.logtev,3) - 0.0014327641212992*pow(datum.logtev,4) + 0.0002012250284791*pow(datum.logtev,5) + 0.0000866396324309*pow(datum.logtev,6) - 0.00002585009680264*pow(datum.logtev,7) + 2.4555011970392e-6*pow(datum.logtev,8) - 8.06838246118e-8*pow(datum.logtev,9)) : 2.56e-9*pow(datum.tev,1.78186)",
    },
    "k16": {
        "equation": "H⁻ + H⁺ → 2H",
        "source": "Croft et al. 1999",
        "formula": "2.4e-6*(1+datum.T/2.0e4)/sqrt(datum.T)",
    },
    "k17": {
        "equation": "H⁻ + H⁺ → H2⁺ + e⁻",
        "source": None,
        "formula": "datum.T < 1e4 ? 1.0e-8*pow(datum.T,-0.4) : 4.0e-4*pow(datum.T,-1.4)*exp(-15100.0/datum.T)",
    },
    "k18": {
        "equation": "H2⁺ + e⁻ → 2H",
        "source": None,
        "formula": "datum.T > 617 ? 1.32e-6*pow(datum.T,-0.76) : 1.0e-8",
    },
    "k19": {
        "equation": "H2⁺ + H⁻ → H + H2",
        "source": None,
        "formula": "5.0e-7*sqrt(100.0/datum.T)",
    },
    "k21": {
        "equation": "2H + H2 → 2H2",
        "source": None,
        "formula": "2.8e-31*pow(datum.T,-0.6)",
    },
    "k22": {
        "equation": "2H + H → H2 + H",
        "source": "Glover 2008 (three-body formation channel)",
        "formula": "datum.T > 300.0 ? 1.3e-32*pow(datum.T/300.0,-0.38) : 1.3e-32*pow(datum.T/300.0,-1.0)",
        "default_preset": "threebody=4",
        "presets": {
            "threebody=0": "datum.T > 300.0 ? 1.3e-32*pow(datum.T/300.0,-0.38) : 1.3e-32*pow(datum.T/300.0,-1.0)",
            "threebody=1": "5.5e-29/datum.T",
            "threebody=2": "8.8e-33",
            "threebody=3": "1.44e-26/pow(datum.T,1.54)",
            "threebody=4": "7.7e-31/pow(datum.T,0.464)",
            "threebody=5": "(6e-32/pow(datum.T,0.25)) + (2.0e-31/pow(datum.T,0.5))",
        },
    },
    "k23": {
        "equation": "H2 + H2 → 2H + H2",
        "source": None,
        "formula": "(8.125e-8/sqrt(datum.T))*exp(-52000.0/datum.T)*(1.0-exp(-6000.0/datum.T))",
    },
}

