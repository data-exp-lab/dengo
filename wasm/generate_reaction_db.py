#!/usr/bin/env python3
"""Prototype: export the *full* primordial reaction network as one JSON
"reaction database" -- every species and every reaction dengo's
primordial chemistry already knows about (build_primordial(), the same
network the "Primordial H/He/H2" fiducial widget uses), rather than one
fixed, pre-selected subset of it.

This is the data half of a "pick species/reactions from a checklist,
get a running solver, with no per-selection recompile" prototype (see
NOTES.md): a generic, hand-written mass-action-kinetics RHS/Jacobian
assembler (generic_kinetics.js) can drive *any* subset of the reactions
exported here directly from this data, entirely in JS -- no sympy, no
per-selection C++ codegen, no em++ invocation ever, for any subset a
user picks. Only the (already-generic, already-compiled-once) Newton
integrator (wasm/generic_solver/dengo_generic.cpp) still needs
Emscripten, and it never needs rebuilding when the selection changes.

Also exports every cooling action whose sympy equation is fully self-
contained once species/T/its own rate tables are supplied (see
`export_cooling_action()` below) -- unlike reactions, a cooling
action's equation genuinely is bespoke per-action math (not one
universal formula), so each one's equation is lowered *once*, here, at
export time via sympy's own `jscode` printer into a small JS expression
string embedded in the JSON -- still no per-*selection* codegen (every
action in the catalog is lowered regardless of what a user later
checks), and no em++/compile step involved either way, `new Function()`
is just JS's own built-in "make a callable from a source string"
primitive. Two of the network's seventeen cooling actions
(`gloverabel08`, `cie_cooling`) reference symbols that are *not*
resolvable from their own equation tree -- dengo's C codegen only
resolves them via hand-written surrounding C (a critical-density/
optical-depth-approximation formula each), not from the symbolic
equation alone -- so they're detected (not hardcoded by name) and
skipped; see export_cooling_action()'s docstring.

Run from anywhere:  python3 wasm/generate_reaction_db.py [output_dir]
"""
import json
import os
import sys

import sympy
from sympy.printing import jscode

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fiducial_networks import FIDUCIAL_NETWORKS, build_primordial  # noqa: E402
from dengo.chemistry_constants import kboltz, mh  # noqa: E402

# The compiled solvers' own rate tables use 1024 T-bins (see
# ChemicalNetwork.init_temperature()'s default and _write_solver_tables_
# bin()) -- that's needless precision for a hand-interpolated, exported-
# as-JSON table meant for a fixed-T (not integrated-over-T) demo widget,
# and it isn't free: 22 reactions x 1024 float64s is ~180KB before JSON
# text overhead even doubles it. Downsampled from the network's own
# already-validated log-spaced grid (not re-derived), rather than
# re-running init_temperature() at a coarser resolution and re-evaluating
# every rate function against a different T array than the one the
# compiled fiducial widget was actually checked against.
DOWNSAMPLE = 8


def export_cooling_action(name, action, network, downsample):
    """Lowers one CoolingAction's sympy equation to a JS expression
    string, or returns None if it can't be (see module docstring).

    `action.equation` is already fully resolved against species symbols
    and this action's own "temporaries" (dengo's own substitution, not
    reimplemented here) -- what's left in it should be only: species
    symbols, `T`, `z`, and each of this action's own rate-table symbols
    (an ugly compound name like "ceHI_ceHI[i]", `ReactionCoefficient`'s
    C-array-indexing convention -- renamed here to a plain scalar
    variable per table before printing, since this prototype
    interpolates each table to one scalar at the current T, like
    reactions' own activeRatesAtT(), rather than keeping a live array).

    `ReactionCoefficient.free_symbols` (reaction_classes.py) always
    forces `ge` into the set regardless of whether the expression
    actually involves it (dengo's own machinery for symbolically
    differentiating a rate coefficient w.r.t. energy) -- harmless here
    (this prototype never asks jscode to print a bare "ge"), so it's
    just always allowed rather than tripping the "unresolvable" check
    below. `z` (redshift; only `compton` uses it) is substituted to 0
    before printing -- this prototype, like the compiled widget it sits
    next to, always runs cooling at z=0.

    Any symbol left over after that isn't a species/T/table/ge/z is a
    sign this action's equation isn't self-contained -- dengo's own C
    codegen resolves it from hand-written surrounding code (a specific
    critical-density or optical-depth-approximation formula), not from
    the equation tree, so there's nothing to lower here. Detected
    generically (whatever's left in `eq.free_symbols`), not hardcoded
    by action name, so this keeps working correctly if primordial_
    cooling.py's own set of actions ever changes.
    """
    eq = action.equation
    eq = eq.subs(sympy.Symbol("z"), 0)

    tables = {}
    for table_name, sym in action.table_symbols.items():
        clean = "rate_%s" % table_name
        eq = eq.subs(sym, sympy.Symbol(clean))
        rate = action.tables[table_name](network)[::downsample]
        tables[clean] = rate.tolist()

    species_symbols = {s.symbol: s.name for s in network.required_species}
    accounted = set(species_symbols) | {sympy.Symbol("T"), sympy.Symbol("ge")}
    accounted |= {sympy.Symbol(c) for c in tables}
    leftover = eq.free_symbols - accounted
    if leftover:
        print("!!! cooling action %r not exportable (references %s, not "
              "resolvable from its own equation) -- skipped"
              % (name, sorted(str(s) for s in leftover)), file=sys.stderr)
        return None

    species_names = sorted(
        species_symbols[s] for s in eq.free_symbols
        if s in species_symbols and species_symbols[s] != "ge"
    )
    return {
        "name": name,
        "species": species_names,
        "tables": tables,
        "js": jscode(eq),
    }


def build_reaction_db(out_dir):
    """Writes reaction_db.json into out_dir; also called directly from
    generate_site.py's main(), not just this script's own CLI entry
    point below."""
    os.makedirs(out_dir, exist_ok=True)

    network = build_primordial()
    T_grid = network.T[::DOWNSAMPLE].tolist()

    # Reused, not re-authored: the same starting-point fractions the
    # compiled "Primordial H/He/H2" widget defaults to (fiducial_networks.
    # py's own FIDUCIAL_NETWORKS["primordial"]["default_ics"]) -- a
    # species with no entry there (shouldn't happen for anything in the
    # primordial network today) falls back to a trace value. `ge` isn't
    # a chemistry species (no reactions produce/consume it) -- it's
    # tracked separately, as the one energy/temperature variable every
    # cooling action feeds -- so it's excluded here and handled on its
    # own client-side (see generic_kinetics.js's geFromT()/TFromGe()).
    default_ics = FIDUCIAL_NETWORKS["primordial"]["default_ics"]
    species = [
        {"name": s.name, "weight": s.weight, "default_fraction": default_ics.get(s.name, 1e-12)}
        for s in sorted(network.required_species)
        if s.name != "ge"
    ]

    reactions = []
    for name, rxn in sorted(network.reactions.items()):
        rate = rxn.coeff_fn(network)[::DOWNSAMPLE]
        reactions.append({
            "name": name,
            "left": [[n, s.name] for n, s in rxn.left_side],
            "right": [[n, s.name] for n, s in rxn.right_side],
            "rate": rate.tolist(),
        })

    cooling = []
    skipped = []
    for name, action in sorted(network.cooling_actions.items()):
        entry = export_cooling_action(name, action, network, DOWNSAMPLE)
        if entry is None:
            skipped.append(name)
        else:
            cooling.append(entry)

    # `T_index` is a forward-looking no-op today (every reaction here is
    # indexed by T, like every primordial_rates.py rate function) -- it's
    # here so a *future* project (see NOTES.md: CHIANTI ion-by-ion rates
    # are also T-indexed and would slot in with no format change, but
    # reaction_classes.py's photoionization rates are z-/redshift-indexed
    # instead) has an explicit field to key off of, rather than needing
    # to add one retroactively once a non-T-indexed reaction shows up.
    #
    # `constants`/`gamma`: this prototype uses a single, constant
    # monatomic-ideal-gas gamma for the ge<->T conversion regardless of
    # composition -- unlike the compiled solver, which interpolates a
    # T-dependent gamma for H2-bearing gas (roto-vibrational degrees of
    # freedom activating) -- see generic_kinetics.js's geFromT(). A real
    # simplification (H2-heavy gas's heat capacity will read a bit off),
    # not merely a labeling one; flagged here and in NOTES.md/README-
    # generic.md rather than silently assumed.
    db = {
        "T_grid": T_grid, "T_index": "T",
        "constants": {"kboltz": kboltz, "mh": mh, "gamma": 5.0 / 3.0},
        "species": species, "reactions": reactions, "cooling": cooling,
    }
    ofn = os.path.join(out_dir, "reaction_db.json")
    with open(ofn, "w") as f:
        json.dump(db, f)
    print("wrote %s (%d species, %d reactions, %d cooling actions [%d skipped: %s], %d T points)"
          % (ofn, len(species), len(reactions), len(cooling), len(skipped), skipped, len(T_grid)))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_site", "generic")
    build_reaction_db(out_dir)


if __name__ == "__main__":
    main()
