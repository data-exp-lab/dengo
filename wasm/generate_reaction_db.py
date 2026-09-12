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

Deliberately narrower than the full compiled solver in one respect:
chemistry only, no thermal (ge) coupling -- every reaction here is
ordinary mass-action kinetics (rate(T) times a product of reactant
number densities), which is generic, but "how does gas temperature
respond to this reaction's heat release" is not -- that's each cooling
action's own bespoke sympy expression, and isn't in scope for this
prototype. Run at a fixed, user-dialed T instead.

Run from anywhere:  python3 wasm/generate_reaction_db.py [output_dir]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fiducial_networks import FIDUCIAL_NETWORKS, build_primordial  # noqa: E402

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
    # primordial network today) falls back to a trace value.
    default_ics = FIDUCIAL_NETWORKS["primordial"]["default_ics"]
    species = [
        {"name": s.name, "weight": s.weight, "default_fraction": default_ics.get(s.name, 1e-12)}
        for s in sorted(network.required_species)
        if s.name != "ge"  # no thermal coupling in this prototype -- see module docstring
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

    # `T_index` is a forward-looking no-op today (every reaction here is
    # indexed by T, like every primordial_rates.py rate function) -- it's
    # here so a *future* project (see NOTES.md: CHIANTI ion-by-ion rates
    # are also T-indexed and would slot in with no format change, but
    # reaction_classes.py's photoionization rates are z-/redshift-indexed
    # instead) has an explicit field to key off of, rather than needing
    # to add one retroactively once a non-T-indexed reaction shows up.
    db = {"T_grid": T_grid, "T_index": "T", "species": species, "reactions": reactions}
    ofn = os.path.join(out_dir, "reaction_db.json")
    with open(ofn, "w") as f:
        json.dump(db, f)
    print("wrote %s (%d species, %d reactions, %d T points)"
          % (ofn, len(species), len(reactions), len(T_grid)))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_site", "generic")
    build_reaction_db(out_dir)


if __name__ == "__main__":
    main()
