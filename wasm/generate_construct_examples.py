#!/usr/bin/env python3
"""Packages each fiducial network (fiducial_networks.py) as a
construct.html-compatible "project" file -- the same
{tool: "generic-construct", T, dtf, cards: [...], species_initial: {...}}
shape construct_ui.js's own exportProject()/importProject() already
read and write -- so the real fiducial chemistry (not a toy example) is
loadable there as a starting point and edited inline, live, the same
way a hand-written project is.

Each reaction's card is not a reimplementation: inspect.getsource()
pulls the *real* Python source straight out of primordial_rates.py's
own `@reaction`-decorated closures -- the exact same code the compiled
per-network widgets run, byte-for-byte, just re-wrapped (decorator
line dropped, function renamed so several cards open side by side
aren't all named "rxn") into a self-contained
Species(...)/def .../Reaction(...) card. Every one of these primordial-
rates functions references only `state.T`/`tev`/`logtev`/`logT`/
`threebody`, `numpy`, and (the only network-wide constant any of them
need -- confirmed by scanning every reaction in all three fiducial
networks via `dis`, not assumed) `tiny` from
dengo.chemistry_constants -- so construct_ui.js's own `_State`
stand-in (collectReactionDb()) needs to provide all five of those, not
just `.T`, for one of these cards (or, for that matter, any hand-
written card that happens to reference the same attributes) to run
there at all.

Run from anywhere:  python3 wasm/generate_construct_examples.py [output_dir]
"""
import inspect
import json
import os
import re
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fiducial_networks import FIDUCIAL_NETWORKS  # noqa: E402

# Matches the compiled widget's own default starting density (see
# generate_site.py's PARAM_SPECS: nH default = 4, i.e. 10^4 cm^-3) --
# FIDUCIAL_NETWORKS' own `default_ics` are *fractions* of this, not
# absolute number densities, the same convention app.js's currentFractions()
# already uses. Total run time similarly matches the widget's own cool-
# mode default (PARAM_SPECS: dtf default = 13, i.e. 10^13 s) -- these
# are real networks, not "really boring" toy rates, so an arbitrarily
# short/small default here would just show a flat line.
DEFAULT_NH = 1e4
DEFAULT_DTF = 1e13

# The one network-wide, non-species/non-T config value a handful of
# reactions (k13, k22 -- the three-body H2 formation/dissociation
# channels) read off `state` directly, gated on ChemicalNetwork's own
# default (chemical_network.py: self.threebody = 4) -- not something
# construct.html has any UI for, so every example card gets this fixed
# default value baked into its own validation state below and its
# generated source is correct only for this choice (same as the
# compiled widget's own unexposed default).
DEFAULT_THREEBODY = 4


def _card_source(name, src, rxn):
    """Turns one reaction's raw inspect.getsource() text (the
    `@reaction(...)`-decorated closure, as found in primordial_rates.py)
    into a self-contained construct.html card: drops the decorator line
    (this reaction's own left/right stoichiometry is reconstructed
    separately, below, from `rxn.left_side`/`right_side` -- the actual
    registered values -- rather than re-parsed out of that decorator
    text), renames the function (every one of these is named `rxn` in
    primordial_rates.py itself -- fine there, where each lives in its
    own closure, but confusing with several cards open side by side
    here), and prepends Species(...) definitions for every species this
    reaction actually touches.
    """
    src = textwrap.dedent(src)
    def_idx = src.index("def ")
    src = src[def_idx:]
    orig_name = re.match(r"def\s+(\w+)\s*\(", src).group(1)
    new_name = "%s_rate" % name
    src = re.sub(r"^def\s+%s\s*\(" % re.escape(orig_name), "def %s(" % new_name, src, count=1)

    species_lines = []
    seen = set()
    for _, sp in list(rxn.left_side) + list(rxn.right_side):
        if sp.name in seen:
            continue
        seen.add(sp.name)
        species_lines.append('%s = Species("%s", %r)' % (sp.name, sp.name, sp.weight))

    def side_repr(side):
        return "[" + ", ".join("(%d, %s)" % (n, sp.name) for n, sp in side) + "]"

    lines = [
        "from dengo.reaction_classes import Species, Reaction",
        "import numpy as np",
        "from dengo.chemistry_constants import tiny",
        "",
        *species_lines,
        "",
        src.rstrip(),
        "",
        'Reaction("%s", %s, %s, %s)' % (name, new_name, side_repr(rxn.left_side), side_repr(rxn.right_side)),
    ]
    return new_name, "\n".join(lines) + "\n"


class _ValidationState:
    """Mirrors construct_ui.js's own `_State` stand-in (collectReactionDb())
    closely enough to exercise every attribute a card might reference --
    used only to validate a generated card runs and matches the real,
    already-registered reaction's own tabulated rate, never shipped
    anywhere itself."""

    pass


def build_example(key, network, raw_sources, out_dir):
    """Writes <out_dir>/<key>.json -- one construct.html project file
    for fiducial network `key`. `raw_sources` is `{(key, name): source
    text}`, pre-extracted (inspect.getsource()) for *every* network
    before this function's own exec()/compile() calls for any of
    them -- necessary, not just tidy: found directly that exec()-ing a
    generated card for one network's reaction, then later calling
    inspect.getsource() again on a *different* network's build of that
    same underlying function (shared reaction registry -- k01 in
    hydrogen_minimal is the identical function object as k01 in
    primordial) raised "OSError: could not get source code" -- some
    interaction between linecache's synthetic-filename bookkeeping for
    the exec()'d card and inspect's own source lookup for the real
    module, not chased further since collecting every real source up
    front, before any exec() happens, sidesteps it entirely (confirmed
    directly)."""
    cfg = FIDUCIAL_NETWORKS[key]
    T_grid = network.T  # the network's own already-configured, already-validated T grid -- reused, not re-derived, for the cross-check below

    cards = []
    skipped = []
    for name, rxn in sorted(network.reactions.items()):
        raw_src = raw_sources[(key, name)]
        new_name, card_src = _card_source(name, raw_src, rxn)

        # Cross-check: the real, already-tabulated rate (rxn.coeff_fn(
        # network), exactly what generate_reaction_db.py's own export
        # and the compiled widget both use) against this card's rate
        # function evaluated the same way construct.html's own
        # collectReactionDb() would call it. Any mismatch means the
        # text transform above broke something -- skip that one
        # reaction's card (reported, not silent) rather than ship a
        # subtly-wrong example, same "detect and skip, don't hardcode"
        # convention export_cooling_action() already uses.
        ns = {}
        exec(compile(card_src, "<construct-example:%s>" % name, "exec"), ns)  # noqa: S102 -- build-time only, own generated source
        state = _ValidationState()
        state.T = T_grid
        state.tev = T_grid / 11605.0
        state.logtev = np.log(state.tev)
        state.logT = np.log(T_grid)
        state.threebody = DEFAULT_THREEBODY
        via_card = np.asarray(ns[new_name](state), dtype=float)
        via_real = np.asarray(rxn.coeff_fn(network), dtype=float)
        relerr = np.abs(via_card - via_real) / np.maximum(np.abs(via_real), 1e-300)
        if not np.all(relerr < 1e-9):
            print("!!! construct.html example for %r/%r not exportable (card doesn't reproduce the "
                  "real registered reaction -- max relative error %.3g) -- skipped"
                  % (key, name, float(np.max(relerr))), file=sys.stderr)
            skipped.append(name)
            continue

        cards.append(card_src)

    default_ics = cfg["default_ics"]
    species_initial = {
        s.name: default_ics.get(s.name, 1e-12) * DEFAULT_NH
        for s in sorted(network.required_species) if s.name != "ge"
    }

    out = {
        "tool": "generic-construct",
        "network": key,
        "title": cfg["title"],
        "T": cfg["default_T"],
        "dtf": DEFAULT_DTF,
        "cards": cards,
        "species_initial": species_initial,
    }
    os.makedirs(out_dir, exist_ok=True)
    ofn = os.path.join(out_dir, "%s.json" % key)
    with open(ofn, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote %s (%d reaction card(s), %d skipped: %s)" % (ofn, len(cards), len(skipped), skipped))


def build_all_examples(out_dir):
    # Phase 1: build every network and extract every reaction's raw
    # source *before* phase 2's exec()/compile() calls for any of them
    # -- see build_example()'s own docstring for why the ordering
    # matters here, not just style.
    networks = {}
    raw_sources = {}
    for key in FIDUCIAL_NETWORKS:
        network = FIDUCIAL_NETWORKS[key]["build"]()
        networks[key] = network
        for name, rxn in sorted(network.reactions.items()):
            raw_sources[(key, name)] = inspect.getsource(rxn.coeff_fn)

    # Phase 2: build + validate + write each example, now that every
    # source has already been safely extracted.
    for key in FIDUCIAL_NETWORKS:
        build_example(key, networks[key], raw_sources, out_dir)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_site", "generic", "examples")
    build_all_examples(out_dir)


if __name__ == "__main__":
    main()
