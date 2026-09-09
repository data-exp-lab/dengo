"""Codegen smoke tests: does the Jinja template render into something that
looks like the network it was given, without needing a C++ compiler.
These are intentionally fast (no `solver_build.build_solver` call)."""
import os
import re

import numpy as np

from conftest import make_hydrogen_network, make_primordial_network


def test_write_cython_solver_creates_expected_files(tmp_path):
    network = make_hydrogen_network()
    network.write_cython_solver("t", output_dir=str(tmp_path))
    for fn in ("t_solver.h", "t_solver.C", "t_solver_run.pyx",
               "t_tables.bin", "BE_chem_solve.C"):
        assert (tmp_path / fn).exists(), fn


def test_generated_source_declares_every_species(tmp_path):
    network = make_primordial_network()
    network.write_cython_solver("t", output_dir=str(tmp_path))
    source = (tmp_path / "t_solver.C").read_text()
    for sp in network.required_species:
        assert ("double %s;" % sp.name) in source


def test_generated_source_references_every_reaction(tmp_path):
    network = make_primordial_network()
    network.write_cython_solver("t", output_dir=str(tmp_path))
    source = (tmp_path / "t_solver.C").read_text()
    for name in network.reactions:
        assert ("rs_%s" % name) in source


def test_tables_bin_size_matches_expected_layout(tmp_path):
    network = make_primordial_network()
    network.write_cython_solver("t", output_dir=str(tmp_path))
    n_T = len(network.T)
    n_reaction_tables = len(network.reactions)
    n_cooling_tables = sum(
        len(action.tables) for action in network.cooling_actions.values()
    )
    n_gamma_tables = 2 * len(network.interpolate_gamma_species)
    expected_doubles = n_T * (n_reaction_tables + n_cooling_tables + n_gamma_tables)
    actual_bytes = os.path.getsize(str(tmp_path / "t_tables.bin"))
    assert actual_bytes == expected_doubles * 8


def test_tables_bin_contents_match_write_order(tmp_path):
    """Regression test: the generated C's fread() order used to silently
    diverge from _write_solver_tables_bin's write order for any pair of
    table names whose case-sensitive and case-insensitive alphabetical
    order disagree (e.g. "ciHI" vs "ciHeI", "gaHI" vs "gaHe") -- Jinja's
    `dictsort`/`sort` filters default to case-insensitive, while Python's
    `sorted()` (used by the writer) is case-sensitive. That silently
    corrupted cooling rates by tens of orders of magnitude with no error
    at all (see NOTES.md, 2026-09-08). This reads the actual bytes back
    with the *exact* order the writer used and checks they still match
    what each reaction/cooling function independently computes -- so any
    reintroduced write/read order mismatch fails loudly here instead.
    """
    network = make_primordial_network()
    network.write_cython_solver("t", output_dir=str(tmp_path))

    raw = np.fromfile(str(tmp_path / "t_tables.bin"), dtype="float64")
    n = len(network.T)
    offset = 0
    for name, rxn in sorted(network.reactions.items()):
        expected = rxn.coeff_fn(network).astype("float64")
        assert np.allclose(raw[offset:offset + n], expected, rtol=1e-10), name
        offset += n
    for name, action in sorted(network.cooling_actions.items()):
        for tab in sorted(action.tables):
            expected = action.tables[tab](network).astype("float64")
            assert np.allclose(raw[offset:offset + n], expected, rtol=1e-10), (name, tab)
            offset += n
    for sp in sorted(network.interpolate_gamma_species):
        expected = network.interpolate_species_gamma(sp).astype("float64")
        assert np.allclose(raw[offset:offset + n], expected, rtol=1e-10), (sp.name, "gamma")
        offset += n
        expected = network.interpolate_species_gamma(sp, deriv=True).astype("float64")
        assert np.allclose(raw[offset:offset + n], expected, rtol=1e-10), (sp.name, "dgamma_dT")
        offset += n
    assert offset == len(raw)

    # Also confirm the *generated C source's* fread order textually
    # matches -- not just the file bytes -- so a future template change
    # that reintroduces a case-insensitive sort is caught even if it
    # happens to still read/write the same total byte count.
    source = (tmp_path / "t_solver.C").read_text()
    read_order = re.findall(r"fread\(data->(\w+),", source)
    expected_order = (
        ["r_%s" % name for name, _ in sorted(network.reactions.items())]
        + [
            "c_%s_%s" % (name, tab)
            for name, action in sorted(network.cooling_actions.items())
            for tab in sorted(action.tables)
        ]
        + [
            name
            for sp in sorted(network.interpolate_gamma_species)
            for name in ("g_gamma%s" % sp.name, "g_dgamma%s_dT" % sp.name)
        ]
    )
    assert read_order == expected_order
