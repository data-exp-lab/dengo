"""Species/reaction/cooling registration bookkeeping."""
import pytest

from dengo.reaction_classes import reaction_registry, species_registry
from dengo.chemical_network import ChemicalNetwork
import dengo.primordial_rates as primordial_rates
import dengo.primordial_cooling as primordial_cooling  # noqa: F401

primordial_rates.setup_primordial()


def test_species_registered():
    for name in ("H_1", "H_2", "He_1", "He_2", "He_3", "H_m0",
                 "H2_1", "H2_2", "de", "ge"):
        assert name in species_registry


def test_reaction_registered_with_correct_stoichiometry():
    k01 = reaction_registry["k01"]
    left_names = sorted(s.name for _, s in k01.left_side)
    right_names = sorted(s.name for _, s in k01.right_side)
    assert left_names == ["H_1", "de"]
    assert right_names == ["H_2", "de"]  # de appears with count 2 on the right
    assert k01.net_change("de") == 1  # one electron produced per ionization
    assert k01.net_change("H_1") == -1
    assert k01.net_change("H_2") == 1


def test_add_species_and_reaction():
    network = ChemicalNetwork()
    network.add_species("H_1")
    network.add_species("H_2")
    network.add_species("de")
    network.add_reaction("k01", auto_add=False)
    assert "k01" in network.reactions
    assert {s.name for s in network.required_species} >= {"H_1", "H_2", "de"}


def test_duplicate_reaction_raises():
    network = ChemicalNetwork()
    network.add_species("H_1")
    network.add_species("H_2")
    network.add_species("de")
    network.add_reaction("k01", auto_add=False)
    with pytest.raises(RuntimeError):
        network.add_reaction("k01", auto_add=False)


def test_reaction_requires_species_when_not_auto_adding():
    network = ChemicalNetwork()
    network.add_species("H_1")
    # "de" was never added, so this should fail rather than silently add it
    with pytest.raises(RuntimeError):
        network.add_reaction("k01", auto_add=False)


def test_species_list_is_sorted_and_complete():
    network = ChemicalNetwork()
    network.add_collection(
        species_names=["H_1", "H_2", "de"],
        cooling_names=[],
        reaction_names=["k01"],
    )
    species_list = network.species_list()
    assert species_list == sorted(species_list)
    assert set(species_list) == {"H_1", "H_2", "de", "ge"}
