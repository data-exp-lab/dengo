"""
Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: Columbia University
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2012 Matthew Turk.  All Rights Reserved.

  This file is part of the dengo package.

  This file is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as na
from chemistry_constants import tevk, tiny, mh
from .reaction_classes import reaction_registry, cooling_registry, \
    count_m, index_i, species_registry
import types
import sympy
from sympy.printing import ccode

class ChemicalNetwork(object):

    energy_term = None

    def __init__(self):
        self.reactions = {}
        self.cooling_actions = {}
        self.required_species = set([])

    def add_reaction(self, reaction):
        if isinstance(reaction, types.StringTypes):
            reaction = reaction_registry[reaction]
        if reaction.name in self.reactions:
            raise RuntimeError
        self.reactions[reaction.name] = reaction
        
        for n, s in reaction.left_side:
            self.required_species.add(s)
        for n, s in reaction.right_side:
            self.required_species.add(s)
        print "Adding reaction: %s" % reaction

    def add_cooling(self, cooling_term):
        if self.energy_term is None:
            self.energy_term = species_registry["ge"]
            self.required_species.add(self.energy_term)
        if cooling_term.name in self.cooling_actions:
            raise RuntimeError
        self.cooling_actions[cooling_term.name] = cooling_term
        self.required_species.update(cooling_term.species)
    
    def init_temperature(self, T_bounds = (1, 1e8), n_bins=1024):
        self.n_bins = 1024
        self.T = na.logspace(na.log(T_bounds[0]),
                             na.log(T_bounds[1]),
                             n_bins, base = na.e)
        self.logT = na.log(self.T)
        self.tev = self.T / tevk
        self.logtev = na.log(self.tev)
        self.T_bounds = T_bounds

    def species_total(self, species):
        eq = sympy.sympify("0")
        for rn, rxn in sorted(self.reactions.items()):
            rxn.update(self.energy_term.symbol)
            eq += rxn.species_equation(species)
        return eq

    def species_reactions(self, species):
        tr = []
        for rn, rxn in sorted(self.reactions.items()):
            if species in rxn: tr.append(rxn)
        return tr

    def species_list(self):
        species_list = []
        for s in sorted(self.required_species):
            species_list.append(s.name)
        return species_list

    def __iter__(self):
        for rname, rxn in sorted(self.reactions.items()):
            yield rxn

    def print_ccode(self, species, assign_to = None):
        #assign_to = sympy.IndexedBase("d_%s" % species.name, (count_m,))
        if assign_to is None: assign_to = sympy.Symbol("d_%s[i]" % species.name)
        return ccode(self.species_total(species), assign_to = assign_to)

    def print_jacobian_component(self, s1, s2, assign_to = None):
        st = self.species_total(s1)
        if assign_to is None:
            assign_to = sympy.Symbol("d_%s_%s" % (s1.name, s2.name))
        if isinstance(st, (list, tuple)):
            codes = []
            for temp_name, temp_eq in st[0]:
                teq = sympy.sympify(temp_eq)
                codes.append(ccode(teq, assign_to = temp_name))
            codes.append(ccode(st[1], assign_to = assign_to))
            return "\n".join(codes)
        return ccode(sympy.diff(st, s2.symbol), assign_to = assign_to)

    def print_number_density(self):
        eq = sympy.sympify("0")
        for s in sorted(self.required_species):
            if s.name != 'ge': 
                eq += (1.0/s.weight * sympy.sympify(s.name))
        return ccode(eq)

    def species_gamma(self, species):
        if species.name == 'H2I' or species.name == 'H2II':
            gamma = sympy.Symbol('gammaH2')
        else:
            gamma = sympy.Symbol('gamma')
        return gamma

    def gamma_factor(self):
        eq = sympy.sympify("0")
        for s in sorted(self.required_species):
            if s.name != 'ge':
                eq += (1.0/s.weight * sympy.sympify(s.name)) / \
                    (self.species_gamma(s) - 1.0)
        return eq

    def temperature_calculation(self, derivative=False, assign_to = None):
        # If derivative=True, returns the derivative of
        # temperature with respect to ge.  Otherwise,
        # returns just the temperature function
        ge = sympy.Symbol('ge')
        function_eq = (sympy.Symbol('density') * ge) / \
            (sympy.Symbol('kb') * (self.gamma_factor()))
        deriv_eq = sympy.diff(function_eq, ge)
        if derivative == True:
            eq = deriv_eq
            if assign_to == None:
                assign_to = sympy.Symbol("d_T_ge")
        else:
            eq = function_eq
            if assign_to == None:
                assign_to = sympy.Symbol("temperature")
        return ccode(eq, assign_to = assign_to)
