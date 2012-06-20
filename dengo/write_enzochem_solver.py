"""
Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: Colubmia University
Homepage: http://yt-project.org/
License:
  Copyright (C) 2011 Matthew Turk.  All Rights Reserved.

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

from chemistry_constants import tiny, kboltz, mh
import jinja2
import h5py
import numpy as na

years = lambda a: a * 365*24*3600

def create_tables(rate_list, cooling_rate_list, solver_name):
    f = h5py.File("%s_rate_tables.h5" % solver_name, "w")
    for name, rate in rate_list.items():
        f.create_dataset("/%s" % name, data = rate.values.astype("float64"))
    f.close()
    f = h5py.File("%s_cooling_tables.h5" % solver_name, "w")
    for name, rate in cooling_rate_list.items():
        f.create_dataset("/%s" % name, data = rate.values.astype("float64"))
    f.close()

def create_cvode_solver(rate_table, reaction_table, species_table,
                        solver_name, cooling_rate_table, cooling_action_table,
                        pp_class):
    # What we are handed here is:
    #   * rate_table, which is a dict of "kXX" to ReactionRate objects.  These
    #     will be used in the reading of rates from disk, but will not directly
    #     be used in the calculation of the derivatives.
    #   * reaction_table, which is a dict of "rXX" to Reaction objects.  These
    #     objects have left_side, right_side, and considered attributes that
    #     describe which species are members of each and how they contribute.
    #     Note that *left_side* and *right_side* are tuples of (n, S), where S
    #     is an instance of Species, but that *considered* is a set of strings.
    #   * species_table, which is a dict of "H2I" (etc) to Species objects.
    #     The Species objects are notable for having information about whether
    #     they are in *equilibrium*, as well as their atomic weight.
    #
    # To utilize these inside our template, we will generate convenience
    # handlers that will explicitly number them.
    ireaction_table = dict([(rid, rname) 
            for rid, rname in enumerate(sorted(reaction_table))])
    reaction_ids = dict([(a, b) for b, a in ireaction_table.items()])
    reaction_varnames = dict([(a, "r_%02i" % i)
            for i, a in enumerate(sorted(reaction_table))])
    irate_table = dict([(rid, rname) 
            for rid, rname in enumerate(sorted(rate_table))])
    rate_ids = dict([(a, b) for b, a in irate_table.items()])
    # Note here that we have three mechanisms for numbering species.  We have
    # our equilibrium species, which are not calculated in the derivative
    # calculation (this should include charge conservation calculations, as
    # well) and we have our actual local variable names.  These variables are
    # local to the cell, so this may give us better cache performance, and it
    # allows us to address things more cleanly.
    #   * non_eq_species_table: maps species name to Species object
    #   * eq_species_table: maps species name to Species object
    #   * species_varnames: maps species name to local variable name
    non_eq_species_table = dict([ (a, b)
            for a, b in species_table.items() if not b.equilibrium
                    and not b.computed])
    non_eq_species_ids = dict([ (a, b)
            for b, a in enumerate(sorted(non_eq_species_table))])
    eq_species_table = dict([ (a, b)
            for a, b in species_table.items() if b.equilibrium 
                    and not b.computed])
    eq_species_ids = dict([ (a, b)
            for b, a in enumerate(sorted(eq_species_table))])
    species_varnames = dict([(a, "s_%02i" % i)
            for i, a in enumerate(sorted(species_table))])
    num_solved_species = len(non_eq_species_table)
    num_total_species = len(species_varnames)
    env = jinja2.Environment(extensions=['jinja2.ext.loopcontrols'],
            loader = jinja2.FileSystemLoader(["cvode_templates/","."]))
    solver_template = env.get_template(
        "cvode_templates/%s_cvode_solver.c.template" % (solver_name))
    # Now the cooling stuff
    icooling_rate_table = dict([(cid, cname) 
            for cid, cname in enumerate(sorted(cooling_rate_table))])
    cooling_rate_ids = dict([(a, b) for b, a in icooling_rate_table.items()])
    #from IPython.Shell import IPShellEmbed
    #IPShellEmbed()()
    template_vars = dict(num_solved_species = num_solved_species,
                         num_total_species = num_total_species,
                         rate_table = rate_table,
                         rate_ids = rate_ids,
                         irate_table = irate_table, 
                         reaction_table = reaction_table,
                         reaction_ids = reaction_ids,
                         reaction_varnames = reaction_varnames,
                         ireaction_table = ireaction_table,
                         solver_name = solver_name,
                         species_table = species_table,
                         non_eq_species_table = non_eq_species_table,
                         non_eq_species_ids = non_eq_species_ids,
                         eq_species_table = eq_species_table,
                         eq_species_ids = eq_species_ids,
                         species_varnames = species_varnames,
                         cooling_rate_table = cooling_rate_table,
                         cooling_rate_ids = cooling_rate_ids,
                         icooling_rate_table = icooling_rate_table,
                         cooling_action_table = cooling_action_table)
    template_vars['pp'] = pp_class(template_vars)
    solver_out = solver_template.render(**template_vars)
    f = open("cvode_templates/%s_cvode_solver.c" % solver_name, "w")
    f.write(solver_out)

def create_initial_conditions(values, solver_name, tfinal):
    f = h5py.File("%s_initial_conditions.h5" % solver_name, "w")
    f["/"].attrs["tfinal"] = tfinal
    for n, v in values.items():
        f.create_dataset("/%s" % n, data=v)
    f.close()

if __name__ == "__main__":
    from primordial_rates import reaction_rates_table, reaction_table, \
        species_table
    from primordial_cooling import cooling_action_table, CVODEPrinter, \
        cooling_rates_table
    #cooling_action_table.pop("gloverabel08")

    NCELLS = 32
    Temperature = 1000
    rho = 1.0e15 # total rho in amu/cc
    X = 1e-4 # ionization fraction
    fH2 = 0.01 # h2 fraction

    # This is the initial fraction of every species
    fracs = dict(HI    = 1.0 - X - fH2,
                 HII   = X,
                 HeI   = tiny,
                 HeII  = tiny,
                 HeIII = tiny,
                 HM    = tiny,
                 de    = X,
                 H2I   = fH2,
                 H2II  = tiny)
    values = dict( [(n, rho*v*na.ones(NCELLS, dtype='float64'))
                    for n, v in fracs.items()] )
    # Approximate, does not include H2
    number_density = sum([values[n] / species_table[n].weight for n in values])
    values['ge'] = ( (Temperature * number_density * kboltz)
                   / (rho * mh * (5.0/3.0 - 1))) # gamma ~ 5/3
    print values['ge']
    #loki = raw_input("enter?")
    values['rho'] = na.ones(NCELLS, dtype='float64')*rho
    #ee = rho * 
    tdyn = na.sqrt(3.0*na.pi / (6.67e-8*1.67e-24*rho))
    create_initial_conditions(values, "primordial", tdyn*10.0)

    create_tables(reaction_rates_table, cooling_rates_table, "primordial")
    create_cvode_solver(reaction_rates_table, reaction_table, species_table,
                        "primordial",
                        cooling_rates_table,
                        cooling_action_table, CVODEPrinter)