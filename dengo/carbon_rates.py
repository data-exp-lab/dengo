"""
Author: Devin Silvia <devin.silvia@gmail.com>
Affiliation: UC Boulder
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

from reaction_classes import Species, chianti_rate, species_registry

for i in range(7):
    ion_state = i + 1
    speciesName = "c_%s" % ion_state
    # Check if the species already exists
    # in the species registry, if it does
    # we don't want to create it again
    if (speciesName in species_registry) == False:
        s = Species(speciesName, 12, i)
    else:
        s = species_registry[speciesName]

    if ion_state != 7:
        # we need to do this to make sure the 'ion_state + 1' species
        # exists when chianti_rate is called
        speciesNamePlusOne = "c_%s" % (ion_state+1)
        if (speciesNamePlusOne in species_registry) == False:
            splusone = Species(speciesNamePlusOne, 12, i+1)
    chianti_rate(s)