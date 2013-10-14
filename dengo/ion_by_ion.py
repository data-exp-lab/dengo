"""
Author: Devin Silvia <devin.silvia@gmail.com>
Affiliation: Michigan State University
Homepage: https://bitbucket.org/MatthewTurk/dengo
License:
  Copyright (C) 2013 Devin Silvia.  All Rights Reserved.

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

from reaction_classes import Species, chianti_rate, \
                             ion_cooling_rate, \
                             ion_photoionization_rate, \
                             ion_photoheating_rate, \
                             species_registry
import docutils.utils.roman as roman


def ion_by_ion_rates(atomicSymbol, atomicNumber,
                     atomicWeight, photo_background=None):
    nIons = atomicNumber + 1

    for i in range(nIons):
        ion_state = i + 1
        speciesName = "%s%s" %(atomicSymbol, roman.toRoman(ion_state))
        # Check if the species already exists
        # in the species registry, if it does
        # we don't want to create it again
        if speciesName not in species_registry:
            s = Species(speciesName, atomicNumber, atomicWeight, i)
        else:
            s = species_registry[speciesName]

        if ion_state != nIons:
            # we need to do this to make sure the 'ion_state + 1' species
            # exists when chianti_rate is called
            speciesNamePlusOne = "%s%s" % (atomicSymbol,
                                           roman.toRoman(ion_state+1))
            if speciesNamePlusOne not in species_registry:
                splusone = Species(speciesNamePlusOne, atomicNumber,
                                   atomicWeight, i+1)

        chianti_rate(s)
        if ion_state != nIons:
            if photo_background != None:
                ion_photoionization_rate(s, photo_background='HM12')

def ion_by_ion_cooling(atomicSymbol, atomicNumber,
                       atomicWeight, photo_background=None):
    nIons = atomicNumber + 1

    for i in range(nIons):
        ion_state = i + 1
        speciesName = "%s%s" %(atomicSymbol, roman.toRoman(ion_state))
        # Check if the species already exists
        # in the species registry, if it does
        # we don't want to create it again
        if (speciesName in species_registry) == False:
            s = Species(speciesName, atomicNumber, atomicWeight, i)
        else:
            s = species_registry[speciesName]
        ion_cooling_rate(s)
        if ion_state != nIons:
            if photo_background != None:
                ion_photoheating_rate(s, photo_background='HM12')

# Generate all the ion-by-ion rates
# Note: all the ones that that "None" for the background
# don't yet have the appropriate tables to allow for
# photo-terms
ion_by_ion_rates('H', 1, 1.00794, photo_background='HM12')
ion_by_ion_rates('He', 2, 4.002602, photo_background='HM12')
ion_by_ion_rates('C', 6, 12.0107, photo_background=None)
ion_by_ion_rates('N', 7, 14.0067, photo_background=None)
ion_by_ion_rates('O', 8, 15.9994, photo_background='HM12')
ion_by_ion_rates('Ne', 10, 20.1797, photo_background=None)
ion_by_ion_rates('Mg', 12, 24.3050, photo_background=None)
ion_by_ion_rates('Si', 14, 28.0855, photo_background=None)
ion_by_ion_rates('S', 16, 32.065, photo_background=None)

# Generate all the ion-by-ion cooling rates
# Note: all the ones that that "None" for the background
# don't yet have the appropriate tables to allow for
# photo-terms
ion_by_ion_cooling('H', 1, 1.00794, photo_background='HM12')
ion_by_ion_cooling('He', 2, 4.002602, photo_background='HM12')
ion_by_ion_cooling('C', 6, 12.0107, photo_background=None)
ion_by_ion_cooling('N', 7, 14.0067, photo_background=None)
ion_by_ion_cooling('O', 8, 15.9994, photo_background='HM12')
ion_by_ion_cooling('Ne', 10, 20.1797, photo_background=None)
ion_by_ion_cooling('Mg', 12, 24.3050, photo_background=None)
ion_by_ion_cooling('Si', 14, 28.0855, photo_background=None)
ion_by_ion_cooling('S', 16, 32.065, photo_background=None)
