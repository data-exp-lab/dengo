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

import types
from .reaction_classes import reaction_registry, species_registry

def ensure_reaction(rxn):
    if isinstance(rxn, types.StringTypes):
        rxn = reaction_registry[rxn]
    return rxn

def ensure_species(s):
    if isinstance(s, types.StringTypes):
        s = species_registry[rxn]
    return s