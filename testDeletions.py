# -*- coding: utf-8 -*-

import sys
from os.path import expanduser
sys.path.append(expanduser("~/OneDrive/Rich/Software/PythonFoodWebs"))

from FoodWeb.deleteSpecies import deleteSpecies
from FoodWeb.fromFile import foodwebFromFile

fwFile = "smallWeb.txt"

fw = foodwebFromFile(fwFile)

deleteSpecies('2', fw)