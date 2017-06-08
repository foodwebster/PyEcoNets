# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:05:58 2016

@author: rich
"""

import numpy as np
from NicheModel import nicheModel
from TrophicLevel import computeTL
from TrophicLevel import computeShortPathTL

np.random.seed(12)

nw, nn, cc, rr, tl = nicheModel(30, 0.15)
if nw != None:
    tl = computeTL(nw)
    shortTL = computeShortPathTL(nw)
