# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 12:52:24 2016

@author: richard.williams
"""

import sys
sys.path.append("..")
from config import basepath

from Sloppy.atnTemporalParamSens import atnTemporalParamSens

#resultsFiles = atnTemporalParamSens(S=30, C=0.15, runParallel=False, )  # for debugging
resultsFiles = atnTemporalParamSens()

####################################################################
