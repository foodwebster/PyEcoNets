# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 12:52:24 2016

@author: richard.williams
"""

import sys
import os
sys.path.append("..")
from config import basepath
from config import staticResultsPath

from Sloppy.atnParamSens import atnParamSens

from Sloppy.atnParamSens import atnParamSens_Analyze
from Sloppy.atnParamSens import atnParamSens_ProcessJacobians

# for testing/debugging - smaller systems, less run time, fewer iterations
#resultsFiles = atnParamSens(S=20, C=0.15, runParallel=False, outbase="Test", finalT=1000, nIter=5)

# run some extra as a few get stuck running slowly due to fluctuating dynamics
outbase = os.path.join(staticResultsPath, "Results")
#resultsFiles = atnParamSens(outbase=staticResultsPath, nIter=115, runParallel=True)
#resultsFiles = atnParamSens(outbase=basepath+"/SloppyResults/ActivityResults/Results", activity=True, nIter=110)

#atnParamSens_Analyze(runParallel=False, nIter=115, outbase=outbase, seed=None)
atnParamSens_ProcessJacobians(nIter=115, outbase=outbase, seed=None)
####################################################################
