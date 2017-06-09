# -*- coding: utf-8 -*-

"""
@author: richard.williams

Analyze results of ATN parameter sensitivity analysis
and get a per-model summary of network structure and results
"""

import sys
sys.path.append("..")
from config import staticResultsPath
from Sloppy.analyzeResults import getResults
from Sloppy.resultsSummary import resultsSummary

res = getResults(staticResultsPath, count=100)   # delete count or count=0 to get all data
resultsSummary(res, staticResultsPath)

