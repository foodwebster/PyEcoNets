# -*- coding: utf-8 -*-

"""
@author: richard.williams

Analyze results of ATN parameter sensitivity analysis
"""

import sys
sys.path.append("..")
from config import basepath
from Sloppy.analyzeResults import analyzeResponseMagnitude
from Sloppy.analyzeResults import getStaticResults
from Sloppy.resultsSummary import resultsSummary

resultspath = basepath+'/SloppyResults'

res = getStaticResults(resultspath, count=100)   # delete count or count=0 to get all data

resultsSummary(res)
analyzeResponseMagnitude(res)

