# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:56:21 2016

@author: richard.williams

Plot results of ATN parameter sensitivity analysis
"""

import sys
sys.path.append("..")
from config import basepath
from Sloppy.plotResults import plotResults
from Sloppy.analyzeResults import getStaticResults
#from Sloppy.analyzeResults import getTimeSeriesResults
#from Sloppy.analyzeResults import propertyDistrSignificance

resultspath = basepath+'/SloppyResults'

res = getStaticResults(resultspath, count=10)   # delete count or count=0 to get all data
plotResults(res, respath=resultspath)
#propertyDistrSignificance(res, savepath=resultspath)

#plotResults(getTimeSeriesResults())

