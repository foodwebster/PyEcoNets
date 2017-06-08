# -*- coding: utf-8 -*-

import sys
sys.path.append("..")
from config import basepath
from Sloppy.analyzeResults import getStaticResults
from Sloppy.plot2DHist import plot2DHistParamVsAttrs
from Sloppy.plot2DHist import plot2DHistParamParam

resultspath = basepath+'/SloppyResults'

res = getStaticResults(resultspath)
#plot2DHistParamVsAttrs(res, savepath=resultspath)
#plot2DHistParamVsAttrs(res, savepath=resultspath, datatype='Consumer')
plot2DHistParamVsAttrs(res, savepath=resultspath, datatype='Producer')

#plot2DHistParamParam(res, savepath=resultspath)
