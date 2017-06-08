# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:28:47 2016

@author: rich
"""

import numpy as np
import pandas as pd

from NicheModel import nicheModel
from atn import atn
from atn import params

nIter = 20
np.random.seed(12)
nExt = []
basal = []   
top = []   
basal = []   
for i in range(nIter):
    nw, nn, cc, rr, tl = nicheModel(30, 0.15)
    if nw == None:
        break
    p = params(q=0.2, d=0.5, r=1.0, k=5.0)
    model = atn(nw, tl, p)
    basal.append(model.B/float(model.s))

    results = model.run(0, 5000, 5000)
    isExtinct = results[-1][1] == 0
    nExt.append(isExtinct.sum())
    print("nBasal = %d, nExtinct = %d"%(model.B, nExt[-1]))

print(sum(nExt)/float(nIter)) 
print(sum(basal)/float(nIter))

# build dataframe timeseries, index is time
resultsdf = pd.DataFrame([res[1] for res in results], index=pd.Series([res[0] for res in results]))
resultsdf.plot(logy=True)  
    #model.removeExtinct(isExtinct)
    #results2 = model.run(0, 1000, 10)
    #isExtinct = results2[-1][1] == 0
    #print("nExtinct2 = %d"%isExtinct.sum())

