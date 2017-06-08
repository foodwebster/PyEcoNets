# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:05:58 2016

@author: rich
"""

import numpy as np
from drawFoodweb import drawFoodweb
from NicheModel import nicheModel

nIter = 100
np.random.seed(12)
basal = []   
top = []   

for i in range(nIter):
    nw, nn, cc, rr, tl = nicheModel(30, 0.15)
    if nw == None:
        break
    nTop = len([n for n,d in nw.in_degree_iter() if d == 0])
    nBasal = len([n for n,d in nw.out_degree_iter() if d == 0]) 
    basal.append(nBasal/float(nw.number_of_nodes()))
    top.append(nTop/float(nw.number_of_nodes()))

print("Top: %f Basal: %f"%(sum(top)/float(nIter), sum(basal)/float(nIter))) 
