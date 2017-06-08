# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:38:48 2016

@author: rich
"""

import numpy as np
import scipy as sp

from computeHessian import finiteDiffHessian
from computeHessian import LMHessian
from computeHessian import toDisk
from computeHessian import fromDisk
from computeHessian import plotEigenvectors
from computeHessian import compareMethods

from NicheModel import nicheModel
from atn import atn
from atn import params

np.random.seed(0)

S=20
C=0.15

# get a network that has 10 or fewer basal species and at least 80% species after running ATN
print("Building niche model network")
done = False
while not done:
    nw, nn, cc, rr, tl = nicheModel(S, C)
    print("setting up atn")
    p = params(q=0.2, d=0.5, r=1.0, k=5.0)
    model = atn(nw, tl, p)
    if model.B <= 0.35*S:
        print("Running atn")
        results = model.run(0, 5000, 5000)
        isExtinct = results[-1][1] == 0
        model.removeExtinct(isExtinct)
        initB = model.binit.copy()
        print("final S = %d"%model.s)
        if model.s > 0.8*S:
            done = True

steps = 100
newB = model.run(0, steps, steps, init=initB, threshold=False)[-1][1]
frac = (newB-initB)/initB

var = {'x': True, 'y': True, 'r': True, 'h': True, 'd': True, 'b0': True}
model.setupVarParams(var)
    
hessRel = finiteDiffHessian(model, initB, absDiff=False)
#(eigvalRel, eigvecRelLM) = np.linalg.eigh(hessRel)
lmhessRel = LMHessian(model, initB, absDiff=False)
(eigvalRelLM, eigvecRelLM) = np.linalg.eigh(lmhessRel)

