# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:38:48 2016

@author: rich
"""

import numpy as np
import scipy as sp
import pandas as pd
import collections

from computeHessian import finiteDiffHessian
from computeHessian import LMHessian
from computeHessian import toDisk
from computeHessian import fromDisk
from computeHessian import plotEigenvalues
from computeHessian import plotEigenvectors
from computeHessian import compareEigvals
from computeHessian import compareEigvecs

from NicheModel import nicheModel
from atnVarParam import atnVarParam
from atn import params

np.random.seed(1)

def getATNModel(S=20, C=0.15):
    # get a network that has 10 or fewer basal species and at least 24 species after running ATN
    print("Building niche model network")
    done = False
    while not done:
        nw, nn, cc, rr, tl = nicheModel(S, C)
        print("setting up atn")
        p = params(q=0.2, d=0.5, r=1.0, k=5.0)
        model = atnVarParam(nw, tl, p)
        if model.B <= 0.35*S:
            print("Running atn")
            results = model.run(0, 5000, 5000)
            isExtinct = results[-1][1] == 0
            model.removeExtinct(isExtinct)
            print("final S = %d"%model.s)
            if model.s > 0.8*S:
                # plot time series
                # build dataframe timeseries, index is time
                resultsdf = pd.DataFrame([res[1] for res in results], index=pd.Series([res[0] for res in results]))
                resultsdf.plot(logy=True)  
                return model
                
model = getATNModel()
initB = model.binit.copy()

var = {'x': True, 'xy': True, 'r': True, 'k': True, 'h': True, 'd': True, 'b0': True}
varNoX = {'x': False, 'xy': True, 'r': True, 'k': True, 'h': True, 'd': True, 'b0': True}
varNoXY = {'x': False, 'xy': False, 'r': True, 'k': True, 'h': True, 'd': True, 'b0': True}
    
# Brown and Sethna use SVD and singular value spectra instead of eigenvalue decomposition due to the
# potential numerical instability of poorly conditioned Hessians (widely separated eigenvalues)
# singular values and eigenvalues are the same for positive definite (all positive eigenvalue) matrices
# and Hessian of cost fn local to cost minimum is positive definite
# eigenvectors are then the rows of Vt (Strang 1976, p243, box 6D)
def svdEigen(mat):
    U, sv, Vt = sp.linalg.svd(mat)
    return sv, Vt


# compute hessian and eigenvalues using absolute cost function
hessAbs = finiteDiffHessian(model, initB, residExp=0.0)
toDisk(hessAbs, "hessAbs")
# find eigenvalues
(eigvalAbs, eigvecAbs) = svdEigen(hessAbs)

# compute hessian and eigenvalues using relative cost function
hessRel = finiteDiffHessian(model, initB, residExp=1.0)
toDisk(hessRel, "hessRel")
# find eigenvalues
(eigvalRel, eigvecRel) = svdEigen(hessRel)

# check the eigenvalues
np.allclose(hessRel, np.dot(eigvecRel, np.dot(np.diag(eigvalRel), eigvecRel.T)))

lmhessAbs = LMHessian(model, initB, residExp=0.0)
toDisk(lmhessAbs, "lmhessAbs")
#lmhessAbs = fromDisk("lmhessAbs")
(eigvalAbsLM, eigvecAbsLM) = svdEigen(lmhessAbs)

lmhessRel = LMHessian(model, initB, residExp=1.0)
toDisk(lmhessRel, "lmhessRel")
#lmhessRel = fromDisk("lmhessRel")
(eigvalRelLM, eigvecRelLM) = svdEigen(lmhessRel)

#lmhessSqrt = LMHessian(model, initB, residExp=0.5)
#toDisk(lmhessSqrt, "lmhessSqrt")
#lmhessSqrt = fromDisk("lmhessSqrt")
#(eigvalSqrtLM, eigvecSqrtLM) = svdEigen(lmhessSqrt)

# plot eigenvalues and eigenvectors
plotEigenvalues((eigvalAbs, eigvalRel), "Finite Diff")
plotEigenvalues((eigvalAbsLM, eigvalRelLM), "LM Approximation")

plotEigenvectors(eigvecAbs, "Finite Diff, Absolute")
plotEigenvectors(eigvecRel, "Finite Diff, Relative")

plotEigenvectors(eigvecAbsLM, "LM Approx, Absolute")
plotEigenvectors(eigvecRelLM, "LM Approx, Relative")
#plotEigenvectors(eigvecSqrtLM, "LM Approx, Sqrt weighting")

compareEigvals(eigvalAbs, eigvalAbsLM, "Absolute Difference")
compareEigvals(eigvalRel, eigvalRelLM, "Relative Difference")

compareEigvecs(eigvecAbs, eigvecAbsLM, "Absolute Difference")
compareEigvecs(eigvecRel, eigvecRelLM, "Relative Difference")

# compared to finite diff approximation, largest LM eigenvalues are close, smaller eigenvalues are not 
# the agreement of the stiff (largest) eigenvalues is in agreement with Brown and Sethna's results

