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
from computeHessian import plotEigenvalues
from computeHessian import plotEigenvectors
from computeHessian import compareEigvals
from computeHessian import compareEigvecs

from NicheModel import nicheModel
from atnVarParam import atnVarParam
from atn import params

np.random.seed(0)

S=20
C=0.15

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
        initB = model.binit.copy()
        print("final S = %d"%model.s)
        if model.s > 0.8*S:
            done = True

var = {'x': False, 'xy': True, 'r': True, 'k': True, 'h': True, 'd': True, 'b0': True}
model.setupVarParams(var)
    
# Brown and Sethna use SVD and singular value spectra instead of eigenvalue decomposition due to the
# potential numerical instability of poorly conditioned Hessians (widely separated eigenvalues)
# singular values and eigenvalues are the same for positive definite (all positive eigenvalue) matrices
# and Hessian of cost fn local to cost minimum is positive definite
# eigenvectors are then the rows of Vt (Strang 1976, p243, box 6D)
def svdEigen(mat):
    U, sv, Vt = sp.linalg.svd(mat)
    return sv, Vt

# compute hessian and eigenvalues using absolute cost function
#hessAbs = finiteDiffHessian(model, initB, residExp=0.0)
#toDisk(hessAbs, "hessAbs")
# find eigenvalues
#(eigvalAbs, eigvecAbs) = svdEigen(hessAbs)

# compute hessian and eigenvalues using relative cost function
#hessRel = finiteDiffHessian(model, initB, residExp=1.0)
#toDisk(hessRel, "hessRel")
# find eigenvalues
#(eigvalRel, eigvecRel) = svdEigen(hessRel)

# check the eigenvalues
#np.allclose(hessRel, np.dot(eigvecRel, np.dot(np.diag(eigvalRel), eigvecRel.T)))

lmhessAbs = fromDisk("lmhessAbs")
(eigvalAbsLM, eigvecAbsLM) = svdEigen(lmhessAbs)

lmhessRel = fromDisk("lmhessRel")
(eigvalRelLM, eigvecRelLM) = svdEigen(lmhessRel)

lmhessSqrt = fromDisk("lmhessSqrt")
(eigvalSqrtLM, eigvecSqrtLM) = svdEigen(lmhessSqrt)

# eigVal are sorted descending
def processEigValVecResults(model, initB, eigVal, eigVec):
    # get node indices descended sort by abundance
    biomassIdx = np.argsort(initB)[::-1]
    relBiomass = initB/initB[biomassIdx[0]]
    # get all eigenvalues that are some fraction of largest eigenvector
    # ellipsoid axis length is sq root of eigenvalue
    eigRatioThr = 0.01
    relEigVal = eigVal/eigVal[0]
    relEigVal = relEigVal[relEigVal > eigRatioThr]
    
    for i in range(len(relEigVal)):
        # for each eigenvalue, get directions (parameters) in eigenvector above 
        # some threshold magnitude
        eigvecThr = 0.1
        valRatio = relEigVal[i]
        print("Axis scale: %f"%np.sqrt(valRatio))
        vec = eigVec[i]
        vecIndices = np.argsort(np.abs(vec))[::-1]
        for idx in vecIndices:
            if np.abs(vec[idx]) < eigvecThr:
                break
            paramInfo = model.getVarParam(idx)
            nodeIdx = paramInfo[1]
            print("Eigvec val: %f, param: %s, node idx: %d, biomass ratio: %f"%(vec[idx], paramInfo[0], nodeIdx, relBiomass[nodeIdx]))
            
            
processEigValVecResults(model, initB, eigvalAbsLM, eigvecAbsLM)

processEigValVecResults(model, initB, eigvalRelLM, eigvecRelLM)

processEigValVecResults(model, initB, eigvalSqrtLM, eigvecSqrtLM)