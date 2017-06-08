# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:38:48 2016

@author: rich
"""

import sys
import os
import numpy as np
import scipy as sp
import math
import dill
import networkx as nx
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing

#sys.path.append(os.path.expanduser("~/OneDrive/Rich/Software/PythonFoodWebs"))
sys.path.append("..")
from Sloppy.computeHessian import LMHessian
from Sloppy.computeHessian import Jacobian
from FoodWeb.NicheModel import nicheModel
from ATN.atnVarParam import atnVarParam
from ATN.atn import params

var = {'x': True, 'xy': True, 'r': True, 'k': True, 'h': True, 'd': True, 'b0': True}

def getFilename(outbase, i):
    return outbase+"_"+str(i)+".pickle"

def toDisk(obj, name):
    print("Saving to %s"%name)
    f = open(name, 'w')
    dill.dump(obj, f)
    f.close()

def fromDisk(name):
    if os.path.exists(name):
        f = open(name, 'r')
        obj = dill.load(f)
        f.close()
        return obj
    return None

# build niche model network and ATN model
# run and only keep models with near steady-state dynamics
def getATNModel(S, C, activity, thr=1e-10, finalT=5000):
    # detects models with significant fluctuations in higher biomass species
    def getRelChange(b1, b2):
        nonzero = (b1 != 0)
        b1 = b1[nonzero]
        b2 = b2[nonzero]
        return (np.abs(b2 - b1)/np.sqrt(b1)).mean()

    # get a network that has 10 or fewer basal species and at least 24 species after running ATN
    print("Building niche model network")
    done = False
    while not done:
        nw, nn, cc, rr, tl = nicheModel(S, C)
        print("setting up atn")
        p = params(q=0.2, d=0.5, r=1.0, k=5.0)
        model = atnVarParam(nw, tl, p, includeActivity=activity)
        if model.B <= 0.35*S:
            print("Running atn")
            results = model.run(0, finalT, finalT)
            relChg = getRelChange(results[-100][1], results[-1][1])
            if relChg < 0.001:
                if model.removeExtinct(isExtinct = results[-1][1] < thr):
                    print("final S = %d"%model.s)
                    if model.s > 0.8*S:
                        return model
            else:
                print("Discarding model with relative biomass change = %f"%relChg)

# Brown and Sethna use SVD and singular value spectra instead of eigenvalue decomposition due to the
# potential numerical instability of poorly conditioned Hessians (widely separated eigenvalues)
# singular values and eigenvalues are the same for positive definite (all positive eigenvalue) matrices
# and Hessian of cost fn local to cost minimum is positive definite
# eigenvectors are then the rows of Vt (Strang 1976, p243, box 6D)
def svdEigen(mat):
    U, sv, Vt = sp.linalg.svd(mat)
    return sv, Vt

# compared to finite diff approximation, largest LM eigenvalues are close, smaller eigenvalues are not
# the agreement of the stiff (largest) eigenvalues is in agreement with Brown and Sethna's results
# and allows the LM approximation to give useful results
#
# first compute and save to disk the Hessian matrices
def computeLMHessian(model, initB, modelIdx, var, title):
    def computeJacobian(model, initB, modelIdx):
        try:
            (jac, refB) = Jacobian(model, initB, modelIdx)
        except ValueError as e:
            print("Value error computing Jacobian in iteration %d"%modelIdx)
            print(e)
            jac = None
            refB = None
        return (jac, refB)

    def computeHessian(modelIdx, jac, refB, residExp):
        try:
            hess = LMHessian(jac, refB, residExp)
        except ValueError as e:
            print("Value error computing Hessian in iteration %d"%modelIdx)
            print(e)
            hess = None
        return hess

    model.setupVarParams(var)
    (jac, refB) = computeJacobian(model, initB, modelIdx)

    lmhessRel = computeHessian(modelIdx, jac, refB, 1.0) if jac is not None else None
    lmhessAbs = computeHessian(modelIdx, jac, refB, 0.0) if jac is not None else None
    return jac, lmhessRel, lmhessAbs

# compared to finite diff approximation, largest LM eigenvalues are close, smaller eigenvalues are not
# the agreement of the stiff (largest) eigenvalues is in agreement with Brown and Sethna's results
# and allows the LM approximation to give useful results
def analyzeLMHessian(lmhess, modelIdx):
    try:
        eigvalLM, eigvecLM = svdEigen(lmhess)
    except ValueError as e:
        print("Value error in iteration %d"%modelIdx)
        print(e)
        eigvalLM = None
        eigvecLM = None
    return (eigvalLM, eigvecLM)

    # output top eigenvalues and information about the associated eigenvectors
# eigVal are sorted descending
def processEigValVecResults(model, initB, eigVal, eigVec, modelIdx):
    # get relative trophic level
    relTL = (model.tl-model.tl.min())/(model.tl.max()-model.tl.min())
    results = []
    # get node indices descending sort by abundance
    biomassIdx = np.argsort(initB)[::-1]
    relBiomass = initB/initB[biomassIdx[0]]
    # get all eigenvalues that are some fraction of largest eigenvector
    # ellipsoid axis length is sq root of eigenvalue
    eigRatioThr = 0.1
    eigvecThr = 0.1   # minimum relative vector component size
    relEigVal = eigVal/eigVal[0]
    nBigVal = len(relEigVal > eigRatioThr)

    for i in range(len(relEigVal)):
        # for each eigenvalue, get directions (parameters) in eigenvector above
        # some threshold magnitude
        valRatio = relEigVal[i]
        print("Model %d, axis scale: %g"%(modelIdx, np.sqrt(valRatio)))
        vec = eigVec[i]
        vecIndices = np.argsort(np.abs(vec))[::-1]
        maxComponent = np.abs(vec[vecIndices[0]])
        for idx in vecIndices:
            if np.abs(vec[idx])/maxComponent < eigvecThr:
                break
            paramInfo = model.getVarParam(idx)
            # convert nodeid to position in list, accounts for extinctions in original network
            nodeId = model.nw.nodes()[paramInfo[1]]
            nodeIdx = model.keepId[nodeId]
            degr = model.nw.degree()[nodeId]
            res = {"modelIdx": modelIdx,
                   "vecIdx": i,
                   "eigvalRatio": valRatio,
                   "axisRatio": math.sqrt(valRatio),
                   "eigvecComponent": vec[idx],
                   "relComponent": vec[idx]/maxComponent,
                   "param": paramInfo[0],
                   "node": nodeId,
                   "relBiomass": relBiomass[nodeIdx],
                   "TL": model.tl[nodeIdx],
                   "relTL": relTL[nodeIdx],
                   "degree": degr,
                   "outFrac": model.nw.out_degree()[nodeId]/float(degr)
                   }
            #print("Eigvec val: %g, param: %s, node idx: %d, biomass ratio: %g, tl: %g"%(vec[idx], paramInfo[0], nodeIdx, relBiomass[nodeIdx], model.tl[nodeIdx]))
            results.append(res)
    return results, nBigVal

# compared to finite diff approximation, largest LM eigenvalues are close, smaller eigenvalues are not
# the agreement of the stiff (largest) eigenvalues is in agreement with Brown and Sethna's results
# and allows the LM approximation to give useful results
def getLMHessian(model, initB, modelIdx, var, title):
    jac, lmhessRel, lmhessAbs = computeLMHessian(model, initB, modelIdx, var, title)
    return model, initB, title, jac, lmhessRel, lmhessAbs


# compared to finite diff approximation, largest LM eigenvalues are close, smaller eigenvalues are not
# the agreement of the stiff (largest) eigenvalues is in agreement with Brown and Sethna's results
# and allows the LM approximation to give useful results
def runLMAnalysis(model, initB, modelIdx, var, title, jac, lmhessRel, lmhessAbs):
    (eigvalRelLM, eigvecRelLM) = analyzeLMHessian(lmhessRel, modelIdx)
    resultsRel, nBigRel = processEigValVecResults(model, initB, eigvalRelLM, eigvecRelLM, modelIdx) if eigvalRelLM is not None else (None,None)

    (eigvalAbsLM, eigvecAbsLM) = analyzeLMHessian(lmhessAbs, modelIdx)
    resultsAbs, nBigAbs = processEigValVecResults(model, initB, eigvalAbsLM, eigvecAbsLM, modelIdx) if eigvalAbsLM is not None else (None,None)

    return resultsRel, nBigRel, eigvalRelLM, eigvecRelLM, \
           resultsAbs, nBigAbs, eigvalAbsLM, eigvecAbsLM if eigvalRelLM is not None else None

# perform a (numbered) iteration of the model and return a block of results (Hessian matrices etc)
# set up to allow easy parallelization
# results written to (local) disk, returns filename
# filename uses suffix _intermed to indicate intermediate results
def runModel(i, S, C, activity, outbase, finalT):
    print("runModel %d"%i)
    np.random.seed(i)
    while True:
        model = getATNModel(S, C, activity, finalT=finalT)
        wtdMnTL = (model.tl*model.binit).sum()/model.binit.sum()
        modelInfo = {"modelIdx": i,
                          "wtdMnTL": wtdMnTL,
                          "S": model.s,
                          "C": model.C,
                          "B": model.B,
                          "network": [li for li in nx.generate_edgelist(model.nw)],
                          "binit": model.binit.copy()}
        initB = model.binit.copy()
        try:
            results = getLMHessian(model, initB, i, var, "All params")
            if results is not None and results[0] is not None:
                filename = getFilename(outbase+"_intermed", i)
                toDisk((modelInfo,)+results, filename)
                return filename
        except ValueError as e:
            print("Value error in iteration %d"%i)
            print(e)

# read intermediate results (filename uses suffix _intermed) from a (numbered) iteration of the model
# analyze Hessian and return a block of results
# set up to allow easy parallelization
# results written to (local) disk, returns filename
def analyzeHessian(i, outbase):
    print("analyzeModel %d"%i)
    res = fromDisk(getFilename(outbase+"_intermed", i))
    if res is not None:
        modelInfo, model, initB, title, jacRel, lmhessRel, jacAbs, lmhessAbs = res
        res = runLMAnalysis(model, initB, i, var, title, jacRel, lmhessRel, jacAbs, lmhessAbs)
        if res is not None and res[0] is not None:
            filename = getFilename(outbase, i)
            toDisk((modelInfo,)+res, filename)
            return filename
    return None

# read intermediate results (filename uses suffix _intermed) from a (numbered) iteration of the model
# analyze Hessian and return a block of results
# set up to allow easy parallelization
# results written to (local) disk, returns filename
def analyzeJacobian(i, outbase):
    def responseMagnitude(jac):
        return math.sqrt((jac*jac).sum())

    def responseMax(jac):
        return np.max(np.abs(jac))

    res = fromDisk(getFilename(outbase+"_intermed", i))
    if res is not None:
        modelInfo, model, initB, title, jac, lmhessRel, lmhessAbs = res
        return [i, responseMagnitude(jac), responseMax(jac)]
    return [i]

# use joblib to parallelize model iteration loop
# possibly use dask.distributed, integrated with joblib, to distribute across a cluster like AWS EC2
# but currently each iteration is pickled so as to save partial reults in case of a crash
#
# separate computing Hessian from its analysis since Hessian compute is expensive and analysis steps might change
#
def atnParamSens_BuildData(S=30, C=0.15, runParallel=True, nIter=100, outbase="SloppyResults", seed=None, activity=False, finalT=5000):
    seed = 0 if seed is None else seed
    seedRange = range(seed, seed+nIter)
    if runParallel:
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores-1)(delayed(runModel)(i, S, C, activity, outbase, finalT) for i in seedRange)
    else:
        results = [runModel(i, S, C, activity, outbase, finalT) for i in seedRange]
    print("Done processing %d iterations"%len(results))
    return results

def atnParamSens_ProcessHessians(runParallel=True, nIter=100, outbase="SloppyResults", seed=None):
    seed = 0 if seed is None else seed
    seedRange = range(seed, seed+nIter)
    if runParallel:
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores-1)(delayed(analyzeHessian)(i, outbase) for i in seedRange)
    else:
        results = [analyzeHessian(i, outbase) for i in seedRange]
    print("Done processing %d iterations"%len(results))
    return results

# process jacobian matrices to find various measures of system sensitivity to parameter perturbation
#
def atnParamSens_ProcessJacobians(nIter=100, outbase="SloppyResults", seed=None):
    seed = 0 if seed is None else seed
    seedRange = range(seed, seed+nIter)
    cols = ["idx", "respMag", "maxResp"]
    results = [analyzeJacobian(i, outbase) for i in seedRange]
    pd.DataFrame(results, columns=cols).to_csv(outbase+"_ResponseMag.csv", index=None)

def atnParamSens(S=30, C=0.15, runParallel=True, nIter=100, finalT=5000, outbase="SloppyResults", seed=None, activity=False):
    atnParamSens_BuildData(S=S, C=C, runParallel=runParallel, nIter=nIter, outbase=outbase, seed=seed, activity=activity, finalT=finalT)
    atnParamSens_ProcessHessians(runParallel=runParallel, nIter=nIter, outbase=outbase, seed=seed)
    atnParamSens_ProcessJacobians(nIter=nIter, outbase=outbase, seed=seed)