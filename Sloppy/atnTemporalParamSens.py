# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:38:48 2016

@author: rich
"""

import sys
from os.path import expanduser
sys.path.append(expanduser("~/OneDrive/Rich/Software/PythonFoodWebs"))
import numpy as np
import scipy as sp
import math
import pickle
import networkx as nx

from joblib import Parallel, delayed
import multiprocessing

from computeHessian import LMHessian
from NicheModel import nicheModel
from atnVarParam import atnVarParam
from atn import params

var = {'x': True, 'xy': True, 'r': True, 'k': True, 'h': True, 'd': True, 'b0': True}
#varNoX = {'x': False, 'xy': True, 'r': True, 'k': True, 'h': True, 'd': True, 'b0': True}
#varNoXY = {'x': False, 'xy': False, 'r': True, 'k': True, 'h': True, 'd': True, 'b0': True}
    
def getATNModel(S, C):
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
        model = atnVarParam(nw, tl, p)
        if model.B <= 0.35*S:
            print("Running atn")
            results = model.run(0, 5000, 5000)
            relChg = getRelChange(results[-100][1], results[-1][1])
            if relChg < 0.001:
                if model.removeExtinct(isExtinct = results[-1][1] == 0):
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
def runLMAnalysis(model, initB, modelIdx, var, title):
    # output top eigenvalues and information about the associated eigenvectors
    # eigVal are sorted descending
    def processEigValVecResults(model, initB, eigVal, eigVec):
        # get relative trophic level
        relTL = (model.tl-model.tl.min())/(model.tl.max()-model.tl.min())
        results = []
        # get node indices descending sort by abundance
        biomassIdx = np.argsort(initB)[::-1]
        relBiomass = initB/initB[biomassIdx[0]]
        # get all eigenvalues that are some fraction of largest eigenvector
        # ellipsoid axis length is sq root of eigenvalue
        eigRatioThr = 0.1
        relEigVal = eigVal/eigVal[0]
        relEigVal = relEigVal[relEigVal > eigRatioThr]
        
        for i in range(len(relEigVal)):
            # for each eigenvalue, get directions (parameters) in eigenvector above 
            # some threshold magnitude
            eigvecThr = 0.2
            valRatio = relEigVal[i]
            print("Model %d, axis scale: %g"%(modelIdx, np.sqrt(valRatio)))
            vec = eigVec[i]
            vecIndices = np.argsort(np.abs(vec))[::-1]
            for idx in vecIndices:
                if np.abs(vec[idx]) < eigvecThr:
                    break
                paramInfo = model.getVarParam(idx)
                # convert nodeid to position in list, accounts for extinctions in original network
                nodeId = model.nw.nodes()[paramInfo[1]]
                nodeIdx = model.keepId[nodeId]
                degr = model.nw.degree()[nodeId]
                res = {"modelIdx": modelIdx,
                       "eigvalRatio": valRatio,
                       "axisRatio": math.sqrt(valRatio),
                       "eigvecComponent": vec[idx],
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
        return results
        
    # runs model for temporal analysis - 10 equally spaced time intervals over 200 steps
    # initial biomasses are 50% of equilibrium, look at behavior of model as it returns towards equilibrium
    #
    def runAnalysis(model, initB, residExp):
        try:
            (jac, lmhess) = LMHessian(model, initB/2, modelIdx, nTimes=10, steps=200, residExp=residExp)
            (eigvalLM, eigvecLM) = svdEigen(lmhess)
        except ValueError as e:
            print("Value error in iteration %d"%modelIdx)
            print(e)
            eigvalLM = None
            eigvecLM = None
        return (eigvalLM, eigvecLM)

    model.setupVarParams(var)

    (eigvalRelLM, eigvecRelLM) = runAnalysis(model, initB, 1.0)
    if eigvalRelLM is None:
        return None
    resultsRel = processEigValVecResults(model, initB, eigvalRelLM, eigvecRelLM)

    (eigvalAbsLM, eigvecAbsLM) = runAnalysis(model, initB, 0.0)
    if eigvalAbsLM is None:
        return None
    resultsAbs = processEigValVecResults(model, initB, eigvalAbsLM, eigvecAbsLM) if eigvalAbsLM is not None else None

    #processEigValVecResults(model, initB, eigvalSqrtLM, eigvecSqrtLM)
    return resultsRel, eigvalRelLM, eigvecRelLM, resultsAbs, eigvalAbsLM, eigvecAbsLM if eigvalRelLM is not None else None

# perform a (numbered) iteration of the model and return a block of results
# set up to allow easy parallelization
# results written to (local) disk, returns filename
#
def runModel(i, S, C, outbase):
    print("runModel %d"%i)
    np.random.seed(i)
    while True:
        model = getATNModel(S, C)
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
            results = runLMAnalysis(model, initB, i, var, "All params")
            if results is not None:
                filename = outbase+"_"+str(i)+".pickle"
                f = open(filename, "w")
                pickle.dump((modelInfo,)+results, f)
                f.close()
                return filename
        except ValueError as e:
            print("Value error in iteration %d"%i)
            print(e)

# use joblib to parallelize model iteration loop
# possibly use dask.distributed, integrated with joblib, to distribute across a cluster like AWS EC2
# but currently each iteration is pickled so as to save partial reults in case of a crash
#
def atnTemporalParamSens(S=30, C=0.15, runParallel=True, nIter=100, outbase="TemporalSloppyResults", firstIdx=0):
    if runParallel:
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores-1)(delayed(runModel)(i+firstIdx, S, C, outbase) for i in range(nIter))
    else:
        results = [runModel(i+firstIdx, S, C, outbase) for i in range(nIter)]
    print("Done processing %d iterations"%len(results))
    return results

