# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle

import pandas as pd
import statsmodels.formula.api as sm

# map node ids to indices
def getIdIdxMap(nw):
    ids = np.array(nw.nodes(), dtype=int)
    ids.sort()
    idx = np.arange(len(ids))
    return dict(zip(ids.astype(str), idx))

def getResults(resultspath, count, prefix='Results_', delta=1):
    files = [resultspath+'/'+p for p in os.listdir(resultspath) if p.startswith(prefix) and p.split('_')[1].split('.')[0].isdigit()]

    # results is a tuple of (modelInfo, resultsAbs, eigvalAbsLM, eigvecAbsLM, resultsRel, eigvalRelLM, eigvecRelLM)
    def loadFile(fname):
        f = open(fname)
        results = pickle.load(f)
        f.close()
        return results

    results = []
    idx = 0
    nFiles = len(files)
    for i in range(nFiles):
        if i%5 == 0:
            print("Getting results for %d of %d"%(i, nFiles))
        fname = files[idx]
        idx += delta
        count -= 1
        results.append(loadFile(fname))
        if count == 0 or idx >= nFiles:
            break
    return results

def getStaticResults(basepath, count=0, delta=1):
    plotPrefix = ""
    resultspath = basepath+'/StaticResults'
    prefix = 'Results_'
    results = getResults(resultspath, prefix, count, delta)
    return results, plotPrefix


def getTimeSeriesResults(basepath, count=0, delta=1):
    resultspath = basepath+'/TemporalResults'
    prefix = 'Results_'
    results = getResults(resultspath, prefix, count, delta)
    plotPrefix = "Time Series"
    return results, plotPrefix


def getActivityResults(basepath, count=0, delta=1):
    resultspath = basepath+'/ActivityResults'
    prefix = 'Results_'
    results = getResults(resultspath, prefix, count, delta)
    plotPrefix = "Activity"
    return results, plotPrefix

def analyzeResponseMagnitude(results):
    def getResponseData(idx, modelInfo, eigval):
        return (idx,
                eigval.max(),
                modelInfo['wtdMnTL'],
                modelInfo['B']/float(modelInfo['S']),
                modelInfo['binit'].sum())

    # rv - response variable, pv - list of predictor variables
    def runResponseModel(df, rv, pv):
        f = rv+'~'+'+'.join(pv)
        return sm.ols(formula=f, data=df).fit()

    relData = []
    absData = []
    cols = ["networkID", "response", "mnTL", "fracB", "totalBiomass"]
    for idx, (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in enumerate(results[0]):
        relData.append(getResponseData(idx, modelInfo, eigvalRel))
        absData.append(getResponseData(idx, modelInfo, eigvalAbs))
    reldf = pd.DataFrame(relData, columns = cols)
    absdf = pd.DataFrame(absData, columns = cols)

    relres = runResponseModel(reldf, cols[1], cols[4:5])
    print(relres.summary())
    absres = runResponseModel(absdf, cols[1], cols[4:5])
    print(absres.summary())
