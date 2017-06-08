# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict

from FoodWeb.TrophicLevel import computeSWTL
from analyzeResults import getIdIdxMap

# get features associated with eigenvector components
# 
# each component is a direction (parameter and species id) in the parameter space
# ech component has a weight that is a combination of the eigenvalue and vector component
#
def getCompFeatures(modelInfo, results):
    def getMeanTot(vals, fname, f, log=False):
        if len(vals) > 0:
            if log:
                f[fname+'Mean'] = np.log10(vals.mean())
                f[fname+'Tot'] = np.log10(vals.sum())
            else:
                f[fname+'Mean'] = vals.mean()
                f[fname+'Tot'] = vals.sum()
        else:
            f[fname+'Mean'] = 0
            f[fname+'Tot'] = 0
    def getTot(vals, fname, f, log=False):
        if len(vals) > 0:
            if log:
                f[fname+'Tot'] = np.log10(vals.sum())
            else:
                f[fname+'Tot'] = vals.sum()
        else:
            f[fname+'Tot'] = 0
    def getMean(vals, fname, f, log=False):
        if len(vals) > 0:
            if log:
                f[fname+'Mean'] = np.log10(vals.mean())
            else:
                f[fname+'Mean'] = vals.mean()
        else:
            f[fname+'Mean'] = 0

    def getMeanFromAll(nodes, values, fname, f):
        vals = np.array([values[n] for n in nodes])
        if len(vals) > 0:
            f[fname+'Mean'] = vals.mean()
        else:
            f[fname+'Mean'] = 0
    def getTotFromAll(nodes, values, fname, f):
        vals = np.array([values[n] for n in nodes])
        if len(vals) > 0:
            f[fname+'Tot'] = vals.sum()
        else:
            f[fname+'Tot'] = 0
    def getMeanTotFromAll(nodes, values, fname, f):
        vals = np.array([values[n] for n in nodes])
        if len(vals) > 0:
            f[fname+'Mean'] = vals.mean()
            f[fname+'Tot'] = vals.sum()
        else:
            f[fname+'Mean'] = 0
            f[fname+'Tot'] = 0
    features = []
    nw = nx.parse_edgelist(modelInfo['network'], create_using=nx.DiGraph())
    idMap = getIdIdxMap(nw)
    # compute network properties
    swtl = computeSWTL(nw)
    indeg = nw.in_degree()
    outdeg = nw.out_degree()
    
    # extract eigenvector components from the results list
    vecs = defaultdict(list)
    for res in results:
        vecs[res['vecIdx']].append(res)
    vecid = vecs.keys()
    vecid.sort(reverse=True)
    biomass = modelInfo['binit']
    for vid in vecid:
        vec = pd.DataFrame(vecs[vid])
        for idx, row in vec.iterrows():
            node = str(row.node)
            f = {}
            f['node'] = str("%d_%d"%(row['modelIdx'], row.node))
            f['vecSize'] = row.axisRatio
            f['componentSize'] = np.abs(row.relComponent)
            #f['nSpecies'] = vec['node'].nunique()
            #f['param_'+row.param] = 1
            f['param'] = row.param
            f['indegree'] = indeg[node]
            f['outdegree'] = outdeg[node]
            f['TL'] = swtl[node]
            f['logRelBiomass'] = np.log10(row.relBiomass)
            
            # get properties of neighbors (mean biomass)
            toBiomass = []
            fromBiomass = []
            #neighborBiomass = []
            preys = []
            preds = []
            succ = nw.successors(node)
            pred = nw.predecessors(node)
            #neigh = succ+pred
            if len(succ) > 0:
                toBiomass.extend(biomass[np.array([idMap[v] for v in succ])].tolist())
                preys.extend(succ)
            if len(pred) > 0:
                fromBiomass.extend(biomass[np.array([idMap[v] for v in pred])].tolist())
                preds.extend(pred)
            getMean(np.array(toBiomass), "preyBiomass", f, log=True)
            getMean(np.array(fromBiomass), "predBiomass", f, log=True)
            getTotFromAll(preds, indeg, 'predIndeg', f)
            getTotFromAll(preds, outdeg, 'predOutdeg', f)
            getTotFromAll(preys, indeg, 'preyIndeg', f)
            getTotFromAll(preys, outdeg, 'preyOutdeg', f)

            features.append(f)
    return features

def getComponentFeatures(results):
    relf = []
    absf = []
    for (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in results:
        relf.extend(getCompFeatures(modelInfo, resultsRel))
        absf.extend(getCompFeatures(modelInfo, resultsAbs))
    return relf, absf