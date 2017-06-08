# -*- coding: utf-8 -*-

import math
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict

from FoodWeb.TrophicLevel import computeSWTL
from analyzeResults import getIdIdxMap

# get features associated with eigenvectors
#

def getEigenvecFeatures(modelInfo, results):
    def getMean(vals, fname, f, log=False):
        if len(vals) > 0:
            if log:
                f[fname+'Mean'] = np.log10(vals.mean())
            else:
                f[fname+'Mean'] = vals.mean()
        else:
            f[fname+'Mean'] = 0

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
    def getMeanFromAll(nodes, values, fname, f):
        vals = np.array([values[n] for n in nodes])
        if len(vals) > 0:
            f[fname+'Mean'] = vals.mean()
        else:
            f[fname+'Mean'] = 0
    def getMeanTotFromAll(nodes, values, fname, f):
        vals = np.array([values[n] for n in nodes])
        if len(vals) > 0:
            f[fname+'Mean'] = vals.mean()
            f[fname+'Tot'] = vals.sum()
        else:
            f[fname+'Mean'] = 0
            f[fname+'Tot'] = 0
    def getShortestPathLen(nw, n1, n2):
        try:
            pl1 = nx.shortest_path_length(nw, n1, n2)
        except nx.NetworkXNoPath:
            pl1 = float('inf')
        try:
            pl2 = nx.shortest_path_length(nw, n2, n1)
        except nx.NetworkXNoPath:
            pl2 = float('inf')
        return min(pl1, pl2)
        
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
        f = {}
        vec = pd.DataFrame(vecs[vid])
        nodesStr = vec['node'].unique().astype(str)
        pvals = vec['param'].value_counts()
        f['relEigval'] = vec.iloc[0]['eigvalRatio']
        f['nSpecies'] = vec['node'].nunique()
        f['nParam'] = len(pvals)

        # add network property summaries
        getMeanFromAll(nodesStr, indeg, 'indegree', f)
        getMeanFromAll(nodesStr, outdeg, 'outdegree', f)
        getMeanFromAll(nodesStr, swtl, 'SWTL', f)

        # get biomass data for unique nodes
        vec = vec.drop_duplicates('node')
        getMeanTot(np.log10(vec['relBiomass'].values), 'logRelBiomass', f)
        
        # add pairwise node properties
        nodes = vec['node'].values.astype(str)
        relBiomass = vec['relBiomass'].values
        tlDiff = []
        biomassRatio = []
        pathLen = []
        for i in range(len(nodes)):
            n1 = nodes[i]
            for j in range(i+1,len(nodes)):
                n2 = nodes[j]
                tlDiff.append(abs(swtl[n1]-swtl[n2]))
                biomassRatio.append(abs(math.log10(relBiomass[i]/relBiomass[j])))
                pl = getShortestPathLen(nw, n1, n2)
                if pl != float('inf'):
                    pathLen.append(pl)
        getMean(np.array(biomassRatio), 'logBiomassRatio', f)
        getMean(np.array(tlDiff), 'TLDiff', f)
        getMean(np.array(pathLen), 'pathLen', f)
        
        # get properties of neighbors (mean biomass)
        toBiomass = []
        fromBiomass = []
        preys = []
        preds = []
        for i in range(len(nodes)):
            n1 = nodes[i]
            succ = nw.successors(n1)
            pred = nw.predecessors(n1)
            if len(succ) > 0:
                toBiomass.extend(biomass[np.array([idMap[v] for v in succ])].tolist())
                preys.extend(succ)
            if len(pred) > 0:
                fromBiomass.extend(biomass[np.array([idMap[v] for v in pred])].tolist())
                preds.extend(pred)
        getMeanTot(np.array(toBiomass), "preyBiomass", f)
        getMeanTot(np.array(fromBiomass), "predBiomass", f)
        getMeanTotFromAll(preds, indeg, 'predOfPred', f)
        getMeanTotFromAll(preds, outdeg, 'preyOfPred', f)
        getMeanTotFromAll(preys, indeg, 'predOfPrey', f)
        getMeanTotFromAll(preys, outdeg, 'preyOfPrey', f)
        f['params'] = '|'.join(pvals.keys())
        features.append(f)
    return features

def getResultsFeatures(results):
    relf = []
    absf = []
    for (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in results:
        relf.extend(getEigenvecFeatures(modelInfo, resultsRel))
        absf.extend(getEigenvecFeatures(modelInfo, resultsAbs))
    return relf, absf