# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:31:59 2016

@author: richard.williams
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import math
import matplotlib.pyplot as plt

from drawFoodweb import drawFoodweb
from analyzeResults import getStaticResults
from TrophicLevel import computeSWTL

results = getStaticResults(count=5, delta=5)[0]
vecThr = 1.0

ii = 0
for (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in results:
    def draw(nw, results, ax, title):
        vecs = defaultdict(list)
        for res in results:
            vecs[res['vecIdx']].append(res)
        vecid = vecs.keys()
        vecid.sort()
        allNodes = []
        allParam = []
        for vid in vecid:
            if vecs[vid][0]['axisRatio'] < vecThr:
                break
            vec = pd.DataFrame(vecs[vid])
            allNodes.extend(vec['node'].astype(str))
            allParam.extend(vec['param'])
        labels = defaultdict(list)
        for nid, p in zip(allNodes, allParam):
            labels[nid].append(p)
        labels = {nid:','.join(p) for nid,p in labels.iteritems()}
        allNodes = labels.keys()
        highlight = np.zeros(nNodes, dtype=bool)
        colors = ['0.5']*nNodes
        for nid in allNodes:
            idx = idMap[nid]
            colors[idx] = 'red'
            highlight[idx] = True
        drawFoodweb(nw, tl, ax=ax, szAttr=bId, node_color=colors, highlight=highlight, labels=labels, title=title)
    
    ii += 1
    nw = nx.parse_edgelist(modelInfo['network'], create_using=nx.DiGraph())
    nNodes = nw.number_of_nodes()
    nodes = nw.nodes()
    tl = computeSWTL(nw)
    idMap = {}
    for i in range(nNodes):
        idMap[nodes[i]] = i
    # build biomass map
    biomass = modelInfo['binit']
    biomass /= biomass.max()
    biomass = np.clip(biomass, 1e-5, 1.0)
    bId = {nid:math.log10(biomass[idMap[nid]]) for nid in nw.nodes()}
    # extract eigenvector components from the results list
    # and get nodes of large vectors
    
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(hspace=0.1)
    ax1 = plt.subplot(1, 2, 1)
    draw(nw, resultsRel, ax1, "Relative " + str(ii))
    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    draw(nw, resultsAbs, ax2, "Absolute " + str(ii))
