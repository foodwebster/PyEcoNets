# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 15:35:50 2014

@author: rich
"""

import networkx as nx
import numpy as np
from scipy.sparse import identity
from scipy.linalg import inv
#from scipy.sparse import issparse
#from scipy.sparse.linalg import inv as inv_sp

# computes prey-averaged trophic level and returns dict of {nodeId:TLValue}
def computeTL(network):
    if not isinstance(network, nx.DiGraph):
        return None
    n = len(network.node)
    a = nx.adjacency_matrix(network).astype('float64')
    # normalize the adjacency matrix (divide each row by row sum)
    asum = np.array(a.sum(axis=1).T)[0]  # force matrix to array
    asum = [(1.0 if val == 0 else 1./val) for val in asum]    
    b = a.todense()
    for row in range(len(asum)):
        b[row] *= asum[row]
    try: 
        # use normalized matrix b in tl computation
        m = identity(n) - b
        mInv = inv(m)
        tl = mInv.sum(axis=1)
        # return results as {node, tl}
        results = dict(zip(network.nodes(), tl))
        return results
    except:
        return None

# return length and basal spp nodeId(s) of shortest path to basal
def shortestPathToBasal(nw, nodeId):
    def isBasal(nw, nid):
        return nw.out_degree(nid) == 0
    if isBasal(nw, nodeId):
        return [(0, nodeId)]
    else:
        shortest = 0.0 
        pathlens = nx.shortest_path_length(nw, source=nodeId)
        for nid, plen in pathlens.iteritems():
            if isBasal(nw, nid):
                if plen < shortest or shortest == 0.0:
                    shortest = plen
                    nodeIds = [nid]
                elif plen == shortest:
                    nodeIds.append(nid)
        if shortest > 0.0:
            return [(shortest, nid) for nid in nodeIds]
        else:
            return []
            
def computeShortPathTL(network):
    shortest = {}
    for nid in network.nodes():
        sh = shortestPathToBasal(network, nid)
        if len(sh) > 0:
            shortest[nid] = 1.0 + sh[0][0]
        else:   # no connection to basal
            shortest[nid] = 0.0
    return shortest
    
def computeSWTL(network):
    tl = computeTL(network)
    stl = computeShortPathTL(network)
    if tl == None:
        return None
    return {n:(tl[n]+stl[n])/2.0 for n in tl.iterkeys()}
    