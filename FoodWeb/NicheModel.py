# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 08:11:21 2016

@author: rich
"""

import numpy as np
import networkx as nx
from TrophicLevel import computeTL

def nicheModel(s, c, errfrac=0.05, niter=1000, allowDisconnected=False, singleComponent=True, nonSingular=True, seed=None):
    if seed != None:    
        np.random.seed(seed)    
    l = c*s*s       # expected number of links
    maxerr = errfrac*l
    for iter in range(niter):
        # pick niche values
        nn = np.random.rand(s)
        nn.sort()
        # pick width of feeding range, beta distr with beta=1
        mn = 2.0*c
        beta = 1.0/mn - 1.0
        rr = nn*np.random.beta(1.0, beta, size=s)
        # compute max and min values for center of feeding niche
        cmin = rr/2
        cmax = np.minimum(1.0-rr/2.0, nn)
        # pick feeding niche positions
        cc = cmin + (cmax-cmin)*np.random.rand(s)
        # pick min and mix feeding range
        rmin = cc - rr/2.0
        rmax = cc + rr/2.0
        # force no diet on species 0 - otherwise it will sometimes be a 
        # cannibal with no other links and cause a singularity
        cc[0] = rr[0] = 0.0
        # compute diets of each species
        diets = np.array([np.logical_and(nn >= rmin[i], nn < rmax[i]) for i in range(s)])
        nlinks = diets.sum()
        #print(nlinks)
        reject = abs(nlinks-l) > maxerr
        if not reject:     # check number of links is in range
            # check for disconnected or multiple components
            ll = diets.nonzero()
            links = zip(ll[0], ll[1])
            if singleComponent:
                nw = nx.Graph(links)     # networkx component routines only work on undirected graph
                reject = singleComponent and not nx.is_connected(nw)
            if not reject:
                nw = nx.DiGraph(links)      # build the network object
                if nw.number_of_nodes() != s:
                    if allowDisconnected:
                        nw.add_nodes_from(range(s))
                    else:
                        reject = True
            if not reject:
                tl = computeTL(nw)
                reject = reject or nonSingular and tl == None
            if not reject:
                return (nw, nn, cc, rr, tl)
    print("Niche model failed")
    return (None, None, None, None, None)
