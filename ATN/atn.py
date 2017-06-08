# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:07:48 2016

@author: rich
"""

import numpy as np
from scipy.integrate import ode
import networkx as nx
from FoodWeb.TrophicLevel import computeSWTL

class spType:
    prod, invert, ectoVert, endoVert = range(4)

class producerModel:
    speciesK, systemK = range(2)
    
# atn model parameters
#
class params:
    # a: competitive exclusion
    # q: type III Hill coeff
    # d: predator interference/prey protection
    # b0: half sat
    def __init__(self, a = 1.0, q=0.2, d=0, b0=0.5, r=1.0, prodModel=producerModel.systemK, k=5.0, speciesType=None, sizeRatio=100.0):
        self.q = q
        self.d = d
        self.a = a
        self.b0 = b0
        self.prodModel = prodModel
        self.k = k
        self.r = r
        self.speciesType = speciesType      # array of spType values
        self.sizeRatio = sizeRatio

class atn:
    # initialize the various model parameter arrays
    def __init__(self, nw, tl, params, thr=1e-15, includeActivity=False):
        self.thr = thr
        if includeActivity:
            self.fm = 0.1
            self.fa = 0.4
        else:
            self.fm = 1.0
            self.fa = 1.0            
        self.params = params
        self.initNetwork(nw, tl)
        self.paramsSet = False
        self.solver = ode(self.dbdt).set_integrator('vode', method='adams')
        self.solver = ode(self.dbdt).set_integrator('dopri5')
    
    def initNetwork(self, nw, tl):
        self.nw = nw
        self.s = nw.number_of_nodes()
        # boolean flag arrays
        self.basal = np.array(self.nw.out_degree().values()) == 0
        self.consumer = np.logical_not(self.basal)
        self.C = self.consumer.sum()
        self.B = self.basal.sum()
        if tl == None:
            tl = computeSWTL(nw)
        self.tl = np.array([tl[n] for n in nw.nodes_iter()]) if tl is not None else None
        return tl is not None
    
    # set the standard body-size based parameters
    def setStandardParams(self):
        aT = {spType.prod: 0.0, spType.invert: 0.31, spType.ectoVert: 0.88, spType.endoVert: 3.0}
        aR = 1.0
        yi = {spType.prod: 0.0, spType.invert: 8.0, spType.ectoVert: 4.0, spType.endoVert: 4.0}        
        efficiency = {spType.prod: 0.45, spType.invert: 0.85, spType.ectoVert: 0.85, spType.endoVert: 0.85}

        self.paramsSet = True
        params = self.params
        
        # set species types
        if params.speciesType is None:
            params.speciesType = np.empty(self.s)
            params.speciesType.fill(spType.invert)
        params.speciesType[self.basal] = spType.prod

        # species mass
        self.m = np.power(params.sizeRatio, self.tl-1.0)

        # primary producer properties
        self.r = np.zeros(self.s)
        self.r[self.basal] = params.r
        self.k = np.zeros(self.s)
        self.k[self.basal] = params.k if params.prodModel == producerModel.speciesK else params.k/self.basal.sum()

        # consumer node properties
        self.x = np.array([aT[st] for st in params.speciesType])*np.power(self.m, -0.25)/aR
        self.b0 = np.full(self.s, params.b0)
        self.h = np.full(self.s, 1.0+params.q)
        self.d = np.full(self.s, params.d)

        # consumer link properties
        y = np.array([yi[st] for st in params.speciesType])        
        self.xy = self.x*y        
        self.e = np.array([efficiency[st] for st in params.speciesType])

        # compute even diet fractions and assign
        self.w = np.zeros((self.s,self.s))
        vals = np.array(self.nw.out_degree().values()).astype(np.float)
        vals[vals.nonzero()] = 1.0/vals[vals.nonzero()]
        self.w[:,] = vals
        self.w = self.w.T
        # set diet frac to zero if no link present
        self.w[np.array(nx.adjacency_matrix(self.nw).todense() == 0)] = 0
        # initial values
        self.binit = np.zeros(self.s)
        self.var = None
        
    def randInitCond(self, minb = 0.1, maxb = 0.5):
        self.binit = np.random.uniform(minb, maxb, self.s)
        
    # combined type II/III and predator interference functional response
    def fnResp_TypeIII_PI(self, b):
        b0h = np.power(self.b0, self.h)
        bh = np.power(b, self.h)
        num = self.w*bh     
        denom = b0h*(1.0 + self.d*b) + self.w.dot(bh)
        return num/denom
        
    def growthFn(self, b):
        g = np.zeros(self.s)
        g[self.basal] = 1.0 - b[self.basal]/self.k[self.basal]
        return g
        
    # compute the time derivative of biomasses
    def dbdt(self, t, b):
        b[b < 0.0] = 0.0
        fnResp = self.fnResp_TypeIII_PI(b)
        # primary producer growth
        growth = b*self.r*self.growthFn(b)
        # metabolic loss
        metab = self.fm*self.x*b
        # gain from consumption
        consumption = self.fa*b*self.xy*np.sum(fnResp, axis=1)
        # loss from being consumed
        consumed = np.sum((fnResp.T*self.xy*b)/self.e, axis=1)
        return growth - metab + consumption - consumed
        
    # run the model over a time interval, saving the results at each step
    # optionally provide initial values
    def run(self, initt, finalt, steps, init=None, threshold=True):
        if self.paramsSet == False:
            self.setStandardParams()
        if init is None:
            self.randInitCond()
        else:
            self.binit = init
        self.solver.set_initial_value(self.binit, initt)
        tarray = np.linspace(initt, finalt, num=steps+1)
        results = []
        for t in tarray:
            if t == initt:
                results.append((t, self.binit))
            elif self.solver.successful():
                b = self.solver.integrate(t)
                b[b < (self.thr if threshold else 0.0)] = 0.0
                self.solver.set_initial_value(b, t)
                results.append((t, b))
            else:
                break
        return results
        
    # remove extinct species from the network
    # isExtinct: boolean array of extinct species
    # return false if new network is singular (cannot compute trophic levels)
    def removeExtinct(self, isExtinct):
        isPresent = np.logical_not(isExtinct)
        binit = self.solver.y[isPresent]
        self.params.speciesType = self.params.speciesType[isPresent]
        removeId = np.arange(self.s)[isExtinct]
        # mapping between original node ids and ids in new, smaller network
        self.keepId = isPresent.cumsum()-1        
        self.nw.remove_nodes_from(removeId)
        if self.initNetwork(self.nw, None):
            self.m = self.m[isPresent]
            self.r = self.r[isPresent]
            self.k = self.k[isPresent]
            self.x = self.x[isPresent]
            #self.y = self.y[isPresent]
            self.xy = self.xy[isPresent]
            self.b0 = self.b0[isPresent]
            self.h = self.h[isPresent]
            self.d = self.d[isPresent]
            self.e = self.e[isPresent]
            self.w = self.w[isPresent,:][:,isPresent]
            if self.var is not None:
                self.var['rinit'] = self.var['rinit'][isPresent]
                self.var['xinit'] = self.var['xinit'][isPresent]
                self.var['yinit'] = self.var['yinit'][isPresent]
                self.var['hinit'] = self.var['hinit'][isPresent]
                self.var['dinit'] = self.var['dinit'][isPresent]
            self.binit = binit.copy()
            return True
        return False
                            