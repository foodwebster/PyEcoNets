# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:06:46 2016

@author: rich
"""

import numpy as np
from atn import atn

class atnVarParam(atn):
    def __init__(self, nw, tl, params, thr=1e-15, includeActivity=False):
        atn.__init__(self, nw, tl, params, thr=thr, includeActivity=includeActivity)
        
    def setupVarParams(self, var = {'x': True, 'xy': True, 'r': True, 'k': True, 'h': True, 'd': True, 'b0': True}):
        self.var = var
        if self.var['x']:
            self.var['xinit'] = np.copy(self.x)
        if self.var['xy']:
            self.var['xyinit'] = np.copy(self.xy)
        if self.var['r']:
            self.var['rinit'] = np.copy(self.r)
        if self.var['k']:
            self.var['kinit'] = np.copy(self.k)
        if self.var['h']:
            self.var['hinit'] = np.copy(self.h)
        if self.params.d > 0 and self.var['d']:
            self.var['dinit'] = np.copy(self.d)
        else:
            self.var['d'] = False
        if self.var['b0']:
            self.var['b0init'] = np.copy(self.b0)
        
    # return the number of variable parameters
    def nVarParams(self):
        nVar = 0
        if self.var['x']:
            nVar += self.C
        if self.var['xy']:
            nVar += self.C
        if self.var['r']:
            nVar += self.B
        if self.var['k']:
            nVar += self.B
        if self.var['h']:
            nVar += self.C
        if self.var['d']:
            nVar += self.C
        if self.var['b0']:
            nVar += self.C
        return int(nVar)

    def getVarParam(self, idx):
        if self.var['x']:
            if idx < self.C:
                return ('x', self.consumer.nonzero()[0][idx])
            idx -= self.C
        if self.var['xy']:
            if idx < self.C:
                return ('xy', self.consumer.nonzero()[0][idx])
            idx -= self.C
        if self.var['h']:
            if idx < self.C:
                return ('h', self.consumer.nonzero()[0][idx])
            idx -= self.C
        if self.var['d']:
            if idx < self.C:
                return ('d', self.consumer.nonzero()[0][idx])
            idx -= self.C
        if self.var['b0']:
            if idx < self.C:
                return ('b0', self.consumer.nonzero()[0][idx])
            idx -= self.C
        if self.var['r']:
            if idx < self.B:
                return ('r', self.basal.nonzero()[0][idx])
            idx -= self.B
        if self.var['k']:
            if idx < self.B:
                return ('k', self.basal.nonzero()[0][idx])

    # alter one of the variable parameters by a fixed fraction
    # return initial and varied parameter values
    def varParam(self, idx, neg=False, varFrac=0.01):
        varFrac = -varFrac if neg else varFrac
        vp = self.getVarParam(idx)
        if vp[0] == 'x':
            val = self.x[vp[1]]
            self.x[vp[1]] += varFrac*val
            newVal = self.x[vp[1]]
        elif vp[0] == 'xy':
            val = self.xy[vp[1]]
            self.xy[vp[1]] += varFrac*val
            newVal = self.xy[vp[1]]
        elif vp[0] == 'r':
            val = self.r[vp[1]]
            self.r[vp[1]] += varFrac*val
            newVal = self.r[vp[1]]
        elif vp[0] == 'k':
            val = self.k[vp[1]]
            self.k[vp[1]] += varFrac*val
            newVal = self.k[vp[1]]
        elif vp[0] == 'h':
            val = self.h[vp[1]]
            self.h[vp[1]] += varFrac*val
            newVal = self.h[vp[1]]
        elif vp[0] == 'd':
            val = self.d[vp[1]]
            self.d[vp[1]] += varFrac*val
            newVal = self.d[vp[1]]
        elif vp[0] == 'b0':
            val = self.b0[vp[1]]
            self.b0[vp[1]] += varFrac*val
            newVal = self.b0[vp[1]]
        return val, newVal
    
    # alter one of the variable parameters
    def rndVarParam(self, idx, varFrac=0.1):
        def _vary(val):
            return val * max(0.0, np.random.normal(1.0, varFrac))

        vp = self.getVarParam(idx)            
        if vp[0] == 'x':
            self.x[vp[1]] = _vary(self.x[vp[1]])
        elif vp[0] == 'xy':
            self.xy[vp[1]] = _vary(self.xy[vp[1]])
        elif vp[0] == 'r':
            self.r[vp[1]] = _vary(self.r[vp[1]])
        elif vp[0] == 'k':
            self.k[vp[1]] = _vary(self.k[vp[1]])
        elif vp[0] == 'h':
            self.h[vp[1]] = _vary(self.h[vp[1]])
        elif vp[0] == 'd':
            self.d[vp[1]] = _vary(self.d[vp[1]])
        elif vp[0] == 'b0':
            self.b0[vp[1]] = _vary(self.b0[vp[1]])
    
    def restoreParam(self, idx):
        vp = self.getVarParam(idx)            
        if vp[0] == 'x':
            self.x[vp[1]] = self.var['xinit'][vp[1]]
        elif vp[0] == 'xy':
            self.xy[vp[1]] = self.var['xyinit'][vp[1]]
        elif vp[0] == 'r':
            self.r[vp[1]] = self.var['rinit'][vp[1]]
        elif vp[0] == 'k':
            self.k[vp[1]] = self.var['kinit'][vp[1]]
        elif vp[0] == 'h':
            self.h[vp[1]] = self.var['hinit'][vp[1]]
        elif vp[0] == 'd':
            self.d[vp[1]] = self.var['dinit'][vp[1]]
        elif vp[0] == 'b0':
            self.b0[vp[1]] = self.var['b0init'][vp[1]]
        
    