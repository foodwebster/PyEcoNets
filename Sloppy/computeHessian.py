# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 21:31:09 2016

@author: rich
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:38:48 2016

@author: rich
"""

import numpy as np

# steps is total number of time steps;
# nTimes is the number of (equally-spaced) times the model is "measured" at
# indices are negative, so index into the time series from the end (last time) in the simulation
def getIndices(steps, nTimes):
    delta = steps/nTimes
    return [-1 - i*delta for i in range(nTimes)]

# Both Hessians are computed using the difference between perturbed and reference states.
# The perturbed state has an altered parameter, both states are run an equal
# number of timestep forward from an initial state that is quasi-equilibrium but
# sometimes has rare species that are exponetially decaying.  Using a forward-integrated
# reference state removes any spurious contribution from those species

# build approximate cost fn Hessian
# cost is sum of square of absolute differences, so is zero at equilibrium
# Hessian is linearized around the equilibrium params so is accurate for
# small changes from equilibrium
# use central difference finite difference approximation
# dC^2/dxdy = [C(x+dx, y+dy) - C(x+dx, y-dy) - C(x-dx, y+dy) + C(x-dx, y-dy)]/4dxdy
# residExp - residual normalization exponent, 0 - no normalization, absolute
# nTimes is the number of equally spaced times to compute model cost at
# logDelta true computes dC/dlogp, otherwise dC/dp
def finiteDiffHessian(model, initB, residExp=0.0, vf=0.001, steps=500, nTimes=1, logDelta=True):

    def computeCost(initB, newB, norm):
        # cost is sum of squares of normalized change
        # sums columns (each species) and rows (each time)
        return np.square((newB - initB)/norm).sum()/2

    nVarParams = model.nVarParams()
    hess = np.zeros((nVarParams,nVarParams))
    print("Computing sensitivity to changes in %d parameters"%nVarParams)
    # reference state runs forward same amount as used to compute state with altered params.
    # this means that any species that are not at equilibrium change over the same amount
    # of time in reference and altered states
    indices = getIndices(steps, nTimes)
    results = model.run(0, steps, steps, init=initB, threshold=False)
    refB = np.vstack([results[idx][1] for idx in indices])
    norm = np.power(refB, residExp) if residExp != 0 else refB.sum()

    for i in range(nVarParams):
        for j in range(i, nVarParams):
            print("Computing sensitivity to changes in parameters %d, %d"%(i, j))
            if i == j:
                def varyOneParam(i, posi):
                    initPi, finalPi = model.varParam(i, neg=not posi, varFrac=vf)
                    deltai = (initPi - finalPi)
                    results = model.run(0, steps, steps, init=initB, threshold=False)
                    newB = np.vstack([results[idx][1] for idx in indices])
                    model.restoreParam(i)
                    return computeCost(refB, newB, norm), deltai
                currDiff, deltai = varyOneParam(i, True)
                currDiff += varyOneParam(i, False)[0]
                hess[i,i] = currDiff/(deltai*deltai)
            else:
                def varyParam(i, j, posi, posj):
                    initPi, finalPi = model.varParam(i, neg=not posi, varFrac=vf)
                    initPj, finalPj = model.varParam(j, neg=not posj, varFrac=vf)
                    deltai = (finalPi - initPi)
                    deltaj = (finalPj - initPj)
                    if logDelta:
                        deltai /= initPi
                        deltaj /= initPj
                    results = model.run(0, steps, steps, init=initB, threshold=False)
                    newB = np.vstack([results[idx][1] for idx in indices])
                    model.restoreParam(i)
                    model.restoreParam(j)
                    return computeCost(refB, newB, norm), deltai, deltaj
                currDiff, deltai, deltaj = varyParam(i, j, True, True)
                currDiff -= varyParam(i, j, True, False)[0]
                currDiff -= varyParam(i, j, False, True)[0]
                currDiff += varyParam(i, j, False, False)[0]
                hess[j, i] = hess[i, j] = currDiff/(4*deltai*deltaj)
    return hess

# Levenberg-Marguardt Hessian is an approximation of the exact Hessian
# in the neighborhood of cost C = 0
# when the cost fn is based on sum of squares of residuals r
# alternative way to compute Hessian of least squares cost fn is H=JTJ
# where Jacobean Jij = dri/dpj, r is residual and p is parameters
# based on first derivatives, so computation is O(n) rather than O(n^2)
# vf - fraction change in parameter values
# residExp - residual normalization exponent
# logDelta true computes dC/dlogp, otherwise dC/dp
def Jacobian(model, initB, modelIdx, vf = 0.001, steps=1000, nTimes=1, logDelta=True):
    indices = getIndices(steps, nTimes)

    def varyParam(j, initB, refB, posj):
        initPj, finalPj = model.varParam(j, neg=not posj, varFrac=vf)
        deltaj = (finalPj - initPj)
        if logDelta:
            if initPj != 0 and not np.isnan(initPj):
                deltaj /= initPj
            else:
                print("Bad parameter value: param index: %d value: %g"%(j, initPj))
                return 0.0, deltaj
        results = model.run(0, steps, steps, init=initB, threshold=False)
        newB = np.hstack([results[idx][1] for idx in indices])
        model.restoreParam(j)
        return (newB - refB), deltaj

    # reference state runs forward same amount as used to compute state with altered params.
    # this means that any species that are not at equilibrium change over the same amount
    # of time in reference and altered states
    results = model.run(0, steps, steps, init=initB, threshold=False)
    refB = np.hstack([results[idx][1] for idx in indices])

    nVarParams = model.nVarParams()
    nSp = model.s
    jac = np.zeros((nSp*nTimes,nVarParams))
    #norm = np.power(refB, residExp) if residExp != 0 else refB.sum()
    # compute rows of Jacardian
    print("[LMHessian] Computing Jacobian")
    for j in range(nVarParams):
        print("[LMHessian] %d Computing sensitivity to changes in parameter %d"%(modelIdx, j))
        resPos, deltaPos = varyParam(j, initB, refB, True)
        resNeg, deltaNeg = varyParam(j, initB, refB, False)
        #jac.T[j] = (resPos - resNeg)/((deltaPos-deltaNeg)*norm)
        jac.T[j] = (resPos - resNeg)/(deltaPos-deltaNeg)
    return jac, refB

def LMHessian(jac, refB, residExp):
    if residExp != 0:
        norm = np.power(refB, residExp)
        jac /= norm[:,np.newaxis]
    else:
        jac /= refB.sum()
    return np.dot(jac.T, jac)

'''
def toDisk(obj, name):
    f = open(name+".pickle", 'w')
    pickle.dump(obj, f)
    f.close()

def fromDisk(name):
    f = open(name+".pickle", 'r')
    obj = pickle.load(f)
    f.close()
    return obj
'''