# -*- coding: utf-8 -*-

import os
import sys
from os.path import expanduser
import pandas as pd
import itertools as it
import random
import numpy as np
from collections import defaultdict

sys.path.append(expanduser("~/OneDrive/Rich/Software/PythonFoodWebs"))

from FoodWeb.deleteSpecies import deleteSpecies
from FoodWeb.fromFile import foodwebFromFile
from FoodWeb.TrophicLevel import computeSWTL
import FoodWeb.foodwebUtil as fwu

os.chdir(expanduser("~/OneDrive/Rich/Research/Sharks"))

pred = 'con.taxonomy'
prey = 'res.taxonomy'

fwFile = "Weddell.txt"
spFile = "WeddellSpp.csv"

def runInvasionExpt(invaderLinkFile):
    def getLossHelper(initdeg, finaldeg):
        loss = []
        for init, deg in initdeg.iteritems():
            if init in finaldeg:
                final = finaldeg[init]
                delta = (deg-final)/float(deg) if deg > 0 else 0
                if delta > 0:
                    loss.append(delta)
        loss = np.array(loss)
        maxloss = loss.max()
        return {'n':len(loss), 'mean':loss.mean(), 'max':maxloss}
    
    # get number of species that have fewer predators
    # and average fraction of predators lost
    def getPredLoss(initfw, finalfw):
        return getLossHelper(initfw.in_degree(), finalfw.in_degree())
    
    # prey are things a node is linked to
    # get number of species that have fewer prey
    # and average fraction of prey lost
    def getPreyLoss(initfw, finalfw):
        return getLossHelper(initfw.out_degree(), finalfw.out_degree())

    ildf = pd.read_excel(invaderLinkFile)
    invaders = set(ildf[pred].unique().tolist())
    
    fw = foodwebFromFile(fwFile)
    tls = computeSWTL(fw)
    basal = set(fwu.getBasal(fw))
    
    sdf = pd.read_csv(spFile)
    sMap = dict(zip(sdf[pred], sdf['id']))
    nSpp = len(sdf)
    invaderLinks = zip(ildf[pred], ildf[prey])
    invaderPrey = defaultdict(list)
    for sl in invaderLinks:
        preyId = sMap[sl[1]]
        # some invader links might not be in the food web or are to other invaders
        if str(preyId) in fw and sl[1] not in invaders:
            invaderPrey[sMap[sl[0]]].append(preyId)
    invaderIds = invaderPrey.keys()
    
    # Variables: number of sharks to add
    #            fraction of sharks prey to make extinct
    #            number of iterations of each treatment
    nInvaders = xrange(1,len(invaderPrey)+1)
    fracExtinct = [0.25, 0.5, 0.75, 1.00]
    nIter = 10
    
    results = []
    for frac in fracExtinct:
        nIt = nIter if frac < 1 else 1
        for ns in nInvaders:
            invaderCombos = it.combinations(invaderIds, ns)
            for invaderList in invaderCombos:
                targets = list(set(it.chain(*[invaderPrey[s] for s in invaderList])))
                nDel = int(frac*len(targets))
                for i in xrange(nIt):
                    print("Adding %d invaders, deleting %d (%f) species, iteration %d"%(ns, nDel, frac, i))
                    tempfw = fw.copy()
                    random.shuffle(targets)
                    nExtinct = 0
                    deleted = targets[:nDel]
                    for sp in deleted:
                        nExtinct += deleteSpecies(str(sp), tempfw, basal=basal)
                    nSecondary = nExtinct - nDel
                    preyLoss = getPreyLoss(fw, tempfw)
                    predLoss = getPredLoss(fw, tempfw)
                    results.append({'frac': frac,
                                    'nFinal': nSpp - nExtinct,
                                    'nInvader': ns, 
                                    'nDeleted': nDel, 
                                    'nExtinct': nExtinct,
                                    'nSecondary': nSecondary,
                                    'fracDeleted': nDel/float(nSpp), 
                                    'fracExtinct': nExtinct/float(nSpp),
                                    'fracSecondary': nSecondary/float(nSpp),
                                    'nPreyLoss' : preyLoss['n'],
                                    'fracPreyLoss': preyLoss['n']/float(nSpp - nExtinct),
                                    'mnPreyLoss' : preyLoss['mean'],
                                    'maxPreyLoss' : preyLoss['max'],
                                    'nPredLoss' : predLoss['n'],
                                    'fracPredLoss': predLoss['n']/float(nSpp - nExtinct),
                                    'mnPredLoss' : predLoss['mean'],
                                    'maxPredLoss' : predLoss['max']
                                    })
    return pd.DataFrame(results)
    
sharkLinkFile = "WeddellSharkLinks.xls"
crabLinkFile = "WeddellCrabLinks.xls"
bothLinkFile = "WeddellSharkCrabLinks.xls"

#rdf = runInvasionExpt(sharkLinkFile)
#rdf.to_csv("SharkInvasionResults.csv", index=False)

#rdf = runInvasionExpt(crabLinkFile)
#rdf.to_csv("CrabInvasionResults.csv", index=False)

rdf = runInvasionExpt(bothLinkFile)
rdf.to_csv("SharkCrabInvasionResults.csv", index=False)

