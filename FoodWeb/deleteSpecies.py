# -*- coding: utf-8 -*-

import foodwebUtil as fwu

# delete a species from a foodweb; 
# optionally passed a set of basal species
# so basal list doesn't have to be repeatedly recomputed
# return number deleted
def deleteSpecies(sp, fw, basal=None):
    def _deleteHelper(sp, fw, basal):
        fw.remove_node(sp)
        newB = fwu.getBasal(fw)
        return [s for s in newB if s not in basal]
    if basal is None:
        basal = fwu.getBasal(fw)
    nDel = 0
    if sp in fw:
        spp = [sp]
        while len(spp) > 0:
            nDel += 1
            sp = spp.pop(0)
            spp.extend(_deleteHelper(sp, fw, basal))
    else:
        print("%s not in food web"%sp)
    return nDel