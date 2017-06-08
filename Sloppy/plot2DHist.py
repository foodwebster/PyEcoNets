# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 08:58:13 2016

Plot sloppy modeling results as 2D histograms - grid shaded by density

@author: richard.williams
"""

import sys
from os.path import expanduser
sys.path.append(expanduser("~/OneDrive/Rich/Software/PythonFoodWebs"))

import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from FoodWeb.TrophicLevel import computeSWTL
from FoodWeb.foodwebUtil import getBasal

def drawHistogs(v1data, v1name, v1cats, v1delta,
                v2data, v2name, v2cats, v2delta,
                bins, paramHist=None, globalHist=None,
                showXHist=True, showYHist=True, title=None, base=None):
    # definitions for the axes
    left, width = 0.19, 0.5
    bottom, height = 0.14, 0.5
    left_h = left + width + 0.04
    bottom_h = bottom + height + 0.04
    hist_ht = 0.18

    plt.figure(figsize=(4, 4))
    # define the drawing axes
    ax2d = plt.axes([left, bottom, width, height])
    if showXHist:
        rect_histx = [left, bottom_h, width, hist_ht]
        axHistx = plt.axes(rect_histx)
    if showYHist:
        rect_histy = [left_h, bottom, hist_ht, height]
        axHisty = plt.axes(rect_histy)

    # the 2d histogram plot:
    bins1 = bins[0]
    bins2 = bins[1]
    ax2d.hist2d(v1data, v2data, cmap=plt.cm.binary, bins=[bins1, bins2])
    if v1cats is not None:
        ax2d.set_xticks(bins1[:len(v1cats)] + 0.5*v1delta)
        ax2d.set_xticklabels(v1cats)
    if v2cats is not None:
        ax2d.set_yticks(bins2[:len(v2cats)] + 0.5*v2delta)
        ax2d.set_yticklabels(v2cats)
    ax2d.tick_params(axis='both', which='major', labelsize=14)
    #ax2d.set_xlabel(v1name, fontsize=20)
    ax2d.set_ylabel(v2name, fontsize=16)
    # the 1d histograms
    if showXHist:
        align = 'left' if v1cats is not None else'mid'
        (n, bins, patches) = axHistx.hist(v1data, bins=bins1, color='0.5', normed=True, align=align)
        axHistx.xaxis.tick_top()
        #if v1delta != 1:
        # set y axis tick labels
        ymin, ymax = axHistx.get_ylim()
        if ymax < 0.5:
            tickdelta = 0.1
        else:
            tickdelta = 0.2
        axHistx.set_yticklabels([(str(v*v1delta) if v%tickdelta<0.01 else '') for v in axHistx.get_yticks()])
        # set x-axis ticks
        if v1cats is not None:
            axHistx.set_xticks(bins1[:len(v1cats)])
            axHistx.set_xticklabels(v1cats)
        #axHistx.set_ylabel('Fraction', fontsize=20)
        axHistx.tick_params(axis='both', which='major', labelsize=14)
        if paramHist is not None:
            # add the global parameter distribution
            axHistx.bar(bins1[:len(v1cats)]-0.5, paramHist, width=1.0, ec='k', fc='none')#, ls='dotted')
            axHistx.set_xlim(right=len(paramHist)-1.0)#6.0)
    if showYHist:
        align = 'left' if v2cats is not None else'mid'
        (n, bins, patches) = axHisty.hist(v2data, orientation='horizontal', bins=bins2, color='0.5', normed=True, align=align)
        axHisty.yaxis.tick_right()
        axHisty.set_yticklabels([])
        # set x-axis tick labels
        xmin, xmax = axHisty.get_xlim()
        if xmax < 0.3:
            tickdelta = 0.1
        elif xmax < 0.7:
            tickdelta = 0.2
        else:
            tickdelta = 0.25
        if xmax > 1.0:
            xmax = 1.0
            axHisty.set_xlim(right=xmax)
        axHisty.set_xticklabels([(str(v*v2delta) if v%tickdelta<0.01 else '') for v in axHisty.get_xticks()], rotation=90)
        #axHisty.set_xticklabels(axHisty.xaxis.get_ticklabels(), rotation=90)
        if v2cats is None:
            axHisty.set_ylim(bins2[0], bins2[-1])
        else:
            axHisty.set_yticks(bins2[:len(v2cats)])
            axHisty.set_yticklabels(v2cats)
        axHisty.tick_params(axis='both', which='major', labelsize=14)
        if globalHist is not None:
            # add the global property value distribution
            ht = globalHist[1][1] - globalHist[1][0]
            axHisty.barh(globalHist[1][:-1], globalHist[0], height=ht, ec='k', fc='none')#, ls='dotted')
    if base is not None:
        plt.savefig(base+".pdf")
        #plt.savefig(base+".eps")
    plt.show()

def getBins(data, cats, delta, minVal=None):
    if cats is not None:
        bins = np.arange(-0.5*delta,len(cats)+0.5*delta,delta)
    else:
        minv = min(data) if minVal is None else minVal
        maxVal = max(data)
        maxv = math.ceil(maxVal/delta)*delta
        if maxv == maxVal:  # fix for when data ends on bin boundary
            maxv += delta
        bins = np.arange(minv,maxv+delta,delta)
    return bins

def getDataForHistograms(results, datatype, axisThr, componentThr):
    if datatype=='Producer':
        params = ['r', 'k']
        tlSelectFn = lambda tl: tl  == 1.0
    elif datatype == 'Consumer':
        params = ['x', 'xy', 'h', 'b0', 'd']
        tlSelectFn = lambda tl: tl  > 1.0
    else:
        params = ['r', 'k', 'x', 'xy', 'h', 'b0', 'd']
        tlSelectFn = lambda tl: True

    pMap = {params[i]:i for i in range(len(params))}
    nProd = 0
    nCons = 0
    # build lists of node properties associated with significant eigenvectors
    def buildLists(nw, res, params, tls, degs, indegs, outdegs, biomasses, tlSelFn):
        indeg = nw.in_degree()
        outdeg = nw.out_degree()
        res = [res[i] for i in range(len(res)) if res[i]['axisRatio'] >= axisThr and abs(res[i]['relComponent']) >= componentThr]
        for r in res:
            if tlSelFn(r['TL']):
                params.append(pMap[r['param']])
                tls.append(r['TL'])
                degs.append(r['degree'])
                indegs.append(indeg[str(r['node'])])
                outdegs.append(outdeg[str(r['node'])])
                biomasses.append(-math.log10(r['relBiomass']))

    globalIn = []; globalOut = []; globalTL = []; globalB = []
    absParams = []; absTLs = []; absDegrees = []; absInDegrees = []; absOutDegrees = []; absBiomasses = []
    relParams = []; relTLs = []; relDegrees = []; relInDegrees = []; relOutDegrees = []; relBiomasses = []
    for (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in results[0]:
        nw = nx.parse_edgelist(modelInfo['network'], create_using=nx.DiGraph())
        tls = computeSWTL(nw)
        keep = [tlSelectFn(tl) for tl in tls.values()]
        buildLists(nw, resultsAbs[:nBigAbs], absParams, absTLs, absDegrees, absInDegrees, absOutDegrees, absBiomasses, tlSelectFn)
        buildLists(nw, resultsRel[:nBigRel], relParams, relTLs, relDegrees, relInDegrees, relOutDegrees, relBiomasses, tlSelectFn)
        keep = [node for node, tl in tls.iteritems() if tlSelectFn(tl)]
        allnodes = np.array([int(node) for node in tls.iterkeys()])
        allnodes.sort()
        allnodes = [str(n) for n in allnodes]
        biomasses = dict(zip(allnodes, -np.log10(modelInfo['binit']/modelInfo['binit'].max())))
        globalIn.extend(nw.in_degree(keep).values())
        globalOut.extend(nw.out_degree(keep).values())
        globalTL.extend([tls[n] for n in keep])
        globalB.extend([biomasses[n] for n in keep])
        nB = len(getBasal(nw))
        nProd += nB
        nCons += nw.number_of_nodes() - nB
    results = {'nProd': nProd, 'nCons': nCons, 'params': params,
                'globalIn': globalIn, 'globalOut': globalOut, 'globalTL': globalTL, 'globalB': globalB,
                'absParams': absParams, 'absTLs': absTLs, 'absDegrees': absDegrees,
                'absInDegrees': absInDegrees, 'absOutDegrees': absOutDegrees, 'absBiomasses': absBiomasses,
                'relParams': relParams, 'relTLs': relTLs, 'relDegrees': relDegrees,
                'relInDegrees': relInDegrees, 'relOutDegrees': relOutDegrees, 'relBiomasses': relBiomasses}
    return results

def plot2DHistParamVsAttrs(results, savepath=None, datatype="All", axisThr=0.7, componentThr=0.2):
    histData = getDataForHistograms(results, datatype, axisThr, componentThr)

    deltaB = 0.5 if datatype=="Producer" else 1.0
    params = histData['params']
    nProd = histData['nProd']
    nCons = histData['nCons']
    absParams = np.array(histData['absParams'])
    absTLs = np.array(histData['absTLs'])
    absInDegrees = np.array(histData['absInDegrees'])
    absOutDegrees = np.array(histData['absOutDegrees'])
    absBiomasses = np.array(histData['absBiomasses'])
    relParams = np.array(histData['relParams'])
    relTLs = np.array(histData['relTLs'])
    relInDegrees = np.array(histData['relInDegrees'])
    relOutDegrees = np.array(histData['relOutDegrees'])
    relBiomasses = np.array(histData['relBiomasses'])
    histB = np.histogram(np.array(histData['globalB']), bins=np.arange(0,13,deltaB), density=True)
    histTL = np.histogram(np.array(histData['globalTL']), bins=np.arange(1,5.5,0.5), density=True)
    histOut = np.histogram(np.array(histData['globalOut']), bins=np.arange(0,max(histData['globalOut'])+2), density=True)
    histIn = np.histogram(np.array(histData['globalIn']), bins=np.arange(0,max(histData['globalIn'])+2), density=True)

    if datatype == "Producer" or datatype == "Consumer":
        paramHist = np.full(len(params), 1.0/len(params))
    else:
        # compute parameter histogram frequencies
        fracP = nProd/float(nProd+nCons)
        fracC = 1.0 - fracP
        frac1 = fracP/(2*fracP+5*fracC)
        frac2 = fracC/(2*fracP+5*fracC)
        print("Parameter frequencies: producers: %f, Consumers: %f"%(frac1, frac2))
        paramHist = np.array([frac1, frac1, frac2, frac2, frac2, frac2, frac2])

    # get parameters bins
    binsParams = getBins(None, params, 1.0)
    # trophic levels
    if (histTL[0]>0).sum() > 1: # check that more than one bin is occupied
        # get attribute bins
        binsRel = getBins(relTLs, None, 0.5)
        binsAbs = getBins(absTLs, None, 0.5)
        binsAttr = binsRel if (len(binsRel) > len(binsAbs)) else binsAbs
        # plot histograms
        drawHistogs(absParams, "Parameters", params, 1, absTLs,
                    "Trophic Level", None, 0.5, [binsParams, binsAttr],
                    paramHist=paramHist, globalHist=histTL, base=savepath+"/ParamsVsTLAbs_"+datatype if savepath else None)
        drawHistogs(relParams, "Parameters", params, 1, relTLs,
                    "Trophic Level", None, 0.5, [binsParams, binsAttr],
                    paramHist=paramHist, globalHist=histTL, base=savepath+"/ParamsVsTLRel_"+datatype if savepath else None)

    # In degree
    # get attribute bins
    binsRel = getBins(relInDegrees, None, 1)
    binsAbs = getBins(absInDegrees, None, 1)
    binsAttr = binsRel if (len(binsRel) > len(binsAbs)) else binsAbs
    # plot histograms
    drawHistogs(absParams, "Parameters", params, 1, absInDegrees,
                "In Degree", None, 1, [binsParams, binsAttr],
                globalHist=histIn, base=savepath+"/ParamsVsInDegAbs_"+datatype if savepath else None, showXHist=False)
    drawHistogs(relParams, "Parameters", params, 1, relInDegrees,
                "In Degree", None, 1, [binsParams, binsAttr],
                globalHist=histIn, base=savepath+"/ParamsVsInDegRel_"+datatype if savepath else None, showXHist=False)

    # out degree
    if (histOut[0]>0).sum() > 1: # check that more than one bin is occupied
        # get attribute bins
        binsRel = getBins(relOutDegrees, None, 1)
        binsAbs = getBins(absOutDegrees, None, 1)
        binsAttr = binsRel if (len(binsRel) > len(binsAbs)) else binsAbs
        # plot histograms
        drawHistogs(absParams, "Parameters", params, 1, absOutDegrees,
                    "Out Degree", None, 1, [binsParams, binsAttr],
                    globalHist=histOut, base=savepath+"/ParamsVsOutDegAbs_"+datatype if savepath else None, showXHist=False)
        drawHistogs(relParams, "Parameters", params, 1, relOutDegrees,
                    "Out Degree", None, 1, [binsParams, binsAttr],
                    globalHist=histOut, base=savepath+"/ParamsVsOutDegRel_"+datatype if savepath else None, showXHist=False)

    # biomass
    # get attribute bins
    binsRel = getBins(relBiomasses, None, deltaB, minVal=0)
    binsAbs = getBins(absBiomasses, None, deltaB, minVal=0)
    binsAttr = binsRel if (len(binsRel) > len(binsAbs)) else binsAbs
    # plot histograms
    drawHistogs(absParams, "Parameters", params, 1, absBiomasses,
                r'-log$_{10}$Biomass', None, deltaB, [binsParams, binsAttr],
                globalHist=histB, base=savepath+"/ParamsVsBAbs_"+datatype if savepath else None, showXHist=False)
    drawHistogs(relParams, "Parameters", params, 1, relBiomasses,
                r'-log$_{10}$Biomass', None, deltaB, [binsParams, binsAttr],
                globalHist=histB, base=savepath+"/ParamsVsBRel_"+datatype if savepath else None, showXHist=False)

def plot2DHistParamParam(results, savepath=None, axisThr=0.7, componentThr=0.2):
    params = ['r', 'k', 'x', 'xy', 'h', 'b0', 'd']
    pMap = {params[i]:i for i in range(len(params))}
    # build parameter pairs
    absPairs = []
    relPairs = []
    for (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in results[0]:
        def buildParamPairList(res, pairs):
            res = [res[i] for i in range(len(res)) if res[i]['axisRatio'] >= axisThr and abs(res[i]['relComponent']) >= componentThr]
            if len(res) > 1:
                for r1 in res:
                    p1 = r1['param']
                    for r2 in res:
                        if r1 != r2:
                            p2 = r2['param']
                            if pMap[p2] < pMap[p1]: # order the parameters
                                temp = p2
                                p2 = p1
                                p1 = temp
                            pairs.append((pMap[p1], pMap[p2]))
                            pairs.append((pMap[p2], pMap[p1]))

        buildParamPairList(resultsAbs[:nBigAbs], absPairs)
        buildParamPairList(resultsRel[:nBigRel], relPairs)

    absParams = np.array(absPairs).T
    relParams = np.array(relPairs).T

    bins = getBins(None, params, 1.0)

    drawHistogs(absParams[0], "Parameters", params, 1, absParams[1], "Parameters", params, 1, [bins, bins], showYHist=False, base=savepath+"/ParamPairsAbs" if savepath else None)
    drawHistogs(relParams[0], "Parameters", params, 1, relParams[1], "Parameters", params, 1, [bins, bins], showYHist=False, base=savepath+"/ParamPairsRel" if savepath else None)
