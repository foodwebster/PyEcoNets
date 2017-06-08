# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 08:58:13 2016

@author: richard.williams
"""

from collections import defaultdict
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from FoodWeb.TrophicLevel import computeSWTL

# build log normalized eigenvalue histograms for absolute and relative eigenvalues
def eigenvalueHistograms(results, savepath):

    # return lognorm of all eigenvalues that are > 0
    def logNormEig(eig):
        eig = eig[eig>0]
        return np.log10(eig/eig.max())

    # determine whether to keep a tick mark
    def keepTick(v, tickdelta):
        return (v%tickdelta<tickdelta/100 or ((tickdelta-v)%tickdelta<tickdelta/100))

    allEigAbs = []
    allEigRel = []
    # get lognorm of all eigenvalues
    for (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in results:
        allEigAbs.append(logNormEig(eigvalAbs))
        allEigRel.append(logNormEig(eigvalRel))
    # merge list of arrays into a single array
    allEigAbs = np.hstack(allEigAbs)
    allEigRel = np.hstack(allEigRel)

    # get counts, build bins
    nAbs = len(allEigAbs)
    nRel = len(allEigRel)
    eigAbs = allEigAbs[allEigAbs > -8]
    eigRel = allEigRel[allEigRel > -8]
    bins = np.arange(-8,0.5,0.5)
    absHist = np.histogram(eigAbs, bins=bins )
    valsAbs = absHist[0]/float(nAbs)
    relHist = np.histogram(eigRel, bins=bins)
    valsRel = relHist[0]/float(nRel)

    #nAbs = len(allEigAbs)
    #nRel = len(allEigRel)
    pctAbs = int(100*float(len(eigAbs))/nAbs)
    pctRel = int(100*float(len(eigRel))/nRel)

    ymax = 0.04
    # plot sum of squares histogram
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.15, bottom=0.25, top=0.88, wspace=0.05)
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(absHist[1][0:len(valsAbs)], valsAbs, width=0.5, color='0.5', edgecolor='0', align='edge')
    ax1.set_ylim([0,ymax])
    ax1.set_xlim([-8.0, 0.0])
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.tick_params(axis='x', direction='out', top='off')
    ax1.set_yticklabels([(str(v) if keepTick(v, 0.01) else '') for v in ax1.get_yticks()])
    ax1.set_xticklabels([(str("%d"%v) if abs(v%2) < 0.01 and v > -7 else '') for v in ax1.get_xticks()])
    ax1.set_title('Fractional', fontsize=16)
    ax1.annotate(str('%d%% of vals'%(pctAbs)), xy=(-7.7, 0.036), fontsize=14)
    ax1.set_ylabel("Fraction", fontsize=16)
    ax1.set_title('Sum of squares', fontsize=16)

    # plot fractional histogram
    ax2 = plt.subplot(1, 2, 2, sharex=ax1)
    ax2.bar(relHist[1][0:len(valsRel)], valsRel, width=0.5, color='0.5', edgecolor='0', align='edge')
    ax2.set_ylim([0,ymax])
    ax2.set_xlim([-8.0, 0.0])
    ax2.annotate(str('%d%% of vals'%(pctRel)), xy=(-7.7, 0.036), fontsize=14)
    ax2.set_yticklabels([])
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='x', direction='out', top='off')
    ax2.set_title('Fractional', fontsize=16)

    ax1.annotate(r'log$_{10}$(eigval/max eigval)', xy=(-4, -0.011), annotation_clip=False, fontsize=16)
    #ax1.annotate('Eigenvalue Distriution', xy=(-4, 0.048), annotation_clip=False, fontsize=16)
    ax1.annotate('(a)', xy=(-1, 0.036), fontsize=14)
    ax2.annotate('(b)', xy=(-1, 0.036), fontsize=14)
    if savepath:
        plt.savefig(savepath+"/eigenvalHistog.pdf")

# build response magnitude (based on max eigenval) histogram
def responseHistograms(results, savepath):
    def keepTick(v, tickdelta):
        return (v%tickdelta<tickdelta/100 or ((tickdelta-v)%tickdelta<tickdelta/100))
    maxEigAbs = []
    maxEigRel = []

    for (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in results:
        maxEigAbs.append(eigvalAbs.max())
        maxEigRel.append(eigvalRel.max())

    responseAbs = np.log10(np.sqrt(np.array(maxEigAbs)))
    responseRel = np.log10(np.sqrt(np.array(maxEigRel)))

    nAbs = len(responseAbs)
    nRel = len(responseRel)
    bins = np.arange(1,4,0.25)
    absHist = np.histogram(responseAbs, bins=bins )
    valsAbs = absHist[0]/float(nAbs)
    relHist = np.histogram(responseRel, bins=bins)
    valsRel = relHist[0]/float(nRel)

    ymax = 0.5
    # plot sum of squares histogram
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.15, bottom=0.25, top=0.88, wspace=0.05)
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(absHist[1][0:len(valsAbs)], valsAbs, width=0.25, color='0.5', edgecolor='0', align='edge')
    ax1.set_ylim([0,ymax])
    ax1.set_xlim([1.0, 4.0])
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.tick_params(axis='x', direction='out', top='off')
    ax1.set_yticklabels([(str(v) if keepTick(v, 0.01) else '') for v in ax1.get_yticks()])
    #ax1.set_xticklabels([(str("%d"%v) if abs(v%1) < 0.01 and v > 0 else '') for v in ax1.get_xticks()])
    ax1.set_title('Fractional', fontsize=16)
    ax1.set_ylabel('Fraction', fontsize=16)
    ax1.set_title('Sum of squares', fontsize=16)

    # plot fractional histogram
    ax2 = plt.subplot(1, 2, 2, sharex=ax1)
    ax2.bar(relHist[1][0:len(valsRel)], valsRel, width=0.25, color='0.5', edgecolor='0', align='edge')
    ax2.set_ylim([0,ymax])
    ax2.set_xlim([1.0, 4])
    ax2.set_yticklabels([])
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='x', direction='out', top='off')
    ax2.set_title('Fractional', fontsize=16)

    ax1.annotate('Response', xy=(3.5, -0.16), annotation_clip=False, fontsize=16)
    #ax1.annotate('Eigenvalue Distriution', xy=(-4, 0.048), annotation_clip=False, fontsize=16)
    ax1.annotate('(a)', xy=(1.25, 0.4), fontsize=14)
    ax2.annotate('(b)', xy=(1.25, 0.4), fontsize=14)
    if savepath:
        plt.savefig(savepath+"/responseHistog.pdf")

# plot histogram of number of "large" components in each stiff eigenvector
def componentHistograms(results, savepath, axisThr, componentThr):
    def keepTick(v, tickdelta):
        return (v%tickdelta<tickdelta/100 or ((tickdelta-v)%tickdelta<tickdelta/100))
    # count number of large components in each eigenvector
    relCounts = defaultdict(int)
    absCounts = defaultdict(int)
    for (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in results:
        def buildComponentHist(res, counts):
            rdf = pd.DataFrame(res)
            brdf = rdf[(rdf.axisRatio >= axisThr) & (rdf.relComponent.abs() >= componentThr)]
            for cnt in brdf['vecIdx'].value_counts():
                counts[cnt] += 1
        buildComponentHist(resultsAbs[:nBigAbs], absCounts)
        buildComponentHist(resultsRel[:nBigRel], relCounts)

    # plot histograms of number of "large" components in each stiff vector
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.15, top=0.88, bottom=0.25, wspace=0.05)
    xmax = max(max(absCounts.keys()), max(relCounts.keys()))

    bins = np.arange(1,xmax+1)
    absHist = np.array([absCounts[val] if val in absCounts else 0 for val in bins])
    relHist = np.array([relCounts[val] if val in relCounts else 0 for val in bins])

    ymax = 0.15
    ax1 = plt.subplot(1,2, 1)
    vals = absHist/float(absHist.sum())
    ax1.bar(bins, vals, width=1.0, color='0.5', edgecolor='0', align='center')
    ax1.set_ylim([0,ymax])
    ax1.set_xlim(left=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.tick_params(axis='x', direction='out', top='off')
    ax1.set_yticklabels([(str(v) if keepTick(v, 0.1) else '') for v in ax1.get_yticks()])
    ax1.set_ylabel("Fraction", fontsize=16)
    #ax1.set_xlabel("Component Count", fontsize=16)
    ax1.set_title('Sum of squares', fontsize=16)

    ax2 = plt.subplot(1, 2, 2, sharex=ax1)
    vals = relHist/float(relHist.sum())
    ax2.bar(bins, vals, width=1.0, color='0.5', edgecolor='0', align='center')
    ax2.set_ylim([0,ymax])
    ax2.set_xlim(left=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='x', direction='out', top='off')
    ax2.set_yticklabels([])
    #ax2.set_xlabel("Component Count", fontsize=16)
    ax2.set_title('Fractional', fontsize=16)

    ax1.annotate('Component Count', xy=(20, -0.05), annotation_clip=False, fontsize=16)
    ax1.annotate('(a)', xy=(23, .13), fontsize=14)
    ax2.annotate('(b)', xy=(23, .13), fontsize=14)

    if savepath:
        plt.savefig(savepath+"/eigenvecCompHistog.pdf")

#
# plot frequency distribution across parameters and their expected distribution
#
def paramDistribution(results, savepath, axisThr, componentThr):
    # build parameter histograms
    def initParamAccumulators():
        paramTypes = {'r':0, 'k':0, 'x': 0, 'xy':0, 'h':0, 'd':0, 'b0':0}
        return paramTypes

    # build histograms of node properties associated with significant eigenvectors
    def buildHist(nw, res, paramHist):
        res = [res[i] for i in range(len(res)) if res[i]['axisRatio'] >= axisThr and abs(res[i]['relComponent']) >= componentThr]
        for r in res:
            paramHist[r['param']] += 1

    # build histogram of node properties for all nodes
    def buildGlobalHist(nw, paramHist):
        outdeg = nw.out_degree().values()
        nBasal = (np.array(outdeg)==0).sum()
        nConsumer = len(outdeg)-nBasal
        paramHist['r'] += nBasal
        paramHist['k'] += nBasal
        paramHist['x'] += nConsumer
        paramHist['xy'] += nConsumer
        paramHist['h'] += nConsumer
        paramHist['d'] += nConsumer
        paramHist['b0'] += nConsumer

    absParamHist = initParamAccumulators()
    relParamHist = initParamAccumulators()
    globalParamHist = initParamAccumulators()
    for (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in results:
        nw = nx.parse_edgelist(modelInfo['network'], create_using=nx.DiGraph())
        buildHist(nw, resultsAbs[:nBigAbs], absParamHist)
        buildHist(nw, resultsRel[:nBigRel], relParamHist)
        buildGlobalHist(nw, globalParamHist)

    # compute and plot parameter z-scores
    params = ['r', 'k', 'x', 'xy', 'h', 'b0', 'd']
    glTot = float(sum(globalParamHist.values()))
    globalFreq = np.array([globalParamHist[p]/glTot for p in params])
    relTot = float(sum(relParamHist.values()))
    absTot = float(sum(absParamHist.values()))
    absParamHist = [absParamHist[p]/absTot for p in params]
    relParamHist = [relParamHist[p]/relTot for p in params]

    # got values computed, now do the plots with actual and global expected values
    def subplot(ax, hist, glHist, annot, axy, showYLabel=True):
        xVal = np.arange(0, len(hist), 1.0)
        ax.bar(xVal, hist, width=1.0, color='0.5', edgecolor='0', align='center')
        ax.bar(xVal, glHist, width=1.0, color='none', edgecolor='0', align='center')
        ax.annotate(annot, xy=axy, fontsize=16, annotation_clip=False)
        if showYLabel:
            ax.set_ylabel("Fraction", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xticks(xVal)
        ax.set_xlim([xVal[0]-0.5, xVal[-1]+0.5])
        return ax

    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.15, top=0.88, wspace=0.25)

    # left plot
    ax1 = subplot(plt.subplot(1, 2, 1), absParamHist, globalFreq, 'Sum of squares', (1, 0.26))
    ax1.set_xticklabels(params)
    # right plot
    ax2 = subplot(plt.subplot(1, 2, 2), relParamHist, globalFreq, 'Fractional', (1, 0.51), showYLabel=False)
    ax2.set_xticklabels(params)

    ax1.annotate('(a)', xy=(0, .22), fontsize=14)
    ax2.annotate('(b)', xy=(5.5, .43), fontsize=14)

    if savepath:
        plt.savefig(savepath+"/paramDistributions.pdf")


#
# are the species that are involved in stiff eigenvetors randomly placed in the network?
#
# compare distribution of trophic levels to random (tl distr of all species)
# compare degree distr to degree distr of all species
#
def propertyDistrSignificance(results, savepath, axisThr, componentThr):
    def plotZScores(zRel, zAbs, minBin, delta, title, catVals=None, offset=False):
        def subplot(ax, data, annot, axy):
            xVal = np.arange(minBin, delta*len(data)+minBin, delta)
            ax.bar(xVal, data, width=delta, color='0.5')
            ax.annotate(annot, xy=axy)
            ax.axhline(2, c='k', ls='dotted')
            ax.axhline(-2, c='k', ls='dotted')
            ax.set_ylabel("Z-Score")
            if offset:
                ax.set_xticks(xVal + 0.5*delta)
            return ax

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(hspace=0.1)

        annotX = minBin + delta*0.03*max(len(zRel), len(zAbs))
        annotY = 0.95*max(max(zAbs), max(zRel))
        # upper plot
        ax1 = subplot(plt.subplot(2, 1, 1), zAbs, 'Sum of squares deviation', (annotX, annotY))
        ax1.set_xticklabels([])
        #ax1.get_xaxis().set_visible(False)
        # lower plot
        ax2 = subplot(plt.subplot(2, 1, 2, sharey=ax1), zRel, 'Fractional deviation', (annotX, annotY))
        ax2.set_xlabel(title)
        if catVals is not None:
            ax2.set_xticklabels(catVals)

        #plt.title(title, fontsize=12)
        if savepath:
            plt.savefig(savepath+"/ZScores"+title+".pdf")
            plt.savefig(savepath+"/ZScores"+title+".eps")

    # compute normalized deviation from expected value of each category
    # is hasVal is True, a category must have at least on occurence for a non-zero z-score
    def cat_zscore(sample, prob, hasVal=False):
        sample.resize(len(prob), refcheck=False)
        n = sample.sum()
        # compute normalized deviation from expected value of each category
        #zsc = np.zeros(len(sample))
        zsc = np.empty(len(sample))
        #zsc.fill(np.nan)
        zsc.fill(0)
        if n > 0:
            # compute std dev of a draw of n values from global distr
            sd = np.sqrt(n*prob*(1-prob))
            nonzero = sd != 0
            if hasVal:
                nonzero = np.logical_and(nonzero, sample != 0)
            # compute category value deviations = (observed-expected)/sd
            zsc[nonzero] = (sample-n*prob)[nonzero]/sd[nonzero]
        return zsc

    # build parameter histograms
    def initParamAccumulators():
        paramTypes = {'r':0, 'k':0, 'x': 0, 'xy':0, 'h':0, 'd':0, 'b0':0}
        return paramTypes

    # build histograms of node properties associated with significant eigenvectors
    def buildHist(nw, res, paramHist, tls, degs, indegs, outdegs, biomasses):
        indeg = nw.in_degree()
        outdeg = nw.out_degree()
        res = [res[i] for i in range(len(res)) if res[i]['axisRatio'] >= axisThr and abs(res[i]['relComponent']) >= componentThr]
        for r in res:
            paramHist[r['param']] += 1
            tls.append(r['TL'])
            degs.append(r['degree'])
            indegs.append(indeg[str(r['node'])])
            outdegs.append(outdeg[str(r['node'])])
            biomasses.append(math.log10(r['relBiomass']))

    # build histogram of node properties for all nodes
    def buildGlobalHist(nw, tls, degs, indegs, outdegs, paramHist):
        outdeg = nw.out_degree().values()
        nBasal = (np.array(outdeg)==0).sum()
        nConsumer = len(outdeg)-nBasal
        tls.extend(computeSWTL(nw).values())
        degs.extend(nw.degree().values())
        indegs.extend(nw.in_degree().values())
        outdegs.extend(outdeg)
        paramHist['r'] += nBasal
        paramHist['k'] += nBasal
        paramHist['x'] += nConsumer
        paramHist['xy'] += nConsumer
        paramHist['h'] += nConsumer
        paramHist['d'] += nConsumer
        paramHist['b0'] += nConsumer

    absParamHist = initParamAccumulators()
    absTLs = []; absDegrees = []; absInDegrees = []; absOutDegrees = []; absBiomasses = []
    tlsGlobal = []; degGlobal = []; indegGlobal = []; outdegGlobal = []
    globalBiomasses = np.array([])
    relParamHist = initParamAccumulators()
    relTLs = []; relDegrees = []; relInDegrees = []; relOutDegrees = []; relBiomasses = []
    globalParamHist = initParamAccumulators()
    for (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in results[0]:
        nw = nx.parse_edgelist(modelInfo['network'], create_using=nx.DiGraph())
        buildHist(nw, resultsAbs[:nBigAbs], absParamHist, absTLs, absDegrees, absInDegrees, absOutDegrees, absBiomasses)
        buildHist(nw, resultsRel[:nBigRel], relParamHist, relTLs, relDegrees, relInDegrees, relOutDegrees, relBiomasses)
        buildGlobalHist(nw, tlsGlobal, degGlobal, indegGlobal, outdegGlobal, globalParamHist)
        # log10 of relative (normalized by maximum) biomass
        globalBiomasses = np.concatenate((globalBiomasses, np.log10(modelInfo['binit']/modelInfo['binit'].max())))

    # compare relative and absolute distributions against global distributions

    # trophic levels
    # put tls into integer groups
    def makeTLGroups(tls):
        tls = (np.array(tls)*2).round().astype(int)
        grps = np.histogram(tls, bins = tls.max()-1)[0]
        return grps

    tlTot = len(tlsGlobal)
    tlFrac = makeTLGroups(tlsGlobal)/float(tlTot)
    zTLRel = cat_zscore(makeTLGroups(relTLs), tlFrac)
    zTLAbs = cat_zscore(makeTLGroups(absTLs), tlFrac)
    plotZScores(zTLRel, zTLAbs, 1, 0.5, "Trophic Level")
    print("Trophic Level Relative: %s"%str(zTLRel))
    print("Trophic Level Absolute: %s"%str(zTLAbs))

    # degree, in-degree and out-degree
    #
    def computeDegreeZscore(degGl, relDeg, absDeg):
        # degree distribution
        # put degrees into integer groups
        def makeDegGroups(degs):
            degs = np.array(degs, dtype=int)
            return np.histogram(degs, bins = degs.max())[0]

        degTot = len(degGl)
        degFrac = makeDegGroups(degGl)/float(degTot)
        zDegRel = cat_zscore(makeDegGroups(relDeg), degFrac)
        zDegAbs = cat_zscore(makeDegGroups(absDeg), degFrac)
        return zDegRel, zDegAbs

    zDegRel, zDegAbs = computeDegreeZscore(degGlobal, relDegrees, absDegrees)
    plotZScores(zDegRel, zDegAbs, 0, 1, "Degree")
    print("Degree Relative: %s"%str(zDegRel))
    print("Degree Absolute: %s"%str(zDegAbs))

    zDegRel, zDegAbs = computeDegreeZscore(indegGlobal, relInDegrees, absInDegrees)
    plotZScores(zDegRel, zDegAbs, 0, 1, "InDegree")
    print("nPred Relative: %s"%str(zDegRel))
    print("nPred Absolute: %s"%str(zDegAbs))

    zDegRel, zDegAbs = computeDegreeZscore(outdegGlobal, relOutDegrees, absOutDegrees)
    plotZScores(zDegRel, zDegAbs, 0, 1, "OutDegree")
    print("nPrey Relative: %s"%str(zDegRel))
    print("nPrey Absolute: %s"%str(zDegAbs))

    # compute and plot parameter z-scores
    params = ['r', 'k', 'x', 'xy', 'h', 'b0', 'd']
    glTot = float(sum(globalParamHist.values()))
    globalFreq = np.array([globalParamHist[p]/glTot for p in params])
    zParamRel = cat_zscore(np.array([relParamHist[p] for p in params]), globalFreq)
    zParamAbs = cat_zscore(np.array([absParamHist[p] for p in params]), globalFreq)
    plotZScores(zParamRel, zParamAbs, 0, 1, "Parameters", catVals=params, offset=True)

    # compute and plot biomass z-scores
    # trophic levels
    # put tls into integer groups
    def makeBiomassGroups(biomasses):
        biomasses = -(biomasses.round().astype(int))
        grps = np.histogram(biomasses, bins = biomasses.max())[0]
        return grps

    bTot = len(globalBiomasses)
    bFrac = makeBiomassGroups(globalBiomasses)/float(bTot)
    zBRel = cat_zscore(makeBiomassGroups(np.array(relBiomasses)), bFrac, hasVal=True)
    zBAbs = cat_zscore(makeBiomassGroups(np.array(absBiomasses)), bFrac, hasVal=True)
    plotZScores(zBRel, zBAbs, 0, 1, "-Log10 Biomass")
    print("Biomass Relative Z: %s"%str(zBRel))
    print("Biomass Absolute Z: %s"%str(zBAbs))


def plotResults(res, respath=None, axisThr=1.0, componentThr=0.2):
    results, plotPrefix = res

    responseHistograms(results, respath)
    #eigenvalueHistograms(results, respath)
    #componentHistograms(results, respath, axisThr, componentThr)
    #paramDistribution(results, respath, axisThr, componentThr)

    #tlDeltaHistograms(results, plotPrefix)
    #tlDeltaHistograms(results, plotPrefix, basalOnly=True)
