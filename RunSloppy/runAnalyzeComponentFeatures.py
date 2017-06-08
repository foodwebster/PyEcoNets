# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn import preprocessing
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

sys.path.append("..")
from config import basepath
from Sloppy.analyzeResults import getStaticResults
from Sloppy.componentFeatures import getComponentFeatures

resultspath = os.path.join(basepath,'SloppyResults')
relfFile = os.path.join(resultspath, "relativeFeatures.csv")
absfFile = os.path.join(resultspath, "absoluteFeatures.csv")

if os.path.exists(relfFile):
    # load feature table if it already exists
    reldf = pd.read_csv(relfFile)
    absdf = pd.read_csv(absfFile)
else:
    # compute the feature table and save for future re-use
    results = getStaticResults(resultspath)[0]
    relf, absf = getComponentFeatures(results)
    reldf = pd.DataFrame(relf)
    absdf = pd.DataFrame(absf)
    reldf.to_csv(relfFile, index=False)
    absdf.to_csv(absfFile, index=False)

def graphvizTree(tree, feature_names, fbase):
    f = open(fbase + ".dot", 'w')
    export_graphviz(tree, out_file=f, feature_names=feature_names)
    f.close()

def analyzeDecisionTree(df):
    df = df.fillna(0)
    rowVal = np.log10(df['componentSize'])
    keep = rowVal > -10
    rowVal = rowVal[keep]
    df = df.drop('componentSize', 1)
    df = df[keep]

    dtr = DecisionTreeRegressor(random_state=0)
    dtr.fit(df.values, rowVal)
    return dtr

def analyzeRandomForest(df):
    df = df.fillna(0)
    rowVal = np.log10(df['componentSize'])
    keep = rowVal > -10
    rowVal = rowVal[keep]
    df = df.drop(['param', 'componentSize'], 1)
    df = df[keep]

    rfr = RandomForestRegressor(random_state=0)
    rfr.fit(df.values, rowVal)
    fImp = np.array(rfr.feature_importances_)
    fOrder = fImp.argsort()
    paramScores = zip(df.columns[fOrder], fImp[fOrder])
    paramScores.reverse()
    return rfr, paramScores

def analyzeFeatures(df, name, maxP=[2]):
    print(name)
    rf, paramScores = analyzeRandomForest(df)
    print(paramScores)

    # do least squares on top-N parameters
    df = df.fillna(0)
    df['logCompSize'] = np.log10(df['componentSize'])
    df = df[df['logCompSize'] > -10]
    for maxParam in maxP:
        print("Doing least squares fit for top %d parameters"%maxParam)
        formula = "logCompSize ~ " + '+'.join([sc[0] for sc in paramScores[:maxParam]])
        result = sm.ols(formula=formula, data=df).fit()
        print result.summary()

# clean up and scale the raw features
def buildFeatureDf(alldf, componentThr=0.2):
    # all entities
    alldf = alldf.fillna(0)
    # only "large" components
    df = alldf[alldf['componentSize'] > componentThr]

    # remove unuser/unneeded columns
    drop = ['componentSize']
    initdrop = ['node', 'param']
    # save parameters
    paramVals = df['param']
    # drop non-numeric columns
    df = df.drop(initdrop,1)
    alldf = alldf.drop(initdrop,1)
    # look for numerics with zero range and remove them
    diff = (df.max()-df.min())
    for idx, val in diff.iteritems():
        if val == 0:
            drop.append(idx)
    df = df.drop(drop,1)

    # scale the features and rebuild dataframes
    scaler = preprocessing.StandardScaler().fit(df)
    featuredf = pd.DataFrame(scaler.transform(df), columns=df.columns)
    alldf = alldf.drop(drop,1)
    allscaleddf = pd.DataFrame(scaler.transform(alldf), columns=alldf.columns)
    return allscaleddf, featuredf, paramVals

# get features associated with node of stiff eigenvectors
# and analyze those features
def analyzeStiffVectors(df, title, filename, ncomp=2):
    def jitter(arr, ptSz):
        sd = .005*ptSz*(max(arr)-min(arr))
        return arr + np.random.randn(len(arr)) * sd

    statusLines = []
    alldf, featuredf, paramVals = buildFeatureDf(df)
    statusLines.append(title)
    statusLines.append(str("Number of vector components: %d\n"%len(alldf)))
    statusLines.append(str("Number of stiff vector components: %d\n"%len(featuredf)))

    pca = PCA(n_components=ncomp)
    pcamodel = pca.fit(featuredf)

    X_r = pcamodel.transform(featuredf)
    allX_r = pcamodel.transform(alldf)
    # for axis limits
    xMax = np.ceil(X_r[:, 0].max()*10)/10.0
    xMin = np.floor(X_r[:, 0].min()*10)/10.0
    yMax = np.ceil(X_r[:, 1].max()*10)/10.0
    yMin = np.floor(X_r[:, 1].min()*10)/10.0

    print(title)
    print(pca.explained_variance_ratio_)
    statusLines.append("Explained Variance Fractions:\n")
    statusLines.append(str("%s\n"%str(pca.explained_variance_ratio_)))

    # plot all data points as density map in PCA space
    plt.figure(figsize=(7, 5))
    #plt.scatter(allX_r[:, 0], allX_r[:, 1], marker='.', c='k')
    ax = plt.axes()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #ax.hexbin(allX_r[:, 0], allX_r[:, 1], gridsize=25, cmap='Greys')
    ax.hist2d(allX_r[:, 0], allX_r[:, 1], bins=25, range=[[xMin, xMax], [yMin, yMax]], cmap='Greys')
    # set axis limits
    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)
    plt.title(title, fontsize=14)

    # get PCA component
    cols = featuredf.columns
    for i in range(ncomp):
        statusLines.append(str("\nComponent %d\n"%i))
        cmps = pcamodel.components_[i]
        order = cmps.argsort()
        for j in order:
            statusLines.append(str("%s: %f\n"%(cols[j], cmps[j])))
    featuredf.columns

    # plot large parameters in PCA coordinate space
    #plt.figure(figsize=(6, 4.5))
    #ax = plt.axes()
    colors = ['blue', 'turquoise', 'red', 'orange', 'darkviolet', 'brown', 'limegreen']
    params = ['r', 'k', 'x', 'xy', 'h', 'd', 'b0']
    plts = []
    finalParams = []
    # plot each parameter in a different color
    for color, param in zip(colors, params):
        paramIdx = (paramVals==param).values
        if paramIdx.sum() > 0: # make sure at least one of the parameter exists
            plts.append(plt.scatter(jitter(X_r[paramIdx, 0],1), jitter(X_r[paramIdx, 1],1), s=3, marker='o', facecolors=color, edgecolors=color))
            finalParams.append(param)
    # set axis limits
    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)
    # add legend and format plot
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(plts, finalParams, scatterpoints=1, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.title(title, fontsize=14)
    # save plot to file
    plt.savefig(os.path.join(resultspath, filename+".pdf"))
    # save pca info to text file
    txtFile = open(os.path.join(resultspath, filename+".txt"), 'w')
    txtFile.writelines(statusLines)
    txtFile.close()
    return pcamodel

def runNMDS(df):
    alldf, featuredf, paramVals = buildFeatureDf(df)
    sims = euclidean_distances(featuredf)
    nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                        dissimilarity="precomputed", random_state=0, n_jobs=1,
                        n_init=1)
    npos = nmds.fit_transform(sims)
    colors = ['blue', 'turquoise', 'red', 'orange', 'darkviolet', 'brown', 'limegreen']
    params = ['r', 'k', 'x', 'xy', 'h', 'd', 'b0']
    colorMap = dict(zip(params, colors))
    colorArray = [colorMap[p] for p in paramVals]
    plt.scatter(npos[:,0], npos[:,1], c=colorArray)

'''
prodreldf = reldf[reldf['TL']<=1.0]
consreldf = reldf[reldf['TL']>1.0]
prodabsdf = absdf[absdf['TL']<=1.0]
consabsdf = absdf[absdf['TL']>1.0]

analyzeStiffVectors(reldf, "Fractional", "RelativePCA")
analyzeStiffVectors(consreldf, "Consumers, Relative", "RelativePCACons")
analyzeStiffVectors(prodreldf, "Producers, Relative", "RelativePCAProd")

analyzeStiffVectors(absdf, "Absolute", "AbsolutePCA")
analyzeStiffVectors(consabsdf, "Consumers, Absolute", "AbsolutePCACons")
analyzeStiffVectors(prodabsdf, "Producers, Absolute", "AbsolutePCAProd")
'''

def analyzeNodes(alldf, bigdf, params, title, filename, ncomp=2):
    def jitter(arr, ptSz):
        sd = .005*ptSz*(max(arr)-min(arr))
        return arr + np.random.randn(len(arr)) * sd

    statusLines = []
    statusLines.append(title)
    statusLines.append(str("Number of vector components: %d\n"%len(alldf)))
    statusLines.append(str("Number of stiff vector components: %d\n"%len(bigdf)))

    pca = PCA(n_components=ncomp)
    pcamodel = pca.fit(bigdf)

    X_r = pcamodel.transform(bigdf)
    allX_r = pcamodel.transform(alldf)
    # for axis limits
    xMax = np.ceil(X_r[:, 0].max()*10)/10.0
    xMin = np.floor(X_r[:, 0].min()*10)/10.0
    yMax = np.ceil(X_r[:, 1].max()*10)/10.0
    yMin = np.floor(X_r[:, 1].min()*10)/10.0

    print(title)
    print(pca.explained_variance_ratio_)
    statusLines.append("Explained Variance Fractions:\n")
    statusLines.append(str("%s\n"%str(pca.explained_variance_ratio_)))

    # plot all data points as density map in PCA space
    plt.figure(figsize=(7, 5))
    ax = plt.axes()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #ax.hexbin(allX_r[:, 0], allX_r[:, 1], gridsize=25, cmap='Greys')
    ax.hist2d(allX_r[:, 0], allX_r[:, 1], bins=25, range=[[xMin, xMax], [yMin, yMax]], cmap='Greys')
    # set axis limits
    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)
    plt.title(title, fontsize=14)

    # get PCA component
    cols = bigdf.columns
    for i in range(ncomp):
        statusLines.append(str("\nComponent %d\n"%i))
        cmps = pcamodel.components_[i]
        order = cmps.argsort()
        for j in order:
            statusLines.append(str("%s: %f\n"%(cols[j], cmps[j])))
    bigdf.columns

    # plot large parameters in PCA coordinate space
    colorList = [('r', 'turquoise'), ('k', 'cyan'), ('k,r', 'lime'),
                ('xy', 'red'), ('x,xy', 'orange'), ('x,xy,b0', 'orangered'), ('x', 'crimson'),
                ('h', 'darkviolet'), ('h,x', 'magenta'), ('h,xy', 'blue'),
                ('h,xy,x', 'cornflowerblue'), ('h,b0', 'green'), ('h,b0,x', 'green'),
                ('h,xy,b0', 'darkgreen'), ('xy,b0', 'seagreen'),
                ('h,xy,b0,x', 'darkgreen'), ('other', 'grey')]
    colorMap = dict(colorList)
    paramEq = {'x,xy,h':'h,xy,x', 'x,h':'h,x'}
    paramCnts = params.value_counts()
    paramNames = {pn:(pn if val > 5 else "other") for pn,val in paramCnts.iteritems()}
    paramVals = params.apply(lambda x: paramNames[x])
    plts = {}
    # plot each parameter in a different color
    for param in paramNames.itervalues():
        paramIdx = (paramVals==param).values
        if param in paramEq:
            param = paramEq[param]
        color = colorMap[param]
        plts[param] = plt.scatter(X_r[paramIdx, 0], X_r[paramIdx, 1], s=8, marker='o', facecolors=color, edgecolors=color, label=param)
    # set axis limits
    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)
    # add legend and format plot
    pltParams = [ci[0] for ci in colorList if ci[0] in plts]
    plts = [plts[ci[0]] for ci in colorList if ci[0] in plts]
    plt.legend(plts, pltParams, scatterpoints=1,
               loc='center left', bbox_to_anchor=(1, 0.5),
               labelspacing=0.3, markerscale=1.5,
               handletextpad=0, borderpad=0, frameon=False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.title(title, fontsize=14)
    # save plot to file
    plt.savefig(os.path.join(resultspath, filename+".pdf"))
    # save pca info to text file
    txtFile = open(os.path.join(resultspath, filename+".txt"), 'w')
    txtFile.writelines(statusLines)
    txtFile.close()
    return pcamodel

def getNodeFeaturesDf(alldf, axisThr=0.7, componentThr=0.2):
    def getNodes(df):
        nodegrps = df.groupby('node')
        #cnts = nodegrps.size().rename('count')
        plists = nodegrps['param'].apply(lambda x: ','.join(list(set(x))))
        nodedf = nodegrps[cols].first()
        #nodedf = nodedf.join(cnts)
        #nodedf = nodedf.join(plists)
        #nodedf['paramCnt'] = nodedf['param'].apply(lambda x: len(x.split(',')))
        return nodedf, plists
    cols = [elem for elem in alldf.columns.tolist() if elem not in set(['node', 'param', 'vecSize', 'componentSize'])]
    # keep only largest eigenvalue
    df = alldf[alldf['vecSize'] > axisThr]
    # keep large components of largest eigenvec
    df = df[df['componentSize'] > componentThr]
    df, params = getNodes(df)
    alldf, allparams = getNodes(alldf)
    # scale the features and rebuild dataframes
    scaler = preprocessing.StandardScaler().fit(df)
    featuredf = pd.DataFrame(scaler.transform(df), columns=df.columns)
    allscaleddf = pd.DataFrame(scaler.transform(alldf), columns=alldf.columns)
    return allscaleddf, featuredf, params

allabsdf, bigabsdf, params = getNodeFeaturesDf(absdf)
allreldf, bigreldf, params = getNodeFeaturesDf(reldf)

analyzeNodes(allabsdf, bigabsdf, params, 'Absolute', 'AbsNodePCA')
analyzeNodes(allreldf, bigreldf, params, 'Fractional', 'RelNodePCA')

analyzeNodes(allabsdf[allabsdf['TL']<=1.0], bigabsdf[bigabsdf['TL']<=1.0], params, 'Absolute', 'AbsProdNodePCA')
analyzeNodes(allreldf[allreldf['TL']<=1.0], bigreldf[bigreldf['TL']<=1.0], params, 'Fractional', 'RelProdNodePCA')

analyzeNodes(allabsdf[allabsdf['TL']>1.0], bigabsdf[bigabsdf['TL']>1.0], params, 'Absolute', 'AbsConsNodePCA')
analyzeNodes(allreldf[allreldf['TL']>1.0], bigreldf[bigreldf['TL']>1.0], params, 'Fractional', 'RelConsNodePCA')

#analyzeFeatures(reldf, "Relative", maxP=range(1,6))
#analyzeFeatures(absdf, "Absolute", maxP=range(1,6))

