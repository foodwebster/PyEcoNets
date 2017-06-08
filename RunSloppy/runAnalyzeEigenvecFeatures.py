# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

sys.path.append("..")
from config import basepath
from Sloppy.analyzeResults import getStaticResults
from Sloppy.eigenvecFeatures import getResultsFeatures

resultspath = os.path.join(basepath,'SloppyResults')
relfFile = os.path.join(resultspath, "relVecFeatures.csv")
absfFile = os.path.join(resultspath, "absVecFeatures.csv")

if os.path.exists(relfFile):
    reldf = pd.read_csv(relfFile)
    absdf = pd.read_csv(absfFile)
else:
    results = getStaticResults(resultspath)[0]
    relf, absf = getResultsFeatures(results)
    reldf = pd.DataFrame(relf)
    absdf = pd.DataFrame(absf)
    reldf.to_csv(relfFile, index=False)
    absdf.to_csv(absfFile, index=False)

def buildFeatureDf(alldf):
    # all entities
    alldf = alldf.fillna(0)
    # only "large" components
    df = alldf[alldf['relEigval'] > 0.2]

    # remove unuser/unneeded columns
    drop = ['relEigval']
    initdrop = ['params', 'nParam', 'nSpecies']
    # save parameters
    paramVals = df['params']
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
    plts = []
    '''
    colors = ['blue', 'turquoise', 'red', 'orange', 'darkviolet', 'brown', 'limegreen']
    params = ['r', 'k', 'x', 'xy', 'h', 'd', 'b0']
    finalParams = []
    # plot each parameter in a different color
    for color, param in zip(colors, params):
        paramIdx = (paramVals==param).values
        if paramIdx.sum() > 0: # make sure at least one of the parameter exists
            plts.append(plt.scatter(jitter(X_r[paramIdx, 0],1), jitter(X_r[paramIdx, 1],1), s=3, marker='o', facecolors=color, edgecolors=color))
            finalParams.append(param)
    '''
    plts.append(plt.scatter(jitter(X_r[:,0],1), jitter(X_r[:,1],1), s=3, marker='o', facecolors='b', edgecolors='b'))
    # set axis limits
    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)
    # add legend and format plot
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #plt.legend(plts, finalParams, scatterpoints=1, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.title(title, fontsize=14)
    # save plot to file
    plt.savefig(os.path.join(resultspath, filename+".pdf"))
    # save pca info to text file
    txtFile = open(os.path.join(resultspath, filename+".txt"), 'w')
    txtFile.writelines(statusLines)
    txtFile.close()
    return pcamodel

analyzeStiffVectors(reldf, "Fractional", "RelVelPCA")
analyzeStiffVectors(absdf, "Absolute", "AbsVecPCA")

