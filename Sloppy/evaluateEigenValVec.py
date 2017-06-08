# -*- coding: utf-8 -*-
#
# fns to do simple assessments of eignevalues and eigenvectors
#

import numpy as np
import matplotlib.pyplot as plt

# plot eigenvalues and eigenvectors
# plot absolute and relative cost eigenvalues
def plotEigenvalues(values, title, minVal):
    plt.figure()
    i = 0
    for vals in values:
        eig = vals[vals>0]
        eig = np.log10(eig/eig.max())
        plt.scatter(np.ones(len(eig))*i, eig, marker='_')
        i += 1
    plt.ylim(minVal, 1)
    plt.xlim(-1, len(values))
    plt.title(title)
    plt.show()

# plot 10 stiffest eigenvectors
def plotEigenvectors(vecs, title):
    x = np.arange(len(vecs[0]))
    plt.figure()
    plt.subplots_adjust(hspace=0.001)
    ax1 = plt.subplot(10, 1, 1)
    plt.title(title)
    ax1.plot(x, vecs[0])
    #plt.yticks(np.arange(-0.8, 1.0, 0.4))
    plt.ylim(-1, 1)

    plt.subplot(10, 1, 2, sharex=ax1).plot(x, vecs[1])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 3, sharex=ax1).plot(x, vecs[2])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 4, sharex=ax1).plot(x, vecs[3])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 5, sharex=ax1).plot(x, vecs[4])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 6, sharex=ax1).plot(x, vecs[5])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 7, sharex=ax1).plot(x, vecs[6])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 8, sharex=ax1).plot(x, vecs[7])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 9, sharex=ax1).plot(x, vecs[8])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 10, sharex=ax1).plot(x, vecs[9])
    plt.ylim(-1, 1)
    plt.show()

# graphically compare finite difference and LM approx eigenvalues
def compareEigvals(valFD, valLM, title):
    lmlog = np.log10(valLM)
    lmnorm = lmlog - lmlog[np.logical_not(np.isnan(lmlog))].max()
    fdlog = np.log10(valFD)
    fdnorm = fdlog - fdlog[np.logical_not(np.isnan(fdlog))].max()
    plt.figure()
    plt.scatter(lmnorm, fdnorm, c='k')
    plt.ylim(-12, 0)
    plt.xlim(-20, 0)
    plt.xlabel(r'LM $log(\lambda/\lambda_{max})$')
    plt.ylabel(r'FiniteDiff $log(\lambda/\lambda_{max})$')
    plt.title(title)
    plt.show()

# compare 5 stiffest eigenvectors
def compareEigvecs(vec1, vec2, title):
    x = np.arange(len(vec1[0]))
    plt.figure()
    plt.subplots_adjust(hspace=0.001)
    plt.title(title)
    ax1 = plt.subplot(10, 1, 1)
    ax1.plot(x, vec1[0])
    #plt.yticks(np.arange(-0.8, 1.0, 0.4))
    plt.ylim(-1, 1)

    plt.subplot(10, 1, 2, sharex=ax1).plot(x, vec2[0])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 3, sharex=ax1).plot(x, vec1[1])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 4, sharex=ax1).plot(x, vec2[1])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 5, sharex=ax1).plot(x, vec1[2])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 6, sharex=ax1).plot(x, vec2[2])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 7, sharex=ax1).plot(x, vec1[3])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 8, sharex=ax1).plot(x, vec2[3])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 9, sharex=ax1).plot(x, vec1[4])
    plt.ylim(-1, 1)
    plt.subplot(10, 1, 10, sharex=ax1).plot(x, vec2[4])
    plt.ylim(-1, 1)
    plt.show()
