# -*- coding: utf-8 -*-

import os
from os.path import expanduser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(expanduser("~/OneDrive/Rich/Research/Sharks"))

def plotSharkInvasionResults():
    invfile = "SharkInvasionResults.csv"
    
    df = pd.read_csv(invfile)
    nInv = df['nInvader'].max()
    
    resultsDf = df.groupby(by=['frac', 'nInvader']).mean()
    imShape = (4,nInv)
    predloss = resultsDf['mnPredLoss'].reshape(imShape)
    preyloss = resultsDf['mnPreyLoss'].reshape(imShape)
    fpredloss = resultsDf['fracPredLoss'].reshape(imShape)
    fpreyloss = resultsDf['fracPreyLoss'].reshape(imShape)
    
    plt.figure(figsize=(7, 5.5))
    plt.subplots_adjust(hspace=0., wspace=0.2, top=0.88, left=0.15, bottom=0.1, right=0.93)
    
    ax1 = plt.subplot(2, 2, 1)
    cax1 = ax1.imshow(predloss, interpolation='nearest', cmap=plt.cm.binary, vmin=0, vmax=0.7)
    ax1.set_xticklabels([])
    ax1.set_ylabel("Fraction Extinct")
    ax1.set_yticks(np.arange(0,4))
    ax1.set_yticklabels(['0.25','0.5','0.75','1'])
    ax1.set_title("Mean Predator Loss", fontsize=12)
    # Add grayscale bar
    cbar = plt.colorbar(cax1,fraction=0.031, pad=0.05, ticks=[0.1, 0.3, 0.5, 0.7])
    cbar.ax.set_yticklabels(['0.1', '0.3', '0.5', '0.7'])  # vertically oriented colorbar
    
    ax1.annotate("Shark Invasions", xy=(4,-1.5), annotation_clip=False, fontsize=14)
    
    #ax1.set_yticks([0,1,2,3])
    #ax1.set_yticklabels(['0.25','0.5','0.75','1'])
    ax2 = plt.subplot(2, 2, 2)
    cax2 = ax2.imshow(preyloss, interpolation='nearest', cmap=plt.cm.binary, vmin=0, vmax=0.5)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_title("Mean Prey Loss", fontsize=12)
    # Add grayscale bar
    cbar = plt.colorbar(cax2,fraction=0.031, pad=0.05, ticks=[0, 0.25, 0.5])
    cbar.ax.set_yticklabels(['0', '0.25', '0.5'])  # vertically oriented colorbar
    
    ax3 = plt.subplot(2, 2, 3)
    cax3 = ax3.imshow(fpredloss, interpolation='nearest', cmap=plt.cm.binary, vmin=0.5, vmax=1)
    ax3.set_xlabel("Number of Invaders")
    ax3.set_xticks(np.arange(0,nInv))
    ax3.set_xticklabels([str(n) for n in range(1, nInv+1)])
    ax3.set_ylabel("Fraction Extinct")
    ax3.set_yticks(np.arange(0,4))
    ax3.set_yticklabels(['0.25','0.5','0.75','1'])
    ax3.set_title("Fraction spp w/ predator loss", fontsize=12)
    # Add grayscale bar
    cbar = plt.colorbar(cax3,fraction=0.031, pad=0.05, ticks=[0.5, 0.75, 1])
    cbar.ax.set_yticklabels(['0.5', '0.75', '1'])  # vertically oriented colorbar
    
    ax4 = plt.subplot(2, 2, 4)
    cax4 = ax4.imshow(fpreyloss, interpolation='nearest', cmap=plt.cm.binary, vmin=0, vmax=0.4)
    ax4.set_xlabel("Number of Invaders")
    ax4.set_xticks(np.arange(0,nInv))
    ax4.set_xticklabels([str(n) for n in range(1, nInv+1)])
    ax4.set_yticklabels([])
    ax4.set_title("Fraction spp w/ prey loss", fontsize=12)
    # Add grayscale bar
    cbar = plt.colorbar(cax4,fraction=0.031, pad=0.05, ticks=[0, 0.2, 0.4])
    cbar.ax.set_yticklabels(['0', '0.2', '0.4'])  # vertically oriented colorbar

    plt.savefig("SharkInvasions.pdf")
    
def plotCrabInvasionResults():
    invfile = "CrabInvasionResults.csv"
    
    df = pd.read_csv(invfile)
    nInv = df['nInvader'].max()
    
    resultsDf = df.groupby(by=['frac', 'nInvader']).mean()
    imShape = (4,df['nInvader'].max())
    predloss = resultsDf['mnPredLoss'].reshape(imShape)
    preyloss = resultsDf['mnPreyLoss'].reshape(imShape)
    fpredloss = resultsDf['fracPredLoss'].reshape(imShape)
    fpreyloss = resultsDf['fracPreyLoss'].reshape(imShape)
    
    plt.figure(figsize=(7, 5.5))
    plt.subplots_adjust(hspace=0., wspace=0.2, top=0.88, left=0.15, bottom=0.1, right=0.93)
    
    ax1 = plt.subplot(2, 2, 1)
    cax1 = ax1.imshow(predloss, interpolation='nearest', cmap=plt.cm.binary, vmin=0, vmax=0.7)
    ax1.set_xticklabels([])
    ax1.set_ylabel("Fraction Extinct")
    ax1.set_yticks(np.arange(0,4))
    ax1.set_yticklabels(['0.25','0.5','0.75','1'])
    ax1.set_title("Mean Predator Loss", fontsize=12)
    # Add grayscale bar
    cbar = plt.colorbar(cax1,fraction=0.031, pad=0.05, ticks=[0.1, 0.3, 0.5, 0.7])
    cbar.ax.set_yticklabels(['0.1', '0.3', '0.5', '0.7'])  # vertically oriented colorbar
    
    ax1.annotate("Crab Invasions", xy=(4,-1.5), annotation_clip=False, fontsize=14)
    
    #ax1.set_yticks([0,1,2,3])
    #ax1.set_yticklabels(['0.25','0.5','0.75','1'])
    ax2 = plt.subplot(2, 2, 2)
    cax2 = ax2.imshow(preyloss, interpolation='nearest', cmap=plt.cm.binary, vmin=0, vmax=0.5)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_title("Mean Prey Loss", fontsize=12)
    # Add grayscale bar
    cbar = plt.colorbar(cax2,fraction=0.031, pad=0.05, ticks=[0, 0.25, 0.5])
    cbar.ax.set_yticklabels(['0', '0.25', '0.5'])  # vertically oriented colorbar
    
    ax3 = plt.subplot(2, 2, 3)
    cax3 = ax3.imshow(fpredloss, interpolation='nearest', cmap=plt.cm.binary, vmin=0.5, vmax=1)
    ax3.set_xlabel("Number of Invaders")
    ax3.set_xticks(np.arange(0,nInv))
    ax3.set_xticklabels([str(n) for n in range(1, nInv+1)])
    ax3.set_ylabel("Fraction Extinct")
    ax3.set_yticks(np.arange(0,4))
    ax3.set_yticklabels(['0.25','0.5','0.75','1'])
    ax3.set_title("Fraction spp w/ predator loss", fontsize=12)
    # Add grayscale bar
    cbar = plt.colorbar(cax3,fraction=0.031, pad=0.05, ticks=[0.5, 0.75, 1])
    cbar.ax.set_yticklabels(['0.5', '0.75', '1'])  # vertically oriented colorbar
    
    ax4 = plt.subplot(2, 2, 4)
    cax4 = ax4.imshow(fpreyloss, interpolation='nearest', cmap=plt.cm.binary, vmin=0, vmax=0.4)
    ax4.set_xlabel("Number of Invaders")
    ax4.set_xticks(np.arange(0,nInv))
    ax4.set_xticklabels([str(n) for n in range(1, nInv+1)])
    ax4.set_yticklabels([])
    ax4.set_title("Fraction spp w/ prey loss", fontsize=12)
    # Add grayscale bar
    cbar = plt.colorbar(cax4,fraction=0.031, pad=0.05, ticks=[0, 0.2, 0.4])
    cbar.ax.set_yticklabels(['0', '0.2', '0.4'])  # vertically oriented colorbar

    plt.savefig("CrabInvasions.pdf")
    
def plotSharkCrabInvasionResults():
    invfile = "SharkCrabInvasionResults.csv"
    
    df = pd.read_csv(invfile)
    nInv = df['nInvader'].max()
    
    resultsDf = df.groupby(by=['frac', 'nInvader']).mean()
    imShape = (4,df['nInvader'].max())
    predloss = resultsDf['mnPredLoss'].reshape(imShape)
    preyloss = resultsDf['mnPreyLoss'].reshape(imShape)
    fpredloss = resultsDf['fracPredLoss'].reshape(imShape)
    fpreyloss = resultsDf['fracPreyLoss'].reshape(imShape)
    
    plt.figure(figsize=(7, 5.5))
    plt.subplots_adjust(hspace=0., wspace=0.2, top=0.88, left=0.15, bottom=0.1, right=0.93)
    
    ax1 = plt.subplot(2, 2, 1)
    cax1 = ax1.imshow(predloss, interpolation='nearest', cmap=plt.cm.binary, aspect=2, vmin=0, vmax=0.7)
    ax1.set_xticklabels([])
    ax1.set_ylabel("Fraction Extinct")
    ax1.set_yticks(np.arange(0,4))
    ax1.set_yticklabels(['0.25','0.5','0.75','1'])
    ax1.set_title("Mean Predator Loss", fontsize=12)
    # Add grayscale bar
    cbar = plt.colorbar(cax1,fraction=0.031, pad=0.05, ticks=[0.1, 0.3, 0.5, 0.7])
    cbar.ax.set_yticklabels(['0.1', '0.3', '0.5', '0.7'])  # vertically oriented colorbar
    
    ax1.annotate("Shark and Crab Invasions", xy=(4,-1.5), annotation_clip=False, fontsize=14)
    
    #ax1.set_yticks([0,1,2,3])
    #ax1.set_yticklabels(['0.25','0.5','0.75','1'])
    ax2 = plt.subplot(2, 2, 2)
    cax2 = ax2.imshow(preyloss, interpolation='nearest', cmap=plt.cm.binary, aspect=2, vmin=0, vmax=0.5)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_title("Mean Prey Loss", fontsize=12)
    # Add grayscale bar
    cbar = plt.colorbar(cax2,fraction=0.031, pad=0.05, ticks=[0, 0.25, 0.5])
    cbar.ax.set_yticklabels(['0', '0.25', '0.5'])  # vertically oriented colorbar
    
    ax3 = plt.subplot(2, 2, 3)
    cax3 = ax3.imshow(fpredloss, interpolation='nearest', cmap=plt.cm.binary, aspect=2, vmin=0.5, vmax=1)
    ax3.set_xlabel("Number of Invaders")
    ax3.set_xticks(np.arange(0,nInv))
    ax3.set_xticklabels([str(n) if n%3==0 else "" for n in range(1, nInv+1)])
    ax3.set_ylabel("Fraction Extinct")
    ax3.set_yticks(np.arange(0,4))
    ax3.set_yticklabels(['0.25','0.5','0.75','1'])
    ax3.set_title("Fraction spp w/ predator loss", fontsize=12)
    # Add grayscale bar
    cbar = plt.colorbar(cax3,fraction=0.031, pad=0.05, ticks=[0.5, 0.75, 1])
    cbar.ax.set_yticklabels(['0.5', '0.75', '1'])  # vertically oriented colorbar
    
    ax4 = plt.subplot(2, 2, 4)
    cax4 = ax4.imshow(fpreyloss, interpolation='nearest', cmap=plt.cm.binary, aspect=2, vmin=0, vmax=0.4)
    ax4.set_xlabel("Number of Invaders")
    ax4.set_xticks(np.arange(0,nInv))
    ax4.set_xticklabels([str(n) if n%3==0 else "" for n in range(1, nInv+1)])
    ax4.set_yticklabels([])
    ax4.set_title("Fraction spp w/ prey loss", fontsize=12)
    # Add grayscale bar
    cbar = plt.colorbar(cax4,fraction=0.031, pad=0.05, ticks=[0, 0.2, 0.4])
    cbar.ax.set_yticklabels(['0', '0.2', '0.4'])  # vertically oriented colorbar

    plt.savefig("SharkCrabInvasions.pdf")

#plotSharkInvasionResults()
#plotCrabInvasionResults()
plotSharkCrabInvasionResults()
