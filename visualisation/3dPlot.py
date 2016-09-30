#! /usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fancyPlot(states, style='b-'):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot(states[:,0], states[:,1], states[:,2], style)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return fig

def fancyPlot2(states1, states2, style1='b-', style2='r-'):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot(states1[:,0], states1[:,1], states1[:,2], style1)
    ax.plot(states2[:,0], states2[:,1], states2[:,2], style2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return fig

def fancyPlot3(states1, states2, states3, style1='b-', style2='r-', style3='g+'):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot(states1[:,0], states1[:,1], states1[:,2], style1)
    ax.plot(states2[:,0], states2[:,1], states2[:,2], style2)
    ax.plot(states3[:,0], states3[:,1], states3[:,2], style3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return fig

def fancyPlot4(states1, states2, states3, states4, style1='b-', style2='r-', style3='g+', style4='ro'):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot(states1[:,0], states1[:,1], states1[:,2], style1)
    ax.plot(states2[:,0], states2[:,1], states2[:,2], style2)
    ax.plot(states3[:,0], states3[:,1], states3[:,2], style3)
    ax.plot(states4[:,0], states4[:,1], states4[:,2], style4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return fig

outputDir  = '/Users/aFarchi/Desktop/test/Lorenz/'
outputFile = outputDir + 'xt_record.bin'
f          = open(outputFile, 'rb')
truth      = np.fromfile(f)
f.close()

outputFile = outputDir + 'xa_record.bin'
f          = open(outputFile, 'rb')
analyse    = np.fromfile(f)
f.close()

outputFile = outputDir + 'xo_record.bin'
f          = open(outputFile, 'rb')
obs        = np.fromfile(f)
f.close()

outputFile = outputDir + 'nt_obs.bin'
f          = open(outputFile, 'rb')
ntObs      = np.fromfile(f).astype(int)
f.close()

#outputFile = outputDir + 'nt_resampling.bin'
#f          = open(outputFile, 'rb')
#resampling = np.fromfile(f).astype(int)
#f.close()

Nt      = truth.size / 3
truth   = truth.reshape((Nt, 3))
analyse = analyse.reshape((Nt, 3))
obs     = obs.reshape((Nt, 3))[ntObs, :]

#resampling = resampling.tolist()
#resampling = [i for i in resampling if i >= imin]

ntMin = min(50, Nt)
ntMax = min(200, Nt)

#fig = fancyPlot(truth[ntMin:ntMax])
fig = fancyPlot3(truth[ntMin:ntMax], analyse[ntMin:ntMax], obs[25:100])

#fig = fancyPlot2(states1, states2)
#fig = fancyPlot4(states1[imin:], states2[imin:], states3[imin:], states2[resampling])

