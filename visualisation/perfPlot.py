#! /usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

def fancyPlot(x, style='b-'):
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, style)
    plt.show()
    return fig

outputDir  = '/Users/aFarchi/Desktop/test/Lorenz/'
outputFile = outputDir + 'meanFP.bin'
f          = open(outputFile, 'rb')
perf       = np.fromfile(f)
f.close()

fig = fancyPlot(perf)

