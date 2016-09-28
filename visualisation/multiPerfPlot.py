#! /usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

outputDir  = '/Users/aFarchi/Desktop/test/Lorenz/'
outputFile = outputDir + 'meanFP.bin'
f          = open(outputFile, 'rb')
meanFP     = np.fromfile(f)
f.close()

outputDir  = '/Users/aFarchi/Desktop/test/Lorenz/'
outputFile = outputDir + 'FP.bin'
f          = open(outputFile, 'rb')
FP         = np.fromfile(f)
f.close()

nfilters = 3
Nt       = meanFP.size / nfilters
meanFP   = meanFP.reshape((nfilters, Nt))
FP       = FP.reshape((nfilters, Nt))

fig = plt.figure()
ax  = fig.gca()

#ax.plot(FP[0], 'b--', label='OISIR, instantaneous')
ax.plot(meanFP[0], 'b-', label='OISIR, average')

#ax.plot(FP[1], 'r--', label='RMI, instantaneous')
ax.plot(meanFP[1], 'r-', label='RMI, average')

#ax.plot(FP[2], 'c--', label='EnKF, instantaneous')
ax.plot(meanFP[2], 'c-', label='EnKF, average')

plt.legend(loc='best')
plt.savefig(outputDir+'FP.pdf')

