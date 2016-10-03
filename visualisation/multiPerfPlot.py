#! /usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

outputDir  = '/Users/aFarchi/Desktop/test/Lorenz/50-10000/'

outputFile = outputDir + 'meanFP.bin'
f          = open(outputFile, 'rb')
meanFP     = np.fromfile(f)
f.close()
outputFile = outputDir + 'FP.bin'
f          = open(outputFile, 'rb')
FP         = np.fromfile(f)
f.close()

nfilters = 4
Nt       = meanFP.size / nfilters
meanFP   = meanFP.reshape((nfilters, Nt))
FP       = FP.reshape((nfilters, Nt))

fig = plt.figure()
ax  = fig.gca()

ax.plot(meanFP[0], label='EnKF')
ax.plot(meanFP[1], label='SIR')
ax.plot(meanFP[2], label='OISIR')
ax.plot(meanFP[3], label='ASIR')
#ax.plot(meanFP[4], label='MSIR')
#ax.plot(meanFP[5], label='AMSIR')

ax.set_ylabel('cummean RMSE')

plt.legend(loc='best')
plt.savefig(outputDir+'FP.pdf')

