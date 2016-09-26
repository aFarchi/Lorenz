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

Nt     = meanFP.size / 2
meanFP = meanFP.reshape((2,Nt))
FP     = FP.reshape((2,Nt))

fig = plt.figure()
ax  = fig.gca()

#ax.plot(FP[0], 'b--', label='OISIR, instantaneous')
ax.plot(meanFP[0], 'b-', label='OISIR, average')

#ax.plot(FP[1], 'r--', label='EnKF, instantaneous')
ax.plot(meanFP[1], 'r-', label='EnKF, average')

plt.legend(loc='best')
plt.savefig(outputDir+'FP.pdf')

