#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/random/
# stochasticuniversalsampler.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# class to handle a Stochastic Universal Resampler
#

import numpy as np
import numpy.random as rnd

from abstractresampler import AbstractResampler

#__________________________________________________

class StochasticUniversalResampler(AbstractResampler):

    #_________________________

    def __init__(self):
        AbstractResampler.__init__(self)

    #_________________________

    def sampleIndices(self, t_Ns, t_w):
        # sample indices according to Proba(X = i) = exp(t_w[i])
        i   = np.arange(t_Ns)
        # cumulative weights
        wc  = np.exp(t_w).cumsum()

        # draw random number in [0,1/t_Ns]
        cursor = rnd.rand() * wc[-1] / t_Ns
        sample = 0

        for ns in np.arange(t_Ns):
            # search for particle to duplicate
            while wc[sample] < cursor:
                sample += 1
            i[ns]   = sample
            # forward cursor
            cursor += wc[-1] / t_Ns
        return i

#__________________________________________________

