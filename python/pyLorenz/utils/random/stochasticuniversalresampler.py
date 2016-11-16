#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/random/
# stochasticuniversalresampler.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# class to handle a Stochastic Universal Resampler
#

import numpy as np

from utils.random.abstractresampler import AbstractResampler

#__________________________________________________

class StochasticUniversalResampler(AbstractResampler):

    #_________________________

    def __init__(self, t_rng):
        AbstractResampler.__init__(self, t_rng)

    #_________________________

    def sampleIndices(self, t_Ns, t_w):
        # sample indices according to Proba(X = i) = exp(t_w[i])
        i   = np.arange(t_Ns)
        # cumulative weights
        wc  = np.exp(t_w).cumsum()

        if wc[-1] == 0.0: 
            return i

        # draw random number in [0,1/t_Ns]
        cursor = self.m_rng.rand() * wc[-1] / t_Ns
        sample = 0

        for ns in range(t_Ns):
            # search for particle to duplicate
            while wc[sample] < cursor:
                sample += 1
            i[ns]   = sample
            # forward cursor
            cursor += wc[-1] / t_Ns
        return i

#__________________________________________________

