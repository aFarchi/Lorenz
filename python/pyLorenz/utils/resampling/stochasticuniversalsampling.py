#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/resampling/
# stochasticuniversalsampling.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/29
#__________________________________________________
#
# class to handle a Stochastic Universal Resampler
#

import numpy as np
import numpy.random as rnd

#__________________________________________________

class StochasticUniversalResampler(object):

    #_________________________

    def __init__(self):
        pass

    #_________________________

    def sampleIndices(self, t_w):
        # sample indices according to Proba(X = i) = exp(t_w[i])
        # number of particles
        Ns  = t_w.size
        # cumulative weights
        wc  = np.exp(t_w).cumsum()
        # make sure weights are normalized
        wc /= wc[-1]
        # array for the new indices
        i   = np.arange(Ns)
        # draw random number in [0,1/Ns]
        cursor = rnd.rand() / Ns
        sample = 0
        for ns in np.arange(Ns):
            # search for particle to duplicate
            while wc[sample] < cursor:
                sample += 1
            i[ns]   = sample
            # forward cursor
            cursor += 1.0 / Ns
        return i

    #_________________________

    def resample(self, t_w, t_x):
        # resample particles t_x from the weights t_w
        indices = self.sampleIndices(t_w)
        Ns      = t_w.size
        return (-np.ones(Ns)*np.log(Ns), t_x[indices])

#__________________________________________________

