#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/resampling/
# directresampling.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/29
#__________________________________________________
#
# class to handle a Direct Resampler
#

import numpy as np
import numpy.random as rnd

#__________________________________________________

class DirectResampler(object):

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
        # array for the new indices
        i   = np.arange(Ns)

        for ns in np.arange(Ns):
            # search for particle to duplicate
            cursor = rnd.rand() * wc[-1]
            sample = 0
            while wc[sample] < cursor:
                sample += 1
            i[ns]   = sample
        return i

    #_________________________

    def resample(self, t_w, t_x):
        # resample particles t_x from the weights t_w
        indices = self.sampleIndices(t_w)
        Ns      = t_w.size
        return (-np.ones(Ns)*np.log(Ns), t_x[indices])

#__________________________________________________

