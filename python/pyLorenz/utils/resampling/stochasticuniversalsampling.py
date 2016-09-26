#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/resampling/
# stochasticuniversalsampling.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
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

    def resample(self, t_w, t_x):
        # resample from the weights t_w
        # number of particles
        Ns = t_w.size
        # cumulative weights
        wc = np.exp(t_w).cumsum()
        # make sure weights are normalized
        wc /= wc[-1]
        # array for the new particles
        x = np.zeros(t_x.shape)
        # draw random number in [0,1/Ns]
        cursor = rnd.rand() / Ns
        sample = 0
        for ns in np.arange(Ns):
            # search for particle to duplicate
            while wc[sample] < cursor:
                sample += 1
            x[ns]   = t_x[ns]
            # forward cursor
            cursor += 1.0 / Ns
        return (-np.ones(Ns)*np.log(Ns), x)

#__________________________________________________

