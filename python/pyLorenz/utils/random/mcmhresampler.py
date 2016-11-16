#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/random/
# mcmhresampler.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/16
#__________________________________________________
#
# class to handle a Monte Carlo Metropolis-Hastings Resampler
#

import numpy as np

from utils.random.abstractresampler import AbstractResampler

#__________________________________________________

class MCMHResampler(AbstractResampler):

    #_________________________

    def __init__(self, t_rng):
        AbstractResampler.__init__(self, t_rng)

    #_________________________

    def sampleIndices(self, t_Ns, t_w):
        # sample indices
        i   = np.arange(t_Ns)
        # normal weights
        w   = np.exp(t_w)

        if w.sum() == 0.0: 
            return i

        # choose first sample
        sample = 0
        for ns in range(t_Ns):
            cursor = w[sample] * self.m_rng.rand()
            # accept next sample with proba w[next]/w[previous]
            if cursor < w[ns]:
                sample = ns
            # else duplicate previous sample
            else:
                i[ns]  = sample

        return i

#__________________________________________________

