#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/random/
# abstractresampler.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# base class for a resampler
#

import numpy as np

#__________________________________________________

class AbstractResampler(object):

    #_________________________

    def __init__(self, t_rng):
        self.setAbstractResamplerParameters(t_rng)

    #_________________________

    def setAbstractResamplerParameters(self, t_rng):
        # random number generator
        self.m_rng = t_rng

    #_________________________

    def sample(self, t_Ns, t_w, t_x):
        # sample t_Ns particles from initial particles t_x with weights t_w
        indices = self.sampleIndices(t_Ns, t_w)
        return (-np.ones(t_Ns)*np.log(t_Ns), t_x[indices])

#__________________________________________________

