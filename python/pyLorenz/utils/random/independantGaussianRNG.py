#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/random/
# independantGaussianRNG.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# class to handle a gaussian random number generator
# when the covariance matrix is diagonal
#

import numpy as np
import numpy.random as rnd

#__________________________________________________

class IndependantGaussianRNG:

    #_________________________

    def __init__(self):
        self.m_relSigmaMin = 1.0e-5
        self.setParameters()

    #_________________________

    def setParameters(self, t_mean = np.zeros(0), t_sigma = np.zeros(0)):
        # set parameter for the gaussian distribution
        self.m_mean  = t_mean
        self.m_sigma = np.maximum(t_sigma, self.m_relSigmaMin)
        self.m_exact = self.m_sigma < self.m_relSigmaMin

    #_________________________

    def addError(self, t_x):
        # add random error to x according to the gaussian distribution
        shape = t_x.shape
        if len(shape) == 1:
            # for one vector
            t_x += self.drawSample()
        else:
            # for multiple vectors
            t_x += self.drawSamples(shape[0])

    #_________________________

    def drawSample(self):
        # draw one sample according to the gaussian distribution
        return rnd.normal(self.m_mean, self.m_sigma) * ( 1.0 - self.m_exact ) + self.m_mean * self.m_exact

    #_________________________

    def drawSamples(self, t_Ns):
        # draw t_Ns samples according to the gaussian distribution
        return ( rnd.normal( np.tile(self.m_mean, (t_Ns, 1)) , np.tile(self.m_sigma, (t_Ns, 1)) ) * ( 1.0 - np.tile(self.m_exact, (t_Ns, 1)) ) +
                np.tile(self.m_mean*self.m_exact, (t_Ns, 1)) )

#__________________________________________________

