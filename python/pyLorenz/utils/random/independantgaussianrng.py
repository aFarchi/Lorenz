#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/random/
# independantgaussianrng.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# class to handle a gaussian random number generator
# when the covariance matrix is diagonal
#

import numpy as np
import numpy.random as rnd

#__________________________________________________

class IndependantGaussianRNG(object):

    #_________________________

    def __init__(self, t_mean = np.zeros(0), t_sigma = np.zeros(0), t_relSigmaMin = 1.0e-5):
        self.setIndependantGaussianRNGParameters(t_mean, t_sigma, t_relSigmaMin)

    #_________________________

    def setIndependantGaussianRNGParameters(self, t_mean = np.zeros(0), t_sigma = np.zeros(0), t_relSigmaMin = 1.0e-5):
        # set parameter for the gaussian distribution
        self.m_relSigmaMin = t_relSigmaMin
        self.m_mean        = t_mean
        self.m_sigma       = np.maximum(t_sigma, self.m_relSigmaMin)
        self.m_exact       = self.m_sigma < self.m_relSigmaMin

    #_________________________

    def addError(self, t_x):
        # add random error to x according to the gaussian distribution
        shape = t_x.shape
        if len(shape) == 1:
            # for one vector
            error = self.drawSample()
        else:
            # for multiple vectors
            error = self.drawSamples(shape[0])
        return t_x + error

    #_________________________

    def drawSample(self):
        # draw one sample according to the gaussian distribution
        return rnd.normal(self.m_mean, self.m_sigma) * ( 1.0 - self.m_exact ) + self.m_mean * self.m_exact

    #_________________________

    def drawSamples(self, t_Ns):
        # draw t_Ns samples according to the gaussian distribution
        return ( rnd.normal( np.tile(self.m_mean, (t_Ns, 1)) , np.tile(self.m_sigma, (t_Ns, 1)) ) * ( 1.0 - np.tile(self.m_exact, (t_Ns, 1)) ) +
                np.tile(self.m_mean*self.m_exact, (t_Ns, 1)) )

    #_________________________

    def pdf(self, t_x, t_inflation = 1.0):
        # compute the pdf of the noise process at point t_x
        shape = t_x.shape
        if len(shape) == 1:
            return np.exp ( - np.power ( ( t_x - self.m_mean ) / ( t_inflation * self.m_sigma ) , 2 ) / 2.0 ) . prod()
        else:
            return np.exp ( - np.power ( ( t_x - np.tile( self.m_mean , (shape[0], 1) ) ) / np.tile ( t_inflation * self.m_sigma , (shape[0], 1) ) , 2 ) / 2.0 ) . prod(axis = 1)

#__________________________________________________

