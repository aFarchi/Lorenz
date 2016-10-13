#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/random/
# independantgaussianrng.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/6
#__________________________________________________
#
# class to handle a gaussian random number generator
# when the covariance matrix is diagonal
#

import numpy        as np
import numpy.random as rnd

#__________________________________________________

class IndependantGaussianRNG(object):

    #_________________________

    def __init__(self, t_mean = np.zeros(0), t_sigma = np.zeros(0)):
        self.setIndependantGaussianRNGParameters(t_mean, t_sigma)

    #_________________________

    def setIndependantGaussianRNGParameters(self, t_mean = np.zeros(0), t_sigma = np.zeros(0)):
        # parameters for the gaussian distribution
        self.m_mean  = t_mean
        self.m_sigma = t_sigma

        # space dimension
        self.m_spaceDimension = self.m_mean.size

    #_________________________

    def addError(self, t_x, t_t):
        # add random error to x according to the gaussian distribution
        shape = t_x.shape
        if len(shape) == len(self.m_mean.shape):
            # for one vector
            error = self.drawSample(t_t)
        else:
            # for multiple vectors
            error = self.drawSamples(shape[0], t_t)
        return t_x + error

    #_________________________

    def drawSample(self, t_t):
        # draw one sample according to the gaussian distribution
        return self.m_sigma * rnd.standard_normal(self.m_mean.shape) + self.m_mean

    #_________________________

    def drawSamples(self, t_Ns, t_t):
        # draw t_Ns samples according to the gaussian distribution
        shape = list(self.m_mean.shape)
        shape.insert(0, t_Ns)
        return self.m_sigma * rnd.standard_normal(tuple(shape)) + self.m_mean

    #_________________________

    def square(self, t_x):
        # just a convenience function
        return t_x * t_x

    #_________________________

    def pdf(self, t_x, t_t, t_inflation = 1.0):
        # compute the pdf in log scale of the noise process at point x and at time t
        return - self.square( ( t_x - self.m_mean ) / ( t_inflation * self.m_sigma ) ) . sum(axis = -1) / 2.0

    #_________________________

    def covarianceMatrix(self, t_t):
        return np.diag(self.square(self.m_sigma))

    #_________________________

    def covarianceMatrix_diag(self, t_t):
        return self.square(self.m_sigma)

    #_________________________

    def stdDevMatrix_diag(self, t_t):
        return self.m_sigma

#__________________________________________________

