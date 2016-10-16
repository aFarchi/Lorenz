#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/random/
# independantgaussianerrorgenerator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/13
#__________________________________________________
#
# class to handle a time independant gaussian random error generator
# when error covariance matrix is diagonal
#

import numpy        as np
import numpy.random as rnd

#__________________________________________________

class IndependantGaussianErrorGenerator(object):

    #_________________________

    def __init__(self, t_stdDev):
        self.setIndependantGaussianRNGParameters(t_stdDev)

    #_________________________

    def setIndependantGaussianRNGParameters(self, t_stdDev):
        # space dimension
        self.m_spaceDimension = t_stdDev.size
        # standard deviation (expected as a one dimensional array)
        self.m_stdDev         = t_stdDev
        # performance tweak: compute variance once and for all
        self.m_sigma          = t_stdDev * t_stdDev

    #_________________________

    def drawSamples(self, t_t, t_shape, t_inflation = 1.0):
        # draw error samples at time t
        # result is expected to have the specified shape 
        # (which must be compatible with self.m_stdDev.size, i.e shape[-1] == self.m_stdDev.size)
        # note: the standard deviation of the error is inflated by a factor inflation
        return ( t_inflation * self.m_stdDev ) * rnd.standard_normal(t_shape)

    #_________________________

    def drawSyntheticSamples(self, t_t, t_tW, t_shape, t_inflation = 1.0):
        # draw synthetic error samples
        # i.e. with stdDev = average( stdDev[time = t] , weight = tW )
        #
        # here since stdDev does not depend on time, it does not change anything compared to the drawSamples() method
        return self.drawSamples(t_nt[0], t_shape, t_inflation)

    #_________________________

    def square(self, t_x):
        # just a convenience function
        return t_x * t_x

    #_________________________

    def pdf(self, t_x, t_t, t_inflation = 1.0):
        # return the pdf in log scale at time t taken at point x
        # note: the standard deviation of the error is inflated by a factor inflation
        return - self.square( t_x / ( t_inflation * self.m_stdDev ) ) . sum(axis = -1) / 2.0

    #_________________________

    def covarianceMatrix(self, t_t):
        # return the covariance matrix of the error at time t
        return np.diag(self.m_sigma)

    #_________________________

    def covarianceMatrix_diag(self, t_t):
        # return the diagonal of the covariance matrix at time t
        return self.m_sigma

    #_________________________

    def stdDevMatrix_diag(self, t_t):
        # return the diagonal of the standard deviation matrix at time t
        return self.m_stdDev

#__________________________________________________

