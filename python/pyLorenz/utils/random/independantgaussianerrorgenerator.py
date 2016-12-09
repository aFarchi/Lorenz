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

import numpy as np

#__________________________________________________

class IndependantGaussianErrorGenerator(object):

    #_________________________

    def __init__(self, t_stdDev, t_rng):
        self.setIndependantGaussianRNGParameters(t_stdDev, t_rng)

    #_________________________

    def setIndependantGaussianRNGParameters(self, t_stdDev, t_rng):
        # space dimension
        self.m_spaceDimension = t_stdDev.size
        # standard deviation (expected as a one dimensional array)
        self.m_stdDev         = t_stdDev
        # performance tweak: compute variance once and for all
        self.m_sigma          = t_stdDev * t_stdDev
        # random number generator
        self.m_rng            = t_rng

    #_________________________

    def drawSamples(self, t_t, t_shape, t_inflation = 1.0):
        # draw error samples at time t
        # result is expected to have the specified shape 
        # (which must be compatible with self.m_stdDev.size, i.e shape[-1] == self.m_stdDev.size)
        # note: the standard deviation of the error is inflated by a factor inflation
        return ( t_inflation * self.m_stdDev ) * self.m_rng.standard_normal(t_shape)

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
        return - 0.5 * self.square( t_x / ( t_inflation * self.m_stdDev ) ) . sum(axis = -1)

    #_________________________

    def local_pdf(self, t_x, t_t, t_dimension, t_inflation = 1.0):
        # return the pdf at time t taken at point x
        # note: the standard deviation of the error is inflated by a factor inflation
        return np.exp ( - 0.5 * self.square( t_x / ( t_inflation * self.m_stdDev[t_dimension] ) ) )

    #_________________________

    def applyLeftCovMatrix_inv(self, t_x):
        # return sigma^(-1) . x
        return np.transpose( np.transpose(t_x) / self.m_sigma )

    #_________________________

    def applyLeftStdDevMatrix_inv(self, t_x):
        # return stdDev^(-1) . x
        return np.transpose( np.transpose(t_x) / self.m_stdDev )

    #_________________________

    def applyLeftStdDevMatrix_inv_local(self, t_x, t_dimensions):
        # return stdDev^(-1) [dimensions, dimensions] . x[dimensions, ...]
        return np.transpose( np.transpose(t_x)[..., t_dimensions] / self.m_stdDev[t_dimensions] )

    #_________________________
        
    def applyRightCovMatrix_inv(self, t_x):
        # return x . sigma^(-1)
        return t_x / self.m_sigma

    #_________________________

    def applyRightStdDevMatrix_inv(self, t_x):
        # return x . stdDev^(-1)
        return t_x / self.m_stdDev

    #_________________________

    def applyRightStdDevMatrix_inv_local(self, t_x, t_dimensions):
        # return x[..., dimensions] .  stdDev^(-1)[dimensions, dimensions]
        return t_x[..., t_dimensions] / self.m_stdDev[t_dimensions]

    #_________________________

    def covarianceMatrix(self, t_t):
        # return the covariance matrix of the error at time t
        return np.diag(self.m_sigma)

    #_________________________

    def stdDevMatrix_inv(self, t_t):
        # return the inverse of the standard deviation matrix at time t
        return np.diag( 1.0 / self.m_stdDev )

    #_________________________

    def covarianceMatrix_diag(self, t_t):
        # return the diagonal of the covariance matrix at time t
        return self.m_sigma

    #_________________________

    def stdDevMatrix_diag(self, t_t):
        # return the diagonal of the standard deviation matrix at time t
        return self.m_stdDev

#__________________________________________________

