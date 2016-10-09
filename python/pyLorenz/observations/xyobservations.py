#! /usr/bin/env python

#__________________________________________________
# pyLorenz/observations/
# xyobservations.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# class to handle an observation operator that is the identity
#

import numpy as np

from iobservations                         import StochasticIObservations
from ..utils.process.abstractprocess       import AbstractStochasticProcess
from ..utils.random.independantgaussianrng import IndependantGaussianRNG

#__________________________________________________

class StochasticXYObservations(StochasticIObservations):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG()):
        StochasticIObservations.__init__(self, t_eg)

    #_________________________

    def deterministicProcess(self, t_x, t_t):
        # just observe everything with the identity operator
        # time is ignored
        shape = t_x.shape
        if len(shape) == 1:
            return t_x[:self.m_spaceDimension]
        else:
            return t_x[:, :self.m_spaceDimension]

    #_________________________

    def isLinear(self):
        return True

    #_________________________

    def differential(self, t_x, t_t):
        # linearisation of deterministicProcess about t_x and t_t
        xDimension = t_x.shape[-1]
        H          = np.eye(xDimension)
        return H[:self.m_spaceDimension, :]

    #_________________________

    def differential_diag(self, t_x, t_t):
        xDimension                = t_x.shape[-1]
        H                         = np.zeros(xDimension)
        H[:self.m_spaceDimension] = 1.0
        return H
            
    #_________________________

    def errorCovarianceMatrix_diag(self, t_t, t_spaceDimension):
        sigma                         = np.ones(t_spaceDimension) # return 1 instead of 0 for unobserved dimensions to avoid division by zero
        sigma[:self.m_spaceDimension] = self.m_errorGenerator.covarianceMatrix_diag(t_t)
        return sigma

    #_________________________

    def castObservationToStateSpace(self, t_observation, t_t, t_spaceDimension):
        y                         = np.zeros(t_spaceDimension)
        y[:self.m_spaceDimension] = t_observation
        return y

#__________________________________________________

