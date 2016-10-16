#! /usr/bin/env python

#__________________________________________________
# pyLorenz/observations/
# observenfirstoperator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/13
#__________________________________________________
#
# class to handle an observation operator that is the identity
# on the N first dimensions
#
# note: N is fixed by the dimension of the error generator
# it will be self.m_spaceDimension
#

import numpy as np

from abstractobservationoperator import AbstractObservationOperator

#__________________________________________________

class ObserveNFirstOperator(AbstractObservationOperator):

    #_________________________

    def __init__(self, t_errorGenerator):
        AbstractObservationOperator.__init__(self, t_errorGenerator)

    #_________________________

    def deterministicObserve(self, t_x, t_t):
        # observe the self.m_spaceDimension first dimensions with the identity operator
        return t_x[..., :self.m_spaceDimension]

    #_________________________

    def isLinear(self):
        # true
        return True

    #_________________________

    def differential(self, t_x, t_t):
        # differential is the identity matrix in the observed subspace
        xDimension = t_x.shape[-1]
        H          = np.eye(xDimension)
        return H[:self.m_spaceDimension]

    #_________________________

    def differential_diag(self, t_x, t_t):
        # diagonal of the identity matrix in the observed subspace
        # and zero in the unobserved subspace
        xDimension                = t_x.shape[-1]
        H                         = np.zeros(xDimension)
        H[:self.m_spaceDimension] = 1.0
        return H
            
    #_________________________

    def errorCovarianceMatrix_diag(self, t_t, t_spaceDimension):
        # cast operator
        # note: return 1 instead of 0 for unobserved dimensions to avoid division by zero
        sigma                         = np.ones(t_spaceDimension)
        sigma[:self.m_spaceDimension] = self.m_errorGenerator.covarianceMatrix_diag(t_t)
        return sigma

    #_________________________

    def errorStdDevMatrix_diag(self, t_t, t_spaceDimension):
        # cast operator
        # note: return 1 instead of 0 for unobserved dimensions to avoid division by zero
        stdDev                         = np.ones(t_spaceDimension)
        stdDev[:self.m_spaceDimension] = self.m_errorGenerator.stdDevMatrix_diag(t_t)
        return stdDev

    #_________________________

    def castObservationToStateSpace(self, t_observation, t_t, t_spaceDimension):
        # cast operator
        y                         = np.zeros(t_spaceDimension)
        y[:self.m_spaceDimension] = t_observation
        return y

#__________________________________________________

