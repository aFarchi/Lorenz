#! /usr/bin/env python

#__________________________________________________
# pyLorenz/observations/
# observealloperator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# class to handle an observation operator that is the identity at all time steps
#

import numpy as np

from observations.abstractobservationoperator import AbstractObservationOperator

#__________________________________________________

class ObserveAllOperator(AbstractObservationOperator):

    #_________________________

    def __init__(self, t_errorGenerator):
        AbstractObservationOperator.__init__(self, t_errorGenerator)

    #_________________________

    def deterministicObserve(self, t_x, t_t, t_y):
        # observe everything with the identity operator
        t_y[:] = t_x[:]

    #_________________________

    def isLinear(self):
        # true
        return True

    #_________________________

    def differential(self, t_x, t_t):
        # differential is the identity matrix
        return np.eye(self.m_spaceDimension)

    #_________________________

    def differential_diag(self, t_x, t_t):
        # diagonal of the identity matrix
        return np.ones(self.m_spaceDimension)

    #_________________________

    def errorCovarianceMatrix_diag(self, t_t, t_spaceDimension):
        # since observation operator is the identity, the "cast" operator is also the identity
        return self.m_errorGenerator.covarianceMatrix_diag(t_t)

    #_________________________

    def errorStdDevMatrix_diag(self, t_t, t_spaceDimension):
        # since observation operator is the identity, the "cast" operator is also the identity
        return self.m_errorGenerator.stdDevMatrix_diag(t_t)

    #_________________________

    def castObservationToStateSpace(self, t_observation, t_t, t_spaceDimension):
        # since observation operator is the identity, the "cast" operator is also the identity
        return t_observation

    #_________________________

    def nearest_y_dimensions(self, t_nearest_x_dimensions):
        # nearest dimensions in observation space
        return t_nearest_x_dimensions

#__________________________________________________

