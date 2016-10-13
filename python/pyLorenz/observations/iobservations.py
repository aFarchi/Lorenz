#! /usr/bin/env python

#__________________________________________________
# pyLorenz/observations/
# iobservations.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# class to handle an observation operator that is the identity
#

import numpy as np

from ..utils.process.abstractprocess       import AbstractStochasticProcess
from ..utils.random.independantgaussianrng import IndependantGaussianRNG

#__________________________________________________

class StochasticIObservations(AbstractStochasticProcess):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG()):
        AbstractStochasticProcess.__init__(self, t_eg)
        self.m_spaceDimension = t_eg.m_spaceDimension

    #_________________________

    def deterministicProcess(self, t_x, t_t):
        # just observe everything with the identity operator
        # time is ignored
        return t_x

    #_________________________

    def isLinear(self):
        # return true if and only if deterministicProcess is a linear operator
        return True

    #_________________________

    def differential(self, t_x, t_t):
        # linearisation of deterministicProcess about t_x and t_t
        return np.eye(self.m_spaceDimension)

    #_________________________

    def differential_diag(self, t_x, t_t):
        # linearisation of deterministicProcess about t_x and t_t
        # return only the diagonal of the differential
        return np.ones(self.m_spaceDimension)

    #_________________________

    def drawErrorSamples(self, t_Ns, t_t):
        return self.m_errorGenerator.drawSamples(t_Ns, t_t)

    #_________________________

    def pdf(self, t_observation, t_x, t_t, t_inflation = 1.0):
        # observation pdf in log scale at obs - H(x)
        return self.m_errorGenerator.pdf(t_observation-self.deterministicProcess(t_x, t_t), t_inflation)

    #_________________________

    def errorCovarianceMatrix_diag(self, t_t, t_spaceDimension):
        return self.m_errorGenerator.covarianceMatrix_diag(t_t)

    #_________________________

    def errorStdDevMatrix_diag(self, t_t, t_spaceDimension):
        return self.m_errorGenerator.stdDevMatrix_diag(t_t)

    #_________________________

    def castObservationToStateSpace(self, t_observation, t_t, t_spaceDimension):
        return t_observation

#__________________________________________________

