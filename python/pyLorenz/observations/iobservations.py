#! /usr/bin/env python

#__________________________________________________
# pyLorenz/observations/
# iobservations.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/27
#__________________________________________________
#
# class to handle an observation operator that is the identity
#

import numpy as np

from ..utils.process.abstractprocess       import AbstractStochasticProcess
from ..utils.process.abstractprocess       import AbstractDeterministicProcess
from ..utils.random.independantgaussianrng import IndependantGaussianRNG

#__________________________________________________

class AbstractIObservations(object):

    #_________________________

    def __init__(self):
        pass

    #_________________________

    def deterministicProcess(self, t_x, t_t):
        # just observe everything with the identity operator
        # time is ignored
        return t_x

    #_________________________

    def observationPDF(self, t_obs, t_x, t_t, t_inflation = 1.0):
        # observation pdf in log scale at obs - H(x)
        # error variance is inflated by factor t_inflation
        # time is ignored
        shape = t_x.shape
        if len(shape) == 1:
            return self.m_errorGenerator.pdf(t_obs-t_x, t_inflation)
        else:
            return self.m_errorGenerator.pdf(np.tile(t_obs, (shape[0], 1))-t_x, t_inflation)

    #_________________________

    def isLinear(self):
        # return true if and only if deterministicProcess is a linear operator
        return True

    #_________________________

    def diagonalDifferential(self, t_x, t_t):
        # linearisation of deterministicProcess about t_x and t_t
        # here, since deterministicProcess is linear t_x and t_t do not interfere
        return np.ones(t_x.shape)

#__________________________________________________

class StochasticIObservations(AbstractIObservations, AbstractStochasticProcess):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG()):
        AbstractIObservations.__init__(self)
        AbstractStochasticProcess.__init__(self, t_eg)

    #_________________________

    def deterministicObservationOperator(self):
        return DeterministicIObservations()

#__________________________________________________

class DeterministicIObservations(AbstractIObservations, AbstractDeterministicProcess):

    #_________________________

    def __init__(self):
        AbstractIObservations.__init__(self)
        AbstractDeterministicProcess.__init__(self)


#__________________________________________________

