#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# abstractenkf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/12
#__________________________________________________
#
# abstract class to handle an EnKF
#

import numpy as np

from ..abstractfilter                   import AbstractFilter
from ...utils.integration.rk4integrator import DeterministicRK4Integrator
from ...observations.iobservations      import StochasticIObservations

#__________________________________________________

class AbstractEnKF(AbstractFilter):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_Ns = 10, t_covInflation = 1.0):
        AbstractFilter.__init__(self, t_integrator, t_obsOp)
        self.setAbstractEnKFParameters(t_Ns, t_covInflation)

    #_________________________

    def setAbstractEnKFParameters(self, t_Ns = 10, t_covInflation = 1.0):
        self.m_Ns             = t_Ns
        self.m_covInflation   = t_covInflation
        self.m_spaceDimension = self.m_integrator.m_spaceDimension

    #_________________________

    def initialise(self, t_initialiser, t_Nt):
        # particles / samples
        self.m_x              = t_initialiser.initialiseSamples(self.m_Ns)
        # estimations
        self.m_estimate       = np.zeros((t_Nt, self.m_spaceDimension))
        # fill first guess
        self.m_estimate[0, :] = self.estimate()

    #_________________________

    def forecast(self, t_ntStart, t_ntEnd, t_observation):
        # integrate particles from ntStart to ntEnd, given the observation at ntEnd
        # here, EnKF ignores the 'future' observation
        for nt in t_ntStart + np.arange(t_ntEnd-t_ntStart):
            self.m_x                 = self.m_integrator.process(self.m_x, nt)
            self.m_estimate[nt+1, :] = self.estimate()

    #_________________________

    def estimate(self):
        # mean of x
        return self.m_x.mean(axis = 0)

    #_________________________

    def recordToFile(self, t_outputDir = './', t_filterPrefix = 'kalman'):
        self.m_estimate.tofile(t_outputDir+t_filterPrefix+'_estimation.bin')

    #_________________________

    def estimateAndInflate(self):
        # computes estimation and inflate ensemble around this estimation
        x_m      = self.estimate()
        self.m_x = x_m + self.m_covInflation * ( self.m_x - x_m )
        return x_m

#__________________________________________________

