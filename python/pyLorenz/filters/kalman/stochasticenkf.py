#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# stochasticenkf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# class to handle a stochastic EnKF
#

import numpy as np

from ..abstractfilter                   import AbstractFilter
from ...utils.integration.rk4integrator import DeterministicRK4Integrator
from ...observations.iobservations      import StochasticIObservations

#__________________________________________________

class StochasticEnKF(AbstractFilter):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_Ns = 10):
        AbstractFilter.__init__(self, t_integrator, t_obsOp)
        self.setStochasticEnKFParameters(t_Ns)

    #_________________________

    def setStochasticEnKFParameters(self, t_Ns = 10):
        self.m_Ns             = t_Ns
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

    def analyse(self, t_nt, t_observation):
        # analyse observation at time nt

        # perturb observations
        oe = self.m_observationOperator.drawErrorSamples(self.m_Ns, t_nt)

        # Ensemble means
        xf_m  = self.m_x.mean(axis = 0)
        oe_m  = oe.mean(axis = 0)
        Hxf   = self.m_observationOperator.process(self.m_x, t_nt)
        Hxf_m = Hxf.mean(axis = 0)

        # Normalized anomalies
        Xf = ( self.m_x - xf_m ) / np.sqrt( self.m_Ns - 1.0 )
        Yf = ( Hxf - Hxf_m - oe + oe_m ) / np.sqrt( self.m_Ns - 1.0 )

        try:
            # Kalman gain
            K = np.dot ( np.transpose ( Xf ) , np.dot ( Yf , np.linalg.inv ( np.dot ( np.transpose ( Yf ) , Yf ) ) ) )
            # Update
            self.m_x = self.m_x + np.tensordot( t_observation + oe - Hxf , K , axes = (1, 1) )
        except:
            # no update if Kalman gain is singular
            pass

        self.m_estimate[t_nt, :] = self.estimate()

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

#__________________________________________________

