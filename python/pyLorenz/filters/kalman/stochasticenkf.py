#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# stochasticenkf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
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
        self.m_Ns = t_Ns

    #_________________________

    def initialise(self, t_initialiser, t_Nt):
        # particles / samples
        self.m_x           = t_initialiser.drawSamples(self.m_Ns)

        # estimations
        self.m_estimate    = np.zeros((t_Nt, self.m_integrator.m_model.m_stateDimension))

        # fill first guess
        self.m_estimate[0] = self.estimate()

    #_________________________

    def Neff(self):
        return 1.0

    #_________________________

    def analyse(self, t_nt, t_obs):
        # analyse observation at time nt

        # perturb observations
        oe = self.m_observationOperator.m_errorGenerator.drawSamples(self.m_Ns)

        # Ensemble means
        xf_m  = self.m_x.mean(axis = 0)
        oe_m  = oe.mean(axis = 0)
        Hxf   = self.m_observationOperator.process(self.m_x, t_nt*self.m_integrator.m_dt)
        Hxf_m = Hxf.mean(axis = 0)

        # Normalized anomalies
        Xf = ( self.m_x - np.tile(xf_m, (self.m_Ns, 1)) ) / np.sqrt( self.m_Ns - 1.0 )
        Yf = ( Hxf - np.tile(Hxf_m, (self.m_Ns, 1)) - oe + np.tile(oe_m, (self.m_Ns, 1)) ) / np.sqrt( self.m_Ns - 1.0 )

        try:
            # Kalman gain
            K = np.dot ( np.transpose ( Xf ) , np.dot ( Yf , np.linalg.inv ( np.dot ( np.transpose ( Yf ) , Yf ) ) ) )
            # Update
            self.m_x = self.m_x + np.tensordot( np.tile ( t_obs , (self.m_Ns, 1) ) + oe - Hxf , K , axes = (1, 1) )
        except:
            # no update
            pass

        self.m_estimate[t_nt] = self.estimate()

    #_________________________

    def forecast(self, t_ntStart, t_ntEnd, t_observation):
        # integrate particles from ntStart to ntEnd, given the observation at ntEnd
        # here, EnKF ignores the 'future' observation
        for nt in t_ntStart + np.arange(t_ntEnd-t_ntStart):
            self.m_x              = self.m_integrator.process(self.m_x, nt)
            self.m_estimate[nt+1] = self.estimate()

    #_________________________

    def estimate(self):
        # mean of x
        return self.m_x.mean(axis = 0)

    #_________________________

    def recordToFile(self, t_outputDir = './', t_filterPrefix = 'kalman'):
        self.m_estimate.tofile(t_outputDir+t_filterPrefix+'_estimation.bin')

#__________________________________________________

