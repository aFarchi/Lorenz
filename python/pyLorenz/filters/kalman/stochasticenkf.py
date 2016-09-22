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

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations()):
        AbstractFilter.__init__(self, t_integrator, t_obsOp)

    #_________________________

    def initialise(self, t_x):
        # particles / samples
        self.m_x = t_x
        # number of particles / samples
        self.m_Ns = t_x.shape[0]

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

        return self.estimate()

    #_________________________

    def forecast(self, t_ntStart, t_ntEnd, t_observation):
        # integrate particles from ntStart to ntEnd, given the observation at ntEnd
        # here, EnKF ignores the 'future' observation
        estimation = np.zeros((t_ntEnd-t_ntStart, self.m_integrator.m_model.m_stateDimension))
        for nt in np.arange(t_ntEnd-t_ntStart):
            self.m_x       = self.m_integrator.process(self.m_x, t_ntStart+nt)
            estimation[nt] = self.estimate()
        return estimation

    #_________________________

    def estimate(self):
        # mean of x
        return self.m_x.mean(axis = 0)

#__________________________________________________

