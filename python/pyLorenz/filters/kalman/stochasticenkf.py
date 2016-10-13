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

from abstractenkf                       import AbstractEnKF
from ...utils.integration.rk4integrator import DeterministicRK4Integrator
from ...observations.iobservations      import StochasticIObservations

#__________________________________________________

class StochasticEnKF(AbstractEnKF):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_Ns = 10, t_covInflation = 1.0):
        AbstractEnKF.__init__(self, t_integrator, t_obsOp, t_Ns, t_covInflation)

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

        # Estimation and inflation
        self.m_estimate[t_nt, :] = self.estimateAndInflate()

#__________________________________________________

