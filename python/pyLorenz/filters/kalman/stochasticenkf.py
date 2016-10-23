#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# stochasticenkf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/20
#__________________________________________________
#
# class to handle a stochastic EnKF
#

import numpy as np

from abstractenkf import AbstractEnKF

#__________________________________________________

class StochasticEnKF(AbstractEnKF):

    #_________________________

    def __init__(self, t_label, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation):
        AbstractEnKF.__init__(self, t_label, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation)

    #_________________________

    def analyse(self, t_index, t_t, t_observation):
        # analyse observation at time t

        # perturb observations
        shape = (self.m_Ns, t_observation.size)
        oe    = self.m_observationOperator.drawErrorSamples(t_t, shape)

        # forecast
        xf    = self.m_x[t_index]
        self.m_observationOperator.deterministicObserve(xf, t_t, self.m_Hxf)

        # Ensemble means
        xf_m  = xf.mean(axis = 0)
        oe_m  = oe.mean(axis = 0)
        Hxf_m = self.m_Hxf.mean(axis = 0)

        # Normalized anomalies
        Xf    = ( xf - xf_m ) / np.sqrt( self.m_Ns - 1.0 )
        Yf    = ( self.m_Hxf - Hxf_m - oe + oe_m ) / np.sqrt( self.m_Ns - 1.0 )

        # Kalman gain
        K     = np.dot ( np.linalg.pinv ( np.dot ( np.transpose ( Yf ) , Yf ) ) , np.dot ( np.transpose ( Yf ) , Xf ) )

        # Update
        xf   += np.dot ( t_observation + oe - self.m_Hxf , K )

#__________________________________________________

