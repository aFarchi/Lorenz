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

from abstractenkf import AbstractEnKF

#__________________________________________________

class StochasticEnKF(AbstractEnKF):

    #_________________________

    def __init__(self, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation):
        AbstractEnKF.__init__(self, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation)

    #_________________________

    def analyse(self, t_index, t_t, t_observation):
        # analyse observation at time t

        # perturb observations
        shape = (self.m_Ns, self.m_spaceDimension)
        oe    = self.m_observationOperator.drawErrorSamples(t_t, shape)

        # shortcut
        xf    = self.m_x[t_index]

        # Ensemble means
        xf_m  = xf.mean(axis = 0)
        oe_m  = oe.mean(axis = 0)
        Hxf   = self.m_observationOperator.observe(xf, t_t)
        Hxf_m = Hxf.mean(axis = 0)

        # Normalized anomalies
        Xf    = ( xf - xf_m ) / np.sqrt( self.m_Ns - 1.0 )
        Yf    = ( Hxf - Hxf_m - oe + oe_m ) / np.sqrt( self.m_Ns - 1.0 )

        # Kalman gain
        K   = np.dot ( np.linalg.inv ( np.dot ( np.transpose ( Yf ) , Yf ) ) , np.dot ( np.transpose ( Yf ) , Xf ) )
        # Update
        xf += np.dot ( t_observation + oe - Hxf , K )

#__________________________________________________

