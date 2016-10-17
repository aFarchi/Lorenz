#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# entkf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/15
#__________________________________________________
#
# class to handle an ensemble transform kalman filter
#

import numpy as np

from abstractenkf import AbstractEnKF

#__________________________________________________

class EnTKF(AbstractEnKF):

    #_________________________

    def __init__(self, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation, t_U = None):
        AbstractEnKF.__init__(self, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation)
        self.setEnTKFParameters(t_U)

    #_________________________

    def setEnTKFParameters(self, t_U):
        # U
        if t_U is None:
            self.m_U = np.eye(self.m_Ns)
        else:
            self.m_U = t_U

    #_________________________

    def analyse(self, t_index, t_t, t_observation):
        # analyse observation at time t

        # shortcut
        xf    = self.m_x[t_index]

        # Ensemble means
        xf_m  = xf.mean(axis = 0)
        Hxf   = self.m_observationOperator.observe(xf, t_t)
        Hxf_m = Hxf.mean(axis = 0)

        # Normalized anomalies
        Xf    = ( xf - xf_m ) / np.sqrt( self.m_Ns - 1.0 )
        Yf    = ( Hxf - Hxf_m ) / np.sqrt( self.m_Ns - 1.0 )

        # Analyse
        rsom1    = self.m_observationOperator.errorStdDevMatrix_inv(t_t)

        S        = np.dot ( Yf , rsom1 )
        delta    = np.dot ( rsom1 , t_observation - Hxf_m )

        Tm1      = np.eye(self.m_Ns) + np.dot( S , np.transpose(S) )
        U, s, V  = np.linalg.svd(Tm1) # T^-1 = U * s * V, T = tV / s * tU

        # w = T * S * delta 
        w = np.dot ( np.dot ( np.transpose(V) / s , np.transpose(U) ) , np.dot ( S , delta ) )

        # update
        # x <- xa + self.m_U * T ^ (1/2) * Xf = xf_m + w * Xf + self.m_U * T ^ (1/2) * Xf
        self.m_x[t_index] = xf_m + np.dot( w + np.sqrt( self.m_Ns - 1.0 ) * np.dot( self.m_U , np.dot( np.transpose(V) / np.sqrt(s) , np.transpose(U) ) ) , Xf )

#__________________________________________________

