#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# entkfn.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/15
#__________________________________________________
#
# class to handle an ensemble transform kalman filter of finite size
# i.e. the hierarchical counterpart of EnTKF
#

import numpy as np

from abstractenkf import AbstractEnKF

#__________________________________________________

class EnTKF_N(AbstractEnKF):

    #_________________________

    def __init__(self, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation, t_minimiser, t_U = None):
        AbstractEnKF.__init__(self, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation)
        self.setEnTKF_NParameters(t_minimiser, t_U)

    #_________________________

    def setEnTKF_NParameters(self, t_minimiser, t_U):
        # minimiser
        self.m_minimiser = t_minimiser
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
        rsom1 = self.m_observationOperator.errorStdDevMatrix_inv(t_t)

        S                      = np.dot ( Yf , rsom1 )
        U, D, V                = np.linalg.svd(S)
        diag                   = np.zeros((self.m_Ns, t_observation.size))
        diag[:D.size, :D.size] = np.diag(D)

        delta = np.dot ( V , np.dot ( rsom1 , t_observation - Hxf_m ) )

        # dual cost to minimise
        diagonal          = np.zeros(delta.size)
        diagonal[:D.size] = D * D
        def dualCost(zeta):
            return ( 0.5 * ( delta * delta / ( 1.0 + diagonal * ( self.m_Ns - 1.0 ) / zeta ) ).sum() +
                    ( self.m_Ns + 1.0 ) * zeta / ( 2.0 * self.m_Ns ) +
                    ( self.m_Ns + 1.0 ) * 0.5 * np.log( ( self.m_Ns + 1.0 ) / zeta ) )

        zeta0           = np.array([self.m_Ns])
        (zetaa, h, nit) = self.m_minimiser.minimise(dualCost, zeta0)
        zetaa           = zetaa[0]

        # analyse weights
        delta              = np.dot(diag, delta)
        diagonal           = np.zeros(self.m_Ns)
        diagonal[:D.size]  = D * D
        diagonal          += zetaa / ( self.m_Ns - 1.0 )
        delta             /= diagonal
        wa                 = np.dot( U , delta )

        # new ensemble
        Xa = np.dot( self.m_U , np.dot( U / np.sqrt( diagonal ) , np.transpose( U ) ) )
        self.m_x[t_index] = xf_m + np.dot( wa + np.sqrt( self.m_Ns - 1.0 ) * Xa , Xf )

#__________________________________________________

