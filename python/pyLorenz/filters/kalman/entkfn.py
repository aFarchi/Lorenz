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

class EnTKF_N_dual(AbstractEnKF):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation,
            t_minimiser, t_epsilon, t_maxZeta, t_U = None):
        AbstractEnKF.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation)
        self.setEnTKF_N_dualParameters(t_minimiser, t_epsilon, t_maxZeta, t_U)

    #_________________________

    def setEnTKF_N_dualParameters(self, t_minimiser, t_epsilon, t_maxZeta, t_U):
        # minimiser
        self.m_minimiser = t_minimiser
        # epsilon
        self.m_epsilon   = t_epsilon
        # max value for zeta
        self.m_maxZeta   = t_maxZeta
        # U
        if t_U is None:
            self.m_U     = np.eye(self.m_Ns)
        else:
            self.m_U     = t_U

    #_________________________

    def analyse(self, t_t, t_observation):
        # analyse observation at time t

        # shortcut for forecast ensemble
        xf    = self.m_x[self.m_integrationIndex]
        # apply observation operator to forecast ensemble
        self.m_observationOperator.deterministicObserve(xf, t_t, self.m_Hxf)

        # Ensemble means
        xf_m  = xf.mean(axis = 0)
        Hxf_m = self.m_Hxf.mean(axis = 0)

        # Normalized anomalies
        Xf    = ( xf - xf_m ) / np.sqrt( self.m_Ns - 1.0 )
        Yf    = ( self.m_Hxf - Hxf_m ) / np.sqrt( self.m_Ns - 1.0 )

        # Analyse
        S       = self.m_observationOperator.applyRightErrorStdDevMatrix_inv(Yf)
        U, D, V = np.linalg.svd(S)
        delta   = np.dot ( V , self.m_observationOperator.applyLeftErrorStdDevMatrix_inv ( t_observation - Hxf_m ) )

        # dual cost to minimise
        diagonal          = np.zeros(delta.size)
        diagonal[:D.size] = D * D
        def dualCost(zeta):
            return ( ( delta * delta / ( 1.0 + diagonal * ( self.m_Ns - 1.0 ) / zeta ) ).sum() +
                    self.m_epsilon * zeta -
                    ( self.m_Ns + 1.0 ) * np.log( zeta ) )

        (zetaa, nit)       = self.m_minimiser.minimiseInterval(dualCost, 0.0, self.m_maxZeta, 0.5*self.m_maxZeta)

        # analyse weights
        wa                 = np.zeros(self.m_Ns)
        wa[:D.size]        = D * delta[:D.size]
        diagonal           = np.zeros(self.m_Ns)
        diagonal[:D.size]  = D * D
        diagonal          += zetaa / ( self.m_Ns - 1.0 )
        wa                /= diagonal
        wa                 = np.dot( U , wa )

        # new ensemble
        Xa                 = np.dot( self.m_U , np.dot( U / np.sqrt( diagonal ) , np.transpose(U) ) )

        # update
        self.m_x[self.m_integrationIndex] = xf_m + np.dot( wa + np.sqrt( self.m_Ns - 1.0 ) * Xa , Xf )

#__________________________________________________

