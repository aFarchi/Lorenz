#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# entkf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# class to handle an ensemble transform kalman filter
#

import numpy as np

from abstractenkf                       import AbstractEnKF
from ...utils.integration.rk4integrator import DeterministicRK4Integrator
from ...observations.iobservations      import StochasticIObservations

#__________________________________________________

class EnTKF_diag(AbstractEnKF):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_Ns = 10, t_covInflation = 1.0, t_U = None):
        AbstractEnKF.__init__(self, t_integrator, t_obsOp, t_Ns, t_covInflation)
        self.setEnTKFParameters(t_U)

    #_________________________

    def setEnTKFParameters(self, t_U = None):
        if t_U is None:
            self.m_U = np.eye(self.m_Ns)
        else:
            self.m_U = t_U

    #_________________________

    def analyse(self, t_nt, t_observation):
        # analyse observation at time nt

        # Ensemble means
        xf_m  = self.m_x.mean(axis = 0)
        Hxf   = self.m_observationOperator.process(self.m_x, t_nt)
        Hxf_m = Hxf.mean(axis = 0)

        # Normalized anomalies
        Xf = ( self.m_x - xf_m ) / np.sqrt( self.m_Ns - 1.0 )
        Yf = ( Hxf - Hxf_m ) / np.sqrt( self.m_Ns - 1.0 )

        # Analyse
        rsigma_o = self.m_observationOperator.errorStdDevMatrix_diag(t_nt, self.m_spaceDimension)
        S        = Yf / rsigma_o
        delta    = ( t_observation - Hxf_m ) / rsigma_o

        Tm1      = np.eye(self.m_Ns) + np.dot( S , np.transpose(S) )
        D_T, V_T = np.linalg.eigh(Tm1)

        # w = T * S * delta 
        #   = ( (V/D) * V^T * S * delta
        w   = np.dot( V_T / D_T , np.dot ( np.transpose(V_T) , np.dot( S , delta ) ) )

        # update
        # x <- xa + U * T ^ (1/2) * Xf = xa + U * (V/D^(1/2)) * V^T * Xf
        self.m_x = xf_m + np.dot( w , Xf )
        self.m_x = self.m_x + np.sqrt( self.m_Ns - 1.0 ) * np.dot( self.m_U , np.dot( V_T / np.sqrt( D_T ) , np.dot( np.transpose(V_T) , Xf ) ) )

        # Estimation and inflation 
        self.m_estimate[t_nt, :] = self.estimateAndInflate() 

#__________________________________________________

