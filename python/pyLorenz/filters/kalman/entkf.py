#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# entkf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/20
#__________________________________________________
#
# class to handle an ensemble transform kalman filter
#

import numpy as np

from abstractenkf import AbstractEnKF

#__________________________________________________

class EnTKF(AbstractEnKF):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation, t_U = None):
        AbstractEnKF.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation)
        self.setEnTKFParameters(t_U)

    #_________________________

    def setEnTKFParameters(self, t_U):
        # U
        if t_U is None:
            self.m_U = np.eye(self.m_Ns)
        else:
            self.m_U = t_U

    #_________________________

    def analyse(self, t_t, t_observation):
        # analyse observation at time t

        # shortcut for forecast ensemble
        xf    = self.m_x[self.m_integrationIndex]
        # apply observation operators to forecast ensemble
        self.m_observationOperator.deterministicObserve(xf, t_t, self.m_Hxf)

        # Ensemble means
        xf_m  = xf.mean(axis = 0)
        Hxf_m = self.m_Hxf.mean(axis = 0)

        # Normalized anomalies
        Xf    = ( xf - xf_m ) / np.sqrt( self.m_Ns - 1.0 )
        Yf    = ( self.m_Hxf - Hxf_m ) / np.sqrt( self.m_Ns - 1.0 )

        # Analyse
        S        = self.m_observationOperator.applyRightErrorStdDevMatrix_inv(Yf)
        delta    = self.m_observationOperator.applyLeftErrorStdDevMatrix_inv(t_observation-Hxf_m)

        Tm1      = np.eye(self.m_Ns) + np.dot( S , np.transpose(S) ) # T^-1
        U, s, V  = np.linalg.svd(Tm1) 
        # note: T^-1 = ( U * s ) * V hence T = ( tV / s ) * tU

        # w = T * S * delta 
        w = np.dot ( np.dot ( np.transpose(V) / s , np.transpose(U) ) , np.dot ( S , delta ) )

        # update
        # x <- xa + self.m_U * T ^ (1/2) * Xf = xf_m + w * Xf + self.m_U * T ^ (1/2) * Xf
        self.m_x[self.m_integrationIndex] = ( xf_m + 
                np.dot( w + np.sqrt( self.m_Ns - 1.0 ) * np.dot( self.m_U , np.dot( np.transpose(V) / np.sqrt(s) , np.transpose(U) ) ) , Xf ) )

#__________________________________________________

