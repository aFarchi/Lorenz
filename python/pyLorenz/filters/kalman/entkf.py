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

class EnTKF_diag(AbstractEnKF):

    #_________________________

    def __init__(self, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation, t_U = None):
        AbstractEnKF.__init__(self, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation)
        self.setEnTKFParameters(t_U)

    #_________________________

    def setEnTKFParameters(self, t_U):
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
        rsigma_o = self.m_observationOperator.errorStdDevMatrix_diag(t_t, self.m_spaceDimension)
        S        = Yf / rsigma_o
        delta    = ( t_observation - Hxf_m ) / rsigma_o

        Tm1      = np.eye(self.m_Ns) + np.dot( S , np.transpose(S) )
        try:
            U, s, V  = np.linalg.svd(Tm1)
        except:
            print 'SVD *** time =', t_t
            raise Exception

        """
        try:
            D, V = np.linalg.eigh(Tm1)
            eps  = 1.0e-8
            zeros = 1.0 * ( np.abs(D) > eps )
            if zeros.min() == 0:
                print 'ZEROS *** time =', t_t
                print 'zeros ='
                print zeros
                print 'D ='
                print D
                raise Exception
            pos  = 1.0 * ( D > 0.0 )
            if pos.min() == 0:
                print 'POS **** time =', t_t
                print 'pos ='
                print pos
                print 'D ='
                print D

            #print np.abs(D).max(), np.abs(D).min()
                
        except:
            print 'EIGH *** time = ', t_t
            raise Exception

        D     = zeros * D + ( 1.0 - zeros )
        """

        # w = T * S * delta 
        #   = ( (V/D) * V^T * S * delta
        """w   = np.dot( V / D , np.dot ( np.transpose(V) , np.dot( S , delta ) ) )"""
        w = np.dot ( np.dot ( np.transpose(V) / s , np.transpose(U) ) , np.dot ( S , delta ) )

        # update
        # x <- xa + U * T ^ (1/2) * Xf = xa + U * (V/D^(1/2)) * V^T * Xf
        #xf = xf_m + np.dot( w , Xf ) + np.sqrt( self.m_Ns - 1.0 ) * np.dot( self.m_U , np.dot( V_T / np.sqrt( D_T ) , np.dot( np.transpose(V_T) , Xf ) ) )
        """xf = xf_m + np.dot( w + np.sqrt( self.m_Ns - 1.0 ) * np.dot( self.m_U , np.dot( V / np.sqrt( D ) , np.transpose(V) ) ) , Xf )"""
        E  = np.dot( np.dot( np.transpose(V) / np.sqrt(s) , np.transpose(U) ) , Xf )
        print E.mean(axis=0)
        xf = xf_m + np.dot(w, Xf) + np.sqrt(self.m_Ns-1.0) * E

#__________________________________________________

