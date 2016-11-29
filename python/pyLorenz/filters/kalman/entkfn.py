#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/kalman
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

from filters.kalman.abstractenkf import AbstractEnKF

#__________________________________________________

class EnTKF_N_dual(AbstractEnKF):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation, t_rcond, 
            t_minimiser, t_epsilon, t_maxZeta, t_U = None, t_order = 1):
        AbstractEnKF.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation, t_rcond)
        self.setEnTKF_N_dualParameters(t_minimiser, t_epsilon, t_maxZeta, t_U, t_order)

    #_________________________

    def setEnTKF_N_dualParameters(self, t_minimiser, t_epsilon, t_maxZeta, t_U, t_order):
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
        # order
        self.m_order     = t_order

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

        # minimise dual cost
        zetaa = self.m_minimiser.argmin(dualCost, bounds = (0.0, self.m_maxZeta))

        # analyse weights
        wa                 = np.zeros(self.m_Ns)
        wa[:D.size]        = D * delta[:D.size]
        diagonal           = np.zeros(self.m_Ns)
        diagonal[:D.size]  = D * D
        diagonal          += zetaa / ( self.m_Ns - 1.0 )
        wa                *= self.reciprocal(diagonal)
        wa                 = np.dot( U , wa )

        if self.m_order == 1:
            # new ensemble
            Xa             = np.dot( self.m_U , np.dot( U * self.reciprocal(np.sqrt(diagonal)) , np.transpose(U) ) )
        else:
            # include the second-order corrections
            wawaT          = wa * np.transpose( np.broadcast_to( wa , (self.m_Ns, self.m_Ns) ) )
            H              = np.dot( U * diagonal , np.transpose(U) ) - ( 2.0 / ( self.m_Ns + 1.0 ) ) * ( ( zetaa / ( self.m_Ns - 1.0 ) ) ** 2 ) * wawaT
            UH, DH, VH     = np.linalg.svd(H)
            Xa             = np.dot( self.m_U , np.dot( UH * self.reciprocal(np.sqrt(DH)) , VH ) )
            # note: UH / sqrt(DH) * VH = transpose(VH) / sqrt(DH) * UH since transpose(H) = H

        # update
        self.m_x[self.m_integrationIndex] = xf_m + np.dot( wa + np.sqrt( self.m_Ns - 1.0 ) * Xa , Xf )

#__________________________________________________

class EnTKF_N_primal(AbstractEnKF):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation, t_rcond, 
            t_minimiser, t_epsilon, t_U = None):
        AbstractEnKF.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation, t_rcond)
        self.setEnTKF_N_primalParameters(t_minimiser, t_epsilon, t_U)

    #_________________________

    def setEnTKF_N_primalParameters(self, t_minimiser, t_epsilon, t_U):
        # minimiser
        self.m_minimiser = t_minimiser
        # epsilon
        self.m_epsilon   = t_epsilon
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
        # ensemble mean
        xf_m  = xf.mean(axis = 0)
        # normalised anomalies
        Xf    = ( xf - xf_m ) / np.sqrt( self.m_Ns - 1.0 )

        # primal cost to minimise
        def primalCost(w):
            x     = xf_m + np.dot(w, Xf)
            self.m_observationOperator.deterministicObserve(x, t_t, self.m_Hxf[0])
            delta = t_observation - self.m_Hxf[0]
            delta = self.m_observationOperator.applyLeftErrorStdDevMatrix_inv(delta)
            return 0.5 * ( np.dot(delta, delta) + (self.m_Ns+1.0)*np.log(self.m_epsilon+np.dot(w, w)/(self.m_Ns-1.0)) )

        # gradient of primal cost
        def gradientPrimalCost(w):
            x     = xf_m + np.dot(w, Xf)
            self.m_observationOperator.deterministicObserve(x, t_t, self.m_Hxf[0])
            delta = t_observation - self.m_Hxf[0]
            delta = self.m_observationOperator.applyRightErrorCovMatrix_inv(delta)
            H     = self.m_observationOperator.differential(x, t_t)
            return - np.dot(np.dot(delta, H), np.transpose(Xf)) + ((self.m_Ns+1.0)/(self.m_Ns-1.0)) * w / (self.m_epsilon+np.dot(w, w)/(self.m_Ns-1.0))

        # first guess
        wa = np.zeros(self.m_Ns)
        # minimisation of primal cost
        wa     = self.m_minimiser.argmin(primalCost, wa, jac = gradientPrimalCost)
        argln  = self.m_epsilon + np.dot(wa, wa) / ( self.m_Ns - 1.0 )
        argln *= ( self.m_Ns - 1.0 )


        # apply observation operator to forecast ensemble
        self.m_observationOperator.deterministicObserve(xf, t_t, self.m_Hxf)
        # ensemble mean
        Hxf_m = self.m_Hxf.mean(axis = 0)
        # normalised anomalies
        Yf    = ( self.m_Hxf - Hxf_m ) / np.sqrt( self.m_Ns - 1.0 )
        Yf    = self.m_observationOperator.applyRightErrorStdDevMatrix_inv(Yf)

        # Hessian order 0
        Ha    = np.dot( Yf , np.transpose(Yf) )
        # order 1
        Ha   += ( self.m_Ns + 1.0 ) / ( argln ) * np.eye(self.m_Ns)
        # order 2
        wawaT = wa * np.transpose( np.broadcast_to( wa , (self.m_Ns, self.m_Ns) ) )
        Ha   -= 2.0 * ( self.m_Ns + 1.0 ) / ( argln**2 ) * wawaT

        # svd of H and new ensemble
        UH, DH, VH     = np.linalg.svd(Ha)
        Xa             = np.dot( self.m_U , np.dot( UH * self.reciprocal(np.sqrt(DH)) , VH ) )
        # note: UH / sqrt(DH) * VH = transpose(VH) / sqrt(DH) * UH since transpose(H) = H

        # update
        self.m_x[self.m_integrationIndex] = xf_m + np.dot( wa + np.sqrt( self.m_Ns - 1.0 ) * Xa , Xf )

#__________________________________________________

