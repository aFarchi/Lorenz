#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# lentkf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/21
#__________________________________________________
#
# class to handle an local ensemble transform kalman filter
#

import numpy as np

from filters.kalman.abstractenkf import AbstractEnKF

#__________________________________________________

class LEnTKF(AbstractEnKF):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observation_operator, t_observation_times, t_output, t_label, t_Ns, t_output_fields, t_inflation, 
            t_rcond, t_localisation_length, t_U = None):
        AbstractEnKF.__init__(self, t_initialiser, t_integrator, t_observation_operator, t_observation_times, t_output, t_label, t_Ns, t_output_fields, t_inflation, t_rcond)
        self.set_LEnTKF_parameters(t_localisation_length, t_U)

    #_________________________

    def set_LEnTKF_parameters(self, t_localisation_length, t_U):
        # localisation length
        self.m_localisation_length = t_localisation_length
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

        for dimension in range(self.m_spaceDimension):
            # localisation
            nearest_x_dimensions = self.m_integrator.m_integrationStep.m_model.nearest_dimensions(dimension, self.m_localisation_length)
            nearest_y_dimensions = self.m_observationOperator.nearest_y_dimensions(nearest_x_dimensions)

            # local analyse
            S       = self.m_observationOperator.applyRightErrorStdDevMatrix_inv_local(Yf, nearest_y_dimensions)
            delta   = self.m_observationOperator.applyLeftErrorStdDevMatrix_inv_local(t_observation-Hxf_m, nearest_y_dimensions)

            Tm1     = np.eye(self.m_Ns) + np.dot(S, np.transpose(S)) # T^-1
            U, s, V = np.linalg.svd(Tm1)
            # note: T^-1 = ( U * s ) * V hence T = ( tV / s ) * tU

            # w = T * S * delta
            w       = np.dot ( np.dot ( np.transpose(V) * self.reciprocal(s) , np.transpose(U) ) , np.dot ( S , delta ) )

            # local update
            self.m_x[self.m_integrationIndex, :, dimension] = ( xf_m[dimension] + 
                    np.dot( w + np.sqrt( self.m_Ns - 1.0 ) * np.dot( self.m_U , np.dot( np.transpose(V) * self.reciprocal(np.sqrt(s)) , np.transpose(U) ) ) , 
                        Xf[:, dimension] ) )

#__________________________________________________

