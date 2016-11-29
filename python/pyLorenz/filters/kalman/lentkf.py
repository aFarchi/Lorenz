#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/kalman
# lentkf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/29
#__________________________________________________
#
# class to handle an local ensemble transform kalman filter
#

import numpy as np

from filters.kalman.abstractenkf import AbstractEnKF
from utils.localisation.taper    import gaussian_tapering, gaspari_cohn_tapering, heaviside_tapering

#__________________________________________________

class LEnTKF(AbstractEnKF):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observation_operator, t_observation_times, t_output, t_label, t_Ns, t_output_fields, t_inflation, 
            t_rcond, t_localisation_radius, t_taper_function, t_U = None):
        AbstractEnKF.__init__(self, t_initialiser, t_integrator, t_observation_operator, t_observation_times, t_output, t_label, t_Ns, t_output_fields, t_inflation, t_rcond)
        self.set_LEnTKF_parameters(t_localisation_radius, t_taper_function, t_U)

    #_________________________

    def set_LEnTKF_parameters(self, t_localisation_radius, t_taper_function, t_U):
        # U
        if t_U is None:
            self.m_U = np.eye(self.m_Ns)
        else:
            self.m_U = t_U

        # tapper function
        if t_taper_function == 'Gaussian':
            taper = gaussian_tapering
        elif t_taper_function == 'Gaspari-Cohn':
            taper = gaspari_cohn_tapering
        elif t_taper_function == 'Heaviside':
            taper = heaviside_tapering
        # localisation coefficients
        self.m_localisation_coefficients = np.zeros((self.m_spaceDimension, self.m_observationOperator.m_spaceDimension))
        for d in range(self.m_spaceDimension):
            coefficients = taper(self.m_integrator.m_integrationStep.m_model.distance_to_dimension(d), t_localisation_radius)
            self.m_localisation_coefficients[d, :] = self.m_observationOperator.cast_localisation_coefficients_to_observation_space(coefficients)

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
            # local analyse for the given dimension
            S       = self.m_observationOperator.applyRightErrorStdDevMatrix_inv(Yf*self.m_localisation_coefficients[dimension])
            delta   = self.m_observationOperator.applyLeftErrorStdDevMatrix_inv((t_observation-Hxf_m)*self.m_localisation_coefficients[dimension])

            Tm1     = np.eye(self.m_Ns) + np.dot(S, np.transpose(S)) # T^-1
            U, s, V = np.linalg.svd(Tm1)
            # note: T^-1 = ( U * s ) * V hence T = ( tV / s ) * tU

            # w = T * S * delta
            w       = np.dot ( np.dot ( np.transpose(V) * self.reciprocal(s) , np.transpose(U) ) , np.dot ( S , delta ) )

            # local update for the given dimension
            # the other values are dropped
            self.m_x[self.m_integrationIndex, :, dimension] = ( xf_m[dimension] + 
                    np.dot( w + np.sqrt( self.m_Ns - 1.0 ) * np.dot( self.m_U , np.dot( np.transpose(V) * self.reciprocal(np.sqrt(s)) , np.transpose(U) ) ) , 
                        Xf[:, dimension] ) )

#__________________________________________________

