#!/usr/bin/env python

#__________________________________________________
# pyLorenz/filters/kalman/
# lastochasticenkf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/12/5
#__________________________________________________
#
# class to handle a stochastic EnKF
# that uses localisation (Local Analyse version)
#

import numpy as np

from filters.kalman.abstractenkf import AbstractEnKF
from utils.localisation.taper    import gaussian_tapering, gaspari_cohn_tapering, heaviside_tapering

#__________________________________________________

class LAStochasticEnKF(AbstractEnKF):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation, t_rcond,
            t_localisation_radius, t_taper_function):
        AbstractEnKF.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation, t_rcond)
        self.set_LAStochasticEnKF_parameters(t_localisation_radius, t_taper_function)

    #_________________________

    def set_LAStochasticEnKF_parameters(self, t_localisation_radius, t_taper_function):
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

        # perturb observations
        oe    = self.m_observationOperator.drawErrorSamples(t_t, (self.m_Ns, t_observation.size))
        sigma = self.m_observationOperator.errorCovarianceMatrix(t_t)
        # shortcut for forecast ensemble
        xf    = self.m_x[self.m_integrationIndex]
        # apply observation operator to forecast ensemble
        self.m_observationOperator.deterministicObserve(xf, t_t, self.m_Hxf)

        # Ensemble means
        xf_m  = xf.mean(axis = 0)
        oe_m  = oe.mean(axis = 0)
        Hxf_m = self.m_Hxf.mean(axis = 0)

        # Normalized anomalies
        Xf    = ( xf - xf_m ) / np.sqrt( self.m_Ns - 1.0 )
        Yf    = ( self.m_Hxf - Hxf_m ) / np.sqrt( self.m_Ns - 1.0 )

        for dimension in range(self.m_spaceDimension):
            # local analyse for the given dimension
            S     = Yf * self.m_localisation_coefficients[dimension]
            delta = ( t_observation + oe - oe_m - self.m_Hxf ) * self.m_localisation_coefficients[dimension]
            # local Kalman gain
            K     = np.dot ( np.linalg.pinv ( np.dot ( np.transpose ( S ) , S ) + sigma , self.m_rcond ) , np.dot ( np.transpose ( S ) , Xf[:, dimension] ) )
            # local update for the given dimension
            # the other values are dropped
            xf[:, dimension] += np.dot ( delta , K )

#__________________________________________________

