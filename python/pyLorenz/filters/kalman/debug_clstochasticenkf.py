#!/usr/bin/env python

#__________________________________________________
# pyLorenz/filters/kalman/
# debug_clstochasticenkf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/12/5
#__________________________________________________
#
# class to handle a stochastic EnKF
# that uses localisation (Covariance Localisation version)
#
# debug version that checks if background covariance matrix is indeed positive definite
#

import numpy as np

from filters.kalman.abstractenkf import AbstractEnKF
from utils.localisation.taper    import gaussian_tapering, gaspari_cohn_tapering, heaviside_tapering

#__________________________________________________

class CLStochasticEnKF(AbstractEnKF):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation, t_rcond,
            t_localisation_radius, t_taper_function):
        AbstractEnKF.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation, t_rcond)
        self.set_CLStochasticEnKF_parameters(t_localisation_radius, t_taper_function)

    #_________________________

    def set_CLStochasticEnKF_parameters(self, t_localisation_radius, t_taper_function):
        # tapper function
        if t_taper_function == 'Gaussian':
            taper = gaussian_tapering
        elif t_taper_function == 'Gaspari-Cohn':
            taper = gaspari_cohn_tapering
        elif t_taper_function == 'Heaviside':
            taper = heaviside_tapering

        # localisation coefficients
        xspace_localisation_matrix = np.zeros((self.m_spaceDimension, self.m_spaceDimension))
        for d in range(self.m_spaceDimension):
            xspace_localisation_matrix[d, :] = taper(self.m_integrator.m_integrationStep.m_model.distance_to_dimension(d), t_localisation_radius)

        self.m_localisation_matrix = self.m_observationOperator.cast_localisation_matrix_to_observation_space(xspace_localisation_matrix)

        # to count iterations where back. cov. matrix has neg. eigenvalues
        self.m_iter = 0
        self.m_neg  = 0

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

        # Background covariance matrix
        B     = self.m_localisation_matrix * np.dot ( np.transpose ( Yf ) , Yf )
        eig   = np.linalg.eigvals(B)
        if eig.min() < 0:
            print(' >>> Background covariance matrix has negative eigenvalue(s) <<< ', eig.min(), eig.max())
            self.m_neg += 1
        self.m_iter += 1

        # localised Kalman gain
        K     = np.dot ( np.linalg.pinv ( B + sigma , self.m_rcond ) , self.m_localisation_matrix * np.dot ( np.transpose ( Yf ) , Xf ) )

        # Update
        xf   += np.dot ( t_observation + oe - oe_m - self.m_Hxf , K )

    #_________________________

    def finalise(self):
        print('Fraction of iterations where the background covariance matrix has negative eigenvalue(s) :', self.m_neg/self.m_iter)
        AbstractEnKF.finalise(self)

#__________________________________________________

