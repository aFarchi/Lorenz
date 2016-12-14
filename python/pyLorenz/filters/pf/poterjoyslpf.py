#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# poterjoyslpf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/12/8
#__________________________________________________
#
# class to handle a SIR particle filter
# with localisation
#

import numpy as np

from filters.abstractensemblefilter import AbstractEnsembleFilter
from utils.localisation.taper       import gaussian_tapering, gaspari_cohn_tapering, heaviside_tapering

#__________________________________________________

class PoterjoysLPF(AbstractEnsembleFilter):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_resampler,
            t_relaxation, t_taper_function, t_localisation_radius):
        AbstractEnsembleFilter.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields)
        self.set_PoterjoysLPF_parameters(t_resampler, t_relaxation, t_taper_function, t_localisation_radius)
        self.set_PoterjoysLPF_tmp_arrays()

    #_________________________

    def set_PoterjoysLPF_parameters(self, t_resampler, t_relaxation, t_taper_function, t_localisation_radius):
        # resampler
        self.m_resampler  = t_resampler
        # relaxation
        self.m_relaxation = t_relaxation

        # tapper function
        if t_taper_function == 'Gaussian':
            taper = gaussian_tapering
        elif t_taper_function == 'Gaspari-Cohn':
            taper = gaspari_cohn_tapering
        elif t_taper_function == 'Heaviside':
            taper = heaviside_tapering
        # Localisation coefficients
        self.m_localisation_coefficients = np.zeros((self.m_spaceDimension, self.m_observationOperator.m_spaceDimension))
        for d in range(self.m_spaceDimension):
            coefficients = taper(self.m_integrator.m_integrationStep.m_model.distance_to_dimension(d), t_localisation_radius)
            self.m_localisation_coefficients[d, :] = self.m_observationOperator.cast_localisation_coefficients_to_observation_space(coefficients)

    #_________________________

    def set_PoterjoysLPF_tmp_arrays(self):
        # allocate temporary arrays
        self.m_Hx  = np.zeros((self.m_Ns, self.m_observationOperator.m_spaceDimension))
        self.m_Hcx = np.zeros((self.m_Ns, self.m_observationOperator.m_spaceDimension))

    #_________________________

    def initialise(self):
        AbstractEnsembleFilter.initialise(self)

    #_________________________

    def analyse(self, t_t, t_observation):
        # shortcut
        xf = self.m_x[self.m_integrationIndex]

        # apply observation operator to ensemble
        self.m_observationOperator.deterministicObserve(xf, t_t, self.m_Hx)

        prior = xf.copy()
        w     = np.ones((self.m_Ns, self.m_spaceDimension))


        # process each component of observation independantly
        for ny in range(t_observation.size):

            # sir weighting and resampling
            self.m_observationOperator.deterministicObserve(xf, t_t, self.m_Hcx)
            p     = self.m_observationOperator.local_pdf(t_observation, self.m_Hcx, t_t, ny)
            sir_w = self.m_relaxation * ( p - 1.0 ) + 1.0
            sir_W = sir_w.sum()
            sir_resampling_indices = self.m_resampler.resampling_indices(sir_w)

            # minimise adjustment for surviving particles
            u, counts = np.unique(sir_resampling_indices, return_counts = True)
            sir_resampling_indices = - np.ones(self.m_Ns, dtype = np.int)
            sir_resampling_indices[u] = u
            remaining = []
            for (i, c) in zip(u, counts):
                remaining.extend([i]*c)
            for i in range(self.m_Ns-1, -1, -1):
                if sir_resampling_indices[i] < 0:
                    sir_resampling_indices[i] = remaining.pop()

            # reweight according to observation weights
            p = self.m_observationOperator.local_pdf(t_observation, self.m_Hx, t_t, ny)
            w[:, :] *= 1 + self.m_relaxation * self.m_localisation_coefficients[:, ny] * ( np.broadcast_to(p, (self.m_spaceDimension, self.m_Ns)).transpose() - 1 )
            W     = w.sum(axis = 0)
            mean  = ( w * prior / W ) . sum ( axis = 0 )
            sigma = ( w * ( prior - mean )**2 / W ) . sum ( axis = 0 )

            epsilon = 1.e-8

            # vectorised version
            l  = self.m_localisation_coefficients[:, ny]
            c  = ( l > epsilon ) * self.m_Ns * ( 1.0 - self.m_relaxation * l ) / ( np.maximum(l, epsilon) * self.m_relaxation * sir_W )
            d  = ( ( xf[sir_resampling_indices, :] - mean + c * ( xf - mean ) ) ** 2 ) . sum (axis = 0)
            r1 = ( ( l > epsilon ) * ( d > epsilon ) * np.sqrt( sigma * ( self.m_Ns - 1.0 ) / np.maximum(d, epsilon) ) +
                    ( l > epsilon ) * ( d <= epsilon ) * 1 )
            r2 = ( ( l > epsilon ) * ( d > epsilon ) * c * r1 +
                    ( l <= epsilon ) * 1 )

            xf[:, :] = mean + r1 * ( xf[sir_resampling_indices] - mean ) + r2 * ( xf - mean )

    #_________________________

    def estimate(self):
        # mean of x
        self.m_estimation = self.m_x[self.m_integrationIndex].mean(axis = -2 )

#__________________________________________________

