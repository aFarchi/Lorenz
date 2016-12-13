#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# pennyslpf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/12/12
#__________________________________________________
#
# class to handle a SIR particle filter
# with localisation
#

import numpy as np

from filters.abstractensemblefilter import AbstractEnsembleFilter
from utils.localisation.taper       import gaussian_tapering, gaspari_cohn_tapering, heaviside_tapering

#__________________________________________________

class PennysLPF(AbstractEnsembleFilter):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_resampler,
            t_taper_function, t_localisation_radius, t_smoothing_strength, t_adaptative_inflation, t_rng):
        AbstractEnsembleFilter.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields)
        self.set_PennysLPF_parameters(t_resampler, t_taper_function, t_localisation_radius, t_smoothing_strength)
        self.set_PennysLPF_tmp_arrays()

    #_________________________

    def set_PennysLPF_parameters(self, t_resampler, t_taper_function, t_localisation_radius, t_smoothing_strength, t_adaptative_inflation, t_rng):
        # resampler
        self.m_resampler  = t_resampler
        # smoothing strength
        self.m_smoothing_strength = t_smoothing_strength
        # apply adaptative inflation
        self.m_adaptative_inflation = t_adaptative_inflation
        # RNG
        self.m_rng = t_rng

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

    def set_PennysLPF_tmp_arrays(self):
        # allocate temporary arrays
        self.m_Hx  = np.zeros((self.m_Ns, self.m_observationOperator.m_spaceDimension))

    #_________________________

    def initialise(self):
        AbstractEnsembleFilter.initialise(self)

    #_________________________

    def analyse(self, t_t, t_observation):
        # shortcut
        xf = self.m_x[self.m_integrationIndex]

        xa  = np.zeros(xf.shape)
        xas = np.zeros(xf.shape)

        # apply observation operator to ensemble
        self.m_observationOperator.deterministicObserve(xf, t_t, self.m_Hx)

        res_ind = np.zeros(xf.shape)

        # local analyse
        for dimension in range(self.m_spaceDimension):
            # localisation coefficients
            loc_c  = self.m_localisation_coefficients[dimension]
            # log-likelihood
            log_w  = self.m_observationOperator.pdf(t_observation*loc_c, self.m_Hx*loc_c, t_t)
            # normalise log-likelihood
            max_w  = log_w.max()
            log_w -= max_w + np.log(np.exp(log_w-max_w).sum())
            w      = np.exp(log_w)
            # resample according to the likelihood
            res_ind[:, dimension] = self.m_resampler.resampling_indices(w)
            xa[:, dimension]      = xf[res_ind[:, dimension], dimension]

        if self.m_smoothing_strength > 0:
            # smoothing by weigths
            for dimension in range(self.m_spaceDimension):
                # localisation coefficients
                loc_c  = self.m_localisation_coefficients[dimension]
                loc_c /= loc_c.sum()
                # smoothing
                for dimension_near in range(self.m_spaceDimension):
                    xas[:, dimension] += xf[res_ind[:, dimension_near], dimension] * loc_c[dimension_near]

        xf[:] = self.m_smoothing_strength * xas[:] + ( 1 - self.m_smoothing_strength ) * xa[:]

        if self.m_adaptative_inflation > 0:
            std     = xf.std(axis = 0) # std is set to the ensemble spread [gaussian kernels with bandwidth=1]
            std     = np.maximum(std, self.m_adaptative_inflation)
            errors  = std * self.m_rng.standard_normal(xf.shape)
            xf     += errors - errors.mean(axis = 0) # add error samples and remove sample mean

    #_________________________

    def estimate(self):
        # mean of x
        self.m_estimation = self.m_x[self.m_integrationIndex].mean(axis = -2 )

#__________________________________________________

