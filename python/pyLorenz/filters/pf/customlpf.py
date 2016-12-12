#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# customlpf.py
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

class CustomLPF(AbstractEnsembleFilter):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_resampler, t_rng,
            t_taper_function, t_localisation_radius):
        AbstractEnsembleFilter.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields)
        self.set_CustomLPF_parameters(t_resampler, t_rng, t_taper_function, t_localisation_radius)
        self.set_CustomLPF_tmp_arrays()

    #_________________________

    def set_CustomLPF_parameters(self, t_resampler, t_rng, t_taper_function, t_localisation_radius):
        # random number generator
        self.m_rng = t_rng
        # resampler
        self.m_resampler  = t_resampler

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

    def set_CustomLPF_tmp_arrays(self):
        # allocate temporary arrays
        self.m_Hx  = np.zeros((self.m_Ns, self.m_observationOperator.m_spaceDimension))
        self.m_log_w = - np.log(self.m_Ns) * np.ones(self.m_Ns)

    #_________________________

    def initialise(self):
        AbstractEnsembleFilter.initialise(self)

    #_________________________

    def forecast(self, t_tStart, t_tEnd, t_observation):

        # auxiliary variables
        sigma_m = self.m_integrator.errorCovarianceMatrix_diag(t_tStart, t_tEnd) # note: this line only works for BasicStochasticIntegrator instances
        sigma_o = self.m_observationOperator.errorCovarianceMatrix_diag(t_tEnd, self.m_spaceDimension)
        
        # deterministic integration
        self.m_integrationIndex = self.m_integrator.deterministicIntegrate(self.m_x, t_tStart, t_tEnd, self.m_dx)
        if self.m_integrationIndex == 0:
            return # if no integration, then just return

        fx = np.copy(self.m_x[self.m_integrationIndex])
        H  = self.m_observationOperator.differential_diag(fx, t_tEnd)
        y  = self.m_observationOperator.castObservationToStateSpace(t_observation, t_tEnd, self.m_spaceDimension)

        # proposal
        sigma_p = 1.0 / ( 1.0 / sigma_m + H * ( 1.0 / sigma_o ) * H )
        mean_p  = sigma_p * ( ( 1.0 / sigma_m ) * fx + H * ( 1.0 / sigma_o ) * y )

        # draw x at tEnd from proposal
        self.m_x[self.m_integrationIndex] = mean_p + np.sqrt(sigma_p) * self.m_rng.standard_normal(self.m_x[self.m_integrationIndex].shape)

        # Compute weights
        # w = p ( observation | x[tStart] ) / p( observation | x[tEnd] )
        s = 1.0 / ( sigma_o + H * sigma_m * H )
        d = y - H * fx
        self.m_log_w = - 0.5 * ( d * s * d ).sum(axis = -1)
        self.m_observationOperator.deterministicObserve(self.m_x[self.m_integrationIndex], t_tEnd, self.m_Hx)
        self.m_log_w -= self.m_observationOperator.pdf(t_observation, self.m_Hx, t_tEnd)

        # Normalise weights
        log_w_max     = self.m_log_w.max()
        self.m_log_w -= log_w_max + np.log(np.exp(self.m_log_w-log_w_max).sum())

    #_________________________

    def analyse(self, t_t, t_observation):
        # shortcut
        xf = self.m_x[self.m_integrationIndex]
        xa = np.zeros(xf.shape)

        # apply observation operator to ensemble
        self.m_observationOperator.deterministicObserve(xf, t_t, self.m_Hx)

        # local analyse
        for dimension in range(self.m_spaceDimension):
            # localisation coefficients
            loc_c  = self.m_localisation_coefficients[dimension]
            # log-likelihood
            log_w  = self.m_log_w + self.m_observationOperator.pdf(t_observation*loc_c, self.m_Hx*loc_c, t_t)
            # normalise log-likelihood
            max_w  = log_w.max()
            log_w -= max_w + np.log(np.exp(log_w-max_w).sum())
            w      = np.exp(log_w)
            # resample according to the likelihood
            ind    = self.m_resampler.resampling_indices(w)

            xa[:, dimension] = xf[ind, dimension]

        xf[:] = xa[:]

    #_________________________

    def estimate(self):
        # mean of x
        self.m_estimation = self.m_x[self.m_integrationIndex].mean(axis = -2 )

#__________________________________________________

