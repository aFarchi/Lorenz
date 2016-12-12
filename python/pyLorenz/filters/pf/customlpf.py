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

        # deterministic integration
        self.m_integrationIndex = self.m_integrator.deterministicIntegrate(self.m_x, t_tStart, t_tEnd, self.m_dx)
        if self.m_integrationIndex == 0:
            return # if no integration, then just return

        # auxiliary variables
        sigma_m = self.m_integrator.errorCovarianceMatrix_diag(t_tStart, t_tEnd)[0] # note: this line only works for BasicStochasticIntegrator instances
        sigma_o = self.m_observationOperator.errorCovarianceMatrix_diag(t_tEnd, self.m_spaceDimension)[0]
        
        fx = np.copy(self.m_x[self.m_integrationIndex])
        H  = self.m_observationOperator.differential_diag(fx, t_tEnd)[0]

        # shortcuts
        xf = self.m_x[self.m_integrationIndex]
        y  = t_observation

        # "local" forecast
        for dimension in range(self.m_spaceDimension):
            # localisation coefficients
            loc_c  = self.m_localisation_coefficients[dimension]

            # proposal
            sigma_p = 1.0 / ( 1.0 / sigma_m + ( ( H * loc_c[dimension] ) ** 2 ) / sigma_o ) # 1
            mean_p  = sigma_p * ( fx[:, dimension] / sigma_m + H * loc_c[dimension] * y[dimension] / sigma_o ) # Ns

            # draw x at tEnd from proposal
            xf[:, dimension] = mean_p + np.sqrt(sigma_p) * self.m_rng.standard_normal(self.m_Ns) # Ns

            # re-weight according to p ( obs | x[tStart] )
            s     = 1.0 / ( sigma_o + sigma_m * ( H * loc_c ) ** 2 ) # Nx
            d     = loc_c * ( y - H * fx ) # Ns * Nx
            log_w = - 0.5 * ( d * s * d ) . sum ( axis = -1 ) # Ns

            # normalise log-weights
            log_w_max  = log_w.max()
            log_w     -= log_w_max + np.log(np.exp(log_w-log_w_max).sum())

            # resample
            w   = np.exp(log_w)
            ind = self.m_resampler.resampling_indices(w)
            xf[:, dimension] = xf[ind, dimension]

    #_________________________

    def analyse(self, t_t, t_observation):
        pass

    #_________________________

    def estimate(self):
        # mean of x
        self.m_estimation = self.m_x[self.m_integrationIndex].mean(axis = -2 )

#__________________________________________________

