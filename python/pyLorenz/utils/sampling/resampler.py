#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/sampling/
# resampler.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/21
#__________________________________________________
#
# class to handle a Resampler object
#

import numpy as np
from utils.sampling.sampling import probabilistic_sampling, stochastic_universal_sampling, monte_carlo_metropolis_hastings_sampling

#__________________________________________________

class Resampler(object):

    #_________________________

    def __init__(self, t_rng, t_method, t_trigger, t_trigger_arg, t_regulariser):
        # random number generator
        self.m_rng         = t_rng
        # resampling method
        if t_method == 'Probabilistic':
            self.m_method  = probabilistic_sampling
        elif t_method == 'StochasticUniversal':
            self.m_method  = stochastic_universal_sampling
        elif t_method == 'MonteCarloMetropolisHastings':
            self.m_method  = monte_carlo_metropolis_hastings_sampling
        # trigger args
        if t_trigger_arg == None:
            def trigger(*t_args):
                return t_trigger()
        elif t_trigger_arg == 'Neff':
            def trigger(t_Neff, t_max_w):
                return t_trigger(t_Neff)
        elif t_trigger_arg == 'max_w':
            def trigger(t_Neff, t_max_w):
                return not t_trigger(t_max_w)
        # trigger
        self.m_trigger     = trigger
        # regulariser
        self.m_regulariser = t_regulariser

    #_________________________
    
    def resampling_indices(self, t_w):
        # return indices that resample from the weights w
        return self.m_method(t_w, t_w.size, self.m_rng, False)

    #_________________________

    def resample(self, t_weights, t_x):
        # check if resampling is needed
        self.m_resampled = self.m_trigger(t_weights.m_Neff, t_weights.m_max_w)
        if self.m_resampled:
            # resampling indices
            indices = self.resampling_indices(t_weights.m_w)
            # resampling and regularisation
            self.m_regulariser.regularisation(t_x, indices, t_weights.m_w)
            # re-initialise weights
            t_weights.initialise()

    #_________________________

    def record(self, t_forecast_or_analyse, t_output, t_label):
        if t_forecast_or_analyse == 'analyse':
            t_output.record(t_label, 'analyse_resampled', 1.0*self.m_resampled)

#__________________________________________________

