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
from scipy.optimize          import linprog

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

    def regularisation(self, t_x, t_weights):
        # return regularisation errors
        return self.m_regulariser.regularisation(t_x, t_weights)

    #_________________________

    def resample(self, t_weights, t_x):
        # check if resampling is needed
        self.m_resampled = self.m_trigger(t_weights.m_Neff, t_weights.m_max_w)
        if self.m_resampled:
            # resampling indices
            indices = self.resampling_indices(t_weights.m_w)
            # regularisation
            errors  = self.regularisation(t_x, t_weights.m_w)
            # resample
            t_x[:]  = t_x[indices] + errors
            # re-initialise weights
            t_weights.initialise()

    #_________________________

    def record(self, t_forecast_or_analyse, t_output, t_label):
        if t_forecast_or_analyse == 'analyse':
            t_output.record(t_label, 'analyse_resampled', 1.0*self.m_resampled)

#__________________________________________________

class OTCouplingResampler(object):

    #_________________________

    def __init__(self, t_rng, t_trigger, t_trigger_arg, t_regulariser):
        # random number generator
        self.m_rng         = t_rng
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

    def optimal_coupling(self, t_x, t_w):
        # distance matrix
        Ns = t_x.shape[0]
        c  = np.zeros((Ns, Ns))
        for i in range(Ns):
            c[i] = ( ( t_x - t_x[i] ) ** 2 ) . sum ( axis = 1 )
        c = c.reshape((Ns*Ns,))

        # constraint matrix
        A = np.zeros((2*Ns-1, Ns))
        # sum over each column
        for j in range(Ns):
            A[:Ns, j*Ns:(j+1)*Ns] = np.eye(Ns)
        # sum over each row
        for i in range(Ns-1):
            A[Ns+i, i*Ns:(i+1)*Ns] = np.ones(Ns)
        
        # constraint vector
        b      = np.zeros(2*Ns-1)
        b[:Ns] = 1 / Ns
        b[Ns:] = t_w[:-1]

        # solve linear programming problem
        res = linprog(c, A_eq=A, b_eq=b)
        return Ns * res.x.reshape((Ns, Ns))

    #_________________________
    
    def regularisation(self, t_x, t_w):
        # return regularisation errors
        return self.m_regulariser.regularisation(t_x, t_w)

    #_________________________

    def resample(self, t_weights, t_x):
        # check if resampling is needed
        self.m_resampled = self.m_trigger(t_weights.m_Neff, t_weights.m_max_w)
        if self.m_resampled:
            # coupling
            p = self.optimal_coupling(t_x, t_weights.m_w)
            # regularisation
            errors  = self.regularisation(t_x, t_weights.m_w)
            # resample
            t_x[:, :] = np.dot( p.transpose() , t_x ) + errors
            # re-initialise weights
            t_weights.initialise()

    #_________________________

    def record(self, t_forecast_or_analyse, t_output, t_label):
        if t_forecast_or_analyse == 'analyse':
            t_output.record(t_label, 'analyse_resampled', 1.0*self.m_resampled)

#__________________________________________________

