#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# asir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/6
#__________________________________________________
#
# class to handle an ASIR particle filter
#

import numpy as np

from sir                                          import SIRPF
from ...utils.integration.rk4integrator           import DeterministicRK4Integrator
from ...observations.iobservations                import StochasticIObservations
from ...utils.random.stochasticuniversalresampler import StochasticUniversalResampler
from ...utils.trigger.thresholdtrigger            import ThresholdTrigger

#__________________________________________________

class ASIRPF(SIRPF):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_resampler = StochasticUniversalResampler(), 
            t_observationVarianceInflation = 1.0, t_resamplingTrigger = ThresholdTrigger(0.3), t_Ns = 10):
        SIRPF.__init__(self, t_integrator, t_obsOp, t_resampler, t_observationVarianceInflation, t_resamplingTrigger, t_Ns)

    #_________________________

    def forecast(self, t_ntStart, t_ntEnd, t_observation):
        # integrate particles from ntStart to ntEnd, given the observation at ntEnd

        for nt in t_ntStart + np.arange(t_ntEnd-t_ntStart-1):
            self.m_x                 = self.m_integrator.process(self.m_x, nt)
            self.m_estimate[nt+1, :] = self.estimate()

        self.m_neff[t_ntStart+1:t_ntEnd+1] = self.Neff()

        # for the last integration, we use the ASIR

        # first stage
        fsx = self.m_integrator.deterministicProcess(self.m_x, t_ntEnd-1)
        fsw = self.m_observationOperator.pdf(t_observation, fsx, t_ntEnd, self.m_observationVarianceInflation) # p(obs|fsx)

        # sample indices from Proba(i = k) = w[k] + fsw[k]
        # note that we do not need the weights to be normalized since the resampler make sure of it
        indices = self.m_resampler.sampleIndices(self.m_Ns, self.m_w+fsw)

        # now integrate particles from ntEnd-1 to ntEnd given the indices
        # i.e. particle #k at time ntEnd will be integrated from particle #indices[k]
        # this is equivalent to setting self.m_x[:] = self.m_x[indices[:]] and then integrating
        self.m_x = self.m_integrator.process(self.m_x[indices], t_ntEnd-1)

        # correct weigths to account for the resampling step self.m_x[:] <- self.m_x[indices[:]]
        # i.e. w[i] = 1 / p ( obs | fsx[indices[i]] )
        self.m_w = - fsw[indices]
        # note : the values of self.m_estimate[t_ntEnd] and self.m_neff[t_ntEnd] do not matter since it will be replaced after analyse

#__________________________________________________

