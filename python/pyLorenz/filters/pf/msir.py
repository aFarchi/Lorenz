#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# msir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/29
#__________________________________________________
#
# class to handle a marginal SIR particle filter
#

import numpy as np
import numpy.random as rnd

from sir                                             import SIRPF
from ...utils.integration.rk4integrator              import DeterministicRK4Integrator
from ...observations.iobservations                   import StochasticIObservations
from ...utils.resampling.stochasticuniversalsampling import StochasticUniversalResampler
from ...utils.resampling.directresampling            import DirectResampler
from ...utils.trigger.thresholdtrigger               import ThresholdTrigger

#__________________________________________________

class MSIRPF(SIRPF):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_resampler = StochasticUniversalResampler(), 
            t_observationVarianceInflation = 1.0, t_resamplingTrigger = ThresholdTrigger(0.3), t_Ns = 10, t_iSampler = DirectResampler()):
        SIRPF.__init__(self, t_integrator, t_obsOp, t_resampler, t_observationVarianceInflation, t_resamplingTrigger, t_Ns)
        self.setMSIRPFParameters(t_iSampler)

    #_________________________

    def setMSIRPFParameters(self, t_iSampler = DirectResampler()):
        self.m_indicesSampler = t_iSampler

    #_________________________

    def forecast(self, t_ntStart, t_ntEnd, t_observation):
        # integrate particles from ntStart to ntEnd, given the observation at ntEnd

        for nt in t_ntStart + np.arange(t_ntEnd-t_ntStart-1):
            self.m_x              = self.m_integrator.process(self.m_x, nt)
            self.m_estimate[nt+1] = self.estimate()

        self.m_neff[t_ntStart+1:t_ntEnd+1] = self.Neff()

        # sample x[ntEnd] according to sum ( w[i] * p ( x[ntEnd] | x[ntEnd-1, i] )
        indices  = self.m_indicesSampler.sampleIndices(self.m_w)
        self.m_x = self.m_integrator.process(self.m_x[indices], t_ntEnd-1)

        # correct weights to account for the proposal
        self.m_w = np.ones(self.m_Ns)
        # note : the values of self.m_estimate[t_ntEnd] and self.m_neff[t_ntEnd] do not matter since it will be replaced after analyse

#__________________________________________________

