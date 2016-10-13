#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# oisir_diag.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/28
#__________________________________________________
#
# class to handle a SIR particle filter that uses the optimal importance function as proposal
#
# Tweak when all matrices are diagonal
#

import numpy as np
import numpy.random as rnd

from sir                                          import SIRPF
from ...utils.integration.rk4integrator           import DeterministicRK4Integrator
from ...observations.iobservations                import StochasticIObservations
from ...utils.random.stochasticuniversalresampler import StochasticUniversalResampler
from ...utils.trigger.thresholdtrigger            import ThresholdTrigger
from ...utils.random.independantgaussianrng       import IndependantGaussianRNG

#__________________________________________________

class OISIRPF_diag(SIRPF):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_resampler = StochasticUniversalResampler(), 
            t_observationVarianceInflation = 1.0, t_resamplingTrigger = ThresholdTrigger(0.3), t_Ns = 10):
        SIRPF.__init__(self, t_integrator, t_obsOp, t_resampler, t_observationVarianceInflation, t_resamplingTrigger, t_Ns)

    #_________________________

    def forecast(self, t_ntStart, t_ntEnd, t_observation):
        # integrate particles from ntStart to ntEnd, given the observation at ntEnd
        # this includes analyse step for conveniance

        for nt in t_ntStart + np.arange(t_ntEnd-t_ntStart-1):
            self.m_x              = self.m_integrator.process(self.m_x, nt)
            self.m_estimate[nt+1] = self.estimate()

        self.m_neff[t_ntStart+1:t_ntEnd+1] = self.Neff()

        # for the last integration, we use the optimal importance proposal

        # auxiliary variables
        sigma_m = self.m_integrator.errorCovarianceMatrix_diag(t_ntEnd-1)
        sigma_o = self.m_observationOperator.errorCovarianceMatrix_diag(t_ntEnd-1, self.m_spaceDimension)
        fx      = self.m_integrator.deterministicProcess(self.m_x, t_ntEnd-1)
        H       = self.m_observationOperator.differential_diag(self.m_x, t_ntEnd)
        y       = self.m_observationOperator.castObservationToStateSpace(t_observation, t_ntEnd, self.m_spaceDimension)

        if not self.m_observationOperator.isLinear():
            y   = H * fx - self.m_observationOperator.deterministicProcess(fx, t_ntEnd) + y

        # proposal
        sigma_p  = 1.0 / ( 1.0 / sigma_m + H * ( 1.0 / sigma_o ) * H )
        mean_p   = sigma_p * ( ( 1.0 / sigma_m ) * fx + H * ( 1.0 / sigma_o ) * y )
        proposal = IndependantGaussianRNG(mean_p, np.sqrt(sigma_p))

        # sample x from N(mean_p, sigma_p)
        self.m_x = proposal.drawSample(0)

        # reweight ensemble to account for proposal

        if self.m_observationOperator.isLinear():
            # p ( y | x[ntEnd-1] )
            s         = 1.0 / ( sigma_o + H * sigma_m * H )
            d         = y - H * fx
            self.m_w -= ( d * s * d ).sum(axis = -1) / 2.0

        else:
            # p( x[ntEnd] | x[ntEnd-1] )
            me        = self.m_x - fx
            self.m_w += self.m_integrator.m_errorGenerator.pdf(me, t_ntEnd-1) # small tweak

            # proposal
            self.m_w -= proposal.pdf(self.m_x, 0)

    #_________________________

    def reweight(self, t_nt, t_observation):
        if self.m_observationOperator.isLinear():
            if t_nt == 0:
                SIRPF.reweight(self, t_nt, t_observation)
        else:
            SIRPF.reweight(self, t_nt, t_observation)

#__________________________________________________

