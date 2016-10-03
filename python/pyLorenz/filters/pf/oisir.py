#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# oisir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/28
#__________________________________________________
#
# class to handle a SIR particle filter that uses the optimal importance function as proposal
#
# We assume that noise are additive (i.e. StochasticProcess class is used) and gaussian (i.e. the statistics are 2nd order)
# and components are independant (i.e observation operator [or its differential if it is not linear] is diagonal, covariance matrices 
# for observation and model noise too)
#
# The model does not need to be linear, however the observation operator does. If not, then it will be linearized.
#

import numpy as np
import numpy.random as rnd

from sir                                             import SIRPF
from ...utils.integration.rk4integrator              import DeterministicRK4Integrator
from ...observations.iobservations                   import StochasticIObservations
from ...utils.resampling.stochasticuniversalsampling import StochasticUniversalResampler
from ...utils.trigger.thresholdtrigger               import ThresholdTrigger

#__________________________________________________

class OISIRPF(SIRPF):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_resampler = StochasticUniversalResampler(), 
            t_observationVarianceInflation = 1.0, t_resamplingTrigger = ThresholdTrigger(0.3), t_Ns = 10):
        SIRPF.__init__(self, t_integrator, t_obsOp, t_resampler, t_observationVarianceInflation, t_resamplingTrigger, t_Ns)
        self.m_deterministicIntegrator          = self.m_integrator.deterministicIntegrator()
        self.m_deterministicObservationOperator = self.m_observationOperator.deterministicObservationOperator()

    #_________________________

    def forecast(self, t_ntStart, t_ntEnd, t_observation):
        # integrate particles from ntStart to ntEnd, given the observation at ntEnd
        # this includes analyse step for conveniance

        for nt in t_ntStart + np.arange(t_ntEnd-t_ntStart-1):
            self.m_x              = self.m_integrator.process(self.m_x, nt)
            self.m_estimate[nt+1] = self.estimate()

        self.m_neff[t_ntStart+1:t_ntEnd+1] = self.Neff()

        # for the last integration, we use the optimal importance proposal

        sigma_m = np.tile(self.m_integrator.m_errorGenerator.m_sigma, (self.m_Ns, 1))
        sigma_o = np.tile(self.m_observationOperator.m_errorGenerator.m_sigma, (self.m_Ns, 1))
        fx      = self.m_deterministicIntegrator.process(self.m_x, t_ntEnd-1)
        H       = self.m_observationOperator.diagonalDifferential(self.m_x, t_ntEnd*self.m_integrator.m_dt)

        # proposal
        sigmaP = 1.0 / ( 1.0 / sigma_m + H * ( 1.0 / sigma_o ) * H )
        ycorr  = np.tile(t_observation, (self.m_Ns, 1))
        if not self.m_observationOperator.isLinear():
            ycorr += H * fx - self.m_deterministicObservationOperator.process(fx, t_ntEnd*self.m_integrator.m_dt)
        meanP  = sigmaP * ( ( 1.0 / sigma_m ) * fx + H * ( 1.0 / sigma_o ) * ycorr )

        # sample x[ntEnd] according to N(meanP, sigmaP)
        self.m_x = rnd.normal(meanP, sigmaP)

        # classic analyse process
        if self.m_observationOperator.isLinear():
            sigma = 1.0 / ( sigma_m + H * sigma_o * H )
            ino   = ycorr - H * fx
            w     = - 0.5 * (ino*sigma*ino).sum(axis = 1) # p(obs|x[nt-1])
        else:
            ino   = np.tile(t_observation, (self.m_Ns, 1)) - self.m_observationOperator.process(self.m_x)
            w     = - 0.5 * (ino*(1.0/sigma_o)*ino).sum(axis = 1) # p(obs|x[nt])
            me    = self.m_x - fx
            w    += - 0.5 * (me*(1.0/sigma_m)*me).sum(axis = 1) # p(x[nt]|x[nt-1])
            pe    = self.m_x - meanP
            w    -= - 0.5 * (pe*(1.0/self.sigmaP)*pe).sum(axis = 1) # proposal

        self.m_w += w
        self.normaliseWeights()
        self.resample(t_ntEnd)

        # estimation after the analyse
        self.m_estimate[t_ntEnd] = self.estimate()
        self.m_neff[t_ntEnd]     = self.Neff()

    #_________________________

    def analyse(self, t_nt, t_obs):
        if t_nt == 0:
            SIRPF.analyse(self, t_nt, t_obs)

#__________________________________________________

