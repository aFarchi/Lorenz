#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# sir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/27
#__________________________________________________
#
# class to handle a SIR particle filter that uses the optimal importance function
# assuming noise are additive (i.e. StochasticProcess class is used) gaussian and independant
# for both observation operator and integrator
#

import numpy as np
import numpy.random as rnd

from sir                                             import SIRPF
from ...utils.integration.rk4integrator              import DeterministicRK4Integrator
from ...observations.iobservations                   import StochasticIObservations
from ...utils.resampling.stochasticuniversalsampling import StochasticUniversalResampler

#__________________________________________________

class OISIRPF(SIRPF):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_resampler = StochasticUniversalResampler(), 
            t_observationVarianceInflation = 50.0, t_resamplingThreshold = 0.3):
        SIRPF.__init__(self, t_integrator, t_obsOp, t_resampler, t_observationVarianceInflation, t_resamplingThreshold)
        self.m_deterministicIntegrator          = self.m_integrator.deterministicIntegrator()
        self.m_deterministicObservationOperator = self.m_observationOperator.deterministicObservationOperator()

    #_________________________

    def reweight(self, t_nt, t_obs):
        if t_nt == 0:
            # no forecast was performed yet
            w = self.m_observationOperator.observationPDF(t_obs, self.m_x, t_nt*self.m_integrator.m_dt, self.m_observationVarianceInflation) # p(obs|x[nt])
        else:

            # shape = (3,)
            sigma_m = self.m_integrator.m_errorGenerator.m_sigma
            sigma_o = self.m_observationOperator.m_errorGenerator.m_sigma

            # shape = (Ns, 3)
            sigma_m = np.tile(sigma_m, (self.m_Ns, 1))
            sigma_o = np.tile(sigma_o, (self.m_Ns, 1))
            H       = self.m_observationOperator.diagonalDifferential(self.m_x, t_nt*self.m_integrator.m_dt)

            if self.m_observationOperator.isLinear():
                sigma   = 1.0 / ( sigma_m + H * sigma_o * H )
                ino     = np.tile(t_obs, (self.m_Ns, 1)) - H * self.m_fx 
                w       = - 0.5 * (ino*sigma*ino).sum(axis = 1) # p(obs|x[nt-1])

            else:
                y    = np.tile(t_obs, (self.m_Ns, 1))
                ino  = y - self.m_observationOperator.process(self.m_x)
                w    = - 0.5 * (ino*(1.0/sigma_o)*ino).sum(axis = 1) # p(obs|x[nt])
                me   = self.m_x - self.m_fx
                w   += - 0.5 * (me*(1.0/sigma_m)*me).sum(axis = 1) # p(x[nt]|x[nt-1])
                pe   = self.m_x - self.meanP
                w   -= - 0.5 * (pe*(1.0/self.sigmaP)*pe).sum(axis = 1) # proposal

        # reweight ensemble
        self.m_w += w

    #_________________________

    def forecast(self, t_ntStart, t_ntEnd, t_observation):
        # integrate particles from ntStart to ntEnd, given the observation at ntEnd
        estimation = np.zeros((t_ntEnd-t_ntStart, self.m_integrator.m_model.m_stateDimension))
        for nt in np.arange(t_ntEnd-t_ntStart-1):
            #self.m_x         = self.m_deterministicIntegrator.process(self.m_x, t_ntStart+nt)
            self.m_x         = self.m_integrator.process(self.m_x, t_ntStart+nt)
            estimation[nt] = self.estimate()

        # shape = (3,)
        sigma_m   = self.m_integrator.m_errorGenerator.m_sigma
        sigma_o   = self.m_observationOperator.m_errorGenerator.m_sigma

        # shape = (Ns, 3)
        sigma_m   = np.tile(sigma_m, (self.m_Ns, 1))
        sigma_o   = np.tile(sigma_o, (self.m_Ns, 1))
        self.m_fx = self.m_deterministicIntegrator.process(self.m_x, t_ntEnd - 1)
        H         = self.m_observationOperator.diagonalDifferential(self.m_x, t_ntEnd*self.m_integrator.m_dt)

        # proposal
        self.sigmaP = 1.0 / ( 1.0 / sigma_m + H * ( 1.0 / sigma_o ) * H )
        ycorr       = np.tile(t_observation, (self.m_Ns, 1))
        if not self.m_observationOperator.isLinear():
            ycorr  += - self.m_deterministicObservationOperator(self.m_fx) + H * self.m_fx
        self.meanP  = self.sigmaP * ( ( 1.0 / sigma_m ) * self.m_fx + H * ( 1.0 / sigma_o ) * ycorr )

        # sample x[ntEnd] according to N(meanP, sigmaP)
        self.m_x = rnd.normal(self.meanP, self.sigmaP)
        # note : the value of estimation[t_ntEnd] does not matter since it will be replaced after analyse
        # so we do not compute it
        return estimation

#__________________________________________________

