#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# sir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/22
#__________________________________________________
#
# class to handle a SIR particle filter that uses the optimal importance function
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
            t_observationVarianceInflation = 50.0, t_resamplingThreshold = 0.3, t_weightsTolerance = 1.0e-8):
        SIRPF.__init__(self, t_integrator, t_obsOp, t_resampler, t_observationVarianceInflation, t_resamplingThreshold, t_weightsTolerance)
        self.m_deterministicIntegrator = self.m_integrator.deterministicIntegrator()

    #_________________________

    def analyse(self, t_nt, t_obs):
        # analyse observation at time nt

        try:
            # shape = (3,)
            sigma_m = self.m_integrator.m_errorGenerator.m_sigma
            sigma_o = self.m_observationOperator.m_errorGenerator.m_sigma
            H       = self.m_observationOperator.diagonalDifferential()
            sigma   = 1.0 / ( sigma_m + H * sigma_o * H )

            # shape = (Ns, 3)
            sigma_m = np.tile(sigma_m, (self.m_Ns, 1))
            sigma_o = np.tile(sigma_o, (self.m_Ns, 1))
            H       = np.tile(H, (self.m_Ns, 1))
            sigma   = np.tile(sigma, (self.m_Ns, 1))
            y       = np.tile(t_obs, (self.m_Ns, 1))
            ino     = y - self.m_observationOperator.process(self.m_fx)

            # p(obs|x[nt-1])
            w = np.exp ( - 0.5 * np.dot ( ino , sigma * ino ) )

        except:
            # observation weights p(obs|x[nt])
            w = self.m_observationOperator.observationPDF(t_obs, self.m_x, t_nt*self.m_integrator.m_dt, self.m_observationVarianceInflation)

        if w.max() < self.m_weightsTolerance:
            ###_____________________________
            ### --->>> TO IMPROVE <<<--- ###
            ###_____________________________
            # filter has diverged from the truth...
            # ignore observation
            print('filter divergence, nt='+str(t_nt))
            w = 1.0

        # reweight ensemble
        self.m_w *= w
        # normalize weights
        self.m_w /= self.m_w.sum()
        # resample if needed
        if self.Neff() < self.m_resamplingThreshold:
            ###_______________________________
            ### --->>> REMOVE PRINT <<<--- ###
            ###_______________________________
            print('resampling, nt='+str(t_nt))
            (self.m_w, self.m_x) = self.m_resampler.resample(self.m_w, self.m_x)
            self.m_resampled.append(t_nt)

        return self.estimate()

    #_________________________

    def forecast(self, t_ntStart, t_ntEnd, t_observation):
        # integrate particles from ntStart to ntEnd, given the observation at ntEnd
        estimation = np.zeros((t_ntEnd-t_ntStart, self.m_integrator.m_model.m_stateDimension))
        for nt in np.arange(t_ntEnd-t_ntStart-1)+t_ntStart:
            self.m_x         = self.m_deterministicIntegrator.process(self.m_x, nt)
            #self.m_x         = self.m_integrator.process(self.m_x, nt)
            estimation[nt] = self.estimate()

        # shape = (3,)
        sigma_m = self.m_integrator.m_errorGenerator.m_sigma
        sigma_o = self.m_observationOperator.m_errorGenerator.m_sigma
        H       = self.m_observationOperator.diagonalDifferential()
        sigma   = 1.0 / ( 1.0 / sigma_m + H * ( 1.0 / sigma_o ) * H )

        # shape = (Ns, 3)
        sigma_m = np.tile(sigma_m, (self.m_Ns, 1))
        sigma_o = np.tile(sigma_o, (self.m_Ns, 1))
        H       = np.tile(H, (self.m_Ns, 1))
        sigma   = np.tile(sigma, (self.m_Ns, 1))
        self.m_fx = self.m_deterministicIntegrator.process(self.m_x, t_ntEnd - 1)
        
        mean    = sigma * ( ( 1.0 / sigma_m ) * self.m_fx +
                H * ( 1.0 / sigma_o ) * np.tile(t_observation, (self.m_Ns, 1)) )

        # sample x[ntEnd] according to N(mean, sigma)
        self.m_x = rnd.normal(mean, sigma)
        #wCorrection = ( np.exp ( - 0.5 * np.dot ( self.m_x - self.m_fx , ( 1.0 / sigma_m ) * ( self.m_x - self.m_fx ) ) ) /
                #np.exp ( - 0.5 * np.dot ( self.m_x - mean , ( 1.0 / sigma ) * ( self.m_x - mean ) ) )
        #estimation[t_ntEnd-1] = np.average(self.m_x, axis = 0 , weights = self.m_w*wCorrection)
        # in fact the value of estimation[t_ntEnd] does not matter since it will be replaced after analyse...
        return estimation

#__________________________________________________

