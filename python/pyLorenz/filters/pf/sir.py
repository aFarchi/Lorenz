#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# sir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/22
#__________________________________________________
#
# class to handle a SIR particle filter
#

import numpy as np

from ..abstractfilter                                import AbstractFilter
from ...utils.integration.rk4integrator              import DeterministicRK4Integrator
from ...observations.iobservations                   import StochasticIObservations
from ...utils.resampling.stochasticuniversalsampling import StochasticUniversalResampler

#__________________________________________________

class SIRPF(AbstractFilter):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_resampler = StochasticUniversalResampler(), 
            t_observationVarianceInflation = 50.0, t_resamplingThreshold = 0.3, t_weightsTolerance = 1.0e-8):
        AbstractFilter.__init__(self, t_integrator, t_obsOp)
        self.setSIRParameters(t_resampler, t_observationVarianceInflation, t_resamplingThreshold, t_weightsTolerance)
        self.m_resampled = []

    #_________________________

    def setSIRParameters(self, t_resampler = StochasticUniversalResampler(), t_observationVarianceInflation = 50.0, t_resamplingThreshold = 0.3, t_weightsTolerance = 1.0e-8):
        # set resampler
        self.m_resampler = t_resampler
        # inflation of the variance of the observation pdf
        self.m_observationVarianceInflation = t_observationVarianceInflation
        # resampling threshold (for Neff)
        self.m_resamplingThreshold          = t_resamplingThreshold
        # tolerance value for the weights given by the observation pdf
        self.m_weightsTolerance = 1.0e-8

    #_________________________

    def initialise(self, t_x):
        # particles / samples
        self.m_x  = t_x
        # number of particles / samples
        self.m_Ns = t_x.shape[0]
        # relative weights
        self.m_w  = np.ones(self.m_Ns) / self.m_Ns

    #_________________________

    def Neff(self):
        # empirical effective relative sample size
        return 1.0 / ( np.power(self.m_w, 2).sum() * self.m_Ns )

    #_________________________

    def resampledSteps(self):
        return 1.0 * np.array(self.m_resampled)

    #_________________________

    def analyse(self, t_nt, t_obs):
        # analyse observation at time nt

        # observation weights
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
        for nt in np.arange(t_ntEnd-t_ntStart):
            self.m_x       = self.m_integrator.process(self.m_x, t_ntStart+nt)
            estimation[nt] = self.estimate()
        return estimation

    #_________________________

    def estimate(self):
        # mean of x
        return np.average(self.m_x, axis = 0, weights = self.m_w)

#__________________________________________________

