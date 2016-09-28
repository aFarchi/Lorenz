#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
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
            t_observationVarianceInflation = 50.0, t_resamplingThreshold = 0.3):
        AbstractFilter.__init__(self, t_integrator, t_obsOp)
        self.setSIRParameters(t_resampler, t_observationVarianceInflation, t_resamplingThreshold)
        self.m_resampled = []

    #_________________________

    def setSIRParameters(self, t_resampler = StochasticUniversalResampler(), t_observationVarianceInflation = 50.0, t_resamplingThreshold = 0.3):
        # set resampler
        self.m_resampler                    = t_resampler
        # inflation of the variance of the observation pdf
        self.m_observationVarianceInflation = t_observationVarianceInflation
        # resampling threshold (for Neff)
        self.m_resamplingThreshold          = t_resamplingThreshold

    #_________________________

    def initialise(self, t_x):
        # particles / samples
        self.m_x  = t_x
        # number of particles / samples
        self.m_Ns = t_x.shape[0]
        # relative weights in ln scale
        self.m_w  = - np.ones(self.m_Ns) * np.log(self.m_Ns)

    #_________________________

    def Neff(self):
        # empirical effective relative sample size
        # Neff = 1 / sum ( w_i ^ 2 ) / Ns
        return 1.0 / ( np.exp(2.0*self.m_w).sum() * self.m_Ns )

    #_________________________

    def resampledSteps(self):
        return 1.0 * np.array(self.m_resampled)

    #_________________________

    def reweight(self, t_nt, t_obs):
        # first step of analyse : reweight ensemble according to observation weights
        self.m_w += self.m_observationOperator.observationPDF(t_obs, self.m_x, t_nt*self.m_integrator.m_dt, self.m_observationVarianceInflation)

    #_________________________

    def normaliseWeights(self):
        # second step of analyse : normalise weigths so that they sum up to 1
        # note that wmax is extracted so that there is no zero argument for np.log() in the next line
        wmax      = self.m_w.max() 
        self.m_w -= wmax + np.log ( np.exp ( self.m_w - wmax ) . sum () )

    #_________________________

    def resample(self, t_nt):
        # third step of analyse : resample
        # note that here we only resample if Neff < resamplingThreshold
        if self.Neff() < self.m_resamplingThreshold:
            #-----------------------------------
            # print('resampling, nt='+str(t_nt))
            #-----------------------------------
            (self.m_w, self.m_x) = self.m_resampler.resample(self.m_w, self.m_x)
            # keep record of resampled steps...
            self.m_resampled.append(t_nt)

    #_________________________

    def analyse(self, t_nt, t_obs):
        # analyse observation at time nt
        self.reweight(t_nt, t_obs)
        self.normaliseWeights()
        self.resample(t_nt)
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
        return np.average(self.m_x, axis = 0, weights = np.exp(self.m_w))

#__________________________________________________

