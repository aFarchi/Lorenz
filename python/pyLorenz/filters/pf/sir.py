#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# sir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/6
#__________________________________________________
#
# class to handle a SIR particle filter
#

import numpy as np

from ..abstractfilter                             import AbstractFilter
from ...utils.integration.rk4integrator           import DeterministicRK4Integrator
from ...observations.iobservations                import StochasticIObservations
from ...utils.random.stochasticuniversalresampler import StochasticUniversalResampler
from ...utils.trigger.thresholdtrigger            import ThresholdTrigger

#__________________________________________________

class SIRPF(AbstractFilter):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_resampler = StochasticUniversalResampler(), 
            t_observationVarianceInflation = 1.0, t_resamplingTrigger = ThresholdTrigger(0.3), t_Ns = 10):
        AbstractFilter.__init__(self, t_integrator, t_obsOp)
        self.setSIRParameters(t_resampler, t_observationVarianceInflation, t_resamplingTrigger, t_Ns)
        self.m_resampled = []

    #_________________________

    def setSIRParameters(self, t_resampler = StochasticUniversalResampler(), t_observationVarianceInflation = 1.0, t_resamplingTrigger = ThresholdTrigger(0.3), t_Ns = 10):
        # number of particles
        self.m_Ns                           = t_Ns
        # resampler
        self.m_resampler                    = t_resampler
        # inflation of the variance of the observation pdf
        self.m_observationVarianceInflation = t_observationVarianceInflation
        # resampling trigger
        self.m_resamplingTrigger            = t_resamplingTrigger
        # space dimension
        self.m_spaceDimension               = self.m_integrator.m_spaceDimension

    #_________________________

    def initialise(self, t_initialiser, t_Nt):
        # particles / samples
        self.m_x        = t_initialiser.initialiseSamples(self.m_Ns)
        # relative weights in ln scale
        self.m_w        = - np.ones(self.m_Ns) * np.log(self.m_Ns)

        # estimations
        self.m_estimate = np.zeros((t_Nt, self.m_spaceDimension))
        self.m_neff     = np.ones(t_Nt)

        # fill first guess
        self.m_estimate[0, :] = self.estimate()
        self.m_neff[0]        = self.Neff()

    #_________________________

    def Neff(self):
        # empirical effective relative sample size
        # Neff = 1 / sum ( w_i ^ 2 ) / Ns
        return 1.0 / ( np.exp(2.0*self.m_w).sum() * self.m_Ns )

    #_________________________

    def resampledSteps(self):
        #-------------------
        # TODO: improve this
        #-------------------
        return 1.0 * np.array(self.m_resampled)

    #_________________________

    def reweight(self, t_nt, t_observation):
        # first step of analyse : reweight ensemble according to observation weights
        self.m_w += self.m_observationOperator.pdf(t_observation, self.m_x, t_nt, self.m_observationVarianceInflation)

    #_________________________

    def normaliseWeights(self):
        # second step of analyse : normalise weigths so that they sum up to 1
        # note that wmax is extracted so that there is no zero argument for np.log() in the next line
        wmax      = self.m_w.max() 
        self.m_w -= wmax + np.log ( np.exp ( self.m_w - wmax ) . sum () )

    #_________________________

    def resample(self, t_nt):
        # third step of analyse : resample
        if self.m_resamplingTrigger.trigger(self.Neff(), t_nt):
            #-----------------------------------
            # print('resampling, nt='+str(t_nt))
            #-----------------------------------
            (self.m_w, self.m_x) = self.m_resampler.sample(self.m_Ns, self.m_w, self.m_x)
            # keep record of resampled step
            self.m_resampled.append(t_nt)

        self.m_neff[t_nt] = self.Neff()

    #_________________________

    def analyse(self, t_nt, t_obs):
        # analyse observation at time nt
        self.reweight(t_nt, t_obs)
        self.normaliseWeights()
        self.resample(t_nt) # and write Neff()
        self.m_estimate[t_nt :] = self.estimate()

    #_________________________

    def forecast(self, t_ntStart, t_ntEnd, t_observation):
        # integrate particles from ntStart to ntEnd, given the observation at ntEnd
        for nt in t_ntStart + np.arange(t_ntEnd-t_ntStart):
            self.m_x                 = self.m_integrator.process(self.m_x, nt)
            self.m_estimate[nt+1, :] = self.estimate()

        self.m_neff[t_ntStart+1:t_ntEnd+1] = self.Neff()
        # note : the values of self.m_estimate[t_ntEnd] and self.m_neff[t_ntEnd] do not matter since it will be replaced after analyse

    #_________________________

    def estimate(self):
        # mean of x
        return np.average(self.m_x, axis = 0, weights = np.exp(self.m_w))

    #_________________________

    def recordToFile(self, t_outputDir = './', t_filterPrefix = 'sir'):
        self.m_estimate.tofile(t_outputDir+t_filterPrefix+'_estimation.bin')
        self.m_neff.tofile(t_outputDir+t_filterPrefix+'_neff.bin')
        self.resampledSteps().tofile(t_outputDir+t_filterPrefix+'_resampled.bin')

#__________________________________________________

