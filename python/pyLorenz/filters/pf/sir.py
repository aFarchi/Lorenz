#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# sir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/15
#__________________________________________________
#
# class to handle a SIR particle filter
#

import numpy as np

from ..abstractensemblefilter import AbstractEnsembleFilter

#__________________________________________________

class SIRPF(AbstractEnsembleFilter):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields,
            t_resampler, t_resamplingTrigger):
        AbstractEnsembleFilter.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields)
        self.setSIRPFParameters(t_resampler, t_resamplingTrigger)
        self.setSIRPFTemporaryArrays()

    #_________________________

    def setSIRPFParameters(self, t_resampler, t_resamplingTrigger):
        # resampler
        self.m_resampler         = t_resampler
        # resampling trigger
        self.m_resamplingTrigger = t_resamplingTrigger

    #_________________________

    def setSIRPFTemporaryArrays(self):
        # allocate temporary arrays
        self.m_Hx = np.zeros((self.m_Ns, self.m_observationOperator.m_spaceDimension))

    #_________________________

    def initialise(self):
        AbstractEnsembleFilter.initialise(self)
        # relative weights in ln scale
        self.m_w         = - np.log(self.m_Ns) * np.ones(self.m_Ns)
        # Neff
        self.m_Neff      = 1.0
        # Resampled
        self.m_resampled = False

    #_________________________

    def Neff(self):
        # empirical effective relative sample size
        # Neff = 1 / sum ( w_i ^ 2 ) / Ns
        self.m_Neff = 1.0 / ( np.exp(2.0*self.m_w).sum() * self.m_Ns )

    #_________________________

    def reweight(self, t_t, t_observation):
        # first step of analyse : reweight ensemble according to observation weights
        self.m_observationOperator.deterministicObserve(self.m_x[self.m_integrationIndex], t_t, self.m_Hx)
        self.m_w += self.m_observationOperator.pdf(t_observation, self.m_Hx, t_t)

    #_________________________

    def normaliseWeights(self):
        # second step of analyse : normalise weigths so that they sum up to 1
        # note that wmax is extracted so that there is no zero argument for np.log() in the next line
        wmax      = self.m_w.max() 
        self.m_w -= wmax + np.log ( np.exp ( self.m_w - wmax ) . sum () )

    #_________________________

    def resample(self, t_t):
        # third step of analyse : resample
        self.Neff()
        self.m_resampled = self.m_resamplingTrigger.trigger(self.m_Neff, t_t)
        if self.m_resampled:
            (self.m_w, self.m_x[self.m_integrationIndex]) = self.m_resampler.sample(self.m_Ns, self.m_w, self.m_x[self.m_integrationIndex])
            self.Neff()

    #_________________________

    def analyse(self, t_t, t_observation):
        # analyse observation at time t
        self.reweight(t_t, t_observation)
        self.normaliseWeights()
        self.resample(t_t)

    #_________________________

    def estimate(self):
        # mean of x
        self.m_estimation = np.average(self.m_x[self.m_integrationIndex], axis = -2, weights = np.exp(self.m_w))

    #_________________________

    def record(self, t_forecastOrAnalyse):
        AbstractEnsembleFilter.record(self, t_forecastOrAnalyse)
        # neff
        self.m_output.record(self.m_label, t_forecastOrAnalyse+'_neff', self.m_Neff)

    #_________________________

    def recordAnalyse(self):
        AbstractEnsembleFilter.recordAnalyse(self)
        # resampling
        out = 0.0
        if self.m_resampled:
            out = 1.0
        self.m_output.record(self.m_label, 'analyse_resampled', out)

#__________________________________________________

