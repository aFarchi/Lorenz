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

    def __init__(self, t_label, t_integrator, t_observationOperator, t_Ns, t_resampler, t_resamplingTrigger):
        AbstractEnsembleFilter.__init__(self, t_label, t_integrator, t_observationOperator, t_Ns)
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

    def initialise(self, t_initialiser, t_sizeX, t_sizeDX):
        AbstractEnsembleFilter.initialise(self, t_initialiser, t_sizeX, t_sizeDX)
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

    def reweight(self, t_index, t_t, t_observation):
        # first step of analyse : reweight ensemble according to observation weights
        self.m_observationOperator.deterministicObserve(self.m_x[t_index], t_t, self.m_Hx)
        self.m_w += self.m_observationOperator.pdf(t_observation, self.m_Hx, t_t)

    #_________________________

    def normaliseWeights(self):
        # second step of analyse : normalise weigths so that they sum up to 1
        # note that wmax is extracted so that there is no zero argument for np.log() in the next line
        wmax      = self.m_w.max() 
        self.m_w -= wmax + np.log ( np.exp ( self.m_w - wmax ) . sum () )

    #_________________________

    def resample(self, t_index, t_t):
        # third step of analyse : resample
        self.Neff()
        self.m_resampled = self.m_resamplingTrigger.trigger(self.m_Neff, t_t)
        if self.m_resampled:
            (self.m_w, self.m_x[t_index]) = self.m_resampler.sample(self.m_Ns, self.m_w, self.m_x[t_index])
            self.Neff()

    #_________________________

    def analyse(self, t_index, t_t, t_observation):
        # analyse observation at time t
        self.reweight(t_index, t_t, t_observation)
        self.normaliseWeights()
        self.resample(t_index, t_t)

    #_________________________

    def estimate(self, t_index):
        # mean of x
        return np.average(self.m_x[t_index], axis = -2, weights = np.exp(self.m_w))

    #_________________________

    def writeForecast(self, t_output, t_iEnd):
        AbstractEnsembleFilter.writeForecast(self, t_output, t_iEnd)
        t_output.writeFilterForecastNeff(self.m_label, self.m_Neff)

    #_________________________

    def writeAnalyse(self, t_output, t_iEnd):
        AbstractEnsembleFilter.writeAnalyse(self, t_output, t_iEnd)
        t_output.writeFilterAnalyseNeff(self.m_label, self.m_Neff)
        t_output.writeFilterAnalyseResampled(self.m_label, self.m_resampled)

#__________________________________________________

