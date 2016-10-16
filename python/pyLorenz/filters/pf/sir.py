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

    def __init__(self, t_integrator, t_observationOperator, t_Ns, t_resampler, t_resamplingTrigger):
        AbstractEnsembleFilter.__init__(self, t_integrator, t_observationOperator, t_Ns)
        self.setSIRPFParameters(t_resampler, t_resamplingTrigger)
        self.m_resampled = []

    #_________________________

    def setSIRPFParameters(self, t_resampler, t_resamplingTrigger):
        # resampler
        self.m_resampler         = t_resampler
        # resampling trigger
        self.m_resamplingTrigger = t_resamplingTrigger

    #_________________________

    def initialise(self, t_initialiser, t_Nt, t_sizeX, t_sizeDX):
        AbstractEnsembleFilter.initialise(self, t_initialiser, t_Nt, t_sizeX, t_sizeDX)

        # relative weights in ln scale
        self.m_w = - np.log(self.m_Ns) * np.ones(self.m_Ns)

        #--------------------------------------------------------------
        # Array for estimation (if there is enough memory to afford it)
        #--------------------------------------------------------------
        self.m_NeffF = np.ones(t_Nt)
        self.m_NeffA = np.ones(t_Nt)

    #_________________________

    def Neff(self):
        # empirical effective relative sample size
        # Neff = 1 / sum ( w_i ^ 2 ) / Ns
        return 1.0 / ( np.exp(2.0*self.m_w).sum() * self.m_Ns )

    #_________________________

    def resampledTimes(self):
        #-------------------
        # TODO: improve this
        #-------------------
        return np.array(self.m_resampled)

    #_________________________

    def reweight(self, t_index, t_t, t_observation):
        # first step of analyse : reweight ensemble according to observation weights
        self.m_w += self.m_observationOperator.pdf(t_observation, self.m_x[t_index], t_t)

    #_________________________

    def normaliseWeights(self):
        # second step of analyse : normalise weigths so that they sum up to 1
        # note that wmax is extracted so that there is no zero argument for np.log() in the next line
        wmax      = self.m_w.max() 
        self.m_w -= wmax + np.log ( np.exp ( self.m_w - wmax ) . sum () )

    #_________________________

    def resample(self, t_index, t_t):
        # third step of analyse : resample
        if self.m_resamplingTrigger.trigger(self.Neff(), t_t):
            #-----------------------------------
            # print('resampling, t = '+str(t_t))
            #-----------------------------------
            (self.m_w, self.m_x[t_index]) = self.m_resampler.sample(self.m_Ns, self.m_w, self.m_x[t_index])
            # keep record of resampled times
            self.m_resampled.append(t_t)

    #_________________________

    def analyse(self, t_index, t_t, t_observation):
        # analyse observation at time t
        self.reweight(t_index, t_t, t_observation)
        self.normaliseWeights()
        self.resample(t_index, t_t)

    #_________________________

    def computeForecastPerformance(self, t_xt, t_iEnd, t_index):
        # performance of the forecast
        AbstractEnsembleFilter.computeForecastPerformance(self, t_xt, t_iEnd, t_index)
        self.m_NeffF[t_index] = self.Neff()

    #_________________________

    def computeAnalysePerformance(self, t_xt, t_iEnd, t_index):
        # performance of the analyse
        AbstractEnsembleFilter.computeAnalysePerformance(self, t_xt, t_iEnd, t_index)
        self.m_NeffA[t_index] = self.Neff()

    #_________________________

    def estimate(self, t_index):
        # mean of x
        return np.average(self.m_x[t_index], axis = -2, weights = np.exp(self.m_w))

    #_________________________

    def recordToFile(self, t_outputDir, t_filterPrefix):
        AbstractEnsembleFilter.recordToFile(self, t_outputDir, t_filterPrefix)
        self.m_NeffF.tofile(t_outputDir+t_filterPrefix+'_NeffF.bin')
        self.m_NeffA.tofile(t_outputDir+t_filterPrefix+'_NeffA.bin')
        self.resampledTimes().tofile(t_outputDir+t_filterPrefix+'_resampledTimes.bin')

#__________________________________________________

