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

from filters.abstractensemblefilter import AbstractEnsembleFilter
from filters.pf.weights             import Weights

#__________________________________________________

class SIRPF(AbstractEnsembleFilter):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_resampler):
        AbstractEnsembleFilter.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields)
        self.setSIRPFParameters(t_resampler)
        self.setSIRPFTemporaryArrays()

    #_________________________

    def setSIRPFParameters(self, t_resampler):
        # resampler
        self.m_resampler = t_resampler
        # Weights
        self.m_weights   = Weights(self.m_Ns)

    #_________________________

    def setSIRPFTemporaryArrays(self):
        # allocate temporary arrays
        self.m_Hx = np.zeros((self.m_Ns, self.m_observationOperator.m_spaceDimension))

    #_________________________

    def initialise(self):
        AbstractEnsembleFilter.initialise(self)

    #_________________________

    def analyse(self, t_t, t_observation):
        # apply observation operator to ensemble
        self.m_observationOperator.deterministicObserve(self.m_x[self.m_integrationIndex], t_t, self.m_Hx)
        # reweight according to observation weights
        self.m_weights.re_weight(self.m_observationOperator.pdf(t_observation, self.m_Hx, t_t))
        # normalise weights and also update normal weights
        self.m_weights.normalise()
        # resample
        self.m_resampler.resample(self.m_weights, self.m_x[self.m_integrationIndex])

    #_________________________

    def estimate(self):
        # mean of x
        self.m_estimation = np.average(self.m_x[self.m_integrationIndex], axis = -2, weights = self.m_weights.m_w)

    #_________________________

    def record(self, t_forecastOrAnalyse):
        AbstractEnsembleFilter.record(self, t_forecastOrAnalyse)
        self.m_resampler.record(t_forecastOrAnalyse, self.m_output, self.m_label)
        self.m_weights.record(t_forecastOrAnalyse, self.m_output, self.m_label)

#__________________________________________________

