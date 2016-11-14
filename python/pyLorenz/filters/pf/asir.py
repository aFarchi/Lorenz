#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# asir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/17
#__________________________________________________
#
# class to handle an ASIR particle filter
#

import numpy as np

from filters.pf.sir import SIRPF

#__________________________________________________

class ASIRPF(SIRPF):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields,
            t_resampler, t_resamplingTrigger):
        SIRPF.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields,
                t_resampler, t_resamplingTrigger)

    #_________________________

    def forecast(self, t_tStart, t_tEnd, t_observation):
        # integrate particles from tStart to tEnd

        # first stage
        fsIndex = self.m_integrator.deterministicIntegrate(self.m_x, t_tStart, t_tEnd, self.m_dx) # note that one could also use integrate() instead of deterministicIntegrate()
        if fsIndex == 0:
            return # if no integration, then just return
        self.m_observationOperator.deterministicObserve(self.m_x[fsIndex], t_tEnd, self.m_Hx)
        fsw = self.m_observationOperator.pdf(t_observation, self.m_Hx, t_tEnd)

        # resample at time tStart given these weights
        rw          = self.normaliseWeights(self.m_w+fsw)
        indices     = self.m_resampler.sampleIndices(self.m_Ns, rw)
        self.m_x[0] = self.m_x[0][indices]

        # now integrate particles from tStart to tEnd
        self.m_integrationIndex = self.m_integrator.integrate(self.m_x, t_tStart, t_tEnd, self.m_dx)

        # correct weigths to account for the resampling step at time tStart
        # i.e. w[i] = 1 / p ( observation | fsx[indices[i]] )
        self.m_w = - fsw[indices]

#__________________________________________________

