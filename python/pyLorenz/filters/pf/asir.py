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

from sir import SIRPF

#__________________________________________________

class ASIRPF(SIRPF):

    #_________________________

    def __init__(self, t_label, t_integrator, t_observationOperator, t_Ns, t_resampler, t_resamplingTrigger):
        SIRPF.__init__(self, t_label, t_integrator, t_observationOperator, t_Ns, t_resampler, t_resamplingTrigger)

    #_________________________

    def forecast(self, t_tStart, t_tEnd, t_observation):
        # integrate particles from tStart to tEnd

        # number of integration sub-steps
        iEnd = self.m_integrator.indexTEnd(t_tStart, t_tEnd)
        if iEnd == 0:
            return # if no integration, then just return

        # first stage
        self.m_integrator.deterministicIntegrate(self.m_x, t_tStart, t_tEnd, self.m_dx) # note that one could also use integrate() instead of deterministicIntegrate()
        self.m_observationOperator.deterministicObserve(self.m_x[iEnd], t_tEnd, self.m_Hx)
        fsw = self.m_observationOperator.pdf(t_observation, self.m_Hx, t_tEnd)

        # resample at time tStart given these weights
        # note that we do not need the weights to be normalized since the resampler make sure of it
        indices     = self.m_resampler.sampleIndices(self.m_Ns, self.m_w+fsw)
        self.m_x[0] = self.m_x[0][indices]

        # now integrate particles from tStart to tEnd
        self.m_integrator.integrate(self.m_x, t_tStart, t_tEnd, self.m_dx)

        # correct weigths to account for the resampling step at time tStart
        # i.e. w[i] = 1 / p ( observation | fsx[indices[i]] )
        self.m_w = - fsw[indices]

#__________________________________________________

