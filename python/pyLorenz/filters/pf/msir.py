#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# msir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/17
#__________________________________________________
#
# class to handle a marginal SIR particle filter
#

import numpy as np

from sir import SIRPF

#__________________________________________________

class MSIRPF(SIRPF):

    #_________________________

    def __init__(self, t_integrator, t_observationOperator, t_Ns, t_resampler, t_resamplingTrigger, t_iSampler):
        SIRPF.__init__(self, t_integrator, t_observationOperator, t_Ns, t_resampler, t_resamplingTrigger)
        self.setMSIRPFParameters(t_iSampler)

    #_________________________

    def setMSIRPFParameters(self, t_iSampler):
        # indices sampler
        self.m_indicesSampler = t_iSampler

    #_________________________

    def forecast(self, t_tStart, t_tEnd, t_observation):
        # integrate particles from tStart to tEnd

        # number of integration sub-steps
        iEnd = self.m_integrator.indexTEnd(t_tStart, t_tEnd)
        if iEnd == 0:
            return # if no integration, then just return

        # sample x[tEnd] according to sum ( w[i] * p ( x[tEnd] | x[tStart-1, i] )
        # this is equivalent to resampling at time tStart, isn't it ???
        indices     = self.m_indicesSampler.sampleIndices(self.m_Ns, self.m_w)
        self.m_x[0] = self.m_x[0][indices]
        self.m_integrator.integrate(self.m_x, t_tStart, t_tEnd, self.m_dx)

        # correct weights to account for the proposal
        self.m_w    = np.ones(self.m_Ns)

#__________________________________________________

