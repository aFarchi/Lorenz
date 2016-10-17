#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# amsir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/29
#__________________________________________________
#
# class to handle an AM SIR particle filter
#

import numpy as np

from msir import MSIRPF

#__________________________________________________

class AMSIRPF(MSIRPF):

    #_________________________

    def __init__(self, t_integrator, t_observationOperator, t_Ns, t_resampler, t_resamplingTrigger, t_iSampler):
        MSIRPF.__init__(self, t_integrator, t_observationOperator, t_Ns, t_resampler, t_resamplingTrigger, t_iSampler)

    #_________________________

    def forecast(self, t_tStart, t_tEnd, t_observation):
        # integrate particles from tStart to tEnd

        # number of integration sub-steps
        iEnd = self.m_integrator.indexTEnd(t_tStart, t_tEnd)
        if iEnd == 0:
            return # if no integration, then just return

        # first stage
        self.m_integrator.deterministicIntegrate(self.m_x, t_tStart, t_tEnd, self.m_dx) # note that one could also use integrate() instead of deterministicIntegrate()
        fsx = np.copy(self.m_x[iEnd]) # copy is necessary since self.m_x[iEnd] will be rewritten when forecasting [see l. 53]
        fsw = self.m_observationOperator.pdf(t_observation, self.m_x[iEnd], t_tEnd)

        # sample x[tEnd] according to sum ( w[i] * fsw[i] * p ( x[tEnd] | x[tStart, i] )
        # this is equivalent to resampling at time tStart, isn't it ???
        indices     = self.m_indicesSampler.sampleIndices(self.m_Ns, self.m_w+fsw)
        self.m_x[0] = self.m_x[0][indices]
        self.m_integrator.integrate(self.m_x, t_tStart, t_tEnd, self.m_dx)

        # correct weights to account for the proposal
        wp = np.zeros(self.m_Ns)

        for ns in np.arange(self.m_Ns):
            # transition probability self.m_x[t-1, j] -> self.m_x[t, ns]
            me     = self.m_x[iEnd, ns] - fsx
            tp     = np.exp(self.m_integrator.errorPDF(me, t_tStart, t_tEnd)) # note: this line only works for BasicStochasticIntegrator instances
            wp[ns] = ( np.exp(self.m_w) * tp ).sum() / ( np.exp(self.m_w) * np.exp(fsw) * tp ).sum()

        self.m_w = np.log(wp)

#__________________________________________________

