#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# oisir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/17
#__________________________________________________
#
# class to handle a SIR particle filter that uses the optimal importance function as proposal
#

import numpy as np
import numpy.random as rnd

from sir import SIRPF

#__________________________________________________

class OISIRPF_diag(SIRPF):

    #_________________________

    def __init__(self, t_label, t_integrator, t_observationOperator, t_Ns, t_resampler, t_resamplingTrigger):
        SIRPF.__init__(self, t_label, t_integrator, t_observationOperator, t_Ns, t_resampler, t_resamplingTrigger)

    #_________________________

    def forecast(self, t_tStart, t_tEnd, t_observation):
        # integrate particles from tStart to tEnd, given the observation at tEnd

        # number of integration sub-steps
        iEnd = self.m_integrator.indexTEnd(t_tStart, t_tEnd)
        if iEnd == 0:
            return # if no integration, then just return

        # auxiliary variables
        sigma_m = self.m_integrator.errorCovarianceMatrix_diag(t_tStart, t_tEnd) # note: this line only works for BasicStochasticIntegrator instances
        sigma_o = self.m_observationOperator.errorCovarianceMatrix_diag(t_tEnd, self.m_spaceDimension)
        self.m_integrator.deterministicIntegrate(self.m_x, t_tStart, t_tEnd, self.m_dx)
        fx = np.copy(self.m_x[iEnd]) # copy is necessary since self.m_x[iEnd] will be rewritten when forecasting [see l. 62 and 71]
        H  = self.m_observationOperator.differential_diag(fx, t_tEnd)
        y  = self.m_observationOperator.castObservationToStateSpace(t_observation, t_tEnd, self.m_spaceDimension)

        if not self.m_observationOperator.isLinear(): # non linear correction
            y = H * fx - self.m_observationOperator.deterministicObserve(fx, t_tEnd) + y

        # proposal
        sigma_p = 1.0 / ( 1.0 / sigma_m + H * ( 1.0 / sigma_o ) * H )
        mean_p  = sigma_p * ( ( 1.0 / sigma_m ) * fx + H * ( 1.0 / sigma_o ) * y )

        # draw x at tEnd from proposal
        self.m_x[iEnd] = mean_p + np.sqrt(sigma_p) * rnd.standard_normal(self.m_x[iEnd].shape)

        # reweight ensemble to account for proposal

        if self.m_observationOperator.isLinear():
            # w *= p ( observation | x[tStart] )
            s         = 1.0 / ( sigma_o + H * sigma_m * H )
            d         = y - H * fx
            self.m_w -= 0.5 * ( d * s * d ).sum(axis = -1)

            # w /= p( observation | x[tEnd] )
            self.m_observationOperator.deterministicObserve(self.m_x[iEnd], t_tEnd, self.m_Hx)
            self.m_w -= self.m_observationOperator.pdf(t_observation, self.m_Hx, t_tEnd)

        else:
            # w *= p( x[tEnd] | x[tStart] )
            # small tweak here since we already performed a deterministic integration of x[tStart] (and stored it in xf)
            me        = self.m_x[iEnd] - fx
            self.m_w -= 0.5 * ( me * sigma_m * me ).sum(axis = -1)

            # w /= proposal_pdf( x[tEnd] )
            pe        = self.m_x[iEnd] - mean_p
            self.m_w += 0.5 * ( pe * sigma_p * pe ).sum(axis = -1)

        # Other option:
        # remove the step : w /= p( observation | x[tEnd] ) in the linear case
        # add the step    : w *= p( observation | x[tEnd] ) in the non-linear case
        # and only perform reweight if no forecast was performed yet (i.e. if one wants to perform an analyse at t=0)
        #
        # But then one would lose information about forecast performance

#__________________________________________________

class OISIRPF(SIRPF):

    #___________________________
    # TODO: complete this filter
    #___________________________

    #_________________________

    def __init__(self, t_integrator, t_observationOperator, t_Ns, t_resampler, t_resamplingTrigger):
        SIRPF.__init__(self, t_integrator, t_observationOperator, t_Ns, t_resampler, t_resamplingTrigger)

    #_________________________

    def forecast(self, t_tStart, t_tEnd, t_observation):
        # integrate particles from tStart to tEnd, given the observation at tEnd

        # number of integration sub-steps
        iEnd = self.m_integrator.indexTEnd(t_tStart, t_tEnd)
        if iEnd == 0:
            return # if no integration, then just return

        # auxiliary variables
        sigma_m = self.m_integrator.errorCovarianceMatrix(t_tStart, t_tEnd) # note: this line only works for BasicStochasticIntegrator instances
        sigma_o = self.m_observationOperator.errorCovarianceMatrix(t_tEnd, self.m_spaceDimension)
        self.m_integrator.deterministicIntegrate(self.m_x, t_tStart, t_tEnd, self.m_dx)
        fx = np.copy(self.m_x[iEnd]) # copy is necessary since self.m_x[iEnd] will be rewritten when forecasting [see l. 62 and 71]
        H  = self.m_observationOperator.differential(fx, t_tEnd)

        if self.m_observationOperator.isLinear():

            # proposal
            sigma_p = 1.0 / ( 1.0 / sigma_m + H * ( 1.0 / sigma_o ) * H )
            mean_p  = sigma_p * ( ( 1.0 / sigma_m ) * fx + H * ( 1.0 / sigma_o ) * y )

        else: # non linear correction
            y = ( H * fx[:, np.newaxis, :] ).sum(axis = -1) - self.m_observationOperator.deterministicObserve(fx, t_tEnd) + t_observation

        # proposal
        sigma_p = 1.0 / ( 1.0 / sigma_m + H * ( 1.0 / sigma_o ) * H )
        mean_p  = sigma_p * ( ( 1.0 / sigma_m ) * fx + H * ( 1.0 / sigma_o ) * y )

        # draw x at tEnd from proposal
        self.m_x[iEnd] = mean_p + np.sqrt(sigma_p) * rnd.standard_normal(self.m_x[iEnd].shape)

        # reweight ensemble to account for proposal

        if self.m_observationOperator.isLinear():
            # w *= p ( observation | x[tStart] )
            s         = 1.0 / ( sigma_o + H * sigma_m * H )
            d         = y - H * fx
            self.m_w -= ( d * s * d ).sum(axis = -1) / 2.0

            # w /= p( observation | x[tEnd] )
            self.m_w -= self.m_observationOperator.pdf(t_observation, self.m_x[iEnd], t_tEnd)

        else:
            # w *= p( x[tEnd] | x[tStart] )
            # small tweak here since we already performed a deterministic integration of x[tStart] (and stored it in xf)
            me        = self.m_x[iEnd] - fx
            self.m_w -= ( me * sigma_m * me ).sum(axis = -1) / 2.0

            # w /= proposal_pdf( x[tEnd] )
            pe        = self.m_x[iEnd] - mean_p
            self.m_w += ( pe * sigma_p * pe ).sum(axis = -1) / 2.0

        # Other option:
        # remove the step : w /= p( observation | x[tEnd] ) in the linear case
        # add the step    : w *= p( observation | x[tEnd] ) in the non-linear case
        # and only perform reweight if no forecast was performed yet (i.e. if one wants to perform an analyse at t=0)
        #
        # But then one would lose information about forecast performance

#__________________________________________________

