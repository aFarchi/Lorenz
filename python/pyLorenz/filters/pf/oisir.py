#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# oisir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# class to handle a SIR particle filter that uses the optimal importance function as proposal
#
# TODO : FINISH THIS FILTER !!!!!
#

import numpy as np
import numpy.random as rnd

from sir                                          import SIRPF
from ...utils.integration.rk4integrator           import DeterministicRK4Integrator
from ...observations.iobservations                import StochasticIObservations
from ...utils.random.stochasticuniversalresampler import StochasticUniversalResampler
from ...utils.trigger.thresholdtrigger            import ThresholdTrigger

#__________________________________________________

class OISIRPF(SIRPF):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_resampler = StochasticUniversalResampler(), 
            t_observationVarianceInflation = 1.0, t_resamplingTrigger = ThresholdTrigger(0.3), t_Ns = 10):
        SIRPF.__init__(self, t_integrator, t_obsOp, t_resampler, t_observationVarianceInflation, t_resamplingTrigger, t_Ns)

    #_________________________

    def forecast(self, t_ntStart, t_ntEnd, t_observation):
        # integrate particles from ntStart to ntEnd, given the observation at ntEnd
        # this includes analyse step for conveniance

        for nt in t_ntStart + np.arange(t_ntEnd-t_ntStart-1):
            self.m_x              = self.m_integrator.process(self.m_x, nt)
            self.m_estimate[nt+1] = self.estimate()

        self.m_neff[t_ntStart+1:t_ntEnd+1] = self.Neff()

        # for the last integration, we use the optimal importance proposal

        # auxiliary variables
        sigma_m     = self.m_integrator.errorCovarianceMatrix(t_ntEnd-1)
        sigma_o     = self.m_observationOperator.errorCovarianceMatrix(t_ntEnd-1)
        sigma_m_inv = np.linalg.inv(sigma_m)
        sigma_o_inv = np.linalg.inv(sigma_o)
        fx          = self.m_integrator.deterministicProcess(self.m_x, t_ntEnd-1)
        H           = self.m_observationOperator.differential(self.m_x, t_ntEnd)

        #---------------------------- 
        # TODO : write this filter...
        # (like in the diag case...)
        #---------------------------- 

        # proposal
        sigma_p     = np.linalg.inv( sigma_m_inv + np.dot( np.transpose(H) , np.dot( sigma_o_inv , H ) ) )

            # S     = np.linalg.inv( sigma_m + np.dot( np.transpose(H) , np.dot( sigma_o , H ) ) )
            #------------------------------------- 
            S     = np.linalg.inv( sigma_o + np.dot( H , np.dot( sigma_m , np.transpose(H) ) ) )

            for ns in np.arange(self.m_Ns):
                #--------------------------------------- 
                # COULD BE VECTORIZED WITH TENSORDOT ???
                #--------------------------------------- 
                mean = np.dot( sigma , np.dot ( sigma_m_inv , fx[ns] ) + np.dot ( np.transpose(H) , np.dot ( sigma_o_inv , t_observation ) ) )

                # sample from N(mean, sigma)
                self.m_x[ns] = rnd.multivariate_normal(mean, sigma)
                # ------------------------------------------------------------------------------
                # Note that python seem to be bad at sampling from multivariate_normal
                # e.g. if sigma is diagonal, a better behavior is observed when sampling using :
                # self.m_x[ns] = rnd.normal(mean, np.diag(sigma))
                # although there is no theoretical difference
                # ------------------------------------------------------------------------------

                # correct weight
                ino = t_observation - np.dot( H , fx[ns] )
                self.m_w[ns] += - np.dot( ino , np.dot( S , ino ) ) / 2.0 # p(obs|x[nt-1])

    #_________________________

    def reweight(self, t_nt, t_observation):
        if t_nt == 0:
            SIRPF.reweight(self, t_nt, t_observation)

#__________________________________________________

