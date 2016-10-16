#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/
# abstractensemblefilter.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/15
#__________________________________________________
#
# abstract class to handle an ensemble filtering process
#

import numpy as np

#__________________________________________________

class AbstractEnsembleFilter(object):

    #_________________________

    def __init__(self, t_integrator, t_observationOperator, t_Ns):
        self.setAbstractEnsembleFilterParameters(t_integrator, t_observationOperator, t_Ns)

    #_________________________

    def setAbstractEnsembleFilterParameters(self, t_integrator, t_observationOperator, t_Ns):
        # integrator
        self.m_integrator          = t_integrator
        # observation operator
        self.m_observationOperator = t_observationOperator
        # number of samples
        self.m_Ns                  = t_Ns
        # space dimension
        self.m_spaceDimension      = t_integrator.m_spaceDimension

    #_________________________

    def initialise(self, t_initialiser, t_Nt, t_sizeX, t_sizeDX):
        # initialise samples
        self.m_x           = t_initialiser.initialiseSamples((t_sizeX, self.m_Ns, self.m_spaceDimension))
        self.m_dx          = np.zeros((t_sizeDX, self.m_Ns, self.m_spaceDimension))
        #--------------------------------------------------------------
        # Array for estimation (if there is enough memory to afford it)
        #--------------------------------------------------------------
        self.m_rmseA       = np.zeros(t_Nt)
        self.m_rmseF       = np.zeros(t_Nt)
        self.m_estimate    = np.zeros((t_Nt, self.m_spaceDimension))

    #_________________________

    def forecast(self, t_tStart, t_tEnd, t_observation):
        # integrate particles from tStart to tEnd
        self.m_integrator.integrate(self.m_x, t_tStart, t_tEnd, self.m_dx)

    #_________________________

    def analyse(self, t_index, t_t, t_observation):
        # analyse observation at time t
        raise NotImplementedError

    #_________________________

    def rms(self, t_a):
        # conveniance function to compute root mean square along the last axis
        return np.sqrt( ( t_a * t_a ) . mean(axis = -1) )

    #_________________________

    def computeForecastPerformance(self, t_xt, t_iEnd, t_index):
        # compute performance of the forecast from tStart to tEnd

        # estimate the mean
        estimate              = self.estimate(t_iEnd)
        # compare to the truth
        self.m_rmseF[t_index] = self.rms( estimate - t_xt )

    #_________________________

    def computeAnalysePerformance(self, t_xt, t_iEnd, t_index):
        # record estimation and compute performance between tStart (excluded) and tEnd (included)

        # estimate the mean
        self.m_estimate[t_index] = self.estimate(t_iEnd)
        # compare to the truth
        self.m_rmseA[t_index]    = self.rms( self.m_estimate[t_index] - t_xt )

    #_________________________

    def permute(self, t_index):
        # permute self.m_x array to prepare next cycle
        self.m_x[0] = self.m_x[t_index]

    #_________________________

    def estimate(self, t_index):
        # estimation
        raise NotImplementedError

    #_________________________

    def recordToFile(self, t_outputDir, t_filterPrefix):
        self.m_rmseA.tofile(t_outputDir+t_filterPrefix+'_rmseA.bin')
        self.m_rmseF.tofile(t_outputDir+t_filterPrefix+'_rmseF.bin')
        self.m_estimate.tofile(t_outputDir+t_filterPrefix+'_estimation.bin')

#__________________________________________________

