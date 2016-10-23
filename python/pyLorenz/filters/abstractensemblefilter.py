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

    def __init__(self, t_label, t_integrator, t_observationOperator, t_Ns):
        self.setAbstractEnsembleFilterParameters(t_label, t_integrator, t_observationOperator, t_Ns)

    #_________________________

    def setAbstractEnsembleFilterParameters(self, t_label, t_integrator, t_observationOperator, t_Ns):
        # filter label
        self.m_label               = t_label
        # integrator
        self.m_integrator          = t_integrator
        # observation operator
        self.m_observationOperator = t_observationOperator
        # number of samples
        self.m_Ns                  = t_Ns
        # space dimension
        self.m_spaceDimension      = t_integrator.m_spaceDimension

    #_________________________

    def initialise(self, t_initialiser, t_sizeX, t_sizeDX):
        # samples arrays
        self.m_x          = np.zeros((t_sizeX, self.m_Ns, self.m_spaceDimension))
        # temporary array
        self.m_dx         = np.zeros((t_sizeDX, self.m_Ns, self.m_spaceDimension))
        # estimate
        self.m_estimation = np.zeros(self.m_spaceDimension)
        # rmse
        self.m_rmse       = 0.0
        # initialise samples
        t_initialiser.initialiseSamples(self.m_x[0])

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

    def computePerformance(self, t_xt, t_iEnd):
        # compute performance

        # estimate the mean
        self.m_estimation = self.estimate(t_iEnd)
        # compare to the truth
        self.m_rmse       = self.rms(self.m_estimation-t_xt)

    #_________________________

    def computeForecastPerformance(self, t_xt, t_iEnd):
        self.computePerformance(t_xt, t_iEnd)

    #_________________________

    def computeAnalysePerformance(self, t_xt, t_iEnd):
        self.computePerformance(t_xt, t_iEnd)

    #_________________________

    def permute(self, t_index):
        # permute self.m_x array to prepare next cycle
        self.m_x[0] = self.m_x[t_index]

    #_________________________

    def estimate(self, t_index):
        # estimation
        raise NotImplementedError

    #_________________________

    def writeForecast(self, t_output, t_iEnd):
        # estimation
        t_output.writeFilterForecast(self.m_label, self.m_estimation)
        # rmse
        t_output.writeFilterForecastRMSE(self.m_label, self.m_rmse)
        # ensemble
        t_output.writeFilterForecastEnsemble(self.m_label, self.m_x[t_iEnd])

    #_________________________

    def writeAnalyse(self, t_output, t_iEnd):
        # estimation
        t_output.writeFilterAnalyse(self.m_label, self.m_estimation)
        # rmse
        t_output.writeFilterAnalyseRMSE(self.m_label, self.m_rmse)
        # ensemble
        t_output.writeFilterAnalyseEnsemble(self.m_label, self.m_x[t_iEnd])

#__________________________________________________

