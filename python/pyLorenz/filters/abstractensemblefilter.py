#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/
# abstractensemblefilter.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/5
#__________________________________________________
#
# abstract class to handle an ensemble filtering process
#

import numpy as np

#__________________________________________________

class AbstractEnsembleFilter(object):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields):
        self.setAbstractEnsembleFilterParameters(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields)

    #_________________________

    def setAbstractEnsembleFilterParameters(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields):
        # initialiser
        self.m_initialiser         = t_initialiser
        # integrator
        self.m_integrator          = t_integrator
        # observation operator
        self.m_observationOperator = t_observationOperator
        # observation times
        self.m_observationTimes    = t_observationTimes
        # output
        self.m_output              = t_output
        # filter label
        self.m_label               = t_label
        # number of samples
        self.m_Ns                  = t_Ns
        # output fields
        self.m_outputFields        = t_outputFields
        # space dimension
        self.m_spaceDimension      = t_integrator.m_spaceDimension
        # index of current state
        self.m_integrationIndex    = 0

    #_________________________

    def temporaryRecordShape(self, t_nRecord, t_field):
        # shape for temporary array recording the given field
        if 'Ensemble' in t_field or 'ensemble' in t_field:
            return (t_nRecord, self.m_Ns, self.m_spaceDimension)
        else:
            return (t_nRecord,)

    #_________________________

    def initialise(self):
        # size for integration arrays
        (sizeX, sizeDX)   = self.m_integrator.maximumIntegrationSubStepsPerCycle(self.m_observationTimes.longestCycle())
        # alloc arrays
        self.m_x          = np.zeros((sizeX, self.m_Ns, self.m_spaceDimension))
        self.m_dx         = np.zeros((sizeDX, self.m_Ns, self.m_spaceDimension))
        self.m_estimation = np.zeros(self.m_spaceDimension)
        self.m_rmse       = 0.0
        # initialise samples
        self.m_initialiser.initialise(self.m_x[self.m_integrationIndex])
        # initialise output
        self.m_output.initialiseFilterOutput(self.m_label, self.m_outputFields, self.temporaryRecordShape)

    #_________________________

    def forecast(self, t_tStart, t_tEnd, t_observation):
        # integrate particles from tStart to tEnd
        self.m_integrationIndex = self.m_integrator.integrate(self.m_x, t_tStart, t_tEnd, self.m_dx)

    #_________________________

    def analyse(self, t_t, t_observation):
        # analyse observation at time t
        raise NotImplementedError

    #_________________________

    def rms(self, t_a):
        # conveniance function to compute root mean square along the last axis
        return np.sqrt( ( t_a * t_a ) . mean(axis = -1) )

    #_________________________

    def computePerformance(self, t_xt):
        # estimate the mean
        self.estimate()
        # compare to the truth
        self.m_rmse = self.rms(self.m_estimation-t_xt)

    #_________________________

    def computeForecastPerformance(self, t_xt):
        # compute forecast performance
        self.computePerformance(t_xt)

    #_________________________

    def computeAnalysePerformance(self, t_xt):
        # compute analyse performance
        self.computePerformance(t_xt)

    #_________________________

    def permute(self):
        # permute self.m_x array to prepare next cycle
        self.m_x[0]             = self.m_x[self.m_integrationIndex]
        self.m_integrationIndex = 0

    #_________________________

    def estimate(self):
        # estimation
        raise NotImplementedError

    #_________________________

    def record(self, t_forecastOrAnalyse):
        # estimation
        self.m_output.record(self.m_label, t_forecastOrAnalyse+'_estimation', self.m_estimation)
        # rmse
        self.m_output.record(self.m_label, t_forecastOrAnalyse+'_rmse', self.m_rmse)
        # ensemble
        self.m_output.record(self.m_label, t_forecastOrAnalyse+'_ensemble', self.m_x[self.m_integrationIndex])

    #_________________________

    def recordForecast(self):
        # record forecast
        self.record('forecast')

    #_________________________

    def recordAnalyse(self):
        # record analyse
        self.record('analyse')

    #_________________________

    def finalise(self):
        # end simulation
        pass

#__________________________________________________

