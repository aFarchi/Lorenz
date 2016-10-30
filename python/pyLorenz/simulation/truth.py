#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# truth.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/30
#__________________________________________________
#
# class to handle the truth of a simulation
#

import numpy as np

#__________________________________________________

class Truth(object):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_truthOutputFields, t_observationsOutputFields):
        self.setTruthParameters(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_truthOutputFields, t_observationsOutputFields)

    #_________________________

    def setTruthParameters(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_truthOutputFields, t_observationsOutputFields):
        # initialiser
        self.m_initialiser              = t_initialiser
        # integrator
        self.m_integrator               = t_integrator
        # observation operator
        self.m_observationOperator      = t_observationOperator
        # observation times
        self.m_observationTimes         = t_observationTimes
        # output
        self.m_output                   = t_output
        # truth output fields
        self.m_truthOutputFields        = t_truthOutputFields
        # observations output fields
        self.m_observationsOutputFields = t_observationsOutputFields
        # x, y space dimension
        self.m_xSpaceDimension          = t_integrator.m_spaceDimension
        self.m_ySpaceDimension          = t_observationOperator.m_spaceDimension
        # index of current truth state
        self.m_integrationIndex         = 0

    #_________________________

    def temporaryRecordShape(self, t_nRecord, t_truthOrObservations, t_field):
        # shape for temporary array recording the given field
        if t_field == 'trajectory':
            if t_truthOrObservations == 'truth':
                return (t_nRecord, self.m_xSpaceDimension)
            elif t_truthOrObservations == 'observations':
                return (t_nRecord, self.m_ySpaceDimension)
        return (t_nRecord, 0)

    #_________________________

    def observation(self):
        # access function for observation array
        return self.m_y

    #_________________________

    def truth(self):
        # access function for truth array
        return self.m_x[self.m_integrationIndex]

    #_________________________

    def initialise(self):
        # size for integration arrays
        (sizeX, sizeDX) = self.m_integrator.maximumIntegrationSubStepsPerCycle(self.m_observationTimes.longestCycle())
        # alloc arrays
        self.m_x  = np.zeros((sizeX, self.m_xSpaceDimension))
        self.m_dx = np.zeros((sizeDX, self.m_xSpaceDimension))
        self.m_y  = np.zeros(self.m_ySpaceDimension)
        # initialise truth
        self.m_initialiser.initialiseTruth(self.m_x[self.m_integrationIndex])
        # initialise output
        self.m_output.initialiseTruthOutput(self.m_truthOutputFields, self.m_observationsOutputFields, self.temporaryRecordShape)

    #_________________________

    def forecast(self, t_tStart, t_tEnd):
        # integrate truth from tStart to tEnd
        self.m_integrationIndex = self.m_integrator.integrate(self.m_x, t_tStart, t_tEnd, self.m_dx)
        # observe truth
        self.m_observationOperator.observe(self.m_x[self.m_integrationIndex], t_tEnd, self.m_y)

    #_________________________

    def permute(self):
        # permute array to prepare next cycle
        self.m_x[0]             = self.m_x[self.m_integrationIndex]
        self.m_integrationIndex = 0

    #_________________________

    def record(self):
        # record truth
        self.m_output.record('truth', 'trajectory', self.m_x[self.m_integrationIndex])
        # record observation
        self.m_output.record('observations', 'trajectory', self.m_y)

#__________________________________________________

