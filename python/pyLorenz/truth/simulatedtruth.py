#! /usr/bin/env python

#__________________________________________________
# pyLorenz/truth/
# simulatedtruth.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/5
#__________________________________________________
#
# class to handle the truth of a simulation
#

import numpy as np

from abstracttruth import AbstractTruth

#__________________________________________________

class SimulatedTruth(AbstractTruth):

    #_________________________

    def __init__(self, t_truthInitialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_truthOutputFields):
        AbstractTruth.__init__(self, t_integrator.m_spaceDimension, t_observationOperator.m_spaceDimension, t_observationTimes, t_output, t_truthOutputFields)
        self.setSimulatedTruthParameters(t_truthInitialiser, t_integrator, t_observationOperator)

    #_________________________

    def setSimulatedTruthParameters(self, t_truthInitialiser, t_integrator, t_observationOperator):
        # initialiser
        self.m_initialiser         = t_truthInitialiser
        # integrator
        self.m_integrator          = t_integrator
        # observation operator
        self.m_observationOperator = t_observationOperator

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
        self.m_initialiser.initialise(self.m_x[0])
        # initialise output
        self.m_output.initialiseTruthOutput(self.m_truthOutputFields, self.temporaryRecordShape)
        # index of current truth state
        self.m_integrationIndex    = 0

    #_________________________

    def forecast(self, t_tStart, t_tEnd):
        # integrate truth from tStart to tEnd
        self.m_integrationIndex = self.m_integrator.integrate(self.m_x, t_tStart, t_tEnd, self.m_dx)
        # observe truth
        self.m_observationOperator.observe(self.m_x[self.m_integrationIndex], t_tEnd, self.m_y)
        # record time
        self.m_time = t_tEnd

    #_________________________

    def permute(self):
        # permute array to prepare next cycle
        self.m_x[0]             = self.m_x[self.m_integrationIndex]
        self.m_integrationIndex = 0

#__________________________________________________

